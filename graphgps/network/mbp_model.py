import torch
import torch_geometric.graphgym.register as register
import torch_sparse
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import BatchNorm1dNode, new_layer_config
from torch_geometric.graphgym.register import register_network
from torch_scatter import scatter


def compute_full_index(batch: torch.Tensor):
    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="add")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    Ns = num_nodes.tolist()
    full_index_list = [
        torch.ones(
            (Ns[idx], Ns[idx]), dtype=torch.short, device=batch.device
        ).nonzero(as_tuple=False).t() + cum_nodes[idx]
        for idx in range(batch_size)
    ]
    batch_index_full = torch.cat(full_index_list, dim=1).contiguous()
    return batch_index_full


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """

    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.mbp_model.hidden_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(new_layer_config(
                    cfg.mbp_model.hidden_dim, -1, -1, has_act=False, has_bias=False, cfg=cfg,
                ))
            # Update dim_in to reflect the new dimension fo the node features
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.mbp_model.hidden_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(new_layer_config(
                    cfg.mbp_model.hidden_dim, -1, -1, has_act=False, has_bias=False, cfg=cfg
                ))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network("MbpModel")
class MbpModel(torch.nn.Module):
    """
    The proposed MbpModel
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        assert (cfg.posenc_Poly.power) > 2 ** (cfg.mbp_model.messaging.num_blocks - 2)
        if cfg.posenc_Poly.method in {"mixed_bern"}:
            emb_dim = cfg.posenc_Poly.power + 2
        elif cfg.posenc_Poly.method in {"adj_powers"}:
            emb_dim = cfg.posenc_Poly.power + 1

        MbpMessagingBlock = register.layer_dict["MbpMessagingBlock"]
        MbpFullBlock = register.layer_dict["MbpFullBlock"]

        self.feat_encoder = FeatureEncoder(cfg.mbp_model.hidden_dim)
        self.block_dict = nn.ModuleDict()

        poly_method = "sparse_" + cfg.posenc_Poly.method
        # "outerprod_linear" if cfg.mbp_model.messaging.layer_type in {"gine", "cat"} else "simple_linear"
        # pe_layer = "simple_linear"
        pe_layer = cfg.mbp_model.pe_layer
        PELayer = register.layer_dict[pe_layer]
        for lidx in range(cfg.mbp_model.messaging.num_blocks):
            loop_layer = PELayer(emb_dim, cfg.mbp_model.hidden_dim)
            conn_layer = PELayer(emb_dim, cfg.mbp_model.hidden_dim)
            repeats = max(2, cfg.mbp_model.messaging.repeats) if lidx == 0 else cfg.mbp_model.messaging.repeats
            messaging_block = MbpMessagingBlock(poly_method, repeats, cfg.mbp_model)
            self.block_dict[f"{lidx}_loop_enc"] = loop_layer
            self.block_dict[f"{lidx}_conn_enc"] = conn_layer
            self.block_dict[f"{lidx}_messaging"] = messaging_block

        if cfg.mbp_model.messaging.num_blocks == 0:
            assert cfg.mbp_model.full.enable
            PELayer = register.layer_dict[pe_layer]
            loop_layer = PELayer(emb_dim, cfg.mbp_model.hidden_dim)
            conn_layer = PELayer(emb_dim, cfg.mbp_model.hidden_dim)
            self.block_dict["all_loop_enc"] = loop_layer
            self.block_dict["all_conn_enc"] = conn_layer

        if cfg.mbp_model.full.enable:
            repeats = cfg.mbp_model.full.repeats
            poly_method = "full_" + cfg.posenc_Poly.method
            full_block = MbpFullBlock(poly_method, repeats, cfg.mbp_model)
            self.block_dict["full"] = full_block

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(cfg.mbp_model.hidden_dim, dim_out)

        if cfg.posenc_Poly.method == "mixed_bern":
            # orders: [1, 2, 2, 4, 4, ..., 2**(K-1), 2**(K-1), 2**K, 2**K, 2**K]
            self._poly_order_map = {
                lidx: 2 ** lidx - 1
                for lidx in range(cfg.mbp_model.messaging.num_blocks)
            }
        elif cfg.posenc_Poly.method == "adj_powers":
            # orders: [0, 1, 2, 3, 4,]
            self._poly_order_map = {
                lidx: 2 ** lidx
                for lidx in range(cfg.mbp_model.messaging.num_blocks)
            }
        else:
            raise NotImplementedError

    def forward(self, batch: Batch):
        all_loop_val = batch[cfg.posenc_Poly.method + "_loop"]
        all_poly_idx = batch[cfg.posenc_Poly.method + "_index"]
        all_poly_val = batch[cfg.posenc_Poly.method + "_conn"]

        batch = self.feat_encoder(batch)
        sparse_poly = "sparse_" + cfg.posenc_Poly.method
        batch[sparse_poly + "_index"] = batch.edge_index
        batch[sparse_poly + "_conn"] = batch.edge_attr

        for lidx in range(cfg.mbp_model.messaging.num_blocks):
            order = self._poly_order_map[lidx]
            if lidx < cfg.mbp_model.messaging.num_blocks - 1:
                poly_sgn = all_poly_val[:, order] != 0
                poly_idx = all_poly_idx[:, poly_sgn]
                poly_val = all_poly_val[poly_sgn, :]
            else:
                # poly_sgn: all true
                poly_idx = all_poly_idx
                poly_val = all_poly_val

            loop_h = self.block_dict[f"{lidx}_loop_enc"](all_loop_val)
            poly_h = self.block_dict[f"{lidx}_conn_enc"](poly_val)
            batch.x += loop_h
            if poly_idx.size(1) != batch[sparse_poly + "_index"].size(1):
                poly_idx_add, poly_h_add = torch_sparse.coalesce(
                    torch.cat([poly_idx, batch[sparse_poly + "_index"]], dim=1),
                    torch.cat([poly_h, batch[sparse_poly + "_conn"]], dim=0),
                    batch.num_nodes, batch.num_nodes, op="add",
                )
                batch[sparse_poly + "_index"] = poly_idx_add
                batch[sparse_poly + "_conn"] = poly_h_add
            else:  # poly_idx == batch[sparse_poly + "_index"]
                batch[sparse_poly + "_conn"] = batch[sparse_poly + "_conn"] + poly_h

            batch = self.block_dict[f"{lidx}_messaging"](batch)

        if cfg.mbp_model.messaging.num_blocks == 0:
            all_loop_h = self.block_dict["all_loop_enc"](all_loop_val)
            all_poly_h = self.block_dict["all_conn_enc"](all_poly_val)
            poly_idx_add, poly_h_add = torch_sparse.coalesce(
                torch.cat([all_poly_idx, batch[sparse_poly + "_index"]], dim=1),
                torch.cat([all_poly_h, batch[sparse_poly + "_conn"]], dim=0),
                batch.num_nodes, batch.num_nodes, op="add",
            )
            batch.x += all_loop_h
            batch[sparse_poly + "_index"] = poly_idx_add
            batch[sparse_poly + "_conn"] = poly_h_add

        if cfg.mbp_model.full.enable:
            full_poly = "full_" + cfg.posenc_Poly.method
            if cfg.mbp_model.full.layer_type in {"grit", "ppgn", "gine", "gat"}:
                if "full_index" in batch:
                    full_index = batch["full_index"]
                else:
                    full_index = compute_full_index(batch.batch)
                if full_index.size(1) > batch[sparse_poly + "_index"].size(1):
                    full_val = all_poly_val.new_zeros((full_index.size(1), cfg.mbp_model.hidden_dim))
                    full_index, full_h = torch_sparse.coalesce(
                        torch.cat([full_index, batch[sparse_poly + "_index"]], dim=1),
                        torch.cat([full_val, batch[sparse_poly + "_conn"]], dim=0),
                        batch.num_nodes, batch.num_nodes, op="add",
                    )
                    batch[full_poly + "_index"] = full_index
                    batch[full_poly + "_conn"] = full_h
                else:
                    batch[full_poly + "_index"] = full_index
                    batch[full_poly + "_conn"] = batch[sparse_poly + "_conn"]

            batch = self.block_dict["full"](batch)

        batch = self.post_mp(batch)
        return batch

    def reset_parameters(self):
        self.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else m)
