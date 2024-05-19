import torch
import torch_geometric.graphgym.register as register
import torch_sparse
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import BatchNorm1dNode, new_layer_config
from torch_geometric.graphgym.register import register_network
from torch_scatter import scatter


def compute_full_edge_index(batch: torch.Tensor):
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
            self.node_encoder = NodeEncoder(cfg.gse_model.hidden_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(new_layer_config(
                    cfg.gse_model.hidden_dim, -1, -1, has_act=False, has_bias=False, cfg=cfg,
                ))
            # Update dim_in to reflect the new dimension fo the node features
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gse_model.hidden_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(new_layer_config(
                    cfg.gse_model.hidden_dim, -1, -1, has_act=False, has_bias=False, cfg=cfg
                ))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class InitialLayer(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        hidden_dim (int): Input feature dimension
    """

    def __init__(self,):
        super(InitialLayer, self).__init__()
        GseMessagingBlock = register.layer_dict["GseMessagingBlock"]
        self.mpnn = GseMessagingBlock(
            cfg.gse_model.messaging.repeats,
            cfg.gse_model.messaging.layer_type,
            cfg.gse_model.hidden_dim,
            cfg.gse_model.hidden_dim,
            cfg.gse_model.num_heads,
            cfg.gse_model,
            cfg.gse_model.dropout,
            cfg.gse_model.attn_dropout,
        )

    def forward(self, batch: Batch):
        batch.poly_val = batch.edge_attr
        batch.poly_idx = batch.edge_index
        self.mpnn(batch)
        batch.edge_attr = batch.poly_val
        return batch


@register_network("GseModel")
class GseModel(torch.nn.Module):
    """
    The proposed GseModel
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        assert (cfg.posenc_Poly.emb_dim - 2) == 2 ** (cfg.gse_model.messaging.num_layers)
        self.feat_encoder = FeatureEncoder(cfg.gse_model.hidden_dim)
        self.init_layer = InitialLayer(
            cfg.gse_model.hidden_dim, cfg.gse_model.num_heads, cfg.gse_model,
            cfg.gse_model.dropout, cfg.gse_model.attn_dropout
        )

        GseMessagingBlock = register.layer_dict["GseMessagingBlock"]
        GseFullBlock = register.layer_dict["GseFullBlock"]
        AbsEncoder = register.node_encoder_dict["poly_sparse"]
        RelEncoder = register.edge_encoder_dict["poly_sparse"]

        self.poly_method = cfg.posenc_Poly.method + "_sparse"
        self.block_dict = nn.ModuleDict()
        for lidx in range(1, cfg.gse_model.messaging.num_layers + 1):
            emb_dim = 2**lidx if lidx < cfg.gse_model.messaging.num_layers else cfg.posenc_Poly.emb_dim
            abs_encoder = AbsEncoder(self.poly_method, emb_dim, cfg.gse_model.hidden_dim)
            rel_encoder = RelEncoder(self.poly_method, emb_dim, cfg.gse_model.hidden_dim)
            messaging_block = GseMessagingBlock(
                cfg.gse_model.messaging.repeats,
                cfg.gse_model.messaging.layer_type,
                cfg.gse_model.hidden_dim,
                cfg.gse_model.hidden_dim,
                cfg.gse_model.num_heads,
                cfg.gse_model,
                cfg.gse_model.dropout,
                cfg.gse_model.attn_dropout,
            )
            self.block_dict[f"{lidx}_abs_enc"] = abs_encoder
            self.block_dict[f"{lidx}_rel_enc"] = rel_encoder
            self.block_dict[f"{lidx}_messaging"] = messaging_block

        for lidx in range(1, cfg.gse_model.full.num_layers + 1):
            full_block = GseFullBlock(
                cfg.gse_model.full.repeats,
                cfg.gse_model.full.layer_type,
                cfg.gse_model.hidden_dim,
                cfg.gse_model.hidden_dim,
                cfg.gse_model.num_heads,
                cfg.gse_model,
                cfg.gse_model.dropout,
                cfg.gse_model.attn_dropout
            )
            self.block_dict[f"full_{lidx}"] = full_block

        GNNHead = register.head_dict[cfg.gse_model.head]
        self.post_mp = GNNHead(dim_in=cfg.gse_model.hidden_dim, dim_out=dim_out)

        if cfg.posenc_Poly.method == "mixed_bern":
            # orders = [0] + [
            #     (idx + 1) // 2**2 for idx in range(2, cfg.posenc_Poly.order + 1)
            # ] + [cfg.posenc_Poly.order]
            # orders: [0, 2, 2, 4, 4, ..., 2**(K-1), 2**(K-1), 2**K, 2**K, 2**K]
            # order2idx_map: {2: 2, 4: 4, 8: 8, 16: 16}
            self._poly_order_map = {
                lidx: 2**lidx
                for lidx in range(1, cfg.gse_model.messaging.num_layers + 1)
            }
        else:
            raise NotImplementedError

    def forward(self, batch: Batch):
        batch = self.feat_encoder(batch)
        batch = self.init_layer(batch)

        whole_poly_idx = batch[f"{cfg.posenc_Poly.method}_index"]
        whole_poly_val = batch[f"{cfg.posenc_Poly.method}_val"]
        whole_abs_val = batch[f"{cfg.posenc_Poly.method}"]
        order1_flag = batch[f"{cfg.posenc_Poly.method}_order1_flag"]

        for lidx in range(1, cfg.gse_model.messaging.num_layers + 1):
            order = self._poly_order_map[lidx]
            if lidx == 1:
                abs_val = whole_abs_val[:, :order]
                order1_idx = whole_poly_idx[:, order1_flag]
                order1_val = whole_poly_val[order1_flag, :order]
                order1_h = self.block_dict[f"{lidx}_rel_enc"](order1_val)
                poly_idx, poly_val = torch_sparse.coalesce(
                    torch.cat([batch.edge_index, order1_idx], dim=1),
                    torch.cat([batch.edge_attr, order1_h], dim=0),
                    batch.num_nodes, batch.num_nodes, op="add",
                )
                batch.poly_idx = poly_idx
                batch.poly_val = poly_val
            else:
                if lidx < cfg.gse_model.messaging.num_layers:
                    abs_val = whole_abs_val[:, :order]
                    poly_flag = whole_poly_val[:, order] != 0
                    poly_idx = whole_poly_idx[:, poly_flag]
                    poly_val = whole_poly_val[poly_flag, :order]
                else:
                    abs_val = whole_abs_val
                    poly_idx = whole_poly_idx
                    poly_val = whole_poly_val

                abs_h = self.block_dict[f"{lidx}_abs_enc"](abs_val)
                batch.x = batch.x + abs_h
                poly_h = self.block_dict[f"{lidx}_rel_enc"](poly_val)
                poly_idx, poly_val = torch_sparse.coalesce(
                    torch.cat([batch.poly_idx, poly_idx], dim=1),
                    torch.cat([batch.poly_val, poly_h], dim=0),
                    batch.num_nodes, batch.num_nodes, op="add",
                )
                batch.poly_idx = poly_idx
                batch.poly_val = poly_val

            batch = self.block_dict[f"{lidx}_messaging"](batch)

        if cfg.gse_model.full.num_layers > 0:
            if 'full_edge_index' in batch:
                full_edge_index = batch['full_edge_index']
            else:
                full_edge_index = compute_full_edge_index(batch.batch)

            if full_edge_index.size(1) > batch.poly_idx.size(1):
                full_pad = poly_h.new_zeros((full_edge_index.size(1), cfg.gse_model.hidden_dim))
                poly_idx, poly_val = torch_sparse.coalesce(
                    torch.cat([batch.poly_idx, full_edge_index], dim=1),
                    torch.cat([batch.poly_val, full_pad], dim=0),
                    batch.num_nodes, batch.num_nodes, op="add",
                )
                batch.poly_idx = poly_idx
                batch.poly_val = poly_val

        for lidx in range(1, cfg.gse_model.full.num_layers + 1):
            batch = self.block_dict[f"full_{lidx}"](batch)

        batch = self.post_mp(batch)
        return batch
