import torch
import torch_geometric.graphgym.register as register
import torch_sparse
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import BatchNorm1dNode, new_layer_config
from torch_geometric.graphgym.register import register_network


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features
    """

    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        model_cfg = cfg.MbpModel
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(model_cfg.hidden_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(new_layer_config(
                    model_cfg.hidden_dim, -1, -1, has_act=False, has_bias=False, cfg=cfg,
                ))
            # Update dim_in to reflect the new dimension fo the node features
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(model_cfg.hidden_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(new_layer_config(
                    model_cfg.hidden_dim, -1, -1, has_act=False, has_bias=False, cfg=cfg
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
        model_cfg = cfg.MbpModel
        poly_cfg = cfg.posenc_Poly
        poly_method = poly_cfg.method

        if model_cfg.poly.num_blocks > 0:
            assert poly_method in {"mixed_sym_bern", "deco_bern"}

        assert (poly_cfg.power) > 2 ** (model_cfg.poly.num_blocks - 2)
        if poly_method in {"mixed_sym_bern", "bern", "deco_bern"}:
            emb_dim = poly_cfg.power + 2
        elif poly_method in {"adj_powers"}:
            emb_dim = poly_cfg.power + 1
        elif poly_method in {"mixed_low_bern"}:
            emb_dim = poly_cfg.power + 1

        PolyBlock = register.layer_dict["PolyBlock"]
        FullBlock = register.layer_dict["FullBlock"]
        PELayer = register.layer_dict[model_cfg.pe_layer]

        self.feat_encoder = FeatureEncoder(model_cfg.hidden_dim)
        loop_layer = PELayer(emb_dim, model_cfg.hidden_dim)
        self.loop_enc = loop_layer

        self.block_dict = nn.ModuleDict()
        for lidx in range(model_cfg.poly.num_blocks):
            repeats = max(2, model_cfg.poly.repeats) if lidx == 0 else model_cfg.poly.repeats
            poly_block = PolyBlock(repeats, model_cfg)
            conn_layer = PELayer(emb_dim, model_cfg.hidden_dim)
            self.block_dict[f"{lidx}_conn_enc"] = conn_layer
            self.block_dict[f"{lidx}_poly_blk"] = poly_block

        if model_cfg.poly.num_blocks == 0:
            conn_layer = PELayer(emb_dim, model_cfg.hidden_dim)
            self.block_dict["all_conn_enc"] = conn_layer
        else:
            JK_features = (model_cfg.poly.num_blocks + 1) * model_cfg.hidden_dim
            self.JK_mlp = nn.Sequential(
                nn.Linear(JK_features, model_cfg.hidden_dim),
                nn.BatchNorm1d(model_cfg.hidden_dim),
                register.act_dict[model_cfg.act](),
                nn.Dropout(model_cfg.drop_prob),
            ) if model_cfg.jumping_knowledge else None

        repeats = model_cfg.full.repeats
        full_block = FullBlock(repeats, model_cfg)
        self.block_dict["full_blk"] = full_block

        GNNHead = register.head_dict[cfg.gnn.head]
        if cfg.gnn.head == 'default':
            self.post_mp = GNNHead(model_cfg.hidden_dim, dim_out)
        else:
            self.post_mp = GNNHead(model_cfg.hidden_dim, dim_out, cfg.gnn.layers_post_mp)

    def forward(self, batch: Batch):
        model_cfg = cfg.MbpModel
        poly_cfg = cfg.posenc_Poly
        poly_method = poly_cfg.method

        all_loop_val = batch[poly_method + "_loop"]
        all_poly_idx = batch[poly_method + "_index"]
        all_poly_val = batch[poly_method + "_conn"]

        batch = self.feat_encoder(batch)
        batch["poly_index"] = batch.edge_index
        batch["poly_conn"] = batch.edge_attr
        loop_h = self.loop_enc(all_loop_val)
        batch['x'] = batch['x'] + loop_h

        x_list = [batch['x']]

        all_nonzero_flag = False
        for lidx in range(model_cfg.poly.num_blocks):
            K = 0 if lidx == 0 else 2 ** lidx
            if all_nonzero_flag or lidx >= model_cfg.poly.num_blocks - 1:
                # poly_sgn: all true
                poly_idx = all_poly_idx
                poly_val = all_poly_val
            else:
                poly_sgn = all_poly_val[:, K] != 0.0
                poly_idx = all_poly_idx[:, poly_sgn]
                poly_val = all_poly_val[poly_sgn, :]
                if torch.all(poly_sgn):
                    all_nonzero_flag = True

            if poly_idx.size(1) > batch["poly_index"].size(1):
                poly_h = self.block_dict[f"{lidx}_conn_enc"](poly_val)
                poly_idx_add, poly_h_add = torch_sparse.coalesce(
                    torch.cat([poly_idx, batch["poly_index"]], dim=1),
                    torch.cat([poly_h, batch["poly_conn"]], dim=0),
                    batch.num_nodes, batch.num_nodes, op="add",
                )
                batch["poly_index"] = poly_idx_add
                batch["poly_conn"] = poly_h_add
            else:
                pass

            batch = self.block_dict[f"{lidx}_poly_blk"](batch)
            x_list.append(batch['x'])

        full_idx = batch["full_index"]
        if model_cfg.poly.num_blocks == 0:
            all_poly_emb = self.block_dict["all_conn_enc"](all_poly_val)
            if full_idx.size(1) > all_poly_idx.size(1):
                full_val = all_poly_val.new_zeros((full_idx.size(1), model_cfg.hidden_dim))
                poly_idx_add, poly_h_add = torch_sparse.coalesce(
                    torch.cat([full_idx, all_poly_idx, batch["poly_index"]], dim=1),
                    torch.cat([full_val, all_poly_emb, batch["poly_conn"]], dim=0),
                    batch.num_nodes, batch.num_nodes, op="add",
                )
            else:
                poly_idx_add, poly_h_add = torch_sparse.coalesce(
                    torch.cat([all_poly_idx, batch["poly_index"]], dim=1),
                    torch.cat([all_poly_emb, batch["poly_conn"]], dim=0),
                    batch.num_nodes, batch.num_nodes, op="add",
                )
            batch["poly_index"] = poly_idx_add
            batch["poly_conn"] = poly_h_add
        else:
            if self.JK_mlp is not None:
                x_cat = torch.concat(x_list, 1)
                x = self.JK_mlp(x_cat)
                batch['x'] = x
            if full_idx.size(1) > all_poly_idx.size(1):
                # pad to complete graphs
                full_val = all_poly_val.new_zeros((full_idx.size(1), model_cfg.hidden_dim))
                poly_idx_add, poly_h_add = torch_sparse.coalesce(
                    torch.cat([full_idx, batch["poly_index"]], dim=1),
                    torch.cat([full_val, batch["poly_conn"]], dim=0),
                    batch.num_nodes, batch.num_nodes, op="add",
                )
                batch["poly_index"] = poly_idx_add
                batch["poly_conn"] = poly_h_add
            else:
                # already are complete graphs
                pass

        batch = self.block_dict["full_blk"](batch)
        batch = self.post_mp(batch)
        return batch

    def reset_parameters(self):
        self.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else m)
