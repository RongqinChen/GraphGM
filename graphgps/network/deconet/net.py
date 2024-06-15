import torch
import torch_geometric.graphgym.register as register
import torch_sparse
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (BatchNorm1dNode,
                                                   new_layer_config)
from torch_sparse import SparseTensor

from .attn import ConditionalAttentionBlock
from .conv import DecoConvBlock


class FeatureEncoder(torch.nn.Module):

    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.DecoNet.hidden_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(new_layer_config(
                    cfg.DecoNet.hidden_dim, -1, -1, has_act=False, has_bias=False, cfg=cfg,
                ))
            # Update dim_in to reflect the new dimension fo the node features
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.DecoNet.hidden_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(new_layer_config(
                    cfg.DecoNet.hidden_dim, -1, -1, has_act=False, has_bias=False, cfg=cfg
                ))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register.register_network("DecoNet")
class DecoNet(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        assert (cfg.posenc_Poly.power) > 2 ** (cfg.DecoNet.conv.num_blocks - 2)
        emb_dim = cfg.posenc_Poly.power + 2
        poly_method = cfg.posenc_Poly.method

        self.feat_encoder = FeatureEncoder(cfg.DecoNet.hidden_dim)
        self.block_dict = nn.ModuleDict()

        for lidx in range(cfg.DecoNet.conv.num_blocks):
            repeats = max(2, cfg.DecoNet.conv.repeats) if lidx == 0 else cfg.DecoNet.conv.repeats
            conv_block = DecoConvBlock(poly_method, repeats, cfg.DecoNet)
            self.block_dict[f"{lidx}_conv"] = conv_block

        JK_features = (cfg.DecoNet.conv.num_blocks + 2) * cfg.DecoNet.hidden_dim
        self.JK_mlp = nn.Sequential(
            nn.Linear(JK_features, cfg.DecoNet.hidden_dim),
            nn.BatchNorm1d(cfg.DecoNet.hidden_dim),
            register.act_dict[cfg.DecoNet.act](),
            nn.Dropout(cfg.DecoNet.drop_prob),
        )

        pe_layer = cfg.DecoNet.pe_layer
        PELayer = register.layer_dict[pe_layer]
        self.loop_layer = PELayer(emb_dim, cfg.DecoNet.hidden_dim)
        self.conn_layer = PELayer(emb_dim, cfg.DecoNet.hidden_dim)
        full_block = ConditionalAttentionBlock(repeats, cfg.DecoNet)
        self.block_dict["full"] = full_block

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(cfg.DecoNet.hidden_dim, dim_out, cfg.gnn.layers_post_mp)
        self.reset()

    def forward(self, batch: Batch | Data):
        N = batch.num_nodes
        poly_method = cfg.posenc_Poly.method
        all_loop_val = batch[poly_method + "_loop"]
        all_poly_idx = batch[poly_method + "_index"]
        all_poly_val = batch[poly_method + "_conn"]

        nh_list = []
        batch = self.feat_encoder(batch)
        nh_list.append(batch['x'])
        power_idx_dict = {
            lidx: 2 ** lidx for lidx in range(1, 11)
        }
        power_idx_dict[0] = 0
        for lidx in range(cfg.DecoNet.conv.num_blocks):
            if lidx < cfg.DecoNet.conv.num_blocks - 1:
                # power_idx = power_idx_dict[lidx]
                # mask = all_poly_val[:, power_idx] != 0.0
                # poly_idx = all_poly_idx[:, mask]
                # poly_val = all_poly_val[mask, power_idx]
                k = 2 ** lidx
                poly_idx = batch[f"{poly_method}_{k}_index"]
                poly_val = batch[f"{poly_method}_{k}_conn"]

            else:
                # poly_sgn: all true
                poly_idx = all_poly_idx
                poly_val = all_poly_val[:, -1]

            poly_adj = SparseTensor.from_edge_index(
                poly_idx, poly_val, (N, N), True, True
            )
            batch = self.block_dict[f"{lidx}_conv"](batch, poly_adj)
            nh_list.append(batch['x'])

        all_loop_val = self.loop_layer(all_loop_val)
        all_poly_val = self.conn_layer(all_poly_val)

        nh_list.append(all_loop_val)
        nh_cat = torch.concat(nh_list, 1)
        nh = self.JK_mlp(nh_cat)
        batch['x'] = nh

        full_idx = batch["full_index"]
        if full_idx.size(1) > all_poly_idx.size(1):
            full_val = all_poly_val.new_zeros((full_idx.size(1), cfg.DecoNet.hidden_dim))
            full_idx, full_val = torch_sparse.coalesce(
                torch.cat([full_idx, all_poly_idx, batch.edge_index], dim=1),
                torch.cat([full_val, all_poly_val, batch.edge_attr], dim=0),
                N, N, op="add",
            )
        else:
            full_idx, full_val = torch_sparse.coalesce(
                torch.cat([all_poly_idx, batch.edge_index], dim=1),
                torch.cat([all_poly_val, batch.edge_attr], dim=0),
                N, N, op="add",
            )

        batch["full_index"] = full_idx
        batch["full_conn"] = full_val
        batch = self.block_dict["full"](batch)
        batch = self.post_mp(batch)

        torch.cuda.empty_cache()
        return batch

    def reset(self):
        self.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else m)
