import torch
import torch_geometric.graphgym.register as register
import torch_sparse
from torch import nn
import math
from typing import Mapping
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import (BatchNorm1dNode,
                                                   new_layer_config)
from torch_sparse import SparseTensor

from .conv import DecoConv
from .attn import GlobalAttentionBlock


class FeatureEncoder(torch.nn.Module):

    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        if cfg.dataset.node_encoder:
            k1_hidden_dim = cfg.DecoNet.conv.k1_hidden_dim
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(k1_hidden_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(new_layer_config(
                    k1_hidden_dim, -1, -1, has_act=False, has_bias=False, cfg=cfg,
                ))
            # Update dim_in to reflect the new dimension fo the node features
        if cfg.dataset.edge_encoder:
            hidden_dim = cfg.DecoNet.conv.hidden_dim
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(hidden_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(new_layer_config(
                    hidden_dim, -1, -1, has_act=False, has_bias=False, cfg=cfg
                ))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register.register_network("DecoNet")
class DecoNet(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        ccfg = cfg.DecoNet.conv
        assert (cfg.posenc_Poly.power) >= 2 ** (ccfg.num_layers - 2)
        pe_dim = int(math.log2(cfg.posenc_Poly.power) + 1) * 3 - 1

        self.feat_encoder = FeatureEncoder(ccfg.k1_hidden_dim)
        self.conv_dict: Mapping[str, DecoConv] = nn.ModuleDict()
        self.conv_dict["1_conv"] = DecoConv(2, ccfg.k1_hidden_dim, ccfg.k1_hidden_dim, ccfg)
        self.conv_dict["2_conv"] = DecoConv(2, ccfg.k1_hidden_dim, ccfg.hidden_dim, ccfg)

        for lidx in range(3, ccfg.num_layers + 1):
            conv = DecoConv(3, ccfg.hidden_dim, ccfg.hidden_dim, ccfg)
            self.conv_dict[f"{lidx}_conv"] = conv

        JK_features = ccfg.k1_hidden_dim * 2 + ccfg.hidden_dim * ccfg.num_layers
        gcfg = cfg.DecoNet.global_attention
        self.JK_mlp = nn.Sequential(
            nn.Linear(JK_features, gcfg.hidden_dim),
            nn.BatchNorm1d(gcfg.hidden_dim),
            register.act_dict[gcfg.act](),
            nn.Dropout(gcfg.drop_prob),
        )

        pe_layer = cfg.DecoNet.pe_layer
        PELayer = register.layer_dict[pe_layer]
        self.loop_layer = PELayer(pe_dim, gcfg.hidden_dim)
        self.conn_layer = PELayer(pe_dim, gcfg.hidden_dim)
        self.global_block = GlobalAttentionBlock(gcfg)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(2 * gcfg.hidden_dim, dim_out, cfg.gnn.layers_post_mp)
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

        ccfg = cfg.DecoNet.conv
        k_idx_list = [
            [k * 3 - 1, (k + 1) * 3 - 1]
            for k in range(ccfg.num_layers)
        ]

        poly_val = all_poly_val[:, :2]
        mask = poly_val[:, 0] > 0.0
        poly_val = poly_val[mask, :]
        poly_idx = all_poly_idx[:, mask]
        dst, src = poly_idx
        mdst = torch.hstack((dst, dst + N))
        msrc = torch.hstack((src, src))
        mval = poly_val.permute((1, 0)).flatten()
        poly_adj = SparseTensor(
            row=mdst, col=msrc, value=mval,
            sparse_sizes=(N * 2, N), is_sorted=True, trust_data=True
        )
        for k in range(1, 3):
            conv = self.conv_dict[f"{k}_conv"]
            batch = conv.forward(batch, poly_adj)
            nh_list.append(batch['x'])

        for k in range(3, ccfg.num_layers + 1):
            l, r = k_idx_list[k - 2]
            poly_val = all_poly_val[:, l:r]
            mask = poly_val[:, 0] > 0.0
            poly_val = poly_val[mask, :]
            poly_idx = all_poly_idx[:, mask]
            dst, src = poly_idx
            mdst = torch.cat((dst, dst + N, dst + N * 2))
            msrc = torch.cat((src, src, src))
            mval = poly_val.permute((1, 0)).flatten()
            poly_adj = SparseTensor(
                row=mdst, col=msrc, value=mval,
                sparse_sizes=(N * 3, N), is_sorted=True, trust_data=True
            )
            conv = self.conv_dict[f"{k}_conv"]
            batch = conv.forward(batch, poly_adj)
            nh_list.append(batch['x'])

        all_loop_val = self.loop_layer(all_loop_val)
        all_poly_val = self.conn_layer(all_poly_val)

        nh_list.append(all_loop_val)
        nh_cat = torch.concat(nh_list, 1)
        nh_conv = self.JK_mlp(nh_cat)
        batch['x'] = nh_conv

        full_idx = batch["full_index"]
        if full_idx.size(1) > all_poly_idx.size(1):
            hidden_dim = cfg.DecoNet.global_attention.hidden_dim
            full_val = all_poly_val.new_zeros((full_idx.size(1), hidden_dim))
            full_idx, full_val = torch_sparse.coalesce(
                torch.cat([full_idx, all_poly_idx, batch.edge_index], dim=1),
                torch.cat([full_val, all_poly_val, batch.edge_attr], dim=0),
                N, N, op="add",
            )
            batch["full_index"] = full_idx
            batch["full_conn"] = full_val
        else:
            full_idx, full_val = torch_sparse.coalesce(
                torch.cat([all_poly_idx, batch.edge_index], dim=1),
                torch.cat([all_poly_val, batch.edge_attr], dim=0),
                N, N, op="add",
            )
            batch["full_conn"] = full_val

        batch = self.global_block(batch)
        batch['x'] = torch.hstack((nh_conv, batch['x']))
        batch = self.post_mp(batch)
        return batch

    def reset(self):
        self.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else m)
