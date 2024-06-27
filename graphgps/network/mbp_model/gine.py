import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.graphgym.register import act_dict
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data, Batch
from torch_scatter import scatter, scatter_add, scatter_max
from yacs.config import CfgNode


def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out


class GINE(nn.Module):
    def __init__(self, cfg: CfgNode):
        in_features=cfg.hidden_dim
        attn_heads=cfg.attn_heads
        clamp=cfg.clamp
        attn_drop_prob=cfg.attn_drop_prob
        drop_prob=cfg.drop_prob
        weight_fn=cfg.weight_fn
        agg=cfg.agg
        act=cfg.act
        bn_momentum=cfg.bn_momentum
        super().__init__()
        self.attn_heads = attn_heads
        self.attn_features = in_features // attn_heads
        self.weight_fn = weight_fn
        self.agg = agg
        self.act = act_dict[act]()
        self.dropout = nn.Dropout(drop_prob)
        self.attn_dropout = nn.Dropout(attn_drop_prob)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.qkv_weight = nn.Parameter(torch.empty((3 * in_features, in_features)))
        self.qkv_bias = nn.Parameter(torch.empty(3 * in_features))
        self.conn_lin1 = nn.Linear(in_features, 2 * in_features)
        # self.Wscore = nn.Parameter(torch.zeros(self.attn_features, self.attn_heads, 1))
        self.conn_lin2 = nn.Linear(in_features, in_features)
        self.conn_norm = nn.BatchNorm1d(in_features, eps=1e-5, momentum=bn_momentum)

        self.deg_coef = nn.Parameter(torch.zeros(1, in_features, 2))
        self.FFN_n_layer1 = nn.Linear(in_features, in_features * 2)
        self.FFN_n_layer2 = nn.Linear(in_features * 2, in_features)
        self.norm1_h = nn.BatchNorm1d(in_features, eps=1e-5, momentum=bn_momentum)
        self.norm2_h = nn.BatchNorm1d(in_features, eps=1e-5, momentum=bn_momentum)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_weight)
        nn.init.constant_(self.qkv_bias, 0.)
        nn.init.xavier_normal_(self.conn_lin1.weight)
        nn.init.xavier_normal_(self.conn_lin2.weight)
        nn.init.constant_(self.conn_lin1.bias, 0.)
        nn.init.constant_(self.conn_lin2.bias, 0.)
        # nn.init.xavier_normal_(self.Wscore)
        self.conn_norm.reset_parameters()

        nn.init.xavier_normal_(self.deg_coef)
        self.FFN_n_layer1.reset_parameters()
        self.FFN_n_layer2.reset_parameters()
        self.norm1_h.reset_parameters()
        self.norm2_h.reset_parameters()

    def forward(self, batch: Data | Batch):
        x = batch['x']
        Ex: Tensor = batch["poly_conn"]
        xh = self.conn_lin1(Ex)
        xh = xh.view((-1, 2, self.attn_heads * self.attn_features))
        xh = xh.transpose(0, 1).contiguous()
        Qh, Kh = xh[0], xh[1]
        dst, src = batch["poly_index"]
        Qdst = Qh[dst]
        Ksrc = Kh[src]

        Ex: Tensor = batch["poly_conn"]
        Eh = self.conn_lin1(Ex)
        Eh = Eh.view((-1, 2, self.attn_heads * self.attn_features))
        Eh = Eh.transpose(0, 1).contiguous()
        Ew, Eb = Eh[0], Eh[1]

        conn = Qdst + Ksrc
        conn = conn * Ew
        # conn = torch.sqrt(torch.relu(conn)) - torch.sqrt(torch.relu(-conn))
        conn = conn + Eb
        conn = self.act(conn)
        conn = self.dropout(conn)

        conn2 = self.conn_lin2(conn)
        conn2 = conn2 + Ex
        conn2 = self.conn_norm(conn2)
        conn2 = self.act(conn2)

        # Aggregate with Attn-Score
        nh = scatter(conn2, dst, dim=0, dim_size=batch.num_nodes, reduce=self.agg)

        sqrt_deg = batch["sqrt_deg"]
        nh = torch.stack([nh, nh * sqrt_deg], dim=-1)
        nh = (nh * self.deg_coef).sum(dim=-1)

        nh = h_res = nh + x
        nh = self.norm1_h(nh)
        nh = self.dropout(nh)
        nh = self.FFN_n_layer1(nh)
        nh = self.act(nh)
        nh = self.dropout(nh)
        nh = self.FFN_n_layer2(nh)
        nh = nh + h_res
        nh = self.norm2_h(nh)

        batch['x'] = nh
        batch["poly_conn"] = conn2.flatten(1)
        return batch
