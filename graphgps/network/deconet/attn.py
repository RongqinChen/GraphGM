import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.graphgym.register import act_dict
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data, Batch
from torch_scatter import scatter, scatter_add, scatter_max
from yacs.config import CfgNode
import opt_einsum as oe


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


class ConditionalAttention(nn.Module):

    def __init__(
        self, in_features, attn_heads, clamp,
        attn_drop_prob, drop_prob, weight_fn, agg, act, bn_momentum
    ):
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
        # self.conn_lin2 = nn.Conv1d(in_features, in_features, 1, groups=attn_heads)
        # self.score_lin = nn.Conv1d(in_features, attn_heads, 1, groups=attn_heads)

        self.Aw = nn.Parameter(torch.zeros(self.attn_features, self.attn_heads, 1))
        self.Bw = nn.Parameter(torch.zeros(self.attn_features, self.attn_heads, self.attn_features))

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
        self.conn_lin1.reset_parameters()
        # self.score_lin.reset_parameters()
        # self.conn_lin2.reset_parameters()
        nn.init.xavier_uniform_(self.Aw)
        nn.init.xavier_uniform_(self.Bw)
        self.conn_norm.reset_parameters()

        nn.init.xavier_normal_(self.deg_coef)
        self.FFN_n_layer1.reset_parameters()
        self.FFN_n_layer2.reset_parameters()
        self.norm1_h.reset_parameters()
        self.norm2_h.reset_parameters()

    def forward(self, batch: Data | Batch):
        x = batch['x']
        Qh, Kh, Vh = F._in_projection_packed(x, x, x, self.qkv_weight, self.qkv_bias)
        dst, src = batch["full_index"]
        Qdst = Qh[dst]
        Ksrc = Kh[src]
        Vsrc = Vh[src]

        Ex: Tensor = batch["full_conn"]
        Eh = self.conn_lin1(Ex)
        Eh = Eh.view((-1, 2, self.attn_heads * self.attn_features))
        Eh = Eh.transpose(0, 1).contiguous()
        Ew, Eb = Eh[0], Eh[1]

        conn = Qdst + Ksrc
        conn = conn * Ew
        conn = torch.sqrt(torch.relu(conn)) - torch.sqrt(torch.relu(-conn))
        conn = conn + Eb
        conn = self.act(conn)
        conn = self.dropout(conn)
        # conn = conn.unsqueeze(-1)
        # conn: N, self.attn_heads * self.attn_features, 1
        # conn2 = self.conn_lin2(conn).squeeze(-1)
        # conn2 = conn.squeeze(-1)

        conn = conn.view((-1, self.attn_heads, self.attn_features))
        conn2 = oe.contract("nhd, dhc -> nhc", conn, self.Bw, backend="torch")
        conn2 = conn2 + Ex.view((-1, self.attn_heads, self.attn_features))
        # conn2 = self.conn_norm(conn2)
        conn2 = self.act(conn2)
        conn2 = self.dropout(conn2)

        score = oe.contract("ehd, dhc->ehc", conn, self.Aw, backend="torch")
        # score = self.score_lin(conn)
        # score: N, self.attn_heads, 1
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)
        score = pyg_softmax(score, dst)  # (num relative) x attn_heads x 1
        score = self.attn_dropout(score)

        # Aggregate with Attn-Score
        Vsrc = Vsrc.view(-1, self.attn_heads, self.attn_features)
        nagg = scatter(Vsrc * score, dst, dim=0, dim_size=batch.num_nodes, reduce=self.agg)
        cagg = scatter(conn2 * score, dst, dim=0, dim_size=batch.num_nodes, reduce=self.agg)
        nh = (nagg + cagg).flatten(1)

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
        batch["full_conn"] = conn2.flatten(1)
        return batch


class ConditionalAttentionBlock(nn.Module):

    def __init__(self, repeats, cfg: CfgNode):
        super().__init__()
        self.attn_list = nn.ModuleList()
        for _ in range(repeats):
            attn = ConditionalAttention(
                cfg.hidden_dim, cfg.attn_heads, cfg.clamp, cfg.attn_drop_prob,
                cfg.drop_prob, cfg.weight_fn, cfg.agg, cfg.act, cfg.bn_momentum
            )
            self.attn_list.append(attn)

    def forward(self, batch: Data | Batch):
        for attn in self.attn_list:
            batch = attn(batch)
        return batch
