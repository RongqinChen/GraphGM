import numpy as np
import opt_einsum as oe
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.graphgym.register import act_dict
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_add, scatter_max


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


def tanh(x: torch.Tensor):
    ex1 = torch.exp(x)
    ex2 = torch.exp(-x)
    out = (ex1 - ex2) / (ex1 + ex2)
    return out


def sigmoid(x: torch.Tensor):
    ex1 = torch.exp(-x)
    out = 1. / (1. + ex1)
    return out


class GritMessagePassing(nn.Module):

    def __init__(self, poly_method, hidden_dim, attn_dim, attn_heads, clamp, attn_drop_prob, weight_fn, agg, act):
        super().__init__()
        self.poly_method = poly_method
        self.attn_dim = attn_dim
        self.attn_heads = attn_heads
        self.weight_fn = weight_fn
        self.agg = agg
        self.attn_dropout = nn.Dropout(attn_drop_prob)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.qkv_weight = nn.Parameter(torch.empty((3 * attn_dim * attn_heads, hidden_dim)))
        self.qkv_bias = nn.Parameter(torch.empty(3 * attn_dim * attn_heads))
        self.E = nn.Linear(hidden_dim, 2 * attn_dim * attn_heads)
        self.Aw = nn.Parameter(torch.zeros(self.attn_dim, self.attn_heads, 1))
        self.Bw = nn.Parameter(torch.zeros(self.attn_dim, self.attn_heads, self.attn_dim))

        nn.init.xavier_uniform_(self.qkv_weight)
        nn.init.constant_(self.qkv_bias, 0.)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.constant_(self.E.bias, 0.)
        nn.init.xavier_normal_(self.Aw)
        nn.init.xavier_normal_(self.Bw)
        self.act = act_dict[act]()

    def propagate_attention(self, batch):
        dst, src = batch[self.poly_method + "_index"]
        Qdst = batch.Qh[dst]
        Ksrc = batch.Kh[src]
        Vsrc = batch.Vh[src]
        Ew = batch.Ew
        Eb = batch.Eb

        msg1 = Qdst + Ksrc
        conn1 = msg1 * Ew
        conn2 = torch.sqrt(torch.relu(conn1)) - torch.sqrt(torch.relu(-conn1))
        conn3 = conn2 + Eb
        conn = self.act(conn3)
        # output edge
        batch.Eo = conn

        # final attn
        conn = conn.view(-1, self.attn_heads, self.attn_dim).contiguous()
        score = oe.contract("ehd, dhc->ehc", conn, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        # raw_attn = score
        if self.weight_fn == 'softmax':
            score = pyg_softmax(score, dst)  # (num relative) x attn_heads x 1
        elif self.weight_fn == 'sigmoid':
            score = sigmoid(score)
        elif self.weight_fn == 'tanh':
            score = tanh(score)
        else:
            raise NotImplementedError

        score = self.attn_dropout(score)
        # Aggregate with Attn-Score
        Vsrc = Vsrc.view(-1, self.attn_heads, self.attn_dim)
        agg = scatter(Vsrc * score, dst, dim=0, dim_size=batch.num_nodes, reduce=self.agg)
        rowV = scatter(conn * score, dst, dim=0, dim_size=batch.num_nodes, reduce=self.agg)
        rowV = oe.contract("nhd, dhc -> nhc", rowV, self.Bw, backend="torch")
        No = (agg + rowV).flatten(1)
        batch.No = No

    def forward(self, batch):
        x = batch.x
        batch.Qh, batch.Kh, batch.Vh = F._in_projection_packed(x, x, x, self.qkv_weight, self.qkv_bias)

        Eh = self.E(batch[self.poly_method + "_conn"])
        Eh = Eh.unflatten(-1, (2, self.attn_heads * self.attn_dim))
        Eh = Eh.unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        batch.Ew, batch.Eb = Eh[0], Eh[1]

        self.propagate_attention(batch)
        h_out = batch.No
        e_out = batch.Eo
        return h_out, e_out


class ConditionalAttention(nn.Module):
    def __init__(
        self, poly_method, hidden_dim, attn_heads, drop_prob, attn_drop_prob,
        residual, layer_norm, batch_norm, bn_momentum, bn_no_runner,
        rezero, deg_scaler, clamp, weight_fn, agg, act,
    ):

        super().__init__()
        assert hidden_dim % attn_heads == 0
        self.poly_method = poly_method
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.attn_dim = attn_dim = hidden_dim // attn_heads
        self.drop_prob = drop_prob
        self.attn_drop_prob = attn_drop_prob
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.bn_momentum = bn_momentum
        self.bn_no_runner = bn_no_runner
        self.rezero = rezero
        self.act = act_dict[act]()
        self.deg_scaler = deg_scaler

        self.message_pass = GritMessagePassing(
            poly_method, hidden_dim, attn_dim, attn_heads,
            clamp, attn_drop_prob, weight_fn, agg, act
        )

        # -------- Deg Scaler Option ------
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, hidden_dim, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.norm1_h = nn.LayerNorm(hidden_dim)
            self.norm2_h = nn.LayerNorm(hidden_dim)
            self.norm1_e = nn.LayerNorm(hidden_dim)
            self.norm2_e = nn.LayerNorm(hidden_dim)

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.norm1_h = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            self.norm2_h = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            self.norm1_e = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            self.norm2_e = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)

        if not self.layer_norm and not self.batch_norm:
            self.norm1_h = nn.Identity()
            self.norm2_h = nn.Identity()
            self.norm1_e = nn.Identity()
            self.norm2_e = nn.Identity()

        self.FFN_h_layer1 = nn.Linear(hidden_dim, int(hidden_dim * 1.5))
        self.FFN_h_layer2 = nn.Linear(int(hidden_dim * 1.5), hidden_dim)
        self.FFN_e_layer1 = nn.Linear(hidden_dim, int(hidden_dim * 1.5))
        self.FFN_e_layer2 = nn.Linear(int(hidden_dim * 1.5), hidden_dim)

    def forward(self, batch):
        h_res = batch.x
        e_res = batch[self.poly_method + "_conn"]

        h_attn_out, e_attn_out = self.message_pass(batch)

        h = h_res + h_attn_out
        if self.deg_scaler:
            sqrt_deg = batch["sqrt_deg"]
            h = torch.stack([h, h * sqrt_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        # h = h_res = h + h_res
        h = self.norm1_h(h)
        h = F.dropout(h, self.drop_prob, training=self.training)
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.drop_prob, training=self.training)
        h = self.FFN_h_layer2(h)
        h = h + h_res
        h = self.norm2_h(h)

        e = e_res = e_attn_out + e_res
        e = self.norm1_e(e)
        e = F.dropout(e, self.drop_prob, training=self.training)
        e = self.FFN_e_layer1(e)
        e = self.act(e)
        e = F.dropout(e, self.drop_prob, training=self.training)
        e = self.FFN_e_layer2(e)
        e = e + e_res
        e = self.norm2_e(e)

        batch.x = h
        batch[self.poly_method + "_conn"] = e
        return batch
