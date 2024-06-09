import warnings

import numpy as np
import opt_einsum as oe
import torch
import torch.nn.functional as F
import torch_geometric as pyg
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
        self.Q = nn.Linear(hidden_dim, attn_dim * attn_heads, bias=False)
        self.K = nn.Linear(hidden_dim, attn_dim * attn_heads, bias=False)
        self.V = nn.Linear(hidden_dim, attn_dim * attn_heads, bias=False)
        self.Ew = nn.Linear(hidden_dim, attn_dim * attn_heads, bias=False)
        self.Eb = nn.Linear(hidden_dim, attn_dim * attn_heads, bias=True)
        self.Eo = nn.Linear(attn_dim * attn_heads, attn_dim * attn_heads, bias=True)
        self.Aw = nn.Parameter(torch.zeros(self.attn_dim, self.attn_heads, 1))
        self.BW = nn.Parameter(torch.zeros(self.attn_dim, self.attn_heads, self.attn_dim))
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.xavier_normal_(self.Ew.weight)
        nn.init.xavier_normal_(self.Eb.weight)
        nn.init.xavier_normal_(self.Eo.weight)
        nn.init.xavier_normal_(self.Aw)
        nn.init.xavier_normal_(self.BW)
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
        conn = self.Eo(conn)
        # output edge
        batch.Oe = conn

        # final attn
        conn = conn.view(-1, self.attn_heads, self.attn_dim)
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
        conn = conn.view(-1, self.attn_heads, self.attn_dim)
        agg = scatter(Vsrc * score, dst, dim=0, reduce=self.agg)
        rowV = scatter(conn * score, dst, dim=0, reduce=self.agg)
        rowV = oe.contract("nhd, dhc -> nhc", rowV, self.BW, backend="torch")
        On = agg + rowV
        batch.On = On.flatten(1)

    def forward(self, batch):
        batch.Qh = self.Q(batch.x)
        batch.Kh = self.K(batch.x)
        batch.Vh = self.V(batch.x)
        batch.Ew = self.Ew(batch[self.poly_method + "_conn"])
        batch.Eb = self.Eb(batch[self.poly_method + "_conn"])
        self.propagate_attention(batch)
        h_out = batch.On
        e_out = batch.Oe
        return h_out, e_out


class GritMessagePassingLayer(nn.Module):
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

        self.O_h = nn.Linear(hidden_dim, hidden_dim)
        self.O_e = nn.Linear(hidden_dim, hidden_dim)

        # -------- Deg Scaler Option ------
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, hidden_dim, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(hidden_dim)
            self.layer_norm1_e = nn.LayerNorm(hidden_dim)

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            self.batch_norm1_e = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.FFN_h_layer2 = nn.Linear(hidden_dim * 2, hidden_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(hidden_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, batch):
        h_in1 = batch.x  # for first residual connection
        e_in1 = batch[self.poly_method + "_conn"]
        # multi-head attention out
        h_attn_out, e_attn_out = self.message_pass(batch)

        # h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h_attn_out, self.drop_prob, training=self.training)
        # degree scaler
        if self.deg_scaler:
            sqrt_deg = get_sqrt_deg(batch)
            h = torch.stack([h, h * sqrt_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        # e = e_attn_out.flatten(1)
        e = F.dropout(e_attn_out, self.drop_prob, training=self.training)
        e = self.O_e(e)

        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
            h = h_in1 + h  # residual connection
            if e is not None:
                if self.rezero:
                    e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.drop_prob, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero:
                h = h * self.alpha2_h
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)
        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        batch[self.poly_method + "_conn"] = e
        return batch


@torch.no_grad()
def get_sqrt_deg(batch):
    if "sqrt_deg" in batch:
        sqrt_deg = batch.sqrt_deg
    elif "deg" in batch:
        deg = batch.deg
        sqrt_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        warnings.warn("Compute the degree on the fly; Might be problematric if have applied edge-padding to complete graphs")
        deg = pyg.utils.degree(batch.poly_idx[1], num_nodes=batch.num_nodes, dtype=torch.float)
        sqrt_deg = torch.log(deg + 1)
    sqrt_deg = sqrt_deg.view(batch.num_nodes, 1)
    return sqrt_deg
