import numpy as np
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


class MbpGINEMessagePassing(nn.Module):

    def __init__(self, poly_method, hidden_dim, attn_dim, attn_heads, clamp, attn_drop_prob, weight_fn, agg, act):
        super().__init__()
        self.poly_method = poly_method
        self.attn_dim = attn_dim
        self.attn_heads = attn_heads
        self.weight_fn = weight_fn
        self.agg = agg
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.dropout = nn.Dropout(attn_drop_prob)
        self.Q = nn.Linear(hidden_dim, attn_dim * attn_heads, bias=True)
        self.K = nn.Linear(hidden_dim, attn_dim * attn_heads, bias=True)
        self.E = nn.Linear(hidden_dim, attn_dim * attn_heads, bias=False)
        self.conn_lin = nn.Linear(attn_dim * attn_heads, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.conn_lin.weight)
        self.act = act_dict[act]()

    def forward(self, batch):
        Eh = self.E(batch[self.poly_method + "_conn"])
        dst, src = batch[self.poly_method + "_index"]
        Qdst = self.Q(batch.x)[dst]
        Ksrc = self.K(batch.x)[src]
        conn1 = Qdst + Ksrc + Eh
        conn2 = self.act(conn1)
        conn3 = self.dropout(conn2)
        conn = self.conn_lin(conn3)
        agg = scatter(conn, dst, dim=0, reduce=self.agg)
        return agg, conn


class MbpGINELayer(nn.Module):
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

        self.message_pass = MbpGINEMessagePassing(
            poly_method, hidden_dim, attn_dim, attn_heads,
            clamp, attn_drop_prob, weight_fn, agg, act
        )

        # -------- Deg Scaler Option ------
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, hidden_dim, 2))
            nn.init.xavier_normal_(self.deg_coef)

        self.nodelin = nn.Linear(hidden_dim, hidden_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(hidden_dim)
            self.layer_norm1_e = nn.LayerNorm(hidden_dim)

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            self.batch_norm1_e = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, batch):
        h_in1 = batch.x  # for first residual connection
        e_in1 = batch[self.poly_method + "_conn"]
        h_out, e_out = self.message_pass(batch)

        h = F.dropout(h_out, self.drop_prob, training=self.training)
        # degree scaler
        if self.deg_scaler:
            sqrt_deg = batch["sqrt_deg"]
            h = torch.stack([h, h * sqrt_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.nodelin(h)
        e = F.dropout(e_out, self.drop_prob, training=self.training)
        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
                e = e * self.alpha1_e
            h = h_in1 + h  # residual connection
            e = e_in1 + e

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        batch.x = h
        batch[self.poly_method + "_conn"] = e
        return batch
