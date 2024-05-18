import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add
from torch_geometric.graphgym.register import act_dict

import opt_einsum as oe

import warnings


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

    def __init__(self, in_dim, out_dim, num_heads, clamp=5., dropout=0.,
                 weight_fn='softmax', agg='add', act=None, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.weight_fn = weight_fn
        self.agg = agg
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.Ew = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.Eb = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.Eo = nn.Linear(out_dim * num_heads, out_dim * num_heads)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.xavier_normal_(self.Ew.weight)
        nn.init.xavier_normal_(self.Eb.weight)
        nn.init.xavier_normal_(self.Eo.weight)
        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1))
        self.BW = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim))
        nn.init.xavier_normal_(self.Aw)
        nn.init.xavier_normal_(self.BW)
        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

    def propagate_attention(self, batch):
        dst, src = batch.poly_edge_index
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

        Eo = self.Eo(conn)
        # output edge
        batch.Oe = Eo.flatten(1)

        # final attn
        score = oe.contract("ehd, dhc->ehc", conn, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        # raw_attn = score
        if self.weight_fn == 'softmax':
            score = pyg_softmax(score, dst)  # (num relative) x num_heads x 1
        elif self.weight_fn == 'sigmoid':
            score = sigmoid(score)
        elif self.weight_fn == 'tanh':
            score = tanh(score)
        else:
            raise NotImplementedError

        score = self.dropout(score)

        # Aggregate with Attn-Score
        agg = scatter(Vsrc * score, dst, dim=0, reduce=self.agg)
        rowV = scatter(Eo * score, dst, dim=0, reduce=self.agg)
        rowV = oe.contract("nhd, dhc -> nhc", rowV, self.BW, backend="torch")
        batch.On = agg + rowV

    def forward(self, batch):
        Qh = self.Q(batch.x)
        Kh = self.K(batch.x)
        Vh = self.V(batch.x)
        Ew = self.Ew(batch.edge_attr)
        Eb = self.Eb(batch.edge_attr)

        batch.Qh = Qh.view(-1, self.num_heads, self.out_dim)
        batch.Kh = Kh.view(-1, self.num_heads, self.out_dim)
        batch.Vh = Vh.view(-1, self.num_heads, self.out_dim)
        batch.Ew = Ew.view(-1, self.num_heads, self.out_dim)
        batch.Eb = Eb.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch)
        h_out = batch.On
        e_out = batch.Oe
        return h_out, e_out


class GritMessagePassingLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, attn_dropout=0.0,
                 layer_norm=False, batch_norm=True, residual=True, act='relu',
                 norm_e=True, cfg=dict(), **kwargs):
        super().__init__()
        assert out_dim % num_heads == 0
        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner
        self.rezero = cfg.get("rezero", False)
        self.act = act_dict[act]() if act is not None else nn.Identity()
        if cfg.get("attn", None) is None:
            cfg.attn = dict()
        self.deg_scaler = cfg.attn.get("deg_scaler", True)

        att_dim = out_dim // num_heads
        self.message_pass = GritMessagePassing(
            in_dim=in_dim,
            out_dim=att_dim,
            num_heads=num_heads,
            clamp=cfg.attn.get("clamp", 5.),
            dropout=attn_dropout,
            weight_fn=cfg.attn.get("weight_fn", "softmax"),
            add=cfg.attn.get("add", "softmax"),
            agg=cfg.attn.get("agg", "add"),
            act=cfg.attn.get("act", "relu"),
        )

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        self.O_e = nn.Linear(out_dim // num_heads * num_heads, out_dim)

        # -------- Deg Scaler Option ------
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim // num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)
            if norm_e:
                self.batch_norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)
            else:
                self.batch_norm1_e = nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, batch):
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)
        h_in1 = batch.x  # for first residual connection
        e_in1 = batch.edge_attr
        # multi-head attention out
        h_attn_out, e_attn_out = self.message_pass(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)
        # degree scaler
        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)

        e = e_attn_out.flatten(1)
        e = F.dropout(e, self.dropout, training=self.training)
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
        h = F.dropout(h, self.dropout, training=self.training)
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
        batch.edge_attr = e
        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual,
            super().__repr__(),
        )


@torch.no_grad()
def get_log_deg(batch):
    if "log_deg" in batch:
        log_deg = batch.log_deg
    elif "deg" in batch:
        deg = batch.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        warnings.warn("Compute the degree on the fly; Might be problematric if have applied edge-padding to complete graphs")
        deg = pyg.utils.degree(batch.poly_edge_index[1], num_nodes=batch.num_nodes, dtype=torch.float)
        log_deg = torch.log(deg + 1)
    log_deg = log_deg.view(batch.num_nodes, 1)
    return log_deg
