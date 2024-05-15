import numpy as np
import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.graphgym.register import act_dict, register_layer
from torch_scatter import scatter, scatter_add, scatter_max


def pyg_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int,): The number of nodes.

    :rtype: :class:`Tensor`
    """
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out


class AdditiveAttn(nn.Module):
    def __init__(
        self, in_dim, h_dim, num_heads, use_bias, clamp=5.0, dropout=0.0, act=None
    ):
        super().__init__()
        self.out_dim = h_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.Q = nn.Linear(in_dim, h_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, h_dim * num_heads, bias=True)
        self.E = nn.Linear(in_dim, h_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, h_dim * num_heads, bias=True)
        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1))
        self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim))
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)
        nn.init.xavier_normal_(self.Aw)
        nn.init.xavier_normal_(self.VeRow)
        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

    def propagate_attention(self, gbatch: Data):
        src, dst = gbatch.edge_index
        src_k = gbatch.Kh[src]
        dst_q = gbatch.Qh[dst]
        score1 = src_k + dst_q
        Ex = gbatch.Ex.view(-1, self.num_heads, self.out_dim * 2)
        Ex1, Ex2 = Ex[:, :, :self.out_dim], Ex[:, :, self.out_dim:]

        score2 = Ex1 * Ex2  # num_edges, num_heads, out_dim
        score2 = torch.sqrt(torch.relu(score2)) - torch.sqrt(torch.relu(-score2))
        score = score1 + score2
        conn = self.act(score)
        gbatch.Eo = conn

        # final attn
        score = oe.contract("ehd, dhc->ehc", conn, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        # raw_attn = score
        score = pyg_softmax(score, dst, gbatch.num_nodes)
        # (num relative) x num_heads x 1
        score = self.dropout(score)
        gbatch.attn = score

        # Aggregate with Attn-Score
        msg = gbatch.Vh[src] * score
        # (num relative) x num_heads x out_dim
        Vo = gbatch.Qh + scatter(msg, dst, dim=0, reduce="add")
        rowV = scatter(conn * score, dst, dim=0, reduce="add")
        rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
        Vo = Vo + rowV
        gbatch.Vo = Vo

    def forward(self, gbatch: Data):
        Qh = self.Q(gbatch.x)
        Kh = self.K(gbatch.x)
        Vh = self.V(gbatch.x)
        gbatch.Ex = self.E(gbatch.edge_attr)
        gbatch.Qh = Qh.view(-1, self.num_heads, self.out_dim)
        gbatch.Kh = Kh.view(-1, self.num_heads, self.out_dim)
        gbatch.Vh = Vh.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(gbatch)
        n_out = gbatch.Vo.flatten(1)
        e_out = gbatch.Eo.flatten(1)
        return n_out, e_out


@register_layer("AdditiveAttn")
class AdditiveAttnLayer(nn.Module):
    def __init__(
        self, in_dim, out_dim, num_heads, dropout=0.0, attn_dropout=0.0,
        layer_norm=False, batch_norm=True, residual=True,
        act="relu", norm_e=True, cfg=dict(), **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner
        self.rezero = cfg.get("rezero", False)
        self.act = act_dict[act]()
        if cfg.get("attn", None) is None:
            cfg.attn = dict()

        head_dim = in_dim // num_heads
        self.attention = AdditiveAttn(
            in_dim=in_dim,
            h_dim=head_dim,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            clamp=cfg.attn.get("clamp", 5.0),
            dropout=attn_dropout,
            act=cfg.attn.get("act", "relu"),
        )
        self.deg_coef = nn.Parameter(torch.zeros(1, head_dim * num_heads, 2))
        nn.init.xavier_normal_(self.deg_coef)
        self.No = nn.Linear(head_dim * num_heads, out_dim)
        self.Eo = nn.Linear(head_dim * num_heads, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small,
            # use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_n = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not self.bn_no_runner,
                eps=1e-5, momentum=cfg.bn_momentum,
            )
            if norm_e:
                self.batch_norm1_e = nn.BatchNorm1d(
                    out_dim,
                    track_running_stats=not self.bn_no_runner,
                    eps=1e-5, momentum=cfg.bn_momentum,
                )
            else:
                self.batch_norm1_e = nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not self.bn_no_runner,
                eps=1e-5,
                momentum=cfg.bn_momentum,
            )

        if self.rezero:
            self.alpha1_n = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, gbatch: Data):
        nr1 = gbatch.x
        er1 = gbatch.edge_attr
        log_deg = gbatch.log_deg
        # for first residual connection
        nh, eh = self.attention(gbatch)
        nh = F.dropout(nh, self.dropout, training=self.training)
        eh = F.dropout(eh, self.dropout, training=self.training)
        # degree scaler
        nh = torch.stack([nh, nh * log_deg], dim=-1)
        nh = (nh * self.deg_coef).sum(dim=-1)
        nh = self.No(nh)
        eh = self.Eo(eh)

        if self.residual:
            if self.rezero:
                nh = nh * self.alpha1_n
                eh = eh * self.alpha1_e
            nh = nr1 + nh  # residual connection
            eh = er1 + eh  # residual connection

        if self.layer_norm:
            nh = self.layer_norm1_h(nh)
            eh = self.layer_norm1_e(eh)

        if self.batch_norm:
            nh = self.batch_norm1_n(nh)
            eh = self.batch_norm1_e(eh)

        nr2 = nh  # for second residual connection
        nh = self.FFN_h_layer1(nh)
        nh = self.act(nh)
        nh = F.dropout(nh, self.dropout, training=self.training)
        nh = self.FFN_h_layer2(nh)

        if self.residual:
            if self.rezero:
                nh = nh * self.alpha2_h
            nh = nr2 + nh  # residual connection

        if self.layer_norm:
            nh = self.layer_norm2_h(nh)

        if self.batch_norm:
            nh = self.batch_norm2_h(nh)

        gbatch.x = nh
        gbatch.edge_attr = eh
        return gbatch
