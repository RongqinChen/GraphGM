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
        self.deg_scaler = deg_scaler
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.weight_fn = weight_fn
        self.agg = agg
        self.act = act_dict[act]()
        self.attn_dropout = nn.Dropout(attn_drop_prob)
        self.act = act_dict[act]()

        self.qkv_weight = nn.Parameter(torch.empty((3 * attn_dim * attn_heads, hidden_dim)))
        self.qkv_bias = nn.Parameter(torch.empty(3 * attn_dim * attn_heads))
        self.E = nn.Linear(hidden_dim, 2 * attn_dim * attn_heads)

        self.Aw = nn.Parameter(torch.zeros(self.attn_dim, self.attn_heads, 1))
        self.Bw = nn.Parameter(torch.zeros(self.attn_dim, self.attn_heads, self.attn_dim))
        # -------- Deg Scaler Option ------
        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, hidden_dim, 2))

        self.conn_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.conn_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.FFN_h_layer1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.FFN_h_layer2 = nn.Linear(hidden_dim * 2, hidden_dim)

        if self.layer_norm:
            self.norm1_h = nn.LayerNorm(hidden_dim)
            self.norm2_h = nn.LayerNorm(hidden_dim)
            self.norm1_conn = nn.LayerNorm(hidden_dim)
            self.norm2_conn = nn.LayerNorm(hidden_dim)

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.norm1_h = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            self.norm2_h = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            self.norm1_conn = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)
            self.norm2_conn = nn.BatchNorm1d(hidden_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=self.bn_momentum)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_weight)
        nn.init.constant_(self.qkv_bias, 0.)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.constant_(self.E.bias, 0.)
        nn.init.xavier_normal_(self.Aw)
        nn.init.xavier_normal_(self.Bw)
        if self.deg_scaler:
            nn.init.xavier_normal_(self.deg_coef)
        nn.init.xavier_uniform_(self.conn_layer1.weight)
        nn.init.constant_(self.conn_layer1.bias, 0.)
        nn.init.xavier_uniform_(self.conn_layer2.weight)
        nn.init.constant_(self.conn_layer2.bias, 0.)
        nn.init.xavier_uniform_(self.FFN_h_layer1.weight)
        nn.init.constant_(self.FFN_h_layer1.bias, 0.)
        nn.init.xavier_uniform_(self.FFN_h_layer2.weight)
        nn.init.constant_(self.FFN_h_layer2.bias, 0.)
        if self.layer_norm or self.batch_norm:
            self.norm1_h.reset_parameters()
            self.norm2_h.reset_parameters()
            self.norm1_conn.reset_parameters()
            self.norm2_conn.reset_parameters()

    def compute_score(self, conn, dst):
        conn = conn.view(-1, self.attn_heads, self.attn_dim).contiguous()
        score = oe.contract("ehd, dhc->ehc", conn, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)
        if self.weight_fn == 'softmax':
            score = pyg_softmax(score, dst)  # (num relative) x attn_heads x 1
        elif self.weight_fn == 'sigmoid':
            score = sigmoid(score)
        elif self.weight_fn == 'tanh':
            score = tanh(score)
        else:
            raise NotImplementedError

        score = self.attn_dropout(score)
        return score

    def forward(self, batch):
        x, e = batch.x, batch[self.poly_method + "_conn"]
        dst, src = batch[self.poly_method + "_index"]

        Qh, Kh, Vh = F._in_projection_packed(x, x, x, self.qkv_weight, self.qkv_bias)
        Qdst, Ksrc, Vsrc = Qh[dst], Kh[src], Vh[src]

        Eh = self.E(e)
        Eh = Eh.unflatten(-1, (2, self.attn_heads * self.attn_dim))
        Eh = Eh.unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        Ew, Eb = Eh[0], Eh[1]

        conn1 = (Qdst + Ksrc) * Ew
        conn2 = torch.sqrt(torch.relu(conn1)) - torch.sqrt(torch.relu(-conn1))
        conn3 = conn2 + Eb
        conn = self.act(conn3)
        score = self.compute_score(conn, dst)

        conn = F.dropout(conn, self.drop_prob, training=self.training)
        conn = self.conn_layer1(conn)
        msg = Vsrc + conn
        msg = msg.view(-1, self.attn_heads, self.attn_dim)
        agg = scatter(msg * score, dst, dim=0, dim_size=batch.num_nodes, reduce=self.agg)
        agg = agg.flatten(1)
        h = x + agg

        if self.deg_scaler:
            sqrt_deg = batch["sqrt_deg"]
            h = torch.stack([h, h * sqrt_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h_res = h
        h = self.norm1_h(h)
        h = F.dropout(h, self.drop_prob, training=self.training)
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.drop_prob, training=self.training)
        h = self.FFN_h_layer2(h)
        h = h + h_res
        h = self.norm2_h(h)

        conn = self.norm1_conn(conn)
        conn = self.act(conn)
        conn = F.dropout(conn, self.drop_prob, training=self.training)
        conn = self.conn_layer2(conn)
        conn = conn + e
        conn = self.norm2_conn(conn)

        batch.x = h
        batch[self.poly_method + "_conn"] = conn
        return batch
