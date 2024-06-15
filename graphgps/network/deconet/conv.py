from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.register import act_dict
from torch_sparse import SparseTensor, matmul
from yacs.config import CfgNode


class DecoConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias, batch_norm, act, drop_prob):
        super().__init__()
        self.lin = nn.Linear(in_channel, out_channel, bias)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_channel)
        else:
            self.batch_norm = None
        self.act = act_dict[act]()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, batch: Data | Batch, poly_adj):
        x = batch["x"]
        h = matmul(poly_adj, x)
        h = self.lin(h)
        h = h + x
        if self.batch_norm is not None:
            h = self.batch_norm(h)
        h = self.act(h)
        h = self.dropout(h)
        batch["x"] = x
        return batch


class DecoConvBlock(nn.Module):
    def __init__(self, poly_method, repeats, cfg: CfgNode):
        super().__init__()
        self.poly_method = poly_method
        self.repeats = repeats
        self.conv_list = nn.ModuleList()
        for _ in range(repeats):
            conv = DecoConv(
                cfg.hidden_dim, cfg.hidden_dim, cfg.bias,
                cfg.batch_norm, cfg.act, cfg.drop_prob
            )
            self.conv_list.append(conv)

    def forward(self, batch: Data | Batch, poly_adj: SparseTensor):
        for conv in self.conv_list:
            batch = conv(batch, poly_adj)
        return batch
