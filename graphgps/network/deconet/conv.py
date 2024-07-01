from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.register import act_dict
from torch_sparse import SparseTensor, matmul
from yacs.config import CfgNode


class DecoConv(nn.Module):
    def __init__(self, num_kernels, in_channel, out_channel, cfg: CfgNode):
        super().__init__()
        self.num_kernels = num_kernels
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.lin = nn.Linear(num_kernels * in_channel, out_channel, cfg.bias)
        if cfg.batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_channel)
        else:
            self.batch_norm = None
        self.act = act_dict[cfg.act]()
        self.dropout = nn.Dropout(cfg.drop_prob)

    def forward(self, batch: Data | Batch, poly_adj: SparseTensor):
        x = batch["x"]
        h = matmul(poly_adj, x)
        h = h.reshape((x.shape[0], self.num_kernels * self.in_channel))
        h = self.lin(h)
        if self.batch_norm is not None:
            h = self.batch_norm(h)
        h = self.act(h)
        h = self.dropout(h)
        batch["x"] = h
        return batch
