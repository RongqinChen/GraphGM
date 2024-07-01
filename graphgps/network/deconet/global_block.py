import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.graphgym.register import act_dict
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_add, scatter_max
from yacs.config import CfgNode

from .dot_prod_attn import DotProductAttention
from .cattn import ConditionalAttention


class GlobalBlock(nn.Module):

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.attn_list = nn.ModuleList()
        for _ in range(cfg.num_layers):
            if cfg.layer_type == 'dot_prod_attn':
                attn = DotProductAttention(cfg.hidden_dim, cfg.attn_heads, cfg.drop_prob, cfg.attn_drop_prob)
            elif cfg.layer_type == 'conditional':
                attn = ConditionalAttention(
                    cfg.hidden_dim, cfg.attn_heads, cfg.clamp, cfg.attn_drop_prob,
                    cfg.drop_prob, cfg.weight_fn, cfg.agg, cfg.act, cfg.bn_momentum
                )
            self.attn_list.append(attn)

    def forward(self, batch: Data | Batch):
        for attn in self.attn_list:
            batch = attn(batch)
        return batch
