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



class ConditionalMessagePassing(nn.Module):

    def __init__(self, poly_method, hidden_dim, attn_dim, attn_heads, clamp, attn_drop_prob, weight_fn, agg, act):

        self.poly_method = poly_method
        self.attn_dim = attn_dim
        self.attn_heads = attn_heads
        self.weight_fn = weight_fn
        self.agg = agg
        self.dropout = nn.Dropout(attn_drop_prob)

