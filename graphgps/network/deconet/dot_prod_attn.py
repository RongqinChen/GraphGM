import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data, Batch


class DotProductAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, attention_dropout: float):
        """Implementation of the DotProductAttention (Graphormer) layer.
        This layer is based on the implementation at:
        https://github.com/microsoft/Graphormer/tree/v1.0
        Note that this refers to v1 of Graphormer.

        Args:
            embed_dim: The number of hidden dimensions of the model
            num_heads: The number of heads of the Graphormer model
            dropout: Dropout applied after the attention and after the MLP
            attention_dropout: Dropout applied within the attention
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, attention_dropout, batch_first=True)
        self.input_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

        # We follow the paper in that all hidden dims are
        # equal to the embedding dim
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, data: Data | Batch):
        x = self.input_norm(data.x)
        x, real_nodes = to_dense_batch(x, data.batch)

        if hasattr(data, "attn_bias"):
            h1 = self.attention(x, x, x, ~real_nodes, attn_mask=data.attn_bias)[0][real_nodes]
        else:
            h1 = self.attention(x, x, x, ~real_nodes)[0][real_nodes]

        h2 = self.dropout(h1) + data.x
        data.x = self.mlp(h2) + h2
        return data
