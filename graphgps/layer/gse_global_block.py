from torch import nn
from torch_geometric.data import Batch
from .gse_grit_mp import GritMessagePassingLayer
from .gse_dense_attn import GraphDenseAttn
from torch_geometric.graphgym.register import register_layer


Layer_dict = {
    'grit': GritMessagePassingLayer,
    'dense': GraphDenseAttn,
}


@register_layer("GseGlobalBlock")
class GseGlobalBlock(nn.Module):
    def __init__(
        self, num_layers, layer_type, in_dim, out_dim, num_heads,
        dropout=0.0, attn_dropout=0.0, mlp_dropout=0.0, input_norm=True
    ) -> None:

        super().__init__()
        self.num_layers = num_layers
        Layer = Layer_dict[layer_type]
        self.layer_list = nn.ModuleList()
        if layer_type == 'grit':
            for _ in range(num_layers):
                layer = Layer(in_dim, out_dim, num_heads, dropout, attn_dropout)
                self.layer_list.append(layer)
        elif layer_type == 'dense':
            for _ in range(num_layers):
                layer = Layer(in_dim, num_heads, dropout, attn_dropout, mlp_dropout, input_norm)
                self.layer_list.append(layer)
        else:
            raise NotImplementedError

    def forward(self, batch: Batch):
        for idx in range(self.num_layers):
            batch = self.layer_list[idx](batch)
        return batch
