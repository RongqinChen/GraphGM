from torch import nn
from torch_geometric.data import Batch
from .gse_grit_mp import GritMessagePassingLayer
from torch_geometric.graphgym.register import register_layer


Layer_dict = {
    'grit': GritMessagePassingLayer
}


@register_layer("GseMessagingBlock")
class GseMessagingBlock(nn.Module):
    def __init__(self, num_layers, layer_type, in_dim, out_dim, num_heads, dropout=0.0, attn_dropout=0.0) -> None:
        super().__init__()
        self.num_layers = num_layers
        Layer = Layer_dict[layer_type]
        self.layer_list = nn.ModuleList()
        if layer_type == 'grit':
            for _ in range(num_layers):
                layer = Layer(in_dim, out_dim, num_heads, dropout, attn_dropout)
                self.layer_list.append(layer)
        else:
            raise NotImplementedError

    def forward(self, batch: Batch):
        for idx in range(self.num_layers):
            batch = self.layer_list[idx](batch)
        return batch
