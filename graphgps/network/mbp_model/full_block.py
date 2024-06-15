from torch import nn
from torch_geometric.data import Batch
from torch_geometric.graphgym.register import register_layer
from yacs.config import CfgNode

from .dense_attn import GraphDenseAttn
from .cattn import ConditionalAttention
from .gine import GINE


Layer_dict = {
    'gine': GINE,
    'cattn': ConditionalAttention,
    'dense': GraphDenseAttn,
}


@register_layer("FullBlock")
class FullBlock(nn.Module):
    def __init__(self, repeats, cfg: CfgNode) -> None:
        super().__init__()
        self.repeats = repeats
        Layer = Layer_dict[cfg.full.layer_type]
        self.layer_list = nn.ModuleList()
        if cfg.full.layer_type == 'dense':
            for _ in range(repeats):
                layer = Layer(
                    cfg.hidden_dim, cfg.attn_heads, cfg.drop_prob, cfg.attn_drop_prob,
                    cfg.drop_prob, cfg.full.input_norm
                )
                self.layer_list.append(layer)
        elif cfg.full.layer_type in Layer_dict:
            for _ in range(repeats):
                layer = Layer(cfg)
                self.layer_list.append(layer)
        else:
            raise NotImplementedError

    def forward(self, batch: Batch):
        for idx in range(self.repeats):
            batch = self.layer_list[idx](batch)
        return batch
