from torch import nn
from torch_geometric.data import Batch
from .gse_grit_mp import GritMessagePassingLayer
from .gse_dense_attn import GraphDenseAttn
from torch_geometric.graphgym.register import register_layer


Layer_dict = {
    'grit': GritMessagePassingLayer,
    'dense': GraphDenseAttn,
}


@register_layer("GseFullBlock")
class GseFullBlock(nn.Module):
    def __init__(self, repeats, layer_type, cfg) -> None:

        super().__init__()
        self.repeats = repeats
        Layer = Layer_dict[layer_type]
        self.layer_list = nn.ModuleList()
        if layer_type == 'grit':
            for _ in range(repeats):
                layer = Layer(
                    cfg.hidden_dim, cfg.attn_heads, cfg.drop_prob,
                    cfg.attn_drop_prob, cfg.residual, cfg.layer_norm,
                    cfg.batch_norm, cfg.bn_momentum, cfg.bn_no_runner,
                    cfg.rezero, cfg.deg_scaler, cfg.clamp,
                    cfg.weight_fn, cfg.agg, cfg.act,
                )
                self.layer_list.append(layer)
        elif layer_type == 'dense':
            for _ in range(repeats):
                layer = Layer(
                    cfg.hidden_dim, cfg.attn_heads, cfg.drop_prob, cfg.attn_drop_prob,
                    cfg.drop_prob, cfg.full.input_norm
                )
                self.layer_list.append(layer)
        else:
            raise NotImplementedError

    def forward(self, batch: Batch):
        for idx in range(self.repeats):
            batch = self.layer_list[idx](batch)
        return batch
