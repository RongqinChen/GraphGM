from torch import nn
from torch_geometric.data import Batch
from .biconn_grit_mp import GritMessagePassingLayer
from torch_geometric.graphgym.register import register_layer
from yacs.config import CfgNode


Layer_dict = {
    'grit': GritMessagePassingLayer
}


@register_layer("BiConnMessagingBlock")
class BiConnMessagingBlock(nn.Module):
    def __init__(self, poly_method, repeats, cfg: CfgNode) -> None:
        super().__init__()
        self.repeats = repeats
        Layer = Layer_dict[cfg.messaging.layer_type]
        self.layer_list = nn.ModuleList()
        if cfg.messaging.layer_type == 'grit':
            for _ in range(self.repeats):
                layer = Layer(
                    poly_method,
                    cfg.hidden_dim, cfg.attn_heads, cfg.drop_prob,
                    cfg.attn_drop_prob, cfg.residual, cfg.layer_norm,
                    cfg.batch_norm, cfg.bn_momentum, cfg.bn_no_runner,
                    cfg.rezero, cfg.deg_scaler, cfg.clamp,
                    cfg.weight_fn, cfg.agg, cfg.act,
                )
                self.layer_list.append(layer)
        else:
            raise NotImplementedError

    def forward(self, batch: Batch):
        for idx in range(self.repeats):
            batch = self.layer_list[idx](batch)
        return batch
