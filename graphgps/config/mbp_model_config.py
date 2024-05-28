from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('mbp_model')
def mbp_model_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    cfg.gnn.enable = False
    cfg.mbp_model = CN()
    cfg.mbp_model.enable = False

    cfg.mbp_model.hidden_dim = 64
    cfg.mbp_model.attn_heads = 8
    cfg.mbp_model.drop_prob = 0.0
    cfg.mbp_model.attn_drop_prob = 0.2
    cfg.mbp_model.residual = True
    cfg.mbp_model.layer_norm = False
    cfg.mbp_model.batch_norm = True
    cfg.mbp_model.bn_momentum = 0.1
    cfg.mbp_model.bn_no_runner = False
    cfg.mbp_model.rezero = False
    cfg.mbp_model.deg_scaler = True
    cfg.mbp_model.clamp = 5.
    cfg.mbp_model.weight_fn = 'softmax'
    cfg.mbp_model.agg = 'add'
    cfg.mbp_model.act = 'relu'

    cfg.mbp_model.messaging = CN()
    cfg.mbp_model.messaging.layer_type = 'grit'
    cfg.mbp_model.messaging.repeats = 2
    cfg.mbp_model.messaging.num_blocks = 3

    cfg.mbp_model.full = CN()
    cfg.mbp_model.full.layer_type = 'grit'
    cfg.mbp_model.full.enable = True
    cfg.mbp_model.full.repeats = 2
    cfg.mbp_model.full.input_norm = False
