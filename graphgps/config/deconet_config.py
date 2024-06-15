from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('DecoNet')
def DecoNet_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    cfg.gnn.enable = False
    cfg.DecoNet = CN()
    cfg.DecoNet.enable = False

    cfg.DecoNet.hidden_dim = 64
    cfg.DecoNet.attn_heads = 8
    cfg.DecoNet.drop_prob = 0.0
    cfg.DecoNet.attn_drop_prob = 0.2
    cfg.DecoNet.bn_momentum = 0.1
    # cfg.DecoNet.bn_no_runner = False
    # cfg.DecoNet.rezero = False
    cfg.DecoNet.batch_norm = True
    cfg.DecoNet.bias = True
    cfg.DecoNet.deg_scaler = True
    cfg.DecoNet.clamp = 5.
    cfg.DecoNet.weight_fn = 'softmax'
    cfg.DecoNet.agg = 'add'
    cfg.DecoNet.act = 'relu'
    cfg.DecoNet.pe_layer = 'simple_linear'

    cfg.DecoNet.conv = CN()
    cfg.DecoNet.conv.repeats = 2
    cfg.DecoNet.conv.num_blocks = 3

    cfg.DecoNet.full = CN()
    cfg.DecoNet.full.enable = True
    cfg.DecoNet.full.repeats = 2
