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

    cfg.DecoNet.conv = CN()
    cfg.DecoNet.conv.num_layers = 7
    cfg.DecoNet.conv.k1_hidden_dim = 64
    cfg.DecoNet.conv.hidden_dim = 64
    cfg.DecoNet.conv.bias = True
    cfg.DecoNet.conv.batch_norm = True
    cfg.DecoNet.conv.act = 'identity'
    cfg.DecoNet.conv.drop_prob = 0.0

    cfg.DecoNet.gblock = CN()
    cfg.DecoNet.gblock.enable = True
    cfg.DecoNet.gblock.layer_type = 'dot_prod_attn'
    cfg.DecoNet.gblock.num_layers = 1
    cfg.DecoNet.gblock.hidden_dim = 64
    cfg.DecoNet.gblock.attn_heads = 8
    cfg.DecoNet.gblock.clamp = 5
    cfg.DecoNet.gblock.attn_drop_prob = 0.2
    cfg.DecoNet.gblock.drop_prob = 0.2
    cfg.DecoNet.gblock.weight_fn = 'softmax'
    cfg.DecoNet.gblock.agg = 'add'
    cfg.DecoNet.gblock.act = 'relu'
    cfg.DecoNet.gblock.bn_momentum = 0.1
    cfg.DecoNet.pe_layer = 'simple_linear'
