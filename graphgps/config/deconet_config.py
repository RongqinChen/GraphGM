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

    cfg.DecoNet.global_attention = CN()
    cfg.DecoNet.global_attention.enable = True
    cfg.DecoNet.global_attention.num_layers = 1
    cfg.DecoNet.global_attention.hidden_dim = 64
    cfg.DecoNet.global_attention.attn_heads = 8
    cfg.DecoNet.global_attention.clamp = 5
    cfg.DecoNet.global_attention.attn_drop_prob = 0.2
    cfg.DecoNet.global_attention.drop_prob = 0.2
    cfg.DecoNet.global_attention.weight_fn = 'softmax'
    cfg.DecoNet.global_attention.agg = 'add'
    cfg.DecoNet.global_attention.act = 'relu'
    cfg.DecoNet.global_attention.bn_momentum = 0.1
    cfg.DecoNet.pe_layer = 'simple_linear'
