from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('gse_gnn')
def gse_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    cfg.gnn.enable = False
    cfg.gse_model = CN()
    cfg.gse_model.enable = False

    cfg.gse_model.hidden_dim = 64
    cfg.gse_model.attn_heads = 8
    cfg.gse_model.drop_prob = 0.0
    cfg.gse_model.attn_drop_prob = 0.2
    cfg.gse_model.residual = True
    cfg.gse_model.layer_norm = False
    cfg.gse_model.batch_norm = True
    cfg.gse_model.bn_momentum = 0.1
    cfg.gse_model.bn_no_runner = False
    cfg.gse_model.rezero = False
    cfg.gse_model.deg_scaler = True
    cfg.gse_model.clamp = 5.
    cfg.gse_model.weight_fn = 'softmax'
    cfg.gse_model.agg = 'add'
    cfg.gse_model.act = 'relu'

    cfg.gse_model.messaging = CN()
    cfg.gse_model.messaging.layer_type = 'grit'
    cfg.gse_model.messaging.repeats = 2
    cfg.gse_model.messaging.num_blocks = 3

    cfg.gse_model.full = CN()
    cfg.gse_model.full.layer_type = 'grit'
    cfg.gse_model.full.enable = True
    cfg.gse_model.full.repeats = 2
    cfg.gse_model.full.input_norm = False
    
    # TODO: remove
    cfg.gse_model.head = 'san_graph'
