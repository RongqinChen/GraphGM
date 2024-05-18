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

    cfg.gse_model.hidden_dim = 96
    cfg.gse_model.num_heads = 12
    cfg.gse_model.dropout = 0.0
    cfg.gse_model.attn_dropout = 0.2
    cfg.gse_model.weight_fn = 'sigmoid'
    cfg.gse_model.agg = 'add'

    cfg.gse_model.messaging = CN()
    cfg.gse_model.messaging.layer_type = 'grit'
    cfg.gse_model.messaging.repeats = 1
    cfg.gse_model.messaging.num_layers = 3

    cfg.gse_model.full = CN()
    cfg.gse_model.full.layer_type = 'grit'
    cfg.gse_model.full.repeats = 1
    cfg.gse_model.full.num_layers = 3

    cfg.gse_model.head = 'san_graph'
