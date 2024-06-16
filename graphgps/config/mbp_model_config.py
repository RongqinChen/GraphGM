from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('MbpModel')
def mbp_model_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    cfg.gnn.enable = False
    cfg.MbpModel = CN()
    cfg.MbpModel.enable = False

    cfg.MbpModel.hidden_dim = 64
    cfg.MbpModel.attn_heads = 8
    cfg.MbpModel.drop_prob = 0.0
    cfg.MbpModel.attn_drop_prob = 0.2
    cfg.MbpModel.residual = True
    cfg.MbpModel.layer_norm = False
    cfg.MbpModel.batch_norm = True
    cfg.MbpModel.bn_momentum = 0.1
    cfg.MbpModel.bn_no_runner = False
    cfg.MbpModel.rezero = False
    cfg.MbpModel.deg_scaler = True
    cfg.MbpModel.clamp = 5.
    cfg.MbpModel.weight_fn = 'softmax'
    cfg.MbpModel.agg = 'add'
    cfg.MbpModel.act = 'relu'
    cfg.MbpModel.pe_layer = 'simple_linear'

    cfg.MbpModel.poly = CN()
    cfg.MbpModel.poly.layer_type = 'cattn'
    cfg.MbpModel.poly.repeats = 2
    cfg.MbpModel.poly.num_blocks = 3
    cfg.MbpModel.jumping_knowledge = False

    cfg.MbpModel.full = CN()
    cfg.MbpModel.full.layer_type = 'cattn'
    cfg.MbpModel.full.enable = True
    cfg.MbpModel.full.repeats = 2
    cfg.MbpModel.full.input_norm = False
