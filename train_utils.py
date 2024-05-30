"""
Utils file for training.
"""

import argparse
import os
import shutil
import time
import torch

# import data_utils
import yaml
from torch_geometric.graphgym.config import cfg
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import StratifiedKFold
from typing import Callable, Tuple


def args_setup():
    r"""Setup argparser."""
    parser = argparse.ArgumentParser("arguments for training and testing")
    # common args
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./save",
        help="Base directory for saving information.",
    )
    parser.add_argument(
        "--seed", type=int, default=234, help="Random seed for reproducibility."
    )

    # training args
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size per GPU."
    )
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Minimum learning rate."
    )
    parser.add_argument("--l2_wd", type=float, default=0.0, help="L2 weight decay.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument(
        "--test_eval_interval",
        type=int,
        default=10,
        help="Interval between validation on test dataset.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.5,
        help="Factor in the ReduceLROnPlateau learning rate scheduler.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience in the ReduceLROnPlateau learning rate scheduler.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="If true, save the wandb log offline. " "Mainly use for debug.",
    )

    # data args
    parser.add_argument(
        "--reprocess", action="store_true", help="Whether to reprocess the dataset"
    )
    parser.add_argument("--conn_mul", action="store_true")

    # model args
    parser.add_argument(
        "--h_dim", type=int, default=96, help="Hidden size of the model."
    )
    parser.add_argument("--num_doubling_layer", type=int, default=5)
    parser.add_argument("--num_full_layer", type=int, default=2)
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        choices=("sum", "max", "mean", "attention", "last", "concat"),
        help="Jumping knowledge method.",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="If ture, use residual connection between each layer.",
    )
    parser.add_argument(
        "--initial_eps", type=float, default=0.0, help="Initial epsilon in GIN."
    )
    parser.add_argument(
        "--train_eps", action="store_true", help="If true, the epsilon is trainable."
    )
    parser.add_argument(
        "--drop_prob",
        type=float,
        default=1 / 8,
        help="Probability of zeroing an activation in dropout models.",
    )
    parser.add_argument("--conn_drop_prob", type=float, default=1 / 8)
    parser.add_argument("--node_drop_prob", type=float, default=1 / 8)
    parser.add_argument(
        "--norm_type",
        type=str,
        default="Batch",
        choices=("Batch", "Layer", "Instance", "GraphSize", "Pair", "None"),
        help="Normalization method in model.",
    )
    parser.add_argument(
        "--act_type",
        type=str,
        default="ReLU",
        choices=("identity", "relu", "gelu", "sigmoid", "tanh"),
        help="Activation function in model.",
    )

    return parser


def get_exp_name(args: argparse.ArgumentParser, add_task=True) -> str:
    """Get experiment name.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """

    arg_list = []
    # if "task" in args and add_task:
    #     arg_list = [str(args.task)]
    # arg_list.extend(
    #     [
    #         args.dataset_name,
    #         str(args.num_doubling_layer),
    #         str(args.num_full_layer),
    #         str(args.h_dim),
    #     ]
    # )

    # if args.residual:
    #     arg_list.append("residual")
    # if args.conn_mul:
    #     arg_list.append("conn_mul")

    exp_name = "_".join(arg_list)
    return exp_name + f"-{time.strftime('%Y%m%d%H%M%S')}"


class CfgWrapper():
    def __init__(self, cfg) -> None:
        self.__opts = cfg.__dict__ if hasattr(cfg, '__dict__') else cfg

    def __getattr__(self, name: str):
        if name not in self.__opts:
            print('cfg missing', name)
            return None

        val = self.__opts[name]
        if isinstance(val, dict):
            return CfgWrapper(val)
        else:
            return val


def update_args(
    args: argparse.ArgumentParser, add_task=True
) -> argparse.ArgumentParser:
    r"""Update argparser given config file.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """

    if args.config_file is not None:
        with open(args.config_file) as f:
            cfg_dict = yaml.safe_load(f)
        for key, value in cfg_dict.items():
            if isinstance(value, list):
                for v in value:
                    getattr(args, key, []).append(CfgWrapper(v))
            else:
                setattr(args, key, CfgWrapper(value))
    args.exp_name = get_exp_name(args, add_task)
    cfg.update(args.__dict__)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return cfg


def data_setup(args: argparse.ArgumentParser) -> Tuple[str, Callable, list]:
    r"""Setup data for experiment.
    Args:
        args (ArgumentParser): Arguments dict from argparser.
    """
    path_arg_list = [f"data/{args.dataset_name}"]
    path = "_".join(path_arg_list)
    if os.path.exists(path + "/processed") and args.reprocess:
        shutil.rmtree(path + "/processed")

    return path


class PostTransform(object):
    r"""Post transformation of dataset.
    Args:
        wo_node_feature (bool): If true, remove path encoding from model.
        wo_edge_feature (bool): If true, remove edge feature from model.
        task (int): Specify the task in dataset if it has multiple targets.
    """

    def __init__(self, wo_node_feature: bool, wo_edge_feature: bool, task: int = None):
        self.wo_node_feature = wo_node_feature
        self.wo_edge_feature = wo_edge_feature
        self.task = task

    def __call__(self, data: Data) -> Data:
        if "x" not in data:
            data.x = torch.zeros([data.num_nodes, 1]).long()

        if self.wo_edge_feature:
            data.edge_attr = None
        if self.wo_node_feature:
            data.x = torch.zeros_like(data.x)
        if self.task is not None:
            data.y = data.y[:, self.task]
        return data


def k_fold(dataset: Dataset, folds: int, seed: int) -> Tuple[list, list, list]:
    r"""Dataset split for K-fold cross-validation.
    Args:
        dataset (Dataset): The dataset to be split.
        folds (int): Number of folds.
        seed (int): Random seed.
    """
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    for _, idx in skf.split(
        torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]
    ):
        test_indices.append(torch.from_numpy(idx).long())

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset)).long()
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def get_seed(seed=234) -> int:
    r"""Return random seed based on current time.
    Args:
        seed (int): base seed.
    """
    t = int(time.time() * 1000.0)
    seed = (
        seed
        + ((t & 0xFF000000) >> 24)
        + ((t & 0x00FF0000) >> 8)
        + ((t & 0x0000FF00) << 8)
        + ((t & 0x000000FF) << 24)
    )
    return seed
