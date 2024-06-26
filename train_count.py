"""
script to train on counting substructure tasks.
"""

from graphgps.loader.dataset.GraphCountDataset import GraphCountDatasetI2
import torch
import torch.nn as nn
import train_utils
from interfaces.pl_model_interface import PlGNNTestonValModule
from interfaces.pl_data_interface import PlPyGDataTestonValModule
from lightning.pytorch import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from lightning.pytorch.callbacks.progress import TQDMProgressBar
import wandb
from torchmetrics import MeanAbsoluteError
from torch_geometric.data import Data
from graphgps.transform.polynomials import compute_polynomials
from torch_geometric.transforms import BaseTransform


class ComputePolynomialBases(BaseTransform):
    r"""Polynomial Bases"""

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, in_data: Data) -> Data:
        paras = self.args.posenc_Poly
        data = compute_polynomials(in_data, paras.method, paras.power, paras.add_full_edge_index)
        return data

    def __repr__(self) -> str:
        paras = self.args.posenc_Poly
        return f"{paras.method}.{paras.power}.{paras.add_full_edge_index}"


class SetTarget(BaseTransform):
    r"""Polynomial Bases"""

    def __init__(self, target):
        super().__init__()
        self.target = target

    def forward(self, data: Data) -> Data:
        data.y = data.y[:, self.target]
        return data


def main():
    parser = train_utils.args_setup()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="count_cycle",
        choices=("count_cycle", "count_graphlet"),
        help="Name of dataset.",
    )
    parser.add_argument(
        "--task", type=int, default=0, choices=(0, 1, 2, 3, 4), help="Train task index."
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of repeat run.")
    args = parser.parse_args()
    args = train_utils.update_args(args)

    train_dataset = GraphCountDatasetI2(
        root='datasets',
        dataname=args.dataset_name,
        split="train",
        pre_transform=ComputePolynomialBases(args),
        transform=SetTarget(args.task)
    )

    val_dataset = GraphCountDatasetI2(
        root='datasets',
        dataname=args.dataset_name,
        split="val",
        pre_transform=ComputePolynomialBases(args),
        transform=SetTarget(args.task)
    )

    test_dataset = GraphCountDatasetI2(
        root='datasets',
        dataname=args.dataset_name,
        split="test",
        pre_transform=ComputePolynomialBases(args),
        transform=SetTarget(args.task)
    )

    y_train_val = torch.cat([train_dataset.data.y, val_dataset.data.y], dim=0)
    mean = y_train_val.mean(dim=0)
    std = y_train_val.std(dim=0)
    train_dataset.data.y = (train_dataset.data.y - mean) / std
    val_dataset.data.y = (val_dataset.data.y - mean) / std
    test_dataset.data.y = (test_dataset.data.y - mean) / std

    for i in range(1, args.runs + 1):
        logger = WandbLogger(
            name=f"run_{str(i)}",
            project=args.exp_name,
            save_dir=args.save_dir,
            offline=True,
        )
        logger.log_hyperparams(args)
        timer = Timer(duration=dict(weeks=4))

        # Set random seed
        seed = train_utils.get_seed(args.seed)
        seed_everything(seed)

        datamodule = PlPyGDataTestonValModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        loss_cri = nn.L1Loss()
        evaluator = MeanAbsoluteError()
        args.mode = "min"
        modelmodule = PlGNNTestonValModule(
            loss_criterion=loss_cri,
            evaluator=evaluator,
            args=args,
        )
        trainer = Trainer(
            accelerator="auto",
            devices="auto",
            max_epochs=args.num_epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=logger,
            callbacks=[
                TQDMProgressBar(refresh_rate=20),
                ModelCheckpoint(monitor="val/metric", mode=args.mode),
                LearningRateMonitor(logging_interval="epoch"),
                timer,
            ],
        )

        trainer.fit(modelmodule, datamodule=datamodule)
        val_result, test_result = trainer.test(
            modelmodule, datamodule=datamodule, ckpt_path="best"
        )
        results = {
            "final/best_val_metric": val_result["val/metric"],
            "final/best_test_metric": test_result["test/metric"],
            "final/avg_train_time_epoch": timer.time_elapsed("train") / args.num_epochs,
        }
        logger.log_metrics(results)
        with open(f'{args.save_dir}/{args.exp_name}/results.txt', 'a') as afile:
            print(results, file=afile)
        wandb.finish()

    return


if __name__ == "__main__":
    main()
