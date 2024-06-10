import time

import torch
import torch.nn as nn
import torch_geometric

from loguru import logger
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_geometric.transforms import BaseTransform
import train_utils
from torch_geometric.data import Data
from graphgps.loader.dataset.BRECDataset_v3 import BRECDataset, part_dict
from graphgps.network.mbp_model import MbpModel
from graphgps.transform.polynomials import compute_polynomials

# part_name_list = ["Basic", "Extension", "CFI", "Regular", "4-Vertex_Condition", "Distance_Regular"]
part_name_list = ["CFI", "Regular", "4-Vertex_Condition", "Distance_Regular"]


class ComputePolynomialBases(BaseTransform):
    r"""Polynomial Bases"""

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, in_data: Data) -> Data:
        paras = self.args.posenc_Poly
        data = compute_polynomials(in_data, paras.method, paras.power, paras.add_full_edge_index)
        return data


def get_dataset(args):
    time_start = time.process_time()
    # transforms = ComputePolynomialBases(args)
    # if args.on_the_fly:
    #     dataset = BRECDataset(transform=transforms)
    # else:
    #     dataset = BRECDataset(pre_transform=transforms)

    dataset = BRECDataset()
    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")
    return dataset


def get_model(args):
    # time_start = time.process_time()
    model = MbpModel(None, 16)
    # time_end = time.process_time()
    # time_cost = round(time_end - time_start, 2)
    # logger.info(f"model construction time cost: {time_cost}")
    return model


def evaluation(run, dataset, args, device):
    logger.info(f"run {run}")

    def T2_calculation(model, dataset):
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=32)
            # loader = DataLoader(dataset, batch_size=args.batch_size)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                data.to(device)
                pred = model(data)[0].detach()
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])

            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            if args.VERBOSE:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = X - Y
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.linalg.pinv(S)
            result = torch.mm(torch.mm(D_mean.T, inv_S), D_mean)
            return result

    time_start = time.process_time()
    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = nn.CosineEmbeddingLoss(margin=args.MARGIN)

    for part_name in part_name_list:
        part_range = part_dict[part_name]
        logger.info("---" * 5)
        logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            if args.VERBOSE:
                logger.info(f"ID: {id}")

            model = get_model(args)
            # if run == 1:
            #     logger.info(model)
            model.to(device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.l2_wd
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            dataset_traintest = dataset[
                id * args.NUM_RELABEL * 2: (id + 1) * args.NUM_RELABEL * 2
            ]
            dataset_reliability = dataset[
                (id + args.SAMPLE_NUM) * args.NUM_RELABEL * 2:
                (id + args.SAMPLE_NUM + 1) * args.NUM_RELABEL * 2
            ]
            transform = ComputePolynomialBases(args)
            dataset_traintest = list(map(transform, dataset_traintest))
            dataset_reliability = list(map(transform, dataset_reliability))

            model.train()
            for _ in range(args.num_epochs):
                traintest_loader = DataLoader(
                    dataset_traintest, batch_size=32, shuffle=True
                    # dataset_traintest, batch_size=args.batch_size, shuffle=True
                )
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    data.to(device)
                    pred = model(data)[0]
                    loss = loss_func(
                        pred[0::2],
                        pred[1::2],
                        torch.tensor([-1] * (len(pred) // 2)).to(device),
                    )
                    loss.backward()
                    optimizer.step()
                    loss_all += len(pred) / 2 * loss.item()

                loss_all /= args.NUM_RELABEL
                if args.VERBOSE:
                    logger.info(f"Loss: {loss_all}")
                if loss_all < args.LOSS_THRESHOLD:
                    if args.VERBOSE:
                        logger.info("Early Stop Here")
                    break
                scheduler.step(loss_all)

            model.eval()
            T_square_traintest = T2_calculation(model, dataset_traintest)
            T_square_reliability = T2_calculation(model, dataset_reliability)

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > args.THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=args.EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < args.THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
                if args.VERBOSE:
                    logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            if args.VERBOSE:
                logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest.cpu().item()}")
                logger.info(f"reliability: {reliability_flag} {T_square_reliability.cpu().item()}")

        end = time.process_time()
        time_cost_part = round(end - start, 2)
        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {part_range[1] - part_range[0]}"
        )

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")
    Acc = round(cnt / args.SAMPLE_NUM, 2)
    logger.info(f"Correct in {cnt} / {args.SAMPLE_NUM}, Acc = {Acc}")
    logger.info(f"Fail in reliability: {fail_in_reliability} / {args.SAMPLE_NUM}")
    logger.info(correct_list)
    logger.info("Real_correct\tCorrect\tFail")
    logger.info(
        f"{cnt-fail_in_reliability:12d}\t{cnt:7d}\t\t{fail_in_reliability}"
    )


def main():
    parser = train_utils.args_setup()
    parser.add_argument("--NUM_RELABEL", type=int, default=32)
    parser.add_argument("--P_NORM", type=int, default=2)
    parser.add_argument("--VERBOSE", action="store_true")
    parser.add_argument("--THRESHOLD", type=float, default=72.34)
    parser.add_argument("--MARGIN", type=float, default=0.0)
    parser.add_argument("--LOSS_THRESHOLD", type=float, default=0.2)
    parser.add_argument("--EPSILON_MATRIX", type=float, default=1e-7)
    parser.add_argument("--EPSILON_CMP", type=float, default=1e-6)
    parser.add_argument("--runs", type=int, default=10, help="Number of repeat run.")
    args = parser.parse_args()
    args = train_utils.update_args(args)
    # torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.SAMPLE_NUM = sum([nums[1] - nums[0] for nums in part_dict.values()])

    # get dataset
    dataset = get_dataset(args)
    logger.add(
        f"{args.save_dir}/{args.exp_name}/result_show.txt",
        format="{message}",
        encoding="utf-8",
    )
    model = get_model(args)
    logger.info(model)
    for run in range(args.runs):
        seed = train_utils.get_seed(args.seed)
        torch_geometric.seed_everything(seed)
        # get model
        evaluation(run + 1, dataset, args, device)


if __name__ == "__main__":
    main()
