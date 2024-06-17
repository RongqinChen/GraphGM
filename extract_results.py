import sys
import os
import os.path as osp
from collections import defaultdict
from typing import Mapping

import json
import numpy as np


def reformat_results(result_fpath, d_collect=False):
    with open(result_fpath, 'r') as rfile:
        result_dict: Mapping = json.load(rfile)
        results_dict = defaultdict(list)
        for label, result in result_dict.items():
            dname, mname, key, timestamp, fold = parse_label(label, d_collect)
            results_dict['dname'].append(dname)
            results_dict['mname'].append(mname)
            results_dict['key'].append(key)
            results_dict['timestamp'].append(f"'{timestamp}")
            results_dict['fold'].append(fold)
            for rkey, rval in result.items():
                if isinstance(rval, dict):
                    continue
                elif isinstance(rval, list):
                    rval = '-'.join(map(str, rval))
                results_dict[rkey].append(rval)

    results_dict2 = dict()
    for rkey, rval_list in results_dict.items():
        if rkey == 'fold':
            results_dict2['num_fold'] = len(rval_list)
        elif rkey == 'best_epoch' or 'loss' in rkey:
            rval_mean, rval_std = np.mean(rval_list), np.std(rval_list)
            results_dict2[f"{rkey}_mean"] = f"{rval_mean:.5f}"
            results_dict2[f"{rkey}_std"] = f"{rval_std:.5f}"
        elif 'ACC' in rkey or 'AP' in rkey or 'MAE' in rkey or 'ROCAUC' in rkey or 'F1' in rkey:
            rval_list = [val for val in rval_list]
            # rval_list = [val * 100. for val in rval_list]
            rval_mean, rval_std = np.mean(rval_list), np.std(rval_list)
            results_dict2[f"{rkey}_mean"] = f"{rval_mean:.5f}"
            results_dict2[f"{rkey}_std"] = f"{rval_std:.5f}"
        elif 'loss' in rkey:
            rval_list = [val for val in rval_list]
            rval_mean, rval_std = np.mean(rval_list), np.std(rval_list)
            results_dict2[f"{rkey}_mean"] = f"{rval_mean:.5f}"
            results_dict2[f"{rkey}_std"] = f"{rval_std:.5f}"
        else:
            rval_set_str = "-".join(
                [f"{val}" for val in sorted(set(rval_list))])
            results_dict2[rkey] = rval_set_str

    return results_dict2


def key_name_factor(metric: str, has_test: bool):
    if has_test:
        alias_keys = [
            'dname', 'mname', f'Test{metric}', 'key', '#Fold', 'TestStd',
            '#Para', 'Time', f'Valid{metric}', 'ValidStd',
            f'Train{metric}', 'TrainStd', 'TestLoss', 'TestL.Std',
            'ValidLoss', 'ValidL.Std', 'TrainLoss', 'TrainL.Std',
            'BestEpoch', 'BestE.Std'
        ]
        all_keys = [
            'dname', 'mname', f'test_{metric}_mean', 'key', 'num_fold',
            f'test_{metric}_std', 'num_parameters', 'timestamp',
            f'valid_{metric}_mean', f'valid_{metric}_std',
            f'train_{metric}_mean', f'train_{metric}_std',
            'test_loss_mean', 'test_loss_std',
            'valid_loss*_mean', 'valid_loss*_std', 'train_loss_mean',
            'train_loss_std', 'best_epoch_mean', 'best_epoch_std'
        ]
    else:
        alias_keys = [
            'dname', 'mname', f'Valid{metric}', 'key', '#Fold', 'ValidStd',
            '#Para', 'Time', f'Train{metric}', 'TrainStd',
            'ValidLoss', 'ValidL.Std', 'TrainLoss', 'TrainL.Std',
            'BestEpoch', 'BestE.Std'
        ]
        all_keys = [
            'dname', 'mname', f'valid_{metric}_mean', 'key', 'num_fold',
            f'valid_{metric}_std', 'num_parameters', 'timestamp',
            f'train_{metric}_mean', f'train_{metric}_std',
            'valid_loss*_mean', 'valid_loss*_std', 'train_loss_mean',
            'train_loss_std', 'best_epoch_mean', 'best_epoch_std'
        ]

    return alias_keys, all_keys


dict_keys = {
    'arxiv': key_name_factor('ACC', True),
    'struct': key_name_factor('MAE', True),
    'func': key_name_factor('AP', True),
    'molpcba': key_name_factor('AP', True),
    'tud': key_name_factor('ACC', False),
    'mnist': key_name_factor('ACC', True),
    'cifar10': key_name_factor('ACC', True),
    'pattern': key_name_factor('ACC-SBM', True),
    'cluster': key_name_factor('ACC-SBM', True),
    'voc': key_name_factor('F1', True),
    'coco': key_name_factor('F1', True),
    'csl': key_name_factor('ACC', True),
    'exp': key_name_factor('ACC', True),
    'sr': key_name_factor('ACC', False),
    'zinc': key_name_factor('MAE', True),
    'zinc_full': key_name_factor('MAE', True),
    'qm9': key_name_factor('MAE', True),
}

data_collection = {'tud', 'cifar10', 'mnist', 'qm9', 'molpcba', 'csl', 'exp'}

if __name__ == '__main__':
    dname = sys.argv[1]
    dname = dname.lower()

    for root, dirnames, _ in os.walk('results'):
        for dirname in dirnames:
            dirname: str = dirname
            if not dirname.startswith(dname + '-'):
                continue

            src_path = f'{root}/{dirname}/result_summary.json'
            if not osp.exists(src_path):
                continue
            dst_path = f'{root}/{dirname}/stat.csv'
            with open(src_path, 'r') as rfile:
                results_dict = json.load(rfile)

            summary_dict = defaultdict(list)
            for result in results_dict.values():
                for k, v in result.items():
                    if isinstance(v, str) and v.startswith('mae: '):
                        v = float(v[5:])
                    summary_dict[k].append(v)

            out_dict = dict()
            for k, v in summary_dict.items():
                if isinstance(v[0], str):
                    out_dict[k] = v[0]
                else:
                    out_dict[k] = np.mean(v)
                    out_dict[k + '_std'] = np.std(v)

            with open(dst_path, 'w') as wfile:
                json.dump(out_dict, wfile, indent='  ')
