import sys
import os
import os.path as osp
from collections import defaultdict
from typing import Mapping

import json
import numpy as np


def parse_label(label, d_collect: bool = False):
    offset = 1 if d_collect else 0
    items = label.split('/')
    dname = items[offset+1]
    mname = items[offset+2]
    key = items[offset+3]
    timestamp = items[offset+4]
    fold = items[offset+5][4]
    return dname, mname, key, timestamp, fold


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


def extract_results(src_dir, dst_path, alias_keys, all_keys, d_collect):

    dst_file = open(dst_path, 'w')
    results_fpath_list = [
        osp.join(folder, file)
        for folder, _, files in os.walk(src_dir)
        for file in files if file == 'results.json'
    ]

    header_flag = True
    for result_fpath in sorted(results_fpath_list):
        print(result_fpath)
        results_dict2 = reformat_results(result_fpath, d_collect)
        if len(results_dict2) > 0:
            if header_flag:
                print(*alias_keys, sep=',', file=dst_file)
                header_flag = False
            values = [results_dict2[key] for key in all_keys]
            print(*values, sep=',', file=dst_file)

    dst_file.close()


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
    'Struct': key_name_factor('MAE', True),
    'Func': key_name_factor('AP', True),
    'molpcba': key_name_factor('AP', True),
    'TUD': key_name_factor('ACC', False),
    'MNIST': key_name_factor('ACC', True),
    'CIFAR10': key_name_factor('ACC', True),
    'PATTERN': key_name_factor('ACC-SBM', True),
    'CLUSTER': key_name_factor('ACC-SBM', True),
    'VOC': key_name_factor('F1', True),
    'COCO': key_name_factor('F1', True),
    'CSL': key_name_factor('ACC', True),
    'EXP': key_name_factor('ACC', True),
    'SR': key_name_factor('ACC', False),
    'ZINC_subset': key_name_factor('MAE', True),
    'ZINC_full': key_name_factor('MAE', True),
    'QM9': key_name_factor('MAE', True),
}

data_collection = {'TUD', 'CIFAR10', 'MNIST', 'QM9', 'molpcba', 'CSL', 'EXP'}

if __name__ == '__main__':
    dname = sys.argv[1]
    dname = dname.lower()
    if 'tu' in dname:
        dname = 'TUD'
        in_dir = f'logs/{dname}'
    elif 'csl' in dname:
        dname = 'CSL'
        in_dir = f'logs/{dname}'
    elif 'exp' in dname:
        dname = 'EXP'
        in_dir = f'logs/{dname}'
    elif 'sr' in dname:
        dname = 'SR'
        in_dir = f'logs/{dname}'
    elif 'str' in dname:
        dname = 'Struct'
        in_dir = f'logs/{dname}'
    elif 'func' in dname:
        dname = 'Func'
        in_dir = f'logs/{dname}'
    elif 'mni' in dname:
        dname = 'MNIST'
        in_dir = f'logs/Bench/{dname}'
    elif 'cif' in dname:
        dname = 'CIFAR10'
        in_dir = f'logs/Bench/{dname}'
    elif 'pat' in dname:
        dname = 'PATTERN'
        in_dir = f'logs/Bench/{dname}'
    elif 'clu' in dname:
        dname = 'CLUSTER'
        in_dir = f'logs/Bench/{dname}'
    elif 'voc' in dname:
        dname = 'VOC'
        in_dir = f'logs/{dname}'
    elif 'co' in dname:
        dname = 'COCO'
        in_dir = f'logs/{dname}'
    elif 'zinc_subset' in dname:
        dname = 'ZINC_subset'
        in_dir = f'logs/{dname}'
    elif 'zinc_full' in dname:
        dname = 'ZINC_full'
        in_dir = f'logs/{dname}'
    elif 'qm9' == dname:
        dname = 'QM9'
        in_dir = f'logs/{dname}'
    elif 'pcba' in dname:
        dname = 'molpcba'
        in_dir = f'logs/OGBG/{dname}'

    dst_path = f'{in_dir}/results.csv'
    alias_keys, all_keys = dict_keys[dname]
    d_collect = dname in data_collection
    extract_results(in_dir, dst_path, alias_keys, all_keys, d_collect)
