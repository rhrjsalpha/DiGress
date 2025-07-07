# 🔧 train_model_entrypoint.py

import argparse
import json
from types import SimpleNamespace
from Graphormer.GP5.models.graphormer_train_eVOsc_GradNorm_2loss_GPUs import train_model_ex_porb
from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset
from torch.utils.data import Subset
import torch.multiprocessing as mp


def load_args(args_path):
    with open(args_path, "r") as f:
        args_dict = json.load(f)
    return SimpleNamespace(**args_dict)


def run(rank, args):
    train_model_ex_porb(rank, args)

def get_fold_subset(full_dataset, fold_idx, n_splits=5, seed=42):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(kf.split(full_dataset))
    train_idx, val_idx = splits[fold_idx]
    return Subset(full_dataset, train_idx), Subset(full_dataset, val_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", type=str, required=True)
    args = parser.parse_args()

    # JSON 로딩
    with open(args.args, "r") as f:
        args_dict = json.load(f)

    args_ns = SimpleNamespace(**args_dict)

    # ✅ 전체 dataset 로딩 및 fold subset 생성
    dataset = SMILESDataset(csv_file="../../data/train_50.csv", attn_bias_w=1.0, target_type=args_ns.target_type)
    train_subset, val_subset = get_fold_subset(dataset, fold_idx=int(args.args[-6]))  # fold3.json → 3

    args_ns.TRAIN_DATASET = train_subset
    args_ns.VAL_DATASET = val_subset

    mp.spawn(train_model_ex_porb, args=(args_ns,), nprocs=args_ns.world_size)
