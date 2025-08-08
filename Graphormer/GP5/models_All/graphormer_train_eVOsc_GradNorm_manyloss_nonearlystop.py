from __future__ import annotations

# ===== 기본 패키지 =====
import os
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from evaluate_metrics import evaluate_model_metrics
from tqdm import tqdm
from collections import OrderedDict

# ===== Graphormer / 사용자 정의 모듈 =====
from Graphormer.GP5.data_prepare.DataLoader_QMData_All import (
    UnifiedSMILESDataset,
    collate_fn,
    get_global_feature_info,  # helper util – 사용자 정의
)
from Graphormer.GP5.models_All.graphormer_3 import GraphormerModel
from Graphormer.GP5.Custom_Loss.soft_dtw_cuda import SoftDTW
from Graphormer.GP5.Custom_Loss.GradNorm import GradNorm
from chemprop.train.loss_functions import sid_loss
from Graphormer.GP5.Custom_Loss.fast_dtw import fastdtw
from rdkit import Chem

# -----------------------------------------------------------------------------
# 유틸리티 함수
# -----------------------------------------------------------------------------

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE 계산 – scikit‑learn 버전 호환"""
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:  # older sklearn
        return np.sqrt(mean_squared_error(y_true, y_pred))


# -----------------------------------------------------------------------------
# 학습 함수(train_model_ex_porb)
# -----------------------------------------------------------------------------
def train_model_eVOsc(
    *,
    config: Dict,
    target_type: str = "ex_prob",
    loss_function_full_spectrum: list[str] = ["MSE"], # MSE, MAE, SoftDTW, SID
    loss_function_ex: list[str] = ["SoftDTW"], # MSE, MAE, SoftDTW, SID
    loss_function_prob: list[str] = ["SoftDTW"], # MSE, MAE, SoftDTW, SID
    num_epochs: int = 10,
    batch_size: int = 64,
    n_pairs: int = 50,
    learning_rate: float = 1e-3,
    dataset_path: str,
    test_dataset_path: str,
    DATASET=None,
    TEST_VAL_DATASET=None,
    alpha: float = 0.12,
    is_cv: bool = False,
    nominal_feature_vocab=None,
    continuous_feature_names=None,
    global_cat_dim=0,
    global_cont_dim=0,
    global_feature_names: List[str] | None = None,
    ex_normalize: str = "ex",
    prob_normalize: str = "prob",
    debug = True
) -> Tuple[Dict, str]:
    """Graphormer 학습 + 인‑샘플 평가.

    Returns
    -------
    results : Dict
        metric / loss history
    best_model_path : str
        저장된 최적 모델 경로
    """

    # ---------------- 데이터 로딩 ----------------
    print("train_model_ex_porb config[mode]",config["mode"])
    print("train_model_ex_porb target_type", config.get("target_type"))
    if DATASET is None:
        print("train_model_ex_porb nominal_feature_vocab",nominal_feature_vocab)
        dataset = UnifiedSMILESDataset(
            csv_file=dataset_path,
            nominal_feature_vocab=nominal_feature_vocab,
            continuous_feature_names=continuous_feature_names,
            global_cat_dim=global_cat_dim,
            global_cont_dim=global_cont_dim,
            ATOM_FEATURES_VOCAB=config["ATOM_FEATURES_VOCAB"],
            float_feature_keys=config["float_feature_keys"],
            BOND_FEATURES_VOCAB=config["BOND_FEATURES_VOCAB"],
            mode=config["mode"],  # "cls", "cls+global_data", "cls+global_model"
            max_nodes=config.get("max_nodes", 128),
            multi_hop_max_dist=config.get("multi_hop_max_dist", 5),
            target_type=config.get("target_type", "default"),
            attn_bias_w=config.get("attn_bias_w", 1.0),
            ex_normalize=config.get("ex_normalize", None),
            prob_normalize=config.get("prob_normalize", None),
            nm_dist_mode=config.get("nm_dist_mode", "hist"),
            nm_gauss_sigma=config.get("nm_gauss_sigma", 10.0),
            intensity_normalize = config.get("intensity_normalize","min_max"),
            intensity_range = config.get("intensity_range",(200,800)),
        )
        print("train_model dataset.mode ,dataset.graphs[0]",dataset.mode,)

        example_graph = dataset.graphs[0]
        has_global_cat = "global_features_cat" in example_graph
        has_global_cont = "global_features_cont" in example_graph

        print("train_model Graph 내부에 global_features_cat 있음:", has_global_cat)
        print("train_model Graph 내부에 global_features_cont 있음:", has_global_cont)
    else:
        dataset = DATASET

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, dataset, n_pairs=n_pairs),
    )

    # ---------------- 모델/옵티마이저 ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("train_model config",config)
    model = GraphormerModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler = GradScaler()

    soft_dtw_fn = SoftDTW(use_cuda=True, gamma=0.2, bandwidth=None, normalize=True)

    def _make_loss(name: str):
        if name == "MSE":
            return nn.MSELoss()
        if name == "MAE":
            return nn.L1Loss()
        if name == "Huber":
            return nn.SmoothL1Loss()
        if name == "SoftDTW":
            return soft_dtw_fn
        if name == "SID":
            # wrap chemprop sid_loss
            def sid_wrapper(pred, tgt, mask, thr):
                return sid_loss(pred, tgt, mask, thr)
            return sid_wrapper
        raise ValueError(f"Unsupported loss name: {name}")

    best_epoch = 0
    best_model_path = "best_model.pth"
    best_combined_loss = math.inf
    if target_type == 'ex_prob':
        crit_ex_dict = {}
        for name_ex in loss_function_ex:
            crit_ex_dict[name_ex] = _make_loss(name_ex)

        crit_prob_dict = {}
        for name_prob in loss_function_prob:
            crit_prob_dict[name_prob] = _make_loss(name_prob)

        num_ex_losses = len(crit_ex_dict)
        num_prob_losses = len(crit_prob_dict)
        total_losses = num_ex_losses + num_prob_losses

        gradnorm = GradNorm(num_losses=total_losses, alpha=alpha)

        # weight_true = torch.tensor([0.5, 0.5], device=device)
        weight_true = torch.full((total_losses,), 1.0 / total_losses, device=device)

        first_losses_ex = None
        first_losses_prob = None

        history: Dict[str, List[float]] = {
            "ex_loss": [],
            "prob_loss": [],
            "total_loss": [],
            "normalized_ex_loss": [],
            "normalized_prob_loss": [],
            "weight_ex": [],
            "weight_prob": [],
        }

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            model.train()
            epoch_loss, ex_losses, prob_losses, wts = 0.0, {}, {}, []
            norm_ex_dict, norm_prob_dict = {}, {}

            for batch in dataloader:
                batch_data = {k: v.to(device) for k, v in batch.items() if k != "targets"}
                targets = batch["targets"].to(device)

                optimizer.zero_grad()
                outputs = model(batch_data, targets=targets, target_type=target_type)
                print("train_model outputs.shape",outputs.shape, targets.shape)
                out_ex, tgt_ex = outputs[:, :, 0:1] + 1e-8, targets[:, :, 0:1] + 1e-8
                out_prob, tgt_prob = outputs[:, :, 1:2] + 1e-8, targets[:, :, 1:2] + 1e-8

                # loss 계산 (배치 내 개별 스펙트럼 평균)
                loss_ex_dict = {}
                for loss_ex_name, crit_ex  in crit_ex_dict.items():
                    if loss_ex_name == "SID":
                        mask_ex = torch.ones_like(out_ex, dtype=torch.bool)
                        loss_ex = torch.stack([
                            crit_ex(out_ex[i].unsqueeze(0), tgt_ex[i].unsqueeze(0), mask_ex[i], 1e-8)
                            for i in range(out_ex.size(0))
                        ]).mean()
                    else:
                        loss_ex = torch.stack([
                            crit_ex(out_ex[i].unsqueeze(0), tgt_ex[i].unsqueeze(0))
                            for i in range(out_ex.size(0))
                        ]).mean()
                    loss_ex_dict[loss_ex_name] = loss_ex

                loss_prob_dict = {}
                for loss_prob_name, crit_prob in crit_prob_dict.items():
                    if loss_prob_name == "SID":
                        mask_prob = torch.ones_like(out_prob, dtype=torch.bool)
                        loss_prob = torch.stack([
                            crit_prob(out_prob[i].unsqueeze(0), tgt_prob[i].unsqueeze(0), mask_prob[i], 1e-8)
                            for i in range(out_prob.size(0))
                        ]).mean()
                    else:
                        loss_prob = torch.stack([
                            crit_prob(out_prob[i].unsqueeze(0), tgt_prob[i].unsqueeze(0))
                            for i in range(out_prob.size(0))
                        ]).mean()
                    loss_prob_dict[loss_prob_name] = loss_prob

                # first losses (정규화 기준)
                # Initialize only once outside the loop
                if first_losses_ex is None:
                    first_losses_ex = {}

                if first_losses_prob is None:
                    first_losses_prob = {}

                # Populate first losses
                if not first_losses_ex:
                    for loss_name_ex, loss_val_ex in loss_ex_dict.items():
                        first_losses_ex[loss_name_ex] = loss_val_ex.item()
                    print("train_model first_losses_ex", first_losses_ex[loss_name_ex])

                if not first_losses_prob:
                    for loss_name_prob, loss_val_prob in loss_prob_dict.items():
                        first_losses_prob[loss_name_prob] = loss_val_prob.item()

                norm_ex = {}
                for loss_ex_name, loss_val_ex in loss_ex_dict.items():
                    if not loss_ex_name in ex_losses.keys():
                        ex_losses[loss_ex_name] = []
                    ex_losses[loss_ex_name].append(loss_val_ex)

                    norm_ex[loss_ex_name] = loss_val_ex / first_losses_ex[loss_ex_name]

                    if not loss_ex_name in norm_ex_dict.keys():
                        norm_ex_dict[loss_ex_name] = []
                    norm_ex_dict[loss_ex_name].append(norm_ex[loss_ex_name])

                norm_prob = {}
                for loss_prob_name, loss_val_prob in loss_prob_dict.items():
                    if not loss_prob_name in prob_losses.keys():
                        prob_losses[loss_prob_name] = []
                    prob_losses[loss_prob_name].append(loss_val_prob)

                    norm_prob[loss_prob_name] = loss_val_prob / first_losses_prob[loss_prob_name]

                    if not loss_prob_name in norm_prob_dict.keys():
                        norm_prob_dict[loss_prob_name] = []
                    norm_prob_dict[loss_prob_name].append(norm_prob[loss_prob_name])

                to_caculate_weights_dict = {"Name":[],"loss_val":[]}
                for loss_name_ex, loss_val_ex in norm_ex.items():
                    to_caculate_weights_dict["Name"].append(loss_name_ex)
                    to_caculate_weights_dict["loss_val"].append(loss_val_ex)
                for loss_name_prob, loss_val_prob in norm_prob.items():
                    to_caculate_weights_dict["Name"].append(loss_name_prob)
                    to_caculate_weights_dict["loss_val"].append(loss_val_prob)

                weights = gradnorm.compute_weights(to_caculate_weights_dict["loss_val"], model)
                print("weights",weights)
                wts.append(weights.detach().cpu().numpy())

                total_loss_list = []
                for loss_val, weight in zip(to_caculate_weights_dict["loss_val"], weight_true):
                    total_loss_list.append(loss_val * weight)
                total_loss = sum(total_loss_list)

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                #ex_losses.append(loss_ex.item())
                #prob_losses.append(loss_prob.item())
                #norm_ex_list.append(norm_ex.item())
                #norm_prob_list.append(norm_prob.item())

            # ---- epoch 통계 ----
            epoch_loss /= len(dataloader)
            weight_true = torch.tensor(np.mean(wts, axis=0), device=device)

            history["total_loss"].append(epoch_loss)

            for i, name in enumerate(to_caculate_weights_dict["Name"]):
                # 예: "SoftDTW", "MSE" 등
                if name in ex_losses:
                    loss_type = "ex"
                    avg_loss = float(torch.stack(ex_losses[name]).mean().item())
                    avg_norm = float(torch.stack(norm_ex_dict[name]).mean().item())
                else:
                    loss_type = "prob"
                    avg_loss = float(torch.stack(prob_losses[name]).mean().item())
                    avg_norm = float(torch.stack(norm_prob_dict[name]).mean().item())

                # 손실 기록
                history_key_loss = f"{loss_type}_loss_{name}"
                history_key_norm = f"normalized_{loss_type}_loss_{name}"
                history_key_weight = f"weight_{name}"

                if history_key_loss not in history:
                    history[history_key_loss] = []
                history[history_key_loss].append(avg_loss)


                if history_key_norm not in history:
                    history[history_key_norm] = []
                history[history_key_norm].append(avg_norm)

                if history_key_weight not in history:
                    history[history_key_weight] = []
                history[history_key_weight].append(float(weight_true[i]))
#
            elapsed = time.time() - t0

            # ex 손실들 중 마지막 값을 문자열로 만듦
            ex_loss_str = " | ".join([
                f"ex_{name}: {history[f'ex_loss_{name}'][-1]:.4f}"
                for name in loss_function_ex
                if f'ex_loss_{name}' in history and len(history[f'ex_loss_{name}']) > 0
            ])

            prob_loss_str = " | ".join([
                f"prob_{name}: {history[f'prob_loss_{name}'][-1]:.4f}"
                for name in loss_function_prob
                if f'prob_loss_{name}' in history and len(history[f'prob_loss_{name}']) > 0
            ])

            print(
                f"Epoch {epoch:03d}/{num_epochs} | total {epoch_loss:.4f} | {ex_loss_str} | {prob_loss_str} | w {weight_true.tolist()} | {elapsed:.1f}s",
                flush=True
            )

            if epoch_loss < best_combined_loss:
                best_combined_loss = epoch_loss
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
    else:
        print("please check training method")
        raise ValueError

    # ---------------- 인‑샘플 평가 ----------------
    time_start = time.time()
    config["out_of_training"] = True
    model = GraphormerModel(config).to(device)
    model.load_state_dict(torch.load(best_model_path))
    results = {}
    results.update(evaluate_model_metrics(model, dataloader, target_type, device, soft_dtw_fn, sid_loss, is_cv=is_cv, best_epoch=best_epoch, is_val=False))
    time_end = time.time()
    print("training_set_evaluate_time:", time_end-time_start)

    time_start = time.time()
    if TEST_VAL_DATASET is None:
        dataset_test = UnifiedSMILESDataset(
            csv_file=test_dataset_path,
            nominal_feature_vocab=nominal_feature_vocab,
            continuous_feature_names=continuous_feature_names,
            global_cat_dim=global_cat_dim,
            global_cont_dim=global_cont_dim,
            ATOM_FEATURES_VOCAB=config["ATOM_FEATURES_VOCAB"],
            float_feature_keys=config["float_feature_keys"],
            BOND_FEATURES_VOCAB=config["BOND_FEATURES_VOCAB"],
            mode=config["mode"],  # "cls", "cls+global_data", "cls+global_model"
            max_nodes=config.get("max_nodes", 128),
            multi_hop_max_dist=config.get("multi_hop_max_dist", 5),
            target_type=config.get("target_type", "default"),
            attn_bias_w=config.get("attn_bias_w", 1.0),
            ex_normalize=config.get("ex_normalize", None),
            prob_normalize=config.get("prob_normalize", None),
            nm_dist_mode=config.get("nm_dist_mode", "hist"),
            nm_gauss_sigma=config.get("nm_gauss_sigma", 10.0),
            intensity_normalize=config.get("intensity_normalize", "min_max"),
            intensity_range=config.get("intensity_range", (200, 800)),
        )
        time_end = time.time()
        print("test set loading time:", time_end - time_start)
        print("dataset.mode ,dataset.graphs[0]", dataset.mode, )

        example_graph = dataset.graphs[0]
        has_global_cat = "global_features_cat" in example_graph
        has_global_cont = "global_features_cont" in example_graph

        print("Graph 내부에 global_features_cat 있음:", has_global_cat)
        print("Graph 내부에 global_features_cont 있음:", has_global_cont)
    else:
        dataset_test = TEST_VAL_DATASET

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, dataset, n_pairs=n_pairs),
    )

    results.update(evaluate_model_metrics(model, dataloader_test, target_type, device, soft_dtw_fn, sid_loss, is_cv=is_cv, best_epoch=best_epoch, is_val=True))


    for k, v in history.items():
        results[f"{k}_history"] = v

    return results, best_model_path


# -----------------------------------------------------------------------------
# 메인 진입점
# -----------------------------------------------------------------------------


PREDEFINED_VOCAB = {
    'Solvent': [
        '1,4-Dioxane', 'Acetonitrile', 'Benzene', 'Chloroform', 'Cyclohexane',
        'Dichloromethane', 'Dimethylformamide', 'Dimethylsulfoxide', 'Ethanol',
        'Ethylacetate', 'Heptane', 'Hexane', 'Methanol', 'N-Methyl-2-pyrrolidone',
        'Tetrahydrofuran', 'Toluene', 'Water', "DMSO", "Acetone"
    ],
}

ATOM_FEATURES_VOCAB = {
        'atomic_num': list(range(1, 119)),  # TODO I need to decrease the range
        'formal_charge': list(range(-5, 6)),  # increase range when diffusion / or add threshold
        'hybridization': [
            Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.OTHER,
            # Chem.rdchem.HybridizationType.UNSPECIFIED # add this when diffusion
        ],
        'is_aromatic': [0, 1],
        'total_num_hs': list(range(0, 9)),  # increase it when diffusion
        'explicit_valence': list(range(0, 8)),  # increase range when diffusion / or add threshold of valence encoding
        'total_bonds': list(range(0, 8)),  # increase range when diffusion / or add threshold
        'partial_charge': float,  # check error and change code when diffusion
        'atomic_mass': float,  # OK when diffusion
    }

float_feature_keys = ['partial_charge', 'atomic_mass']

BOND_FEATURES_VOCAB = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
        # Chem.rdchem.BondType.UNSPECIFIED # add this when diffusion
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS,  # OK when diffusion
    ],
    'is_conjugated': [0, 1],  # OK when diffusion
    'is_in_ring': [0, 1],  # OK when diffusion
}

from types import SimpleNamespace

def main() -> None:
    print("CUDA available:", torch.cuda.is_available())

    # ---------- 글로벌 피처 정보 ----------
    #global nominal_dims, continuous_feature_names, global_cat_dim, global_cont_dim
    global device

    GLOBAL_FEATURE_NAMES = ['Solvent', 'Temperature', 'Pressure']

    dataset_train_path = "../../graphormer_data/train_50_with_features.csv"
    dataset_test_path = "../../graphormer_data/test_10_with_features.csv"

    try:
        nominal_dims, continuous_feature_names, global_cat_dim, global_cont_dim = get_global_feature_info(GLOBAL_FEATURE_NAMES, PREDEFINED_VOCAB)
        print("nominal_dims",nominal_dims)
        print("continuous_feature_names",continuous_feature_names)
        print(global_cat_dim)
        print(global_cont_dim)
    except Exception as e:
        print("[WARN] global feature info 오류, fallback 사용:", e)
        nominal_dims, continuous_feature_names = {}, []
        global_cat_dim, global_cont_dim = 1, 2

    # device (GPU/CPU) 설정 전역 변수화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # ---------- Graphormer config ----------
    config: Dict = {
        "num_atoms": 100,
        "num_in_degree": 10,
        "num_out_degree": 10,
        "num_edges": 100,
        "num_spatial": 100,
        "num_edge_dis": 10,
        "edge_type": "multi_hop",
        "multi_hop_max_dist": 2,
        "num_encoder_layers": 1,
        "embedding_dim": 128,
        "ffn_embedding_dim": 256,
        "num_attention_heads": 8,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.1,
        "activation_fn": "gelu",
        "pre_layernorm": False,
        "q_noise": 0.0,
        "qn_block_size": 8,
        "output_size": 100, # 100(ex_prob), 451(nm_ditribution), exp_spectrum 100
        "num_categorical_features": 7,  # (= 7 atom categorical)
        "num_continuous_features": 2,  # (= 2 atom continuous)
        "mode": "cls_only", # "cls_only" , "cls_global_data", "cls_global_model"
        "target_type": "ex_prob", # "default", "ex_prob", "nm_distribution", "exp_spectrum"
        "intensity_normalize": "min_max",
        "intensity_range" : (1,100), # (150,601) nm_distribution , (1,100) ex_prob exp_spectrum
        "nm_dist_mode" : "gauss"
    }

    config.update({
        "ATOM_FEATURES_VOCAB": ATOM_FEATURES_VOCAB,
        "float_feature_keys": float_feature_keys,
        "BOND_FEATURES_VOCAB": BOND_FEATURES_VOCAB,
        "ex_normalize": "ex_min_max", #
        "prob_normalize": "prob_min_max",
    })
    if config.get("mode") == "cls_global_data" or config.get("mode") == "cls_global_model":
        config.update({
                     "global_cat_dim": global_cat_dim,
                     "global_cont_dim": global_cont_dim,
                      })
    elif config.get("mode") == "cls_only":
        config.update({
            "global_cat_dim": 0,
            "global_cont_dim": 0,
        })
    else:
        print("please use cls_global_data, cls_global_model, cls_only")

    # ---------- 학습 루프 ----------
    loss_candidates = ["MAE"]  # 필요시 확장: ["MAE", "MSE", "SoftDTW", "SID"]
    results_all: List[Dict] = []

    for loss_name in loss_candidates:
        print(f"\n=== START training with loss: {loss_name} ===")
        res, best_path = train_model_eVOsc(
            config=config,
            target_type=config.get("target_type"),
            # loss_function_full_spectrum = ['SoftDTW','SID'],  # MSE, MAE, SoftDTW, SID ['MSE','MAE','SoftDTW','SID']
            # loss_function_nm_point = loss_name,
            loss_function_ex= ['MSE','MAE','SoftDTW','SID'],
            loss_function_prob= ['MSE','MAE','SoftDTW','SID'],
            num_epochs=200,
            batch_size=50,
            n_pairs=50,
            dataset_path=dataset_train_path,
            test_dataset_path=dataset_test_path,
            alpha=0.12,
            global_feature_names=GLOBAL_FEATURE_NAMES,
            nominal_feature_vocab=nominal_dims,
            continuous_feature_names = continuous_feature_names,
            ex_normalize="ex_min_max",
            prob_normalize="prob_min_max",
        )
        res["loss_function"] = loss_name
        res["best_model_path"] = best_path
        results_all.append(res)

        # 중간 저장
        pd.DataFrame(results_all).to_csv("training_results_intermediate.csv", index=False)
        print(f"[SAVE] intermediate results → training_results_intermediate.csv")

    # ---------- 최종 결과 저장 ----------
    out_csv = "training_results.csv"
    pd.DataFrame(results_all).to_csv(out_csv, index=False)
    print("\nAll trainings finished. Saved ⇒", out_csv)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
