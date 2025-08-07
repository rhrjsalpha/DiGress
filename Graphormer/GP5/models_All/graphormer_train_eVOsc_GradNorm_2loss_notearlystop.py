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



def train_model_ex_porb(
    *,
    config: Dict,
    target_type: str = "ex_prob",
    loss_function_full_spectrum: str = "MSE", # MSE, MAE, SoftDTW, SID
    loss_function_nm_point: str = "MSE", # MSE MAE
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
    print("config[mode]",config["mode"])
    print("target_type", config.get("target_type"))
    if DATASET is None:
        print("nominal_feature_vocab",nominal_feature_vocab)
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
        print("dataset.mode ,dataset.graphs[0]",dataset.mode,)

        example_graph = dataset.graphs[0]
        has_global_cat = "global_features_cat" in example_graph
        has_global_cont = "global_features_cont" in example_graph

        print("Graph 내부에 global_features_cat 있음:", has_global_cat)
        print("Graph 내부에 global_features_cont 있음:", has_global_cont)
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
    print("config",config)
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
    total_loss = 0
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
                print("outputs.shape",outputs.shape, targets.shape)
                out_ex, tgt_ex = outputs[:, :, 0:1] + 1e-8, targets[:, :, 0:1] + 1e-8
                out_prob, tgt_prob = outputs[:, :, 1:2] + 1e-8, targets[:, :, 1:2] + 1e-8

                # loss 계산 (배치 내 개별 스펙트럼 평균)
                loss_ex_dict = {}
                for loss_ex_name, crit_ex  in crit_ex_dict.items():
                    if loss_function_ex == "SID":
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
                    if loss_function_prob == "SID":
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
                    print("first_losses_ex", first_losses_ex)

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

                # total_loss = weight_true[0] * norm_ex + weight_true[1] * norm_prob
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

            #history["ex_loss"].append(float(np.mean(ex_losses)))
            #history["prob_loss"].append(float(np.mean(prob_losses)))
            #history["total_loss"].append(epoch_loss)
            #history["normalized_ex_loss"].append(float(np.mean(norm_ex_list)))
            #history["normalized_prob_loss"].append(float(np.mean(norm_prob_list)))
            #history["weight_ex"].append(float(weight_true[0]))
            #history["weight_prob"].append(float(weight_true[1]))
#
            elapsed = time.time() - t0
            #print(
            #    f"Epoch {epoch:03d}/{num_epochs} | total {epoch_loss:.4f} | ex {history['ex_loss'][-1]:.4f} | prob {history['prob_loss'][-1]:.4f} | w {weight_true.tolist()} | {elapsed:.1f}s",
            #    flush=True,
            #)

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
        #num_ex_losses = len(crit_ex_dict)
        #num_prob_losses = len(crit_prob_dict)
        #total_losses = num_ex_losses + num_prob_losses
        # gradnorm = GradNorm(num_losses=total_losses, alpha=alpha)

        if target_type == "exp_spectrum":
            criterion = _make_loss(loss_function_full_spectrum)
        elif target_type == "nm_distribution":
            criterion = _make_loss(loss_function_nm_point)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        history = {
            "loss": [],
        }

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            model.train()
            epoch_losses = []

            for batch in dataloader:
                batch_data = {k: v.to(device) for k, v in batch.items() if k != "targets"}
                targets = batch["targets"].to(device)

                optimizer.zero_grad()
                outputs = model(batch_data, targets=targets, target_type=target_type)

                batch_losses = []
                for i in range(outputs.size(0)):
                    y_pred = outputs[i].unsqueeze(0).unsqueeze(-1)
                    y_true = targets[i].unsqueeze(0).unsqueeze(-1)

                    if target_type == "exp_spectrum":
                        mask = batch["masks"][i].to(device)
                        mask_1d = mask == 1
                        y_pred = y_pred[0][mask_1d].unsqueeze(0).unsqueeze(-1)
                        y_true = y_true[0][mask_1d].unsqueeze(0).unsqueeze(-1)

                    if loss_function_full_spectrum == "SID":
                        mask_tensor = torch.ones_like(y_pred, dtype=torch.bool)
                        loss_i = criterion(y_pred, y_true, mask_tensor, 1e-8)
                    else:
                        loss_i = criterion(y_pred, y_true)

                    batch_losses.append(loss_i)

                total_loss = torch.stack(batch_losses).mean()
                total_loss.backward()
                optimizer.step()

                epoch_losses.append(total_loss.item())

            avg_loss = np.mean(epoch_losses)
            history["loss"].append(avg_loss)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{num_epochs} | loss {avg_loss:.4f} | {elapsed:.1f}s",
                flush=True,
            )

            if avg_loss < best_combined_loss:
                best_combined_loss = avg_loss
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)

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

### 여러 Target type에 맞추어 나중에 바꾸기 ###
def evaluate_model_metrics(model, dataloader, target_type, device, soft_dtw_fn, sid_loss, is_cv=False, best_epoch=None, is_val=False):
    model.eval()

    # 스펙트럼별 결과 저장용 리스트 초기화
    # Initialize lists for individual and combined metrics
    if target_type == "ex_prob":
        sid_spectrum_ex, sid_spectrum_prob, sid_spectrum_combined = [], [], []
        sis_spectrum_ex, sis_spectrum_prob, sis_spectrum_combined = [], [], []
        r2_spectrum_ex, r2_spectrum_prob, r2_spectrum_combined = [], [], []
        mae_spectrum_ex, mae_spectrum_prob, mae_spectrum_combined = [], [], []
        rmse_spectrum_ex, rmse_spectrum_prob, rmse_spectrum_combined = [], [], []
        softdtw_spectrum_ex, softdtw_spectrum_prob, softdtw_spectrum_combined = [], [], []
        fastdtw_spectrum_ex, fastdtw_spectrum_prob, fastdtw_spectrum_combined = [], [], []
    else:
        sid_spectrum = []
        sis_spectrum = []
        r2_spectrum = []
        mae_spectrum = []
        rmse_spectrum = []
        softdtw_spectrum = []
        fastdtw_spectrum = []


    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            all_keys = batch.keys()
            batched_data = {k: batch[k] for k in all_keys if k != "targets"}
            # batched_data = {k: batch[k] for k in
            #                ["x_cat", "x_cont", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
            #                 "attn_edge_type"]}

            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=target_type)
            # outputs = torch.clamp(outputs, min=1e-8)

            sid_spectrum_ex, sid_spectrum_prob, sid_spectrum_combined = [], [], []

            nan_case_ex, nan_case_prob, nan_case_combined = 0, 0, 0  # NaN 카운트

            # Batch에서 각 스펙트럼별로 계산
            for i in range(targets.size(0)):  # batch_size 만큼 루프
                time_start = time.time()
                y_true = targets[i].cpu().numpy()
                # print("y_true.shape)",y_true.shape)
                y_pred = outputs[i].cpu().detach().numpy()
                # print("y_pred.shape",y_pred.shape)
                time_end = time.time()
                #print("to_cpu_time:", time_end-time_start)

                if target_type == "ex_prob":
                    # 2D 인덱싱이 필요한 경우 (배치 크기 x n_pairs x 2)
                    y_true_ex = y_true[:, 0]  # ex 값들
                    y_pred_ex = y_pred[:, 0]  # 예측된 ex 값들
                    y_true_prob = y_true[:, 1]  # prob 값들
                    y_pred_prob = y_pred[:, 1]  # 예측된 prob 값들

                    sid_ex = sid_loss(torch.tensor(y_pred_ex + 1e-8, device=device).unsqueeze(0).to(device),
                                      torch.tensor(y_true_ex + 1e-8, device=device).unsqueeze(0).to(device),
                                      torch.ones_like(torch.tensor(y_pred_ex).unsqueeze(0), dtype=torch.bool).to(
                                          device),
                                      threshold=1e-8).mean().item()
                    sid_prob = sid_loss(torch.tensor(y_pred_prob + 1e-8, device=device).unsqueeze(0).to(device),
                                        torch.tensor(y_true_prob + 1e-8, device=device).unsqueeze(0).to(device),
                                        torch.ones_like(torch.tensor(y_pred_prob).unsqueeze(0), dtype=torch.bool).to(
                                            device),
                                        threshold=1e-8).mean().item()

                    sid_combined = sid_loss(torch.tensor(y_pred + 1e-8, device=device).unsqueeze(0).to(device),
                                            torch.tensor(y_true + 1e-8, device=device).unsqueeze(0).to(device),
                                            torch.ones_like(torch.tensor(y_pred).unsqueeze(0), dtype=torch.bool).to(
                                                device),
                                            threshold=1e-8).mean().item()

                    # 스펙트럼별 계산 결과 저장
                    r2_ex = r2_score(y_true_ex, y_pred_ex)
                    r2_prob = r2_score(y_true_prob, y_pred_prob)
                    r2_combined = r2_score(y_true.flatten(), y_pred.flatten())

                    mae_ex = mean_absolute_error(y_true_ex, y_pred_ex)
                    mae_prob = mean_absolute_error(y_true_prob, y_pred_prob)
                    mae_combined = mean_absolute_error(y_true.flatten(), y_pred.flatten())

                    rmse_ex = root_mean_squared_error(y_true_ex, y_pred_ex, )
                    rmse_prob = root_mean_squared_error(y_true_prob, y_pred_prob, )
                    rmse_combined = root_mean_squared_error(y_true.flatten(), y_pred.flatten(), )

                    # print("softdtw_ex = SoftDTWLoss",torch.tensor(y_pred).unsqueeze(0).unsqueeze(-1).shape)
                    softdtw_ex = soft_dtw_fn(
                        torch.tensor(y_pred_ex).unsqueeze(0).unsqueeze(-1).to(device),
                        # (batch_size=1, seq_len, dimension=1)
                        torch.tensor(y_true_ex).unsqueeze(0).unsqueeze(-1).to(device),
                        # (batch_size=1, seq_len, dimension=1)
                    ).item()
                    softdtw_prob = soft_dtw_fn(
                        torch.tensor(y_pred_prob).unsqueeze(0).unsqueeze(-1).to(device),
                        torch.tensor(y_true_prob).unsqueeze(0).unsqueeze(-1).to(device)
                    ).item()
                    softdtw_combined = soft_dtw_fn(
                        torch.tensor(y_pred).unsqueeze(0).to(device),
                        torch.tensor(y_true).unsqueeze(0).to(device)
                    ).item()

                    fastdtw_ex, _ = fastdtw(torch.tensor(y_pred_ex), torch.tensor(y_true_ex))
                    fastdtw_prob, _ = fastdtw(torch.tensor(y_pred_prob), torch.tensor(y_true_prob))
                    fastdtw_combined, _ = fastdtw(torch.tensor(y_pred.flatten()), torch.tensor(y_true.flatten()))

                    # NaN 체크 및 제외 # Nan 아닌 경우 SIS 계산

                    if not math.isnan(sid_ex):
                        sid_spectrum_ex.append(sid_ex)
                        sis_ex = 1 / (1 + sid_ex)
                    else:
                        nan_case_ex += 1
                        print(f"[SID_ex] NaN detected at case {i}")

                    if not math.isnan(sid_prob):
                        sid_spectrum_prob.append(sid_prob)
                        sis_prob = 1 / (1 + sid_prob)
                    else:
                        nan_case_prob += 1
                        print(f"[SID_prob] NaN detected at case {i}")

                    if not math.isnan(sid_combined):
                        sid_spectrum_combined.append(sid_combined)
                        sis_combined = 1 / (1 + sid_combined)
                    else:
                        nan_case_combined += 1
                        print(f"[SID_combined] NaN detected at case {i}")
                    try:
                        sis_spectrum_ex.append(sis_ex)
                    except UnboundLocalError:
                        print(sis_ex)
                    sis_spectrum_prob.append(sis_prob)
                    sis_spectrum_combined.append(sis_combined)

                    r2_spectrum_ex.append(r2_ex)
                    r2_spectrum_prob.append(r2_prob)
                    r2_spectrum_combined.append(r2_combined)

                    mae_spectrum_ex.append(mae_ex)
                    mae_spectrum_prob.append(mae_prob)
                    mae_spectrum_combined.append(mae_combined)

                    rmse_spectrum_ex.append(rmse_ex)
                    rmse_spectrum_prob.append(rmse_prob)
                    rmse_spectrum_combined.append(rmse_combined)

                    softdtw_spectrum_ex.append(softdtw_ex)
                    softdtw_spectrum_prob.append(softdtw_prob)
                    softdtw_spectrum_combined.append(softdtw_combined)

                    fastdtw_spectrum_ex.append(fastdtw_ex)
                    fastdtw_spectrum_prob.append(fastdtw_prob)
                    fastdtw_spectrum_combined.append(fastdtw_combined)

                elif target_type == "exp_spectrum":
                    for i in range(targets.size(0)):
                        y_true = targets[i].cpu().numpy()
                        y_pred = outputs[i].cpu().detach().numpy()
                        mask = batch["masks"][i].cpu().numpy()

                        y_true_masked = y_true[mask == 1]
                        y_pred_masked = y_pred[mask == 1]

                        r2 = r2_score(y_true_masked, y_pred_masked)
                        mae = mean_absolute_error(y_true_masked, y_pred_masked)
                        rmse = root_mean_squared_error(y_true_masked, y_pred_masked,)
                        softdtw = soft_dtw_fn(
                            torch.tensor(y_pred_masked).unsqueeze(0).unsqueeze(-1).to(device),
                            torch.tensor(y_true_masked).unsqueeze(0).unsqueeze(-1).to(device)
                        ).item()
                        #fastdtw_val, _ = fastdtw(y_pred_masked, y_true_masked)
                        sid = sid_loss(
                            torch.tensor(y_pred_masked + 1e-8).unsqueeze(0).to(device),
                            torch.tensor(y_true_masked + 1e-8).unsqueeze(0).to(device),
                            torch.ones_like(torch.tensor(y_pred_masked).unsqueeze(0), dtype=torch.bool).to(device),
                            threshold=1e-8
                        ).mean().item()
                        sis = 1 / (1 + sid)

                        # 저장
                        r2_spectrum.append(r2)
                        mae_spectrum.append(mae)
                        rmse_spectrum.append(rmse)
                        softdtw_spectrum.append(softdtw)
                        #fastdtw_spectrum.append(fastdtw_val)
                        sid_spectrum.append(sid)
                        sis_spectrum.append(sis)

                elif target_type == "nm_distribution":
                    for i in range(targets.size(0)):
                        time_start = time.time()
                        y_true = targets[i].cpu().numpy()
                        y_pred = outputs[i].cpu().detach().numpy()

                        # 정규화나 마스킹 없이 바로 사용 가능
                        r2 = r2_score(y_true, y_pred)
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = root_mean_squared_error(y_true, y_pred,)
                        softdtw = soft_dtw_fn(
                            torch.tensor(y_pred).unsqueeze(0).unsqueeze(-1).to(device),
                            torch.tensor(y_true).unsqueeze(0).unsqueeze(-1).to(device)
                        ).item()
                        #fastdtw_val, _ = fastdtw(y_pred, y_true)
                        sid = sid_loss(
                            torch.tensor(y_pred + 1e-8).unsqueeze(0).to(device),
                            torch.tensor(y_true + 1e-8).unsqueeze(0).to(device),
                            torch.ones_like(torch.tensor(y_pred).unsqueeze(0), dtype=torch.bool).to(device),
                            threshold=1e-8
                        ).mean().item()

                        #print("y_pred", y_pred)
                        #print("y_true", y_true)
                        print("nm distibution sid",sid)
                        sis = 1 / (1 + sid)

                        r2_spectrum.append(r2)
                        mae_spectrum.append(mae)
                        rmse_spectrum.append(rmse)
                        softdtw_spectrum.append(softdtw)
                        #fastdtw_spectrum.append(fastdtw_val)
                        sid_spectrum.append(sid)
                        sis_spectrum.append(sis)
                        time_end = time.time()
                        total_time = time_end - time_start
                        #print("total_evaluation_time", total_time)

                else:
                    raise ValueError(f"Unknown target_type: {target_type}")

    # 스펙트럼별 평균 계산
    if target_type == "ex_prob":
        r2_avg_ex = np.mean(r2_spectrum_ex)
        r2_avg_prob = np.mean(r2_spectrum_prob)
        r2_avg_combined = np.mean(r2_spectrum_combined)

        mae_avg_ex = np.mean(mae_spectrum_ex)
        mae_avg_prob = np.mean(mae_spectrum_prob)
        mae_avg_combined = np.mean(mae_spectrum_combined)

        rmse_avg_ex = np.mean(rmse_spectrum_ex)
        rmse_avg_prob = np.mean(rmse_spectrum_prob)
        rmse_avg_combined = np.mean(rmse_spectrum_combined)

        softdtw_avg_ex = np.mean(softdtw_spectrum_ex)
        softdtw_avg_prob = np.mean(softdtw_spectrum_prob)
        softdtw_avg_combined = np.mean(softdtw_spectrum_combined)

        fastdtw_avg_ex = np.mean([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in fastdtw_spectrum_ex])
        fastdtw_avg_prob = np.mean([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in fastdtw_spectrum_prob])
        fastdtw_avg_combined = np.mean(
            [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in fastdtw_spectrum_combined])

        sid_avg_ex = np.mean(sid_spectrum_ex) if sid_spectrum_ex else np.nan
        sid_avg_prob = np.mean(sid_spectrum_prob) if sid_spectrum_prob else np.nan
        sid_avg_combined = np.mean(sid_spectrum_combined) if sid_spectrum_combined else np.nan

        sis_avg_ex = np.mean(sis_spectrum_ex)
        sis_avg_prob = np.mean(sis_spectrum_prob)
        sis_avg_combined = np.mean(sis_spectrum_combined)
        results = {}

        print("SID ex 평균 (NaN 제외):", sid_avg_ex, "NaN 개수:", nan_case_ex)
        print("SID prob 평균 (NaN 제외):", sid_avg_prob, "NaN 개수:", nan_case_prob)
        print("SID combined 평균 (NaN 제외):", sid_avg_combined, "NaN 개수:", nan_case_combined)
        # 결과 저장용 딕셔너리 생성
        if is_cv == True and is_val == False:
            column_header_front = "CV_train"
        elif is_cv == True and is_val == True:
            column_header_front = "CV_val"
        elif is_cv == False and is_val == False:
            column_header_front = "train"
        elif is_cv == False and is_val == True:
            column_header_front = "test"
        else:
            column_header_front = "Unknown"

        result_dict = {
            f"best_epoch_{column_header_front}": best_epoch,
            f"r2_avg_ex_{column_header_front}": r2_avg_ex,
            f"r2_avg_prob_{column_header_front}": r2_avg_prob,
            f"r2_avg_combined_{column_header_front}": r2_avg_combined,
            f"mae_avg_ex_{column_header_front}": mae_avg_ex,
            f"mae_avg_prob_{column_header_front}": mae_avg_prob,
            f"mae_avg_combined_{column_header_front}": mae_avg_combined,
            f"rmse_avg_ex_{column_header_front}": rmse_avg_ex,
            f"rmse_avg_prob_{column_header_front}": rmse_avg_prob,
            f"rmse_avg_combined_{column_header_front}": rmse_avg_combined,
            f"softdtw_avg_ex_{column_header_front}": softdtw_avg_ex,
            f"softdtw_avg_prob_{column_header_front}": softdtw_avg_prob,
            f"softdtw_avg_combined_{column_header_front}": softdtw_avg_combined,
            f"fastdtw_avg_ex_{column_header_front}": fastdtw_avg_ex,
            f"fastdtw_avg_prob_{column_header_front}": fastdtw_avg_prob,
            f"fastdtw_avg_combined_{column_header_front}": fastdtw_avg_combined,
            f"sid_avg_ex_{column_header_front}": sid_avg_ex,
            f"sid_avg_prob_{column_header_front}": sid_avg_prob,
            f"sid_avg_combined_{column_header_front}": sid_avg_combined,
            f"sis_avg_ex_{column_header_front}": sis_avg_ex,
            f"sis_avg_prob_{column_header_front}": sis_avg_prob,
            f"sis_avg_combined_{column_header_front}": sis_avg_combined,
        }

    else:
        # exp_spectrum, nm_distribution 공통 처리
        r2_avg = np.mean(r2_spectrum)
        mae_avg = np.mean(mae_spectrum)
        rmse_avg = np.mean(rmse_spectrum)
        softdtw_avg = np.mean(softdtw_spectrum)
        #fastdtw_avg = np.mean([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in fastdtw_spectrum])
        sid_avg = np.mean(sid_spectrum) if sid_spectrum else np.nan
        sis_avg = np.mean(sis_spectrum)

        # 결과 저장용 딕셔너리 생성
        if is_cv and not is_val:
            column_header_front = "CV_train"
        elif is_cv and is_val:
            column_header_front = "CV_val"
        elif not is_cv and not is_val:
            column_header_front = "train"
        elif not is_cv and is_val:
            column_header_front = "test"
        else:
            column_header_front = "Unknown"

        result_dict = {
            f"best_epoch_{column_header_front}": best_epoch,
            f"r2_avg_{column_header_front}": r2_avg,
            f"mae_avg_{column_header_front}": mae_avg,
            f"rmse_avg_{column_header_front}": rmse_avg,
            f"softdtw_avg_{column_header_front}": softdtw_avg,
            #f"fastdtw_avg_{column_header_front}": fastdtw_avg,
            f"sid_avg_{column_header_front}": sid_avg,
            f"sis_avg_{column_header_front}": sis_avg,
        }
    return result_dict

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
    QM_exp_data = "QM" # exp
    if QM_exp_data == "QM":
        dataset_train_path = "../../graphormer_data/train_50_with_features.csv"
        dataset_test_path = "../../graphormer_data/test_10_with_features.csv"
        print(QM_exp_data)
    elif QM_exp_data == "exp":
        dataset_train_path = "../../graphormer_data/NIST_with_fake_golbal.csv"
        dataset_test_path = "../../graphormer_data/NIST_with_fake_golbal.csv"
        print(QM_exp_data)
    else:
        print("path error")

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
        "output_size": 100, # 100(ex_prob), 601(nm_ditribution), exp_spectrum 100
        "num_categorical_features": 7,  # (= 7 atom categorical)
        "num_continuous_features": 2,  # (= 2 atom continuous)
        "mode": "cls_global_data", # "cls_only" , "cls_global_data", "cls_global_model"
        "target_type": "ex_prob", # "default", "ex_prob", "nm_distribution", "exp_spectrum"
        "intensity_normalize": "min_max",
        "intensity_range" : (1,100)
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
        res, best_path = train_model_ex_porb(
            config=config,
            target_type=config.get("target_type"),
            loss_function_full_spectrum = loss_name,  # MSE, MAE, SoftDTW, SID
            loss_function_nm_point = loss_name,
            loss_function_ex= ["MAE", "MSE"],
            loss_function_prob= ["SoftDTW"],
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
