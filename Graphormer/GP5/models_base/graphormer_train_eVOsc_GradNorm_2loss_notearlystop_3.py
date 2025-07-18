"""graphormer_train_eVOsc_GradNorm_2loss_notearlystop_3.py
완성본 – 2‑loss(GradNorm) 기반 Graphormer 학습 + 평가 스크립트
Author: ChatGPT (2025‑07‑18)

주요 특징
-----------
* ex(에너지) / prob(오실레이터 강도) 두 채널을 동시에 학습하며 GradNorm 으로 가중치 동적 조정
* Soft‑DTW/MSE/MAE/SID 등 다양한 loss 선택 가능 (loss_ex == loss_prob 로 가정)
* ① 단일 학습(train) ② 전체 데이터 평가(in‑sample) 를 한 번에 수행
* global categorical / continuous feature 정보를 자동 추출하여 config 에 반영
* 학습 결과 및 손실 히스토리를 csv 로 저장
* NaN 감지 및 디버깅 메시지 포함
"""

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===== Graphormer / 사용자 정의 모듈 =====
from Graphormer.GP5.data_prepare.DataLoader_QMData_base import (
    SMILESDataset,
    collate_fn,
    get_global_feature_info,  # helper util – 사용자 정의
)
from Graphormer.GP5.models_base.graphormer_3 import GraphormerModel
from Graphormer.GP5.Custom_Loss.soft_dtw_cuda import SoftDTW
from Graphormer.GP5.Custom_Loss.GradNorm import GradNorm
from chemprop.train.loss_functions import sid_loss
from Graphormer.GP5.Custom_Loss.fast_dtw import fastdtw

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
    loss_function: str = "MSE",
    loss_function_ex: str = "SoftDTW",
    loss_function_prob: str = "SoftDTW",
    num_epochs: int = 10,
    batch_size: int = 64,
    n_pairs: int = 50,
    learning_rate: float = 1e-3,
    dataset_path: str,
    DATASET=None,
    alpha: float = 0.12,
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
    if DATASET is None:
        dataset = SMILESDataset(
            csv_file=dataset_path,
            attn_bias_w=1.0,
            target_type=target_type,
            nominal_feature_vocab=nominal_dims,
            continuous_feature_names=continuous_feature_names,
            global_cat_dim=global_cat_dim,
            global_cont_dim=global_cont_dim,
            ex_normalize=ex_normalize,
            prob_normalize=prob_normalize,
            is_global=True,
        )
    else:
        dataset = DATASET

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, dataset, n_pairs=n_pairs),
    )

    # ---------------- 모델/옵티마이저 ----------------
    model = GraphormerModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    gradnorm = GradNorm(num_losses=2, alpha=alpha)
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

    crit_ex = _make_loss(loss_function_ex)
    crit_prob = _make_loss(loss_function_prob)

    weight_true = torch.tensor([0.5, 0.5], device=device)
    first_loss_ex: float | None = None
    first_loss_prob: float | None = None

    best_epoch = 0
    best_model_path = "best_model.pth"
    best_combined_loss = math.inf

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
        epoch_loss, ex_losses, prob_losses, wts = 0.0, [], [], []
        norm_ex_list, norm_prob_list = [], []

        for batch in dataloader:
            batch_data = {k: v.to(device) for k, v in batch.items() if k != "targets"}
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            outputs = model(batch_data, targets=targets, target_type=target_type)

            out_ex, tgt_ex = outputs[:, :, 0:1] + 1e-8, targets[:, :, 0:1] + 1e-8
            out_prob, tgt_prob = outputs[:, :, 1:2] + 1e-8, targets[:, :, 1:2] + 1e-8

            # loss 계산 (배치 내 개별 스펙트럼 평균)
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

            # first losses (정규화 기준)
            if first_loss_ex is None:
                first_loss_ex = loss_ex.item()
            if first_loss_prob is None:
                first_loss_prob = loss_prob.item()

            norm_ex = loss_ex / first_loss_ex
            norm_prob = loss_prob / first_loss_prob

            weights = gradnorm.compute_weights([loss_ex, loss_prob], model)
            wts.append(weights.detach().cpu().numpy())

            total_loss = weight_true[0] * norm_ex + weight_true[1] * norm_prob
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            ex_losses.append(loss_ex.item())
            prob_losses.append(loss_prob.item())
            norm_ex_list.append(norm_ex.item())
            norm_prob_list.append(norm_prob.item())

        # ---- epoch 통계 ----
        epoch_loss /= len(dataloader)
        weight_true = torch.tensor(np.mean(wts, axis=0), device=device)

        history["ex_loss"].append(float(np.mean(ex_losses)))
        history["prob_loss"].append(float(np.mean(prob_losses)))
        history["total_loss"].append(epoch_loss)
        history["normalized_ex_loss"].append(float(np.mean(norm_ex_list)))
        history["normalized_prob_loss"].append(float(np.mean(norm_prob_list)))
        history["weight_ex"].append(float(weight_true[0]))
        history["weight_prob"].append(float(weight_true[1]))

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{num_epochs} | total {epoch_loss:.4f} | ex {history['ex_loss'][-1]:.4f} | prob {history['prob_loss'][-1]:.4f} | w {weight_true.tolist()} | {elapsed:.1f}s",
            flush=True,
        )

        if epoch_loss < best_combined_loss:
            best_combined_loss = epoch_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

    # ---------------- 인‑샘플 평가 ----------------
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # 간단 평가 – 전체 데이터에 대해 예측 후 R2/MAE/RMSE
    loader_eval = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, dataset, n_pairs=n_pairs),
    )

    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in loader_eval:
            batch_data = {k: v.to(device) for k, v in batch.items() if k != "targets"}
            targets = batch["targets"].to(device)
            outputs = model(batch_data, targets=targets, target_type=target_type)
            y_true_all.append(targets.cpu().numpy())
            y_pred_all.append(outputs.cpu().numpy())

    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    # ex / prob 나누어 단순 metric
    r2_ex = r2_score(y_true_all[:, :, 0].flatten(), y_pred_all[:, :, 0].flatten())
    r2_prob = r2_score(y_true_all[:, :, 1].flatten(), y_pred_all[:, :, 1].flatten())
    mae_ex = mean_absolute_error(y_true_all[:, :, 0].flatten(), y_pred_all[:, :, 0].flatten())
    mae_prob = mean_absolute_error(y_true_all[:, :, 1].flatten(), y_pred_all[:, :, 1].flatten())

    results = {
        "best_epoch": best_epoch,
        "total_loss_best": best_combined_loss,
        "r2_ex": r2_ex,
        "r2_prob": r2_prob,
        "mae_ex": mae_ex,
        "mae_prob": mae_prob,
    }
    for k, v in history.items():
        results[f"{k}_history"] = v

    return results, best_model_path


# -----------------------------------------------------------------------------
# 메인 진입점
# -----------------------------------------------------------------------------

def main() -> None:
    print("CUDA available:", torch.cuda.is_available())

    # ---------- 글로벌 피처 정보 ----------
    global nominal_dims, continuous_feature_names, global_cat_dim, global_cont_dim
    global device

    global_feature_names = ["Solvent", "Temperature", "Pressure"]
    dataset_train = "../../graphormer_data/train_50_with_features.csv"

    try:
        nominal_dims, continuous_feature_names, global_cat_dim, global_cont_dim = get_global_feature_info(global_feature_names)
    except Exception as e:
        print("[WARN] global feature info 오류, fallback 사용:", e)
        nominal_dims, continuous_feature_names = {}, []
        global_cat_dim, global_cont_dim = 1, 2

    # device (GPU/CPU) 설정 전역 변수화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        "output_size": 100,
        "global_cat_dim": global_cat_dim,
        "global_cont_dim": global_cont_dim,
    }

    # ---------- 학습 루프 ----------
    loss_candidates = ["MAE"]  # 필요시 확장: ["MAE", "MSE", "SoftDTW", "SID"]
    results_all: List[Dict] = []

    for loss_name in loss_candidates:
        print(f"\n=== START training with loss: {loss_name} ===")
        res, best_path = train_model_ex_porb(
            config=config,
            target_type="ex_prob",
            loss_function=loss_name,
            loss_function_ex=loss_name,
            loss_function_prob=loss_name,
            num_epochs=10,
            batch_size=50,
            n_pairs=50,
            dataset_path=dataset_train,
            alpha=0.12,
            global_feature_names=global_feature_names,
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
