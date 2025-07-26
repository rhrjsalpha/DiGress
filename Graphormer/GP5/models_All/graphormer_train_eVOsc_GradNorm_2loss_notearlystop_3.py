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
    is_cv: bool = False,
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
        dataset = UnifiedSMILESDataset(
            csv_file=dataset_path,
            nominal_feature_vocab=nominal_dims,
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
            print("outputs.shape",outputs.shape, targets.shape)
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

    # 스펙트럼별 결과 저장용 리스트 초기화
    # Initialize lists for individual and combined metrics
    sid_spectrum_ex, sid_spectrum_prob, sid_spectrum_combined = [], [], []
    sis_spectrum_ex, sis_spectrum_prob, sis_spectrum_combined = [], [], []
    r2_spectrum_ex, r2_spectrum_prob, r2_spectrum_combined = [], [], []
    mae_spectrum_ex, mae_spectrum_prob, mae_spectrum_combined = [], [], []
    rmse_spectrum_ex, rmse_spectrum_prob, rmse_spectrum_combined = [], [], []
    softdtw_spectrum_ex, softdtw_spectrum_prob, softdtw_spectrum_combined = [], [], []
    fastdtw_spectrum_ex, fastdtw_spectrum_prob, fastdtw_spectrum_combined = [], [], []

    y_true_prob_nan_cases = []
    y_pred_prob_nan_cases = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            batched_data = {k: batch[k] for k in
                            ["x_cat", "x_cont", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
                             "attn_edge_type"]}

            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=target_type)
            #outputs = torch.clamp(outputs, min=1e-8)

            sid_spectrum_ex, sid_spectrum_prob, sid_spectrum_combined = [], [], []

            nan_case_ex, nan_case_prob, nan_case_combined = 0, 0, 0  # NaN 카운트

            # Batch에서 각 스펙트럼별로 계산
            for i in range(targets.size(0)):  # batch_size 만큼 루프
                y_true = targets[i].cpu().numpy()
                #print("y_true.shape)",y_true.shape)
                y_pred = outputs[i].cpu().detach().numpy()
                #print("y_pred.shape",y_pred.shape)
                if target_type == "ex_prob":
                    # 2D 인덱싱이 필요한 경우 (배치 크기 x n_pairs x 2)
                    y_true_ex = y_true[:, 0]  # ex 값들
                    y_pred_ex = y_pred[:, 0]  # 예측된 ex 값들
                    y_true_prob = y_true[:, 1]  # prob 값들
                    y_pred_prob = y_pred[:, 1]  # 예측된 prob 값들


                else:
                    raise ValueError(f"Unknown target_type: {target_type}")

                # SID 및 SIS 계산

                #print("sid shape",torch.tensor(y_pred_ex).unsqueeze(0).to(device).shape, torch.tensor(y_true_ex).unsqueeze(0).to(device).shape)
                #print(torch.ones_like(torch.tensor(y_pred_ex).unsqueeze(0), dtype=torch.bool).to(device).shape)
                sid_ex = sid_loss(torch.tensor(y_pred_ex).unsqueeze(0).to(device),
                                  torch.tensor(y_true_ex).unsqueeze(0).to(device),
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

                rmse_ex = mean_squared_error(y_true_ex, y_pred_ex, squared=False)
                rmse_prob = mean_squared_error(y_true_prob, y_pred_prob, squared=False)
                rmse_combined = mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False)

                #print("softdtw_ex = SoftDTWLoss",torch.tensor(y_pred).unsqueeze(0).unsqueeze(-1).shape)
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


                sis_spectrum_ex.append(sis_ex)
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

    # 스펙트럼별 평균 계산
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

    results.update({
        "best_epoch": best_epoch,
        "r2_avg_ex": r2_avg_ex,
        "r2_avg_prob": r2_avg_prob,
        "r2_avg_combined": r2_avg_combined,
        "mae_avg_ex": mae_avg_ex,
        "mae_avg_prob": mae_avg_prob,
        "mae_avg_combined": mae_avg_combined,
        "rmse_avg_ex": rmse_avg_ex,
        "rmse_avg_prob": rmse_avg_prob,
        "rmse_avg_combined": rmse_avg_combined,
        "softdtw_avg_ex": softdtw_avg_ex,
        "softdtw_avg_prob": softdtw_avg_prob,
        "softdtw_avg_combined": softdtw_avg_combined,
        "fastdtw_avg_ex": fastdtw_avg_ex,
        "fastdtw_avg_prob": fastdtw_avg_prob,
        "fastdtw_avg_combined": fastdtw_avg_combined,
        "sid_avg_ex": sid_avg_ex,
        "sid_avg_prob": sid_avg_prob,
        "sid_avg_combined": sid_avg_combined,
        "sis_avg_ex": sis_avg_ex,
        "sis_avg_prob": sis_avg_prob,
        "sis_avg_combined": sis_avg_combined,
    })

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
    global nominal_dims, continuous_feature_names, global_cat_dim, global_cont_dim
    global device

    GLOBAL_FEATURE_NAMES = ['Solvent', 'Temperature', 'Pressure']
    dataset_train = "../../graphormer_data/train_50_with_features.csv"

    try:
        nominal_dims, continuous_feature_names, global_cat_dim, global_cont_dim = get_global_feature_info(GLOBAL_FEATURE_NAMES, PREDEFINED_VOCAB)
        print(nominal_dims)
        print(continuous_feature_names)
        print(global_cat_dim)
        print(global_cont_dim)
    except Exception as e:
        print("[WARN] global feature info 오류, fallback 사용:", e)
        nominal_dims, continuous_feature_names = {}, []
        global_cat_dim, global_cont_dim = 1, 2

    # device (GPU/CPU) 설정 전역 변수화
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
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
        "num_categorical_features": 7,  # (= 7 atom categorical)
        "num_continuous_features": 2,  # (= 2 atom continuous)
        "mode": "cls_global_model", # "cls_only" , "cls_global_data", "cls_global_model"
        "target_type": "ex_prob", # "default", "ex_prob", "nm_distribution"
    }
    config.update({
        "ATOM_FEATURES_VOCAB": ATOM_FEATURES_VOCAB,
        "float_feature_keys": float_feature_keys,
        "BOND_FEATURES_VOCAB": BOND_FEATURES_VOCAB,
        "ex_normalize": "ex_min_max",
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
            target_type="ex_prob",
            loss_function=loss_name,
            loss_function_ex=loss_name,
            loss_function_prob=loss_name,
            num_epochs=10,
            batch_size=50,
            n_pairs=50,
            dataset_path=dataset_train,
            alpha=0.12,
            global_feature_names=GLOBAL_FEATURE_NAMES,
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
