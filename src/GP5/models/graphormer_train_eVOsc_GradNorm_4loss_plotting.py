import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset, collate_fn
from Graphormer.GP5.models.graphormer import GraphormerModel
import os
from Graphormer.GP5.Custom_Loss.custom_loss import fastdtw_loss
from Graphormer.GP5.Custom_Loss.SID_loss import SIDLoss
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Graphormer.GP5.Custom_Loss.fast_dtw import fastdtw
import time
from chemprop.train.loss_functions import sid_loss
from torch.cuda.amp import autocast, GradScaler
from tslearn.metrics import SoftDTWLossPyTorch
from Graphormer.GP5.Custom_Loss.GradNorm import GradNorm
import math
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

def plot_predictions(y_true_ex_list, y_pred_ex_list, y_true_prob_list, y_pred_prob_list, save_dir="./predictions"):
    os.makedirs(save_dir, exist_ok=True)  # 저장 디렉토리 생성

    num_samples = len(y_true_ex_list)  # 모든 샘플 저장

    for i in range(num_samples):
        fig, axs = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)

        # ✅ (1) eV 값 플롯 (기존 plot 유지)
        axs[0].plot(y_true_ex_list[i], label=f"True Ex {i}", marker='o', linestyle='-', alpha=0.7, color='blue')
        axs[0].plot(y_pred_ex_list[i], label=f"Pred Ex {i}", marker='x', linestyle='--', alpha=0.7, color='orange')
        axs[0].set_title(f"Predicted vs True eV Values (Ex) - Sample {i}")
        axs[0].set_xlabel("Index")
        axs[0].set_ylabel("eV")
        axs[0].legend()
        axs[0].grid(True)

        # ✅ (2) Oscillator Strength 값 플롯 (기존 plot 유지)
        axs[1].plot(y_true_prob_list[i], label=f"True Osc {i}", marker='o', linestyle='-', alpha=0.7, color='blue')
        axs[1].plot(y_pred_prob_list[i], label=f"Pred Osc {i}", marker='x', linestyle='--', alpha=0.7, color='orange')
        axs[1].set_title(f"Predicted vs True Oscillator Strength Values (Prob) - Sample {i}")
        axs[1].set_xlabel("Index")
        axs[1].set_ylabel("Oscillator Strength")
        axs[1].legend()
        axs[1].grid(True)

        # ✅ (3) 전체 스펙트럼 플롯 (수직선 + 마커 추가)
        # True Spectrum → o 마커
        axs[2].vlines(y_true_ex_list[i], ymin=0, ymax=y_true_prob_list[i], color='blue', alpha=0.7)
        axs[2].scatter(y_true_ex_list[i], y_true_prob_list[i], color='blue', label=f"True Spectrum {i}", marker='o')

        # Pred Spectrum → x 마커
        axs[2].vlines(y_pred_ex_list[i], ymin=0, ymax=y_pred_prob_list[i], color='orange', alpha=0.7, linestyles='dashed')
        axs[2].scatter(y_pred_ex_list[i], y_pred_prob_list[i], color='orange', label=f"Pred Spectrum {i}", marker='x')

        axs[2].set_title(f"Predicted vs True Spectrum - Sample {i}")
        axs[2].set_xlabel("eV")
        axs[2].set_ylabel("Oscillator Strength")
        axs[2].legend()
        axs[2].grid(True)

        # ✅ 파일 저장
        save_path = os.path.join(save_dir, f"molecule_{i}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    print(f"Plots saved to '{save_dir}'")  # 저장된 디렉토리 출력



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def loss_normalizer(first_loss, loss):
    # 손실 값이 tensor인 경우 정규화 값도 tensor로 유지
    if first_loss is not None and isinstance(loss, torch.Tensor):
        normalized_loss = loss / first_loss
    else:
        normalized_loss = loss
    return normalized_loss


def train_model_ex_porb(
    config,
    target_type="ex_prob",
    loss_function="MSE",
    loss_function_ex_1="SoftDTW",
    loss_function_ex_2 = "SID",
    loss_function_prob_1="SoftDTW",
    loss_function_prob_2 = "SID",
    # weight_ex=0.5,
    num_epochs=10,
    batch_size=50,
    n_pairs = 50,
    learning_rate=0.001,
    dataset_path="../../data/data_example.csv",
    patience = 20,
    DATASET = None,
    alpha = 0.12
):
    """
    Train the Graphormer model with specified configurations and return the final loss and evaluation metrics.

    Args:
        config (dict): Configuration for the Graphormer model.
        target_type (str): Target type ("default", "ex_prob", "nm_distribution").
        loss_function_ex (str): Loss function for 'ex'.
        loss_function_prob (str): Loss function for 'prob'.
        weight_ex (float): Weight for 'ex' loss. Weight for 'prob' will be 1 - weight_ex.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        n_pairs (int): Number of pairs for 'ex_prob' targets.
        learning_rate (float): Learning rate for the optimizer.
        dataset_path (str): Path to the dataset CSV file.

    Returns:
        dict: A dictionary containing the final loss and evaluation metrics.
    """
    # Initialize dataset and DataLoader
    if DATASET is None:
        dataset = SMILESDataset(csv_file=dataset_path, attn_bias_w=1.0, target_type=target_type)
    else:
        dataset = DATASET

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, dataset, n_pairs=n_pairs),
    )
    #for data in dataloader:
    #    for key in data:
    #        print(key, data[key].shape)
    # Initialize the model, loss function, and optimizer
    model = GraphormerModel(config)
    #SoftDTWLoss = SoftDTW(use_cuda=True, gamma=1.0, bandwidth=None)
    SoftDTWLoss = SoftDTW(use_cuda=True, gamma=0.2, bandwidth=None, normalize=True)

    def loss_fn_gen(loss_fn):
        if loss_fn == 'MSE':
            return nn.MSELoss()
        elif loss_fn == 'MAE':
            return nn.L1Loss()
        elif loss_fn == 'SoftDTW':
            return SoftDTWLoss
        elif loss_fn == 'Huber':
            return nn.SmoothL1Loss()
        elif loss_fn == 'SID':
            def sid_loss_wrapper(model_spectra, target_spectra, mask, threshold):
                return sid_loss(model_spectra, target_spectra, mask, threshold)
            return sid_loss_wrapper

    criterion_ex_1 = loss_fn_gen(loss_function_ex_1)
    criterion_ex_2 = loss_fn_gen(loss_function_ex_2)
    criterion_prob_1 = loss_fn_gen(loss_function_prob_1)
    criterion_prob_2 = loss_fn_gen(loss_function_prob_2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_modifier = GradNorm(num_losses=4, alpha=alpha)


    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scaler = GradScaler()

    #weight_prob = 1 - weight_ex

    # 손실 값 저장을 위한 리스트 초기화
    best_loss_ex_1 = float('inf')
    best_loss_ex_2 = float('inf')
    best_loss_prob_1 = float('inf')
    best_loss_prob_2 = float('inf')
    best_model_path = "./best_model.pth"
    best_epoch = 0
    patience = patience  # Early stopping patience 설정

    # 개선되지 않은 epoch 수 추적
    no_improve_count_ex_1 = 0
    no_improve_count_ex_2 = 0
    no_improve_count_prob_1 = 0
    no_improve_count_prob_2 = 0

    # Training loop
    #loss_dict = {"ex_loss":[], "prob_loss":[], "total_loss":[]}
    loss_history = {"ex_loss_1": [],"ex_loss_2": [], "prob_loss_1": [],"prob_loss_2": [], "total_loss": [],
                    "normalized_ex_loss_1":[], "normalized_ex_loss_2":[], "normalized_prob_loss_1": [], "normalized_prob_loss_2":[],}
    weight_true = [0.5, 0.5, 0.5, 0.5]

    first_loss_ex_1 = None
    first_loss_ex_2 = None
    first_loss_prob_1 = None
    first_loss_prob_2 = None

    weight_history = {"weight_ex_1": [],"weight_ex_2": [], "weight_prob_1": [], "weight_prob_2": []}

    for epoch in range(num_epochs):
        now_time = time.time()
        model.train()
        epoch_loss = 0.0
        loss_ex_list_1 = []
        loss_ex_list_2 = []
        loss_prob_list_1 = []
        loss_prob_list_2 = []
        normalized_loss_ex_list_1 = []
        normalized_loss_ex_list_2 = []
        normalized_loss_prob_list_1 = []
        normalized_loss_prob_list_2 = []
        weight_list = []

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input", "attn_edge_type"]}
            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=target_type)

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN detected in model outputs!")
                print(f"Sample outputs: {outputs}")
                print(targets)
                raise ValueError("NaN values found in model outputs, check data and model configuration.")



            # Compute loss
            if target_type == "ex_prob":
                # SID Loss를 사용할 경우 마스크 생성
                outputs_ex = outputs[:, :, 0:1] + 1e-8
                targets_ex = targets[:, :, 0:1] + 1e-8
                if loss_function_ex_1 == "SID":
                    threshold = 1e-8
                    mask_ex_1 = torch.ones_like(outputs_ex, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                    loss_ex_1 = torch.stack([
                        criterion_ex_1(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0), mask_ex_1[i],threshold)
                        for i in range(outputs_ex.size(0))
                    ]).mean()
                else:
                    loss_ex_1 = torch.stack([
                        criterion_ex_1(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0))
                        for i in range(outputs_ex.size(0))
                    ]).mean()

                if loss_function_ex_2 == "SID":
                    threshold = 1e-8
                    mask_ex_2 = torch.ones_like(outputs_ex, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                    loss_ex_2 = torch.stack([
                        criterion_ex_2(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0), mask_ex_2[i],threshold)
                        for i in range(outputs_ex.size(0))
                    ]).mean()
                else:
                    loss_ex_2 = torch.stack([
                        criterion_ex_2(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0))
                        for i in range(outputs_ex.size(0))
                    ]).mean()

                #outputs_prob = torch.sigmoid(outputs[:, :, 1:2])
                #targets_prob = torch.sigmoid(targets[:, :, 1:2])
                outputs_prob = outputs[:, :, 1:2] + 1e-8
                targets_prob = targets[:, :, 1:2] + 1e-8
                #outputs_prob = torch.clamp(outputs[:, :, 1:2], min=1e-8)
                #targets_prob = torch.clamp(targets[:, :, 1:2], min=1e-8)

                # SID Loss를 사용할 경우 마스크 생성
                if loss_function_prob_1 == "SID":
                    threshold = 1e-8
                    mask_prob_1 = torch.ones_like(outputs_prob, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                    loss_prob_1 = torch.stack([
                        criterion_prob_1(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0), mask_prob_1[i],threshold)
                        for i in range(outputs_prob.size(0))
                    ]).mean()
                else:
                    loss_prob_1 = torch.stack([
                        criterion_prob_1(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0))
                        for i in range(outputs_prob.size(0))
                    ]).mean()

                if loss_function_prob_2 == "SID":
                    threshold = 1e-8
                    mask_prob_2 = torch.ones_like(outputs_prob, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                    loss_prob_2 = torch.stack([
                        criterion_prob_2(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0), mask_prob_2[i],threshold)
                        for i in range(outputs_prob.size(0))
                    ]).mean()
                else:
                    loss_prob_2 = torch.stack([
                        criterion_prob_2(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0))
                        for i in range(outputs_prob.size(0))
                    ]).mean()

                # 손실 값이 tensor인 경우 정규화 값도 tensor로 유지
                normalized_loss_ex_1 = loss_normalizer(first_loss_ex_1, loss_ex_1)
                normalized_loss_ex_2 = loss_normalizer(first_loss_ex_2, loss_ex_2)
                normalized_loss_prob_1 = loss_normalizer(first_loss_prob_1, loss_prob_1)
                normalized_loss_prob_2 = loss_normalizer(first_loss_prob_2, loss_prob_2)

                weight = loss_modifier.compute_weights([loss_ex_1, loss_ex_2, loss_prob_1, loss_prob_2], model)
                #weight = loss_modifier.compute_weights([normalized_loss_ex, normalized_loss_prob], model)
                weight_list.append(weight)

                # Final loss 계산
                #loss = weight_true[0] * loss_ex + weight_true[1] * loss_prob
                loss = weight_true[0] * normalized_loss_ex_1 + weight_true[1] * normalized_loss_ex_2 + weight_true[2] * normalized_loss_prob_1 + weight_true[3] * normalized_loss_prob_2
            else:
                raise ValueError("Invalid target type")

            if first_loss_ex_1 is None:
                first_loss_ex_1 = loss_ex_1.item()
            if first_loss_ex_2 is None:
                first_loss_ex_2 = loss_ex_2.item()
            if first_loss_prob_1 is None:
                first_loss_prob_1 = loss_prob_1.item()
            if first_loss_prob_2 is None:
                first_loss_prob_2 = loss_prob_2.item()

            normalized_loss_ex_1 = loss_ex_1.item() / first_loss_ex_1 if first_loss_ex_1 != 0 else 1.0
            normalized_loss_ex_2 = loss_ex_2.item() / first_loss_ex_2 if first_loss_ex_2 != 0 else 1.0
            normalized_loss_prob_1 = loss_prob_1.item() / first_loss_prob_1 if first_loss_prob_1 != 0 else 1.0
            normalized_loss_prob_2 = loss_prob_2.item() / first_loss_prob_2 if first_loss_prob_2 != 0 else 1.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loss_ex_list_1.append(loss_ex_1.item())
            loss_ex_list_2.append(loss_ex_2.item())
            loss_prob_list_1.append(loss_prob_1.item())
            loss_prob_list_2.append(loss_prob_2.item())

            normalized_loss_ex_list_1.append(normalized_loss_ex_1)
            normalized_loss_ex_list_2.append(normalized_loss_ex_2)
            normalized_loss_prob_list_1.append(normalized_loss_prob_1)
            normalized_loss_prob_list_2.append(normalized_loss_prob_2)

        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_loss_ex_1 = np.mean(loss_ex_list_1)
        avg_loss_ex_2 = np.mean(loss_ex_list_2)
        avg_loss_prob_1 = np.mean(loss_prob_list_1)
        avg_loss_prob_2 = np.mean(loss_prob_list_2)

        avg_normalized_loss_ex_1 = np.mean(normalized_loss_ex_list_1)
        avg_normalized_loss_ex_2 = np.mean(normalized_loss_ex_list_2)
        avg_normalized_loss_prob_1 = np.mean(normalized_loss_prob_list_1)
        avg_normalized_loss_prob_2 = np.mean(normalized_loss_prob_list_2)

        avg_weight = torch.stack(weight_list).mean(dim=0).detach()
        weight_true = avg_weight
        weight_history["weight_ex_1"].append(weight_true[0].item())
        weight_history["weight_ex_2"].append(weight_true[1].item())
        weight_history["weight_prob_1"].append(weight_true[2].item())
        weight_history["weight_prob_2"].append(weight_true[3].item())

        loss_history["ex_loss_1"].append(avg_loss_ex_1)
        loss_history["ex_loss_2"].append(avg_loss_ex_2)
        loss_history["prob_loss_1"].append(avg_loss_prob_1)
        loss_history["prob_loss_2"].append(avg_loss_prob_2)
        loss_history["total_loss"].append(avg_epoch_loss)
        loss_history["normalized_ex_loss_1"].append(avg_normalized_loss_ex_1)
        loss_history["normalized_ex_loss_2"].append(avg_normalized_loss_ex_2)
        loss_history["normalized_prob_loss_1"].append(avg_normalized_loss_prob_1)
        loss_history["normalized_prob_loss_2"].append(avg_normalized_loss_prob_2)

        # Best model 저장 & Early stopping 체크
        if avg_loss_ex_1 < best_loss_ex_1:
            best_loss_ex_1 = avg_loss_ex_1
            no_improve_count_ex_1 = 0
        else:
            no_improve_count_ex_1 += 1

        if avg_loss_ex_2 < best_loss_ex_2:
            best_loss_ex_2 = avg_loss_ex_2
            no_improve_count_ex_2 = 0
        else:
            no_improve_count_ex_2 += 1

        if avg_loss_prob_1 < best_loss_prob_1:
            best_loss_prob_1 = avg_loss_prob_1
            no_improve_count_prob_1 = 0
        else:
            no_improve_count_prob_1 += 1

        if avg_loss_prob_2 < best_loss_prob_2:
            best_loss_prob_2 = avg_loss_prob_2
            no_improve_count_prob_2 = 0
        else:
            no_improve_count_prob_2 += 1

        # ✅ 두 손실 모두 patience 기준 충족 시 종료
        if (no_improve_count_ex_1 >= patience and
                no_improve_count_ex_2 >= patience and
                no_improve_count_prob_1 >= patience and
                no_improve_count_prob_2 >= patience):
            print(f"Early stopping triggered at epoch {epoch + 1} (All losses exceeded patience)")
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "./best_model.pth")
            break

        epoch_time = time.time() - now_time
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, "
              f"Loss_Ex: {avg_loss_ex_1:.4f},{avg_loss_ex_2:.4f}, Loss_Prob: {avg_loss_prob_1:.4f}, {avg_loss_ex_2:.4f} "
              f"Normalized_Loss_Ex: {avg_normalized_loss_ex_1:.4f},{avg_normalized_loss_ex_2:.4f} "
              f"Normalized_Loss_Prob: {avg_normalized_loss_prob_1:.4f}, {avg_normalized_loss_prob_2:.4f}"
              f"Weights: {weight_true}, Time: {epoch_time:.2f},"
              f"no_improve_count: {no_improve_count_ex_1, no_improve_count_ex_2, no_improve_count_prob_1, no_improve_count_prob_2}")

    # Final evaluation metrics 계산
    model.load_state_dict(torch.load(best_model_path,))
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

    y_true_ex_list = []
    y_pred_ex_list = []
    y_true_prob_list = []
    y_pred_prob_list = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in
                            ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
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
                    y_true_ex_list.append(y_true_ex)
                    y_pred_ex_list.append(y_pred_ex)
                    y_true_prob_list.append(y_true_prob)
                    y_pred_prob_list.append(y_pred_prob)
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
                sid_prob = sid_loss(torch.tensor(y_pred_prob).unsqueeze(0).to(device),
                                    torch.tensor(y_true_prob).unsqueeze(0).to(device),
                                    torch.ones_like(torch.tensor(y_pred_prob).unsqueeze(0), dtype=torch.bool).to(
                                        device),
                                    threshold=1e-8).mean().item()

                sid_combined = sid_loss(torch.tensor(y_pred).unsqueeze(0).to(device),
                                        torch.tensor(y_true).unsqueeze(0).to(device),
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
                softdtw_ex = SoftDTWLoss(
                    torch.tensor(y_pred_ex).unsqueeze(0).unsqueeze(-1).to(device),
                    # (batch_size=1, seq_len, dimension=1)
                    torch.tensor(y_true_ex).unsqueeze(0).unsqueeze(-1).to(device),
                    # (batch_size=1, seq_len, dimension=1)
                ).item()
                softdtw_prob = SoftDTWLoss(
                    torch.tensor(y_pred_prob).unsqueeze(0).unsqueeze(-1).to(device),
                    torch.tensor(y_true_prob).unsqueeze(0).unsqueeze(-1).to(device)
                ).item()
                softdtw_combined = SoftDTWLoss(
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

    plot_predictions(y_true_ex_list, y_pred_ex_list, y_true_prob_list, y_pred_prob_list)

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
    print("loss_history_training", loss_history)
    for key, value in loss_history.items():
        results[f"{key}_history_training"] = value


    # 결과 저장
    #with open("./training_results.json", "w") as f:
    #    json.dump(results, f)
    print("Training complete.")
    return results, best_model_path

###############################################################################
config = {
    "num_atoms": 100,              # 분자의 최대 원자 수 (그래프의 노드 개수) 100
    "num_in_degree": 10,           # 그래프 노드의 최대 in-degree
    "num_out_degree": 10,          # 그래프 노드의 최대 out-degree
    "num_edges": 100,               # 그래프의 최대 엣지 개수 50
    "num_spatial": 100,            # 공간적 위치 인코딩을 위한 최대 값 default 100
    "num_edge_dis": 10,            # 엣지 거리 인코딩을 위한 최대 값
    "edge_type": "multi_hop",      # 엣지 타입 ("multi_hop" 또는 다른 값 가능)
    "multi_hop_max_dist": 2,       # Multi-hop 엣지의 최대 거리
    "num_encoder_layers": 1,       # Graphormer 모델에서 사용할 인코더 레이어 개수
    "embedding_dim": 128,          # 임베딩 차원 크기 (노드, 엣지 등)
    "ffn_embedding_dim": 256,      # Feedforward Network의 임베딩 크기
    "num_attention_heads": 8,      # Multi-head Attention에서 헤드 개수
    "dropout": 0.1,                # 드롭아웃 비율
    "attention_dropout": 0.1,      # Attention 레이어의 드롭아웃 비율
    "activation_dropout": 0.1,     # 활성화 함수 이후 드롭아웃 비율
    "activation_fn": "gelu",       # 활성화 함수 ("gelu", "relu" 등)
    "pre_layernorm": False,        # LayerNorm을 Pre-Normalization으로 사용할지 여부
    "q_noise": 0.0,                # Quantization noise (훈련 중 노이즈 추가를 위한 매개변수)
    "qn_block_size": 8,            # Quantization block 크기
    "output_size": 100,            # 모델 출력 크기
}


# Example usage
if __name__ == "__main__":
    target_type = "ex_prob"  # "ex_prob" "default" "nm_distribution"
    loss_functions_ex_1 = ["SoftDTW"] # "MSE", "MAE", "SoftDTW", "Huber", "SID"
    loss_functions_ex_2 = ["SID"]
    loss_functions_prob_1 = ["SoftDTW"]
    loss_functions_prob_2 = ["SID"]

    for loss_ex_1 in loss_functions_ex_1:
        for loss_ex_2 in loss_functions_ex_2:
            for loss_prob_1 in loss_functions_prob_1:
                for loss_prob_2 in loss_functions_prob_2:
                    print(f"Running training with loss_function_ex={loss_ex_1, loss_ex_2}, loss_function_prob={loss_prob_1, loss_prob_2}")
                    final_loss = train_model_ex_porb(
                        config=config,
                        target_type=target_type,
                        dataset_path="../../data/train_1000.csv",
                        loss_function_ex_1=loss_ex_1,
                        loss_function_ex_2=loss_ex_2,
                        loss_function_prob_1=loss_prob_1,
                        loss_function_prob_2=loss_prob_2,
                        learning_rate=0.001,
                        batch_size=50,
                        num_epochs=100,
                        n_pairs=50,
                        patience=20,
                    )
                    print(f"Final loss for loss_function_ex={loss_ex_1, loss_ex_2}, loss_function_prob={loss_prob_1, loss_prob_2}: {final_loss}")

