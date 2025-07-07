import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset, collate_fn
from Graphormer.GP5.models.graphormer import GraphormerModel
import os
from Graphormer.GP5.Custom_Loss.custom_loss import fastdtw_loss
from Graphormer.GP5.Custom_Loss.soft_dtw_cuda import SoftDTW

from Graphormer.GP5.Custom_Loss.SID_loss import SIDLoss
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Graphormer.GP5.Custom_Loss.fast_dtw import fastdtw
import time
from sklearn.model_selection import KFold
from Graphormer.GP5.models.graphormer_train_eVOsc_GradNorm_2loss import train_model_ex_porb
from chemprop.train.loss_functions import sid_loss
from tslearn.metrics import SoftDTWLossPyTorch
from Graphormer.GP5.Custom_Loss.GradNorm import GradNorm
import math
import pandas as pd

def CV_loss_fn_name_gen(loss_fn, ex_prob): ### CV_r2_ex
    if loss_fn == "SID":
        CV_col_name = f"CV_sid_{ex_prob}_avg"
    elif loss_fn == "SoftDTW":
        CV_col_name = f"CV_softdtw_{ex_prob}_avg"
    elif loss_fn == "MAE":
        CV_col_name = f"CV_mae_{ex_prob}_avg"
    elif loss_fn == "MSE":
        CV_col_name = f"CV_mse_{ex_prob}_avg"
    else:
        print("loss function name is incorrect")
    return CV_col_name

def cross_validate_model(
    config,
    target_type="default",
    loss_function="MSE",
    loss_function_ex="SoftDTW",
    loss_function_prob="SoftDTW",
    # weight_ex=0.5,
    num_epochs=10,
    batch_size=64,
    n_pairs=1,
    learning_rate=0.001,
    dataset_path="../../data/data_example.csv",
    testset_path="../../data/data_example.csv",
    n_splits=3, # Number of folds for Cross Validation
    patience = 20,
    DATASET = None,
    TEST_DATASET = None,
    alpha = 0.12
    ):
    """
    Perform cross-validation and final training on the entire dataset.

    Args:
        config (dict): Model configuration.
        Other arguments are the same as train_model.

    Returns:
        dict: Results containing CV metrics and full dataset metrics.
    """
    if DATASET is None:
        dataset = SMILESDataset(csv_file=dataset_path, attn_bias_w=1.0, target_type=target_type)
    else:
        dataset = DATASET

    # Prepare K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_metrics = {
        "CV_train_r2_ex": [], "CV_train_r2_prob": [], "CV_train_r2_combined": [],
        "CV_train_mae_ex": [], "CV_train_mae_prob": [], "CV_train_mae_combined": [],
        "CV_train_rmse_ex": [], "CV_train_rmse_prob": [], "CV_train_rmse_combined": [],
        "CV_train_mse_ex": [], "CV_train_mse_prob": [], "CV_train_mse_combined": [],
        "CV_train_softdtw_ex": [], "CV_train_softdtw_prob": [], "CV_train_softdtw_combined": [],
        "CV_train_sid_ex": [], "CV_train_sid_prob": [], "CV_train_sid_combined": [],
        "CV_train_sis_ex": [], "CV_train_sis_prob": [], "CV_train_sis_combined": [],
        "CV_r2_ex": [], "CV_r2_prob": [], "CV_r2_combined": [],
        "CV_mae_ex": [], "CV_mae_prob": [], "CV_mae_combined": [],
        "CV_rmse_ex": [], "CV_rmse_prob": [], "CV_rmse_combined": [],
        "CV_mse_ex": [], "CV_mse_prob": [], "CV_mse_combined": [],
        "CV_softdtw_ex": [], "CV_softdtw_prob": [], "CV_softdtw_combined": [],
        #"CV_fastdtw_ex": [], "CV_fastdtw_prob": [], "CV_fastdtw_combined": [],
        "CV_sid_ex": [], "CV_sid_prob": [], "CV_sid_combined": [],
        "CV_sis_ex": [], "CV_sis_prob": [], "CV_sis_combined": [],
        "val_loss": [], "CV_best_epoch":[], "CV_weight_ex": [], "CV_weight_prob": []
    }
    loss_history_all_folds = []  # Store loss history for all folds

    train_min, train_max = None, None

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Processing fold {fold + 1}/{n_splits}")

        # Split dataset into train and validation sets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # targets가 없는 경우 dataset.process_targets() 호출해서 min/max 계산
        all_targets = dataset.process_targets(n_pairs=n_pairs)

        # PyTorch Tensor인 경우
        if isinstance(all_targets, torch.Tensor):
            train_min = all_targets.min(dim=0)[0].min(dim=0)[0]  # 두 차원에서 min 계산
            train_min = train_min[0]
            train_max = all_targets.max(dim=0)[0].max(dim=0)[0]
            train_max = train_max[0]
        # Numpy 배열인 경우
        elif isinstance(all_targets, np.ndarray):
            train_min = all_targets.min(axis=(0, 1))
            train_max = all_targets.max(axis=(0, 1))
        else:
            raise TypeError(f"Unexpected type for targets: {type(all_targets)}")

        print(f"train_min: {train_min}, train_max: {train_max}")

        #print("before load batch_size", batch_size)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, dataset, n_pairs=n_pairs),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, dataset, n_pairs=n_pairs),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            #print("len(batch)",len(batch))
            #print("len(batch[targets]",len(batch["targets"]))
            #print(batch["targets"].shape)

        # Initialize model, loss functions, and optimizer
        model = GraphormerModel(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        SoftDTWLoss = SoftDTW(use_cuda=True, gamma=0.2, bandwidth=None, normalize=True)
        #SoftDTWLoss = SoftDTWLossPyTorch(gamma=0.2, normalize=True)

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

        criterion_ex = loss_fn_gen(loss_function_ex)
        criterion_prob = loss_fn_gen(loss_function_prob)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        loss_modifier = GradNorm(num_losses=2, alpha=alpha)

        patience = patience  # Early stopping patience 설정
        ex_no_improve_count = 0
        prob_no_improve_count = 0
        best_epoch = 0
        best_loss = float('inf')
        best_loss_ex = float('inf')
        best_loss_prob = float('inf')

        weight_true = torch.tensor([0.5, 0.5], device=device)
        first_loss_ex = None
        first_loss_prob = None

        loss_history = {"ex_loss": [], "prob_loss": [], "total_loss": [], "normalized_ex_loss": [], "normalized_prob_loss": []}

        start_count_epoch = 100

        torch.autograd.set_detect_anomaly(True)

        for epoch in range(num_epochs):
            #print("epochs left",epoch,"/",num_epochs)
            model.train()
            epoch_loss = 0.0
            loss_ex_list = []
            loss_prob_list = []
            weight_list = []
            normalized_loss_ex_list = []
            normalized_loss_prob_list = []
            epoch_time_start = time.time()

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                batched_data = {k: batch[k] for k in ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input", "attn_edge_type"]}
                targets = batch["targets"]
                outputs = model(batched_data, targets=targets, target_type=target_type)

                if target_type == "ex_prob":
                    outputs_ex = outputs[:, :, 0:1] + 1e-8
                    targets_ex = targets[:, :, 0:1] + 1e-8

                    # SID Loss를 사용할 경우 마스크 생성
                    if loss_function_ex == "SID":
                        threshold = 1e-4
                        mask_ex = torch.ones_like(outputs_ex, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                        loss_ex = torch.stack([
                            criterion_ex(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0), mask_ex[i], threshold)
                            for i in range(outputs_ex.size(0))
                        ]).mean()

                    elif loss_function_ex == "SoftDTW":
                        loss_ex = criterion_ex(outputs_ex, targets_ex).mean()

                    else:
                        loss_ex = torch.stack([
                            criterion_ex(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0))
                            for i in range(outputs_ex.size(0))
                        ]).mean()
                        #loss_ex = criterion_prob(outputs_ex, targets_ex)

                    # outputs_prob = torch.sigmoid(outputs[:, :, 1:2])
                    # targets_prob = torch.sigmoid(targets[:, :, 1:2])
                    outputs_prob = outputs[:, :, 1:2] + 1e-8
                    targets_prob = targets[:, :, 1:2] + 1e-8
                    #outputs_prob = torch.clamp(outputs[:, :, 1:2], min=1e-8)
                    #targets_prob = torch.clamp(targets[:, :, 1:2], min=1e-8)

                    # SID Loss를 사용할 경우 마스크 생성
                    if loss_function_prob == "SID":
                        threshold = 1e-4
                        mask_prob = torch.ones_like(outputs_prob, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                        loss_prob = torch.stack([
                            criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0), mask_prob[i],
                                           threshold)
                            for i in range(outputs_prob.size(0))
                        ]).mean()

                    elif loss_function_prob == "SoftDTW":
                        loss_prob = criterion_prob(outputs_prob, targets_prob).mean()

                    else:
                        loss_prob = torch.stack([
                            criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0))
                            for i in range(outputs_prob.size(0))
                        ]).mean()
                        #loss_prob = criterion_prob(outputs_prob, targets_prob)

                    # ✅ 첫 배치에서 손실 값 저장
                    if first_loss_ex is None:
                        first_loss_ex = loss_ex.item()
                        first_loss_prob = loss_prob.item()

                    # 손실 값이 tensor인 경우 정규화 값도 tensor로 유지
                    if first_loss_ex is not None and isinstance(loss_ex, torch.Tensor):
                        normalized_loss_ex = loss_ex / first_loss_ex
                    else:
                        normalized_loss_ex = loss_ex

                    if first_loss_prob is not None and isinstance(loss_prob, torch.Tensor):
                        normalized_loss_prob = loss_prob / first_loss_prob
                    else:
                        normalized_loss_prob = loss_prob

                    # ✅ 매 배치마다 GradNorm 적용
                    #weight = loss_modifier.compute_weights([loss_ex, loss_prob], model)
                    weight = loss_modifier.compute_weights([loss_ex, loss_prob], model)
                    weight_list.append(weight.detach().cpu().numpy()) ###### 바꾼부분

                    # ✅ 손실 값 계산
                    # loss = weight_true[0] * loss_ex + weight_true[1] * loss_prob
                    loss = weight_true[0] * normalized_loss_ex + weight_true[1] * normalized_loss_prob
                else:
                    raise ValueError("Invalid target type")

                optimizer.zero_grad()
                loss.backward(retain_graph=False)

                optimizer.step()

                epoch_loss += loss.item()
                loss_ex_list.append(loss_ex.item())
                loss_prob_list.append(loss_prob.item())
                normalized_loss_ex_list.append(normalized_loss_ex.item())
                normalized_loss_prob_list.append(normalized_loss_prob.item())

            weight_true = torch.tensor(np.mean(weight_list, axis=0), device=device)
            print(weight_true)

            avg_loss_ex = np.mean(loss_ex_list)
            avg_loss_prob = np.mean(loss_prob_list)
            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_normalized_loss_ex = np.mean(normalized_loss_ex_list)
            avg_normalized_loss_prob = np.mean(normalized_loss_prob_list)

            print(weight_list)
            avg_weight = np.mean(weight_list, axis=0)
            print(avg_weight)
            cv_metrics["CV_weight_ex"].append(avg_weight[0])
            cv_metrics["CV_weight_prob"].append(avg_weight[1])

            cv_metrics["CV_weight_ex"].append(float(avg_weight[0]))
            cv_metrics["CV_weight_prob"].append(float(avg_weight[1]))

            loss_history["total_loss"].append(avg_epoch_loss)
            loss_history["ex_loss"].append(avg_loss_ex)
            loss_history["prob_loss"].append(avg_loss_prob)

            loss_history["normalized_ex_loss"].append(
                np.mean([x for x in normalized_loss_ex_list])
            )
            loss_history["normalized_prob_loss"].append(
                np.mean([x for x in normalized_loss_prob_list])
            )

            # ✅ Early Stopping 개별 손실 기준 적용
            if epoch >= start_count_epoch:
                if avg_loss_ex < best_loss_ex:
                    best_loss_ex = avg_loss_ex
                    ex_no_improve_count = 0
                    best_epoch = epoch + 1
                    best_loss = avg_epoch_loss
                    torch.save(model.state_dict(), "./best_model.pth")
                else:
                    ex_no_improve_count += 1

                if avg_loss_prob < best_loss_prob:
                    best_loss_prob = avg_loss_prob
                    prob_no_improve_count = 0
                    best_epoch = epoch + 1
                    best_loss = avg_epoch_loss
                    torch.save(model.state_dict(), "./best_model.pth")
                else:
                    prob_no_improve_count += 1

            epoch_time_end = time.time()
            epoch_time = epoch_time_start - epoch_time_end

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, "
                  f"Loss_Ex: {avg_loss_ex:.4f}, Loss_Prob: {avg_loss_prob:.4f}, "
                  f"Normalized_Loss_Ex: {avg_normalized_loss_ex:.4f}, Normalized_Loss_Prob: {avg_normalized_loss_prob:.4f}, "
                  f"Weights: {weight_true}, Time: {epoch_time:.2f},no_improve_count: {ex_no_improve_count, prob_no_improve_count}")

            if (ex_no_improve_count >= patience and
                    prob_no_improve_count >= patience
            ):
                print(f"Early stopping triggered at epoch {epoch + 1} (All losses exceeded patience)")
                break

            if epoch == num_epochs - 1:
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), "./best_model.pth")

        if epoch == num_epochs - 1:
            best_epoch = num_epochs

        cv_metrics["CV_best_epoch"].append(best_epoch)

        model.load_state_dict(torch.load("./best_model.pth", ))
        # Validation step
        model.eval()
        print("best_loss",best_loss)
        val_loss = best_loss
        val_metrics = {
            "CV_r2_ex": [], "CV_r2_prob": [], "CV_r2_combined": [],
            "CV_mae_ex": [], "CV_mae_prob": [], "CV_mae_combined": [],
            "CV_rmse_ex": [], "CV_rmse_prob": [], "CV_rmse_combined": [],
            "CV_mse_ex": [], "CV_mse_prob": [], "CV_mse_combined": [],
            "CV_softdtw_ex": [], "CV_softdtw_prob": [], "CV_softdtw_combined": [],
            #"CV_fastdtw_ex": [], "CV_fastdtw_prob": [], "CV_fastdtw_combined": [],
            "CV_sid_ex": [], "CV_sid_prob": [], "CV_sid_combined": [],
            "CV_sis_ex": [], "CV_sis_prob": [], "CV_sis_combined": []
        }
        train_metrics = {
            "CV_train_r2_ex": [], "CV_train_r2_prob": [], "CV_train_r2_combined": [],
            "CV_train_mae_ex": [], "CV_train_mae_prob": [], "CV_train_mae_combined": [],
            "CV_train_rmse_ex": [], "CV_train_rmse_prob": [], "CV_train_rmse_combined": [],
            "CV_train_mse_ex":[], "CV_train_mse_prob":[], "CV_train_mse_combined": [],
            "CV_train_softdtw_ex": [], "CV_train_softdtw_prob": [], "CV_train_softdtw_combined": [],
            "CV_train_sid_ex": [], "CV_train_sid_prob": [], "CV_train_sid_combined": [],
            "CV_train_sis_ex": [], "CV_train_sis_prob": [], "CV_train_sis_combined": []
        }
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                batched_data = {k: batch[k] for k in ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input", "attn_edge_type"]}
                targets = batch["targets"]
                outputs = model(batched_data, targets=targets, target_type=target_type)

                for i in range(targets.size(0)):  # batch_size 만큼 루프
                    if target_type == "ex_prob":
                        y_true = targets[i].cpu().numpy()
                        y_pred = outputs[i].cpu().detach().numpy()

                        y_true_ex = y_true[:, 0]
                        y_pred_ex = y_pred[:, 0]
                        y_true_prob = y_true[:, 1]
                        y_pred_prob = y_pred[:, 1]

                        # R2, MAE, RMSE 계산
                        val_metrics["CV_r2_ex"].append(r2_score(y_true_ex, y_pred_ex))
                        val_metrics["CV_r2_prob"].append(r2_score(y_true_prob, y_pred_prob))
                        val_metrics["CV_r2_combined"].append(r2_score(y_true.flatten(), y_pred.flatten()))

                        val_metrics["CV_mae_ex"].append(mean_absolute_error(y_true_ex, y_pred_ex))
                        val_metrics["CV_mae_prob"].append(mean_absolute_error(y_true_prob, y_pred_prob))
                        val_metrics["CV_mae_combined"].append(mean_absolute_error(y_true.flatten(), y_pred.flatten()))

                        val_metrics["CV_rmse_ex"].append(mean_squared_error(y_true_ex, y_pred_ex, squared=False))
                        val_metrics["CV_rmse_prob"].append(mean_squared_error(y_true_prob, y_pred_prob, squared=False))
                        val_metrics["CV_rmse_combined"].append(
                            mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False))

                        val_metrics["CV_mse_ex"].append(mean_squared_error(y_true_ex, y_pred_ex, ))
                        val_metrics["CV_mse_prob"].append(mean_squared_error(y_true_prob, y_pred_prob, ))
                        val_metrics["CV_mse_combined"].append(mean_squared_error(y_true.flatten(), y_pred.flatten(), ))

                        # SoftDTW 및 FastDTW 계산
                        val_metrics["CV_softdtw_ex"].append(SoftDTWLoss(torch.tensor(y_pred_ex).unsqueeze(0).unsqueeze(-1).to(device),
                                                                     torch.tensor(y_true_ex).unsqueeze(0).unsqueeze(-1).to(
                                                                         device)).item())
                        val_metrics["CV_softdtw_prob"].append(SoftDTWLoss(torch.tensor(y_pred_prob).unsqueeze(0).unsqueeze(-1).to(device),
                                                                       torch.tensor(y_true_prob).unsqueeze(0).unsqueeze(-1).to(
                                                                           device)).item())
                        val_metrics["CV_softdtw_combined"].append(SoftDTWLoss(torch.tensor(y_pred).unsqueeze(0).to(device),
                                                                           torch.tensor(y_true).unsqueeze(0).to(
                                                                               device)).item())

                        #fastdtw_ex, _ = fastdtw(torch.tensor(y_pred_ex), torch.tensor(y_true_ex))
                        #fastdtw_prob, _ = fastdtw(torch.tensor(y_pred_prob), torch.tensor(y_true_prob))
                        #fastdtw_combined, _ = fastdtw(torch.tensor(y_pred.flatten()), torch.tensor(y_true.flatten()))
                        #val_metrics["CV_fastdtw_ex"].append(fastdtw_ex)
                        #val_metrics["CV_fastdtw_prob"].append(fastdtw_prob)
                        #val_metrics["CV_fastdtw_combined"].append(fastdtw_combined)

                        # SID 및 SIS 계산
                        sid_ex = sid_loss(torch.tensor(y_pred_ex).unsqueeze(0).to(device),
                                          torch.tensor(y_true_ex).unsqueeze(0).to(device),
                                          torch.ones_like(torch.tensor(y_pred_ex).unsqueeze(0), dtype=torch.bool).to(
                                              device),
                                          threshold=1e-4).mean().item()
                        sid_prob = sid_loss(torch.tensor(y_pred_prob).unsqueeze(0).to(device),
                                            torch.tensor(y_true_prob).unsqueeze(0).to(device),
                                            torch.ones_like(torch.tensor(y_pred_prob).unsqueeze(0), dtype=torch.bool).to(
                                                device),
                                            threshold=1e-4).mean().item()
                        sid_combined = sid_loss(torch.tensor(y_pred).unsqueeze(0).to(device),
                                                torch.tensor(y_true).unsqueeze(0).to(device),
                                                torch.ones_like(torch.tensor(y_pred).unsqueeze(0), dtype=torch.bool).to(
                                                    device),
                                                threshold=1e-4).mean().item()

                        # SID 결과값에 NaN이 있는지 확인 후 추가
                        if not math.isnan(sid_ex):
                            val_metrics["CV_sid_ex"].append(sid_ex)
                            val_metrics["CV_sis_ex"].append(1 / (1 + sid_ex))
                        else:
                            print(f"NaN detected in SID_ex at fold {fold + 1}, skipping this spectrum for ex.")

                        if not math.isnan(sid_prob):
                            val_metrics["CV_sid_prob"].append(sid_prob)
                            val_metrics["CV_sis_prob"].append(1 / (1 + sid_prob))
                        else:
                            print(f"NaN detected in SID_prob at fold {fold + 1}, skipping this spectrum for prob.")

                        if not math.isnan(sid_combined):
                            val_metrics["CV_sid_combined"].append(sid_combined)
                            val_metrics["CV_sis_combined"].append(1 / (1 + sid_combined))
                        else:
                            print(
                                f"NaN detected in SID_combined at fold {fold + 1}, skipping this spectrum for combined.")
                    else:
                        pass####

                for i in range(targets.size(0)):  # batch_size 만큼 루프
                    if target_type == "ex_prob":
                        y_true = targets[i].cpu().numpy()
                        y_pred = outputs[i].cpu().detach().numpy()

                        y_true_ex = y_true[:, 0]
                        y_pred_ex = y_pred[:, 0]
                        y_true_prob = y_true[:, 1]
                        y_pred_prob = y_pred[:, 1]

                        # R2, MAE, RMSE 계산
                        train_metrics["CV_train_r2_ex"].append(r2_score(y_true_ex, y_pred_ex))
                        train_metrics["CV_train_r2_prob"].append(r2_score(y_true_prob, y_pred_prob))
                        train_metrics["CV_train_r2_combined"].append(r2_score(y_true.flatten(), y_pred.flatten()))

                        train_metrics["CV_train_mae_ex"].append(mean_absolute_error(y_true_ex, y_pred_ex))
                        train_metrics["CV_train_mae_prob"].append(mean_absolute_error(y_true_prob, y_pred_prob))
                        train_metrics["CV_train_mae_combined"].append(mean_absolute_error(y_true.flatten(), y_pred.flatten()))

                        train_metrics["CV_train_rmse_ex"].append(mean_squared_error(y_true_ex, y_pred_ex, squared=False))
                        train_metrics["CV_train_rmse_prob"].append(mean_squared_error(y_true_prob, y_pred_prob, squared=False))
                        train_metrics["CV_train_rmse_combined"].append(mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False))

                        train_metrics["CV_train_mse_ex"].append(mean_squared_error(y_true_ex, y_pred_ex))
                        train_metrics["CV_train_mse_prob"].append(mean_squared_error(y_true_prob, y_pred_prob))
                        train_metrics["CV_train_mse_combined"].append(mean_squared_error(y_true.flatten(), y_pred.flatten()))

                        # SoftDTW 계산
                        train_metrics["CV_train_softdtw_ex"].append(
                            SoftDTWLoss(torch.tensor(y_pred_ex).unsqueeze(0).unsqueeze(-1).to(device),
                                        torch.tensor(y_true_ex).unsqueeze(0).unsqueeze(-1).to(device)).item())
                        train_metrics["CV_train_softdtw_prob"].append(
                            SoftDTWLoss(torch.tensor(y_pred_prob).unsqueeze(0).unsqueeze(-1).to(device),
                                        torch.tensor(y_true_prob).unsqueeze(0).unsqueeze(-1).to(device)).item())
                        train_metrics["CV_train_softdtw_combined"].append(
                            SoftDTWLoss(torch.tensor(y_pred).unsqueeze(0).to(device),
                                        torch.tensor(y_true).unsqueeze(0).to(device)).item())

                        # SID 및 SIS 계산
                        sid_ex = sid_loss(torch.tensor(y_pred_ex).unsqueeze(0).to(device),
                                          torch.tensor(y_true_ex).unsqueeze(0).to(device),
                                          torch.ones_like(torch.tensor(y_pred_ex).unsqueeze(0), dtype=torch.bool).to(device),
                                          threshold=1e-4).mean().item()
                        sid_prob = sid_loss(torch.tensor(y_pred_prob).unsqueeze(0).to(device),
                                            torch.tensor(y_true_prob).unsqueeze(0).to(device),
                                            torch.ones_like(torch.tensor(y_pred_prob).unsqueeze(0), dtype=torch.bool).to(device),
                                            threshold=1e-4).mean().item()
                        sid_combined = sid_loss(torch.tensor(y_pred).unsqueeze(0).to(device),
                                                torch.tensor(y_true).unsqueeze(0).to(device),
                                                torch.ones_like(torch.tensor(y_pred).unsqueeze(0), dtype=torch.bool).to(device),
                                                threshold=1e-4).mean().item()

                        if not math.isnan(sid_ex):
                            train_metrics["CV_train_sid_ex"].append(sid_ex)
                            train_metrics["CV_train_sis_ex"].append(1 / (1 + sid_ex))
                        if not math.isnan(sid_prob):
                            train_metrics["CV_train_sid_prob"].append(sid_prob)
                            train_metrics["CV_train_sis_prob"].append(1 / (1 + sid_prob))
                        if not math.isnan(sid_combined):
                            train_metrics["CV_train_sid_combined"].append(sid_combined)
                            train_metrics["CV_train_sis_combined"].append(1 / (1 + sid_combined))
                    else:
                        pass

        avg_val_loss = val_loss

        cv_metrics["val_loss"].append(avg_val_loss)

        # Store average metrics for this fold
        for key in val_metrics:
            cv_metrics[key].append(np.mean(val_metrics[key]))
        for key in train_metrics:
            cv_metrics[key].append(np.mean(train_metrics[key]))
        loss_history_all_folds.append(loss_history)

    # Compute average CV metrics
    avg_cv_metrics = {f"{key}_avg": np.mean(values) for key, values in cv_metrics.items()}
    avg_cv_metrics["CV_best_epoch_all"] = cv_metrics["CV_best_epoch"]

    CV_loss_ex_name = CV_loss_fn_name_gen(loss_function_ex, "ex")
    CV_loss_prob_name = CV_loss_fn_name_gen(loss_function_prob, "prob")
    CV_loss_combined_exloss = CV_loss_fn_name_gen(loss_function_ex, "combined")
    CV_loss_combined_probloss = CV_loss_fn_name_gen(loss_function_prob, "combined")

    print(avg_cv_metrics.keys())

    score_ex = np.mean(avg_cv_metrics[CV_loss_ex_name])
    score_prob = np.mean(avg_cv_metrics[CV_loss_prob_name])
    score_combined_based_exloss = np.mean(avg_cv_metrics[CV_loss_combined_exloss])
    score_combined_based_probloss = np.mean(avg_cv_metrics[CV_loss_combined_probloss])
    print("bayes_opt_scores", score_ex, score_prob, score_combined_based_exloss, score_combined_based_probloss)

    # Train on the entire dataset and calculate final metrics
    final_results, best_model_path = train_model_ex_porb(
        config=config,
        target_type=target_type,
        loss_function=loss_function,
        loss_function_ex=loss_function_ex,
        loss_function_prob=loss_function_prob,
        #weight_ex=weight_ex,
        num_epochs=num_epochs,
        batch_size=batch_size,
        n_pairs=n_pairs,
        dataset_path=dataset_path,
        DATASET=dataset,
        alpha = alpha
    )

    # Combine CV and final results
    final_results.update(avg_cv_metrics)
    fold = 0
    for fold_history in loss_history_all_folds:
        fold+=1
        for key, value in fold_history.items():
            final_results[f"cv_loss_history_{key}_{fold}"] = value

    fold = 0
    for fold_history in loss_history_all_folds:
        fold += 1
        for key, value in fold_history.items():
            final_results[f"cv_loss_history_{key}_{fold}"] = value
    final_results["weight_ex_history"] = cv_metrics["CV_weight_ex"]
    final_results["weight_prob_history"] = cv_metrics["CV_weight_prob"]

    #final_results["cv_loss_history"] = loss_history_all_folds

    if TEST_DATASET == None:
        test_dataset = SMILESDataset(csv_file=testset_path, attn_bias_w=1.0, target_type=target_type)
    else:
        test_dataset = TEST_DATASET

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, test_dataset, n_pairs=n_pairs, min_max=[train_min, train_max] ),
    )
    print("batch_size",batch_size)
    model.load_state_dict(torch.load(best_model_path,))
    model.eval()

    test_results = evaluate_on_test_set(model, test_loader, device, target_type)
    final_results.update(test_results)

    return final_results

def evaluate_on_test_set(model, test_loader, device, target_type):
    results = {
        "test_r2_ex": [], "test_r2_prob": [], "test_r2_combined": [],
        "test_mae_ex": [], "test_mae_prob": [], "test_mae_combined": [],
        "test_rmse_ex": [], "test_rmse_prob": [], "test_rmse_combined": [],
        "test_softdtw_ex": [], "test_softdtw_prob": [], "test_softdtw_combined": [],
        #"test_fastdtw_ex": [], "test_fastdtw_prob": [], "test_fastdtw_combined": [],
        "test_sid_ex": [], "test_sid_prob": [], "test_sid_combined": [],
        "test_sis_ex": [], "test_sis_prob": [], "test_sis_combined": []
    }
    print("evaluating on test set")
    #SoftDTWLoss = SoftDTW(use_cuda=True, gamma=1.0, bandwidth=None)
    SoftDTWLoss = SoftDTW(use_cuda=True, gamma=0.2, bandwidth=None, normalize=True)
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in
                            ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
                             "attn_edge_type"]}
            targets = batch["targets"]
            #print("targets",targets.shape)
            outputs = model(batched_data, targets=targets, target_type=target_type)
            #outputs = torch.clamp(outputs, min=1e-8)
            #print("targets", outputs.shape)

            for i in range(targets.size(0)):  # batch_size 만큼 루프
                y_true = targets[i].cpu().numpy()
                y_pred = outputs[i].cpu().detach().numpy()

                # ex와 prob 분리
                y_true_ex = y_true[:, 0]
                y_pred_ex = y_pred[:, 0]
                y_true_prob = y_true[:, 1]
                y_pred_prob = y_pred[:, 1]

                # R2, MAE, RMSE 계산
                results["test_r2_ex"].append(r2_score(y_true_ex, y_pred_ex))
                results["test_r2_prob"].append(r2_score(y_true_prob, y_pred_prob))
                results["test_r2_combined"].append(r2_score(y_true.flatten(), y_pred.flatten()))

                results["test_mae_ex"].append(mean_absolute_error(y_true_ex, y_pred_ex))
                results["test_mae_prob"].append(mean_absolute_error(y_true_prob, y_pred_prob))
                results["test_mae_combined"].append(mean_absolute_error(y_true.flatten(), y_pred.flatten()))

                results["test_rmse_ex"].append(mean_squared_error(y_true_ex, y_pred_ex, squared=False))
                results["test_rmse_prob"].append(mean_squared_error(y_true_prob, y_pred_prob, squared=False))
                results["test_rmse_combined"].append(
                    mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False))

                # SoftDTW 및 FastDTW 계산
                results["test_softdtw_ex"].append(
                    SoftDTWLoss(torch.tensor(y_pred_ex).unsqueeze(0).unsqueeze(-1).to(device),
                                torch.tensor(y_true_ex).unsqueeze(0).unsqueeze(-1).to(
                                    device)).item())
                results["test_softdtw_prob"].append(
                    SoftDTWLoss(torch.tensor(y_pred_prob).unsqueeze(0).unsqueeze(-1).to(device),
                                torch.tensor(y_true_prob).unsqueeze(0).unsqueeze(-1).to(
                                    device)).item())
                results["test_softdtw_combined"].append(SoftDTWLoss(torch.tensor(y_pred).unsqueeze(0).to(device),
                                                                      torch.tensor(y_true).unsqueeze(0).to(
                                                                          device)).item())

                #fastdtw_ex, _ = fastdtw(torch.tensor(y_pred_ex), torch.tensor(y_true_ex))
                #fastdtw_prob, _ = fastdtw(torch.tensor(y_pred_prob), torch.tensor(y_true_prob))
                #fastdtw_combined, _ = fastdtw(torch.tensor(y_pred.flatten()), torch.tensor(y_true.flatten()))
                #results["test_fastdtw_ex"].append(fastdtw_ex)
                #results["test_fastdtw_prob"].append(fastdtw_prob)
                #results["test_fastdtw_combined"].append(fastdtw_combined)

                # SID 및 SIS 계산
                sid_ex = sid_loss(torch.tensor(y_pred_ex).unsqueeze(0).to(device),
                                  torch.tensor(y_true_ex).unsqueeze(0).to(device),
                                  torch.ones_like(torch.tensor(y_pred_ex).unsqueeze(0), dtype=torch.bool).to(device),
                                  threshold=1e-4).mean().item()
                sid_prob = sid_loss(torch.tensor(y_pred_prob).unsqueeze(0).to(device),
                                    torch.tensor(y_true_prob).unsqueeze(0).to(device),
                                    torch.ones_like(torch.tensor(y_pred_prob).unsqueeze(0), dtype=torch.bool).to(device),
                                    threshold=1e-4).mean().item()
                sid_combined = sid_loss(torch.tensor(y_pred).unsqueeze(0).to(device),
                                        torch.tensor(y_true).unsqueeze(0).to(device),
                                        torch.ones_like(torch.tensor(y_pred).unsqueeze(0), dtype=torch.bool).to(device),
                                        threshold=1e-4).mean().item()

                # SID 및 SIS NaN 체크 및 저장
                if not math.isnan(sid_ex):
                    results["test_sid_ex"].append(sid_ex)
                    results["test_sis_ex"].append(1 / (1 + sid_ex))
                else:
                    print("NaN detected in SID_ex on Test set, skipping this spectrum for ex.")

                if not math.isnan(sid_prob):
                    results["test_sid_prob"].append(sid_prob)
                    results["test_sis_prob"].append(1 / (1 + sid_prob))
                else:
                    print("NaN detected in SID_prob, skipping this spectrum for prob.")

                if not math.isnan(sid_combined):
                    results["test_sid_combined"].append(sid_combined)
                    results["test_sis_combined"].append(1 / (1 + sid_combined))
                else:
                    print("NaN detected in SID_combined, skipping this spectrum for combined.")

    # 평균 계산
    avg_results = {f"test_{key}_avg": np.mean(values) for key, values in results.items()}
    return avg_results


if __name__ == "__main__":
    config = {
        "num_atoms": 100,  # 분자의 최대 원자 수 (그래프의 노드 개수) 100
        "num_in_degree": 10,  # 그래프 노드의 최대 in-degree
        "num_out_degree": 10,  # 그래프 노드의 최대 out-degree
        "num_edges": 100,  # 그래프의 최대 엣지 개수 50
        "num_spatial": 50,  # 공간적 위치 인코딩을 위한 최대 값 default 100
        "num_edge_dis": 10,  # 엣지 거리 인코딩을 위한 최대 값
        "edge_type": "multi_hop",  # 엣지 타입 ("multi_hop" 또는 다른 값 가능)
        "multi_hop_max_dist": 6,  # Multi-hop 엣지의 최대 거리
        "num_encoder_layers": 3,  # Graphormer 모델에서 사용할 인코더 레이어 개수
        "embedding_dim": 128,  # 임베딩 차원 크기 (노드, 엣지 등)
        "ffn_embedding_dim": 256,  # Feedforward Network의 임베딩 크기
        "num_attention_heads": 8,  # Multi-head Attention에서 헤드 개수
        "dropout": 0.1,  # 드롭아웃 비율
        "attention_dropout": 0.1,  # Attention 레이어의 드롭아웃 비율
        "activation_dropout": 0.1,  # 활성화 함수 이후 드롭아웃 비율
        "activation_fn": "relu",  # 활성화 함수 ("gelu", "relu" 등)
        "pre_layernorm": False,  # LayerNorm을 Pre-Normalization으로 사용할지 여부
        "q_noise": 0.0,  # Quantization noise (훈련 중 노이즈 추가를 위한 매개변수)
        "qn_block_size": 8,  # Quantization block 크기
        "output_size": 100,  # 모델 출력 크기
    }
    #     final_loss = cross_validate_model(config=config, target_type="ex_prob",dataset_path="../../data/train_50.csv", num_epochs=10, n_pairs=50, n_splits=5,loss_function_ex="MAE",loss_function_prob="MAE")
    results_list = []
    loss_fn_list = ["SoftDTW"]
    for loss_fn in loss_fn_list:
        cv_result = cross_validate_model(config=config, target_type="ex_prob", dataset_path="../../data/train_1000.csv",
                                          num_epochs=5, n_pairs=50, n_splits=5, batch_size=32,loss_function_ex=loss_fn,
                                          loss_function_prob=loss_fn)
        cv_result["loss_function"] = loss_fn
        results_list.append(cv_result)
        df = pd.DataFrame(results_list)
        df.to_csv("CV_losses_result_intermedeate.csv", index=False)
        print(cv_result)
        print(f"{loss_fn}: finished")

    # 리스트 → DataFrame으로 변환
    df = pd.DataFrame(results_list)

    # CSV 저장
    df.to_csv("CV_losses_result.csv", index=False)
    print("Saved to .csv")
    #print(f"Final Average Loss: {final_loss:.4f}")
    # train_50 set으로 할시 nan 값이 sid에서 나오눈 문제가 있음