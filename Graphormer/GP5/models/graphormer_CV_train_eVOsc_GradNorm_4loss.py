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
from sklearn.model_selection import KFold
from Graphormer.GP5.models.graphormer_train_eVOsc_GradNorm_4loss import train_model_ex_porb
from chemprop.train.loss_functions import sid_loss
from tslearn.metrics import SoftDTWLossPyTorch
from Graphormer.GP5.Custom_Loss.GradNorm import GradNorm
import math
from Graphormer.GP5.Custom_Loss.soft_dtw_cuda import SoftDTW

def loss_normalizer(first_loss, loss):
    # 손실 값이 tensor인 경우 정규화 값도 tensor로 유지
    if first_loss is not None and isinstance(loss, torch.Tensor):
        normalized_loss = loss / first_loss
    else:
        normalized_loss = loss
    return normalized_loss

def cross_validate_model(
    config,
    target_type="default",
    loss_function="MSE",
    loss_function_ex_1="SoftDTW",
    loss_function_ex_2="SID",
    loss_function_prob_1="SoftDTW",
    loss_function_prob_2="SID",
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
        "CV_r2_ex": [], "CV_r2_prob": [], "CV_r2_combined": [],
        "CV_mae_ex": [], "CV_mae_prob": [], "CV_mae_combined": [],
        "CV_rmse_ex": [], "CV_rmse_prob": [], "CV_rmse_combined": [],
        "CV_softdtw_ex": [], "CV_softdtw_prob": [], "CV_softdtw_combined": [],
        #"CV_fastdtw_ex": [], "CV_fastdtw_prob": [], "CV_fastdtw_combined": [],
        "CV_sid_ex": [], "CV_sid_prob": [], "CV_sid_combined": [],
        "CV_sis_ex": [], "CV_sis_prob": [], "CV_sis_combined": [],
        "val_loss": [], "CV_best_epoch":[], "CV_weight_ex_1": [], "CV_weight_ex_2": [],"CV_weight_prob_1": [], "CV_weight_prob_2": []
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
            train_min = all_targets.min(dim=0)[0].min(dim=0)[0]
            train_min = train_min[0].detach().cpu().numpy()
            train_max = all_targets.max(dim=0)[0].max(dim=0)[0]
            train_max = train_max[0].detach().cpu().numpy()
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
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        loss_modifier = GradNorm(num_losses=4, alpha=alpha)

        patience = patience  # Early stopping patience 설정
        no_improve_count_ex_1 = 0
        no_improve_count_ex_2 = 0
        no_improve_count_prob_1 = 0
        no_improve_count_prob_2 = 0
        best_loss_ex_1 = float('inf')
        best_loss_ex_2 = float('inf')
        best_loss_prob_1 = float('inf')
        best_loss_prob_2 = float('inf')
        best_epoch = 0
        best_loss = 0
        weight_true = torch.tensor([0.5, 0.5, 0.5, 0.5], device=device)
        first_loss_ex_1 = None
        first_loss_ex_2 = None
        first_loss_prob_1 = None
        first_loss_prob_2 = None

        loss_history = {"ex_loss_1": [], "ex_loss_2": [], "prob_loss_1": [], "prob_loss_2": [], "total_loss": [],
                        "normalized_ex_loss_1": [],"normalized_ex_loss_2": [], "normalized_prob_loss_1": [], "normalized_prob_loss_2": []}

        for epoch in range(num_epochs):
            #print("epochs left",epoch,"/",num_epochs)
            model.train()
            epoch_loss = 0.0
            loss_ex_list_1 = []
            loss_ex_list_2 = []
            loss_prob_list_1 = []
            loss_prob_list_2 = []
            weight_list = []
            normalized_loss_ex_list_1 = []
            normalized_loss_ex_list_2 = []
            normalized_loss_prob_list_1 = []
            normalized_loss_prob_list_2 = []
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
                    if loss_function_ex_1 == "SID":
                        threshold = 1e-4
                        mask_ex_1 = torch.ones_like(outputs_ex, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                        loss_ex_1 = torch.stack([
                            criterion_ex_1(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0), mask_ex_1[i], threshold)
                            for i in range(outputs_ex.size(0))
                        ]).mean()

                    else:
                        loss_ex_1 = torch.stack([
                            criterion_ex_1(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0))
                            for i in range(outputs_ex.size(0))
                        ]).mean()

                    if loss_function_ex_2 == "SID":
                        threshold = 1e-4
                        mask_ex_2 = torch.ones_like(outputs_ex, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                        loss_ex_2 = torch.stack([
                            criterion_ex_2(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0), mask_ex_2[i], threshold)
                            for i in range(outputs_ex.size(0))
                        ]).mean()

                    else:
                        loss_ex_2 = torch.stack([
                            criterion_ex_2(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0))
                            for i in range(outputs_ex.size(0))
                        ]).mean()

                    # outputs_prob = torch.sigmoid(outputs[:, :, 1:2])
                    # targets_prob = torch.sigmoid(targets[:, :, 1:2])
                    outputs_prob = outputs[:, :, 1:2] + 1e-8
                    targets_prob = targets[:, :, 1:2] + 1e-8
                    #outputs_prob = torch.clamp(outputs[:, :, 1:2], min=1e-8)
                    #targets_prob = torch.clamp(targets[:, :, 1:2], min=1e-8)

                    # SID Loss를 사용할 경우 마스크 생성
                    if loss_function_prob_1 == "SID":
                        threshold = 1e-4
                        mask_prob_1 = torch.ones_like(outputs_prob, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                        loss_prob_1 = torch.stack([
                            criterion_prob_1(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0), mask_prob_1[i],
                                           threshold)
                            for i in range(outputs_prob.size(0))
                        ]).mean()
                    else:
                        loss_prob_1 = torch.stack([
                            criterion_prob_1(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0))
                            for i in range(outputs_prob.size(0))
                        ]).mean()

                    if loss_function_prob_2 == "SID":
                        threshold = 1e-4
                        mask_prob_2 = torch.ones_like(outputs_prob, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                        loss_prob_2 = torch.stack([
                            criterion_prob_2(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0), mask_prob_2[i],
                                           threshold)
                            for i in range(outputs_prob.size(0))
                        ]).mean()
                    else:
                        loss_prob_2 = torch.stack([
                            criterion_prob_2(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0))
                            for i in range(outputs_prob.size(0))
                        ]).mean()

                    # ✅ 첫 배치에서 손실 값 저장
                    if first_loss_ex_1 is None:
                        first_loss_ex_1 = loss_ex_1.item()
                        first_loss_ex_2 = loss_ex_2.item()
                        first_loss_prob_1 = loss_prob_1.item()
                        first_loss_prob_2 = loss_prob_2.item()


                    # 손실 값이 tensor인 경우 정규화 값도 tensor로 유지
                    if first_loss_ex_1 is not None and isinstance(loss_ex_1, torch.Tensor):
                        normalized_loss_ex_1 = loss_ex_1 / first_loss_ex_1
                    else:
                        normalized_loss_ex_1 = loss_ex_1

                    if first_loss_ex_2 is not None and isinstance(loss_ex_2, torch.Tensor):
                        normalized_loss_ex_2 = loss_ex_2 / first_loss_ex_2
                    else:
                        normalized_loss_ex_2 = loss_ex_2

                    if first_loss_prob_1 is not None and isinstance(loss_prob_1, torch.Tensor):
                        normalized_loss_prob_1 = loss_prob_1 / first_loss_prob_1
                    else:
                        normalized_loss_prob_1 = loss_prob_1

                    if first_loss_prob_2 is not None and isinstance(loss_prob_2, torch.Tensor):
                        normalized_loss_prob_2 = loss_prob_2 / first_loss_prob_2
                    else:
                        normalized_loss_prob_2 = loss_prob_2

                    # ✅ 매 배치마다 GradNorm 적용
                    #weight = loss_modifier.compute_weights([loss_ex, loss_prob], model)
                    weight = loss_modifier.compute_weights([loss_ex_1, loss_ex_2, loss_prob_1, loss_prob_2], model)
                    weight_list.append(weight.detach().cpu().numpy()) ###### 바꾼부분

                    # ✅ 손실 값 계산
                    # loss = weight_true[0] * loss_ex + weight_true[1] * loss_prob
                    loss = weight_true[0] * normalized_loss_ex_1 + weight_true[1] * normalized_loss_ex_2 + weight_true[2] * normalized_loss_prob_1 + weight_true[3] * normalized_loss_prob_2
                else:
                    raise ValueError("Invalid target type")

                optimizer.zero_grad()
                loss.backward(retain_graph=False)

                optimizer.step()

                epoch_loss += loss.item()
                loss_ex_list_1.append(loss_ex_1.item())
                loss_ex_list_2.append(loss_ex_2.item())
                loss_prob_list_1.append(loss_prob_1.item())
                loss_prob_list_2.append(loss_prob_2.item())
                normalized_loss_ex_list_1.append(normalized_loss_ex_1.item())
                normalized_loss_ex_list_2.append(normalized_loss_ex_2.item())
                normalized_loss_prob_list_1.append(normalized_loss_prob_1.item())
                normalized_loss_prob_list_2.append(normalized_loss_prob_2.item())

            weight_true = torch.tensor(np.mean(weight_list, axis=0))

            avg_loss_ex_1 = np.mean(loss_ex_list_1)
            avg_loss_ex_2 = np.mean(loss_ex_list_2)
            avg_loss_prob_1 = np.mean(loss_prob_list_1)
            avg_loss_prob_2 = np.mean(loss_prob_list_2)
            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_normalized_loss_ex_1 = np.mean(normalized_loss_ex_list_1)
            avg_normalized_loss_ex_2 = np.mean(normalized_loss_ex_list_2)
            avg_normalized_loss_prob_1 = np.mean(normalized_loss_prob_list_1)
            avg_normalized_loss_prob_2 = np.mean(normalized_loss_prob_list_2)

            avg_weight = np.mean(weight_list, axis=0)
            cv_metrics["CV_weight_ex_1"].append(avg_weight[0])
            cv_metrics["CV_weight_ex_2"].append(avg_weight[1])
            cv_metrics["CV_weight_prob_1"].append(avg_weight[2])
            cv_metrics["CV_weight_prob_2"].append(avg_weight[3])

            loss_history["ex_loss_1"].append(avg_loss_ex_1)
            loss_history["ex_loss_2"].append(avg_loss_ex_2)
            loss_history["prob_loss_1"].append(avg_loss_prob_1)
            loss_history["prob_loss_2"].append(avg_loss_prob_2)

            loss_history["total_loss"].append(avg_epoch_loss)

            loss_history["normalized_ex_loss_1"].append(
                np.mean([x for x in normalized_loss_ex_list_1])
            )
            loss_history["normalized_ex_loss_2"].append(
                np.mean([x for x in normalized_loss_ex_list_2])
            )
            loss_history["normalized_prob_loss_1"].append(
                np.mean([x for x in normalized_loss_prob_list_1])
            )
            loss_history["normalized_prob_loss_2"].append(
                np.mean([x for x in normalized_loss_prob_list_2])
            )

            # ✅ Early Stopping 개별 손실 기준 적용
            if avg_loss_ex_1 < best_loss_ex_1:
                best_loss_ex_1 = avg_loss_ex_1
                no_improve_count_ex_1 = 0
                best_epoch = epoch + 1
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), "./best_model.pth")
            else:
                no_improve_count_ex_1 += 1

            if avg_loss_ex_2 < best_loss_ex_2:
                best_loss_ex_2 = avg_loss_ex_2
                no_improve_count_ex_2 = 0
                best_epoch = epoch + 1
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), "./best_model.pth")
            else:
                no_improve_count_ex_2 += 1

            if avg_loss_prob_1 < best_loss_prob_1:
                best_loss_prob_1 = avg_loss_prob_1
                no_improve_count_prob_1 = 0
                best_epoch = epoch + 1
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), "./best_model.pth")
            else:
                no_improve_count_prob_1 += 1

            if avg_loss_prob_2 < best_loss_prob_2:
                best_loss_prob_2 = avg_loss_prob_2
                no_improve_count_prob_2 = 0
                best_epoch = epoch + 1
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), "./best_model.pth")
            else:
                no_improve_count_prob_2 += 1

            epoch_time_end = time.time()
            epoch_time = epoch_time_start - epoch_time_end

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, " # 
                  f"Loss_Ex: {avg_loss_ex_1:.4f},{avg_loss_ex_2:.4f}, Loss_Prob: {avg_loss_prob_1:.4f},{avg_loss_prob_2:.4f}, "
                  f"Normalized_Loss_Ex: {avg_normalized_loss_ex_1:.4f},{avg_normalized_loss_ex_2:.4f}, "
                  f"Normalized_Loss_Prob: {avg_normalized_loss_prob_1:.4f},{avg_normalized_loss_prob_2:.4f}"
                  f" Weights: {weight_true}, Time: {epoch_time:.2f},"
                  f"no_improve_count: {no_improve_count_ex_1, no_improve_count_ex_2, no_improve_count_prob_1, no_improve_count_prob_2}, val_loss: {avg_epoch_loss:.4f}")

            if epoch == num_epochs - 1:
                torch.save(model.state_dict(), "./best_model.pth")
                best_loss = avg_epoch_loss

            # ✅ 두 손실 모두 patience 기준 충족 시 종료
            if (no_improve_count_ex_1 >= patience and
                    no_improve_count_ex_2 >= patience and
                    no_improve_count_prob_1 >= patience and
                    no_improve_count_prob_2 >= patience):
                print(f"Early stopping triggered at epoch {epoch + 1} (All losses exceeded patience)")
                #torch.save(model.state_dict(), "./best_model.pth")
                break

        if epoch == num_epochs - 1:
            best_epoch = num_epochs
        print("best_loss",best_loss, best_epoch)
        cv_metrics["CV_best_epoch"].append(best_epoch)

        # Validation step
        # model.load_state_dict(torch.load("./best_model.pth", ))
        model.eval()
        val_loss = best_loss
        val_metrics = {
            "CV_r2_ex": [], "CV_r2_prob": [], "CV_r2_combined": [],
            "CV_mae_ex": [], "CV_mae_prob": [], "CV_mae_combined": [],
            "CV_rmse_ex": [], "CV_rmse_prob": [], "CV_rmse_combined": [],
            "CV_softdtw_ex": [], "CV_softdtw_prob": [], "CV_softdtw_combined": [],
            #"CV_fastdtw_ex": [], "CV_fastdtw_prob": [], "CV_fastdtw_combined": [],
            "CV_sid_ex": [], "CV_sid_prob": [], "CV_sid_combined": [],
            "CV_sis_ex": [], "CV_sis_prob": [], "CV_sis_combined": []
        }
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                batched_data = {k: batch[k] for k in ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input", "attn_edge_type"]}
                targets = batch["targets"]
                #print("batch_size",batch_size)
                #print("targets first",targets.shape)
                outputs = model(batched_data, targets=targets, target_type=target_type)
                #outputs = torch.clamp(outputs,min=1e-8)
                #print("output shape",outputs.shape)
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



                    #Final loss 계산
                    #val_loss += (weight_ex * loss_ex + (1 - weight_ex) * loss_prob).item()
                    else:
                        pass####


        avg_val_loss = val_loss / len(val_loader)
        cv_metrics["val_loss"].append(avg_val_loss)

        # Store average metrics for this fold
        for key in val_metrics:
            cv_metrics[key].append(np.mean(val_metrics[key]))
        loss_history_all_folds.append(loss_history)

    # Compute average CV metrics
    avg_cv_metrics = {f"{key}_avg": np.mean(values) for key, values in cv_metrics.items()}
    avg_cv_metrics["CV_best_epoch_all"] = cv_metrics["CV_best_epoch"]
    print("val_loss_avg",avg_cv_metrics["val_loss_avg"])
    # Train on the entire dataset and calculate final metrics
    final_results, best_model_path = train_model_ex_porb(
        config=config,
        target_type=target_type,
        loss_function=loss_function,
        loss_function_ex_1=loss_function_ex_1,
        loss_function_ex_2=loss_function_ex_2,
        loss_function_prob_1=loss_function_prob_1,
        loss_function_prob_2=loss_function_prob_2,
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
    final_results["weight_ex_history_1"] = cv_metrics["CV_weight_ex_1"]
    final_results["weight_ex_history_2"] = cv_metrics["CV_weight_ex_2"]
    final_results["weight_prob_history_1"] = cv_metrics["CV_weight_prob_2"]
    final_results["weight_prob_history_2"] = cv_metrics["CV_weight_prob_1"]

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
        "num_atoms": 57,  # 분자의 최대 원자 수 (그래프의 노드 개수) 100
        "num_in_degree": 10,  # 그래프 노드의 최대 in-degree
        "num_out_degree": 10,  # 그래프 노드의 최대 out-degree
        "num_edges": 62,  # 그래프의 최대 엣지 개수 50
        "num_spatial": 27,  # 공간적 위치 인코딩을 위한 최대 값 default 100
        "num_edge_dis": 10,  # 엣지 거리 인코딩을 위한 최대 값
        "edge_type": "multi_hop",  # 엣지 타입 ("multi_hop" 또는 다른 값 가능)
        "multi_hop_max_dist": 5,  # Multi-hop 엣지의 최대 거리
        "num_encoder_layers": 6,  # Graphormer 모델에서 사용할 인코더 레이어 개수
        "embedding_dim": 128,  # 임베딩 차원 크기 (노드, 엣지 등)
        "ffn_embedding_dim": 256,  # Feedforward Network의 임베딩 크기
        "num_attention_heads": 8,  # Multi-head Attention에서 헤드 개수
        "dropout": 0.1,  # 드롭아웃 비율
        "attention_dropout": 0.1,  # Attention 레이어의 드롭아웃 비율
        "activation_dropout": 0.1,  # 활성화 함수 이후 드롭아웃 비율
        "activation_fn": "gelu",  # 활성화 함수 ("gelu", "relu" 등)
        "pre_layernorm": False,  # LayerNorm을 Pre-Normalization으로 사용할지 여부
        "q_noise": 0.0,  # Quantization noise (훈련 중 노이즈 추가를 위한 매개변수)
        "qn_block_size": 8,  # Quantization block 크기
        "output_size": 100,  # 모델 출력 크기
    }
    print("CV config", config)
    final_loss = cross_validate_model(config=config, target_type="ex_prob",dataset_path="../../data/train_50.csv", num_epochs=10, n_pairs=50, n_splits=5,
                                      loss_function_ex_1="SoftDTW",loss_function_ex_2="SoftDTW",loss_function_prob_1="SoftDTW",loss_function_prob_2="MAE", patience=5)
    print(final_loss)
    #print(f"Final Average Loss: {final_loss:.4f}")
    # train_50 set으로 할시 nan 값이 sid에서 나오눈 문제가 있음