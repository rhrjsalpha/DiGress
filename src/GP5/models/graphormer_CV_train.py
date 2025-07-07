import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset, collate_fn
from Graphormer.GP5.models.graphormer import GraphormerModel
import os
from Graphormer.GP5.Custom_Loss.custom_loss import fastdtw_loss
from Graphormer.GP5.Custom_Loss.sdtw_cuda_loss import SoftDTW
from Graphormer.GP5.Custom_Loss.SID_loss import SIDLoss
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Graphormer.GP5.Custom_Loss.fast_dtw import fastdtw
import time
from sklearn.model_selection import KFold
from Graphormer.GP5.models.graphormer_train_ex_prob import train_model_ex_porb
from chemprop.train.loss_functions import sid_loss
from tslearn.metrics import SoftDTWLossPyTorch

def cross_validate_model(
    config,
    target_type="default",
    loss_function="MSE",
    loss_function_ex="SoftDTW",
    loss_function_prob="SoftDTW",
    weight_ex=0.5,
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
        "val_loss": [], "CV_best_epoch":[]
    }
    loss_history_all_folds = []  # Store loss history for all folds

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Processing fold {fold + 1}/{n_splits}")

        # Split dataset into train and validation sets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
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
        SoftDTWLoss = SoftDTWLossPyTorch(gamma=0.2, normalize=True)

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

        patience = patience  # Early stopping patience 설정
        no_improve_count = 0  # 개선되지 않은 epoch 수 추적
        best_loss = float('inf')
        first_loss_ex = None
        first_loss_prob = None

        # Train the model on the training set
        loss_history = {"ex_loss": [], "prob_loss": [], "total_loss": [], "normalized_ex_loss": [],
                        "normalized_prob_loss": []}
        for epoch in range(num_epochs):
            #print("epochs left",epoch,"/",num_epochs)
            epoch_start_time = time.time()
            model.train()
            epoch_loss = 0.0
            loss_ex_list = []
            loss_prob_list = []
            normalized_loss_ex_list = []
            normalized_loss_prob_list = []

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

                    else:
                        loss_ex = torch.stack([
                            criterion_ex(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0))
                            for i in range(outputs_ex.size(0))
                        ]).mean()

                    #outputs_prob = outputs[:, :, 1:2] + 1e-8
                    #targets_prob = targets[:, :, 1:2] + 1e-8
                    outputs_prob = torch.clamp(outputs[:, :, 1:2], min=1e-8)
                    targets_prob = torch.clamp(targets[:, :, 1:2], min=1e-8)

                    # SID Loss를 사용할 경우 마스크 생성
                    if loss_function_prob == "SID":
                        threshold = 1e-4
                        mask_prob = torch.ones_like(outputs_prob, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                        loss_prob = torch.stack([
                            criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0), mask_prob[i],
                                           threshold)
                            for i in range(outputs_prob.size(0))
                        ]).mean()
                    else:
                        loss_prob = torch.stack([
                            criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0))
                            for i in range(outputs_prob.size(0))
                        ]).mean()

                    # Final loss 계산
                    loss = weight_ex * loss_ex + (1-weight_ex) * loss_prob
                    #print(loss_ex, loss_prob)

                elif target_type == "default":
                    outputs_ex = outputs[:, :outputs.size(1) // 2]
                    outputs_prob = outputs[:, outputs.size(1) // 2:]
                    targets_ex = targets[:, :targets.size(1) // 2]
                    targets_prob = targets[:, targets.size(1) // 2:]

                    loss_ex = torch.stack([criterion_ex(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0)) for i in
                                           range(outputs_ex.size(0))]).mean()
                    loss_prob = torch.stack(
                        [criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0)) for i in
                         range(outputs_prob.size(0))]).mean()

                    loss = weight_ex * loss_ex + (1-weight_ex) * loss_prob
                    #print(loss_ex, loss_prob)
                else:
                    raise ValueError("Invalid target type")

                if first_loss_ex is None:
                    first_loss_ex = loss_ex.item()
                if first_loss_prob is None:
                    first_loss_prob = loss_prob.item()

                normalized_loss_ex = loss_ex.item() / first_loss_ex if first_loss_ex != 0 else 1.0
                normalized_loss_prob = loss_prob.item() / first_loss_prob if first_loss_prob != 0 else 1.0

                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                optimizer.step()

                epoch_loss += loss.item()
                loss_ex_list.append(loss_ex.item())
                loss_prob_list.append(loss_prob.item())
                normalized_loss_ex_list.append(normalized_loss_ex)
                normalized_loss_prob_list.append(normalized_loss_prob)

            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_loss_ex = np.mean(loss_ex_list)
            avg_loss_prob = np.mean(loss_prob_list)
            avg_normalized_loss_ex = np.mean(normalized_loss_ex_list)
            avg_normalized_loss_prob = np.mean(normalized_loss_prob_list)

            loss_history["ex_loss"].append(avg_loss_ex)
            loss_history["prob_loss"].append(avg_loss_prob)
            loss_history["total_loss"].append(avg_epoch_loss)
            loss_history["normalized_ex_loss"].append(avg_normalized_loss_ex)
            loss_history["normalized_prob_loss"].append(avg_normalized_loss_prob)

            # Best model 저장 & Early stopping 체크
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_epoch = epoch
                no_improve_count = 0  # 개선되었으므로 리셋
            else:
                no_improve_count += 1  # 개선되지 않았으므로 증가

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, "
                  f"Loss_Ex: {avg_loss_ex:.4f}, Loss_Prob: {avg_loss_prob:.4f}, "
                  f"Normalized_Loss_Ex: {avg_normalized_loss_ex:.4f}, Normalized_Loss_Prob: {avg_normalized_loss_prob:.4f}, "
                  f"Time: {epoch_time:.2f},no_improve_count: {no_improve_count}")
            # Early stopping 조건
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        cv_metrics["CV_best_epoch"].append(best_epoch)

        # Validation step
        model.eval()
        val_loss = 0.0
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

                outputs = model(batched_data, targets=targets, target_type=target_type)
                outputs = torch.clamp(outputs, min=1e-8)

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

                        val_metrics["CV_sid_ex"].append(sid_ex)
                        val_metrics["CV_sid_prob"].append(sid_prob)
                        val_metrics["CV_sid_combined"].append(sid_combined)

                        val_metrics["CV_sis_ex"].append(1 / (1 + sid_ex))
                        val_metrics["CV_sis_prob"].append(1 / (1 + sid_prob))
                        val_metrics["CV_sis_combined"].append(1 / (1 + sid_combined))

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
    # Train on the entire dataset and calculate final metrics
    final_results, best_model_path = train_model_ex_porb(
        config=config,
        target_type=target_type,
        loss_function=loss_function,
        loss_function_ex=loss_function_ex,
        loss_function_prob=loss_function_prob,
        weight_ex=weight_ex,
        num_epochs=num_epochs,
        batch_size=batch_size,
        n_pairs=n_pairs,
        dataset_path=dataset_path,
        DATASET=dataset
    )

    # Combine CV and final results
    final_results.update(avg_cv_metrics)
    fold = 0
    for fold_history in loss_history_all_folds:
        fold += 1
        for key, value in fold_history.items():
            final_results[f"cv_loss_history_{key}_{fold}"] = value

    if TEST_DATASET == None:
        test_dataset = SMILESDataset(csv_file=testset_path, attn_bias_w=1.0, target_type=target_type)
    else:
        test_dataset = TEST_DATASET

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, test_dataset, n_pairs=n_pairs),
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
    SoftDTWLoss = SoftDTWLossPyTorch(gamma=0.2, normalize=True)
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in
                            ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
                             "attn_edge_type"]}
            targets = batch["targets"]
            #print("targets",targets.shape)
            outputs = model(batched_data, targets=targets, target_type=target_type)
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

                results["test_sid_ex"].append(sid_ex)
                results["test_sid_prob"].append(sid_prob)
                results["test_sid_combined"].append(sid_combined)

                results["test_sis_ex"].append(1 / (1 + sid_ex))
                results["test_sis_prob"].append(1 / (1 + sid_prob))
                results["test_sis_combined"].append(1 / (1 + sid_combined))

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
    final_loss = cross_validate_model(config=config, target_type="ex_prob",dataset_path="../../data/train_50.csv", num_epochs=5, n_pairs=50,n_splits=5,loss_function_ex="SID",loss_function_prob="SID")
    print(final_loss)
    #print(f"Final Average Loss: {final_loss:.4f}")


