from typing import final

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from chemprop.train.loss_functions import sid_loss

from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset, collate_fn
from Graphormer.GP5.models.graphormer import GraphormerModel
from Graphormer.GP5.Custom_Loss.sdtw_cuda_loss import SoftDTW
from Graphormer.GP5.models.graphormer_train_default import train_model
from Graphormer.GP5.Custom_Loss.GradNorm import GradNorm
from tslearn.metrics import SoftDTWLossPyTorch

def cross_validate_model(
    config,
    target_type="default",
    loss_function="MSE",
    weight_ex=0.5,
    num_epochs=10,
    batch_size=50,
    n_pairs=1,
    learning_rate=0.001,
    dataset_path="../../data/data_example.csv",
    testset_path="../../data/data_example.csv",
    n_splits=3,  # Number of folds for Cross Validation
    patience = 20,
    DATASET = None,
    TEST_DATASET = None
):
    """
    Perform cross-validation and final training on the entire dataset.

    Args:
        config (dict): Model configuration.
        loss_function (str): Loss function to be used.
        weight_ex (float): Weight for loss.
        num_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        n_pairs (int): Number of ex_prob pairs.
        learning_rate (float): Learning rate.
        dataset_path (str): Path to the training dataset.
        testset_path (str): Path to the test dataset.
        n_splits (int): Number of cross-validation splits.

    Returns:
        dict: Results containing CV metrics and test set metrics.
    """
    dataset_time_before = time.time()
    if DATASET is None:
        dataset = SMILESDataset(csv_file=dataset_path, attn_bias_w=1.0, target_type=target_type)
    else:
        dataset = DATASET
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    dataset_time_after = time.time()
    #print("dataset_loading_time",dataset_time_after - dataset_time_before)

    # Store results for cross-validation
    cv_metrics = {
        "CV_r2": [], "CV_mae": [], "CV_rmse": [], "CV_softdtw": [],"CV_sid": [], "CV_sis": [], "val_loss": [], "CV_best_epoch":[]
    }

    cv_train_start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Processing fold {fold + 1}/{n_splits}")

        # Split dataset into train and validation sets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        before_loader = time.time()
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
        after_loader = time.time()
        #print("train,val loader time",after_loader - before_loader)

        # Initialize model
        model = GraphormerModel(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        #print("Train model keys:", model.state_dict().keys())

        #SoftDTWLoss = SoftDTW(use_cuda=True, gamma=1.0, bandwidth=None)
        SoftDTWLoss = SoftDTWLossPyTorch(gamma=1, normalize=True)

        # Define loss function
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

        criterion = loss_fn_gen(loss_function)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        patience = patience  # Early stopping patience 설정
        no_improve_count = 0  # 개선되지 않은 epoch 수 추적
        best_loss = float('inf')

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # batch에 포함된 모든 키와 해당 텐서의 크기 출력
                #for key, value in batch.items():
                #    if isinstance(value, torch.Tensor):
                #        print(f"batch items ::: {key}: shape={value.shape}, length={value.size(0)}")

                batched_data = {k: batch[k] for k in ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input", "attn_edge_type"]}
                targets = batch["targets"].T
                #print("targets.shape",targets.shape)
                outputs = model(batched_data, targets=targets, target_type=target_type)
                #print(f"Batch keys: {batch.keys()}")
                #print(f"Batch targets shape: {batch['targets'].shape}")
                #print(f"Batched data keys: {batched_data.keys()}")
                #print(f"Outputs shape: {outputs.shape}")

                outputs = outputs + 1e-6  # Small value to prevent log(0)
                targets = targets + 1e-6
                #print("shape", outputs.shape, targets.shape)
                if outputs.shape[1] != targets.shape[1]:
                    raise ValueError(f"Output size {outputs.shape} does not match target size {targets.shape}")

                if loss_function == "SID":
                    #print("outputs shape:", outputs.shape)
                    #print("targets shape:", targets.shape)
                    mask = torch.ones_like(outputs, dtype=torch.bool)
                    loss = torch.stack([
                        criterion(outputs[i].unsqueeze(0), targets[i].unsqueeze(0), mask[i], 1e-4)
                        for i in range(outputs.size(0))
                    ]).mean()
                elif loss_function == "SoftDTW":
                    loss = torch.stack([
                        criterion(
                            outputs[i].unsqueeze(0).unsqueeze(-1),  # (seq_len, dim) -> (1, seq_len, dim=1)
                            targets[i].unsqueeze(0).unsqueeze(-1)  # (seq_len, dim) -> (1, seq_len, dim=1)
                        )
                        for i in range(outputs.size(0))
                    ]).mean()
                else:
                    loss = torch.stack([
                        criterion(outputs[i].unsqueeze(0), targets[i].unsqueeze(0))
                        for i in range(outputs.size(0))
                    ]).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)

            cv_metrics["val_loss"].append(avg_epoch_loss)

            # Best model 저장 & Early stopping 체크
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_epoch = epoch
                no_improve_count = 0  # 개선되었으므로 리셋
            else:
                no_improve_count += 1  # 개선되지 않았으므로 증가
            print(f"Epoch {epoch + 1}/{num_epochs}, loss:{avg_epoch_loss}, no_improve_count:{no_improve_count}")
            # Early stopping 조건
            if no_improve_count >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")

                break
        cv_metrics["CV_best_epoch"].append(best_epoch)

        cv_train_end_time = time.time()
        print("cv_train_time", cv_train_end_time - cv_train_start_time)
        # Evaluate on validation set
        cv_valid_start_time = time.time()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                batched_data = {k: batch[k] for k in ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input", "attn_edge_type"]}
                targets = batch["targets"].T
                outputs = model(batched_data, targets=targets, target_type=target_type)

                # Calculate metrics
                y_true = targets.cpu().numpy()
                y_pred = outputs.cpu().detach().numpy()

                # R2, MAE, RMSE 계산
                legacy_val_time = time.time()
                #print("faltten",y_true.flatten().shape, y_pred.flatten().shape)
                r2 = r2_score(y_true.flatten(), y_pred.flatten())
                mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
                rmse = mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False)
                #print("R2, MAE, RMSE CV Val time",time.time() - legacy_val_time)

                # SoftDTW 계산
                #print("y_true SoftDTW",y_true.shape)
                #print("y_pred SoftDTW",y_pred.shape)
                start_softdtw_val_t = time.time()
                softdtw_value = SoftDTWLoss(torch.tensor(y_pred).unsqueeze(0).to(device),
                                            torch.tensor(y_true).unsqueeze(0).to(device)).item()
                #print("SoftDTW_time",time.time() - start_softdtw_val_t)
                # FastDTW 계산
                #start_fastdtw_val_t = time.time()
                #fastdtw_value, _ = fastdtw(torch.tensor(y_pred.flatten()), torch.tensor(y_true.flatten()))
                #print("FASTDTW_time",time.time() - start_fastdtw_val_t)

                # SID 및 SIS 계산
                y_true = y_true + 1e-6

                start_sid_val_t = time.time()
                sid_value = sid_loss(
                    torch.tensor(y_pred).unsqueeze(0).to(device),
                    torch.tensor(y_true).unsqueeze(0).to(device),
                    torch.ones_like(torch.tensor(y_pred).unsqueeze(0), dtype=torch.bool).to(device),
                    threshold=1e-4
                ).mean().item()
                #print("SID_time",time.time() - start_sid_val_t)

                sis_value = 1 / (1 + sid_value)

                # cv_metrics에 추가
                cv_metrics["CV_r2"].append(r2)
                cv_metrics["CV_mae"].append(mae)
                cv_metrics["CV_rmse"].append(rmse)
                cv_metrics["CV_softdtw"].append(softdtw_value)
                print(cv_metrics["CV_softdtw"])
                #cv_metrics["CV_fastdtw"].append(fastdtw_value)
                cv_metrics["CV_sid"].append(sid_value)
                cv_metrics["CV_sis"].append(sis_value)


        print(f"Fold {fold + 1} completed.")
        cv_valid_end_time = time.time()
        #print("cv_valid_time", cv_valid_end_time - cv_valid_start_time)
    # Calculate average CV metrics
    avg_cv_metrics = {f"{key}_avg": np.mean(values) for key, values in cv_metrics.items()}
    avg_cv_metrics["CV_best_epoch_all"] = cv_metrics["CV_best_epoch"]
    print("avg_cv_metrics", avg_cv_metrics)
    # Train on the full dataset
    final_results, best_model_path = train_model(
        config=config,
        target_type=target_type,
        loss_function=loss_function,
        weight_ex=weight_ex,
        num_epochs=num_epochs,
        batch_size=batch_size,
        n_pairs=n_pairs,
        dataset_path=dataset_path,
        patience = patience
    )
    print("final_result at graphormer_CV_train_default",final_results)
    # Merge final results with CV metrics
    #print(avg_cv_metrics)
    final_results.update(avg_cv_metrics)

    if TEST_DATASET == None:
        test_dataset = SMILESDataset(csv_file=testset_path, attn_bias_w=1.0, target_type=target_type)
    else:
        test_dataset = TEST_DATASET

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, test_dataset, n_pairs=n_pairs),
    )

    model.load_state_dict(torch.load(best_model_path, ))
    #best_model = GraphormerModel(config).to(device)
    #print("Current model keys:", model.state_dict().keys())
#
    #for batch in train_loader:
    #    batched_data_dummy = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    #    break
    #dummy_targets = torch.zeros(1, config["output_size"]).to(device)  # 가짜 target 데이터 생성
#
    #best_model(batched_data_dummy, targets=dummy_targets, target_type=target_type)

    test_results = evaluate_on_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        target_type=target_type,
        loss_function=loss_function
    )
    final_results.update(test_results)  # 테스트 결과 추가
    #print("final_results", final_results)
    return final_results


def evaluate_on_test_set(model, test_loader, device, target_type, loss_function):
    results = {
        "test_r2": [], "test_mae": [], "test_rmse": [],
        "test_softdtw": [], "test_sid": [], "test_sis": []
    }

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

    criterion = loss_fn_gen(loss_function)

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in
                            ["x", "adj", "in_degree", "out_degree", "spatial_pos",
                             "attn_bias", "edge_input", "attn_edge_type"]}
            targets = batch["targets"]
            targets = targets.T
            #print("targets",targets.shape, targets.dtype)
            outputs = model(batched_data, targets=targets, target_type=target_type)
            #print("outputs test",outputs.shape)
            #print("batch_size from target_]size",targets.size(0))
            for i in range(targets.size(0)):  # batch_size 만큼 루프
                y_true = targets[i].cpu().numpy() + 1e-6  # 작은 값 추가 (NaN 방지)
                #print(y_true.shape)
                y_pred = outputs[i].cpu().detach().numpy() + 1e-6
                #print(y_pred.shape)

                # R2, MAE, RMSE 계산
                #print(y_true.flatten().shape, y_pred.flatten().shape)
                results["test_r2"].append(r2_score(y_true.flatten(), y_pred.flatten()))
                results["test_mae"].append(mean_absolute_error(y_true.flatten(), y_pred.flatten()))
                results["test_rmse"].append(mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False))

                # SoftDTW 손실 계산
                softdtw_value = SoftDTWLoss(
                    torch.tensor(y_pred).unsqueeze(0).unsqueeze(-1).to(device),
                    torch.tensor(y_true).unsqueeze(0).unsqueeze(-1).to(device)
                ).item()
                results["test_softdtw"].append(softdtw_value)

                # FastDTW 계산
                #fastdtw_value, _ = fastdtw(torch.tensor(y_pred.flatten()), torch.tensor(y_true.flatten()))
                #results["test_fastdtw"].append(fastdtw_value)

                # SID 손실 계산
                threshold = 1e-4
                mask = torch.ones_like(torch.tensor(y_pred).unsqueeze(0), dtype=torch.bool).to(device)

                sid_value = sid_loss(
                    torch.tensor(y_pred).unsqueeze(0).to(device),
                    torch.tensor(y_true).unsqueeze(0).to(device),
                    mask,
                    threshold
                ).mean().item()
                results["test_sid"].append(sid_value)
                results["test_sis"].append(1 / (1 + sid_value))
                #if loss_function == "SID":
                #    threshold = 1e-4
                #    mask = torch.ones_like(torch.tensor(y_pred).unsqueeze(0), dtype=torch.bool).to(device)

                #    sid_value = sid_loss(
                #        torch.tensor(y_pred).unsqueeze(0).to(device),
                #        torch.tensor(y_true).unsqueeze(0).to(device),
                #        mask,
                #        threshold
                #    ).mean().item()
                #    results["test_sid"].append(sid_value)
                #    results["test_sis"].append(1 / (1 + sid_value))
                #else:
                #    results["test_sid"].append(0)
                #    results["test_sis"].append(0)

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
        "output_size": 451,  # 모델 출력 크기
    }

    final_loss = cross_validate_model(config=config, target_type="default",
                                      dataset_path="../../data/train_100.csv", testset_path = "../../data/train_50.csv",
                                      batch_size=10, num_epochs=1000, n_pairs=5,loss_function="SoftDTW", #MSE, MAE, SoftDTW, Huber, SID
                                      patience = 20)
    print(final_loss)
    #print(f"Final Average Loss: {final_loss:.4f}")