# 다른 곳 import #

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
from chemprop.train.loss_functions import sid_loss
from tslearn.metrics import SoftDTWLossPyTorch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



def train_model(
    config,
    target_type="default",
    loss_function="MSE",
    loss_function_ex="SoftDTW",
    loss_function_prob="SoftDTW",
    weight_ex=0.5,
    num_epochs=10,
    batch_size=50,
    n_pairs=1,
    learning_rate=0.001,
    dataset_path="../../data/data_example.csv",
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
    dataset = SMILESDataset(csv_file=dataset_path, attn_bias_w=1.0, target_type=target_type)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, dataset, n_pairs=n_pairs),
    )

    # Initialize the model, loss function, and optimizer
    model = GraphormerModel(config)
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    weight_prob = 1 - weight_ex

    # 손실 값 저장을 위한 리스트 초기화
    loss_history = []
    best_loss = float('inf')
    best_model_path = None
    best_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        now_time = time.time()
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input", "attn_edge_type"]}
            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=target_type)

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN detected in model outputs!")
                print(f"Sample outputs: {outputs}")
                raise ValueError("NaN values found in model outputs, check data and model configuration.")

            # Compute loss
            if target_type == "ex_prob":
                outputs_ex = outputs[:, :, 0:1]
                targets_ex = targets[:, :, 0:1]

                # SID Loss를 사용할 경우 마스크 생성
                if loss_function_ex == "SID":
                    threshold = 1e-4
                    mask_ex = torch.ones_like(outputs_ex, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                    loss_ex = torch.stack([
                        criterion_ex(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0), mask_ex[i],threshold)
                        for i in range(outputs_ex.size(0))
                    ]).mean()
                else:
                    loss_ex = torch.stack([
                        criterion_ex(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0))
                        for i in range(outputs_ex.size(0))
                    ]).mean()

                outputs_prob = outputs[:, :, 1:2]
                targets_prob = targets[:, :, 1:2]

                # SID Loss를 사용할 경우 마스크 생성
                if loss_function_prob == "SID":
                    threshold = 1e-4
                    mask_prob = torch.ones_like(outputs_prob, dtype=torch.bool)  # 모든 영역 포함 (조건부 수정 가능)
                    loss_prob = torch.stack([
                        criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0), mask_prob[i],threshold)
                        for i in range(outputs_prob.size(0))
                    ]).mean()
                else:
                    loss_prob = torch.stack([
                        criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0))
                        for i in range(outputs_prob.size(0))
                    ]).mean()

                # Final loss 계산
                loss = weight_ex * loss_ex + weight_prob * loss_prob


            elif target_type == "default":
                print(target_type)

                # `default`에서는 eV를 고려하지 않고 확률값만 예측 (전체 크기 대응)
                outputs_prob = outputs  # 모델 출력이 전체 확률값
                targets_prob = targets  # 타겟도 동일한 확률값 크기
                # 유연한 처리: 가변 크기에 대응
                if outputs_prob.shape[1] != targets_prob.shape[1]:
                    raise ValueError(
                        f"Output size {outputs_prob.shape} does not match target size {targets_prob.shape}")

                if loss_function_prob == "SID":
                    threshold = 1e-4
                    mask_prob = torch.ones_like(outputs_prob, dtype=torch.bool)
                    print(mask_prob.shape)
                    loss_prob = torch.stack([criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0), mask_prob[i],threshold)
                        for i in range(outputs_prob.size(0))]).mean()
                    print(loss_prob.shape)
                else:
                    loss_prob = torch.stack([criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0)) for i in range(outputs_prob.size(0))]).mean()
                print("loss_prob",loss_prob)
                loss = loss_prob  # 최종 손실은 prob 값만 고려
            else:
                raise ValueError("Invalid target type")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)

        # Best model 저장
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch
            best_model_path = "./best_model.pth"
            torch.save(model.state_dict(), best_model_path)
        epoch_time = time.time() - now_time
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.10f}, Time: {epoch_time:.2f}")

    # Final evaluation metrics 계산
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

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in
                            ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
                             "attn_edge_type"]}
            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=target_type)

            # Batch에서 각 스펙트럼별로 계산
            for i in range(targets.size(0)):  # batch_size 만큼 루프
                y_true = targets[i].cpu().numpy()
                print("y_true.shape)",y_true.shape)
                y_pred = outputs[i].cpu().detach().numpy()
                print("y_pred.shape",y_pred.shape)
                if target_type == "ex_prob":
                    # 2D 인덱싱이 필요한 경우 (배치 크기 x n_pairs x 2)
                    y_true_ex = y_true[:, 0]  # ex 값들
                    print("y_true_ex.shape",y_true_ex.shape)
                    y_pred_ex = y_pred[:, 0]  # 예측된 ex 값들
                    print("y_pred_ex.shape",y_pred_ex.shape)
                    y_true_prob = y_true[:, 1]  # prob 값들
                    print("y_true_prob.shape",y_true_prob.shape)
                    y_pred_prob = y_pred[:, 1]  # 예측된 prob 값들
                    print("y_pred_prob.shape",y_pred_prob.shape)
                elif target_type == "default":
                    # default 모드에서는 전체 벡터가 확률값이므로 그대로 사용
                    print("type",type(y_true))
                    y_true_ex = np.expand_dims(y_true, axis=-1)  # 전체 값 그대로 사용 (1D → 2D 변환)
                    y_pred_ex = np.expand_dims(y_pred, axis=-1)  # 전체 값 그대로 사용
                    y_true_prob = np.expand_dims(y_true, axis=-1)  # 동일한 변수로 설정
                    y_pred_prob = np.expand_dims(y_pred, axis=-1)  # 동일한 변수로 설정
                    print("last shape",y_true_ex.shape)
                else:
                    raise ValueError(f"Unknown target_type: {target_type}")

                # SID 및 SIS 계산
                if target_type == "ex_prob":
                    print("sid shape",torch.tensor(y_pred_ex).unsqueeze(0).to(device).shape, torch.tensor(y_true_ex).unsqueeze(0).to(device).shape)
                    print(torch.ones_like(torch.tensor(y_pred_ex).unsqueeze(0), dtype=torch.bool).to(
                                          device).shape)
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
                elif target_type == "default":
                    # 1D 데이터를 3D로 변환 (batch_size=1, seq_len=451, dimension=1)
                    y_true_ex_1 = np.expand_dims(y_true, axis=-1)  # (451, 1)
                    y_true_ex_1 = np.expand_dims(y_true_ex_1, axis=0)  # (1, 451, 1)

                    y_pred_ex_1 = np.expand_dims(y_pred, axis=-1)  # (451, 1)
                    y_pred_ex_1 = np.expand_dims(y_pred_ex_1, axis=0)  # (1, 451, 1)

                    # PyTorch 텐서 변환 및 device로 이동
                    y_true_ex_2 = torch.tensor(y_true_ex_1, dtype=torch.float32).to(device)
                    y_pred_ex_2 = torch.tensor(y_pred_ex_1, dtype=torch.float32).to(device)
                    print("y_true_ex_2",y_true_ex_2,y_pred_ex_2)
                    # SID 손실 계산 (전체 벡터 비교)
                    sid_ex = sid_loss(
                        y_pred_ex_2,
                        y_true_ex_2,
                        torch.ones_like(y_pred_ex_2, dtype=torch.bool).to(device),
                        threshold=1e-4
                    ).mean().item()
                    print("sid_ex",sid_ex)

                    sid_prob = sid_loss(
                        y_pred_ex_2,
                        y_true_ex_2,
                        torch.ones_like(y_pred_ex_2, dtype=torch.bool).to(device),
                        threshold=1e-4
                    ).mean().item()

                    sid_combined = sid_loss(
                        y_pred_ex_2,
                        y_true_ex_2,
                        torch.ones_like(y_pred_ex_2, dtype=torch.bool).to(device),
                        threshold=1e-4
                    ).mean().item()

                # SIS 계산
                sis_ex = 1 / (1 + sid_ex)
                sis_prob = 1 / (1 + sid_prob)
                sis_combined = 1 / (1 + sid_combined)

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

                #print("softdtw_ex = SoftDTWLoss",torch.tensor(y_pred).unsqueeze(0).unsqueeze(-1))
                if target_type == "ex_prob":
                    print("softdtw shape", torch.tensor(y_pred_ex).unsqueeze(0).unsqueeze(-1).to(device).shape)
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
                elif target_type == "default":
                    # PyTorch 텐서로 변환 및 차원 추가 (numpy → torch tensor 변환)
                    y_true_ex = torch.tensor(np.expand_dims(y_true, axis=0), dtype=torch.float32).to(
                        device)  # (1, seq_len)
                    y_pred_ex = torch.tensor(np.expand_dims(y_pred, axis=0), dtype=torch.float32).to(
                        device)  # (1, seq_len)

                    # 배치 크기 및 차원 추가 (batch_size=1, seq_len, 1)
                    y_true_ex = y_true_ex.unsqueeze(-1)
                    y_pred_ex = y_pred_ex.unsqueeze(-1)

                    print("Processed shape for DTW:", y_true_ex.shape)  # 예상: (1, 451, 1)

                    # SoftDTW 손실 계산
                    softdtw_ex = SoftDTWLoss(y_pred_ex, y_true_ex).item()
                    softdtw_prob = SoftDTWLoss(y_pred_ex, y_true_ex).item()  # 동일한 확률값으로 계산
                    softdtw_combined = SoftDTWLoss(y_pred_ex, y_true_ex).item()

                fastdtw_ex, _ = fastdtw(torch.tensor(y_pred_ex), torch.tensor(y_true_ex))
                fastdtw_prob, _ = fastdtw(torch.tensor(y_pred_prob), torch.tensor(y_true_prob))
                fastdtw_combined, _ = fastdtw(torch.tensor(y_pred.flatten()), torch.tensor(y_true.flatten()))

                # Append results to respective lists
                sid_spectrum_ex.append(sid_ex)
                sid_spectrum_prob.append(sid_prob)
                sid_spectrum_combined.append(sid_combined)

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

    sid_avg_ex = np.mean(sid_spectrum_ex)
    sid_avg_prob = np.mean(sid_spectrum_prob)
    sid_avg_combined = np.mean(sid_spectrum_combined)

    sis_avg_ex = np.mean(sis_spectrum_ex)
    sis_avg_prob = np.mean(sis_spectrum_prob)
    sis_avg_combined = np.mean(sis_spectrum_combined)
    results = {}
    # 결과 저장용 딕셔너리 생성
    results.update({
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

    # 결과 저장
    #with open("./training_results.json", "w") as f:
    #    json.dump(results, f)
    print("Training complete.")
    return results, best_model_path


###############################################################################
config = {
    "num_atoms": 57,              # 분자의 최대 원자 수 (그래프의 노드 개수) 100
    "num_in_degree": 10,           # 그래프 노드의 최대 in-degree
    "num_out_degree": 10,          # 그래프 노드의 최대 out-degree
    "num_edges": 62,               # 그래프의 최대 엣지 개수 50
    "num_spatial": 26,            # 공간적 위치 인코딩을 위한 최대 값 default 100
    "num_edge_dis": 10,            # 엣지 거리 인코딩을 위한 최대 값
    "edge_type": "multi_hop",      # 엣지 타입 ("multi_hop" 또는 다른 값 가능)
    "multi_hop_max_dist": 5,       # Multi-hop 엣지의 최대 거리
    "num_encoder_layers": 6,       # Graphormer 모델에서 사용할 인코더 레이어 개수
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
    target_type = "ex_prob" # "ex_prob" "default" "nm_distribution"
    final_loss = train_model(config=config,
                             target_type=target_type,
                             dataset_path="../../data/data_example.csv",
                             loss_function_ex="Huber",
                             loss_function_prob="MSE", #MSE, MAE, SoftDTW, Huber, SID
                             batch_size=1,
                             num_epochs=2,
                             n_pairs=3)
    print(final_loss)
    #print(f"Final Average Loss: {final_loss['final_loss']:.4f}")
