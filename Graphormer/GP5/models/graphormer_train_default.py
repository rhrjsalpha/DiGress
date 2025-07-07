
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
    loss_function="MSE",  # 단일 손실 함수로 통합
    weight_ex=0.5,
    num_epochs=10,
    batch_size=50,
    n_pairs=1,
    learning_rate=0.001,
    dataset_path="../../data/data_example.csv",
    patience = 20,
    DATASET = None
):
    """
    Train the Graphormer model with specified configurations and return the final loss and evaluation metrics.

    Args:
        config (dict): Configuration for the Graphormer model.
        target_type (str): Target type ("default", "ex_prob", "nm_distribution").
        loss_function (str): Loss function to be used.
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

    # Initialize the model, loss function, and optimizer
    model = GraphormerModel(config)
    print("Train model keys:", model.state_dict().keys())
    #SoftDTWLoss = SoftDTW(use_cuda=True, gamma=1.0, bandwidth=None)
    SoftDTWLoss = SoftDTWLossPyTorch(gamma=1, normalize=True)
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

    criterion = loss_fn_gen(loss_function)  # 단일 손실 함수로 설정

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    weight_prob = 1 - weight_ex

    # 손실 값 저장을 위한 리스트 초기화
    loss_history = []
    best_loss = float('inf')
    best_model_path = "./best_model.pth"
    best_epoch = 0
    patience = patience  # Early stopping patience 설정
    no_improve_count = 0  # 개선되지 않은 epoch 수 추적

    # Training loop
    for epoch in range(num_epochs):
        now_time = time.time()
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input", "attn_edge_type"]}
            targets = batch["targets"].T
            outputs = model(batched_data, targets=targets, target_type=target_type)

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN detected in model outputs!")
                print(f"Sample outputs: {outputs}")
                raise ValueError("NaN values found in model outputs, check data and model configuration.")

            # Compute loss
            if target_type == "default":
                outputs = outputs + 1e-6  # 작은 값 추가
                targets = targets + 1e-6  # 작은 값 추가

                if outputs.shape[1] != targets.shape[1]:
                    raise ValueError(
                        f"Output size {outputs.shape} does not match target size {targets.shape}"
                    )

                if loss_function == "SID":
                    threshold = 1e-6
                    mask = torch.ones_like(outputs, dtype=torch.bool)
                    loss = torch.stack([
                        criterion(outputs[i].unsqueeze(0), targets[i].unsqueeze(0), mask[i], threshold)
                        for i in range(outputs.size(0))
                    ]).mean()
                elif loss_function == "SoftDTW":
                    loss = torch.stack([
                        criterion(
                            outputs[i].unsqueeze(0).unsqueeze(-1),  # Add batch and dim dimensions
                            targets[i].unsqueeze(0).unsqueeze(-1)
                        )
                        for i in range(outputs.size(0))
                    ]).mean()
                else:
                    loss = torch.stack([
                        criterion(outputs[i].unsqueeze(0), targets[i].unsqueeze(0))
                        for i in range(outputs.size(0))
                    ]).mean()
                #print("loss:", loss)

            else:
                raise ValueError("Invalid target type")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)

        # Best model 저장 & Early stopping 체크
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch
            no_improve_count = 0  # 개선되었으므로 리셋
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_count += 1  # 개선되지 않았으므로 증가

        # Early stopping 조건
        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        epoch_time = time.time() - now_time
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.10f}, Time: {epoch_time:.2f}, no_improve_count: {no_improve_count}")

    # Final evaluation metrics 계산
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Initialize lists for individual metrics
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
            batched_data = {k: batch[k] for k in
                            ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
                             "attn_edge_type"]}
            targets = batch["targets"].T
            #print("before model target.shape",targets.shape)
            outputs = model(batched_data, targets=targets, target_type=target_type)
            #print(outputs.shape)
            for i in range(targets.size(0)):  # batch_size 만큼 루프
                y_true = targets[i].cpu().numpy()
                y_pred = outputs[i].cpu().detach().numpy()

                # SID 및 SIS 계산
                y_true_tensor = torch.tensor(y_true, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
                y_true_tensor = y_true_tensor + 1e-6
                y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
                # 마스크 생성 및 SID 손실 계산
                mask = torch.ones_like(y_pred_tensor, dtype=torch.bool).to(device)
                sid_value = sid_loss(y_pred_tensor, y_true_tensor, mask, threshold=1e-4).mean().item()
                sis_value = 1 / (1 + sid_value)

                # R², MAE, RMSE 계산
                r2 = r2_score(y_true.flatten(), y_pred.flatten())
                mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
                rmse = mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False)

                # SoftDTW 및 FastDTW 계산
                softdtw_value = SoftDTWLoss(y_pred_tensor, y_true_tensor).item()
                #fastdtw_value, _ = fastdtw(torch.tensor(y_pred.flatten()), torch.tensor(y_true.flatten()))

                # 결과 저장
                sid_spectrum.append(sid_value)
                sis_spectrum.append(sis_value)
                r2_spectrum.append(r2)
                mae_spectrum.append(mae)
                rmse_spectrum.append(rmse)
                softdtw_spectrum.append(softdtw_value)
                #fastdtw_spectrum.append(fastdtw_value)

    # 스펙트럼별 평균 계산
    r2_avg = np.mean(r2_spectrum)
    mae_avg = np.mean(mae_spectrum)
    rmse_avg = np.mean(rmse_spectrum)
    softdtw_avg = np.mean(softdtw_spectrum)
    #fastdtw_avg = np.mean([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in fastdtw_spectrum])
    sid_avg = np.mean(sid_spectrum)
    sis_avg = np.mean(sis_spectrum)

    # 결과 저장용 딕셔너리 생성
    results = {
        "best_epoch": best_epoch,
        "r2_avg": r2_avg,
        "mae_avg": mae_avg,
        "rmse_avg": rmse_avg,
        "softdtw_avg": softdtw_avg,
        #"fastdtw_avg": fastdtw_avg,
        "sid_avg": sid_avg,
        "sis_avg": sis_avg,
    }

    # 결과 출력
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    print("Evaluation complete.")
    return results, best_model_path




###############################################################################
config = {
    "num_atoms": 100,              # 분자의 최대 원자 수 (그래프의 노드 개수) 100
    "num_in_degree": 10,           # 그래프 노드의 최대 in-degree
    "num_out_degree": 10,          # 그래프 노드의 최대 out-degree
    "num_edges": 50,               # 그래프의 최대 엣지 개수 50
    "num_spatial": 100,            # 공간적 위치 인코딩을 위한 최대 값 default 100
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

    target_type = "default" # "ex_prob" "default" "nm_distribution"
    final_loss = train_model(config=config,
                             target_type=target_type,
                             dataset_path="../../data/train_50.csv",
                             loss_function="SoftDTW", #MSE, MAE, SoftDTW, Huber, SID
                             batch_size=10,
                             num_epochs=1500,
                             n_pairs=3,
                             patience = 20
                             )
    print(final_loss)
