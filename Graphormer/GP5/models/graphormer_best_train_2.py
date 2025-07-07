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
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



def train_model_and_plot(
    config,
    target_type="default",
    loss_function="MSE",
    loss_function_ex="SoftDTW",
    loss_function_prob="SoftDTW",
    weight_ex=0.8,
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
        shuffle=True,
    )
    #for batch in dataloader:
    #    print(batch['targets'])

    # Initialize the model, loss function, and optimizer
    model = GraphormerModel(config)
    SoftDTWLoss = SoftDTW(use_cuda=True, gamma=1.0, bandwidth=None)

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
            #print(batch['targets'])
            batched_data = {k: batch[k] for k in ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input", "attn_edge_type"]}
            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=target_type)

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
                outputs_ex = outputs[:, :outputs.size(1) // 2]
                outputs_prob = outputs[:, outputs.size(1) // 2:]
                targets_ex = targets[:, :targets.size(1) // 2]
                targets_prob = targets[:, targets.size(1) // 2:]

                loss_ex = torch.stack([criterion_ex(outputs_ex[i].unsqueeze(0), targets_ex[i].unsqueeze(0)) for i in range(outputs_ex.size(0))]).mean()
                loss_prob = torch.stack([criterion_prob(outputs_prob[i].unsqueeze(0), targets_prob[i].unsqueeze(0)) for i in range(outputs_prob.size(0))]).mean()

                loss = weight_ex * loss_ex + weight_prob * loss_prob
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

    save_dir = "spectrum_plots"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            print(f"Processing batch {batch_count}: {batch['targets'].size()}")
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            print(len(batch), batch.keys())
            #print(batch['targets'])
            batched_data = {k: batch[k] for k in
                            ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
                             "attn_edge_type"]}
            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=target_type)

            # Batch에서 각 스펙트럼별로 계산
            count = 0
            for i in range(targets.size(0)):  # batch_size 만큼 루프
                count+=1
                y_true = targets[i].cpu().numpy()
                y_pred = outputs[i].cpu().detach().numpy()

                # ex와 prob 분리
                y_true_ex = y_true[:, 0]  # ex 값들
                y_pred_ex = y_pred[:, 0]  # 예측된 ex 값들
                y_true_prob = y_true[:, 1]  # prob 값들
                y_pred_prob = y_pred[:, 1]  # 예측된 prob 값들

                plt.figure(figsize=(8, 6))

                # 수직선 플롯 (x 값 고정, y 축을 0에서 해당 강도로 확장)
                plt.vlines(y_true_ex, 0, y_true_prob, color='blue', label='Actual', linewidth=2)
                plt.vlines(y_pred_ex, 0, y_pred_prob, color='red', linestyle='dashed', label='Predicted', linewidth=2)

                # 실제 및 예측된 값에 점 추가
                plt.scatter(y_true_ex, y_true_prob, color='blue', marker='x', s=100)
                plt.scatter(y_pred_ex, y_pred_prob, color='red', marker='o', s=100)

                plt.xlabel("Energy (eV)")
                plt.ylabel("Oscillator Strength")
                plt.title(f"QM Spectrum for Molecule {batch_count}_{count}")
                plt.legend()

                # 플롯 저장
                plot_filename = os.path.join(save_dir, f"molecule_{batch_count}_{count}_spectrum.png")
                plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"Plot saved: {plot_filename}")

                # SID 및 SIS 계산
                sid_ex = sid_loss(torch.tensor(y_pred_ex).unsqueeze(0).to(device),
                                  torch.tensor(y_true_ex).unsqueeze(0).to(device),
                                  torch.ones_like(torch.tensor(y_pred_ex).unsqueeze(0), dtype=torch.bool).to(device),
                                  threshold=1e-4).mean().item()
                sid_prob = sid_loss(torch.tensor(y_pred_prob).unsqueeze(0).to(device),
                                    torch.tensor(y_true_prob).unsqueeze(0).to(device),
                                    torch.ones_like(torch.tensor(y_pred_prob).unsqueeze(0), dtype=torch.bool).to(
                                        device),
                                    threshold=1e-4).mean().item()
                sid_combined = sid_loss(torch.tensor(y_pred).unsqueeze(0).to(device),
                                        torch.tensor(y_true).unsqueeze(0).to(device),
                                        torch.ones_like(torch.tensor(y_pred).unsqueeze(0), dtype=torch.bool).to(device),
                                        threshold=1e-4).mean().item()

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

    fastdtw_avg_ex = np.mean(fastdtw_spectrum_ex)
    fastdtw_avg_prob = np.mean(fastdtw_spectrum_prob)
    fastdtw_avg_combined = np.mean(fastdtw_spectrum_combined)

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
    "num_spatial": 30,            # 공간적 위치 인코딩을 위한 최대 값 default 100
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
    final_loss = train_model_and_plot(config=config,
                             target_type="ex_prob",
                             dataset_path="../../data/train_50.csv",
                             loss_function_ex="Huber",
                             loss_function_prob="SID", #MSE, MAE, SoftDTW, Huber, SID
                             batch_size=5,
                             num_epochs=100,
                             n_pairs=5)
    print(final_loss)
    #print(f"Final Average Loss: {final_loss['final_loss']:.4f}")
