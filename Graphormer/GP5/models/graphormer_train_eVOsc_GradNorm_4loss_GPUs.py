import pandas as pd
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
from torch.cuda.amp import autocast, GradScaler
from tslearn.metrics import SoftDTWLossPyTorch
from Graphormer.GP5.Custom_Loss.GradNorm import GradNorm
import math
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import json

def setup(rank, world_size):
    import os
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # libuv 사용을 명시적으로 비활성화
    os.environ['USE_LIBUV'] = '0'

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method='tcp://127.0.0.1:29500'
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def loss_normalizer(first_loss, loss):
    # 손실 값이 tensor인 경우 정규화 값도 tensor로 유지
    if first_loss is not None and isinstance(loss, torch.Tensor):
        normalized_loss = loss / first_loss
    else:
        normalized_loss = loss
    return normalized_loss

def train_model_ex_porb(
    rank,
    world_size,
    config,
    dataset_path,
    target_type,
    loss_function_ex_1,
    loss_function_ex_2,
    loss_function_prob_1,
    loss_function_prob_2,
    num_epochs,
    batch_size,
    n_pairs,
    learning_rate,
    patience,
    alpha,
    DATASET=None,
    training_result_root = "./training_results_4loss.csv"
):
    setup(rank, world_size)  # DDP 초기화

    # ---------------------
    # 1) 장치와 데이터셋 설정
    # ---------------------
    device = torch.device(f"cuda:{rank}")

    if DATASET is None:
        dataset = SMILESDataset(csv_file=dataset_path, attn_bias_w=1.0, target_type=target_type)
    else:
        dataset = DATASET

    # DistributedSampler
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda batch: collate_fn(batch, dataset, n_pairs=n_pairs),
    )

    # ---------------------
    # 2) 모델 및 옵티마 설정
    # ---------------------
    model = GraphormerModel(config).to(device)
    model = DDP(model, device_ids=[rank])  # DDP 래핑

    # 손실함수 정의(기존 코드와 동일)
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

    criterion_ex_1 = loss_fn_gen(loss_function_ex_1).to(device)
    criterion_ex_2 = loss_fn_gen(loss_function_ex_2).to(device)
    criterion_prob_1 = loss_fn_gen(loss_function_prob_1).to(device)
    criterion_prob_2 = loss_fn_gen(loss_function_prob_2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_modifier = GradNorm(num_losses=4, alpha=alpha)

    # ---------------------
    # 3) 변수 초기화
    # ---------------------
    best_loss_ex_1 = float('inf')
    best_loss_ex_2 = float('inf')
    best_loss_prob_1 = float('inf')
    best_loss_prob_2 = float('inf')
    best_model_path = "./best_model.pth"
    ex_1_no_improve = 0
    ex_2_no_improve = 0
    prob_1_no_improve = 0
    prob_2_no_improve = 0

    first_loss_ex_1 = None
    first_loss_ex_2 = None
    first_loss_prob_1 = None
    first_loss_prob_2 = None


    loss_history = {"ex_loss_1": [],"ex_loss_2": [], "prob_loss_1": [],"prob_loss_2": [], "total_loss": [],
                    "normalized_ex_loss_1":[], "normalized_ex_loss_2":[], "normalized_prob_loss_1": [], "normalized_prob_loss_2":[],}
    # DDP에서는 rank=0(메인 프로세스)에서만 로깅/출력 하는 것이 일반적
    best_epoch = 0
    weight_true = [0.5, 0.5, 0.5, 0.5]
    first_loss_ex_1 = None
    first_loss_ex_2 = None
    first_loss_prob_1 = None
    first_loss_prob_2 = None

    weight_history = {"weight_ex_1": [], "weight_ex_2": [], "weight_prob_1": [], "weight_prob_2": []}

    # ---------------------
    # 4) 학습 루프
    # ---------------------
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # epoch마다 샘플러 재설정(셔플)
        model.train()

        epoch_loss = 0.0
        ex_1_losses, ex_2_losses = [], []
        prob_1_losses, prob_2_losses = [], []
        weights_list = []

        now_time = time.time()
        model.train()
        normalized_loss_ex_list_1 = []
        normalized_loss_ex_list_2 = []
        normalized_loss_prob_list_1 = []
        normalized_loss_prob_list_2 = []


        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {
                k: batch[k] for k in [
                    "x", "adj", "in_degree", "out_degree", "spatial_pos",
                    "attn_bias", "edge_input", "attn_edge_type"
                ]
            }
            targets = batch["targets"]

            outputs = model(batched_data, targets=targets, target_type=target_type)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN detected in model outputs!")
                print(f"Sample outputs: {outputs}")
                print(targets)
                raise ValueError("NaN values found in model outputs, check data and model configuration.")


            # ex, prob
            out_ex = outputs[:, :, 0:1] + 1e-8
            tgt_ex = targets[:, :, 0:1] + 1e-8
            out_prob = outputs[:, :, 1:2] + 1e-8
            tgt_prob = targets[:, :, 1:2] + 1e-8

            # 각 손실 계산
            loss_ex_1 = compute_individual_loss(out_ex, tgt_ex, criterion_ex_1, loss_function_ex_1)
            loss_ex_2 = compute_individual_loss(out_ex, tgt_ex, criterion_ex_2, loss_function_ex_2)
            loss_prob_1 = compute_individual_loss(out_prob, tgt_prob, criterion_prob_1, loss_function_prob_1)
            loss_prob_2 = compute_individual_loss(out_prob, tgt_prob, criterion_prob_2, loss_function_prob_2)

            # 최초 손실 기록
            if first_loss_ex_1 is None: first_loss_ex_1 = loss_ex_1.item()
            if first_loss_ex_2 is None: first_loss_ex_2 = loss_ex_2.item()
            if first_loss_prob_1 is None: first_loss_prob_1 = loss_prob_1.item()
            if first_loss_prob_2 is None: first_loss_prob_2 = loss_prob_2.item()

            # Normalized
            norm_ex_1 = loss_ex_1 / first_loss_ex_1
            norm_ex_2 = loss_ex_2 / first_loss_ex_2
            norm_prob_1 = loss_prob_1 / first_loss_prob_1
            norm_prob_2 = loss_prob_2 / first_loss_prob_2

            normalized_loss_ex_list_1.append(norm_ex_1)
            normalized_loss_ex_list_2.append(norm_ex_2)
            normalized_loss_prob_list_1.append(norm_prob_1)
            normalized_loss_prob_list_2.append(norm_prob_2)

            # GradNorm
            weights = loss_modifier.compute_weights(
                [loss_ex_1, loss_ex_2, loss_prob_1, loss_prob_2],
                model
            )
            weights_list.append(weights)

            # 최종 loss
            loss = (
                weight_true[0] * norm_ex_1 +
                weight_true[1] * norm_ex_2 +
                weight_true[2] * norm_prob_1 +
                weight_true[3] * norm_prob_2
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            ex_1_losses.append(loss_ex_1.item())
            ex_2_losses.append(loss_ex_2.item())
            prob_1_losses.append(loss_prob_1.item())
            prob_2_losses.append(loss_prob_2.item())

            # weight = loss_modifier.compute_weights([normalized_loss_ex, normalized_loss_prob], model)

        # ------ epoch 끝 ------
        avg_loss_ex_1 = np.mean(ex_1_losses)
        avg_loss_ex_2 = np.mean(ex_2_losses)
        avg_loss_prob_1 = np.mean(prob_1_losses)
        avg_loss_prob_2 = np.mean(prob_2_losses)
        avg_epoch_loss = epoch_loss / len(dataloader)

        avg_normalized_loss_ex_1 = np.mean([loss.cpu().item() for loss in normalized_loss_ex_list_1])
        avg_normalized_loss_ex_2 = np.mean([loss.cpu().item() for loss in normalized_loss_ex_list_2])
        avg_normalized_loss_prob_1 = np.mean([loss.cpu().item() for loss in normalized_loss_prob_list_1])
        avg_normalized_loss_prob_2 = np.mean([loss.cpu().item() for loss in normalized_loss_prob_list_2])

        # weight 업데이트
        mean_weight = torch.stack(weights_list).mean(dim=0).detach()
        weight_true = mean_weight

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

        # rank=0 에서만 로깅
        if rank == 0:
            epoch_time = time.time() - now_time

            # Early Stopping 로직
            ex_1_no_improve = 0 if (avg_loss_ex_1 < best_loss_ex_1) else ex_1_no_improve + 1
            if avg_loss_ex_1 < best_loss_ex_1:
                best_loss_ex_1 = avg_loss_ex_1
                best_epoch = epoch+1
                torch.save(model.module.state_dict(), best_model_path)

            ex_2_no_improve = 0 if (avg_loss_ex_2 < best_loss_ex_2) else ex_2_no_improve + 1
            if avg_loss_ex_2 < best_loss_ex_2:
                best_loss_ex_2 = avg_loss_ex_2
                best_epoch = epoch+1
                torch.save(model.module.state_dict(), best_model_path)

            prob_1_no_improve = 0 if (avg_loss_prob_1 < best_loss_prob_1) else prob_1_no_improve + 1
            if avg_loss_prob_1 < best_loss_prob_1:
                best_loss_prob_1 = avg_loss_prob_1
                best_epoch = epoch+1
                torch.save(model.module.state_dict(), best_model_path)

            prob_2_no_improve = 0 if (avg_loss_prob_2 < best_loss_prob_2) else prob_2_no_improve + 1
            if avg_loss_prob_2 < best_loss_prob_2:
                best_loss_prob_2 = avg_loss_prob_2
                best_epoch = epoch+1
                torch.save(model.module.state_dict(), best_model_path)

            if epoch == num_epochs - 1:
                torch.save(model.state_dict(), best_model_path)

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, "
                  f"Loss_Ex: {avg_loss_ex_1:.4f},{avg_loss_ex_2:.4f}, Loss_Prob: {avg_loss_prob_1:.4f}, {avg_loss_prob_2:.4f} "
                  f"Normalized_Loss_Ex: {avg_normalized_loss_ex_1:.4f},{avg_normalized_loss_ex_2:.4f} "
                  f"Normalized_Loss_Prob: {avg_normalized_loss_prob_1:.4f}, {avg_normalized_loss_prob_2:.4f} "
                  f"Weights: {weight_true}, Time: {epoch_time:.2f},"
                  f"no_improve_count: {ex_1_no_improve, ex_2_no_improve, prob_1_no_improve, prob_2_no_improve}")
            # patience 확인
            if (ex_1_no_improve >= patience and
                ex_2_no_improve >= patience and
                prob_1_no_improve >= patience and
                prob_2_no_improve >= patience):
                print(f"Early stopping at epoch {epoch+1}")
                break
    cleanup()  # DDP 종료

    # Final evaluation metrics 계산
    print(f"Config used during loading: {config}")
    model.load_state_dict(torch.load(best_model_path, ))
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
                            ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
                             "attn_edge_type"]}
            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=target_type)
            # outputs = torch.clamp(outputs, min=1e-8)

            sid_spectrum_ex, sid_spectrum_prob, sid_spectrum_combined = [], [], []

            nan_case_ex, nan_case_prob, nan_case_combined = 0, 0, 0  # NaN 카운트

            # Batch에서 각 스펙트럼별로 계산
            for i in range(targets.size(0)):  # batch_size 만큼 루프
                y_true = targets[i].cpu().numpy()
                # print("y_true.shape)",y_true.shape)
                y_pred = outputs[i].cpu().detach().numpy()
                # print("y_pred.shape",y_pred.shape)
                if target_type == "ex_prob":
                    # 2D 인덱싱이 필요한 경우 (배치 크기 x n_pairs x 2)
                    y_true_ex = y_true[:, 0]  # ex 값들
                    y_pred_ex = y_pred[:, 0]  # 예측된 ex 값들
                    y_true_prob = y_true[:, 1]  # prob 값들
                    y_pred_prob = y_pred[:, 1]  # 예측된 prob 값들


                else:
                    raise ValueError(f"Unknown target_type: {target_type}")

                # SID 및 SIS 계산

                # print("sid shape",torch.tensor(y_pred_ex).unsqueeze(0).to(device).shape, torch.tensor(y_true_ex).unsqueeze(0).to(device).shape)
                # print(torch.ones_like(torch.tensor(y_pred_ex).unsqueeze(0), dtype=torch.bool).to(device).shape)
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

                # print("softdtw_ex = SoftDTWLoss",torch.tensor(y_pred).unsqueeze(0).unsqueeze(-1).shape)
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

    results_for_csv = {}
    for key, value in results.items():
        if isinstance(value, list):
            results_for_csv[key] = str(value)  # 리스트를 문자열로 직접 변환
            print(f"List found in results for key {key}: {value}")
        else:
            results_for_csv[key] = value
            print(f"Scalar found in results for key {key}: {value}")

    # 결과 저장
    results_csv = pd.DataFrame([results_for_csv])
    results_csv.to_csv(training_result_root, index=False)
    print("Training complete.")
    return training_result_root, best_model_path

# ----- 개별 손실 계산 편의 함수 -----
def compute_individual_loss(outputs, targets, criterion, loss_type):
    # SID인 경우만 mask/threshold를 적용
    if loss_type == "SID":
        threshold = 1e-8
        mask = torch.ones_like(outputs, dtype=torch.bool)
        loss_val = torch.stack([
            criterion(outputs[i].unsqueeze(0), targets[i].unsqueeze(0), mask[i], threshold)
            for i in range(outputs.size(0))
        ]).mean()
    else:
        loss_val = torch.stack([
            criterion(outputs[i].unsqueeze(0), targets[i].unsqueeze(0))
            for i in range(outputs.size(0))
        ]).mean()
    return loss_val

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

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(
        train_model_ex_porb,
        args=(
            world_size,
            config,
            "../../data/train_50.csv",
            "ex_prob",
            # ex1, ex2, prob1, prob2
            "SoftDTW", "MAE",  # 예시
            "SoftDTW", "MAE",  # 예시
            5,  # num_epochs
            32,  # batch_size
            50,  # n_pairs
            0.001,  # learning_rate
            10,  # patience
            0.12,  # alpha
            None, #Dataset
            "./training_results.csv"
        ),
        nprocs=world_size,
        join=True
    )
    results = pd.read_csv("./training_results.csv")
    print(results)

if __name__ == "__main__":
    main()

