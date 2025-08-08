from __future__ import annotations

# ===== 기본 패키지 =====
import time
import math
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

### 여러 Target type에 맞추어 나중에 바꾸기 ###
def evaluate_model_metrics(model, dataloader, target_type, device, soft_dtw_fn, sid_loss, is_cv=False, best_epoch=None, is_val=False):
    model.eval()

    # 스펙트럼별 결과 저장용 리스트 초기화
    # Initialize lists for individual and combined metrics
    if target_type == "ex_prob":
        sid_spectrum_ex, sid_spectrum_prob, sid_spectrum_combined = [], [], []
        sis_spectrum_ex, sis_spectrum_prob, sis_spectrum_combined = [], [], []
        r2_spectrum_ex, r2_spectrum_prob, r2_spectrum_combined = [], [], []
        mae_spectrum_ex, mae_spectrum_prob, mae_spectrum_combined = [], [], []
        rmse_spectrum_ex, rmse_spectrum_prob, rmse_spectrum_combined = [], [], []
        softdtw_spectrum_ex, softdtw_spectrum_prob, softdtw_spectrum_combined = [], [], []
        fastdtw_spectrum_ex, fastdtw_spectrum_prob, fastdtw_spectrum_combined = [], [], []
    else:
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
            all_keys = batch.keys()
            batched_data = {k: batch[k] for k in all_keys if k != "targets"}
            # batched_data = {k: batch[k] for k in
            #                ["x_cat", "x_cont", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
            #                 "attn_edge_type"]}

            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=target_type)
            # outputs = torch.clamp(outputs, min=1e-8)

            sid_spectrum_ex, sid_spectrum_prob, sid_spectrum_combined = [], [], []

            nan_case_ex, nan_case_prob, nan_case_combined = 0, 0, 0  # NaN 카운트

            # Batch에서 각 스펙트럼별로 계산
            for i in range(targets.size(0)):  # batch_size 만큼 루프
                time_start = time.time()
                y_true = targets[i].cpu().numpy()
                # print("y_true.shape)",y_true.shape)
                y_pred = outputs[i].cpu().detach().numpy()
                # print("y_pred.shape",y_pred.shape)
                time_end = time.time()
                #print("to_cpu_time:", time_end-time_start)

                if target_type == "ex_prob":
                    # 2D 인덱싱이 필요한 경우 (배치 크기 x n_pairs x 2)
                    y_true_ex = y_true[:, 0]  # ex 값들
                    y_pred_ex = y_pred[:, 0]  # 예측된 ex 값들
                    y_true_prob = y_true[:, 1]  # prob 값들
                    y_pred_prob = y_pred[:, 1]  # 예측된 prob 값들

                    sid_ex = sid_loss(torch.tensor(y_pred_ex + 1e-8, device=device).unsqueeze(0).to(device),
                                      torch.tensor(y_true_ex + 1e-8, device=device).unsqueeze(0).to(device),
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

                    rmse_ex = root_mean_squared_error(y_true_ex, y_pred_ex, )
                    rmse_prob = root_mean_squared_error(y_true_prob, y_pred_prob, )
                    rmse_combined = root_mean_squared_error(y_true.flatten(), y_pred.flatten(), )

                    # print("softdtw_ex = SoftDTWLoss",torch.tensor(y_pred).unsqueeze(0).unsqueeze(-1).shape)
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
                    try:
                        sis_spectrum_ex.append(sis_ex)
                    except UnboundLocalError:
                        print(sis_ex)
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

                elif target_type == "exp_spectrum":
                    for i in range(targets.size(0)):
                        y_true = targets[i].cpu().numpy()
                        y_pred = outputs[i].cpu().detach().numpy()
                        mask = batch["masks"][i].cpu().numpy()

                        y_true = y_true.reshape(-1)
                        y_pred = y_pred.reshape(-1)
                        # 두 경우 모두 대응
                        if torch.is_tensor(mask):
                            mask = mask.view(-1).bool()
                        else:
                            mask = mask.reshape(-1).astype(bool)

                        #print(y_true.shape)
                        #print(y_pred.shape)
                        #print(mask.shape)
                        #print(mask)

                        y_true_masked = y_true[mask == 1]
                        y_pred_masked = y_pred[mask == 1]

                        r2 = r2_score(y_true_masked, y_pred_masked)
                        mae = mean_absolute_error(y_true_masked, y_pred_masked)
                        rmse = root_mean_squared_error(y_true_masked, y_pred_masked,)
                        softdtw = soft_dtw_fn(
                            torch.tensor(y_pred_masked).unsqueeze(0).unsqueeze(-1).to(device),
                            torch.tensor(y_true_masked).unsqueeze(0).unsqueeze(-1).to(device)
                        ).item()
                        #fastdtw_val, _ = fastdtw(y_pred_masked, y_true_masked)
                        sid = sid_loss(
                            torch.tensor(y_pred_masked + 1e-8).unsqueeze(0).to(device),
                            torch.tensor(y_true_masked + 1e-8).unsqueeze(0).to(device),
                            torch.ones_like(torch.tensor(y_pred_masked).unsqueeze(0), dtype=torch.bool).to(device),
                            threshold=1e-8
                        ).mean().item()
                        sis = 1 / (1 + sid)

                        # 저장
                        r2_spectrum.append(r2)
                        mae_spectrum.append(mae)
                        rmse_spectrum.append(rmse)
                        softdtw_spectrum.append(softdtw)
                        #fastdtw_spectrum.append(fastdtw_val)
                        sid_spectrum.append(sid)
                        sis_spectrum.append(sis)

                elif target_type == "nm_distribution":
                    for i in range(targets.size(0)):
                        time_start = time.time()
                        y_true = targets[i].cpu().numpy()
                        y_pred = outputs[i].cpu().detach().numpy()

                        # 정규화나 마스킹 없이 바로 사용 가능
                        r2 = r2_score(y_true, y_pred)
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = root_mean_squared_error(y_true, y_pred,)
                        softdtw = soft_dtw_fn(
                            torch.tensor(y_pred).unsqueeze(0).unsqueeze(-1).to(device),
                            torch.tensor(y_true).unsqueeze(0).unsqueeze(-1).to(device)
                        ).item()
                        #fastdtw_val, _ = fastdtw(y_pred, y_true)
                        sid = sid_loss(
                            torch.tensor(y_pred + 1e-8).unsqueeze(0).to(device),
                            torch.tensor(y_true + 1e-8).unsqueeze(0).to(device),
                            torch.ones_like(torch.tensor(y_pred).unsqueeze(0), dtype=torch.bool).to(device),
                            threshold=1e-8
                        ).mean().item()

                        #print("y_pred", y_pred)
                        #print("y_true", y_true)
                        #print("nm distibution sid",sid)
                        sis = 1 / (1 + sid)

                        r2_spectrum.append(r2)
                        mae_spectrum.append(mae)
                        rmse_spectrum.append(rmse)
                        softdtw_spectrum.append(softdtw)
                        #fastdtw_spectrum.append(fastdtw_val)
                        sid_spectrum.append(sid)
                        sis_spectrum.append(sis)
                        time_end = time.time()
                        total_time = time_end - time_start
                        #print("total_evaluation_time", total_time)

                else:
                    raise ValueError(f"Unknown target_type: {target_type}")

    # 스펙트럼별 평균 계산
    if target_type == "ex_prob":
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
        if is_cv == True and is_val == False:
            column_header_front = "CV_train"
        elif is_cv == True and is_val == True:
            column_header_front = "CV_val"
        elif is_cv == False and is_val == False:
            column_header_front = "train"
        elif is_cv == False and is_val == True:
            column_header_front = "test"
        else:
            column_header_front = "Unknown"

        result_dict = {
            f"best_epoch_{column_header_front}": best_epoch,
            f"r2_avg_ex_{column_header_front}": r2_avg_ex,
            f"r2_avg_prob_{column_header_front}": r2_avg_prob,
            f"r2_avg_combined_{column_header_front}": r2_avg_combined,
            f"mae_avg_ex_{column_header_front}": mae_avg_ex,
            f"mae_avg_prob_{column_header_front}": mae_avg_prob,
            f"mae_avg_combined_{column_header_front}": mae_avg_combined,
            f"rmse_avg_ex_{column_header_front}": rmse_avg_ex,
            f"rmse_avg_prob_{column_header_front}": rmse_avg_prob,
            f"rmse_avg_combined_{column_header_front}": rmse_avg_combined,
            f"softdtw_avg_ex_{column_header_front}": softdtw_avg_ex,
            f"softdtw_avg_prob_{column_header_front}": softdtw_avg_prob,
            f"softdtw_avg_combined_{column_header_front}": softdtw_avg_combined,
            f"fastdtw_avg_ex_{column_header_front}": fastdtw_avg_ex,
            f"fastdtw_avg_prob_{column_header_front}": fastdtw_avg_prob,
            f"fastdtw_avg_combined_{column_header_front}": fastdtw_avg_combined,
            f"sid_avg_ex_{column_header_front}": sid_avg_ex,
            f"sid_avg_prob_{column_header_front}": sid_avg_prob,
            f"sid_avg_combined_{column_header_front}": sid_avg_combined,
            f"sis_avg_ex_{column_header_front}": sis_avg_ex,
            f"sis_avg_prob_{column_header_front}": sis_avg_prob,
            f"sis_avg_combined_{column_header_front}": sis_avg_combined,
        }

    else:
        # exp_spectrum, nm_distribution 공통 처리
        r2_avg = np.mean(r2_spectrum)
        mae_avg = np.mean(mae_spectrum)
        rmse_avg = np.mean(rmse_spectrum)
        softdtw_avg = np.mean(softdtw_spectrum)
        #fastdtw_avg = np.mean([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in fastdtw_spectrum])
        sid_avg = np.mean(sid_spectrum) if sid_spectrum else np.nan
        sis_avg = np.mean(sis_spectrum)

        # 결과 저장용 딕셔너리 생성
        if is_cv and not is_val:
            column_header_front = "CV_train"
        elif is_cv and is_val:
            column_header_front = "CV_val"
        elif not is_cv and not is_val:
            column_header_front = "train"
        elif not is_cv and is_val:
            column_header_front = "test"
        else:
            column_header_front = "Unknown"

        result_dict = {
            f"best_epoch_{column_header_front}": best_epoch,
            f"r2_avg_{column_header_front}": r2_avg,
            f"mae_avg_{column_header_front}": mae_avg,
            f"rmse_avg_{column_header_front}": rmse_avg,
            f"softdtw_avg_{column_header_front}": softdtw_avg,
            #f"fastdtw_avg_{column_header_front}": fastdtw_avg,
            f"sid_avg_{column_header_front}": sid_avg,
            f"sis_avg_{column_header_front}": sis_avg,
        }
    return result_dict