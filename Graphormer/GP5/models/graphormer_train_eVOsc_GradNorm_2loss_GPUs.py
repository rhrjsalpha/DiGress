import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset, collate_fn
from Graphormer.GP5.models.graphormer import GraphormerModel

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
from torch.utils.data import Subset


class CustomLossWrapper(nn.Module):
    def __init__(self, loss_fn):
        """
        CustomLossWrapper에 기본적인 mask와 threshold를 저장.
        """
        super(CustomLossWrapper, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, model_spectra, target_spectra, mask, threshold):
        """
        forward에서는 저장된 mask와 threshold를 활용.
        """
        return self.loss_fn(model_spectra, target_spectra, mask, threshold)



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    # libuv 사용을 명시적으로 비활성화
    os.environ['USE_LIBUV'] = '0'

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method='tcp://127.0.0.1:29501'
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_model_ex_porb(rank, args):
                        #world_size, config, train_dataset_path, val_dataset_path, target_type, loss_function_ex, loss_function_prob,
              #num_epochs, batch_size, n_pairs, learning_rate, patience, alpha, TRAIN_DATASET=None, VAL_DATASET=None,training_result_root = "./training_results_2loss.csv", skip_setup=False):

    if not args.skip_setup:
        setup(rank, args.world_size)

    # GPU 할당
    device = torch.device(f"cuda:{rank}")

    # dataset 생성
    # Initialize dataset and DataLoader
    if args.TRAIN_DATASET is None:
        TRAIN_DATASET = SMILESDataset(csv_file=args.train_dataset_path, attn_bias_w=1.0, target_type=args.target_type)
    else:
        TRAIN_DATASET = args.TRAIN_DATASET

    if args.VAL_DATASET is None:
        VAL_DATASET = SMILESDataset(csv_file=args.val_dataset_path, attn_bias_w=1.0, target_type=args.target_type)
    else:
        VAL_DATASET = args.VAL_DATASET

    # DistributedSampler 적용
    train_sampler = torch.utils.data.distributed.DistributedSampler(TRAIN_DATASET, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(
        TRAIN_DATASET,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, TRAIN_DATASET.dataset if isinstance(TRAIN_DATASET,Subset) else TRAIN_DATASET, n_pairs=args.n_pairs)
    )

    val_loader = DataLoader(
        VAL_DATASET,
        batch_size=args.batch_size,
        collate_fn=lambda batch: collate_fn(batch, VAL_DATASET.dataset if isinstance(VAL_DATASET,Subset) else VAL_DATASET, n_pairs=args.n_pairs),
    )

    # 모델 설정 및 DDP로 래핑
    model = GraphormerModel(args.config).to(device)
    model = DDP(model, device_ids=[rank])

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
            #def sid_loss_wrapper(model_spectra, target_spectra, mask, threshold):
            return CustomLossWrapper(sid_loss)
            #return sid_loss_wrapper

    criterion_ex = loss_fn_gen(args.loss_function_ex).to(device)
    criterion_prob = loss_fn_gen(args.loss_function_prob).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_modifier = GradNorm(num_losses=2, alpha=args.alpha)

    # 손실 값 저장을 위한 리스트 초기화
    best_model_path = "./best_model.pth"
    best_epoch = 0
    loss_history = {"ex_loss": [], "prob_loss": [], "total_loss": [], "normalized_ex_loss": [],
                    "normalized_prob_loss": []}
    weight_history = {"weight_ex": [], "weight_prob": []}
    best_loss_ex, best_loss_prob = float('inf'), float('inf')
    ex_no_improve_count, prob_no_improve_count = 0, 0

    first_loss_ex, first_loss_prob = None, None
    weight_true = [0.5, 0.5]

    early_stop_flag = torch.tensor([0], device=device)
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)  # 매 epoch마다 sampler shuffle
        model.train()
        now_time = time.time()
        model.train()
        normalized_loss_ex_list = []
        normalized_loss_prob_list = []
        weight_list = []
        epoch_loss = 0.0
        loss_ex_list, loss_prob_list = [], []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input", "attn_edge_type"]}
            targets = batch["targets"].to(device)

            outputs = model(batched_data, targets=targets, target_type=args.target_type)

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN detected in model outputs!")
                print(f"Sample outputs: {outputs}")
                raise ValueError("NaN values found in model outputs, check data and model configuration.")

                # Compute loss
            if args.target_type == "ex_prob":
                outputs_ex = outputs[:, :, 0:1] + 1e-8
                targets_ex = targets[:, :, 0:1] + 1e-8

                # SID Loss를 사용할 경우 마스크 생성
                if args.loss_function_ex == "SID":
                    threshold = 1e-8
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

                # outputs_prob = torch.sigmoid(outputs[:, :, 1:2])
                # targets_prob = torch.sigmoid(targets[:, :, 1:2])
                outputs_prob = outputs[:, :, 1:2] + 1e-8
                targets_prob = targets[:, :, 1:2] + 1e-8
                # outputs_prob = torch.clamp(outputs[:, :, 1:2], min=1e-8)
                # targets_prob = torch.clamp(targets[:, :, 1:2], min=1e-8)

                # SID Loss를 사용할 경우 마스크 생성
                if args.loss_function_prob == "SID":
                    threshold = 1e-8
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

                # 손실 값이 tensor인 경우 정규화 값도 tensor로 유지
                if first_loss_ex is not None and isinstance(loss_ex, torch.Tensor):
                    normalized_loss_ex = loss_ex / first_loss_ex
                else:
                    normalized_loss_ex = loss_ex

                if first_loss_prob is not None and isinstance(loss_prob, torch.Tensor):
                    normalized_loss_prob = loss_prob / first_loss_prob
                else:
                    normalized_loss_prob = loss_prob

                weight = loss_modifier.compute_weights([loss_ex, loss_prob], model)
                # weight = loss_modifier.compute_weights([normalized_loss_ex, normalized_loss_prob], model)
                weight_list.append(weight)

                # Final loss 계산
                # loss = weight_true[0] * loss_ex + weight_true[1] * loss_prob
                loss = weight_true[0] * normalized_loss_ex + weight_true[1] * normalized_loss_prob
            else:
                raise ValueError("Invalid target type")

            if first_loss_ex is None:
                first_loss_ex = loss_ex.item()
            if first_loss_prob is None:
                first_loss_prob = loss_prob.item()

            normalized_loss_ex = loss_ex.item() / first_loss_ex if first_loss_ex != 0 else 1.0
            normalized_loss_prob = loss_prob.item() / first_loss_prob if first_loss_prob != 0 else 1.0

            optimizer.zero_grad()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[RANK {rank}] NaN detected in loss!")
                print("Outputs:", outputs)
                print("Targets:", targets)
                exit()

            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"[Rank {rank}] Gradient nan or inf in {name}")

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

        avg_weight = torch.stack(weight_list).mean(dim=0).detach()
        weight_true = avg_weight
        weight_history["weight_ex"].append(weight_true[0].item())
        weight_history["weight_prob"].append(weight_true[1].item())

        loss_history["ex_loss"].append(avg_loss_ex)
        loss_history["prob_loss"].append(avg_loss_prob)
        loss_history["total_loss"].append(avg_epoch_loss)
        loss_history["normalized_ex_loss"].append(avg_normalized_loss_ex)
        loss_history["normalized_prob_loss"].append(avg_normalized_loss_prob)

        if rank == 0:
            #print(f"Epoch {epoch+1}, Avg Loss Ex: {avg_loss_ex}, Avg Loss Prob: {avg_loss_prob}")
            # Early stopping 체크
            if avg_loss_ex < best_loss_ex:
                best_epoch = epoch
                best_loss_ex = avg_loss_ex
                ex_no_improve_count = 0
                if rank == 0:
                    torch.save(model.module.state_dict(), best_model_path)
            else:
                ex_no_improve_count += 1

            if avg_loss_prob < best_loss_prob:
                best_epoch = epoch
                best_loss_prob = avg_loss_prob
                prob_no_improve_count = 0
                if rank == 0:
                    torch.save(model.module.state_dict(), best_model_path)
            else:
                prob_no_improve_count += 1

        epoch_time = time.time() - now_time
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_epoch_loss:.4f}, "
              f"Loss_Ex: {avg_loss_ex:.4f}, Loss_Prob: {avg_loss_prob:.4f}, "
              f"Normalized_Loss_Ex: {avg_normalized_loss_ex:.4f}, Normalized_Loss_Prob: {avg_normalized_loss_prob:.4f}, "
              f"Weights: {weight_true}, Time: {epoch_time:.2f},no_improve_count: {ex_no_improve_count, prob_no_improve_count}")

        if epoch == args.num_epochs - 1:
            torch.save(model.module.state_dict(), best_model_path)
            best_epoch = args.num_epochs

        # ✅ 두 손실 모두 patience 기준 충족 시 종료
        if ex_no_improve_count >= args.patience  and prob_no_improve_count >= args.patience :
            early_stop_flag.fill_(1)  # stop 신호
            print(f"Early stopping triggered at epoch {epoch + 1} (Both losses satisfied patience)")
        else:
            early_stop_flag.fill_(0)
        dist.broadcast(early_stop_flag, src=0)
        if early_stop_flag.item() == 1:
            break
    if rank == 0:
        # 손실 히스토리 저장
        print("save csv")
        loss_df = pd.DataFrame(loss_history)
        loss_df.to_csv(f"./loss_history_rank{rank}.csv", index=False)

        # weight 히스토리 저장
        weight_df = pd.DataFrame(weight_history)
        weight_df.to_csv(f"./weight_history_rank{rank}.csv", index=False)

        # best model 정보 따로 저장
        pd.DataFrame([{
            "best_model_path": best_model_path,
            "best_epoch": best_epoch
        }]).to_csv(f"./best_info_rank{rank}.csv", index=False)

    if not args.skip_setup:
        print("clean up")
        cleanup()

    #print("dist.destory_process_group()")
    #dist.destroy_process_group()

    return best_model_path, best_epoch, loss_history, train_loader, val_loader

def evaluate_model(best_model_path, train_loader, val_loader, args, best_epoch, loss_history, weight_history):

    SoftDTWLoss = SoftDTWLossPyTorch(gamma=1, normalize=True)

    # only rank 0 does validation
    device = torch.device("cuda:0")
    args.config["out_of_training"] = True
    args.config["output_size"] = 100

    model = GraphormerModel(args.config).to(device)
    state_dict = torch.load(best_model_path, map_location=device)
    # 'module.' prefix 제거 (DDP 저장 모델이면)
    if next(iter(state_dict)).startswith("module."):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
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
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in
                            ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
                             "attn_edge_type"]}
            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=args.target_type)
            # outputs = torch.clamp(outputs, min=1e-8)

            sid_spectrum_ex, sid_spectrum_prob, sid_spectrum_combined = [], [], []

            nan_case_ex, nan_case_prob, nan_case_combined = 0, 0, 0  # NaN 카운트

            # Batch에서 각 스펙트럼별로 계산
            for i in range(targets.size(0)):  # batch_size 만큼 루프
                y_true = targets[i].cpu().numpy()
                # print("y_true.shape)",y_true.shape)
                y_pred = outputs[i].cpu().detach().numpy()
                # print("y_pred.shape",y_pred.shape)
                if args.target_type == "ex_prob":
                    # 2D 인덱싱이 필요한 경우 (배치 크기 x n_pairs x 2)
                    y_true_ex = y_true[:, 0]  # ex 값들
                    y_pred_ex = y_pred[:, 0]  # 예측된 ex 값들
                    y_true_prob = y_true[:, 1]  # prob 값들
                    y_pred_prob = y_pred[:, 1]  # 예측된 prob 값들


                else:
                    raise ValueError(f"Unknown target_type: {args.target_type}")

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
                    sis_spectrum_ex.append(sis_ex)
                else:
                    nan_case_ex += 1
                    print(f"[SID_ex] NaN detected at case {i}")

                if not math.isnan(sid_prob):
                    sid_spectrum_prob.append(sid_prob)
                    sis_prob = 1 / (1 + sid_prob)
                    sis_spectrum_prob.append(sis_prob)
                else:
                    nan_case_prob += 1
                    print(f"[SID_prob] NaN detected at case {i}")

                if not math.isnan(sid_combined):
                    sid_spectrum_combined.append(sid_combined)
                    sis_combined = 1 / (1 + sid_combined)
                    sis_spectrum_combined.append(sis_combined)
                else:
                    nan_case_combined += 1
                    print(f"[SID_combined] NaN detected at case {i}")

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

    #print("SID ex 평균 (NaN 제외):", sid_avg_ex, "NaN 개수:", nan_case_ex)
    #print("SID prob 평균 (NaN 제외):", sid_avg_prob, "NaN 개수:", nan_case_prob)
    #print("SID combined 평균 (NaN 제외):", sid_avg_combined, "NaN 개수:", nan_case_combined)
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
    #print("loss_history_training", loss_history)
    for key, value in loss_history.items():
        results[f"{key}_history_training"] = value
    for key, value in weight_history.items():
        results[f"{key}_history_training"] = value

    results_for_csv = {}
    for key, value in results.items():
        if isinstance(value, list):
            results_for_csv[key] = str(value)  # 리스트를 문자열로 직접 변환
            #print(f"List found in results for key {key}: {value}")
        else:
            results_for_csv[key] = value
            #print(f"Scalar found in results for key {key}: {value}")

    val_results = {
        "val_r2_ex": [], "val_r2_prob": [], "val_r2_combined": [],
        "val_mae_ex": [], "val_mae_prob": [], "val_mae_combined": [],
        "val_rmse_ex": [], "val_rmse_prob": [], "val_rmse_combined": [],
        "val_softdtw_ex": [], "val_softdtw_prob": [], "val_softdtw_combined": [],
        "val_sid_ex": [], "val_sid_prob": [], "val_sid_combined": [],
        "val_sis_ex": [], "val_sis_prob": [], "val_sis_combined": [],
    }

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batched_data = {k: batch[k] for k in
                            ["x", "adj", "in_degree", "out_degree", "spatial_pos", "attn_bias", "edge_input",
                             "attn_edge_type"]}
            targets = batch["targets"]
            outputs = model(batched_data, targets=targets, target_type=args.target_type)

            for i in range(targets.size(0)):
                y_true = targets[i].cpu().numpy()
                y_pred = outputs[i].cpu().detach().numpy()
                y_true_ex = y_true[:, 0]
                y_pred_ex = y_pred[:, 0]
                y_true_prob = y_true[:, 1]
                y_pred_prob = y_pred[:, 1]

                val_results["val_r2_ex"].append(r2_score(y_true_ex, y_pred_ex))
                val_results["val_r2_prob"].append(r2_score(y_true_prob, y_pred_prob))
                val_results["val_r2_combined"].append(r2_score(y_true.flatten(), y_pred.flatten()))

                val_results["val_mae_ex"].append(mean_absolute_error(y_true_ex, y_pred_ex))
                val_results["val_mae_prob"].append(mean_absolute_error(y_true_prob, y_pred_prob))
                val_results["val_mae_combined"].append(mean_absolute_error(y_true.flatten(), y_pred.flatten()))

                val_results["val_rmse_ex"].append(mean_squared_error(y_true_ex, y_pred_ex, squared=False))
                val_results["val_rmse_prob"].append(mean_squared_error(y_true_prob, y_pred_prob, squared=False))
                val_results["val_rmse_combined"].append(mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False))

                val_results["val_softdtw_ex"].append(SoftDTWLoss(
                    torch.tensor(y_pred_ex).unsqueeze(0).unsqueeze(-1).to(device),
                    torch.tensor(y_true_ex).unsqueeze(0).unsqueeze(-1).to(device)).item())
                val_results["val_softdtw_prob"].append(SoftDTWLoss(
                    torch.tensor(y_pred_prob).unsqueeze(0).unsqueeze(-1).to(device),
                    torch.tensor(y_true_prob).unsqueeze(0).unsqueeze(-1).to(device)).item())
                val_results["val_softdtw_combined"].append(SoftDTWLoss(
                    torch.tensor(y_pred).unsqueeze(0).to(device),
                    torch.tensor(y_true).unsqueeze(0).to(device)).item())

                sid_ex = sid_loss(torch.tensor(y_pred_ex).unsqueeze(0).to(device),
                                  torch.tensor(y_true_ex).unsqueeze(0).to(device),
                                  torch.ones_like(torch.tensor(y_pred_ex).unsqueeze(0), dtype=torch.bool).to(device),
                                  threshold=1e-8).mean().item()
                sid_prob = sid_loss(torch.tensor(y_pred_prob).unsqueeze(0).to(device),
                                    torch.tensor(y_true_prob).unsqueeze(0).to(device),
                                    torch.ones_like(torch.tensor(y_pred_prob).unsqueeze(0), dtype=torch.bool).to(device),
                                    threshold=1e-8).mean().item()
                sid_combined = sid_loss(torch.tensor(y_pred).unsqueeze(0).to(device),
                                        torch.tensor(y_true).unsqueeze(0).to(device),
                                        torch.ones_like(torch.tensor(y_pred).unsqueeze(0), dtype=torch.bool).to(device),
                                        threshold=1e-8).mean().item()

                val_results["val_sid_ex"].append(sid_ex)
                val_results["val_sid_prob"].append(sid_prob)
                val_results["val_sid_combined"].append(sid_combined)

                val_results["val_sis_ex"].append(1 / (1 + sid_ex) if not math.isnan(sid_ex) else np.nan)
                val_results["val_sis_prob"].append(1 / (1 + sid_prob) if not math.isnan(sid_prob) else np.nan)
                val_results["val_sis_combined"].append(1 / (1 + sid_combined) if not math.isnan(sid_combined) else np.nan)

    # ✅ 평균 계산
    for key, values in val_results.items():
        results[f"{key}_avg"] = np.nanmean(values)
        results[f"{key}_list"] = values


    # 리스트가 아닌 항목(평균값)만 저장
    for key, value in results.items():
        if isinstance(value, list):
            continue  # 리스트는 저장하지 않음
        results_for_csv[key] = value

    # 저장
    results_csv = pd.DataFrame([results_for_csv])
    results_csv.to_csv(args.training_result_root, index=False)
    print("Training complete.")

    return args.training_result_root
########################################

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

    mp.spawn(train_model_ex_porb,
             args=(world_size,
                   config,
                   "../../data/train_50.csv",  # train dataset
                   "../../data/train_50.csv",  # validation dataset (임시로 동일하게 사용, 나중에 분리 가능)
                   "ex_prob",
                   "MAE",
                   "MAE",
                   5,  # num_epochs
                   50,  # batch_size
                   50,  # n_pairs
                   0.001,  # learning_rate
                   20,  # patience
                   0.12),  # alpha
             nprocs=world_size,
             join=True)

    results = pd.read_csv("./training_results_2loss.csv")
    print(results)

if __name__ == "__main__":
    print(sid_loss)
    print("Type:", type(sid_loss))

    main()


