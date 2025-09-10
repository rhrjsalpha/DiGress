import torch
from torch import Tensor
import torch.functional as F
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
import wandb
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, SumExceptBatchKL, CrossEntropyMetric, \
    ProbabilityMetric, NLL


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLoss(nn.Module):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if hasattr(self.node_loss, "compute") else -1
        epoch_edge_loss = self.edge_loss.compute() if hasattr(self.edge_loss, "compute") else -1

        yl = getattr(self, "y_loss", None)
        if hasattr(yl, "compute") and getattr(yl, "total_samples", 0) > 0:
            epoch_y_loss = yl.compute()
        else:
            epoch_y_loss = -1

        to_log = {
            "train_epoch/x_CE": epoch_node_loss,
            "train_epoch/E_CE": epoch_edge_loss,
            "train_epoch/y": epoch_y_loss,
        }
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log



class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train, y_loss_mode):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss_mode = y_loss_mode

        # y 손실 핸들러(필요 시에만 사용)
        if self.y_loss_mode == "none":
            self.y_loss = None
        elif self.y_loss_mode in ("mse", "auto"):
            self.y_loss = SumExceptBatchMSE()
        elif self.y_loss_mode == "mae":
            # 배치축 제외 평균 L1; SumExceptBatchMSE와 스케일을 맞추고 싶으면 .sum으로 바꿔도 됨
            self.y_loss = nn.L1Loss(reduction="mean")
        elif self.y_loss_mode == "ce":
            # y에 대해서는 CE를 직접 계산(F.cross_entropy)로 처리
            self.y_loss = "ce_manual"
        else:
            raise ValueError(f"Unknown y_loss_mode={self.y_loss_mode}")

        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        # print("pred_y, true_y", pred_y, true_y)
        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0

        lambda_E = float(self.lambda_train[0]) if len(self.lambda_train) > 0 else 1.0
        lambda_y = float(self.lambda_train[1]) if len(self.lambda_train) > 1 else 0.0

        if (self.y_loss is None) or (true_y is None) or (true_y.numel() == 0) or (lambda_y == 0.0):
            loss_y = torch.as_tensor(0.0, device=flat_pred_X.device)
        else:
            # auto: true_y dtype 기준으로 회귀/분류 판정
            mode = self.y_loss_mode
            if mode == "auto":
                mode = "mse" if torch.is_floating_point(true_y) else "ce"

            if mode in ("mse", "mae"):
                # 회귀: (B, *) -> (B, D) 로 맞춰 계산
                py = pred_y.reshape(pred_y.size(0), -1).to(dtype=torch.float32)
                ty = true_y.reshape(true_y.size(0), -1).to(dtype=py.dtype)
                loss_y_full = self.y_loss(py, ty)  # SumExceptBatchMSE or L1Loss
                loss_y = loss_y_full if loss_y_full.dim() == 0 else loss_y_full.mean()

            elif mode == "ce":
                # 분류: pred_y = (B, ..., C), true_y = one-hot(C) 또는 인덱스
                if pred_y.dim() < 2 or pred_y.size(-1) < 2:
                    raise ValueError(f"y CE needs logits with C>1, got shape {tuple(pred_y.shape)}")
                logits = pred_y.reshape(-1, pred_y.size(-1))  # (K, C)

                if torch.is_floating_point(true_y) and true_y.dim() == pred_y.dim() and true_y.size(-1) == pred_y.size(
                        -1):
                    idx = true_y.argmax(dim=-1).reshape(-1).long()
                else:
                    idx = true_y.reshape(-1).long()

                if self.y_num_classes is not None and logits.size(-1) != self.y_num_classes:
                    raise ValueError(f"pred_y classes ({logits.size(-1)}) != y_num_classes ({self.y_num_classes})")

                loss_y = F.cross_entropy(logits, idx)

            else:
                raise RuntimeError("unreachable")

        total = loss_X + lambda_E * loss_E + lambda_y * loss_y

        # loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0 # 용도 : 조건부 생성을 위한 predictor 학습 위한 loss 임
        # print("loss_X, loss_E, loss_y",loss_X, loss_E, loss_y)

        if log:
            yl = getattr(self, "y_loss", None)
            to_log = {
                "train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                "train_loss/X_CE": self.node_loss.compute() if hasattr(self.node_loss, "compute") else -1,
                "train_loss/E_CE": self.edge_loss.compute() if hasattr(self.edge_loss, "compute") else -1,
                "train_loss/y": float(loss_y.detach()) if isinstance(loss_y, torch.Tensor) else -1,
            }
            if wandb.run:
                wandb.log(to_log, commit=True)

        return total

    def reset(self):
        # None / 문자열("ce_manual") / 메트릭 객체 모두 안전 처리
        for metric in (self.node_loss, self.edge_loss, getattr(self, "y_loss", None)):
            if hasattr(metric, "reset"):
                metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log



