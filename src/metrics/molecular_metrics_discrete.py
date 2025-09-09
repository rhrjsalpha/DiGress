import torch
from torchmetrics import Metric, MetricCollection
from torch import Tensor
import wandb
import torch.nn as nn


class CEPerClass(Metric):
    full_state_update = False
    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        preds: (B, N, C) or (B, N, N, C) or (K, C)  # K = 마스크된 유효 항목 수
        target: (B, N, C) or (B, N, N, C) or (K, C) # 원-핫(0/1)
        """
        # 확률(특정 클래스)의 텐서 준비
        probs_class = self.softmax(preds)[..., self.class_id]  # (B,N) / (B,N,N) / (K,)

        # --- 타깃(해당 클래스의 0/1)과 마스크 만들기 ---
        # target이 원-핫이므로, 마지막 축에 대해 '0이 아닌 항목이 하나라도 있으면 유효' 로 간주
        if target.dim() >= 2:
            # (B,N,C) 또는 (B,N,N,C) 또는 (K,C)
            target_class = target[..., self.class_id].to(probs_class.dtype)  # 동일 rank, 동일 broadcast 축
            valid_mask = (target != 0.).any(dim=-1)  # (B,N) / (B,N,N) / (K,)
        else:
            raise ValueError(f"Unexpected target shape: {target.shape}")

        # --- preds/target가 이미 마스크되어 있는 경우와 아닌 경우를 모두 처리 ---
        # 케이스 A) preds가 2D (K,) 또는 (K,C) → 이미 마스크된 상태
        if probs_class.dim() == 1:
            probs_flat = probs_class  # (K,)
            if target_class.dim() >= 2:
                # target이 (K,) 이면 그대로, (B,N[,*])이면 유효 항만 골라 길이 K가 되도록 맞춤
                if target_class.dim() == 1:
                    tgt_flat = target_class
                else:
                    tgt_full = target_class.reshape(-1)
                    m_full = valid_mask.reshape(-1)
                    tgt_flat = tgt_full[m_full]
            else:
                tgt_flat = target_class

        # 케이스 B) preds가 3D/4D → 아직 마스크 전, 평탄화 후 동일 마스크 적용
        else:
            probs_flat = probs_class.reshape(-1)
            tgt_flat = target_class.reshape(-1)
            m_flat = valid_mask.reshape(-1)
            probs_flat = probs_flat[m_flat]
            tgt_flat = tgt_flat[m_flat]

        # --- 길이 불일치 방지(안전 가드) ---
        if probs_flat.numel() == 0:
            return
        if probs_flat.numel() != tgt_flat.numel():
            m = min(probs_flat.numel(), tgt_flat.numel())
            probs_flat = probs_flat[:m]
            tgt_flat = tgt_flat[:m]

        # --- 수치 안전화 후 BCE 누적 ---
        probs_flat = torch.clamp(probs_flat, 1e-12, 1. - 1e-12)  # log(0) 방지
        tgt_flat = tgt_flat.to(probs_flat.dtype)

        output = self.binary_cross_entropy(probs_flat, tgt_flat)  # reduction='sum'
        self.total_ce += output
        self.total_samples += probs_flat.numel()

    def compute(self):
        return self.total_ce / self.total_samples


class HydrogenCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NoBondCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticCE(CEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetricsCE(MetricCollection):
    def __init__(self, dataset_infos):
        atom_decoder = dataset_infos.atom_decoder

        class_dict = {'H': HydrogenCE, 'C': CarbonCE, 'N': NitroCE, 'O': OxyCE, 'F': FluorCE, 'B': BoronCE,
                      'Br': BrCE, 'Cl': ClCE, 'I': IodineCE, 'P': PhosphorusCE, 'S': SulfurCE, 'Se': SeCE,
                      'Si': SiCE}

        metrics_list = []
        for i, atom_type in enumerate(atom_decoder):
            metrics_list.append(class_dict[atom_type](i))
        super().__init__(metrics_list)


class BondMetricsCE(MetricCollection):
    def __init__(self):
        ce_no_bond = NoBondCE(0)
        ce_SI = SingleCE(1)
        ce_DO = DoubleCE(2)
        ce_TR = TripleCE(3)
        ce_AR = AromaticCE(4)
        super().__init__([ce_no_bond, ce_SI, ce_DO, ce_TR, ce_AR])


class TrainMolecularMetricsDiscrete(nn.Module):
    def __init__(self, dataset_infos):
        super().__init__()
        self.train_atom_metrics = AtomMetricsCE(dataset_infos=dataset_infos)
        self.train_bond_metrics = BondMetricsCE()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        self.train_atom_metrics(masked_pred_X, true_X)
        self.train_bond_metrics(masked_pred_E, true_E)
        if log:
            to_log = {}
            for key, val in self.train_atom_metrics.compute().items():
                to_log['train/' + key] = val.item()
            for key, val in self.train_bond_metrics.compute().items():
                to_log['train/' + key] = val.item()
            if wandb.run:
                wandb.log(to_log, commit=False)

    def reset(self):
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log['train_epoch/' + key] = val.item()
        for key, val in epoch_bond_metrics.items():
            to_log['train_epoch/' + key] = val.item()
        if wandb.run:
            wandb.log(to_log, commit=False)

        for key, val in epoch_atom_metrics.items():
            epoch_atom_metrics[key] = val.item()
        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = val.item()

        return epoch_atom_metrics, epoch_bond_metrics

