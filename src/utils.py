import os
import csv
from copy import deepcopy
from typing import Optional, Union, Dict
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
import pytorch_lightning as pl
from overrides import overrides
from pytorch_lightning.utilities import rank_zero_only
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
from typing import Any

def log_to_csv(log_dict, file_path='logs/train_metrics_discrete.csv'):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_dict)

def create_folders(args):
    try:
        os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


class EMA(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """

    def __init__(self, decay: float = 0.9999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.

        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()

    @overrides
    def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.ema_device:
                self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in
                                       self.ema_state_dict.items()}

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

        self._ema_state_dict_ready = True

    @overrides
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, batch, batch_idx, *args,
                             **kwargs) -> None:
        if self.original_state_dict != {}:
            # Replace EMA weights with training weights
            pl_module.load_state_dict(self.original_state_dict, strict=False)

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
        # Update EMA weights
        with torch.no_grad():
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)

        # Setup EMA for sampling in on_train_batch_end
        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
        ema_state_dict = pl_module.trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
        self.ema_state_dict = ema_state_dict
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the ema static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    @overrides
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

    @overrides
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

    ### 원본 버전 ###
    #@overrides
    #def on_save_checkpoint(
    #        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict
    #) -> dict:
    #    return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

    @overrides
    def on_save_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict
    ) -> None:
        checkpoint["ema_state_dict"] = self.ema_state_dict
        checkpoint["_ema_state_dict_ready"] = self._ema_state_dict_ready

    ### 원본 코드 ###
    #@overrides
    #def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict):
    #    self._ema_state_dict_ready = callback_state["_ema_state_dict_ready"]
    #    self.ema_state_dict = callback_state["ema_state_dict"]

    @overrides
    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]) -> None:
        self._ema_state_dict_ready = checkpoint["_ema_state_dict_ready"]
        self.ema_state_dict = checkpoint["ema_state_dict"]


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)

### 그래프 구조 데이터를 Dense한 텐서 형태로 바꿔서 diffusion 모델이 노드, 엣지 특성을 일괄적으로 처리할 수 있도록 만듦 ###
def to_dense(x, edge_index, edge_attr, batch):

    ## Variable-sized 그래프(batch)를 fixed-size tensor로 변환 ##
    X, node_mask = to_dense_batch(x=x, batch=batch) # 노드 → Dense 변환, 패딩된 그래프 + Mask

    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr) # Self-loop 제거 : 자기 자신에게 향하는 엣지 제거

    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)

    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes) # edge_index → 인접 행렬로 변환
    E = encode_no_edge(E) # 엣지 없는 부분 인코딩

    return PlaceHolder(X=X, E=E, y=None), node_mask

### dense 엣지 행렬 E에서 "엣지가 없는 위치"를 적절히 표시 ###
# 엣지가 없는 위치를 E[:, :, :, 0] = 1로 명시적으로 표시하여
# 이후 diffusion 모델이 모든 위치를 categorical 분포로 처리할 수 있도록 함
"""
입력텐서 E: shape = (B, N, N, D)
B	batch size
N	노드 수 (최대)
D	엣지 타입 one-hot vector 길이 (e.g. 5종류면 D=5)

"""
def encode_no_edge(E):
    assert len(E.shape) == 4 # 4차원 텐서가 아니면 오류
    if E.shape[-1] == 0: # 만약 엣지 타입이 없으면 그대로 반환
        return E

    ## E[b, i, j, :]에 모든 값이 0 → 엣지가 없음
    ## True인 위치가 "엣지 없음"을 나타냄
    no_edge = torch.sum(E, dim=3) == 0 # shape: (B, N, N)

    # "엣지 없음"을 첫 번째 채널 (네번째 차원의 No_edge class)에 표시 (e.g. no-edge용)
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    """
    E[b, i, j, :] = [0, 0, 1, 0, 0]  → edge type 2  
    E[b, i, j, :] = [1, 0, 0, 0, 0]  → no-edge
    마지막 차원이 엣지 타입 분포 (one-hot) 입니다.
    [0, 0, 0, 0, 0] → 엣지 없음인데 softmax나 loss 계산이 안 됨
    그래서 [1, 0, 0, 0, 0]으로 바꿔서 "no-edge" 클래스라고 표시하는 것
    """

    ## 자기 자신에 대한 엣지는 제거 (대각 원소 (i → i)는 제거) ##
    # torch.eye(E.shape[1], dtype=torch.bool)             #
    # E.shape[1] → 노드 수 N                               #
    # torch.eye(N)은 N x N 단위 행렬 (diagonal이 True)      #
    # .unsqueeze(0) → (1, N, N)                           #
    # .expand(E.shape[0], -1, -1) → (B, N, N)             #
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)

    # diag 에서 나온 1 인 대각선 부분을 E 에서 0으로 처리함
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg

###  노드 (X), 엣지 (E), 전역 특성 (y)을 하나의 묶음으로 간편하게 다루기 위한 래퍼 클래스 ###
class PlaceHolder:
    def __init__(self, X, E, y):
        # tensor (X, E, y)를 받아서 인스턴스 속성으로 저장 #
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        # 데이터 타입 (dtype)과 디바이스를 주어진 텐서 x와 같게 맞춤
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        # 유효한 노드/엣지만 유지하고 나머지는 제거(masking)
        # PyTorch는 텐서 크기를 맞춰야 하므로 최대 노드 수로 padding을 합니다.
        # mask는 이 padding된 가짜 노드/엣지를 학습에서 제외하기 위한 필수 처리
        # 그래프 A: 노드 3개
        # 그래프 B: 노드 5개
        # → 최대 노드 수 = 5 → 둘 다 (batch_size=2, num_nodes=5)로 맞춰야 함
        # 그래프 A의 마지막 2개는 의미 없는 가짜 노드지만, loss나 softmax 연산에 포함되어 잡음이 발생

        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            # one-hot → 정수 index 변환 (예: [0, 1, 0] → 1)
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            # 유효하지 않은 노드/엣지는 -1로 마킹
            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            # one hot 그대로 사용
            # 단순 마스킹 (0으로 제거)
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


