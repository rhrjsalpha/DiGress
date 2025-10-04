# -*- coding: utf-8 -*-
import os, sys, torch
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import Draw
import torch.nn.functional as F

# ── repo 루트 기준으로 경로 조정 ───────────────────────────────────────────
# 예: 이 파일을 <repo>/Practice_SRC/diffusion_visualizer/ 에 두었다면:
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(REPO_ROOT, "src"))

# ── 프로젝트 모듈 ────────────────────────────────────────────────────────
from src.main_spec import CSVSpecDataModule, CSVSpecInfos      # src/main_spec.py에서 import 경로 확인
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.analysis.visualization import MolecularVisualization
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.metrics.molecular_metrics import SamplingMolecularMetrics
from src import utils

# ── 1) 설정 불러오기 & 핵심 override ─────────────────────────────────────
# 1) config 불러오기 (없으면 빈 컨피그)
cfg_path = os.path.join(REPO_ROOT, "configs", "config.yaml")
cfg = OmegaConf.load(cfg_path) if os.path.exists(cfg_path) else OmegaConf.create({})

# 2) 필요한 키가 없으면 기본 블록 병합
base = {
    "general": {
        "name": "fwd_noise_vis",
        "gpus": 0,
        "number_chain_steps": 50,
    },
    "train": {
        "n_epochs": 1,
        "batch_size": 1,
        "num_workers": 0,
        "save_model": False,
    },
    "model": {
        "type": "discrete",
        "transition": "marginal",
        "diffusion_steps": 500,
    },
    "dataset": {
        "name": "csvspec",
        "train_csv": r"C:\Users\kogun\PycharmProjects\DiGress\data_for_ssh\csv\EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv",   # ← 여기에 본인 csv 경로
        "val_csv": None,
        "test_csv": None,
        "remove_h": True,                       # H 숨김(그림 깔끔)
        # 필요시 추가 필드
        # "smiles_col": "SMILES",
        # "inchi_col": "InChI",
        # "global_cols": [],
        # "spectrum_start": 200, "spectrum_end": 800,
        # "boolean_cols": [],
        # "add_h": False,
    }
}
cfg = OmegaConf.merge(base, cfg)  # base로 기본값 채우고, 기존 cfg가 있으면 덮어씀

# 이후에 원하는 override
cfg.dataset.name = "csvspec"
cfg.model.transition = "marginal"

# ↓↓↓ 네 CSV 경로로 바꿔 넣기
cfg.dataset.val_csv   = None
cfg.dataset.test_csv  = None

# ── 2) 데이터/Infos 준비 ──────────────────────────────────────────────────
dm = CSVSpecDataModule(cfg,
                       train_csv=cfg.dataset.train_csv,
                       val_csv=None, test_csv=None,
                       smiles_col=getattr(cfg.dataset, "smiles_col", None),
                       inchi_col=getattr(cfg.dataset, "inchi_col", "InChI"),
                       spectrum_start=getattr(cfg.dataset, "spectrum_start", 200),
                       spectrum_end=getattr(cfg.dataset, "spectrum_end", 800),
                       global_cols=getattr(cfg.dataset, "global_cols", []),
                       spectrum_fill_eps=getattr(cfg.dataset, "spectrum_fill_eps", 1e-8),
                       fixed_vocabs=getattr(cfg.dataset, "fixed_vocabs", {}),
                       boolean_cols=getattr(cfg.dataset, "boolean_cols", []),
                       add_h=getattr(cfg.dataset, "add_h", False),
                       batch_size=1, num_workers=0)
dm.setup(stage="fit")

infos = CSVSpecInfos(dm, remove_h=cfg.dataset.remove_h)

# extra/domain features & 메트릭(모델 init에 필요)
extra_features  = ExtraFeatures(cfg.model.extra_features, dataset_info=infos) if cfg.model.extra_features else DummyExtraFeatures()
domain_features = ExtraMolecularFeatures(dataset_infos=infos)
train_metrics   = TrainMolecularMetricsDiscrete(infos)
sampling_metrics= SamplingMolecularMetrics(infos, train_smiles=None)
viz             = MolecularVisualization(remove_h=cfg.dataset.remove_h, dataset_infos=infos)

# 모델
model = DiscreteDenoisingDiffusion(cfg=cfg,
                                   dataset_infos=infos,
                                   train_metrics=train_metrics,
                                   sampling_metrics=sampling_metrics,
                                   visualization_tools=viz,
                                   extra_features=extra_features,
                                   domain_features=domain_features,
                                   y_loss_mode=getattr(cfg.train, "y_loss_mode", "none"))
model.eval(); torch.set_grad_enabled(False)

# ── 3) 한 분자 가져오기 & dense로 변환 ───────────────────────────────────
g = dm.train_dataset[0]
dense, node_mask = utils.to_dense(g.x, g.edge_index, g.edge_attr, torch.zeros(g.x.size(0), dtype=torch.long))
dense = dense.mask(node_mask)
X0, E0, y0 = dense.X, dense.E, g.y

# 원본 PNG 저장(참고용)
os.makedirs("./fwd_noise_snaps", exist_ok=True)
if getattr(g, "smiles", None):
    m0 = Chem.MolFromSmiles(g.smiles)
    if m0 is not None and cfg.dataset.remove_h:
        m0 = Chem.RemoveHs(m0)
    Draw.MolToImage(m0, size=(300, 300)).save("./fwd_noise_snaps/noise_step_original.png")

# ── 4) t를 고정해서 forward-noise 계산하는 함수 ───────────────────────────
def apply_noise_at_t(model, X, E, y, node_mask, t_int_scalar: int):
    """model.apply_noise를 t 고정 버전으로 재현"""
    T = model.T
    device = X.device
    bs = X.size(0)
    t_int = torch.full((bs, 1), t_int_scalar, device=device, dtype=torch.float32)  # (bs,1)
    s_int = t_int - 1

    t_float = t_int / T
    s_float = s_int / T

    beta_t      = model.noise_schedule(t_normalized=t_float)
    alpha_s_bar = model.noise_schedule.get_alpha_bar(t_normalized=s_float)
    alpha_t_bar = model.noise_schedule.get_alpha_bar(t_normalized=t_float)

    Qtb = model.transition_model.get_Qt_bar(alpha_t_bar, device=device)
    probX = X @ Qtb.X                            # (bs, n, DX)
    probE = E @ Qtb.E.unsqueeze(1)               # (bs, n, n, DE)

    sampled = model.dataset_info.__class__  # dummy to silence linter
    sampled = utils.PlaceHolder(**{})       # ignored

    # 모델 util 사용해 이산 샘플
    from src.diffusion import diffusion_utils
    sampled = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

    X_t = F.one_hot(sampled.X, num_classes=model.Xdim_output).to(probX.dtype)
    E_t = F.one_hot(sampled.E, num_classes=model.Edim_output).to(probE.dtype)

    return {"X_t": X_t, "E_t": E_t, "y_t": y, "t_int": t_int, "t": t_float,
            "beta_t": beta_t, "alpha_s_bar": alpha_s_bar, "alpha_t_bar": alpha_t_bar,
            "node_mask": node_mask}

# ── 5) 여러 t에서 스냅샷 저장 ────────────────────────────────────────────
T = int(cfg.model.diffusion_steps)
steps = [0, T//20, T//10, T//5, T//2, T-1]     # 예시: 0, 25, 50, 100, 250, 499 (T=500일 때)
for t in steps:
    noisy = apply_noise_at_t(model, X0, E0, y0, node_mask, t_int_scalar=t)
    # RDKit 그림: one-hot → argmax로 라벨 얻기
    xlab = noisy["X_t"][0].argmax(dim=-1)                 # [N]
    elab = noisy["E_t"][0].argmax(dim=-1)                 # [N,N]
    # 간단한 그리기(모델 내 시각화 도구 이용)
    try:
        viz.visualize("./fwd_noise_snaps", [[xlab.cpu(), elab.cpu()]], save_n=1, file_prefix=f"noise_step_{t}")
    except TypeError:
        # visualize 시그니처가 버전에 따라 다를 수 있어 RDKit 직접 그리기 폴백
        from src.analysis.rdkit_functions import build_molecule_with_partial_charges
        mol = build_molecule_with_partial_charges(xlab.cpu(), elab.cpu(), infos.atom_decoder)
        if mol is not None and cfg.dataset.remove_h:
            mol = Chem.RemoveHs(mol)
        Draw.MolToImage(mol, size=(300,300)).save(f"./fwd_noise_snaps/noise_step_{t}.png")

print("Saved → ./fwd_noise_snaps/")
