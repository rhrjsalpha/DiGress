# transfer_chain_boot.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import json, os, shlex, subprocess, sys
from pathlib import Path
from typing import Optional, List

# ─────────────────────────────────────────────────────────────────────────────
# 0) 경로/데이터셋 설정
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
print("PROJECT_ROOT",PROJECT_ROOT)
ROOT_FOR_BOOT = REPO_ROOT = PROJECT_ROOT.parent
BOOT_PATH    = PROJECT_ROOT / "main_spec_boot_NoSmoke.py"  # 실행 대상
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
OUTPUTS_ROOTS = [REPO_ROOT / "outputs", PROJECT_ROOT / "outputs"]

# ✅ 스테이지별 개별 설정(배치/에폭/GPU 등)
# 순서에 따라 epoch을 크게 해야 함
# 예 : QM10epoch, Abs10epoch, EM10epoch 총 30epoch 진행하고 싶다면
# QM epoch 은 10 epochm, Abs 20, EM은 30으로 설정해야 한다.
# ckpt 파일 내에는 모델 가중치 외에도 "훈련 재개(학습상태까지 복원)" 까지 하기위한 정보가 들어 있기 때문
DATASETS = {
    "QM": {
        "train_csv":  "/home/user/Spectral_Data/QM_stratified_train_resplit_with_mu_eps.csv",
        "test_csv":   "/home/user/Spectral_Data/QM_stratified_test_resplit_with_mu_eps.csv",
        "batch_size": 1024,
        "epochs":     1000,
        "gpus":       3,
    },
    "Abs": {
        "train_csv":  "/home/user/Spectral_Data/train_good_rows.csv",
        "test_csv":   "/home/user/Spectral_Data/test_good_rows.csv",
        "batch_size": 32,
        "epochs":     1000,
        "gpus":       3,
    },
    "Em": {
        "train_csv":  "/home/user/Spectral_Data/EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv",
        "test_csv":   "/home/user/Spectral_Data/EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv",
        "batch_size": 32,
        "epochs":     1000,
        "gpus":       3,
    },
}

# 실행 순서
ORDER = ["QM", "Abs",] #  "Em"

# 공통 기본값 (스테이지 설정에서 누락 시 폴백)
DEFAULT_GPUS       = 3
DEFAULT_EPOCHS     = 10
DEFAULT_BATCH_SIZE = 32

# 샘플/체인 저장 관련 (원하면 조정)
SAMPLES_TO_GENERATE = 40
SAMPLES_TO_SAVE     = 1
CHAINS_TO_SAVE      = 1
N_CHAIN_STEPS       = 50
FINAL_SAMPLES       = 3000
TARGET_SAFE_FOLDS   = 1

# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _shell(cmd: list[str]) -> int:
    env = dict(os.environ)
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("[RUN]", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, text=True, env=env).returncode

def _slug(s: str) -> str:
    # 폴더명 안전 문자열
    import re
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(s)).strip("-")

def _find_latest_run_dir(run_name: str) -> Optional[Path]:
    """
    여러 후보 outputs 루트에서 '*-<run_name>' 형태의 Hydra 런 디렉토리를 모두 모아
    수정시간(mtime) 최신 것을 고른다. (날짜 디렉토리/비날짜 모두 rglob로 커버)
    """
    candidates: List[Path] = []
    for root in OUTPUTS_ROOTS:
        if not root.exists():
            continue
        # 표준 패턴: HH-MM-SS-<run_name> (상위에 YYYY-MM-DD가 있든 없든 rglob로 다 찾음)
        candidates += [p for p in root.rglob(f"*-{run_name}") if p.is_dir()]
        # 보수적으로, checkpoints/<run_name>/last.ckpt가 있는 곳의 run_dir도 수집
        for ck in root.rglob(f"checkpoints/{run_name}/last.ckpt"):
            candidates.append(ck.parents[2])  # .../<run_dir>/checkpoints/<run_name>/last.ckpt → run_dir
    if not candidates:
        print("[WARN] no hydra runs for name=", run_name,
              " under: ", ", ".join(str(r) for r in OUTPUTS_ROOTS))
        return None
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] picked run dir → {best}")
    return best

def _pick_last_ckpt(run_name: str) -> Optional[Path]:
    run_dir = _find_latest_run_dir(run_name)
    if not run_dir:
        return None

    ckpt_dir = run_dir / "checkpoints" / run_name
    # 표준 위치 우선
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last

    # ep*.ckpt 중 최신
    if ckpt_dir.exists():
        cands = sorted(ckpt_dir.glob("ep*.ckpt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]

    # 아주 드물게 디렉토리 구조가 다른 경우에 대비: checkpoints 아래 전역 검색(최신)
    any_ckpt = sorted((run_dir / "checkpoints").rglob("*.ckpt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if any_ckpt:
        return any_ckpt[0]

    print(f"[WARN] checkpoints not found under {ckpt_dir}")
    return None

def _build_overrides(*, resume_ckpt: Optional[str], gpus: int, epochs: int, batch_size: int) -> tuple[list[str], list[str]]:
    base = [
        "dataset.name=csvspec",
        f"general.gpus={gpus}",
        f"train.n_epochs={epochs}",
        f"train.batch_size={batch_size}",
        "train.save_model=True",
        # eval용(안 써도 되지만 유지)
        f"general.samples_to_generate={SAMPLES_TO_GENERATE}",
        f"general.samples_to_save={SAMPLES_TO_SAVE}",
        f"general.chains_to_save={CHAINS_TO_SAVE}",
        f"general.number_chain_steps={N_CHAIN_STEPS}",
        # ★ 핵심: Eval에서도 final_* 기본값(10000)을 반드시 덮어쓴다
        f"general.final_model_samples_to_generate={FINAL_SAMPLES}",
        f"general.final_model_samples_to_save={SAMPLES_TO_SAVE}",
        f"general.final_model_chains_to_save={CHAINS_TO_SAVE}",
    ]
    # full도 같은 값을 갖게 해 두면 혼동이 없음
    full = list(base)
    if resume_ckpt:
        base.append(f'general.resume="{resume_ckpt}"')
        full.append(f'general.resume="{resume_ckpt}"')
    return base, full

def _run_boot_stage(label: str, conf: dict, resume_ckpt: Optional[str]) -> Path:
    # 스테이지별 값 (없으면 기본값 사용)
    gpus       = int(conf.get("gpus", DEFAULT_GPUS))
    epochs     = int(conf.get("epochs", DEFAULT_EPOCHS))
    batch_size = int(conf.get("batch_size", DEFAULT_BATCH_SIZE))

    train_csv = Path(conf["train_csv"])
    test_csv  = Path(conf["test_csv"])
    assert train_csv.exists(), f"[{label}] train_csv not found: {train_csv}"
    assert test_csv.exists(),  f"[{label}] test_csv not found:  {test_csv}"

    # 폴더 이름 구분이 쉬운 접두사 (라벨+배치+에폭 포함)
    name_prefix = _slug(f"{label}_bs{batch_size}_e{epochs}")

    eval_ovr, full_ovr = _build_overrides(
        resume_ckpt=resume_ckpt, gpus=gpus, epochs=epochs, batch_size=batch_size
    )

    # main_spec_boot_NoSmoke.py 실행 (inline)
    py_code = f"""
from pathlib import Path
import importlib.util, sys
p = r"{BOOT_PATH}"
spec = importlib.util.spec_from_file_location("boot", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
stage_out = (Path(r"{ROOT_FOR_BOOT.as_posix()}") / "bootstrap_cv_like_notval_resample" / r"{label}").as_posix()
m.COMMON_SETTINGS.update({{
    "project_root": r"{ROOT_FOR_BOOT.as_posix()}",
    "train_csv": r"{train_csv.as_posix()}",
    "test_csv":  r"{test_csv.as_posix()}",
    "out_root":  stage_out, 
    "name_prefix": r"{name_prefix}",
    "target_safe": {TARGET_SAFE_FOLDS},
    "reuse_existing_safe": True,
    "pick_order": "asc",
    "bootstrap_trials": 2000,
    "bootstrap_size": 1.0,
    "with_replacement": True,
    "seed": 10,
    "stratify_by": "pH_label,type,Solvent",
    "stratified_bootstrap": True,
    "min_test_size": 1,
    "dedupe_by": "train",
    "level_check": True,
    "level_check_mode": "warn",
    "level_check_ignore": "",
    "max_resample_per_fold": 50,
    "eval_epochs": {epochs},
    "eval_extra_overrides": {eval_ovr!r},
    "do_full_retrain": True,
    "full_extra_overrides": {full_ovr!r},
    }})
m.USE_INLINE_SETTINGS = True
m.main()
    """
    ret = _shell([sys.executable, "-u", "-c", py_code])
    if ret != 0:
        raise SystemExit(f"[ERROR] stage '{label}' failed with return code {ret}")

    # 다음 스테이지에 넘길 체크포인트 찾기 (FINAL full 런 이름 기준)
    final_name = f"{name_prefix}_FINAL_full"
    ckpt = _pick_last_ckpt(final_name)
    if not ckpt:
        roots = ", ".join(str(r) for r in OUTPUTS_ROOTS)
        raise FileNotFoundError(
            f"[ERROR] checkpoint not found for '{final_name}'. "
            f"Searched under: {roots}"
        )
    print(f"[OK] stage '{label}' finished. last.ckpt → {ckpt}")
    return ckpt

# ─────────────────────────────────────────────────────────────────────────────
def main():
    assert BOOT_PATH.exists(), f"main_spec_boot_NoSmoke.py not found: {BOOT_PATH}"
    resume: Optional[Path] = None
    summary: list[dict] = []

    for stage in ORDER:
        conf = DATASETS[stage]
        print(f"\n========== [Stage: {stage}] ==========")
        resume = _run_boot_stage(stage, conf, str(resume) if resume else None)
        summary.append({"stage": stage, "resume_ckpt_for_next": str(resume)})

    out = OUTPUTS_ROOT / "transfer_chain_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n[DONE] Transfer chain finished.")
    print(f"Summary → {out}")

if __name__ == "__main__":
    main()
