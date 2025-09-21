# diffusion_model_discrete_debug.py
# -*- coding: utf-8 -*-
"""
디버그용 LightningModule 패치
- train/test step에서 예외 발생 시 배치를 '스킵'하고 0-loss로 계속 진행
- 문제 배치는 ./_bad_batches/bad_rank{LOCAL_RANK}.csv 에 안전하게 기록(csv.writer 사용)
- CSV 컬럼: split, epoch, batch_idx, reason, indices, ident, n_nodes
    * indices : collate에서 마지막에 붙인 전역 인덱스(없으면 빈칸)
    * ident   : SMILES/InChI 등 식별자(가능한 경우)
    * n_nodes : 그래프 노드 수(가능한 경우)
사용법:
    from diffusion_model_discrete_debug import DiscreteDenoisingDiffusionDebug as DiscreteDenoisingDiffusion
"""

from __future__ import annotations

import os
import csv
import traceback
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch

# 원본 모듈
from diffusion_model_discrete import DiscreteDenoisingDiffusion


class DiscreteDenoisingDiffusionDebug(DiscreteDenoisingDiffusion):
    """원본 DiscreteDenoisingDiffusion의 training_step/test_step에 스킵·로깅 기능을 얹은 디버그용 클래스."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        log_dir = Path("./_bad_batches")
        log_dir.mkdir(parents=True, exist_ok=True)

        self._rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._log_path = log_dir / f"bad_rank{self._rank}.csv"

        # csv.writer로 헤더 생성(쉼표/인용 안전)
        if not self._log_path.exists():
            with self._log_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["split", "epoch", "batch_idx", "reason", "indices", "ident", "n_nodes"])

        # 너무 많은 스킵에 대한 안전장치(0이면 무제한)
        self._skip_limit = int(os.environ.get("SKIP_LIMIT", "0"))
        self._skip_count = 0

    # ----------------------------- helpers -----------------------------

    def _split(self) -> str:
        """현재 루프가 train/test 인지."""
        t = getattr(self.trainer, "training", False)
        s = getattr(self.trainer, "testing", False)
        return "train" if t else ("test" if s else "unknown")

    @staticmethod
    def _as_list_str(v: Any) -> List[str]:
        """여러 타입을 CSV 쓰기용 문자열 리스트로 정규화."""
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        if torch.is_tensor(v):
            return [str(x) for x in v.view(-1).tolist()]
        return [str(v)]

    @staticmethod
    def _extract_indices_from_batch(batch: Any) -> Optional[List[int]]:
        """collate에서 마지막에 붙인 long 텐서/필드로부터 전역 인덱스 회수."""
        try:
            # tuple/list 마지막이 long 텐서인 경우
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                last = batch[-1]
                if torch.is_tensor(last) and last.dtype in (torch.long, torch.int64):
                    return [int(x) for x in last.view(-1).tolist()]

            # dict
            if isinstance(batch, dict):
                for k in ("idxs", "indices", "sample_idx", "sample_indices"):
                    v = batch.get(k, None)
                    if torch.is_tensor(v):
                        return [int(x) for x in v.view(-1).tolist()]
                    if isinstance(v, (list, tuple)):
                        return [int(x) for x in v]

            # 객체 속성
            for k in ("idxs", "indices", "sample_idx", "sample_indices"):
                if hasattr(batch, k):
                    v = getattr(batch, k)
                    if torch.is_tensor(v):
                        return [int(x) for x in v.view(-1).tolist()]
                    if isinstance(v, (list, tuple)):
                        return [int(x) for x in v]
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_ident_from_batch(batch: Any) -> List[str]:
        """가능한 식별자(SMILES/InChI)를 문자열 리스트로 추출."""
        idents: List[str] = []
        for k in ("smiles", "SMILES", "inchi", "InChI"):
            try:
                v = batch[k] if isinstance(batch, dict) and k in batch else getattr(batch, k, None)
            except Exception:
                v = None
            idents.extend(DiscreteDenoisingDiffusionDebug._as_list_str(v))
        # 중복 제거 유지
        seen = set()
        uniq = []
        for s in idents:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        return uniq

    def _write_log(
        self,
        split: str,
        epoch: int,
        batch_idx: int,
        reason: str,
        indices: Optional[Sequence[int]] = None,
        ident: Optional[Sequence[str]] = None,
        n_nodes: Optional[int] = None,
    ) -> None:
        """문제 배치 기록(csv.writer 사용 → 쉼표 안전)."""
        # 개행만 정리(쉼표는 csv.writer가 처리)
        reason = reason.replace("\n", " ")
        idx_str = "" if not indices else " ".join(str(int(x)) for x in indices)
        ident_str = "" if not ident else " | ".join(map(str, ident))
        n_nodes_str = "" if n_nodes is None else str(int(n_nodes))

        with self._log_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow([split, int(epoch), int(batch_idx), reason, idx_str, ident_str, n_nodes_str])

    def _zero_loss_out(self) -> dict:
        """DDP 호환을 위해 requires_grad=True인 0-loss 반환."""
        return {"loss": torch.zeros((), device=self.device, requires_grad=True)}

    def _precheck_no_edges(self, data) -> Optional[str]:
        """엣지 없음/노드 수 1 이하 등의 즉시 스킵 사유 메시지 리턴."""
        try:
            n = int(data.x.size(0)) if hasattr(data, "x") and data.x is not None else -1
            m = int(data.edge_index.size(1)) if hasattr(data, "edge_index") and data.edge_index is not None else 0
        except Exception:
            n, m = -1, 0
        if (not hasattr(data, "edge_index")) or (data.edge_index is None) or (data.edge_index.numel() == 0) or (n <= 1):
            return f"no_edges_or_small_graph(n={n},m={m})"
        return None

    # ----------------------------- training -----------------------------

    def training_step(self, data, i):
        split = self._split()

        # 사전 가드(엣지 없음 등)
        msg = self._precheck_no_edges(data)
        if msg:
            idxs = self._extract_indices_from_batch(data)
            ident = self._extract_ident_from_batch(data)
            n_nodes = int(getattr(data, "num_nodes", getattr(data, "x", torch.zeros(0)).size(0) if hasattr(data, "x") else 0))
            self._write_log(split, self.current_epoch, i, msg, idxs, ident, n_nodes)
            self.print(f"[WARN][{split}][rank{self._rank}] skip batch {i}: {msg}")
            return self._zero_loss_out()

        try:
            return super().training_step(data, i)
        except Exception as e:
            reason = "".join(traceback.format_exception_only(type(e), e)).strip()
            idxs = self._extract_indices_from_batch(data)
            ident = self._extract_ident_from_batch(data)
            n_nodes = int(getattr(data, "num_nodes", getattr(data, "x", torch.zeros(0)).size(0) if hasattr(data, "x") else 0))
            self._write_log(split, self.current_epoch, i, reason, idxs, ident, n_nodes)
            self.print(f"[WARN][{split}][rank{self._rank}] skip batch {i}: {reason}")

            self._skip_count += 1
            if self._skip_limit and self._skip_count >= self._skip_limit:
                raise RuntimeError(f"Too many skipped batches: {self._skip_count} >= {self._skip_limit}")

            return self._zero_loss_out()

    # ------------------------------ testing -----------------------------

    def test_step(self, data, i):
        split = self._split()

        # 사전 가드
        msg = self._precheck_no_edges(data)
        if msg:
            idxs = self._extract_indices_from_batch(data)
            ident = self._extract_ident_from_batch(data)
            n_nodes = int(getattr(data, "num_nodes", getattr(data, "x", torch.zeros(0)).size(0) if hasattr(data, "x") else 0))
            self._write_log(split, self.current_epoch, i, msg, idxs, ident, n_nodes)
            self.print(f"[WARN][{split}][rank{self._rank}] skip batch {i}: {msg}")
            return {"loss": torch.zeros((), device=self.device)}

        try:
            return super().test_step(data, i)
        except Exception as e:
            reason = "".join(traceback.format_exception_only(type(e), e)).strip()
            idxs = self._extract_indices_from_batch(data)
            ident = self._extract_ident_from_batch(data)
            n_nodes = int(getattr(data, "num_nodes", getattr(data, "x", torch.zeros(0)).size(0) if hasattr(data, "x") else 0))
            self._write_log(split, self.current_epoch, i, reason, idxs, ident, n_nodes)
            self.print(f"[WARN][{split}][rank{self._rank}] skip batch {i}: {reason}")
            return {"loss": torch.zeros((), device=self.device)}
