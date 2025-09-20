# -*- coding: utf-8 -*-
import os, traceback
from pathlib import Path
from typing import Any, List, Optional
import torch
from diffusion_model_discrete import DiscreteDenoisingDiffusion

class DiscreteDenoisingDiffusionDebug(DiscreteDenoisingDiffusion):
    """
    - train/test step에서 예외 스킵 + CSV 기록
    - CSV: split(train|test),epoch,batch_idx,reason,indices
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        log_dir = Path("./_bad_batches"); log_dir.mkdir(parents=True, exist_ok=True)
        self._rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._log = log_dir / f"bad_rank{self._rank}.csv"
        if not self._log.exists():
            self._log.write_text("split,epoch,batch_idx,reason,indices\n", encoding="utf-8")
        self._skip_limit = int(os.environ.get("SKIP_LIMIT", "0"))
        self._skip_count = 0

    def _split(self) -> str:
        t = getattr(self.trainer, "training", False)
        s = getattr(self.trainer, "testing", False)
        return "train" if t else ("test" if s else "unknown")

    @staticmethod
    def _extract_indices_from_batch(batch: Any) -> Optional[List[int]]:
        try:
            if isinstance(batch, (list, tuple)) and len(batch) > 0:
                last = batch[-1]
                if torch.is_tensor(last) and last.dtype in (torch.long, torch.int64):
                    return [int(x) for x in last.view(-1).tolist()]
            if isinstance(batch, dict):
                for k in ("idxs","indices","sample_idx","sample_indices"):
                    v = batch.get(k, None)
                    if torch.is_tensor(v): return [int(x) for x in v.view(-1).tolist()]
                    if isinstance(v, (list, tuple)): return [int(x) for x in v]
            for k in ("idxs","indices","sample_idx","sample_indices"):
                if hasattr(batch, k):
                    v = getattr(batch, k)
                    if torch.is_tensor(v): return [int(x) for x in v.view(-1).tolist()]
                    if isinstance(v, (list, tuple)): return [int(x) for x in v]
        except Exception:
            pass
        return None

    def _write_log(self, split: str, epoch: int, batch_idx: int, reason: str, indices: Optional[List[int]]):
        reason = reason.replace("\n"," ").replace(",",";")
        idx_str = "" if not indices else " ".join(map(str, indices))
        with self._log.open("a", encoding="utf-8") as f:
            f.write(f"{split},{int(epoch)},{int(batch_idx)},{reason},{idx_str}\n")

    def _zero_loss_out(self):  # train/test 공용
        return {"loss": torch.zeros((), device=self.device, requires_grad=True)}

    def _precheck_no_edges(self, data) -> Optional[str]:
        try:
            n = int(data.x.size(0)) if hasattr(data,"x") and data.x is not None else -1
            m = int(data.edge_index.size(1)) if hasattr(data,"edge_index") and data.edge_index is not None else 0
        except Exception:
            n, m = -1, 0
        if (not hasattr(data,"edge_index")) or (data.edge_index is None) or (data.edge_index.numel()==0) or (n<=1):
            return f"no_edges_or_small_graph(n={n},m={m})"
        return None

    # -------- train --------
    def training_step(self, data, i):
        split = self._split()
        msg = self._precheck_no_edges(data)
        if msg:
            idxs = self._extract_indices_from_batch(data)
            self._write_log(split, self.current_epoch, i, msg, idxs)
            self.print(f"[WARN][{split}][rank{self._rank}] skip batch {i}: {msg}")
            return self._zero_loss_out()
        try:
            return super().training_step(data, i)
        except Exception as e:
            reason = "".join(traceback.format_exception_only(type(e), e)).strip()
            idxs = self._extract_indices_from_batch(data)
            self._write_log(split, self.current_epoch, i, reason, idxs)
            self.print(f"[WARN][{split}][rank{self._rank}] skip batch {i}: {reason}")
            self._skip_count += 1
            if self._skip_limit and self._skip_count >= self._skip_limit:
                raise RuntimeError(f"Too many skipped batches: {self._skip_count} >= {self._skip_limit}")
            return self._zero_loss_out()

    # -------- test --------
    def test_step(self, data, i):
        split = self._split()
        msg = self._precheck_no_edges(data)
        if msg:
            idxs = self._extract_indices_from_batch(data)
            self._write_log(split, self.current_epoch, i, msg, idxs)
            self.print(f"[WARN][{split}][rank{self._rank}] skip batch {i}: {msg}")
            return {"loss": torch.zeros((), device=self.device)}
        try:
            return super().test_step(data, i)
        except Exception as e:
            reason = "".join(traceback.format_exception_only(type(e), e)).strip()
            idxs = self._extract_indices_from_batch(data)
            self._write_log(split, self.current_epoch, i, reason, idxs)
            self.print(f"[WARN][{split}][rank{self._rank}] skip batch {i}: {reason}")
            return {"loss": torch.zeros((), device=self.device)}
