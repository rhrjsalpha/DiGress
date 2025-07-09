import re, torch, pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from ogb.utils.mol import smiles2graph

class FlexibleGraphDataset(Dataset):
    """
    target_mode : {"ex_prob", "spectrum", "lambda_max"}
      • ex_prob   : ex_cols + prob_cols   → (N,2)
      • spectrum  : spectrum_cols         → (M,)
      • lambda_max: lambda_cols           → (1,) 또는 (2,)
    """

    def __init__(
        self,
        csv_path: str,
        target_mode: str = "ex_prob",
        smiles_col: str = "smiles",
        ex_cols=None, prob_cols=None,
        spectrum_cols=None,
        lambda_cols=None,           # ["lambda_max"]  or ["lambda_max","intensity"]
        atom_vocab: int = 120,
        bond_vocab: int = 6,
        global_cols=None,  # 그래프-전역 컬럼 이름(리스트·prefix·None)
        global_schema=None,  # encode_global()에 넘길 스키마 dict
    ):
        self.df = pd.read_csv(csv_path)
        self.mode = target_mode.lower()
        self.smiles_col = smiles_col

        # ---------- 1) 컬럼 선택 ----------
        if self.mode == "ex_prob":
            ex_cols   = self._auto_cols(ex_cols,   r"^ex\d+$") # “ex” + 숫자
            prob_cols = self._auto_cols(prob_cols, r"^prob\d+$") # “prob” + 숫자
            self.target_cols = ex_cols + prob_cols
        elif self.mode == "spectrum":
            self.target_cols = self._auto_cols(spectrum_cols, r"^\d+$|^nm\d+$") # 150, 151, …, 549 또는 nm150, nm151 …
        elif self.mode == "lambda_max":
            self.target_cols = self._auto_cols(lambda_cols, r"^lambda")  #“lambda”로 시작
        else:
            raise ValueError("target_mode must be ex_prob | spectrum | lambda_max")

        missing = [c for c in [smiles_col]+self.target_cols if c not in self.df.columns]
        if missing:
            raise KeyError(f"CSV에 없는 컬럼: {missing}")

        # ---------- 2) 숫자화 ----------
        self.df[self.target_cols] = (
            self.df[self.target_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0).astype("float32")
        )

        # ---------- 3) 그래프 ----------
        self.graphs = [smiles2graph(s) for s in self.df[smiles_col]]
        self.atom_vocab, self.bond_vocab = atom_vocab, bond_vocab

        self.global_cols = []

        if global_cols is not None:
            self.global_cols = self._auto_cols(global_cols, r".*")

            if global_schema is None:
                raise ValueError("global_cols 지정 시 global_schema 도 함께 넘겨 주세요")

            self.global_schema = global_schema
            self.df[self.global_cols] = (
                self.df[self.global_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0).astype("float32")
            )

    # ------------------------------------------------------------
    def _auto_cols(self, cols, regex):
        if cols is None:                         # 미지정 → 정규식 탐색
            return [c for c in self.df.columns if re.match(regex, c)]
        if isinstance(cols, str):                # prefix → prefix1..N
            return sorted([c for c in self.df.columns if c.startswith(cols)])
        return cols                              # 이미 리스트

    def __len__(self):  return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        # 어떤 셀이라도 str 타입이면 True
        row = self.df.loc[idx, self.target_cols]
        tgt  = torch.tensor(row.astype("float32").to_numpy(), dtype=torch.float32)

        if self.mode == "ex_prob":
            half = len(self.target_cols)//2
            tgt = tgt.view(2, half).T            # (N,2)  [[ex,prob]...]

        # -------------- one-hot --------------
        x_idx = torch.tensor(g["node_feat"], dtype=torch.long)
        X0 = F.one_hot(x_idx, self.atom_vocab).float()

        N = g["num_nodes"]
        E0 = torch.zeros((N, N, self.bond_vocab), dtype=torch.float32)
        s,d = map(torch.tensor, g["edge_index"]) #  s  → source indices  (출발 노드들)  d  → dest   indices  (도착 노드들)
        b   = torch.tensor(g["edge_feat"]).squeeze() # edge_types  → 각 간선에 대한 '타입 번호' bond_type, bond_dir, is_aromatic
        print(b)
        E0[s,d,b] = 1.0 # 7월 10일 부터 해야 할곳 실행시키고 오류난거 보자

        # ---------- 전역(global) 벡터 ---------- ★

        if self.global_cols:
            feats = self.df.loc[idx, self.global_cols].to_dict()
            global_vec = encode_global(feats, self.global_schema)  # (D_g,)
        else:
            global_vec = None

        return X0, E0, tgt, global_vec


def encode_global(features: dict, schema: dict) -> torch.Tensor:
    """
    여러 전역 특성(features)을 하나의 1-D 텐서로 인코딩한다.

    Parameters
    ----------
    features : {"col_name": value, ...}
    schema   : {"col_name": {"type": "...", **params}}
               type ∈ {"onehot", "multihot", "continuous"}

    Returns
    -------
    torch.FloatTensor  (총 채널 수,)
    """
    chunks = []
    for name, rule in schema.items():
        value = features[name]

        if rule["type"] == "onehot":
            vec = F.one_hot(torch.tensor(int(value)),
                            num_classes=rule["num_classes"]).float()
            chunks.append(vec)

        elif rule["type"] == "multihot":
            vec = torch.zeros(rule["num_classes"], dtype=torch.float32)
            vec[torch.tensor(value, dtype=torch.long)] = 1.0
            chunks.append(vec)

        elif rule["type"] == "continuous":
            if "mean" in rule and "std" in rule:
                norm = (float(value) - rule["mean"]) / rule["std"]
            elif "min" in rule and "max" in rule:
                norm = (float(value) - rule["min"]) / (rule["max"] - rule["min"])
            else:
                raise ValueError(f"continuous rule for '{name}' requires mean/std or min/max")
            chunks.append(torch.tensor([norm], dtype=torch.float32))
        else:
            raise ValueError(f"Unknown encode type: {rule['type']}")

    return torch.cat(chunks, dim=0)

import torch
from torch.utils.data import DataLoader

# 1) Dataset 인스턴스화  ─── ex / prob 쌍 모드
dataset = FlexibleGraphDataset(
    "train_50.csv",        # CSV 경로
    target_mode="ex_prob", # ex+prob 쌍
    ex_cols="ex",          # "ex1, ex2, ..." 자동 탐색
    prob_cols="prob"       # "prob1, prob2, ..." 자동 탐색
)

# 2) DataLoader  ─── node_mask·패딩은 collate_clean이 처리
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    #collate_fn=collate_clean
)

# 3) 한 배치 뽑아보기
X0, E0, y, node_mask, gvec = next(iter(loader))   # gvec은 없으면 None

print("X0:", X0.shape)        # (B, N_max, D_x)
print("E0:", E0.shape)        # (B, N_max, N_max, D_e)
print("y :", y.shape)         # (B, N_pairs, 2)  ← ex/prob 쌍
print("mask:", node_mask.shape)
print("global_vec:", gvec)    # 전역 컬럼을 지정하지 않았다면 None