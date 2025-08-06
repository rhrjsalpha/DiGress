from rdkit import Chem
import torch
import pandas as pd
from torch.utils.data import Dataset

# --------------------- util -------------------------------------------------
def smiles_to_tensor(
        smiles: str,
        atom_list: list,                 # ex) ["H","C","N","O","F"]
        bond_list: list                  # ex) [Chem.BondType.SINGLE, ...]
):
    d_x = len(atom_list)
    d_e = len(bond_list) + 1            # 0 = no-edge

    bond_id = {b_type: i+1 for i, b_type in enumerate(bond_list)}

    mol = Chem.MolFromSmiles(smiles)
    n   = mol.GetNumAtoms()

    # 노드 one-hot
    atom_ids = [atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]
    X = torch.eye(d_x)[atom_ids]                      # (N, d_x)

    # 결합 one-hot adjacency
    E = torch.zeros((n, n, d_e))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t    = bond_id[b.GetBondType()]
        E[i, j, t] = 1.0
        E[j, i, t] = 1.0

    return X, E
# ----------------------------------------------------------------------------


# autospectra_dataset.py
import pandas as pd, torch
from rdkit import Chem
from torch.utils.data import Dataset

class AutoSpectraDataset(Dataset):
    """
    CSV ──► (X,E,target)  only
    * atom_vocab / bond_vocab 을 내부에서 자동 생성
    * 첫 pass 에서 RDKit Mol 객체들을 캐싱해 두므로
      전체 SMILES 를 두 번 파싱하지 않음
    """
    def __init__(self, csv_path, target_cols=None):
        df = pd.read_csv(csv_path)

        self.smiles  = df['smiles'].tolist()
        target_cols  = target_cols or df.columns.drop('smiles')
        self.targets = torch.tensor(df[target_cols].values, dtype=torch.float32)

        # ---------- 1) 모든 분자 파싱 & vocab 수집 ----------
        atom_set, bond_set, mols = set(), set(), []
        for smi in self.smiles:
            mol = Chem.MolFromSmiles(smi)
            mols.append(mol)
            atom_set |= {a.GetSymbol() for a in mol.GetAtoms()}
            bond_set |= {b.GetBondType() for b in mol.GetBonds()}

        # vocab 확정 (고정된 순서를 위해 정렬)
        self.atom_vocab = sorted(atom_set)                       # ex) ['Br','C','Cl','F','H','N','O', ...]
        self.bond_vocab = sorted(bond_set,
                                 key=lambda x: int(x))           # SINGLE=1,DOUBLE=2,TRIPLE=3,AROMATIC=12 등
        self.d_x = len(self.atom_vocab)
        self.d_e = len(self.bond_vocab) + 1      # 0 = no-edge

        # id 매핑 dict
        self.atom_id  = {sym: i for i, sym in enumerate(self.atom_vocab)}
        self.bond_id  = {b: i+1 for i, b in enumerate(self.bond_vocab)}  # +1 shift

        # RDKit Mol 캐시
        self.mols = mols

    # ------------------------------------------------------------------
    def __len__(self):  return len(self.smiles)

    def __getitem__(self, idx):
        mol = self.mols[idx]
        X, E = self._mol_to_tensor(mol)
        return X, E, self.targets[idx]

    # ------------------------------------------------------------------
    def _mol_to_tensor(self, mol):
        n = mol.GetNumAtoms()

        atom_ids = [self.atom_id[a.GetSymbol()] for a in mol.GetAtoms()]
        X = torch.eye(self.d_x)[atom_ids]                        # (N,d_x)

        E = torch.zeros((n, n, self.d_e))
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            t    = self.bond_id[b.GetBondType()]
            E[i, j, t] = 1.0
            E[j, i, t] = 1.0
        return X, E



# ---------- collate 그대로 ---------------------------------
def pad_nd(t, L, pad_dims):
    tgt_shape = [L]*pad_dims + list(t.shape[pad_dims:])
    out = t.new_zeros(tgt_shape)

    # ex) pad_dims=1 → (slice(0,n), :)
    #     pad_dims=2 → (slice(0,n), slice(0,n), :)
    slices = [slice(0, s) for s in t.shape[:pad_dims]] \
           + [slice(None)]*(t.dim()-pad_dims)
    out[tuple(slices)] = t
    return out
def collate_auto(batch):
    maxN = max(x.size(0) for x,_,_ in batch)
    X0   = torch.stack([pad_nd(x, maxN, 1) for x,_,_ in batch])
    E0   = torch.stack([pad_nd(e, maxN, 2) for _,e,_ in batch])
    y    = torch.stack([t for _,_,t in batch])
    mask = torch.stack([torch.arange(maxN) < x.size(0) for x,_,_ in batch])
    return X0, E0, y, mask, None


from torch.utils.data import DataLoader

ds = AutoSpectraDataset(
        "train_50.csv",
        target_cols=[f"ex{i}" for i in range(1,51)] +
                    [f"prob{i}" for i in range(1,51)])

loader = DataLoader(ds, batch_size=32,
                    shuffle=True, collate_fn=collate_auto)

X0, E0, y, mask, _ = next(iter(loader))

print("X0 :", X0.shape)         # (32, maxN, D_x)
print("E0 :", E0.shape)         # (32, maxN, maxN, D_e)
print(" y :", y.shape)          # (32, 50, 2)  ← ex_prob 예시
print("mask :", mask.shape)     # (32, maxN)

# 첫 2 개 그래프의 마스크를 시각적으로 확인
for i in range(2):
    n = int(mask[i].sum())
    print(f"sample {i}: n_nodes = {n}\n", mask[i])  # 실제 노드 이후 5칸까지


