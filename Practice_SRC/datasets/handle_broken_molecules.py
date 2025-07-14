
'''
이 스크립트는 DiGress와 같은 확산 모델(Diffusion Model)에서 생성되는
중간 단계의 "부서진(broken)" 분자 그래프를 RDKit으로 어떻게 다룰 수 있는지 시연합니다.

이 최종 버전은 실제 `DiscreteDenoisingDiffusion` 모델의 인스턴스를 생성하고,
그 인스턴스의 `apply_noise` 메소드를 직접 호출하여 확산 과정을 가장 정확하게 시뮬레이션합니다.

과정:
1. SMILES 문자열로부터 PyTorch Geometric 데이터 객체를 생성합니다.
2. 시뮬레이션에 필요한 최소한의 설정(dummy config)과 데이터셋 정보(dummy dataset_infos)를 만듭니다.
3. `DiscreteDenoisingDiffusion` 모델을 이 더미 정보로 초기화합니다.
4. 초기화된 모델의 `apply_noise` 메소드를 호출하여 그래프에 실제와 동일한 방식으로 노이즈를 추가합니다.
5. 이 "부서진" 그래프 정보로부터 RDKit 특징을 안전하게(robustly) 추출합니다.
'''
import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from omegaconf import OmegaConf

# --- 프로젝트의 src 모듈을 임포트하기 위해 경로 추가 --- #
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 실제 프로젝트 모듈 임포트 --- #
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src import utils
from src.diffusion.extra_features import DummyExtraFeatures

# --- 더미 클래스 및 함수 정의 --- #
class DummyDatasetInfos:
    def __init__(self, x_classes, e_classes):
        self.input_dims = {'X': x_classes, 'E': e_classes, 'y': 0}
        self.output_dims = {'X': x_classes, 'E': e_classes, 'y': 0}
        self.nodes_dist = torch.distributions.Categorical(torch.tensor([0.0, 0.0] + [1.0/10] * 10))
        self.node_types = torch.ones(x_classes)
        self.edge_types = torch.ones(e_classes)

    def complete_infos(self, a, b):
        pass

def get_dummy_cfg(transition_type):
    return OmegaConf.create({
        'general': {
            'name': 'test_run',
            'log_every_steps': 50,
            'number_chain_steps': 10,
            'sample_every_val': 5,
            'samples_to_generate': 10,
            'samples_to_save': 5,
            'chains_to_save': 2
        },
        'model': {
            'n_layers': 3,
            'hidden_mlp_dims': {'X': 64, 'E': 64, 'y': 64},
            'hidden_dims': {'dx': 64, 'de': 64, 'dy': 64, 'n_head': 4, 'dim_ffX': 256, 'dim_ffE': 256, 'dim_ffy': 256},
            'diffusion_steps': 500,
            'diffusion_noise_schedule': 'cosine',
            'transition': transition_type,
            'lambda_train': [1.0, 1.0, 1.0]
        },
        'train': {'lr': 1e-4, 'weight_decay': 1e-8, 'batch_size': 4}
    })

# --- 1. SMILES로부터 PyTorch Geometric 그래프 생성 --- #
def smiles_to_torch_geometric(smiles_string, x_classes, e_classes):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    try: AllChem.UFFOptimizeMolecule(mol)
    except Exception: pass

    atom_map = {1:1, 6:2, 7:3, 8:4, 9:5}
    atom_features = [atom_map.get(atom.GetAtomicNum(), 0) for atom in mol.GetAtoms()]
    x = F.one_hot(torch.tensor(atom_features), num_classes=x_classes).float()

    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([(i, j), (j, i)])
        bond_type = int(bond.GetBondTypeAsDouble())
        edge_features.extend([bond_type, bond_type])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = F.one_hot(torch.tensor(edge_features), num_classes=e_classes).float()

    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, smiles=smiles_string)
    return data

# --- 3. 부서진 그래프로부터 특징 재계산 --- #
def extract_features_from_broken_graph(broken_graph_tensors, smiles_string, mol_idx, transition_type):
    node_mask = broken_graph_tensors['node_mask'].squeeze(0) # (N,)
    atom_indices = broken_graph_tensors['X_t'].squeeze(0).argmax(dim=-1).cpu().numpy() # (N,)
    edge_attrs_dense = broken_graph_tensors['E_t'].squeeze(0).argmax(dim=-1).cpu().numpy() # (N, N)

    bond_types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}

    # Dense 엣지 텐서에서 엣지 리스트를 직접 추출
    edge_indices, edge_attrs = [], []
    nodes = torch.where(node_mask)[0] # 실제 노드 인덱스
    for i in nodes:
        for j in nodes:
            if i >= j: continue # 상단 삼각형만 순회
            bond_type_idx = edge_attrs_dense[i, j]
            if bond_type_idx in bond_types: # 유효한 bond_type_idx인 경우에만 엣지 추가
                edge_indices.append((i.item(), j.item()))
                edge_attrs.append(bond_type_idx)

    rw_mol = Chem.RWMol()
    node_to_idx = {}
    atom_map_rev = {1:'H', 2:'C', 3:'N', 4:'O', 5:'F'}
    for i in torch.where(node_mask)[0]:
        atom_idx = atom_indices[i]
        if atom_idx == 0:
            atom = Chem.Atom('C')
            atom.SetBoolProp("is_masked", True)
        else:
            atom = Chem.Atom(atom_map_rev.get(atom_idx, 'C'))
        idx = rw_mol.AddAtom(atom)
        node_to_idx[i.item()] = idx

    bond_types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for (u, v), bond_type_idx in zip(edge_indices, edge_attrs):
        rw_mol.AddBond(node_to_idx[u], node_to_idx[v], bond_types.get(bond_type_idx))

    print(f"  [Info] 수동 조립된 분자: {Chem.MolToSmiles(rw_mol, isomericSmiles=False)}")
    try:
        Chem.SanitizeMol(rw_mol)
        print("  [Info] SanitizeMol 성공!")
    except Exception as e:
        print(f"  [Warning] SanitizeMol 실패 (예상된 동작): {e}")

    # 분자 시각화 및 저장
    output_dir = "molecules_output"
    os.makedirs(output_dir, exist_ok=True)
    mol_filename = os.path.join(output_dir, f"molecule_{transition_type}_{smiles_string.replace('/', '_').replace(':', '_')}_{mol_idx}.png")
    try:
        Draw.MolToImage(rw_mol).save(mol_filename)
        print(f"  [Info] 분자 이미지를 {mol_filename}에 저장했습니다.")
    except Exception as e:
        print(f"  [Warning] 분자 이미지 저장 실패: {e}")

    recalculated_features = []
    for i, atom in enumerate(rw_mol.GetAtoms()):
        features = {}
        features['is_masked'] = atom.GetBoolProp("is_masked") if atom.HasProp("is_masked") else False
        try: features['hybridization'] = str(atom.GetHybridization())
        except Exception: features['hybridization'] = "Error"
        try: features['formal_charge'] = atom.GetFormalCharge()
        except Exception: features['formal_charge'] = -999
        try: features['atomic_num'] = atom.GetAtomicNum()
        except Exception: features['atomic_num'] = -1
        try: features['degree'] = atom.GetTotalDegree()
        except Exception: features['degree'] = -1
        try: features['explicit_valence'] = atom.GetExplicitValence()
        except Exception: features['explicit_valence'] = -1
        try: features['implicit_hydrogens'] = atom.GetImplicitHCount()
        except Exception: features['implicit_hydrogens'] = -1
        try: features['is_aromatic'] = atom.GetIsAromatic()
        except Exception: features['is_aromatic'] = False
        try: features['is_in_ring'] = atom.IsInRing()
        except Exception: features['is_in_ring'] = False
        try: features['num_radical_electrons'] = atom.GetNumRadicalElectrons()
        except Exception: features['num_radical_electrons'] = -1
        try: features['num_h_donors'] = atom.GetNumHDonors()
        except Exception: features['num_h_donors'] = -1
        try: features['num_h_acceptors'] = atom.GetNumHAcceptors()
        except Exception: features['num_h_acceptors'] = -1
        recalculated_features.append(features)
    return recalculated_features

if __name__ == "__main__":
    X_CLASSES, E_CLASSES = 6, 5

    # --- 데이터 로드 ---
    csv_path = os.path.join(script_dir, 'train_50.csv')
    try: df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} 파일을 찾을 수 없습니다.")
        exit()

    smiles_list = df['smiles'].head(3).tolist()

    for transition_type in ['uniform', 'marginal']:
        print(f"\n--- Transition Type: {transition_type} ---")
        dummy_cfg = get_dummy_cfg(transition_type)
        dummy_infos = DummyDatasetInfos(x_classes=X_CLASSES, e_classes=E_CLASSES)

        model = DiscreteDenoisingDiffusion(
            cfg=dummy_cfg, dataset_infos=dummy_infos, train_metrics=None,
            sampling_metrics=None, visualization_tools=None,
            extra_features=DummyExtraFeatures(), domain_features=DummyExtraFeatures()
        )
        print("DiscreteDenoisingDiffusion 모델이 성공적으로 초기화되었습니다.")

        for i, smiles in enumerate(smiles_list):
            print(f"--- 처리 시작: 분자 {i+1} ({smiles}) ---")
            original_graph = smiles_to_torch_geometric(smiles, X_CLASSES, E_CLASSES)
            if original_graph is None: continue
            print(f"  [Info] 원본 분자 생성 완료. 원자 수: {original_graph.num_nodes}")

            # --- 2. 실제 `apply_noise` 메소드를 사용하여 노이즈 추가 ---
            dense_data, node_mask = utils.to_dense(original_graph.x, original_graph.edge_index, original_graph.edge_attr, torch.zeros(original_graph.num_nodes, dtype=torch.long))
            dense_data = dense_data.mask(node_mask)

            noisy_data = model.apply_noise(dense_data.X, dense_data.E, torch.zeros(1, 0), node_mask)
            t_val = noisy_data['t_int'].item()
            num_masked_atoms = (noisy_data['X_t'].argmax(dim=-1) == 0).sum().item()
            print(f"  [Info] `apply_noise` 호출 완료 (t={t_val}). 마스크된 원자 수: {num_masked_atoms}")

            # --- 3. 부서진 그래프로부터 특징 재계산 ---
            print("  [Info] 부서진 그래프로부터 특징 재계산을 시작합니다...")
            recalculated_features = extract_features_from_broken_graph(noisy_data, smiles, i, transition_type)
            print("\n  [결과] 재계산된 원자별 특징:")
            for atom_idx, features in enumerate(recalculated_features):
                print(f"    Atom {atom_idx}: {features}")
            print(f"--- 처리 완료: 분자 {i+1} ---\n")
