import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

from src.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, DiscreteUniformTransition

# SMILES → Mol
smiles = "CCCC1=CC=CC=C1"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)

# atom index → symbol 매핑
SYMBOLS = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H']

# 엣지 타입 인덱스 → 결합 스타일 매핑
BOND_STYLE_MAP = {
    0: 'SINGLE',
    1: 'DOUBLE',
    2: 'TRIPLE',
    3: 'AROMATIC',
    4: 'NONE'
}


# Mol → Graph(X, E, G) 생성 + 원자 기호 저장
def mol_to_graph_with_labels(mol):
    num_atoms = mol.GetNumAtoms()
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_indices = [SYMBOLS.index(s) if s in SYMBOLS else 0 for s in atom_symbols]
    X = torch.eye(len(SYMBOLS))[atom_indices]  # one-hot
    E = torch.zeros(num_atoms, num_atoms, 5)

    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i, symbol=atom.GetSymbol())

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(i, j, bond_type=bond.GetBondType())
        E[i, j, 0] = 1
        E[j, i, 0] = 1

    return X, E, G, atom_symbols

X, E, G, atom_symbols = mol_to_graph_with_labels(mol)

""" 노이즈 스케줄 만큼 분자그래프에 trasition(전이행렬) 그래프가 반영됨"""

# Noise Schedule
schedule = PredefinedNoiseScheduleDiscrete(noise_schedule="cosine", timesteps=1000)
print(schedule)

### Discrete Diffusion 모델에서 사용되는 전이 행렬(Transition Matrix)을 정의하는 부분입니다. ###
# 특히 노이즈를 어떻게 추가할 것인지에 대한 규칙을 지정 #
transition = DiscreteUniformTransition(x_classes=len(SYMBOLS), e_classes=5, y_classes=1)
print(transition)

# 시각화 함수: X_t 기반으로 노드 라벨 결정
def visualize_graph_atoms_edges(X_t, E_t, G, pos, title, ax):
    predicted_ids = X_t.argmax(dim=1).tolist()
    labels = {i: SYMBOLS[predicted_ids[i]] for i in G.nodes}
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=10)

    for i, j in G.edges():
        bond_type_idx = int(E_t[i, j].argmax().item())
        bond_style = BOND_STYLE_MAP.get(bond_type_idx, 'SINGLE')

        if bond_style == 'SINGLE':
            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], width=2, edge_color='black', ax=ax)
        elif bond_style == 'DOUBLE':
            offset = 0.03
            line1 = [(pos[i][0] - offset, pos[i][1] - offset), (pos[j][0] - offset, pos[j][1] - offset)]
            line2 = [(pos[i][0] + offset, pos[i][1] + offset), (pos[j][0] + offset, pos[j][1] + offset)]
            ax.plot(*zip(*line1), color='black', linewidth=2)
            ax.plot(*zip(*line2), color='black', linewidth=2)
        elif bond_style == 'TRIPLE':
            offsets = [-0.04, 0, 0.04]
            for o in offsets:
                line = [(pos[i][0] + o, pos[i][1] + o), (pos[j][0] + o, pos[j][1] + o)]
                ax.plot(*zip(*line), color='black', linewidth=1.5)
        elif bond_style == 'AROMATIC':
            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], style='dashed', width=1, edge_color='black', ax=ax)
        else:
            continue  # 'NONE'

    ax.set_title(title)
    ax.axis('off')

# 시각화 시점
timesteps = [0.0, 0.2, 0.4, 0.6, 0.8]
fig, axes = plt.subplots(1, len(timesteps), figsize=(4*len(timesteps), 4))
pos = nx.spring_layout(G, seed=42)

for i, t_scalar in enumerate(timesteps):
    t = torch.tensor([t_scalar])
    beta_t = schedule(t_normalized=t) #### 노이즈 스케줄
    Qt = transition.get_Qt(beta_t, device='cpu') #### transition 을 통해 노이즈 추가

    # 노이즈 추가
    X_logits = X @ Qt.X[0]
    X_t = torch.nn.functional.gumbel_softmax(X_logits, tau=1.0, hard=True)

    E_flat = E.view(-1, E.shape[-1])
    E_logits = E_flat @ Qt.E[0]
    E_t = torch.nn.functional.gumbel_softmax(E_logits, tau=1.0, hard=True)
    E_t = E_t.view(E.shape)

    # 시각화 (E_t 추가됨)
    visualize_graph_atoms_edges(X_t, E_t, G, pos, f"t = {t_scalar:.1f}", axes[i])

plt.tight_layout()
plt.show()





