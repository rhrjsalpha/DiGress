# visualize_molecule_graph.py
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import networkx as nx


def visualize_rdkit_mol_graph(mol, title="Molecule Graph"):
    # RDKit 2D 좌표 생성
    AllChem.Compute2DCoords(mol)

    # NetworkX 그래프 생성
    G = nx.Graph()
    conf = mol.GetConformer()

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        G.add_node(idx, label=atom.GetSymbol(), pos=(pos.x, pos.y))

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        G.add_edge(i, j, bond_type=bond_type)

    # 2D 좌표 사용
    pos = nx.get_node_attributes(G, 'pos')

    # 시각화
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, labels={i: d['label'] for i, d in G.nodes(data=True)}, font_size=10)

    for (i, j, d) in G.edges(data=True):
        bond_type = d['bond_type']
        dx = pos[j][0] - pos[i][0]
        dy = pos[j][1] - pos[i][1]
        norm = (dx ** 2 + dy ** 2) ** 0.5 if dx ** 2 + dy ** 2 != 0 else 1.0
        ox, oy = -0.05 * dy / norm, 0.05 * dx / norm

        if bond_type == Chem.rdchem.BondType.SINGLE:
            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], width=2, edge_color='black')
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            line1 = [(pos[i][0] + ox, pos[i][1] + oy), (pos[j][0] + ox, pos[j][1] + oy)]
            line2 = [(pos[i][0] - ox, pos[i][1] - oy), (pos[j][0] - ox, pos[j][1] - oy)]
            plt.plot(*zip(*line1), color='black', linewidth=2)
            plt.plot(*zip(*line2), color='black', linewidth=2)
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            offsets = [-0.06, 0, 0.06]
            for offset in offsets:
                ox, oy = -offset * dy / norm, offset * dx / norm
                line = [(pos[i][0] + ox, pos[i][1] + oy), (pos[j][0] + ox, pos[j][1] + oy)]
                plt.plot(*zip(*line), color='black', linewidth=1.5)
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], style='dashed', width=1, edge_color='black')

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

