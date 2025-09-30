import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 깨진 PNG도 읽기 시도
import networkx as nx
import numpy as np
import rdkit.Chem
import wandb
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw

class MolecularVisualization:
    def __init__(self, remove_h, dataset_infos):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)
        
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, 'molecule_{}.png'.format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            try:
                Draw.MolToFile(mol, file_path)
                if wandb.run and log is not None:
                    # print(f"Saving {file_path} to wandb")
                    wandb.log({log: wandb.Image(file_path)}, commit=True)
            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")

    def _safe_draw(mol, file_name, legend):
        try:
            # 케쿨화 강제 안 함: invalid aromatic/고리 구조에서도 최대한 그려줌
            dmol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
        except Exception:
            dmol = mol
        try:
            Draw.MolToFile(dmol, file_name, size=(300, 300), legend=legend)
            return True
        except Exception as e:
            print(f"[viz] skip frame (draw failed): {e}")
            return False

    def visualize_chain(self, path, nodes_list, adjacency_matrix, trainer=None):
        RDLogger.DisableLog('rdApp.*')
        # convert graphs to the rdkit molecules
        mols = [self.mol_from_graphs(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]
        mols = [m for m in mols if m is not None and m.GetNumAtoms() > 0]  # ← 유효한 프레임만 남김
        if not mols:
            print("[viz] no valid molecules to draw; skip chain")
            return []
        # find the coordinates of atoms in the final molecule
        final_molecule = mols[-1]
        AllChem.Compute2DCoords(final_molecule)

        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append((positions.x, positions.y, positions.z))

        # align all the molecules
        for i, mol in enumerate(mols):
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))

        # draw gif
        save_paths = []
        # num_frams = nodes_list.shape[0]

        for frame, m in enumerate(mols):  # 유효 프레임만 사용
            file_name = os.path.join(path, f'fram_{frame}.png')
            if self._draw_mol_safe(m, file_name, legend=f"Frame {frame}"):
                save_paths.append(file_name)

        if not save_paths:
            print("[viz] no drawable frames; skip gif")
            return mols

        def _safe_imread(p):
            try:
                return imageio.imread(p)
            except Exception as e:
                print(f"[viz] skip broken frame {p}: {e}")
                return None

        imgs = [im for im in (_safe_imread(fn) for fn in save_paths) if im is not None]
        if not imgs:
            print("[viz] no readable frames; skip gif")
            return mols

        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)

        if wandb.run:
            print(f"Saving {gif_path} to wandb")
            wandb.log({"chain": wandb.Video(gif_path, fps=5, format="gif")}, commit=True)

        # draw grid image
        # --- draw grid image (safe) ---
        try:
            # RDKit의 그리드 함수가 케쿨화/SMILES 경로를 타며 터질 수 있으므로
            # 먼저 kekulize=False로 준비된 mol 리스트를 만들어 시도
            safe_mols = []
            for m in mols:
                try:
                    dm = rdMolDraw2D.PrepareMolForDrawing(m, kekulize=False)
                except Exception:
                    dm = m
                safe_mols.append(dm)

            img = Draw.MolsToGridImage(safe_mols, molsPerRow=10, subImgSize=(200, 200))
            out_grid = os.path.join(path, f"{path.split('/')[-1]}_grid_image.png")
            img.save(out_grid)
        except Exception as e:
            # 여전히 실패하면 그리드 이미지는 건너뜁니다(체인 GIF/패널은 계속 진행)
            print(f"[viz] grid draw failed: {e} -- skip grid")

    def _draw_mol_safe(self, mol, file_name, legend=None):
        """케쿨화/SMILES 의존을 피해서 실패 없이 그리기"""
        try:
            dmol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
        except Exception:
            dmol = mol
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
            drawer.DrawMolecule(dmol, legend=legend or "")
            drawer.FinishDrawing()
            with open(file_name, "wb") as f:
                f.write(drawer.GetDrawingText())
            return True
        except Exception as e:
            print(f"[viz] skip frame (draw failed): {e}")
            return False

    def visualize_panel(self, path: str, gen_atom_edge_pair, cond_y: np.ndarray,
                            spec_len: int | None = None, ref_mol: Chem.Mol | None = None,
                            file_name: str = "panel.png", title: str | None = None, log='graph'):
        """
        gen_atom_edge_pair: tuple (atom_types_tensor, edge_types_tensor) like items of molecule_list
        cond_y: numpy array of condition vector used for generation (spectrum + globals)
        spec_len: length of spectrum part in cond_y. If None, try to infer (use full y)
        ref_mol: RDKit Mol of the reference/original molecule (optional)
        """

        os.makedirs(path, exist_ok=True)

        # 1) 분자 그림(PIL Image) 준비
        gen_mol = self.mol_from_graphs(gen_atom_edge_pair[0].numpy(), gen_atom_edge_pair[1].numpy())
        gen_img = Draw.MolToImage(gen_mol, size=(350, 350)) if gen_mol is not None else None
        ref_img = Draw.MolToImage(ref_mol, size=(350, 350)) if ref_mol is not None else None

        # 2) 스펙트럼/조건 분리
        y = np.asarray(cond_y).reshape(-1)
        if spec_len is None or spec_len <= 0 or spec_len > y.shape[0]:
            spec_len = y.shape[0]  # 안전빵: 전체를 스펙처럼 보여줌
        y_spec = y[:spec_len]
        y_rest = y[spec_len:]  # 글로벌 조건 숫자들(있다면)

        # 3) 패널 구성
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        ax0, ax1, ax2 = axes

        # (왼쪽) 스펙트럼 + 글로벌 조건 요약
        ax0.plot(np.arange(len(y_spec)), y_spec)
        ax0.set_title("Spectrum")
        ax0.set_xlabel("bin")
        ax0.set_ylabel("intensity")
        if y_rest.size > 0:
            txt = "\n".join([f"g[{i}]={v:.3g}" for i, v in enumerate(y_rest[:12])])  # 최대 12개만
            ax0.text(0.02, 0.98, txt, transform=ax0.transAxes, va="top", ha="left")

        # (가운데) 원본
        ax1.axis('off')
        ax1.set_title("Original")
        if ref_img is not None:
            ax1.imshow(ref_img)

        # (오른쪽) 생성 결과
        ax2.axis('off')
        ax2.set_title("Generated")
        if gen_img is not None:
            ax2.imshow(gen_img)

        if title:
            fig.suptitle(title, y=1.02)

        fig.tight_layout()
        out_path = os.path.join(path, file_name)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        # wandb 로깅(선택)
        #if wandb.run and log is not None:
        #    wandb.log({log: wandb.Image(out_path)}, commit=True)


class NonMolecularVisualization:
    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()

        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

        rows, cols = np.where(adjacency_matrix >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adjacency_matrix[edge[0]][edge[1]]
            graph.add_edge(edge[0], edge[1], color=float(edge_type), weight=3 * edge_type)

        return graph

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=100, largest_component=False):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        # Set node colors based on the eigenvectors
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        plt.figure()
        nx.draw(graph, pos, font_size=5, node_size=node_size, with_labels=False, node_color=U[:, 1],
                cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, edge_color='grey')

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, 'graph_{}.png'.format(i))
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)
            im = plt.imread(file_path)
            if wandb.run and log is not None:
                wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_chain(self, path, nodes_list, adjacency_matrix):
        # convert graphs to networkx
        graphs = [self.to_networkx(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]
        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=0)

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            self.visualize_non_molecule(graph=graphs[frame], pos=final_pos, path=file_name)
            save_paths.append(file_name)

        def _safe_imread(p):
            try:
                return imageio.imread(p)
            except Exception as e:
                print(f"[viz] skip broken frame {p}: {e}")
                return None

        imgs = [im for im in (_safe_imread(fn) for fn in save_paths) if im is not None]
        if not imgs:
            print("[viz] no readable frames; skip gif")
            return graphs

        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)

        if wandb.run:
            wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})

        # --- Add to src/analysis/visualization.py (inside MolecularVisualization) ---



