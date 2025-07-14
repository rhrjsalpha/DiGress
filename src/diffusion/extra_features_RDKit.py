'''
This file contains the RDKit-based feature extractor for molecular graphs.
It is designed to handle "broken" molecules that can appear during the diffusion process.
'''
import torch
from rdkit import Chem
import numpy as np

from src import utils


class RDKitExtraFeatures:
    def __init__(self, dataset_info):
        """
        Initialize the RDKit feature extractor.
        :param dataset_info: A class containing information about the dataset, 
                             including atom type mappings (atom_decoder) and max_n_nodes.
        """
        self.dataset_info = dataset_info
        # Create a reverse mapping from atom index to atomic symbol string
        self.atom_decoder = {v: k for k, v in dataset_info.atom_decoder.items()}

    def __call__(self, noisy_data):
        """
        Compute RDKit-based features for a batch of noisy molecular graphs.
        :param noisy_data: A dictionary containing the noisy graph tensors: X_t, E_t, node_mask.
        :return: A PlaceHolder object with the computed features.
        """
        X, E, node_mask = noisy_data['X_t'], noisy_data['E_t'], noisy_data['node_mask']

        # Move tensors to CPU and convert to numpy for RDKit processing
        X_np = X.cpu().numpy()
        E_np = E.cpu().numpy()
        node_mask_np = node_mask.cpu().numpy()

        batch_x_features = []
        batch_e_features = []

        # Iterate over each graph in the batch
        for i in range(X_np.shape[0]):
            n_nodes = int(node_mask_np[i].sum())
            if n_nodes == 0:
                # Handle empty graph
                x_features = np.zeros((0, 6), dtype=np.float32) # 6 is the number of node features
                e_features = np.zeros((0, 0, 1), dtype=np.float32) # 1 is the number of edge features
            else:
                atom_types = np.argmax(X_np[i, :n_nodes], axis=-1)
                edge_types = np.argmax(E_np[i, :n_nodes, :n_nodes], axis=-1)
                x_features, e_features = self._get_features_from_graph(atom_types, edge_types)

            # Pad features to the maximum number of nodes for batching
            padded_x = np.zeros((self.dataset_info.max_n_nodes, x_features.shape[1]), dtype=np.float32)
            if n_nodes > 0:
                padded_x[:n_nodes] = x_features
            batch_x_features.append(padded_x)

            padded_e = np.zeros((self.dataset_info.max_n_nodes, self.dataset_info.max_n_nodes, e_features.shape[2]), dtype=np.float32)
            if n_nodes > 0:
                padded_e[:n_nodes, :n_nodes] = e_features
            batch_e_features.append(padded_e)

        # Convert lists of numpy arrays back to PyTorch tensors
        final_x_features = torch.from_numpy(np.array(batch_x_features)).float().to(X.device)
        final_e_features = torch.from_numpy(np.array(batch_e_features)).float().to(E.device)

        # Return as a PlaceHolder, y is empty as these are node/edge features
        return utils.PlaceHolder(X=final_x_features, E=final_e_features, y=torch.zeros((X.shape[0], 0)).to(X.device))

    def _get_features_from_graph(self, atom_types, edge_types):
        """
        Constructs an RDKit molecule from raw atom and edge types and extracts features.
        Handles "broken" molecules gracefully.
        """
        rw_mol = Chem.RWMol()
        node_to_idx = {}

        # Add atoms
        for i, atom_type_idx in enumerate(atom_types):
            atom_symbol = self.atom_decoder.get(atom_type_idx, 'C')  # Default to Carbon if not found
            atom = Chem.Atom(atom_symbol)
            if atom_type_idx == 0:  # Assuming 0 is the mask token
                atom.SetBoolProp("is_masked", True)

            idx = rw_mol.AddAtom(atom)
            node_to_idx[i] = idx

        # Add bonds
        bond_types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC} # Added AROMATIC
        for i in range(len(atom_types)):
            for j in range(i + 1, len(atom_types)):
                bond_type_idx = edge_types[i, j]
                if bond_type_idx in bond_types and bond_type_idx != 0: # Ensure bond_type_idx is not 0 (no bond)
                    try:
                        rw_mol.AddBond(node_to_idx[i], node_to_idx[j], bond_types.get(bond_type_idx))
                    except Exception:
                        # Handle cases where bond cannot be added (e.g., invalid valence)
                        pass

        # Sanitize molecule, but don't raise an error if it fails
        try:
            Chem.SanitizeMol(rw_mol)
        except Exception:
            pass

        # Extract node (atom) features
        x_features = []
        num_atoms = rw_mol.GetNumAtoms()
        if num_atoms > 0:
            for i in range(num_atoms):
                atom = rw_mol.GetAtomWithIdx(i)
                features = [
                    int(atom.GetHybridization()),
                    atom.GetFormalCharge(),
                    atom.GetTotalDegree(),
                    atom.GetTotalNumHs(includeNeighbors=True),
                    int(atom.GetIsAromatic()),
                    int(atom.IsInRing())
                ]
                x_features.append(features)
        
        if not x_features:
            x_features = np.zeros((len(atom_types), 6)) # 6 features

        # Extract edge (bond) features: Bond Type, Spatial Pos, In Ring, Adjacency
        num_nodes = len(atom_types)
        e_features = np.zeros((num_nodes, num_nodes, 4), dtype=np.float32) # 4 channels for edge features

        # Channel 0: Bond Type
        for bond in rw_mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            e_features[i, j, 0] = bond_type
            e_features[j, i, 0] = bond_type

        # Channel 1: Spatial Position (Shortest Path Distance)
        # Initialize with a large value, 0 for self-loops
        spatial_pos_matrix = np.full((num_nodes, num_nodes), num_nodes + 1, dtype=np.float32)
        np.fill_diagonal(spatial_pos_matrix, 0)

        if num_atoms > 0:
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    try:
                        path = Chem.GetShortestPath(rw_mol, rw_mol.GetAtomWithIdx(i), rw_mol.GetAtomWithIdx(j))
                        distance = len(path)
                        spatial_pos_matrix[i, j, 1] = distance
                        spatial_pos_matrix[j, i, 1] = distance
                    except Exception:
                        # No path found or invalid molecule, distance remains large
                        pass
        e_features[:, :, 1] = spatial_pos_matrix

        # Channel 2: In Ring (Binary)
        if num_atoms > 0:
            for bond in rw_mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                is_in_ring = int(bond.IsInRing())
                e_features[i, j, 2] = is_in_ring
                e_features[j, i, 2] = is_in_ring

        # Channel 3: Adjacency (Binary)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if num_atoms > 0:
            for bond in rw_mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        e_features[:, :, 3] = adj_matrix

        return np.array(x_features, dtype=np.float32), e_features
