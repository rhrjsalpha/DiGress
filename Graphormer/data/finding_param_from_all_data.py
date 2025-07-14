import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdmolops, Draw
import time

def create_filtered_params_with_distribution(dataset_path, spatial_threshold=100):
    """
    Generate fixed parameters for model training, filter molecules with extreme spatial distances,
    plot distributions, and save removed molecules as images.

    Args:
        dataset_path (str): Path to the dataset.
        spatial_threshold (int): Maximum allowed spatial distance.

    Returns:
        dict: Fixed parameters for model configuration.
    """
    dataset = pd.read_csv(dataset_path)
    cols = dataset.columns
    output_size_count = sum(1 for col_name in cols if "ex" in col_name or "prob" in col_name)
    print("output_size_count:", output_size_count)

    # Calculate maximum atoms, edges, and spatial distances using RDKit
    smiles_list = dataset["smiles"]
    atom_counts = []
    edge_counts = []
    spatial_distances = []
    removed_smiles = []

    count = 0
    all = len(smiles_list)
    for smiles in smiles_list:
        count += 1
        print(f"{count}/{all}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            num_edges = len(rdmolops.GetAdjacencyMatrix(mol).nonzero()[0]) // 2
            dist_matrix = rdmolops.GetDistanceMatrix(mol)
            max_spatial = int(dist_matrix.max())

            if max_spatial <= spatial_threshold:
                atom_counts.append(num_atoms)
                edge_counts.append(num_edges)
                spatial_distances.append(max_spatial)
            else:
                removed_smiles.append(smiles)

    # Convert to Pandas Series for analysis
    atom_counts = pd.Series(atom_counts, name="num_atoms")
    edge_counts = pd.Series(edge_counts, name="num_edges")
    spatial_distances = pd.Series(spatial_distances, name="max_spatial")

    # Print summary statistics
    print("\n--- Data Summary After Filtering ---")
    print(atom_counts.describe())
    print(edge_counts.describe())
    print(spatial_distances.describe())
    print(f"\nRemoved {len(removed_smiles)} molecules with max_spatial > {spatial_threshold}")

    # Plot distributions after filtering
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(atom_counts, bins=30, color='blue', alpha=0.7)
    plt.xlabel("Number of Atoms")
    plt.ylabel("Frequency")
    plt.title("Filtered Distribution of Number of Atoms")

    plt.subplot(1, 3, 2)
    plt.hist(edge_counts, bins=30, color='green', alpha=0.7)
    plt.xlabel("Number of Edges")
    plt.ylabel("Frequency")
    plt.title("Filtered Distribution of Number of Edges")

    plt.subplot(1, 3, 3)
    plt.hist(spatial_distances, bins=30, color='red', alpha=0.7)
    plt.xlabel("Max Spatial Distance")
    plt.ylabel("Frequency")
    plt.title("Filtered Distribution of Max Spatial Distance")

    plt.tight_layout()
    plt.show()

    # Save removed molecules as images
    if removed_smiles:
        removed_mols = [Chem.MolFromSmiles(sm) for sm in removed_smiles if Chem.MolFromSmiles(sm)]
        img = Draw.MolsToGridImage(removed_mols[:], molsPerRow=5, subImgSize=(300, 300))
        img.save("removed_molecules.png")
        print(f"Saved removed molecules image as 'removed_molecules.png'")

    # Fixed parameters with dataset-dependent values
    fixed_params = {
        "num_atoms": int(atom_counts.max()),
        "num_edges": int(edge_counts.max()),
        "num_spatial": int(spatial_distances.max() + 10),
        "output_size": int(output_size_count),
    }
    return fixed_params


# 실행 예시
before_time = time.time()
fixed_params = create_filtered_params_with_distribution("all_data_2.csv", spatial_threshold=10000)
print(fixed_params)
after_time = time.time()
print(after_time-before_time)

