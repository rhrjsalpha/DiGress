from bayes_opt import BayesianOptimization
import pandas as pd
from Graphormer.GP5.models.graphormer_train_default import train_model
from Graphormer.GP5.models.graphormer_CV_train_default import cross_validate_model
from rdkit import Chem
from rdkit.Chem import rdmolops
from itertools import product

def config_maker(dataset_path, param_bounds=None):
    """
    Generate configuration combinations for model training.

    Args:
        dataset_path (str): Path to the dataset.
        param_bounds (dict, optional): Hyperparameter bounds for Bayesian Optimization.

    Returns:
        dict: Configuration dictionary for Bayesian Optimization.
    """
    dataset = pd.read_csv(dataset_path)
    cols = dataset.columns
    output_size_count = sum(1 for col_name in cols if "ex" in col_name or "prob" in col_name)

    # Calculate maximum atoms, edges, and spatial distances using RDKit
    smiles_list = dataset["smiles"]
    max_atoms = 0
    max_edges = 0
    max_spatial = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            max_atoms = max(max_atoms, num_atoms)

            num_edges = len(rdmolops.GetAdjacencyMatrix(mol).nonzero()[0]) // 2
            max_edges = max(max_edges, num_edges)

            dist_matrix = rdmolops.GetDistanceMatrix(mol)
            max_spatial = max(max_spatial, dist_matrix.max())
            max_spatial = int(max_spatial)

    # Define fixed values for non-optimizable parameters
    fixed_params = {
        "num_atoms": max_atoms,  # Fixed maximum number of atoms
        "num_edges": max_edges,  # Fixed maximum number of edges
        "num_spatial": max_spatial + 1,  # Fixed maximum spatial encoding value
        "output_size": output_size_count,  # Fixed output size
        "edge_type": "multi_hop",
        "activation_fn": "gelu",
        "pre_layernorm": False,
        "q_noise": 0.0,
        "qn_block_size": 8,
        "batch_size": 64,
        "num_edge_dis": 10,
        "target_type": "default",
        "n_pairs" : 5,
    }

    # Define hyperparameter bounds for optimization
    if param_bounds is None:
        param_bounds = {
            "num_in_degree": (5, 15),  # Example range
            "num_out_degree": (5, 15),  # Example range
            "multi_hop_max_dist": (3, 7),  # Example range
            "num_encoder_layers": (4, 8),  # Example range
            "embedding_dim": (32, 128),  # Example range
            "ffn_embedding_dim": (128, 512, 16),  # Example range
            "num_attention_heads": (4, 16, 4),  # Example range
            "dropout": (0.0, 0.5),  # Example range
            "attention_dropout": (0.0, 0.5),  # Example range
            "activation_dropout": (0.0, 0.5),  # Example range
            "weight_ex": (0, 1),  # 가중치
            "batch_size": (2, 128),  # 배치 크기
            "learning_rate": (0.0001, 0.01),  # 학습률
        }

    return param_bounds, fixed_params

def fixed_param_grid(fixed_param_grid=None):
    # Grid Search용 Fixed Parameters 조합
    fixed_param_grid = {
        "edge_type": ["multi_hop"],
        "activation_fn": ["gelu"],
        "loss_function" : ["SoftDTW"],
        "pre_layernorm": [False],
        "q_noise": [0.0],
        "qn_block_size": [8, 16],
        "batch_size": [int(64)],
        "num_edge_dis": [int(10)],
        "target_type": ["default"],
        "n_pairs": [int(5)],
    }


    keys = list(fixed_param_grid.keys())
    #print(fixed_param_grid.values())
    #print(*fixed_param_grid.values())
    # zip -> 튜플 (key, value)
    grids = [dict(zip(keys, values)) for values in product(*fixed_param_grid.values())]
    return grids

def create_fixed_params(dataset_path):
    """
    Generate fixed parameters for model training.

    Args:
        dataset_path (str): Path to the dataset.

    Returns:
        dict: Fixed parameters for model configuration.
    """
    dataset = pd.read_csv(dataset_path)
    cols = dataset.columns
    output_size_count = sum(1 for col_name in cols if "ex" in col_name or "prob" in col_name)

    # Calculate maximum atoms, edges, and spatial distances using RDKit
    smiles_list = dataset["smiles"]
    max_atoms = 0
    max_edges = 0
    max_spatial = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            num_atoms = mol.GetNumAtoms()
            max_atoms = max(max_atoms, num_atoms)

            num_edges = len(rdmolops.GetAdjacencyMatrix(mol).nonzero()[0]) // 2
            max_edges = max(max_edges, num_edges)

            dist_matrix = rdmolops.GetDistanceMatrix(mol)
            max_spatial = max(max_spatial, dist_matrix.max())
            max_spatial = int(max_spatial)

    # Fixed parameters
    fixed_params = {
        "num_atoms": int(max_atoms),
        "num_edges": int(max_edges),
        "num_spatial": int(max_spatial + 1),
        "output_size": int(output_size_count),
        "edge_type": "multi_hop",
        "activation_fn": "gelu",
        "loss_function" : "SoftDTW",
        "pre_layernorm": False,
        "q_noise": 0.0,
        "qn_block_size": int(8),
        "batch_size": int(64),
        "num_edge_dis": int(10),
        "target_type": "default",
        "n_pairs": int(5),
    }
    return fixed_params

def bayesian_optimization_param_grid(
    param_bounds,
    train_function,
    dataset_path,
    testset_path,
    num_epochs=10,
    verbose=True,
    CV=True,
    init_points=5,
    n_iter=25,
):
    """
    Perform Bayesian Optimization over the provided parameter bounds.

    Args:
        param_bounds (dict): Bounds for hyperparameters to optimize.
        train_function (function): Function to train and evaluate the model.
        dataset_path (str): Path to the training dataset.
        testset_path (str): Path to the test dataset.
        num_epochs (int): Number of epochs for training.
        verbose (bool): If True, print progress and results.
        CV (bool): If True, perform cross-validation.
        init_points (int): Number of random points to explore initially.
        n_iter (int): Number of optimization iterations.

    Returns:
        pd.DataFrame: Results with optimized hyperparameters.
    """
    results = []

    def objective_function(**params):
        # Add fixed parameters
        fixed_params = create_fixed_params(dataset_path)
        merged_params = {**fixed_params, **params}  # Merge fixed and optimizable parameters
        config_params = {}
        # Ensure integers for parameters where required
        for key in [
            "num_atoms",  # 분자의 최대 원자 수 (그래프의 노드 개수) 100
            "num_in_degree",  # 그래프 노드의 최대 in-degree
            "num_out_degree",  # 그래프 노드의 최대 out-degree
            "num_edges",  # 그래프의 최대 엣지 개수 50
            "num_spatial",  # 공간적 위치 인코딩을 위한 최대 값 default 100
            "multi_hop_max_dist",  # Multi-hop 엣지의 최대 거리
            "num_encoder_layers",  # Graphormer 모델에서 사용할 인코더 레이어 개수
            "embedding_dim",  # 임베딩 차원 크기 (노드, 엣지 등)
            "ffn_embedding_dim",  # Feedforward Network의 임베딩 크기
            "num_attention_heads",  # Multi-head Attention에서 헤드 개수
            "qn_block_size",  # Quantization block 크기
        ]:
            if key in merged_params:
                config_params[key] = int(round(merged_params[key]))  # Convert to integer
            else:
                config_params[key] = merged_params.get(key)

        if config_params["embedding_dim"] % config_params["num_attention_heads"] != 0:
            embedding_dim_changed = config_params["embedding_dim"]+(config_params["num_attention_heads"]-(config_params["embedding_dim"] % config_params["num_attention_heads"]))
            print("embedding_dim_changed",config_params["embedding_dim"],embedding_dim_changed, config_params["num_attention_heads"])
            config_params["embedding_dim"] = embedding_dim_changed
            if config_params["embedding_dim"] % config_params["num_attention_heads"] != 0:
                print("Invalid combination: embedding_dim must be divisible by num_attention_heads")

        # For parameters that do not require integer conversion
        for key in [
            "dropout",  # 드롭아웃 비율
            "attention_dropout",  # Attention 레이어의 드롭아웃 비율
            "activation_dropout",  # 활성화 함수 이후 드롭아웃 비율
            "activation_fn",  # 활성화 함수 ("gelu", "relu" 등)
            "pre_layernorm",  # LayerNorm을 Pre-Normalization으로 사용할지 여부
            "q_noise",  # Quantization noise (훈련 중 노이즈 추가를 위한 매개변수)
            "num_edge_dis",  # 엣지 거리 인코딩을 위한 최대 값
            "edge_type",  # 엣지 타입 ("multi_hop" 또는 다른 값 가능)
            "output_size",  # 모델 출력 크기
        ]:
            if key in merged_params:
                config_params[key] = merged_params[key]



        # Check embed_dim divisibility by num_heads
        if config_params["embedding_dim"] % config_params["num_attention_heads"] != 0:
            print("Invalid combination: embedding_dim must be divisible by num_attention_heads")
            return float("inf")  # Penalize invalid combinations

        # Perform cross-validation or training
        print("config_params",config_params)
        print(len(config_params))

        if CV:
            result = cross_validate_model(
                config=config_params,  # Use merged parameters as config
                target_type=merged_params["target_type"],  # Merged target type
                loss_function=merged_params["loss_function"],  # Merged loss function
                weight_ex=merged_params["weight_ex"],  # Optimized weight
                num_epochs=num_epochs,
                batch_size=merged_params["batch_size"],  # Merged batch size
                n_pairs=merged_params["n_pairs"],  # Merged number of pairs
                learning_rate=merged_params["learning_rate"],  # Optimized learning rate
                dataset_path=dataset_path,
                testset_path=testset_path,
            )
        else:
            result = train_function(
                config=config_params,  # Use merged parameters as config
                target_type=merged_params["target_type"],  # Merged target type
                loss_function=merged_params["loss_function"],  # Merged loss function
                weight_ex=merged_params["weight_ex"],  # Optimized weight
                num_epochs=num_epochs,
                batch_size=merged_params["batch_size"],  # Merged batch size
                n_pairs=merged_params["n_pairs"],  # Merged number of pairs
                learning_rate=merged_params["learning_rate"],  # Optimized learning rate
                dataset_path=dataset_path,
            )

        # Use validation loss as the target metric to minimize
        print("result",result)
        val_loss = result["val_loss_avg"] if "val_loss_avg" in result else float("inf")
        results.append({**merged_params, **result,"val_loss": val_loss})
        return -val_loss# Negate because Bayesian Optimization maximizes by default

    # Run Bayesian Optimization
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        verbose=verbose,  # Print status during optimization
        random_state=42,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # Collect results
    results_df = pd.DataFrame(results)
    print(results_df.columns)
    return results_df


# Example usage
param_bounds = {
    "num_in_degree": (5, 15),
    "num_out_degree": (5, 15),
    "multi_hop_max_dist": (3, 7),
    "num_encoder_layers": (4, 8),
    "dropout": (0.0, 0.5),
    "attention_dropout": (0.0, 0.5),
    "activation_dropout": (0.0, 0.5),
    "weight_ex": (0, 1),
    "learning_rate": (0.0001, 0.01),
    "embedding_dim": (16 ,128),
    "ffn_embedding_dim": (128, 512),
    "num_attention_heads": (4, 16),
}

results_df = bayesian_optimization_param_grid(
    param_bounds=param_bounds,
    train_function=train_model,
    dataset_path="../data/train_50.csv",
    testset_path="../data/test_50.csv",
    num_epochs=10,
    verbose=True,
    CV=True,
    init_points=3,
    n_iter=2,
)

results_df.to_csv("BayesianOptimization_with_merged_params.csv", index=False)