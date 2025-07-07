from bayes_opt import BayesianOptimization
import pandas as pd
from Graphormer.GP5.models.graphormer_train_default import train_model
from Graphormer.GP5.models.graphormer_CV_train_default import cross_validate_model
from rdkit import Chem
from rdkit.Chem import rdmolops

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
            "ffn_embedding_dim": (128, 512),  # Example range
            "num_attention_heads": (4, 16),  # Example range
            "dropout": (0.0, 0.5),  # Example range
            "attention_dropout": (0.0, 0.5),  # Example range
            "activation_dropout": (0.0, 0.5),  # Example range
            "weight_ex": (0, 1),  # 가중치
            "batch_size": (2, 128),  # 배치 크기
            "learning_rate": (0.0001, 0.01),  # 학습률
        }

    return param_bounds, fixed_params


def bayesian_optimization_param_grid(
    config_combinations,
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
        config_combinations (list): Model configuration.
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

    # Define the objective function for Bayesian Optimization
    def objective_function(**params):
        # 고정된 값들을 fixed_params에서 가져와 적용
        params.update(fixed_params)

        # Convert float params to int if necessary (최적화 변수에만 해당)
        params["batch_size"] = int(params["batch_size"])
        params["n_pairs"] = int(params["n_pairs"])

        if CV:
            # Perform cross-validation
            result = cross_validate_model(
                config=config_combinations[0],  # Assuming only one config
                target_type=params["target_type"],  # Fixed value from fixed_params
                loss_function=params["loss_function"],
                weight_ex=params["weight_ex"],
                num_epochs=num_epochs,
                batch_size=params["batch_size"],
                n_pairs=params["n_pairs"],
                learning_rate=params["learning_rate"],
                dataset_path=dataset_path,
                testset_path=testset_path,
            )
        else:
            # Perform single training and evaluation
            result = train_function(
                config=config_combinations[0],  # Assuming only one config
                target_type=params["target_type"],  # Fixed value from fixed_params
                loss_function=params["loss_function"],
                weight_ex=params["weight_ex"],
                num_epochs=num_epochs,
                batch_size=params["batch_size"],
                n_pairs=params["n_pairs"],
                learning_rate=params["learning_rate"],
                dataset_path=dataset_path,
            )

        # Use validation loss as the target metric to minimize
        val_loss = result["val_loss_avg"] if "val_loss_avg" in result else float("inf")
        results.append({**params, "val_loss": val_loss})
        return -val_loss  # Negate because Bayesian Optimization maximizes by default

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
    return results_df


param_bounds, fixed_params = config_maker(dataset_path="../data/train_50.csv")
print(param_bounds)

results_df = bayesian_optimization_param_grid(
    config_combinations=[fixed_params],  # 고정된 값 포함
    param_bounds=param_bounds,          # 최적화할 하이퍼파라미터
    train_function=train_model,
    dataset_path="../data/train_50.csv",
    testset_path="../data/test_50.csv",
    num_epochs=10,
    verbose=True,
    CV=True,
    init_points=5,
    n_iter=25,
)

results_df.to_csv("../GridSearch/BayesianOptimization_fixed_values.csv", index=False)

## 고정시킬 것들
##### 최적화할 것들


#            "activation_fn": ["gelu"],  # 활성화 함수 ("gelu", "relu" 등) ->

#   "loss_function": ["MSE", "MAE", "SoftDTW", "Huber", "SID"],  # 단일 손실 함수 적용

