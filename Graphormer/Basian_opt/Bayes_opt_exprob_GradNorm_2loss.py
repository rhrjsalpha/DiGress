from bayes_opt import BayesianOptimization
import pandas as pd
from itertools import product
from rdkit import Chem
from rdkit.Chem import rdmolops
import time
#from Graphormer.Basian_opt.Bayes_opt_default_nm import total_grid_count
from Graphormer.GP5.models.graphormer_train_eVOsc_GradNorm_2loss import train_model_ex_porb
from Graphormer.GP5.models.graphormer_CV_train_eVOsc_GradNorm_2loss import cross_validate_model
from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset

def create_fixed_params(dataset_path, combination, dynamic = True):
    """
    Generate fixed parameters for model training with a given combination.

    Args:
        dataset_path (str): Path to the dataset.
        combination (dict): Fixed parameter combination.

    Returns:
        dict: Fixed parameters for model configuration.
    """
    dataset = pd.read_csv(dataset_path)
    cols = dataset.columns
    output_size_count = sum(1 for col_name in cols if "ex" in col_name or "prob" in col_name)
    print("output_size_count",output_size_count)
    # Calculate maximum atoms, edges, and spatial distances using RDKit
    smiles_list = dataset["smiles"]
    max_atoms, max_edges, max_spatial = 0, 0, 0
    if dynamic == True:
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
    else:
        max_atoms, max_edges, max_spatial = 100, 100, 49

    # Fixed parameters with dataset-dependent values
    fixed_params = {
        "num_atoms": int(max_atoms),
        "num_edges": int(max_edges),
        "num_spatial": int(max_spatial + 1),
        "output_size": int(output_size_count),
        **combination,  # Use the provided combination
    }
    return fixed_params


def bayesian_optimization_param_grid(
        param_bounds,
        train_function,
        dataset_path,
        testset_path,
        fixed_params,
        verbose=True,
        CV=True,
        init_points=5,
        n_iter=25,
        total_grid_count=0,
        grid_count=0,
        DATASET = None,
        TEST_DATASET = None
):
    """
    Perform Bayesian Optimization over the provided parameter bounds.

    Args:
        param_bounds (dict): Bounds for hyperparameters to optimize.
        train_function (function): Function to train and evaluate the model.
        dataset_path (str): Path to the training dataset.
        testset_path (str): Path to the test dataset.
        fixed_params (dict): Fixed parameter combination.
        num_epochs (int): Number of epochs for training.
        verbose (bool): If True, print progress and results.
        CV (bool): If True, perform cross-validation.
        init_points (int): Number of random points to explore initially.
        n_iter (int): Number of optimization iterations.

    Returns:
        pd.DataFrame: Results with optimized hyperparameters.
    """
    results = []

    global iteration_counter
    iteration_counter = 0

    def objective_function(**params):
        # Merge fixed and optimizable parameters
        merged_params = {**fixed_params, **params}
        config_params = {}

        start_time = time.time()
        global iteration_counter
        iteration_counter += 1
        print(f"▶ Running iteration {iteration_counter}/{init_points + n_iter}, fixed_param started {grid_count}/{total_grid_count}")

        merged_params["epoch"] = int(round(merged_params["epoch"]))

        # Ensure integers for specific parameters
        for key in [
            "num_atoms", "num_in_degree", "num_out_degree", "num_edges", "num_spatial",
            "multi_hop_max_dist", "num_encoder_layers", "embedding_dim", "ffn_embedding_dim",
            "num_attention_heads", "qn_block_size"
        ]:
            if key in merged_params:
                config_params[key] = int(round(merged_params[key]))
            else:
                config_params[key] = merged_params.get(key)

        merged_params["batch_size"] = int(round(merged_params["batch_size"]))
        merged_params["n_pairs"] = int(round(merged_params["n_pairs"]))
        merged_params["num_edge_dis"] = int(round(merged_params["num_edge_dis"]))

        # Adjust embedding_dim to be divisible by num_attention_heads
        if config_params["embedding_dim"] % config_params["num_attention_heads"] != 0:
            config_params["embedding_dim"] += (
                    config_params["num_attention_heads"] - (
                        config_params["embedding_dim"] % config_params["num_attention_heads"])
            )

        # Add remaining parameters
        for key in [
            "dropout", "attention_dropout", "activation_dropout", "activation_fn",
            "pre_layernorm", "q_noise", "num_edge_dis", "edge_type", "output_size"
        ]:
            if key in merged_params:
                config_params[key] = merged_params[key]

        if merged_params["target_type"] == "default_nm":
            merged_params["target_type"] = "default"

        config_params["activation_fn"] = activation_fn_selector(int(round(config_params["activation_fn"])))
        merged_params["activation_fn"] = activation_fn_selector(int(round(merged_params["activation_fn"])))

        # Perform cross-validation or training
        if CV:
            print("merged_params[batch_size]",merged_params["batch_size"])
            result = cross_validate_model(
                config=config_params,
                target_type=merged_params["target_type"],
                loss_function=merged_params["loss_function"],
                loss_function_ex=merged_params["loss_function_ex"],
                loss_function_prob=merged_params["loss_function_prob"],
                #weight_ex=merged_params["weight_ex"],
                num_epochs=merged_params["epoch"],
                batch_size=merged_params["batch_size"],
                n_pairs=merged_params["n_pairs"],
                learning_rate=merged_params["learning_rate"],
                dataset_path=dataset_path,
                testset_path=testset_path,
                DATASET=DATASET,
                TEST_DATASET = TEST_DATASET,
                alpha = merged_params["gradnorm_alpha"],
                n_splits=5
            )
        else:
            result = train_function(
                config=config_params,
                target_type=merged_params["target_type"],
                loss_function=merged_params["loss_function"],
                loss_function_ex=merged_params["loss_function_ex"],
                loss_function_prob=merged_params["loss_function_prob"],
                #weight_ex=merged_params["weight_ex"],
                num_epochs=merged_params["epoch"],
                batch_size=merged_params["batch_size"],
                n_pairs=merged_params["n_pairs"],
                learning_rate=merged_params["learning_rate"],
                dataset_path=dataset_path,
                DATASET=DATASET,
                TEST_DATASET = TEST_DATASET
            )

        # Store results
        val_loss = result.get("val_loss_avg")
        results.append({**config_params, **merged_params, **result, "val_loss": val_loss})

        lossfn_ex = merged_params["loss_function_ex"]
        lossfn_prob = merged_params["loss_function_prob"]



        if iteration_counter >= init_points + n_iter:
            res_df = pd.DataFrame(results)
            res_df.to_csv(f"BayesianOptimization_default_exprob_{grid_count}_iter{iteration_counter}_{lossfn_ex}_{lossfn_prob}.csv")
        else:
            print("intermediate results save")
            res_df = pd.DataFrame(results)
            res_df.to_csv(
                f"BayesianOptimization_Intermediate_default_exprob_{grid_count}_{lossfn_ex}_{lossfn_prob}.csv")

        end_time = time.time()
        full_time = end_time-start_time
        print(
            f"▶ total_time {full_time},finish iteration {iteration_counter}/{init_points + n_iter}, fixed_param started {grid_count}/{total_grid_count}, {lossfn_ex},{lossfn_prob}, {val_loss}")

        return -val_loss  # Bayesian Optimization maximizes, so we negate the loss

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=param_bounds,
        verbose=verbose,
        random_state=42,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return pd.DataFrame(results)

def activation_fn_selector(activation_num:int):
    if activation_num == 1:
        return "relu"
    elif activation_num == 2:
        return "leaky_relu"
    elif activation_num == 3:
        return "gelu"
    elif activation_num == 4:
        return "swish"
    else:
        print("Invalid loss function, defaulting to gelu")
        return "gelu"

# Define fixed parameter grid
fixed_param_grid = {
    "edge_type": ["multi_hop"],
    #"activation_fn": ["gelu"],
    "loss_function": ["MSE"],  # 전체 손실 함수 -> "ex_prob" 모드에서는 사용되지 않아서 실제로는 아무 의미 없음
    "loss_function_ex": ["SID","SoftDTW"],  # 'ex'에 대한 손실 함수 "SoftDTW", "MSE", "MAE", "Huber", "SID"
    "loss_function_prob": ["SID","SoftDTW"],  # 'prob'에 대한 손실 함수 "SoftDTW", "MSE", "MAE", "Huber", "SID"
    "pre_layernorm": [False],
    "q_noise": [0.0],
    "qn_block_size": [8],
    #"batch_size": [64],
    #"num_edge_dis": [10],
    "target_type": ["ex_prob"],
    #"n_pairs": [5],
}

# Generate all combinations of fixed parameters
grid_combinations = list(product(*fixed_param_grid.values()))
grid_keys = list(fixed_param_grid.keys())
print(len(grid_keys),grid_keys)
grid_dicts = [dict(zip(grid_keys, values)) for values in grid_combinations]
print(len(grid_dicts),grid_dicts)
time.sleep(10)
# Define hyperparameter bounds for optimization
param_bounds = {
    "num_in_degree": (5, 15),
    "num_out_degree": (5, 15),
    "multi_hop_max_dist": (3, 7),
    "num_encoder_layers": (4, 8),
    "dropout": (0.0, 0.5),
    "attention_dropout": (0.0, 0.5),
    "activation_dropout": (0.0, 0.5),
    #"weight_ex": (0.1, 0.9),
    "learning_rate": (0.0001, 0.01),
    "embedding_dim": (16, 128),
    "ffn_embedding_dim": (128, 512),
    "num_attention_heads": (4, 16),
    "epoch": (3, 5),
    "batch_size": (32, 70),
    "num_edge_dis": (7, 20),
    "activation_fn": (0.5, 4.5),
    "n_pairs": (50, 50),
    "gradnorm_alpha": (0.1, 1.0),
}

# Run Grid Search + Bayesian Optimization

dataset = SMILESDataset(csv_file="../data/train_50.csv", attn_bias_w=1.0, target_type=fixed_param_grid["target_type"][0])
test_dataset = SMILESDataset(csv_file="../data/train_50.csv", attn_bias_w=1.0, target_type=fixed_param_grid["target_type"][0])

all_results = []
total_grid_count = len(grid_dicts)
grid_count = 0
for combination in grid_dicts:
    grid_count += 1
    print(f"Running Bayesian Optimization for fixed parameters: {combination}")

    # Generate dataset-dependent fixed params
    fixed_params = create_fixed_params("../data/train_100.csv", combination, dynamic=False)
    print("fixed_params", fixed_params)

    # Run Bayesian Optimization
    results_df = bayesian_optimization_param_grid(
        param_bounds=param_bounds,
        train_function=train_model_ex_porb,
        dataset_path="../data/train_100.csv",
        testset_path="../data/train_100.csv",
        fixed_params=fixed_params,
        verbose=True,
        CV=True,
        init_points=2,
        n_iter=2,
        total_grid_count=total_grid_count,
        grid_count=grid_count,
        DATASET = dataset,
        TEST_DATASET = test_dataset
    )

    all_results.append(results_df)

n_pair=param_bounds["n_pairs"]
# Concatenate all results
final_results_df = pd.concat(all_results, ignore_index=True)
final_results_df.to_csv(f"BayesianOptimization_exporb_eVOsc_2loss_{n_pair}.csv", index=False)