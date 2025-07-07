from skopt import gp_minimize
from skopt import forest_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import pandas as pd
from itertools import product
from rdkit import Chem
from rdkit.Chem import rdmolops
import time
from Graphormer.GP5.models.graphormer_train_eVOsc_GradNorm_2loss import train_model_ex_porb
from Graphormer.GP5.models.graphormer_CV_train_eVOsc_GradNorm_2loss import cross_validate_model
from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset

def CV_loss_fn_name_gen(loss_fn, ex_prob): ### CV_r2_ex
    if loss_fn == "SID":
        CV_col_name = f"CV_sid_{ex_prob}_avg"
    elif loss_fn == "SoftDTW":
        CV_col_name = f"CV_softdtw_{ex_prob}_avg"
    elif loss_fn == "MAE":
        CV_col_name = f"CV_mae_{ex_prob}_avg"
    elif loss_fn == "MSE":
        CV_col_name = f"CV_mse_{ex_prob}_avg"
    else:
        print("loss function name is incorrect")
    return CV_col_name

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

# ✅ 활성화 함수 선택 함수
def activation_fn_selector(activation_num):
    if activation_num == 1:
        return "relu"
    elif activation_num == 2:
        return "leaky_relu"
    elif activation_num == 3:
        return "gelu"
    elif activation_num == 4:
        return "swish"
    else:
        return "gelu"

# ✅ 하이퍼파라미터 탐색 범위 정의
space = [
    Integer(5, 15, name="num_in_degree"),
    Integer(5, 15, name="num_out_degree"),
    Integer(3, 7, name="multi_hop_max_dist"),
    Integer(4, 8, name="num_encoder_layers"),
    Real(0.0, 0.5, name="dropout"),
    Real(0.0, 0.5, name="attention_dropout"),
    Real(0.0, 0.5, name="activation_dropout"),
    Real(0.0001, 0.01, "log-uniform", name="learning_rate"),
    Integer(16, 128, name="embedding_dim"),
    Integer(128, 512, name="ffn_embedding_dim"),
    Integer(4, 16, name="num_attention_heads"),
    Integer(3, 5, name="epoch"),
    Integer(32, 70, name="batch_size"),
    Integer(7, 20, name="num_edge_dis"),
    Real(0.5, 4.5, name="activation_fn"),
    #Integer(50, 50, name="n_pairs"),
    Real(0.1, 1.0, name="gradnorm_alpha")
]

def bayesian_optimization_param_grid(
        dimension,
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
    results = []

    global iteration_counter
    iteration_counter = 0

    # ✅ 목적 함수 정의
    @use_named_args(space)
    def objective_function(**params):
        start_time = time.time()
        global iteration_counter
        iteration_counter += 1
        fixed_params = create_fixed_params(dataset_path, fixed_combination, dynamic=False)

        merged_params = {**fixed_params, **params}
        config_params = {}

        for key in [
            "num_atoms", "num_in_degree", "num_out_degree", "num_edges", "num_spatial",
            "multi_hop_max_dist", "num_encoder_layers", "embedding_dim", "ffn_embedding_dim",
            "num_attention_heads", "qn_block_size", "num_edge_dis"
        ]:
            if key in merged_params:
                config_params[key] = int(round(merged_params[key]))

        config_params["activation_fn"] = activation_fn_selector(int(round(merged_params["activation_fn"])))
        merged_params["batch_size"] = int(round(merged_params["batch_size"]))
        merged_params["n_pairs"] = int(round(merged_params["n_pairs"]))
        merged_params["num_edge_dis"] = int(round(merged_params["num_edge_dis"]))

        # Adjust embedding_dim to be divisible by num_attention_heads
        if config_params["embedding_dim"] % config_params["num_attention_heads"] != 0:
            config_params["embedding_dim"] += (
                    config_params["num_attention_heads"] - (
                    config_params["embedding_dim"] % config_params["num_attention_heads"]
            )
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

        print("Running with parameters:", config_params)

        result = cross_validate_model(
            config=config_params,
            target_type=merged_params["target_type"],
            loss_function=merged_params["loss_function"],
            loss_function_ex=merged_params["loss_function_ex"],
            loss_function_prob=merged_params["loss_function_prob"],
            num_epochs=merged_params["epoch"],
            batch_size=merged_params["batch_size"],
            n_pairs=merged_params["n_pairs"],
            learning_rate=merged_params["learning_rate"],
            dataset_path=dataset_path,
            testset_path=testset_path,
            DATASET=dataset,
            TEST_DATASET=test_dataset,
            alpha=merged_params["gradnorm_alpha"],
            n_splits=5
        )

        val_loss = result.get("val_loss_avg")
        results.append({**config_params, **merged_params, **result, "val_loss": val_loss})

        lossfn_ex = merged_params["loss_function_ex"]
        lossfn_prob = merged_params["loss_function_prob"]

        CV_col_ex = CV_loss_fn_name_gen(lossfn_ex, "ex")
        CV_col_prob = CV_loss_fn_name_gen(lossfn_prob, "prob")

        CV_loss_ex = result.get(CV_col_ex)
        CV_loss_prob = result.get(CV_col_prob)

        print(f"Loss: {lossfn_ex}: {CV_loss_ex:.5f}, {lossfn_prob}: {CV_loss_prob:.5f}")
        if iteration_counter == n_iter:
            print("full result saved")
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
            f"▶ total_time {full_time},finish iteration {iteration_counter}/{n_iter}, fixed_param started {grid_count}/{total_grid_count}, {lossfn_ex},{lossfn_prob}, {val_loss}")
        return [CV_loss_ex, CV_loss_prob]

    # ✅ 하이퍼파라미터 탐색 수행
    all_results = []
    for fixed_combination in grid_dicts:
        print(f"Running Bayesian Optimization for fixed parameters: {fixed_combination}")

        res_gp = forest_minimize(
            objective_function,
            dimensions=dimension,
            n_random_starts=init_points,
            n_calls=n_iter,
            random_state=42
        )

        # 결과 저장
        #print("res_gp",type(res_gp),res_gp)
        results = pd.DataFrame({
            'param': res_gp.x,
            'value': res_gp.fun
        })
        all_results.append(results)
        print("all_results",all_results)
    return all_results




# ✅ 고정 파라미터 설정
fixed_param_grid = {
    "edge_type": ["multi_hop"],
    "loss_function": ["MSE"],
    "loss_function_ex": ["SID",],
    "loss_function_prob": ["SID",],
    "pre_layernorm": [False],
    "q_noise": [0.0],
    "qn_block_size": [8],
    "target_type": ["ex_prob"],
    "n_pairs":[50]
}

grid_combinations = list(product(*fixed_param_grid.values()))
grid_keys = list(fixed_param_grid.keys())
grid_dicts = [dict(zip(grid_keys, values)) for values in grid_combinations]

dataset_path = "../data/train_50.csv"
testset_path = "../data/train_50.csv"

dataset = SMILESDataset(csv_file=dataset_path, attn_bias_w=1.0, target_type="ex_prob")
test_dataset = SMILESDataset(csv_file=testset_path, attn_bias_w=1.0, target_type="ex_prob")

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
    results = bayesian_optimization_param_grid(
        dimension=space,
        train_function=train_model_ex_porb,
        dataset_path="../data/train_100.csv",
        testset_path="../data/train_100.csv",
        fixed_params=fixed_params,
        verbose=True,
        CV=True,
        init_points=2, # 랜덤서치과정 15
        n_iter=4, # 전체탐색회수 (모델최적화 탐색은 niter-initpoints) 60
        total_grid_count=total_grid_count,
        grid_count=grid_count,
        DATASET = dataset,
        TEST_DATASET = test_dataset
    )
    print("all_res",all_results)
    all_results.append(results)

n_pair=fixed_param_grid["n_pairs"]
print("all_res",all_results)
# Concatenate all results
#final_results_df = pd.concat(all_results, ignore_index=True)

#final_results_df.to_csv(f"BayesianOptimization_exporb_eVOsc_2loss_{n_pair}.csv", index=False)


# ✅ 최종 결과 저장
#final_results_df = pd.concat(all_results, ignore_index=True)
#final_results_df.to_csv("BayesianOptimization_exporb_eVOsc_2loss.csv", index=False)

