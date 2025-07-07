import pandas as pd
import torch
from botorch.models import MultiTaskGP, SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
import time
from Graphormer.GP5.models.graphormer_CV_train_eVOsc_GradNorm_2loss import cross_validate_model
from rdkit import Chem
from rdkit.Chem import rdmolops
from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset
from botorch.models.transforms.outcome import Standardize

def df_to_train_X(df, param_bounds):
    keys = list(param_bounds.keys())
    norm_values = []
    for i, row in df.iterrows():
        values = []
        for key in keys:
            min_v, max_v = param_bounds[key]
            val = row[key]
            norm = (val - min_v) / (max_v - min_v)
            values.append(norm)
        norm_values.append(values)
    print(norm_values)
    return torch.tensor(norm_values, dtype=torch.double)

def custom_objective(samples: torch.Tensor) -> torch.Tensor:
    ev_mae = samples[..., 0]
    osc_mse = samples[..., 1]
    return torch.stack([ev_mae, osc_mse], dim=-1)

def create_fixed_params(dataset_path, combination, dynamic=True):
    dataset = pd.read_csv(dataset_path)
    output_size_count = sum(1 for col in dataset.columns if "ex" in col or "prob" in col)

    smiles_list = dataset["smiles"]
    max_atoms, max_edges, max_spatial = 0, 0, 0

    if dynamic:
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                num_atoms = mol.GetNumAtoms()
                max_atoms = max(max_atoms, num_atoms)

                num_edges = len(rdmolops.GetAdjacencyMatrix(mol).nonzero()[0]) // 2
                max_edges = max(max_edges, num_edges)

                dist_matrix = rdmolops.GetDistanceMatrix(mol)
                max_spatial = max(max_spatial, dist_matrix.max())
    else:
        max_atoms, max_edges, max_spatial = 100, 100, 49

    fixed_params = {
        "num_atoms": int(max_atoms),
        "num_edges": int(max_edges),
        "num_spatial": int(max_spatial + 1),
        "output_size": int(output_size_count),
        **combination,
    }
    return fixed_params

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

class MultiTask_qLogNoisyExpectedHypervolumeImprovement(qLogNoisyExpectedHypervolumeImprovement):
    def __init__(self, model, ref_point: torch.Tensor, X_baseline: torch.Tensor, sampler=None, **kwargs):
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        #objective = GenericMCObjective(custom_objective)
        #print("kwargs",kwargs)
        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            #objective=objective,
            **kwargs
        )

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

def objective_function_multitask(params, fixed_params, dataset_path, testset_path, DATASET, TEST_DATASET):

    # 리스트를 값으로 변환
    for key in fixed_params:
        if isinstance(fixed_params[key], list):
            fixed_params[key] = fixed_params[key][0]

    config_params = {**fixed_params, **params}

    #print("before", config_params)

    # ✅ 매핑 제거 (이미 train_X에서 매핑됨)
    config_params["num_in_degree"] = int(round(float(params["num_in_degree"])))
    config_params["num_out_degree"] = int(round(float(params["num_out_degree"])))
    config_params["multi_hop_max_dist"] = int(round(float(params["multi_hop_max_dist"])))
    config_params["num_encoder_layers"] = int(round(float(params["num_encoder_layers"])))
    config_params["embedding_dim"] = int(round(float(params["embedding_dim"])))
    config_params["ffn_embedding_dim"] = int(round(float(params["ffn_embedding_dim"])))
    config_params["num_attention_heads"] = int(round(float(params["num_attention_heads"])))
    config_params["epoch"] = int(round(float(params["epoch"])))
    config_params["num_edge_dis"] = int(round(float(params["num_edge_dis"])))

    # ✅ 소수 값은 float으로 유지
    config_params["dropout"] = float(params["dropout"])
    config_params["attention_dropout"] = float(params["attention_dropout"])
    config_params["activation_dropout"] = float(params["activation_dropout"])
    config_params["learning_rate"] = float(params["learning_rate"])
    config_params["gradnorm_alpha"] = float(params["gradnorm_alpha"])

    #print("after", config_params)
    # Attention head 크기 조정
    if config_params["embedding_dim"] % config_params["num_attention_heads"] != 0:
        config_params["embedding_dim"] += (
                config_params["num_attention_heads"] - (
                    config_params["embedding_dim"] % config_params["num_attention_heads"])
        )
    #print("after_2", config_params)
    if fixed_params["target_type"] == "default_nm":
        fixed_params["target_type"] = "default"

    config_params["activation_fn"] = activation_fn_selector(int(round(float(config_params["activation_fn"]))))
    #print("after_3",config_params)
    # 학습 실행
    print("config_params", config_params)
    result = cross_validate_model(
        config=config_params,
        target_type=config_params["target_type"],
        loss_function=config_params["loss_function"],
        loss_function_ex=config_params["loss_function_ex"],
        loss_function_prob=config_params["loss_function_prob"],
        num_epochs=int(round(config_params["epoch"])),
        batch_size=config_params["batch_size"],
        n_pairs=config_params["n_pairs"],
        learning_rate=config_params["learning_rate"],
        dataset_path=dataset_path,
        testset_path=testset_path,
        DATASET=DATASET,
        TEST_DATASET=TEST_DATASET,
        alpha=config_params["gradnorm_alpha"],
        n_splits=5
    )
    loss_fn_ex = CV_loss_fn_name_gen(config_params["loss_function_ex"], "ex")
    loss_fn_prob = CV_loss_fn_name_gen(config_params["loss_function_prob"], "prob")
    obj1 = torch.tensor(result.get(loss_fn_ex), dtype=torch.double)
    obj2 = torch.tensor(result.get(loss_fn_prob), dtype=torch.double)

    # ✅ CV 결과 반환
    result["obj1"] = obj1.item()
    result["obj2"] = obj2.item()
    return torch.stack([-obj1, -obj2], dim=-1), result

def objective_function_multilist(params, fixed_params, dataset_path, testset_path, DATASET, TEST_DATASET):

    # 리스트를 값으로 변환
    for key in fixed_params:
        if isinstance(fixed_params[key], list):
            fixed_params[key] = fixed_params[key][0]

    config_params = {**fixed_params, **params}

    #print("before", config_params)

    # ✅ 매핑 제거 (이미 train_X에서 매핑됨)
    config_params["num_in_degree"] = int(round(float(params["num_in_degree"])))
    config_params["num_out_degree"] = int(round(float(params["num_out_degree"])))
    config_params["multi_hop_max_dist"] = int(round(float(params["multi_hop_max_dist"])))
    config_params["num_encoder_layers"] = int(round(float(params["num_encoder_layers"])))
    config_params["embedding_dim"] = int(round(float(params["embedding_dim"])))
    config_params["ffn_embedding_dim"] = int(round(float(params["ffn_embedding_dim"])))
    config_params["num_attention_heads"] = int(round(float(params["num_attention_heads"])))
    config_params["epoch"] = int(round(float(params["epoch"])))
    config_params["num_edge_dis"] = int(round(float(params["num_edge_dis"])))

    # ✅ 소수 값은 float으로 유지
    config_params["dropout"] = float(params["dropout"])
    config_params["attention_dropout"] = float(params["attention_dropout"])
    config_params["activation_dropout"] = float(params["activation_dropout"])
    config_params["learning_rate"] = float(params["learning_rate"])
    config_params["gradnorm_alpha"] = float(params["gradnorm_alpha"])

    #print("after", config_params)
    # Attention head 크기 조정
    if config_params["embedding_dim"] % config_params["num_attention_heads"] != 0:
        config_params["embedding_dim"] += (
                config_params["num_attention_heads"] - (
                    config_params["embedding_dim"] % config_params["num_attention_heads"])
        )
    #print("after_2", config_params)
    if fixed_params["target_type"] == "default_nm":
        fixed_params["target_type"] = "default"

    config_params["activation_fn"] = activation_fn_selector(int(round(float(config_params["activation_fn"]))))
    #print("after_3",config_params)
    # 학습 실행
    print("config_params", config_params)
    result = cross_validate_model(
        config=config_params,
        target_type=config_params["target_type"],
        loss_function=config_params["loss_function"],
        loss_function_ex=config_params["loss_function_ex"],
        loss_function_prob=config_params["loss_function_prob"],
        num_epochs=int(round(config_params["epoch"])),
        batch_size=config_params["batch_size"],
        n_pairs=config_params["n_pairs"],
        learning_rate=config_params["learning_rate"],
        dataset_path=dataset_path,
        testset_path=testset_path,
        DATASET=DATASET,
        TEST_DATASET=TEST_DATASET,
        alpha=config_params["gradnorm_alpha"],
        n_splits=5
    )
    loss_fn_ex = CV_loss_fn_name_gen(config_params["loss_function_ex"], "ex")
    loss_fn_prob = CV_loss_fn_name_gen(config_params["loss_function_prob"], "prob")
    obj1 = torch.tensor(result.get(loss_fn_ex), dtype=torch.double)
    obj2 = torch.tensor(result.get(loss_fn_prob), dtype=torch.double)

    # ✅ CV 결과 반환
    result["obj1"] = obj1.item()
    result["obj2"] = obj2.item()
    return torch.stack([-obj1, -obj2], dim=-1), result
def resume_botorch_optimization_multitaks_2loss(prev_result_path,new_result_path, random_result_path, param_bounds, fixed_params, dataset_path, testset_path,
                                 DATASET, TEST_DATASET, n_iter=20, ):
    # ✅ 두 파일을 합쳐서 이어서 시작
    random_df = pd.read_csv(random_result_path)
    bo_df = pd.read_csv(prev_result_path) if prev_result_path else pd.DataFrame()
    df = pd.concat([random_df, bo_df], ignore_index=True)

    name_loss_fn_e = fixed_params["loss_function_ex"]
    name_loss_fn_p = fixed_params["loss_function_prob"]

    train_Y_1 = torch.tensor(df["obj1"].values, dtype=torch.double)
    train_Y_2 = torch.tensor(df["obj2"].values, dtype=torch.double)
    #train_X = torch.rand(len(train_Y_1), len(param_bounds), dtype=torch.double)#### 수정해야함
    train_X = df[list(param_bounds.keys())].values
    train_X = torch.tensor(train_X, dtype=torch.double)

    train_X_full = torch.cat([
        torch.cat([train_X, torch.zeros(train_X.shape[0], 1)], dim=-1),
        torch.cat([train_X, torch.ones(train_X.shape[0], 1)], dim=-1)
    ], dim=0)
    train_Y_full = torch.cat([train_Y_1.unsqueeze(-1), train_Y_2.unsqueeze(-1)], dim=0)

    model = MultiTaskGP(train_X_full, train_Y_full, task_feature=train_X_full.shape[-1] - 1)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    ref_point = torch.tensor([train_Y_1.min().item(), train_Y_2.min().item()]) - 0.1
    results = []

    for i in range(n_iter):
        start_time = time.time()

        acq_func = MultiTask_qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X,
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1])
        )

        bounds = torch.tensor(list(param_bounds.values()), dtype=torch.double).T  # 실값 기반
        new_x, _ = optimize_acqf(acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=256)

        sample_params = {
            key: val.item()
            for key, val in zip(param_bounds.keys(), new_x[0])
        }
        print("sample_params", sample_params)

        obj, result = objective_function_multitask(sample_params, fixed_params, dataset_path, testset_path, DATASET, TEST_DATASET)

        train_X = torch.cat([train_X, new_x])
        train_Y_1 = torch.cat([train_Y_1, obj[0].view(1)], dim=0)
        train_Y_2 = torch.cat([train_Y_2, obj[1].view(1)], dim=0)

        train_X_full = torch.cat([
            torch.cat([train_X, torch.zeros(train_X.shape[0], 1)], dim=-1),
            torch.cat([train_X, torch.ones(train_X.shape[0], 1)], dim=-1)
        ], dim=0)
        train_Y_full = torch.cat([train_Y_1.unsqueeze(-1), train_Y_2.unsqueeze(-1)], dim=0)

        model.set_train_data(train_X_full, train_Y_full, strict=False)
        fit_gpytorch_mll(mll)

        pareto = torch.cat([train_Y_1.unsqueeze(-1), train_Y_2.unsqueeze(-1)], dim=-1)
        pareto_mask = is_non_dominated(pareto)
        hypervolume = Hypervolume(ref_point).compute(pareto[pareto_mask])

        result.update({
            "iteration": len(df) + i + 1,
            "best_value_1": train_Y_1.max().item(),
            "best_value_2": train_Y_2.max().item(),
            "hypervolume": hypervolume,
            "total_time": time.time() - start_time,
            "pareto_front": str(pareto[pareto_mask].tolist())
        })
        results.append(result)

        updated_df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
        updated_df.to_csv(new_result_path, index=False)

        print(f"[Resume Iter {i + 1}] Hypervolume: {hypervolume:.4f}")

    return results

def resume_botorch_optimization_multilist_2loss(
    prev_result_path,random_result_path,new_result_path, param_bounds, fixed_params, dataset_path, testset_path,
    DATASET, TEST_DATASET, n_iter=20
):

    if random_result_path is not None and prev_result_path is None:
        df = pd.read_csv(random_result_path)
    elif random_result_path is None and prev_result_path is not None:
        df = pd.read_csv(prev_result_path)
    elif random_result_path is not None and prev_result_path is not None:
        random_df = pd.read_csv(random_result_path)
        bo_df = pd.read_csv(prev_result_path)
        df = pd.concat([random_df, bo_df], ignore_index=True)
    else:
        print("error")

    # 이전 결과에서 obj1, obj2와 하이퍼파라미터 불러오기
    train_Y_1 = torch.tensor(df["obj1"].values, dtype=torch.double)
    train_Y_2 = torch.tensor(df["obj2"].values, dtype=torch.double)

    train_X = torch.tensor(df[list(param_bounds.keys())].values, dtype=torch.double)
    print("train_X",train_X)
    # 모델 구성
    model_1 = SingleTaskGP(train_X, train_Y_1.reshape(-1, 1), outcome_transform=Standardize(m=1))
    model_2 = SingleTaskGP(train_X, train_Y_2.reshape(-1, 1), outcome_transform=Standardize(m=1))
    model = ModelListGP(model_1, model_2)

    mll_1 = ExactMarginalLogLikelihood(model_1.likelihood, model_1)
    mll_2 = ExactMarginalLogLikelihood(model_2.likelihood, model_2)

    fit_gpytorch_mll(mll_1)
    fit_gpytorch_mll(mll_2)

    ref_point = torch.tensor([train_Y_1.min().item(), train_Y_2.min().item()]) - 0.1
    results = []

    for i in range(n_iter):
        print("[iter]:", i)
        start_time = time.time()

        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X
        )

        bounds = torch.tensor(list(param_bounds.values()), dtype=torch.double).T  # 실값 기반
        new_x, _ = optimize_acqf(acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=512)

        # unnormalize to evaluate
        sample_params = {
            key: val.item()
            for key, val in zip(param_bounds.keys(), new_x[0])
        }
        print("🧪 new_x raw:", new_x)
        print("🧪 sample_params:", sample_params)
        print("🔎 bounds used:", list(param_bounds.values()))

        obj, result = objective_function_multilist(sample_params, fixed_params, dataset_path, testset_path, DATASET, TEST_DATASET)
        new_y_1, new_y_2 = obj[0], obj[1]

        train_X = torch.cat([train_X, new_x])
        train_Y_1 = torch.cat([train_Y_1, new_y_1.view(1)], dim=0)
        train_Y_2 = torch.cat([train_Y_2, new_y_2.view(1)], dim=0)

        model_1.set_train_data(train_X, train_Y_1, strict=False)
        model_2.set_train_data(train_X, train_Y_2, strict=False)

        fit_gpytorch_mll(mll_1)
        fit_gpytorch_mll(mll_2)

        pareto = torch.cat([train_Y_1.unsqueeze(-1), train_Y_2.unsqueeze(-1)], dim=-1)
        pareto_mask = is_non_dominated(pareto)
        hypervolume = Hypervolume(ref_point).compute(pareto[pareto_mask])

        result.update({
            "iteration": len(df) + i + 1,
            "best_value_1": train_Y_1.max().item(),
            "best_value_2": train_Y_2.max().item(),
            "hypervolume": hypervolume,
            "total_time": time.time() - start_time,
            "pareto_front": str(pareto[pareto_mask].tolist())
        })

        for key in param_bounds:
            result[key] = sample_params[key]

        for key in fixed_params:
            if key not in result:
                val = fixed_params[key][0] if isinstance(fixed_params[key], list) else fixed_params[key]
                result[key] = val

        results.append(result)
        updated_df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
        updated_df.to_csv(new_result_path, index=False)

        print(f"[Resume Iter {i+1}] Hypervolume: {hypervolume:.4f}")

    return results


param_bounds = {
    "num_in_degree": (5, 15),
    "num_out_degree": (5, 15),
    "multi_hop_max_dist": (3, 7),
    "num_encoder_layers": (4, 8),
    "dropout": (0.0, 0.5),
    "attention_dropout": (0.0, 0.5),
    "activation_dropout": (0.0, 0.5),
    "learning_rate": (0.0001, 0.01),
    "embedding_dim": (16, 128),
    "ffn_embedding_dim": (128, 512),
    "num_attention_heads": (4, 16),
    "epoch": (10,20),
    "num_edge_dis": (7, 20),
    "activation_fn": (0.5, 4.5),  # 활성화 함수 선택 (정수형으로 변환)
    #"n_pairs": (50, 50),
    "gradnorm_alpha": (0.1, 1.0),
}

fixed_params = create_fixed_params("../data/train_100.csv", {}, dynamic=False)
fixed_params["target_type"] = ["ex_prob"] # "ex_prob"
fixed_params["loss_function"] = ["MSE"] # "SoftDTW", "MSE", "MAE", "Huber", "SID"
fixed_params["loss_function_ex"] = ["SoftDTW"]
fixed_params["loss_function_prob"] = ["SoftDTW"]
fixed_params["pre_layernorm"] = [False]
fixed_params["q_noise"] = [0.0]
fixed_params["qn_block_size"] = [8]
fixed_params["batch_size"] = [64]
fixed_params["n_pairs"] = [50]
fixed_params["edge_type"] = ["multi_hop"]

dataset = SMILESDataset(csv_file="../data/train_50.csv", attn_bias_w=1.0, target_type="ex_prob")
test_dataset = SMILESDataset(csv_file="../data/test_10.csv", attn_bias_w=1.0, target_type="ex_prob")

resume_botorch_optimization_multilist_2loss(
    prev_result_path="botorch_intermediate_results_SoftDTW_SoftDTW_modellist.csv",
    random_result_path="botorch_intermediate_random_results_SoftDTW_SoftDTW_modellist.csv",
    new_result_path="botorch_intermediate_results_SoftDTW_SoftDTW_resume.csv",
    param_bounds=param_bounds,
    fixed_params=fixed_params,
    dataset_path="../data/train_50.csv",
    testset_path="../data/test_10.csv",
    DATASET=dataset,
    TEST_DATASET=test_dataset,
    n_iter=10
)