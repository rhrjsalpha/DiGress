import torch
import pandas as pd
from itertools import product
from rdkit import Chem
from rdkit.Chem import rdmolops
import time

from botorch.models import MultiTaskGP, ModelListGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from gpytorch.mlls import ExactMarginalLogLikelihood

from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset
from Graphormer.GP5.models.graphormer_train_eVOsc_GradNorm_2loss import train_model_ex_porb
from Graphormer.GP5.models.graphormer_CV_train_eVOsc_GradNorm_2loss import cross_validate_model
import itertools
from botorch.models.transforms.outcome import Standardize

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

# 리스트 값이 여러 개일 경우 조합 생성 함수
def expand_grid(fixed_params):
    keys = fixed_params.keys()
    values = (fixed_params[key] if isinstance(fixed_params[key], list) else [fixed_params[key]] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations

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

# ✅ 초기값 생성
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


# ✅ 최적화 함수 정의
def objective_function(params, fixed_params, dataset_path, testset_path, DATASET, TEST_DATASET):

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

    return torch.stack([-obj1, -obj2], dim=-1)


# ✅ Botorch 기반 Bayesian Optimization 수행 함수
def bayesian_optimization_botorch(param_bounds, fixed_params, dataset_path, testset_path, DATASET, TEST_DATASET,
                                  init_points=5, n_iter=25):
    results = []

    print("initial points search start")
    counter = 0
    for _ in range(init_points):
        counter += 1
        sample = torch.rand(len(param_bounds), dtype=torch.double)
        sample_params = {
            key: param_bounds[key][0] + (param_bounds[key][1] - param_bounds[key][0]) * val
            for key, val in zip(param_bounds.keys(), sample)
        }

        obj1, obj2 = objective_function(
            sample_params,
            fixed_params,
            dataset_path,
            testset_path,
            DATASET,
            TEST_DATASET
        )

        if 'train_X' not in locals():
            train_X = sample.unsqueeze(0)
            train_Y_1 = obj1.unsqueeze(0)
            train_Y_2 = obj2.unsqueeze(0)
        else:
            train_X = torch.cat([train_X, sample.unsqueeze(0)], dim=0)
            train_Y_1 = torch.cat([train_Y_1, obj1.unsqueeze(0)], dim=0)
            train_Y_2 = torch.cat([train_Y_2, obj2.unsqueeze(0)], dim=0)
        print(f"init count:{counter}")
    print("initial points search finished")

    # ✅ 개별 모델 생성
    model_1 = SingleTaskGP(train_X, train_Y_1.reshape(-1, 1), outcome_transform=Standardize(m=1))
    model_2 = SingleTaskGP(train_X, train_Y_2.reshape(-1, 1), outcome_transform=Standardize(m=1))

    # ✅ ModelListGP로 합침
    model = ModelListGP(model_1, model_2)

    mll_1 = ExactMarginalLogLikelihood(model_1.likelihood, model_1)
    mll_2 = ExactMarginalLogLikelihood(model_2.likelihood, model_2)

    fit_gpytorch_mll(mll_1)
    fit_gpytorch_mll(mll_2)

    ref_point = torch.tensor([train_Y_1.min().item(), train_Y_2.min().item()]) - 0.1 #최소값에서 0.1 을 빼서 그곳을 원점으로 사용함

    counter = 0
    for _ in range(n_iter):
        counter += 1
        #print(f"train_X:: {counter}", train_X)
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X
        )

        bounds = torch.tensor(list(param_bounds.values()), dtype=torch.double).T
        new_x, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512
        )
        #print(f"new_x:: {counter}", new_x)

        new_y_1, new_y_2 = objective_function(
            {key: float(val) for key, val in zip(param_bounds.keys(), new_x[0])},
            fixed_params,
            dataset_path,
            testset_path,
            DATASET,
            TEST_DATASET
        )

        train_Y_1 = torch.cat([train_Y_1, new_y_1.reshape(-1)], dim=0)
        train_Y_2 = torch.cat([train_Y_2, new_y_2.reshape(-1)], dim=0)
        train_X = torch.cat([train_X, new_x])

        model_1.set_train_data(train_X, train_Y_1, strict=False)
        model_2.set_train_data(train_X, train_Y_2, strict=False)

        fit_gpytorch_mll(mll_1)
        fit_gpytorch_mll(mll_2)

        # 하이퍼볼륨 계산
        pareto_mask = is_non_dominated(torch.cat([train_Y_1.unsqueeze(-1), train_Y_2.unsqueeze(-1)], dim=-1))
        pareto_front = torch.cat([train_Y_1.unsqueeze(-1), train_Y_2.unsqueeze(-1)], dim=-1)[pareto_mask]
        hv = Hypervolume(ref_point)
        hypervolume = hv.compute(pareto_front)

        result = {
            "iteration": counter,
            "best_value_1": train_Y_1.max().item(),
            "best_value_2": train_Y_2.max().item(),
            "hypervolume": hypervolume
        }

        # ✅ param_bounds 값 저장
        for key in param_bounds.keys():
            value = new_x[0][list(param_bounds.keys()).index(key)].item()
            if key in [
                "num_in_degree", "num_out_degree", "multi_hop_max_dist", "num_encoder_layers",
                "embedding_dim", "ffn_embedding_dim", "num_attention_heads", "epoch",
                "num_edge_dis", "qn_block_size", "batch_size", "n_pairs"
            ]:
                result[key] = int(round(value))  # 정수로 변환
            elif key == "activation_fn":  # activation_fn 값 처리
                result[key] = int(round(value))  # 정수 변환
                result["activation_fn_name"] = activation_fn_selector(result[key])  # 활성화 함수 선택
            else:
                result[key] = value

        # ✅ fixed_params 값 저장 (정수형 변환 추가)
        for key in fixed_params.keys():
            if key not in result:
                if isinstance(fixed_params[key], list):
                    value = fixed_params[key][0]
                else:
                    value = fixed_params[key]

                # 정수 변환 필요 시 수행
                if key in [
                    "num_atoms", "num_edges", "num_spatial", "output_size",
                    "batch_size", "n_pairs", "qn_block_size", "num_edge_dis"
                ]:
                    result[key] = int(value)
                else:
                    result[key] = value
        result["pareto_front"] = str(pareto_front.tolist())
        print(f"result at {counter}:",result)
        results.append(result)

        # ✅ 중간 결과 저장 (매 iteration 마다)
        intermediate_results = pd.DataFrame(results)
        intermediate_results.to_csv("botorch_intermediate_results.csv", index=False)

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
    "epoch": (3, 5),
    "num_edge_dis": (7, 20),
    "activation_fn": (0.5, 4.5),  # 활성화 함수 선택 (정수형으로 변환)
    #"n_pairs": (50, 50),
    "gradnorm_alpha": (0.1, 1.0),
}


# ✅ 실행 코드
fixed_params = create_fixed_params("../data/train_100.csv", {}, dynamic=True)
fixed_params["target_type"] = ["ex_prob"] # "ex_prob"
fixed_params["loss_function"] = ["MSE"] # "SoftDTW", "MSE", "MAE", "Huber", "SID"
fixed_params["loss_function_ex"] = ["SID","SoftDTW"]
fixed_params["loss_function_prob"] = ["SID","SoftDTW"]
fixed_params["pre_layernorm"] = [False]
fixed_params["q_noise"] = [0.0]
fixed_params["qn_block_size"] = [8]
fixed_params["batch_size"] = [64]
fixed_params["n_pairs"] = [50]
fixed_params["edge_type"] = ["multi_hop"]

dataset = SMILESDataset(csv_file="../data/train_50.csv", attn_bias_w=1.0, target_type="ex_prob")
test_dataset = SMILESDataset(csv_file="../data/train_50.csv", attn_bias_w=1.0, target_type="ex_prob")


results = bayesian_optimization_botorch(
    param_bounds=param_bounds,
    fixed_params=fixed_params,
    dataset_path="../data/train_50.csv",
    testset_path="../data/train_50.csv",
    DATASET=dataset,
    TEST_DATASET=test_dataset,
    init_points=3,
    n_iter=3
)

# ✅ 결과 저장
print(results)


## multi task 사용중인데 multitask는 서로 상관관계가 존재할때 사용, 반면 eV와 Osc는 산점도 그릴시 은 서로 상관이 없음
## Cross validation에서 나오는것 전부 csv 에 포함되도록하기, 현재는 hyperapram 및 hyper volume 만이 들어감