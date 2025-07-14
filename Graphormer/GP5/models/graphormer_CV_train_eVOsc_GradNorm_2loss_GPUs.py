import time

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from Graphormer.GP5.models.graphormer_train_eVOsc_GradNorm_2loss_GPUs import train_model_ex_porb, evaluate_model
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from Graphormer.GP5.data_prepare.Dataloader_2 import SMILESDataset, collate_fn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from types import SimpleNamespace
import os
import json
import subprocess

import pandas as pd
def run_ddp_training_external(args_dict, fold_idx):

    json_path = f"./ddp_args_fold{fold_idx}.json"
    # JSON에 저장 가능한 값만 필터링
    json_safe_args = {
        k: v for k, v in args_dict.items()
        if k not in ["TRAIN_DATASET", "VAL_DATASET"]
    }

    with open(json_path, "w") as f:
        json.dump(json_safe_args, f)

    # subprocess 실행
    command = f"python train_model_entrypoint.py --args {json_path}"
    print(f"🧪 Launching subprocess for Fold {fold_idx+1}: {command}")
    subprocess.run(command, shell=True, check=True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    # libuv 사용을 명시적으로 비활성화
    os.environ['USE_LIBUV'] = '0'

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method='tcp://127.0.0.1:29501'
    )
    torch.cuda.set_device(rank)

def flatten_cv_and_train_test(cv_df: pd.DataFrame, train_test_row: pd.Series) -> pd.DataFrame:
    result = {}

    for col in cv_df.columns:
        values = cv_df[col].tolist()

        # 리스트 형태인 컬럼이면 fold별로만 저장 (mean 불가)
        if isinstance(values[0], str) and values[0].startswith("[") and values[0].endswith("]"):
            for i in range(5):
                result[f"{col}_CV{i+1}"] = values[i]
            # 평균은 저장하지 않음
        else:
            # 수치형 컬럼: fold별 + 평균
            for i in range(5):
                result[f"{col}_CV{i+1}"] = values[i]
            result[f"{col}_CV"] = pd.to_numeric(cv_df[col], errors="coerce").mean()

    # 전체 학습 결과 저장 (Train / Test 구분)
    for col in train_test_row.index:
        if col.startswith("val_"):
            new_col = col.replace("val_", "") + "_Test"
        else:
            new_col = col + "_Train"
        result[new_col] = train_test_row[col]

    return pd.DataFrame([result])


def cross_validate_with_splits(train_val_splits, config, target_type, loss_function_ex, loss_function_prob,
                               num_epochs, batch_size, n_pairs, learning_rate, patience, alpha,
                               training_result_root="./cv_result.csv", train_dataset_path=None, val_dataset_path=None, use_ddp=True):
    all_paths = []

    for fold, (train_dataset, val_dataset) in enumerate(train_val_splits):
        print(f"🚀 Fold {fold+1} 시작")
        start_time = time.time()
        world_size = torch.cuda.device_count()

        fold_result_path = f"./fold{fold+1}.csv"

        train_args = SimpleNamespace(
            world_size=world_size,
            config=config,
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            target_type=target_type,
            loss_function_ex=loss_function_ex,
            loss_function_prob=loss_function_prob,
            num_epochs=num_epochs,
            batch_size=batch_size,
            n_pairs=n_pairs,
            learning_rate=learning_rate,
            patience=patience,
            alpha=alpha,
            TRAIN_DATASET=train_dataset,
            VAL_DATASET=val_dataset,
            training_result_root=f"./fold{fold+1}.csv",
            skip_setup=False
        )

        if use_ddp == True:
            #run_ddp_training_external(train_args.__dict__, fold_idx=fold)
            mp.spawn(train_model_ex_porb, args=(train_args,), nprocs=train_args.world_size)
        else:
            train_model_ex_porb(0, train_args)

        # ✅ 훈련 결과 파일에서 best_model_path, best_epoch, loss_history, weight_history 수동으로 추적
        best_model_path = "./best_model.pth"  # 기본 저장 경로에서 로딩
        best_epoch = train_args.num_epochs  # 혹은 기록한 best_epoch 불러오기
        # 아래는 실제로 train_model_ex_porb가 반환한 걸 받으면 좋지만, spawn은 값을 직접 return 못 함

        # ✅ 평가를 위한 로더 생성 (단일 GPU에서 수행)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  collate_fn=lambda batch: collate_fn(batch, train_dataset.dataset, n_pairs=n_pairs))

        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                collate_fn=lambda batch: collate_fn(batch, val_dataset.dataset, n_pairs=n_pairs))

        loss_df = pd.read_csv(f"./loss_history_rank0.csv")
        weight_df = pd.read_csv(f"./weight_history_rank0.csv")
        best_info = pd.read_csv(f"./best_info_rank0.csv").iloc[0]

        loss_history = loss_df.to_dict(orient="list")
        weight_history = weight_df.to_dict(orient="list")
        best_model_path = best_info["best_model_path"]
        best_epoch = int(best_info["best_epoch"])

        # ✅ 단일 GPU 평가 수행
        train_args.skip_setup = True  # DDP 초기화 안함
        evaluate_model(best_model_path, train_loader, val_loader, train_args, best_epoch, loss_history,
                       weight_history)

        all_paths.append(fold_result_path)
        total_time = time.time() - start_time
        print(f"🚀 Fold {fold + 1} end, time:{total_time}")

    # Fold별 결과 통합
    merged = [pd.read_csv(p) for p in all_paths]
    result = pd.concat(merged, axis=0, ignore_index=True)

    # 평균 결과 추가
    mean_row = result.mean(numeric_only=True)
    result = pd.concat([result, pd.DataFrame([mean_row])], ignore_index=True)
    result.to_csv(training_result_root, index=False)
    print(f"✅ CV 결과 저장 완료: {training_result_root}")

def train_on_full_and_eval(config, full_train_dataset, test_dataset, target_type, loss_function_ex, loss_function_prob,
                           num_epochs, batch_size, n_pairs, learning_rate, patience, alpha,
                           training_result_root="./final_train_results.csv", use_ddp=None):
    print("🧪 전체 데이터로 학습 후 테스트셋 평가")
    start_time = time.time()
    world_size = torch.cuda.device_count()
    train_args = SimpleNamespace(
        world_size=world_size,
        config=config,
        train_dataset_path=None,
        val_dataset_path=None,
        target_type=target_type,
        loss_function_ex=loss_function_ex,
        loss_function_prob=loss_function_prob,
        num_epochs=num_epochs,
        batch_size=batch_size,
        n_pairs=n_pairs,
        learning_rate=learning_rate,
        patience=patience,
        alpha=alpha,
        TRAIN_DATASET=full_train_dataset,
        VAL_DATASET=test_dataset,
        training_result_root=training_result_root,
        skip_setup=False,
    )

    if use_ddp == True:
        mp.spawn(train_model_ex_porb, args=(train_args,), nprocs=train_args.world_size)
    else:
        train_model_ex_porb(0, train_args)

    # ✅ 평가를 위한 로더 생성 (단일 GPU에서 수행)
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size,
                              collate_fn=lambda batch: collate_fn(batch, full_train_dataset, n_pairs=n_pairs))

    val_loader = DataLoader(test_dataset, batch_size=batch_size,
                            collate_fn=lambda batch: collate_fn(batch, test_dataset, n_pairs=n_pairs))

    loss_df = pd.read_csv(f"./loss_history_rank0.csv")
    weight_df = pd.read_csv(f"./weight_history_rank0.csv")
    best_info = pd.read_csv(f"./best_info_rank0.csv").iloc[0]

    loss_history = loss_df.to_dict(orient="list")
    weight_history = weight_df.to_dict(orient="list")
    best_model_path = best_info["best_model_path"]
    best_epoch = int(best_info["best_epoch"])

    # ✅ 단일 GPU 평가 수행
    train_args.skip_setup = True  # DDP 초기화 안함
    evaluate_model(best_model_path, train_loader, val_loader, train_args, best_epoch, loss_history,
                   weight_history)

    total_time = time.time() - start_time
    print(f"full training end, time:{total_time}")

def merge_all_results(cv_csv_path, test_csv_path, final_output_path="final_pipeline_results.csv"):
    df_cv = pd.read_csv(cv_csv_path)
    df_test = pd.read_csv(test_csv_path)
    df_all = pd.concat([df_cv, df_test], axis=1)
    df_all.to_csv(final_output_path, index=False)
    print(f"🎉 전체 파이프라인 결과 저장 완료: {final_output_path}")

def create_train_val_splits(full_dataset, n_splits=5, seed=42):
    """
    단일 CSV 경로를 받아 sklearn의 KFold로 5개 분할된 (train, val) Subset 튜플을 반환
    """
    #full_dataset = SMILESDataset(csv_file=dataset_path, attn_bias_w=1.0, target_type="ex_prob")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    train_val_splits = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        train_val_splits.append((train_subset, val_subset))

    return train_val_splits

def cross_validate_model(
    config,
    target_type="ex_prob",
    loss_function_1="MSE",
    loss_function_2="MSE",
    weight_ex=0.5,
    num_epochs=10,
    batch_size=50,
    n_pairs=50,
    learning_rate=0.001,
    patience=20,
    alpha=0.12,
    train_dataset=None,
    test_dataset=None,
    cv_result_root = None,
    training_result_root = None,
    flattend_result_root = None,
    n_splits=5,
    use_ddp=True
):

    if train_dataset is None:
        train_dataset = SMILESDataset(csv_file="../../graphormer_data/train_50.csv", attn_bias_w=1.0, target_type=target_type)
    if test_dataset is None:
        test_dataset = SMILESDataset(csv_file="../../graphormer_data/test_100.csv", attn_bias_w=1.0, target_type=target_type)

    if cv_result_root is None:
        cv_result_root = "cv_results.csv"
    if training_result_root is None:
        training_result_root = "final_train_results.csv"
    if flattend_result_root is None:
        flattend_result_root = "flattened_final_results.csv"
    train_val_splits = create_train_val_splits(full_dataset=train_dataset, n_splits=n_splits)

    cross_validate_with_splits(
        train_val_splits=train_val_splits,
        config=config,
        target_type=target_type,
        loss_function_ex=loss_function_1,
        loss_function_prob=loss_function_2,
        num_epochs=num_epochs,
        batch_size=batch_size,
        n_pairs=n_pairs,
        learning_rate=learning_rate,
        patience=patience,
        alpha=alpha,
        training_result_root=cv_result_root,
        use_ddp=use_ddp
    )
    print("CV_finished")

    train_on_full_and_eval(
        config=config,
        full_train_dataset=train_dataset,
        test_dataset=test_dataset,
        target_type=target_type,
        loss_function_ex=loss_function_1,
        loss_function_prob=loss_function_2,
        num_epochs=num_epochs,
        batch_size=batch_size,
        n_pairs=n_pairs,
        learning_rate=learning_rate,
        patience=patience,
        alpha=alpha,
        training_result_root=training_result_root,
        use_ddp=use_ddp
    )
    print("training finished")

    cv_df = pd.read_csv(cv_result_root).iloc[:5]
    test_row = pd.read_csv(training_result_root).iloc[0]

    final_flattened_df = flatten_cv_and_train_test(cv_df, test_row)
    final_flattened_df.to_csv(flattend_result_root, index=False)
    if len(final_flattened_df) == 1:  # DataFrame에 행이 한 개일 경우
        final_flattened_dict = final_flattened_df.iloc[0].to_dict()  # 단일 딕셔너리 반환
    else:  # 여러 행이 있을 경우
        final_flattened_dict = final_flattened_df.to_dict(orient="records")  # 리스트 내 딕셔너리
    return final_flattened_dict

if __name__ == "__main__":
    dataset_path = "../../graphormer_data/train_50.csv"
    target_type = "ex_prob"
    # 자동 분할 (csv 한 개에서 5fold Subset 생성)
    config = {
        "num_atoms": 100,  # 분자의 최대 원자 수 (그래프의 노드 개수) 100
        "num_in_degree": 10,  # 그래프 노드의 최대 in-degree
        "num_out_degree": 10,  # 그래프 노드의 최대 out-degree
        "num_edges": 100,  # 그래프의 최대 엣지 개수 50
        "num_spatial": 100,  # 공간적 위치 인코딩을 위한 최대 값 default 100
        "num_edge_dis": 10,  # 엣지 거리 인코딩을 위한 최대 값
        "edge_type": "multi_hop",  # 엣지 타입 ("multi_hop" 또는 다른 값 가능)
        "multi_hop_max_dist": 2,  # Multi-hop 엣지의 최대 거리
        "num_encoder_layers": 1,  # Graphormer 모델에서 사용할 인코더 레이어 개수
        "embedding_dim": 128,  # 임베딩 차원 크기 (노드, 엣지 등)
        "ffn_embedding_dim": 256,  # Feedforward Network의 임베딩 크기
        "num_attention_heads": 8,  # Multi-head Attention에서 헤드 개수
        "dropout": 0.1,  # 드롭아웃 비율
        "attention_dropout": 0.1,  # Attention 레이어의 드롭아웃 비율
        "activation_dropout": 0.1,  # 활성화 함수 이후 드롭아웃 비율
        "activation_fn": "gelu",  # 활성화 함수 ("gelu", "relu" 등)
        "pre_layernorm": False,  # LayerNorm을 Pre-Normalization으로 사용할지 여부
        "q_noise": 0.0,  # Quantization noise (훈련 중 노이즈 추가를 위한 매개변수)
        "qn_block_size": 8,  # Quantization block 크기
        "output_size": 100,  # 모델 출력 크기
    }

    train_dataset = SMILESDataset(csv_file=dataset_path, attn_bias_w=1.0, target_type=target_type)
    test_dataset = SMILESDataset(csv_file="../../graphormer_data/test_100.csv", attn_bias_w=1.0, target_type=target_type)
    df = cross_validate_model(
        config,
        target_type="ex_prob",
        loss_function_1="SID",
        loss_function_2="SID",
        weight_ex=0.5,
        num_epochs=200,
        batch_size=50,
        n_pairs=50,
        learning_rate=0.001,
        patience=20,
        alpha=0.12,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        cv_result_root = None,
        training_result_root = None,
        flattend_result_root = None,
    )
    print(df)






