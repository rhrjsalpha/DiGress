import logging
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
#from dask.dataframe import read_csv
from torch.utils.data import DataLoader, TensorDataset

from Graphormer.GP.models.graphormer_torch import GraphormerModel
from Graphormer.GP5.models.graphormer_train import train_model
from Graphormer.GP5.models.graphormer_CV_train import cross_validate_model
from itertools import product
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops

def config_maker(dataset_path, combination_dict):
    #output size 설정 위해 데이터를 불러와 확인#
    dataset=pd.read_csv(dataset_path)
    cols=dataset.columns
    output_size_count = 0
    for col_name in cols:
        if "ex" in col_name:
            output_size_count+=1
        elif 'prob' in col_name:
            output_size_count+=1
        else:
            pass

    #최대 원자수, 최대 edge 수 계산 위해 rdkit 사용#
    smiles_list = dataset["smiles"]
    max_atoms = 0
    max_edges = 0
    max_spatial = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # 원자 수 계산
            num_atoms = mol.GetNumAtoms()
            max_atoms = max(max_atoms, num_atoms)

            # 엣지 수 계산 (결합 수)
            num_edges = len(rdmolops.GetAdjacencyMatrix(mol).nonzero()[0]) // 2
            max_edges = max(max_edges, num_edges)

            # 자내 원자의 최대 거리 계산
            dist_matrix = rdmolops.GetDistanceMatrix(mol)
            max_spatial = max(max_spatial, dist_matrix.max())
            max_spatial = int(max_spatial)

    print(max_atoms, max_edges, max_spatial)

    if combination_dict == None:
        combination_dict = {
            "num_atoms": [max_atoms],  # 분자의 최대 원자 수 (그래프의 노드 개수)
            "num_in_degree": [10],  # 그래프 노드의 최대 in-degree
            "num_out_degree": [10],  # 그래프 노드의 최대 out-degree
            "num_edges": [max_edges],  # 그래프의 최대 엣지 개수
            "num_spatial": [max_spatial+int(1)],  # 공간적 위치 인코딩을 위한 최대 값
            "num_edge_dis": [10],  # 엣지 거리 인코딩을 위한 최대 값
            "edge_type": ["multi_hop"],  # 엣지 타입 ("multi_hop" 또는 다른 값 가능)
            "multi_hop_max_dist": [5],  # Multi-hop 엣지의 최대 거리 ->
            "num_encoder_layers": [10],  # Graphormer 모델에서 사용할 인코더 레이어 개수 ->
            "embedding_dim": [128],  # 임베딩 차원 크기 (노드, 엣지 등) ->
            "ffn_embedding_dim": [256],  # Feedforward Network의 임베딩 크기 ->
            "num_attention_heads": [8],  # Multi-head Attention에서 헤드 개수 ->
            "dropout": [0.1],  # 드롭아웃 비율
            "attention_dropout": [0.1],  # Attention 레이어의 드롭아웃 비율 ->
            "activation_dropout": [0.1],  # 활성화 함수 이후 드롭아웃 비율 ->
            "activation_fn": ["gelu"],  # 활성화 함수 ("gelu", "relu" 등) ->
            "pre_layernorm": [False],  # LayerNorm을 Pre-Normalization으로 사용할지 여부
            "q_noise": [0.0],  # Quantization noise (훈련 중 노이즈 추가를 위한 매개변수)
            "qn_block_size": [8],  # Quantization block 크기
            "output_size": [output_size_count],  # 모델 출력 크기 ->
        }
    else:
        combination_dict = combination_dict

    print(*combination_dict.items())
    keys, values = zip(*combination_dict.items())
    config_combinations = [dict(zip(keys, v)) for v in product(*values)]
    print(len(config_combinations))
    return config_combinations

def grid_combinations(config_combinations, param_grid):
    if param_grid is None:
        param_grid = {
            "target_type": ["default", "ex_prob"],  # 대상 유형
            "loss_function": ["MSE", "MAE"],  # 전체 손실
            "loss_function_ex": ["SoftDTW", "MSE"],  # 'ex'에 대한 손실
            "loss_function_prob": ["SoftDTW", "MAE"],  # 'prob'에 대한 손실
            "weight_ex": [0.3, 0.5, 0.7],  # 가중치
            "batch_size": [2, 4],  # 배치 크기
            "n_pairs": [1, 5],  # 'ex_prob' 쌍 개수
            "learning_rate": [0.001, 0.01],  # 학습률
        }
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    print("len param comb",len(param_combinations))
    for param_combination in param_combinations:
        print(param_combination.keys())
    for config_combination in config_combinations:
        print(config_combination.keys())

    final_param_gird = []
    for config_combination in config_combinations:
        for param_combination in param_combinations:
            param_combination['config'] = config_combination
            final_param_gird.append(param_combination)

    print(len(final_param_gird))
    return final_param_gird


def grid_search_param_grid(final_param_gird, train_function, dataset_path, testset_path, num_epochs=10, verbose=True, CV=True):
    """
    Perform GridSearch over the provided parameter grid.

    Args:
        param_grid (dict): Dictionary of hyperparameter ranges to search.
        train_function (function): Function to train and evaluate the model.
        dataset_path (str): Path to the dataset CSV file.
        num_epochs (int): Number of epochs for training.
        verbose (bool): If True, print progress and results.

    Returns:
        dict: Best parameters and corresponding metrics.
        pd.DataFrame: Full results for all hyperparameter combinations.
    """
    if train_function is None:
        # Hyperparameter combinations count
        print("param_grid_size",len(final_param_gird))
        count=0
        for param_combination in final_param_gird:
            count+=1
            print(len(param_combination),param_combination.keys())
        print(count)

    else:
        if CV == False:
            # Results storage
            results = []

            # Grid Search
            idx = 0
            for params in final_param_gird:
                if verbose:
                    print(f"Testing combination {idx + 1}/{len(final_param_gird)}: {params}")

                # Train the model with the current parameter set
                result = train_function(
                    config=params['config'],
                    target_type=params['target_type'],
                    loss_function=params['loss_function'],
                    loss_function_ex=params['loss_function_ex'],
                    loss_function_prob=params['loss_function_prob'],
                    weight_ex=params['weight_ex'],
                    num_epochs=num_epochs,
                    batch_size=params['batch_size'],
                    n_pairs=params['n_pairs'],
                    learning_rate=params['learning_rate'],
                    dataset_path=dataset_path
                )

                # Save results
                results.append({
                    **params,
                    **result  # Include metrics like final_loss, best_epoch, etc.
                })
                idx+=1

                if verbose:
                    print(f"Result for combination {idx + 1}: {result}")

            # Find best parameters (based on final_loss or other metric)
            results_df = pd.DataFrame(results)

            return results_df

        elif CV == True:
            # Results storage
            results = []

            # Grid Search
            idx = 0
            for params in final_param_gird:
                if verbose:
                    print(f"Testing combination {idx + 1}/{len(final_param_gird)}: {params}")

                # Train the model with the current parameter set
                result = cross_validate_model(
                    config=params['config'],
                    target_type=params['target_type'],
                    loss_function=params['loss_function'],
                    loss_function_ex=params['loss_function_ex'],
                    loss_function_prob=params['loss_function_prob'],
                    weight_ex=params['weight_ex'],
                    num_epochs=num_epochs,
                    batch_size=params['batch_size'],
                    n_pairs=params['n_pairs'],
                    learning_rate=params['learning_rate'],
                    dataset_path=dataset_path,
                    testset_path=testset_path
                )

                # Save results
                results.append({
                    **params,
                    **result  # Include metrics like final_loss, best_epoch, etc.
                })
                idx += 1

                if verbose:
                    print(f"Result for combination {idx + 1}: {result}")

            # Find best parameters (based on final_loss or other metric)
            results_df = pd.DataFrame(results)

            return results_df
        else:
            return None

config_combinations = config_maker("../data/train_1000.csv", combination_dict=None)

param_grid = {
    "target_type": ["ex_prob"],  # 대상 유형
    "loss_function": ["MSE"],  # 전체 손실 # ex porb 일때는 하나만 넣어 놓을것, 이것 무엇을 넣든 손실함수 종류가 변하지 않을것
    "loss_function_ex": ["SoftDTW","MSE", "MAE","Huber","SID"],  # 'ex'에 대한 손실 "SoftDTW","MSE", "MAE","Huber","SID"
    "loss_function_prob": ["SoftDTW","MSE", "MAE","Huber","SID"],  # 'prob'에 대한 손실 "SoftDTW","MSE", "MAE","Huber","SID"
    "weight_ex": [0.3,0.5,0.7],  # 가중치
    "batch_size": [128],  # 배치 크기
    "n_pairs": [5],  # 'ex_prob' 쌍 개수
    "learning_rate": [0.001,0.01],  # 학습률 0.001, 0.01
}

combinations = grid_combinations(config_combinations=config_combinations, param_grid=param_grid)
#print(combinations)
results_df = grid_search_param_grid(final_param_gird=combinations, train_function = train_model, dataset_path="../data/train_1000.csv", testset_path="../data/test_1000.csv",num_epochs=200, verbose=True, CV=True)
results_df.to_csv("../GridSearch/GredSearch_n_pair_5.csv", index=False)


config_example = {
    "num_atoms": 100,              # 분자의 최대 원자 수 (그래프의 노드 개수)
    "num_in_degree": 10,           # 그래프 노드의 최대 in-degree
    "num_out_degree": 10,          # 그래프 노드의 최대 out-degree
    "num_edges": 50,               # 그래프의 최대 엣지 개수
    "num_spatial": 100,            # 공간적 위치 인코딩을 위한 최대 값
    "num_edge_dis": 10,            # 엣지 거리 인코딩을 위한 최대 값
    "edge_type": "multi_hop",      # 엣지 타입 ("multi_hop" 또는 다른 값 가능)
    "multi_hop_max_dist": 5,       # Multi-hop 엣지의 최대 거리 ->
    "num_encoder_layers": 6,       # Graphormer 모델에서 사용할 인코더 레이어 개수 ->
    "embedding_dim": 128,          # 임베딩 차원 크기 (노드, 엣지 등) ->
    "ffn_embedding_dim": 256,      # Feedforward Network의 임베딩 크기 ->
    "num_attention_heads": 8,      # Multi-head Attention에서 헤드 개수 ->
    "dropout": 0.1,                # 드롭아웃 비율
    "attention_dropout": 0.1,      # Attention 레이어의 드롭아웃 비율 ->
    "activation_dropout": 0.1,     # 활성화 함수 이후 드롭아웃 비율 ->
    "activation_fn": "gelu",       # 활성화 함수 ("gelu", "relu" 등) ->
    "pre_layernorm": False,        # LayerNorm을 Pre-Normalization으로 사용할지 여부
    "q_noise": 0.0,                # Quantization noise (훈련 중 노이즈 추가를 위한 매개변수)
    "qn_block_size": 8,            # Quantization block 크기
    "output_size": 100,            # 모델 출력 크기 ->
}



