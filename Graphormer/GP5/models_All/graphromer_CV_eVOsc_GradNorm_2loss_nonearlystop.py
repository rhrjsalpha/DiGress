from sklearn.model_selection import KFold
from Graphormer.GP5.data_prepare.DataLoader_QMData_All import (
    UnifiedSMILESDataset,
    collate_fn,
    get_global_feature_info,  # helper util – 사용자 정의
)
import pandas as pd
import torch
from graphormer_train_eVOsc_GradNorm_2loss_notearlystop import train_model_ex_porb
from rdkit import Chem
from Graphormer.GP5.models_All.graphormer_3 import GraphormerModel
from typing import Dict, List, Tuple
import numpy as np
import os

def run_cv_and_final_training(
        *,
        config: Dict,
        csv_path_train,
        csv_path_test,
        n_splits=5,
        test_csv_path=None,
        save_path="cv_results.csv",
        target_type="ex_prob",
        loss_function='SID',
        loss_function_ex='SID',
        loss_function_prob='SID',
        num_epochs=10,
        batch_size=50,
        n_pairs=50,
        learning_rate = 1e-3,
        dataset_path=None,
        test_dataset_path=None,
        alpha=0.12,
        global_feature_names=None,
        ex_normalize="ex_min_max",
        prob_normalize="prob_min_max",
        nominal_feature_vocab=None,
        continuous_feature_names=None,
        global_cat_dim=0,
        global_cont_dim=0,
):
    # 전체 dataset 불러오기
    full_df = pd.read_csv(csv_path_train)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_df)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)

        # fold용 CSV 임시 저장
        train_df.to_csv(f"temp_train_fold{fold}.csv", index=False)
        val_df.to_csv(f"temp_val_fold{fold}.csv", index=False)

        cv_train_dataset = UnifiedSMILESDataset(
            csv_file=dataset_path,
            nominal_feature_vocab=nominal_feature_vocab,
            continuous_feature_names=continuous_feature_names,
            global_cat_dim=global_cat_dim,
            global_cont_dim=global_cont_dim,
            ATOM_FEATURES_VOCAB=config["ATOM_FEATURES_VOCAB"],
            float_feature_keys=config["float_feature_keys"],
            BOND_FEATURES_VOCAB=config["BOND_FEATURES_VOCAB"],
            mode=config["mode"],  # "cls", "cls+global_data", "cls+global_model"
            max_nodes=config.get("max_nodes", 128),
            multi_hop_max_dist=config.get("multi_hop_max_dist", 5),
            target_type=config.get("target_type", "default"),
            attn_bias_w=config.get("attn_bias_w", 1.0),
            ex_normalize=config.get("ex_normalize", None),
            prob_normalize=config.get("prob_normalize", None),
            nm_dist_mode=config.get("nm_dist_mode", "hist"),
            nm_gauss_sigma=config.get("nm_gauss_sigma", 10.0),
        )

        val_dataset = UnifiedSMILESDataset(
            csv_file=test_dataset_path,
            nominal_feature_vocab=nominal_feature_vocab,
            continuous_feature_names=continuous_feature_names,
            global_cat_dim=global_cat_dim,
            global_cont_dim=global_cont_dim,
            ATOM_FEATURES_VOCAB=config["ATOM_FEATURES_VOCAB"],
            float_feature_keys=config["float_feature_keys"],
            BOND_FEATURES_VOCAB=config["BOND_FEATURES_VOCAB"],
            mode=config["mode"],  # "cls", "cls+global_data", "cls+global_model"
            max_nodes=config.get("max_nodes", 128),
            multi_hop_max_dist=config.get("multi_hop_max_dist", 5),
            target_type=config.get("target_type", "default"),
            attn_bias_w=config.get("attn_bias_w", 1.0),
            ex_normalize=config.get("ex_normalize", None),
            prob_normalize=config.get("prob_normalize", None),
            nm_dist_mode=config.get("nm_dist_mode", "hist"),
            nm_gauss_sigma=config.get("nm_gauss_sigma", 10.0),
        )
        # 학습 실행
        res, best_path = train_model_ex_porb(
            config=config,
            target_type=target_type,
            loss_function=loss_function,
            loss_function_ex=loss_function_ex,
            loss_function_prob=loss_function_prob,
            num_epochs=num_epochs,
            batch_size=batch_size,
            n_pairs=n_pairs,
            learning_rate = learning_rate,
            dataset_path = None,
            test_dataset_path = None,
            DATASET= cv_train_dataset,
            TEST_VAL_DATASET = val_dataset,
            alpha=alpha,
            is_cv = True,
            nominal_feature_vocab=nominal_feature_vocab,
            continuous_feature_names=continuous_feature_names,
            global_cat_dim=global_cat_dim,
            global_cont_dim=global_cont_dim,
            global_feature_names=global_feature_names,
            ex_normalize=ex_normalize,
            prob_normalize=prob_normalize,
        )

        res["fold"] = fold
        res["best_model_path"] = best_path
        all_results.append(res)

        # pd.DataFrame(all_results).to_csv("intermediate_cv_results.csv", index=False)
        df_all = pd.DataFrame(all_results)

        metric_columns = [
            col for col in df_all.columns
            if any(metric in col.lower() for metric in ["mae", "rmse", "r2", "sid", "softdtw"])
        ]

        # 평균/표준편차 계산 및 추가
        if len(df_all) >= 1:
            mean_row = df_all[metric_columns].mean().to_dict()
            std_row = df_all[metric_columns].std().to_dict()

            mean_row["fold"] = "mean"
            std_row["fold"] = "std"
            mean_row["best_model_path"] = ""
            std_row["best_model_path"] = ""

            df_out = pd.concat([
                df_all,
                pd.DataFrame([mean_row, std_row])
            ], ignore_index=True)
        else:
            df_out = df_all

        df_out.to_csv("intermediate_cv_results.csv", index=False)

    # 전체 데이터로 학습 + test 평가
    print("\n=== Final Training on Full Data ===")

    res_final, best_path = train_model_ex_porb(
        config=config,
        target_type=target_type,
        loss_function=loss_function,
        loss_function_ex=loss_function_ex,
        loss_function_prob=loss_function_prob,
        num_epochs=num_epochs,
        batch_size=batch_size,
        n_pairs=n_pairs,
        learning_rate=learning_rate,
        dataset_path=dataset_path,
        test_dataset_path=test_dataset_path,
        DATASET=None,
        TEST_VAL_DATASET=None,
        alpha=alpha,
        is_cv=False,
        nominal_feature_vocab=nominal_feature_vocab,
        continuous_feature_names=continuous_feature_names,
        global_cat_dim=global_cat_dim,
        global_cont_dim=global_cont_dim,
        global_feature_names=global_feature_names,
        ex_normalize=ex_normalize,
        prob_normalize=prob_normalize,
    )

    res_final["fold"] = "FullTrain"
    res_final["best_model_path"] = best_path
    all_results.append(res_final)

    model = GraphormerModel(config)
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.eval()

        # 평가 함수 따로 정의 가능
        # res_test = evaluate_model_on_test(model, test_dataset)
        # res_test["fold"] = "TestSet"
        # all_results.append(res_test)

    # 최종 저장
    pd.DataFrame(all_results).to_csv(save_path, index=False)
    print(f"\n✅ All CV + Final Training Results Saved to: {save_path}")



PREDEFINED_VOCAB = {
    'Solvent': [
        '1,4-Dioxane', 'Acetonitrile', 'Benzene', 'Chloroform', 'Cyclohexane',
        'Dichloromethane', 'Dimethylformamide', 'Dimethylsulfoxide', 'Ethanol',
        'Ethylacetate', 'Heptane', 'Hexane', 'Methanol', 'N-Methyl-2-pyrrolidone',
        'Tetrahydrofuran', 'Toluene', 'Water', "DMSO", "Acetone"
    ],
}

ATOM_FEATURES_VOCAB = {
        'atomic_num': list(range(1, 119)),  # TODO I need to decrease the range
        'formal_charge': list(range(-5, 6)),  # increase range when diffusion / or add threshold
        'hybridization': [
            Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.OTHER,
            # Chem.rdchem.HybridizationType.UNSPECIFIED # add this when diffusion
        ],
        'is_aromatic': [0, 1],
        'total_num_hs': list(range(0, 9)),  # increase it when diffusion
        'explicit_valence': list(range(0, 8)),  # increase range when diffusion / or add threshold of valence encoding
        'total_bonds': list(range(0, 8)),  # increase range when diffusion / or add threshold
        'partial_charge': float,  # check error and change code when diffusion
        'atomic_mass': float,  # OK when diffusion
    }

float_feature_keys = ['partial_charge', 'atomic_mass']

BOND_FEATURES_VOCAB = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
        # Chem.rdchem.BondType.UNSPECIFIED # add this when diffusion
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS,  # OK when diffusion
    ],
    'is_conjugated': [0, 1],  # OK when diffusion
    'is_in_ring': [0, 1],  # OK when diffusion
}
def main() -> None:
    print("CUDA available:", torch.cuda.is_available())

    # ---------- 글로벌 피처 정보 ----------
    #global nominal_dims, continuous_feature_names, global_cat_dim, global_cont_dim
    global device

    GLOBAL_FEATURE_NAMES = ['Solvent', 'Temperature', 'Pressure']
    csv_path_train  = "../../graphormer_data/train_50_with_features.csv"
    csv_path_test  = "../../graphormer_data/train_50_with_features.csv"
    try:
        nominal_feature_vocab, continuous_feature_names, global_cat_dim, global_cont_dim = get_global_feature_info(GLOBAL_FEATURE_NAMES, PREDEFINED_VOCAB)
        print(nominal_feature_vocab)
        print(continuous_feature_names)
        print(global_cat_dim)
        print(global_cont_dim)
    except Exception as e:
        print("[WARN] global feature info 오류, fallback 사용:", e)
        nominal_dims, continuous_feature_names = {}, []
        global_cat_dim, global_cont_dim = 1, 2

    # device (GPU/CPU) 설정 전역 변수화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # ---------- Graphormer config ----------
    config: Dict = {
        "num_atoms": 100,
        "num_in_degree": 10,
        "num_out_degree": 10,
        "num_edges": 100,
        "num_spatial": 100,
        "num_edge_dis": 10,
        "edge_type": "multi_hop",
        "multi_hop_max_dist": 2,
        "num_encoder_layers": 1,
        "embedding_dim": 128,
        "ffn_embedding_dim": 256,
        "num_attention_heads": 8,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.1,
        "activation_fn": "gelu",
        "pre_layernorm": False,
        "q_noise": 0.0,
        "qn_block_size": 8,
        "output_size": 100,
        "num_categorical_features": 7,  # (= 7 atom categorical)
        "num_continuous_features": 2,  # (= 2 atom continuous)
        "mode": "cls_global_model", # "cls_only" , "cls_global_data", "cls_global_model"
        "target_type": "ex_prob", # "default", "ex_prob", "nm_distribution"
    }
    config.update({
        "ATOM_FEATURES_VOCAB": ATOM_FEATURES_VOCAB,
        "float_feature_keys": float_feature_keys,
        "BOND_FEATURES_VOCAB": BOND_FEATURES_VOCAB,
        "ex_normalize": "ex_min_max", #  ex_std, ex_min_max, none(문자열)
        "prob_normalize": "prob_min_max", # prob_std, prob_min_max, none 문자열
        "nm_dist_mode": "hist",  # 가능 값: "hist" | "gauss" | "exp"
    })
    if config.get("mode") == "cls_global_data" or config.get("mode") == "cls_global_model":
        config.update({
                     "global_cat_dim": global_cat_dim,
                     "global_cont_dim": global_cont_dim,
                      })
    elif config.get("mode") == "cls_only":
        config.update({
            "global_cat_dim": 0,
            "global_cont_dim": 0,
        })
    else:
        print("please use cls_global_data, cls_global_model, cls_only")

    # ---------- 학습 루프 ----------
    loss_candidates = ["MAE"]  # 필요시 확장: ["MAE", "MSE", "SoftDTW", "SID"]
    # results_all: List[Dict] = []
    print('run')
    run_cv_and_final_training(
        config=config,
        csv_path_train=csv_path_train,
        csv_path_test=csv_path_test,
        n_splits=5,
        save_path="cv_results.csv",
        target_type="ex_prob",
        loss_function="SID",
        loss_function_ex="SID",
        loss_function_prob="SID",
        num_epochs=30,
        batch_size=64,
        n_pairs=50,
        learning_rate=1e-4,
        dataset_path=csv_path_train,
        test_dataset_path=csv_path_test,
        alpha=0.12,
        nominal_feature_vocab=nominal_feature_vocab,
        continuous_feature_names=continuous_feature_names,
        global_cat_dim=global_cat_dim,
        global_cont_dim=global_cont_dim,
        global_feature_names=["Solvent", "pH"],
        ex_normalize="ex_min_max",
        prob_normalize="prob_min_max"
    )

    # ---------- 최종 결과 저장 ----------
    #out_csv = "training_results.csv"
    #pd.DataFrame(results_all).to_csv(out_csv, index=False)
    print("\nAll trainings finished. Saved ⇒")

main()