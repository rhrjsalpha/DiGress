
'''
This script simulates and visualizes the noise application (diffusion) process of the DiGress model.
It loads a SMILES string from a CSV file and applies noise at different timesteps,
saving the intermediate noisy molecular structures.
'''
import torch                                # PyTorch 라이브러리, 텐서 및 딥러닝 연산에 사용
import os                                   # 운영 체제와 상호 작용하기 위한 모듈 (예: 파일 경로 처리)
import sys                                  # 파이썬 인터프리터와 상호 작용하기 위한 모듈 (예: 시스템 경로 추가)
import numpy as np                          # 수치 연산을 위한 라이브러리 (여기서는 직접 사용되지 않음)
import pandas as pd                         # 데이터 조작 및 분석을 위한 라이브러리, CSV 파일 읽기에 사용
from rdkit import Chem                      # RDKit의 핵심 화학 모듈, 분자 구조 처리에 사용
from rdkit.Chem import Draw, AllChem        # 분자 구조를 그리거나 추가적인 화학 기능을 사용하기 위한 모듈
from omegaconf import OmegaConf             # 계층적 설정을 관리하기 위한 라이브러리
import torch.nn.functional as F             # 원-핫 인코딩 등 신경망 함수를 사용하기 위한 모듈
from torch_geometric.data import Data       # PyTorch Geometric 라이브러리, 그래프 데이터를 표현하는 데 사용

# --- 프로젝트 루트 디렉터리를 시스템 경로에 추가하여 모듈 임포트 ---
script_dir = os.path.dirname(os.path.abspath(__file__))                         # 현재 스크립트 파일이 있는 디렉터리 경로를 가져옴
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))         # 프로젝트의 루트 디렉터리 경로를 계산 (두 단계 상위 폴더)
if project_root not in sys.path:
    sys.path.append(project_root)                                               # 시스템 경로에 추가하여 src 폴더의 모듈을 임포트할 수 있게 함

# --- DiGress 프로젝트 내부 모듈 임포트 ---
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion           # 이산적 확산 모델의 핵심 클래스
from src.analysis.rdkit_functions import build_molecule_with_partial_charges  # 노이즈 데이터로부터 분자를 재구성하는 함수
from src import utils                                                         # 프로젝트 전반에 사용되는 유틸리티 함수
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete         # 훈련 지표를 추상화한 기본 클래스
from src.metrics.molecular_metrics import SamplingMolecularMetrics            # 분자 샘플링 성능을 평가하는 지표 클래스
from src.analysis.visualization import MolecularVisualization                 # 분자 시각화 도구 클래스
from src.diffusion.extra_features import DummyExtraFeatures                   # 더미(dummy) 추가 특징 클래스

# --- SMILES 문자열을 PyTorch Geometric 데이터 객체로 변환하는 헬퍼 함수 ---
def smiles_to_torch_geometric(smiles_string, x_classes, e_classes):
    '''
    SMILES 문자열을 받아 PyTorch Geometric의 Data 객체로 변환합니다.

    Args:
        smiles_string (str): 변환할 분자의 SMILES 문자열.
        x_classes (int): 노드(원자) 특징의 총 클래스 수.
        e_classes (int): 엣지(결합) 특징의 총 클래스 수.

    Returns:
        torch_geometric.data.Data: 분자 그래프 정보를 담은 Data 객체.
    '''
    mol = Chem.MolFromSmiles(smiles_string)                                     # RDKit을 사용하여 SMILES로부터 분자 객체 생성
    if mol is None: return None                                                 # 분자 생성에 실패하면 None을 반환
    mol = Chem.AddHs(mol)                                                       # 분자에 수소 원자를 명시적으로 추가

    atom_map = {1: 1, 6: 2, 7: 3, 8: 4, 9: 5}                                    # 원자 번호(H, C, N, O, F)를 정수 인덱스로 매핑
    atom_features = [atom_map.get(atom.GetAtomicNum(), 0) for atom in mol.GetAtoms()] # 각 원자의 원자 번호를 인덱스로 변환
    x = F.one_hot(torch.tensor(atom_features), num_classes=x_classes).float()    # 원자 특징을 원-핫 인코딩하여 텐서로 만듦

    edge_indices, edge_features = [], []                                       # 엣지(결합)의 인덱스와 특징을 저장할 리스트 초기화
    for bond in mol.GetBonds():                                                 # 분자의 모든 결합에 대해 반복
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()                     # 결합의 시작 원자와 끝 원자 인덱스를 가져옴
        edge_indices.extend([(i, j), (j, i)])                                   # 무방향 그래프이므로 양방향으로 엣지 추가
        bond_type = int(bond.GetBondTypeAsDouble())                             # 결합 종류(단일, 이중 등)를 정수로 변환
        edge_features.extend([bond_type, bond_type])                            # 양방향 엣지에 대해 결합 특징 추가

    if not edge_indices:                                                        # 분자에 결합이 없는 경우 (원자가 하나인 경우 등)
        edge_index = torch.empty((2, 0), dtype=torch.long)                      # 빈 엣지 인덱스 텐서 생성
        edge_attr = torch.empty((0, e_classes), dtype=torch.float)              # 빈 엣지 속성 텐서 생성
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() # 엣지 인덱스 리스트를 텐서로 변환
        edge_attr = F.one_hot(torch.tensor(edge_features), num_classes=e_classes).float() # 엣지 특징을 원-핫 인코딩하여 텐서로 만듦

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles_string) # 최종적으로 PyG Data 객체 생성
    return data                                                                 # 생성된 Data 객체 반환

# --- 노이즈가 적용된 그래프 데이터로부터 분자를 재구성하는 헬퍼 함수 ---
def get_mol_from_data(data_dict, atom_decoder):
    '''
    노이즈가 적용된 데이터 딕셔너리로부터 RDKit 분자 객체를 재구성합니다.

    Args:
        data_dict (dict): 'X_t'와 'E_t' 키를 포함하는, 노이즈가 적용된 데이터 딕셔너리.
        atom_decoder (dict): 정수 인덱스를 원자 기호(예: 'C', 'H')로 변환하는 딕셔너리.

    Returns:
        rdkit.Chem.Mol: 재구성된 RDKit 분자 객체.
    '''
    # 노이즈가 적용된 원자(X_t)와 결합(E_t) 행렬에서 가장 확률이 높은 클래스를 선택하여 분자 재구성
    mol = build_molecule_with_partial_charges(data_dict['X_t'][0].argmax(dim=-1),
                                            data_dict['E_t'][0].argmax(dim=-1),
                                            atom_decoder)
    return mol                                                                  # 재구성된 분자 객체 반환


# --- 모델 초기화를 위한 더미(Dummy) DatasetInfos 클래스 ---
class DummyDatasetInfos:
    '''
    실제 데이터셋을 로드하지 않고 확산 모델을 초기화하는 데 필요한
    최소한의 데이터셋 정보를 제공하는 더미 클래스입니다.
    '''
    def __init__(self, x_classes, e_classes, max_nodes):
        self.atom_decoder = {1: 'H', 2: 'C', 3: 'N', 4: 'O', 5: 'F'}            # 정수 인덱스를 원자 기호로 디코딩하는 맵
        self.input_dims = {'X': x_classes, 'E': e_classes, 'y': 0}               # 모델의 입력 차원 정보
        self.output_dims = {'X': x_classes, 'E': e_classes, 'y': 0}              # 모델의 출력 차원 정보
        self.nodes_dist = torch.distributions.Categorical(torch.tensor([0.1] * max_nodes)) # 노드 수 분포 (여기서는 임의의 값)
        self.n_nodes = torch.ones(max_nodes + 1) / (max_nodes + 1)                  # 노드 개수에 대한 균일 분포
        self.node_types = torch.ones(x_classes) / x_classes                         # 원자 종류에 대한 균일 분포
        self.edge_types = torch.ones(e_classes) / e_classes                         # 결합 종류에 대한 균일 분포
        self.valency_distribution = torch.ones(max_nodes * 2) / (max_nodes * 2)     # 원자가(valency) 분포 (여기서는 임의의 값)
        self.max_n_nodes = max_nodes                                                # 최대 노드 수

    def complete_infos(self, a, b):                                                 # 모델 초기화에 필요하지만 여기서는 사용되지 않는 함수
        pass                                                                        # 아무 작업도 수행하지 않음

# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # 1. CSV 파일에서 분자 로드
    csv_path = os.path.join(project_root, 'Practice_SRC', 'datasets', 'train_50.csv') # 시각화할 분자가 포함된 CSV 파일 경로
    try:                                                                            # 파일 읽기 시 발생할 수 있는 오류 처리
        df = pd.read_csv(csv_path)                                                  # pandas를 사용하여 CSV 파일을 데이터프레임으로 읽음
        print("pandas read csv_path:", csv_path)
        smiles_to_visualize = df['smiles'].iloc[0]                                  # 데이터프레임의 'smiles' 열에서 첫 번째 SMILES 문자열을 가져옴
        print("smiles_to_visualize:", smiles_to_visualize)
    except FileNotFoundError:                                                       # 파일을 찾을 수 없을 때의 예외 처리
        print(f"Error: Could not find {csv_path}")                                # 오류 메시지 출력
        sys.exit(1)                                                                 # 프로그램 종료
    except Exception as e:                                                          # 그 외 다른 예외 처리
        print(f"Error reading CSV: {e}")                                          # 오류 메시지 출력
        sys.exit(1)                                                                 # 프로그램 종료

    # 2. 데이터 및 모델 설정 준비
    X_CLASSES, E_CLASSES = 6, 5                                                 # 원자(H,C,N,O,F + 마스크)와 결합(없음,단일,이중,삼중,방향족) 종류의 수
    MAX_NODES = 128                                                             # 분자 내 최대 원자(노드) 수 설정

    dataset_infos = DummyDatasetInfos(x_classes=X_CLASSES, e_classes=E_CLASSES, max_nodes=MAX_NODES) # 더미 데이터셋 정보 객체 생성
    print("dataset_infos",dataset_infos.__dict__)
    atom_decoder = dataset_infos.atom_decoder                                   # 정수 인덱스를 원자 기호로 변환하는 디코더 가져오기

    # SMILES를 모델이 요구하는 밀집(dense) 텐서 형식으로 변환
    original_graph = smiles_to_torch_geometric(smiles_to_visualize, X_CLASSES, E_CLASSES) # SMILES를 PyG 그래프 객체로 변환
    print("original_graph",original_graph)
    if original_graph is None:                                                  # 변환에 실패한 경우
        print(f"Could not process SMILES: {smiles_to_visualize}")                 # 오류 메시지 출력
        sys.exit(1)                                                                 # 프로그램 종료
    
    batch_indices = torch.zeros(original_graph.x.size(0), dtype=torch.long)     # 모든 노드가 단일 그래프에 속함을 나타내는 배치 인덱스 생성
    print("batch_indices",batch_indices.size())
    data, node_mask = utils.to_dense(original_graph.x, original_graph.edge_index, original_graph.edge_attr, batch_indices) # 그래프를 밀집 텐서로 변환
    data = data.mask(node_mask)                                                 # 패딩된 노드를 가리기 위한 마스크 적용

    print("--- Original Molecule ---")                                           # 원본 분자 정보 출력 시작
    original_mol = Chem.MolFromSmiles(smiles_to_visualize)                      # 시각화를 위해 SMILES로부터 RDKit 분자 객체 다시 생성
    print(f"SMILES: {smiles_to_visualize}")                                      # 원본 SMILES 문자열 출력
    Draw.MolToImage(original_mol).save(os.path.join(script_dir, 'noise_step_original.png')) # 원본 분자 이미지를 파일로 저장
    print(f"Saved original molecule to {os.path.join(script_dir, 'noise_step_original.png')}") # 저장 완료 메시지 출력

    # 3. 더미 확산 모델 초기화
    dummy_model_cfg = OmegaConf.create({                                        # OmegaConf를 사용하여 모델 설정을 계층적으로 정의
        'general': {
            'name': 'dummy_noise_sim',
            'log_every_steps': 50,
            'number_chain_steps': 10,
            'sample_every_val': 5,
            'samples_to_generate': 10,
            'samples_to_save': 5,
            'chains_to_save': 2,
            'final_model_samples_to_generate': 10,
            'final_model_samples_to_save': 5,
            'final_model_chains_to_save': 2
        },
        'model': {
            'n_layers': 2,
            'hidden_mlp_dims': {'X': 32, 'E': 32, 'y': 32},
            'hidden_dims': {'dx': 32, 'de': 32, 'dy': 32, 'n_head': 4, 'dim_ffX': 128, 'dim_ffE': 128, 'dim_ffy': 128},
            'diffusion_steps': 500,
            'diffusion_noise_schedule': 'cosine',
            'transition': 'uniform',
            'lambda_train': [1.0, 1.0, 1.0]
        }
    })
    
    # 모델 초기화에 필요한 인수들의 더미 인스턴스 생성
    train_metrics = TrainAbstractMetricsDiscrete()                              # 더미 훈련 지표 객체
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles=None) # 더미 샘플링 지표 객체
    visualization_tools = MolecularVisualization(False, dataset_infos=dataset_infos) # 더미 시각화 도구 객체
    extra_features = DummyExtraFeatures()                                       # 더미 추가 특징 객체
    domain_features = DummyExtraFeatures()                                      # 더미 도메인 특징 객체

    # 설정과 더미 객체들을 사용하여 확산 모델 초기화
    model = DiscreteDenoisingDiffusion(cfg=dummy_model_cfg, dataset_infos=dataset_infos,
                                       train_metrics=train_metrics, sampling_metrics=sampling_metrics,
                                       visualization_tools=visualization_tools, extra_features=extra_features,
                                       domain_features=domain_features)

    # 4. 다른 타임스텝에서 노이즈 적용
    print("\n--- Applying Noise at Different Timesteps ---")                      # 노이즈 적용 과정 출력 시작
    noise_timesteps = [0, 10, 50, 100, 250, 499]                                # 노이즈를 적용하고 시각화할 타임스텝 리스트

    for t in noise_timesteps:                                                   # 각 타임스텝에 대해 반복
        # apply_noise 함수에 전달할 타임스텝 t를 배치 크기만큼 복제하여 텐서 생성
        t_tensor = torch.full((data.X.size(0),), t, dtype=torch.long)
        
        # 모델의 apply_noise 함수를 호출하여 현재 타임스텝 t에 해당하는 노이즈를 데이터에 적용
        noisy_data = model.apply_noise(data.X, data.E, data.y, node_mask, t_int=t_tensor)
        
        # 노이즈가 적용된 데이터로부터 분자 객체를 재구성
        noisy_mol = get_mol_from_data(noisy_data, atom_decoder)
        
        # 이미지 저장
        filename = os.path.join(script_dir, f'noise_step_{t}.png')               # 저장할 이미지 파일 이름 생성
        Draw.MolToImage(noisy_mol, size=(300,300)).save(filename)                # 노이즈가 적용된 분자 이미지를 파일로 저장
        print(f"Saved noisy molecule at t={t} to {filename}")                    # 저장 완료 메시지 출력

    print("\nNoise application simulation finished.")                         # 시뮬레이션 완료 메시지 출력
