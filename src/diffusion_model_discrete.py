import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import csv
import os

from src.models.transformer_model import GraphTransformer
from src.diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from src.metrics.train_metrics import TrainLossDiscrete
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils

def log_to_csv(log_dict, file_path='logs/val_metrics.csv'):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_dict)


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()
        ### dataset 정보 가져오기 ###
        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist # 노드 개수에 대한 이산 확률 분포(discrete distribution)

        ### 주요 설정값 ###
        self.cfg = cfg # 모델의 주요 설정 값 config 파일 저장
        self.name = cfg.general.name # 현재 모델 실험 이름
        self.model_dtype = torch.float32 # 텐서 타입
        self.T = cfg.model.diffusion_steps # diffusion stp 수

        ### 노드 및 엣지, 그래프 특징들 ###
        self.Xdim = input_dims['X'] # 노드
        self.Edim = input_dims['E'] # 엣지
        self.ydim = input_dims['y'] # 그래프 전역
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist # 노드 개수에 대한 이산 확률 분포(discrete distribution)

        self.dataset_info = dataset_infos

        ### 학습 중 사용할 손실 함수 객체를 초기화 ###
        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        ## KL Divergence Loss , log ##
        # src.metrics.abstract_metrics 확인하기
        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_y_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_y_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMetric()

        ### 학습 중 사용하는 정량 평가 지표 계산 도구###
        # 일반적으로 노드/엣지 예측 정확도, loss, KL divergence 등을 로그로 남깁니다.
        # 내부적으로 TrainMolecularMetricsDiscrete 클래스를 사용합니다.
        self.train_metrics = train_metrics

        ### 샘플링된 분자들에 대해 화학적으로 유효한 분자인지 평가하는 메트릭 계산기입니다.
        #유효성(validity), 고유성(uniqueness), 다양성(diversity), novelty 등의 지표 계산에 사용됩니다.
        #보통 sampling 후 self.sampling_metrics(samples) 형태로 사용됩니다.
        self.sampling_metrics = sampling_metrics

        ### 생성된 분자 구조를 이미지로 시각화 ###
        # src/analysis/visualization.py에 정의되어 있음
        self.visualization_tools = visualization_tools

        ### 노드/엣지/전역 특성 외에 추가적으로 입력되는 ML용 feature 생성기 ###
        self.extra_features = extra_features

        ### 화학 도메인 지식 기반의 특화된 feature 생성 도구 ###
        self.domain_features = domain_features

        ### denoising을 위한 예측을 수행하는 모델 ###
        self.model = GraphTransformer(n_layers=cfg.model.n_layers, # 몇개의 Trasnformer Layer 사용할 것인지
                                      input_dims=input_dims, # 입력 차원 Node, Edge, Graph전역 Y
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims, #MLP 에서 사용하는 차원
                                      hidden_dims=cfg.model.hidden_dims, # Transformer 내 hidden 차원
                                      output_dims=output_dims, # 출력 차원 Node, Edge, Graph전역 Y
                                      act_fn_in=nn.ReLU(), # 입력용 activation function
                                      act_fn_out=nn.ReLU()) # 출력용 activation function

        ### Discrete diffusion 과정에서 timestep별 노이즈 스케줄(β, ᾱ 등)을 정의 ###
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        ### 모든 클래스에 대해 **동일한 확률(균등)**로 노이즈가 분포된다고 가정 ###
        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            # 노드 type 이 5개다 CNOSP -> C일 확률은 1/5로 설정됨
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        ### 실제 데이터셋의 node/edge class 빈도수를 기반으로 확률 분포 생성###
        elif cfg.model.transition == 'marginal':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types) # node type 비율

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types) # edge type 비율
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")

            # 위에서 계산한 실 데이터 기반 확률 분포를 사용하여 전이 모델 (transition model) 설정
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)

            # limit_dist 설정
            # diffusion 과정의 최종 단계 𝑡=𝑇 에 도달했을 때, 데이터가 수렴해야 하는 확률 분포
            # PlaceHolder : x_marginals, e_marginals, torch.ones(...) / ... ← 이 세 개의 분포를 하나로 묶어서 다루기
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)


        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics]) # PyTorch Lightning에서 모델 초기화 시 전달된 모든 인자(config, kwargs 등)를 자동 저장
        self.start_epoch_time = None # 각 epoch 시작 시 시간을 기록하기 위한 변수
        self.train_iterations = None # 한 epoch에 train/val 데이터셋을 몇 번 반복하는지 (iteration 수)
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps  # 몇 step마다 로그를 출력할지 설정
        self.number_chain_steps = cfg.general.number_chain_steps # diffusion 샘플링 과정에서 몇 단계의 중간 결과(chain)를 저장할지
        self.best_val_nll = 1e8 # 현재까지 관측된 최소 validation NLL (Negative Log Likelihood) 를 저장
        self.val_counter = 0 # validation epoch 횟수 추적용 카운터

    def training_step(self, data, i):

        ### PyG의 sparse 형식 그래프 데이터를 dense 텐서 형식으로 변환 ###
        # 노드 feature x를 (B, N, F)로 변환, 가장 큰 그래프에 맞춰서 여러 그래프를 패딩 + node_mask 생성
        # 자기 자신에 대한 엣지를 제거, edge_index -> 인접행렬, 엣지가 없는 위치를 "no-edge 클래스"로 표시
        # 노드/엣지 텐서를 하나로 묶음
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)

        dense_data = dense_data.mask(node_mask) # 유효한 노드만 구분하기 위한 binary mask

        # 노이즈 주입 (forward diffusion 단계)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask) # --> 함수확인하기

        # extra_features, domain_features, t 등을 계산하여 입력에 추가 # 아래쪽 라인 확인
        extra_data = self.compute_extra_data(noisy_data)

        # 모델 예측
        pred = self.forward(noisy_data, extra_data, node_mask)

        # 손실 계산
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)

        # 학습 메트릭 기록
        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    # 학습에 사용할 옵티마이저를 정의 (여기서는 AdamW)
    def configure_optimizers(self):
        """ self.cfg.train.lr과 weight_decay는 설정 파일에서 지정 """
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)
    # 전체 학습 시작 시 한 번만 호출됨
    # train 데이터의 전체 반복 수 계산 (한 epoch 당 step 수)
    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        print("Size of the input features", self.Xdim, self.Edim, self.ydim)

    # 매 epoch 시작 시 호출
    # 시간 측정 시작 (start_epoch_time)
    # 학습 손실 객체 및 메트릭 객체 초기화
    def on_train_epoch_start(self) -> None:
        print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    # 매 epoch 종료 시 호출
    # 손실/정확도 등의 epoch 단위 로그 기록 (wandb.log() 내부에서 수행될 가능성 높음)
    # 경과 시간 포함
    def on_train_epoch_end(self) -> None:
        self.train_loss.log_epoch_metrics(self.current_epoch, self.start_epoch_time)
        self.train_metrics.log_epoch_metrics(self.current_epoch)

    # validation epoch 시작 전 호출
    # validation에서 사용하는 모든 메트릭 객체 초기화
    # 예: NLL, KL divergence, log-likelihood, 샘플링 관련 메트릭 등
    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_y_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_y_logp.reset()
        self.sampling_metrics.reset()


    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch) # PyG의 sparse 형식 그래프 데이터를 dense 텐서 형식으로 변환
        dense_data = dense_data.mask(node_mask)

        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask) # 노이즈 주입 (forward diffusion 단계)
        extra_data = self.compute_extra_data(noisy_data) # extra_features, domain_features, t 등을 계산하여 입력에 추가

        pred = self.forward(noisy_data, extra_data, node_mask) # 모델 예측
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False) # 손실 계산
        return {'loss': nll}

    # validation epoch 시작 후 호출
    #def validation_epoch_end(self, outs) -> None:
    def on_validation_epoch_end(self) -> None:

        # Metric 계산 및 로그 출력
        # nll : Negative Log-Likelihood
        metrics = [self.val_nll.compute(), self.val_X_kl.compute(), self.val_E_kl.compute(),
                   self.val_y_kl.compute(), self.val_X_logp.compute(), self.val_E_logp.compute(),
                   self.val_y_logp.compute()]
        print("validation metrics:", metrics)
        metrics = {
            "epoch": self.current_epoch,
            "val_nll": metrics[0].item(),
            "val_X_kl": metrics[1].item(),
            "val_E_kl": metrics[2].item(),
            "val_y_kl": metrics[3].item(),
            "val_X_logp": metrics[4].item(),
            "val_E_logp": metrics[5].item(),
            "val_y_logp": metrics[6].item()
        }
        print("Validation metrics:", metrics)
        log_to_csv(metrics, file_path='logs/val_metrics.csv')

        val_nll_key = "val_nll"
        val_X_kl_key = f"val_X_kl"
        val_E_kl_key = f"val_E_kl"
        val_y_kl_key = f"val_y_kl"
        print(f"Epoch {self.current_epoch}: Val NLL {metrics[val_nll_key] :.2f} -- Val Atom type KL {metrics[val_X_kl_key] :.2f} -- ",
              f"Val Edge type KL: {metrics[val_E_kl_key] :.2f} -- Val Global feat. KL {metrics[val_y_kl_key] :.2f}\n")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        # Checkpoint 평가 기준 기록
        # 궁금 할 시 main.py 안 checkpoint_callback = ModelCheckpoint 확인해 보기,
        val_nll = metrics[val_nll_key]

        # PyTorch Lightning 내부 logger에 "val/epoch_NLL"이라는 이름으로 값 저장 | callback들이 읽을 수 있는 핵심 로그 키(key)
        self.log("val/epoch_NLL", val_nll)

        # callback을 통해 val 이 nll 기존 보다 낮을 경우 best_val_nll에 저장
        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        ## 샘플 생성 조건 확인 및 실행 ##
        self.val_counter += 1
        # 매 sample_every_val번 validation epoch마다 한 번
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()

            ## 샘플 생성 개수 설정 config 파일 읽기##
            samples_left_to_generate = self.cfg.general.samples_to_generate # 얼마나 많은 샘플을 만들 것인지
            samples_left_to_save = self.cfg.general.samples_to_save # 그중 저장할 샘플 수
            chains_left_to_save = self.cfg.general.chains_to_save # sampling 과정(chain) 저장할 수 등 설정
            samples = []

            ident = 0
            ## 샘플 생성 루프 ##
            while samples_left_to_generate > 0: # 총 samples_to_generate 만큼 생성될 때까지 반복
                bs = 2 * self.cfg.train.batch_size # 한 번에 최대 2 × batch_size 만큼 생성
                to_generate = min(samples_left_to_generate, bs) # 두 변수중 작은 수 만큼 생성되도록 변수지정
                to_save = min(samples_left_to_save, bs) # 두 변수중 작으 수만큼 저장되도록 변수지정
                chains_save = min(chains_left_to_save, bs) # 두 변수중 작은 수만큼 chains (샘플링 중간과정)가 저장됨

                # 실제 샘플 생성은 self.sample_batch(...) 함수에서 수행
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))

                # count 에 대한 변화
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save

            # 생성된 샘플 평가
            print("Computing sampling metrics...")
            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False)
            print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            self.sampling_metrics.reset()

    # 테스트 시작 전에 모든 metric을 초기화함
    def on_test_epoch_start(self) -> None:
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_y_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_y_logp.reset()

    # 테스트용 한 배치에서의 forward → loss 계산
    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch) # PyG의 sparse 형식 그래프 데이터를 dense 텐서 형식으로 변환
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask) # 노이즈 주입 (forward diffusion 단계)
        extra_data = self.compute_extra_data(noisy_data) # extra_features, domain_features, t 등을 계산하여 입력에 추가
        pred = self.forward(noisy_data, extra_data, node_mask) # 예측 수행
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True) # loss 계산,val 에서는 true 였음
        return {'loss': nll}

    def test_epoch_end(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        ## Metric 계산 및 로그 기록##
        # 각 테스트 지표 (NLL, KL, logP 등)를 계산하고 wandb에 기록
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_y_kl.compute(), self.test_X_logp.compute(), self.test_E_logp.compute(),
                   self.test_y_logp.compute()]

        metrics = {
            "epoch": self.current_epoch,
            "test_nll": metrics[0].item(),
            "test_X_kl": metrics[1].item(),
            "test_E_kl": metrics[2].item(),
            "test_y_kl": metrics[3].item(),
            "test_X_logp": metrics[4].item(),
            "test_E_logp": metrics[5].item(),
            "test_y_logp": metrics[6].item()
        }
        test_nll_key = "test_nll"
        test_X_kl_key = "test_X_kl"
        test_E_kl_key = "test_E_kl"
        test_y_kl_key = "test_y_kl"

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[test_nll_key] :.2f} -- Test Atom type KL {metrics[test_X_kl_key] :.2f} -- ",
              f"Test Edge type KL: {metrics[test_E_kl_key] :.2f} -- Test Global feat. KL {metrics[test_y_kl_key] :.2f}\n")

        test_nll = metrics[0]
        #wandb.log({"test/epoch_NLL": test_nll}, commit=False)
        print(f'Test loss: {test_nll :.4f}')

        # configuration 파일에서 얼마나 샘플 생성할지 #
        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate # 얼마나 많은 샘플을 만들 것인지
        samples_left_to_save = self.cfg.general.final_model_samples_to_save  # 그중 저장할 샘플 수
        chains_left_to_save = self.cfg.general.final_model_chains_to_save # sampling 과정(chain) 저장할 수 등 설정

        samples = []

        ## 샘플 생성 루프 ##
        id = 0
        while samples_left_to_generate > 0:
            print(f'Samples left to generate: {samples_left_to_generate}/'
                  f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size # 한번에 batch size * 2 만큼 생성
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        # 샘플링 성능 평가
        # validity (화학적으로 유효한지)
        # uniqueness (중복 없는지)
        # novelty (학습 데이터에 없는 새로운 구조인지)
        # diversity, FCD 등
        print("Computing sampling metrics...")
        self.sampling_metrics.reset()
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True)
        self.sampling_metrics.reset()
        print("Done.")

    ##### "최종적으로 생성된 diffusion의 마지막 단계 T 노이즈 분포" 와 "미리 데이터셋에서 만들어진 사전 분포" 사이에서 얼마나 차이나는지를 측정 #####
    def kl_prior(self, X, E, y, node_mask): # compute_val_loss에서 사용됨
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """


        ### Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device) # 1로 채워진 텐서 (bs, 1)
        Ts = self.T * ones # "1로 채워진 텐서"에 "전체 diffusion step 수" 곱하기
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1) , PredefinedNoiseScheduleDiscrete, step 1부터 t까지 곱해진것, 원본 정보가 얼마나 유지되었는지

        ### 확률 전이 행렬 생성 Q t bar Q̅_T = Qtb
        # DiscreteUniformTransition 또는 MarginalUniformTransition
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device) # transition matrix 생성 | 각 입력 클래스 → 각 출력 클래스 로 갈 확률 분포

        ### Compute transition probabilities
        # Qtb 곱하여 확률 분포 계산 = "최종적으로 생성된 diffusion의 마지막 단계 T 노이즈 분포"
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        proby = y @ Qtb.y if y.numel() > 0 else y
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        # limit_dist : Node 및 Edge Class 에 대한 확률 분포 = "미리 데이터셋에서 만들어진 사전 분포"
        # 노드 클래스별확률분포(C:0.2, O:0.8), 엣지 클래스별 확률분포(단일:0.8, 이중:0.2) 이런 형태를 probX나 probE 등 모델 예측 결과와 같게 만들기
        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)
        uniform_dist_y = torch.ones_like(proby) / self.ydim_output

        ### Make sure that masked rows do not contribute to the loss
        # 마스크 적용 | padding된 node/edge는 KL 계산에서 제외
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        # probX = 모델이 예측한 확률 분포
        # limit_dist_X = 사전 분포
        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
        kl_distance_y = F.kl_div(input=proby.log(), target=uniform_dist_y, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E) + \
               diffusion_utils.sum_except_batch(kl_distance_y)

    ### Diffusion 단계 손실 (전체 시점 통합) ###
    # 모델이 각 시점 t에서 z_t → z_{t-1} 를 얼마나 잘 예측하는지
    # 각 샘플별로 여러 t에서 평균적으로 얼마나 잘 예측했는지를 나타냄
    # 다시 한번 보기 이해 잘 x
    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test): # compute_val_loss에서 사용됨
        ### 모델의 예측 확률화
        # 모델의 출력 logits → softmax로 확률로 변환
        pred_probs_X = F.softmax(pred.X, dim=-1) # node X features (매우 +나 -인 수 가능) -> node X features (0~1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        ### 전이 행렬 계산
        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device) # t 시점의 전이 확률 = t 시점까지의 Qt 곱하여 만든 Qtb | (node X node)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device) # t-1 시점의 전이 확률 = t-1 시점까지의 Qt 곱하여 만든 Qsb
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device) # 직접적인 forward transition = t -> t-1 로 가기위한 Qt

        ### Compute distributions to compare with KL
        bs, n, d = X.shape
        # Ground truth 분포 계산 : 모델이 예측해야 할 정답 분포 # diffusion_utils.posterior_distributions 확인하기
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        # 모델의 출력으로 부터 예측 분포 계산 # diffusion_utils.posterior_distributions 확인하기
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows | 마스크 적용, padding 된 노드 / 엣지를 KL 계산에서 제외
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        # from src.metrics.abstract_metrics import SumExceptBatchKL
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        kl_y = (self.test_y_kl if test else self.val_y_kl)(prob_true.y, torch.log(prob_pred.y)) if pred_probs_y.numel() != 0 else 0
        return kl_x + kl_e + kl_y

    def reconstruction_logp(self, t, X, E, y, node_mask): # compute_val_loss 에서 사용됨
        ### Compute noise values for t = 0. t=0일 때의 transition matrix 계산 ###
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        # 원래 입력 X, E에 Q0를 적용하여 확률 분포 계산 | 각 노드/엣지가 특정 클래스일 확률
        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        # 이 분포로부터 샘플링하여 X0, E0, y0 생성
        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        ### Predictions | 이 noisy한 입력으로 다시 prediction 수행 ###
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        ### Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        ### Set masked rows to arbitrary values that don't contribute to loss
        # 마스크로 비유효한 노드/엣지는 손실에 영향을 미치지 않도록 임의 값(=uniform)으로 설정.
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        # 자기 자신으로의 엣지(self-loop)는 무시
        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        # softmax 확률들을 반환
        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    ##### apply_noise 설명 #####
    # 각 분자 x에 대해 무작위로 하나의 timestep t를 선택하여 해당 시점 zt 의 noisy 데이터 를 만들고, 이로부터 clean data x를 복원하는 것을 학습합니다.
    # 분자 (샘플) inde, 샘플링된 t
    # 0	37
    # 1	92
    # 2	14
    # 3	51
    #
    # 모든 t에 대해 한 번에 학습하는 건 비효율적
    # 대신 매 epoch마다 다른 t를 랜덤하게 선택 → 전체적으로 다양한 시점을 학습하게 됨
    # 이것이 "random timestep training", DDPM의 핵심 전략 중 하나
    ####################################

    def apply_noise(self, X, E, y, node_mask): # 위쪽에 train, val, test 쪽에서 사용됨
        """ Sample noise and apply it to the data. """

        ### Sample a timestep t. | diffusion 시점 t 샘플링
        ### When evaluating, the loss for t=0 is computed separately | 학습 중에는 t=0∼T, 테스트/validation 시에는 t=1∼T만 (t=0은 평가용)
        lowest_t = 0 if self.training else 1
        # lowest_t ~ t+1 사이의 정수중 랜덤/ 각 샘플에 대해 1개의 t값을 할당 : size=(X.size(0), 1) : batch size,1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1) | lowest_t 에서 self.T + 1 까지
        s_int = t_int - 1 # 현재 시점보다 이전시점

        t_float = t_int / self.T
        s_float = s_int / self.T

        ### beta_t and alpha_s_bar are used for denoising/loss computation | 알파/베타 값 계산 (스케줄링)
        # self.noise_schedule = src.diffusion.noise_schedule PredefinedNoiseScheduleDiscrete
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        ## 전이 확률 행렬 Qt 생성
        # from src.diffusion.noise_schedule DiscreteUniformTransition 또는 MarginalUniformTransition
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        ### Compute transition probabilities | 노이즈 추가
        # @ : 행렬 곱 연산자, torch.matmul() 또는 np.matmul()을 호출
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        # 원-핫 인코딩
        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False): # 위쪽 val, test 에서 사용
        """Computes an estimator for the variational lower bound, or the simple loss (MSE).
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        ### 1. ###
        # 사전 확률분포 𝑝(𝑁) : node count 분포, 즉 그래프, 분자의 노드 개수는 몇개 일것인가? 에 대한 분포
        # 샘플별 노드 수의 우도 (likehood)
        N = node_mask.sum(1).long() # 각 샘플의 유효 노드 수
        log_pN = self.node_dist.log_prob(N) # 사전 분포로부터 확률(log), 노드 개수에 대한 이산 확률 분포(discrete distribution)

        ### 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero. ###
        # KL Divergence (prior vs posterior)
        kl_prior = self.kl_prior(X, E, y, node_mask) # 초기 노이즈 분포가 prior와 얼마나 다른지를 측정

        ### 3. Diffusion loss ###
        # Diffusion 단계 손실 (전체 시점 통합)
        # 모델이 각 시점 t에서 z_t → z_{t-1} 를 얼마나 잘 예측하는지
        # 각 샘플별로 여러 t에서 평균적으로 얼마나 잘 예측했는지를 나타냄
        # 보통 CrossEntropy 또는 MSE로 계산
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        ### 4. Reconstruction loss ###
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        # 모델이 t=0 시점에서 원래의 데이터 (X, E, y) 를 얼마나 잘 복원하는지
        prob0 = self.reconstruction_logp(t, X, E, y, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log()) + \
                      self.val_y_logp(y * prob0.y.log())

        ### Combine terms | NLL 통합 계산 ###
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        ### Update NLL metric object and return batch nll ###
        # from src.metrics.abstract_metrics import NLL
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch

        # wandb 로깅
        #wandb.log({"kl prior": kl_prior.mean(),
        #           "Estimator loss terms": loss_all_t.mean(),
        #           "log_pn": log_pN.mean(),
        #           "loss_term_0": loss_term_0,
        #           'test_nll' if test else 'val_nll': nll}, commit=False)

        log_data = {
            "epoch": self.current_epoch,
            "kl_prior": kl_prior.mean().item(),
            "loss_all_t": loss_all_t.mean().item(),
            "log_pn": log_pN.mean().item(),
            "loss_term_0": loss_term_0.item() if isinstance(loss_term_0, torch.Tensor) else loss_term_0,
            "nll": nll.item()
        }
        log_to_csv(log_data, file_path="logs/val_log.csv")


        return nll

    # 모델 예측(denoising) 을 수행하는 핵심 부분
    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask) # self.model = GraphTransformer


    # 이 sample_batch 함수는 Diffusion 모델에서 molecule을 샘플링(생성)하는 전체 과정을 담당
    # Diffusion 모델의 학습이 끝난 뒤,
    # 이 함수를 호출하면 노이즈 상태 z_T로부터 z₀까지 역추론(reverse diffusion) 과정을 통해 새로운 분자 구조를 생성
    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None): # test_epoch_end, val_epoch_end
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s, predicted_graph = self.sample_p_zs_given_zt(t_norm, X, E, y, node_mask,
                                                                                       last_step=s_int == 100)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y



        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            if i < 3:
                print("Example of generated E: ", atom_types)
                print("Example of generated X: ", edge_types)

        predicted_graph_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            predicted_graph_list.append([atom_types, edge_types])


        # Visualize chains
        if self.visualization_tools is not None:
            print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.visualization_tools.visualize(result_path, predicted_graph_list, save_final, log='predicted')
            print("Done.")

        return molecule_list

    # 이 함수 sample_p_zs_given_zt()는 reverse diffusion 과정의 핵심으로,
    # 주어진 noisy 상태 𝑧𝑡에서 한 단계 전 상태𝑧𝑠 (s=t−1)를 샘플링합니다.
    def sample_p_zs_given_zt(self, t, X_t, E_t, y_t, node_mask, last_step: bool): # sample_batch 에서 사용
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        if last_step:
            predicted_graph = diffusion_utils.sample_discrete_features(pred_X, pred_E, node_mask=node_mask)

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t), \
               predicted_graph if last_step else None

    def compute_extra_data(self, noisy_data): #training_setp, Validation_step, test_step, reconstruction_logp, sample_p_zs_given_zt
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        # src -> diffusion -> extra features -> DummyExtraFeatures Class, ExtraFeatures Class 내에서 ExtraFeatures type에 따라 다르게 계산
        extra_features = self.extra_features(noisy_data)
        # src -> diffusion -> extra features molecular -> ExtraMolecularFeatures Class 내에서 ExtraFeatures type에 따라 다르게 계산
        extra_molecular_features = self.domain_features(noisy_data)

        # 위에서 정의된 Class 활용해 features 계산
        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t'] # 현재 diffusion timestep
        extra_y = torch.cat((extra_y, t), dim=1) # 전역 특성에 t 값을 추가로 붙임

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
