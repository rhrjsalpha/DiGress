import torch
from torch import Tensor
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples

class MultiTaskHypervolumeAcquisition(MCAcquisitionFunction):
    def __init__(self, model, ref_point: Tensor, sampler, **kwargs):
        super().__init__(model=model, sampler=sampler)
        self.ref_point = ref_point

    def forward(self, X: Tensor) -> Tensor:
        q = X.shape[-2]
        device = X.device

        # Task 0: MSE
        X0 = torch.cat([X, torch.zeros(q, 1, device=device)], dim=-1)
        posterior_0 = self.model.posterior(X0)
        samples_0 = self.sampler(posterior_0)  # (mc_samples, q, 1)

        # Task 1: MAE
        X1 = torch.cat([X, torch.ones(q, 1, device=device)], dim=-1)
        posterior_1 = self.model.posterior(X1)
        samples_1 = self.sampler(posterior_1)  # (mc_samples, q, 1)

        # Combined objective samples: (mc_samples, q, 2)
        obj_samples = torch.cat([samples_0, samples_1], dim=-1)

        # Hypervolume improvement를 위해 참조점 기준 dominated 영역 계산
        hv = Hypervolume(ref_point=self.ref_point)

        # 평균 개선량 계산 (q=1일 때)
        if q == 1:
            hv_improvements = []
            for i in range(obj_samples.shape[0]):
                y = obj_samples[i]  # shape: (q=1, 2)
                hv_improvements.append(hv.compute(torch.cat([self.ref_point.unsqueeze(0), y], dim=0)) - hv.compute(self.ref_point.unsqueeze(0)))
            return torch.tensor(hv_improvements, device=device).mean().unsqueeze(0)

        # q > 1인 경우는 적절한 generalized HV 계산 필요 (간단히 sum of improvements 등 가능)
        return -obj_samples.mean(dim=0).sum(dim=-1)



# 모델을 "2번 학습하는 건 아니고", "2번 예측하는 구조
# model list gp => 2개의 모델 [input = hyperparam, output=cv_avg_1], [input = hyperparam, output=cv_avg_2]
# multi task gp => 1개의 모델 [input = hyperparam*2 + task (0 또는 1), output = cv_avg 두개 중하나], -> 여기서는 한번 학습 한후에 0, 1 을 넣어 2번 불러냄