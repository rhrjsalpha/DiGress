# -------------------------------------------------------------
# 독립 실행 예시: 무조건부 posterior(prob_X)와
# 회귀기(guidance) 기반 마스크 p_eta_x가 어떻게 섞여서
# 최종 z_{t-1} 클래스를 뽑는지 보여준다.
# -------------------------------------------------------------

import torch
import torch.nn.functional as F

# 출력 설정(지수표기 끄기, 소수점 4자리)
torch.set_printoptions(precision=4, sci_mode=False)

# ------------------------------------------------------------------
# 1) 장난감 설정
#    그래프 1개(bs=1), 노드 1개(n=1), 원자 클래스 3개(C, O, N)
# ------------------------------------------------------------------
bs, n, d = 1, 1, 3

# (1) 모델이 noisy 입력 G_t에 대해 예측한 로짓
pred_logits_x = torch.tensor([[[2.0, 0.5, -1.0]]])   # 모양: (bs, n, d)
print("pred_logits_x:", pred_logits_x)

# softmax → 예측 확률
pred_probs_x = F.softmax(pred_logits_x, dim=-1)
print("softmax(pred_logits_x):", pred_probs_x)

# (2) 무조건부 posterior  p_θ(x_{t-1} | G_t)
#     예시이므로 임의의 값 사용
posterior_xt = torch.tensor([[[0.15, 0.6, 0.25]]])   # (bs, n, d)
print("무조건부 posterior prob_X:", posterior_xt)

# (3) regressor guidance 설정
lambda_guidance = 0.5  # λ 값 (크면 조건을 더 강하게 반영)

# MSE에 대한 그래디언트(예시 값)
# 양수 → 해당 클래스를 억제, 음수 → 선호
grad_x = torch.tensor([[[ 2.0, -1.0, 0.0]]])         # (bs, n, d)
print("guidance grad_x:", grad_x)

# 소프트맥스로 p_eta_x 계산 (논문 식 (7)·(8) 부분)
p_eta_x = F.softmax(-lambda_guidance * grad_x, dim=-1)
print("p_eta_x (softmax(-λ∇)):", p_eta_x)

# (4) 무조건부 분포와 p_eta_x를 곱해 guided posterior 획득
prob_x_guided_unnorm = p_eta_x * posterior_xt
prob_x_guided = prob_x_guided_unnorm / prob_x_guided_unnorm.sum(dim=-1, keepdim=True)
print("guided posterior prob_X:", prob_x_guided)

# (5) 멀티노미얼 샘플링으로 z_{t-1} 클래스 선택
sampled_index = torch.multinomial(prob_x_guided.squeeze(0).squeeze(0), num_samples=1)
print("샘플링된 클래스 인덱스 z_{t-1}:", sampled_index.item())

# 원-핫 벡터로 변환
one_hot_xtm1 = F.one_hot(sampled_index, num_classes=d)
print("one-hot x_{t-1}:", one_hot_xtm1)

