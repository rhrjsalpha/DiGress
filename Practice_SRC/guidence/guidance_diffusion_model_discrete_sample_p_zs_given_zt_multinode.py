# -------------------------------------------------------------
# 다중 노드(n=2) 예시: guided posterior가 노드별로 어떻게 계산되는가
# -------------------------------------------------------------
import torch
import torch.nn.functional as F
torch.set_printoptions(precision=4, sci_mode=False)

bs, n, d = 1, 2, 3          # 그래프 1개, 노드 2개, 클래스 3개(C,O,N)

# (1) 모델(pred) 로짓 ― 노드마다 다르게 구성
pred_logits_x = torch.tensor([
    [[ 2.0,  0.5, -1.0],     # 노드 0
     [-0.5,  1.5,  0.2]]     # 노드 1
])                          # shape (1, 2, 3)
print("\n[1] 모델 softmax 예측:")
print(F.softmax(pred_logits_x, dim=-1))

# (2) 무조건부 posterior p_θ(x_{t-1}|G_t)
posterior_xt = torch.tensor([
    [[0.20, 0.60, 0.20],     # 노드 0
     [0.10, 0.25, 0.65]]     # 노드 1
])                          # (1, 2, 3)
print("\n[2] 무조건부 posterior:")
print(posterior_xt)

# (3) regressor guidance (∇) : 노드마다 다른 기울기
lambda_guidance = 0.5
grad_x = torch.tensor([[
    [ 2.0, -1.0,  0.0],      # 노드 0  → C 억제, O 선호
    [-0.2,  1.0, -1.5]       # 노드 1  → N 선호, O 억제
]])
p_eta_x = F.softmax(-lambda_guidance * grad_x, dim=-1)
print("\n[3] p_eta_x (soft-max(-λ∇)):")
print(p_eta_x)

# (4) guided posterior = p_eta_x ⊙ posterior  (정규화 포함)
prob_x_guided_unnorm = p_eta_x * posterior_xt
prob_x_guided = prob_x_guided_unnorm / prob_x_guided_unnorm.sum(-1, keepdim=True)
print("\n[4] guided posterior:")
print(prob_x_guided)

# (5) 노드별 멀티노미얼 샘플링
sam_indices = torch.multinomial(prob_x_guided.view(-1, d), 1)   # (bs*n, 1)
sam_indices = sam_indices.view(bs, n)                           # (1, 2)
print("\n[5] 샘플링된 클래스 인덱스 (노드0, 노드1):", sam_indices.tolist())

one_hot_xtm1 = F.one_hot(sam_indices, num_classes=d)
print("\n[5′] one-hot x_{t-1}:")
print(one_hot_xtm1)
