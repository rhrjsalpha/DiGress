import torch

def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    print("\n=== compute_posterior_distribution ===")
    print("input of compute_posterior_distribution")
    print("M : 원래 값 (clean)\n", M)
    print("M_t : noisy 값 \n", M_t)
    print("Qt_M (t(tb) -> t-1(sb) 전이 행렬):\n", Qt_M)
    print("Qsb_M (t=0 -> t-1 전이행렬) :\n", Qsb_M)
    print("Qtb_M (t=0 -> t 전이행렬):\n", Qtb_M)

    # 일관된 3차원 텐서 형식으로 계산을 수행하기 위해#
    M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)
    print("Flattened M:\n", M)
    print("Flattened M_t:\n", M_t)

    Qt_M_T = torch.transpose(Qt_M, -2, -1)
    print("Qt_M_T:\n", Qt_M_T)

    # 1. M_t @ Qt_M_T -> q(z_t | z_t-1)
    # - 본래 M_t-1에 Qt_M을 곱하여 M_t를 만들게 됨
    # - 위 과정을 역으로 하기 위해서 M_t에 Qt_M 을 transpose 하여 Qt_M_T를 만듦
    # - 그리고 M_T 와 Qt_M_T를 곱하여 M_t-1을 얻고자 함
    # 2. M @ Qsb_M -> q(z_t-1 | x)

    left_term = M_t @ Qt_M_T # posterior compotation 이미지 파일의 (1)
    right_term = M @ Qsb_M # posterior compotation 이미지 파일의 (2)
    product = left_term * right_term # 이미지 파일대로 곱함

    print("left_term:\n", left_term)
    print("right_term:\n", right_term)
    print("product:\n", product)

    denom = M @ Qtb_M # posterior compotation 이미지 파일의 (3)
    denom = (denom * M_t).sum(dim=-1)
    print("denom:\n", denom)

    prob = product / denom.unsqueeze(-1)
    print("prob:\n", prob)
    return prob

class PlaceHolder:
    def __init__(self, X=None, E=None, y=None):
        self.X = X
        self.E = E
        self.y = y

def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    print("\n=== posterior_distributions ===")
    prob_X = compute_posterior_distribution(X, X_t, Qt.X, Qsb.X, Qtb.X)
    prob_E = compute_posterior_distribution(E, E_t, Qt.E, Qsb.E, Qtb.E)
    return PlaceHolder(X=prob_X, E=prob_E, y=y_t)

# ====== 테스트 데이터 ======
bs, n, d = 1, 3, 3

X = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
X_t = torch.tensor([[[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]]], dtype=torch.float32)

E = torch.zeros((1, 3, 3, 3))
E_t = torch.zeros((1, 3, 3, 3))
for i in range(3):
    for j in range(3):
        if i != j:
            E[0, i, j] = torch.tensor([1, 0, 0]) if i < j else torch.tensor([0, 1, 0])
            E_t[0, i, j] = torch.tensor([0.6, 0.3, 0.1]) if i < j else torch.tensor([0.2, 0.5, 0.3])
        else:
            E[0, i, j] = torch.tensor([0, 0, 1])
            E_t[0, i, j] = torch.tensor([0.1, 0.1, 0.8])

y = torch.rand(bs, 2)
y_t = torch.rand(bs, 2)

class DummyQt:
    def __init__(self):
        self.X = torch.eye(d).unsqueeze(0).repeat(bs, 1, 1) * 0.9 + 0.1 / d
        self.E = torch.eye(d).unsqueeze(0).repeat(bs, 1, 1) * 0.85 + 0.15 / d

Qt = DummyQt()
Qsb = DummyQt()
Qtb = DummyQt()

# ====== 실행 ======
output = posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb)

print("\n=== 최종 출력 ===")
print("Posterior X:\n", output.X)
print("Posterior E:\n", output.E)
print("Posterior y:\n", output.y)
