import numpy as np
import matplotlib.pyplot as plt

def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    steps = timesteps + 2

    # 1. x: [0, steps]
    x = np.linspace(0, steps, steps)

    # 2. 초기 alphas_cumprod (before normalization)
    raw_alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2

    # 3. 정규화된 alphas_cumprod (처음 값을 1로 나눈 것)
    normalized_alphas_cumprod = raw_alphas_cumprod / raw_alphas_cumprod[0]

    # 4. betas 계산
    betas = 1 - (normalized_alphas_cumprod[1:] / normalized_alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)

    # 5. alphas 계산
    alphas = 1. - betas

    # 6. 누적곱 alphas_cumprod 계산
    # cumprod = 누적곱
    alphas_cumprod = np.cumprod(alphas, axis=0)
    print("final alphas_cumprod",alphas_cumprod)

    # 7. raise_to_power 적용 (선택사항)
    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return x, raw_alphas_cumprod, normalized_alphas_cumprod, betas, alphas, alphas_cumprod


# 실행
timesteps = 1000
x, raw_acp, norm_acp, betas, alphas, final_acp = cosine_beta_schedule(timesteps)

# 시각화용 x축
timesteps_range = np.linspace(0, 1, len(final_acp))

# 시각화
plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.plot(np.linspace(0, 1, len(x)), x)
plt.title("x (linspace from 0 to T+2)")
plt.xlabel("Normalized t")
plt.ylabel("x")

plt.subplot(3, 2, 2)
plt.plot(np.linspace(0, 1, len(raw_acp)), raw_acp)
plt.title("Raw alphas_cumprod (cos²)")
plt.xlabel("Normalized t")
plt.ylabel("raw ᾱₜ")

plt.subplot(3, 2, 3)
plt.plot(np.linspace(0, 1, len(norm_acp)), norm_acp)
plt.title("Normalized alphas_cumprod")
plt.xlabel("Normalized t")
plt.ylabel("normalized ᾱₜ")

plt.subplot(3, 2, 4)
plt.plot(np.linspace(0, 1, len(betas)), betas)
plt.title("Betas (βₜ)")
plt.xlabel("Normalized t")
plt.ylabel("βₜ")

plt.subplot(3, 2, 5)
plt.plot(np.linspace(0, 1, len(alphas)), alphas)
plt.title("Alphas (αₜ = 1 - βₜ)")
plt.xlabel("Normalized t")
plt.ylabel("αₜ")

plt.subplot(3, 2, 6)
plt.plot(timesteps_range, final_acp)
plt.title("Final alphas_cumprod (ᾱₜ)")
plt.xlabel("Normalized t")
plt.ylabel("ᾱₜ (final)")

plt.tight_layout()
plt.show()

