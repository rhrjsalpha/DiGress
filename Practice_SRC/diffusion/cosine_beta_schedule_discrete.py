## cosine_beta_schedule_discrete() 함수는 timesteps 개수만큼의 베타 값 (βₜ) 을 반환 ##
import numpy as np
import matplotlib.pyplot as plt


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    raw_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    normalized_cumprod = raw_cumprod / raw_cumprod[0]

    alphas = normalized_cumprod[1:] / normalized_cumprod[:-1]
    betas = 1 - alphas

    return x, raw_cumprod, normalized_cumprod, alphas, betas


def visualize_cosine_beta_schedule_discrete(timesteps):
    x, raw_cumprod, normalized_cumprod, alphas, betas = cosine_beta_schedule_discrete(timesteps)

    t_range = np.linspace(0, 1, timesteps + 1)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(np.linspace(0, 1, len(x)), x)
    plt.title("x (Time Steps Normalized)")
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(np.linspace(0, 1, len(raw_cumprod)), raw_cumprod)
    plt.title("Raw Cosine ᾱₜ (before normalization)")
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(np.linspace(0, 1, len(normalized_cumprod)), normalized_cumprod)
    plt.title("Normalized ᾱₜ (ᾱ₀ = 1)")
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(t_range, alphas)
    plt.title("αₜ = ᾱₜ₊₁ / ᾱₜ")
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(t_range, betas)
    plt.title("βₜ = 1 − αₜ")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 실행
visualize_cosine_beta_schedule_discrete(timesteps=1000)

