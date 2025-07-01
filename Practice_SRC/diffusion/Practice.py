import numpy as np
import matplotlib.pyplot as plt

def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

# 실행
timesteps = 1000
alpha_bar = cosine_beta_schedule(timesteps)

# 일부 출력
print("첫 10개 alphā_t:", alpha_bar[:10])
print("마지막 10개 alphā_t:", alpha_bar[-10:])

# 수정된 plot
plt.plot(np.linspace(0, 1, timesteps + 1), alpha_bar)
plt.xlabel("Normalized timestep t/T")
plt.ylabel("Cumulative alpha (alphā_t)")
plt.title("Cosine Beta Schedule (alphā_t over time)")
plt.grid(True)
plt.show()
