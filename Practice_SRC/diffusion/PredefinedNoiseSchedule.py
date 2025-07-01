import numpy as np
import matplotlib.pyplot as plt
from src.diffusion.diffusion_utils import cosine_beta_schedule  # 기존 코드 기준

def visualize_predefined_schedule(timesteps):
    # alphā_t 가져오기
    alphas2 = cosine_beta_schedule(timesteps) # Cumulative Alpha Squared
    sigmas2 = 1 - alphas2 # 2

    log_alphas2 = np.log(alphas2) #3
    log_sigmas2 = np.log(sigmas2) #4

    log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2 #5
    gamma = -log_alphas2_to_sigmas2 #6

    timesteps_range = np.linspace(0, 1, len(alphas2))

    # 시각화
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(timesteps_range, alphas2)
    plt.title("ᾱ_t (Cumulative Alpha Squared)")
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(timesteps_range, sigmas2)
    plt.title("σ̄²_t = 1 - ᾱ_t")
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(timesteps_range, log_alphas2)
    plt.title("log(ᾱ_t)")
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(timesteps_range, log_sigmas2)
    plt.title("log(σ̄²_t)")
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(timesteps_range, log_alphas2_to_sigmas2)
    plt.title("log(ᾱ_t / σ̄²_t)")
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(timesteps_range, gamma)
    plt.title("gamma = -log(ᾱ_t / σ̄²_t)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 실행
visualize_predefined_schedule(timesteps=1000)
