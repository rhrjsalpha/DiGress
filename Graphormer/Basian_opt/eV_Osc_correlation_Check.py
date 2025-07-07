import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ 파일 경로 설정
file_path = "../data/train_100.csv"  # 파일 경로 수정 필요

# ✅ 데이터 로드
try:
    data = pd.read_csv(file_path)
    print("데이터 로드 성공 ✅")
except FileNotFoundError:
    print(f"파일이 존재하지 않습니다: {file_path}")
    exit()

# ✅ SMILES 열 및 'ex1~50', 'prob1~50' 열 추출
smiles = data['smiles']
ex_columns = [f"ex{i}" for i in range(1, 51)]
prob_columns = [f"prob{i}" for i in range(1, 51)]

# ✅ 상관 행렬 계산을 위한 데이터 생성
ex_data = data[ex_columns]
prob_data = data[prob_columns]

# ✅ 상관 행렬 계산
correlation_matrix = pd.concat([ex_data, prob_data], axis=1).corr()

# ✅ 상관 행렬 출력
print("상관 행렬:")
print(correlation_matrix)

# ✅ 상관 행렬 시각화 (matplotlib 사용)
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title("Correlation Matrix of ex and prob values")

# 행, 열 이름 추가
plt.xticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90, fontsize=8)
plt.yticks(np.arange(len(correlation_matrix.index)), correlation_matrix.index, fontsize=8)

plt.show()

# ✅ ex vs prob 사이의 상관관계 시각화 (matplotlib 사용)
plt.figure(figsize=(10, 6))
plt.scatter(ex_data.values.flatten(), prob_data.values.flatten(), alpha=0.5, c='blue', s=10)
plt.xlabel("ex")
plt.ylabel("prob")
plt.title("Scatter Plot of ex vs prob")
plt.grid(True)
plt.show()

# ✅ ex와 prob의 상관계수 출력
correlation_coeff = ex_data.corrwith(prob_data, axis=1).mean()
print(f"ex와 prob의 평균 상관 계수: {correlation_coeff:.4f}")


