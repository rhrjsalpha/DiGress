import pandas as pd

# CSV 파일 읽기
test_1000 = pd.read_csv("test_1000.csv")

# 랜덤하게 50개의 행을 선택하여 새로운 DataFrame 생성
example = test_1000.sample(n=10, random_state=42).reset_index(drop=True)

# 새로운 CSV 파일로 저장
example.to_csv("test_10.csv", index=False)