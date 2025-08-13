import os
import pandas as pd

# 현재 폴더 내 모든 파일 리스트 가져오기
csv_files = [f for f in os.listdir() if f.endswith('.csv')]

# CSV 파일별 row 개수 세기
for file in csv_files:
    try:
        df = pd.read_csv(file)
        print(f"{file}: {len(df)} rows")
    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

# 전체 CSV 파일 개수 출력
print(f"\n총 {len(csv_files)}개의 CSV 파일을 확인했습니다.")