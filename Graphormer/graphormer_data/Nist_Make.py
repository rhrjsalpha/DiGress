import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# 1. 데이터 통합
know_it_all_1 = pd.read_csv('merged_200_350.csv')
know_it_all_2 = pd.read_csv('merged_200_500.csv')
know_it_all_3 = pd.read_csv('merged_200_800.csv')
know_it_all = pd.concat([know_it_all_1, know_it_all_2, know_it_all_3])
print("전체 shape:", know_it_all.shape)
know_it_all.to_csv("know_it_all.csv", index=False)

# 2. CSV 로드
csv_path = "know_it_all.csv"
df = pd.read_csv(csv_path)

# 3. 농도 문자열 분리: "0.1 g/L" → [0.1, "g/L"]
def split_concentration(val):
    try:
        parts = val.strip().split()
        if len(parts) == 2:
            return pd.Series([float(parts[0]), parts[1]])
        else:
            import re
            match = re.match(r"([0-9.]+)([a-zA-Z/]+)", val)
            if match:
                return pd.Series([float(match.group(1)), match.group(2)])
        return pd.Series([None, None])
    except:
        return pd.Series([None, None])

df[['Concentration_value', 'Concentration_unit']] = df['Concentration'].apply(split_concentration)

# 4. 단위 종류 출력
units = df['Concentration_unit'].dropna().unique()
print("🧪 농도 단위 종류:", units)

# 5. InChI → 분자량 계산
def calc_mol_weight(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            return Descriptors.MolWt(mol)
        else:
            return None
    except:
        return None

df['Molecular_Weight'] = df['InChI'].apply(calc_mol_weight)

# 6. mol/L 계산 (단위가 g/L일 때만)
def convert_to_mol_L(row):
    if row['Concentration_unit'] == 'g/L' and row['Molecular_Weight'] and row['Molecular_Weight'] > 0:
        return row['Concentration_value'] / row['Molecular_Weight']
    return None

df['Concentration_mol_L'] = df.apply(convert_to_mol_L, axis=1)

# 7. 결과 확인
print(df[['Concentration', 'Concentration_value', 'Concentration_unit', 'Molecular_Weight', 'Concentration_mol_L']].head())

# 8. 연속형/범주형 분류
categorical_cols = []
continuous_cols = []

for col in df.columns:
    if col in ['Name', 'SMILES', 'InChI', 'exp_spectrum']:
        continue
    if df[col].dtype == object or df[col].nunique() < 30:
        categorical_cols.append(col)
    else:
        continuous_cols.append(col)

print("🔷 명목형:", categorical_cols)
print("🔶 연속형:", continuous_cols)

df.to_csv("know_it_all_2.csv", index=False)

# 파일 읽기
df = pd.read_csv("know_it_all_2.csv")

# 새로운 컬럼 리스트 생성
new_columns = []
for col in df.columns:
    try:
        col_float = float(col)
        if 200.0 <= col_float <= 800.0:  # 스펙트럼 컬럼 범위 조건
            new_columns.append(str(int(col_float)))  # 200.0 → "200"
        else:
            new_columns.append(col)
    except:
        new_columns.append(col)  # 문자열 컬럼은 그대로 유지

# 컬럼명 적용
df.columns = new_columns

# 저장
df.to_csv("know_it_all_2_reindexed.csv", index=False)
print("✅ 저장 완료: know_it_all_2_reindexed.csv")