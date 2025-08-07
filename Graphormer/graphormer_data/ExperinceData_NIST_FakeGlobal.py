import pandas as pd

import pandas as pd
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
# 1. 파일 로드 및 컬럼 제거
csv = pd.read_csv("NIST_완성_smiles.csv")
csv = csv.drop(['Column0', 'InChI_new', 'SMILES'], axis=1)

# 2. 사전 정의된 solvent 목록
PREDEFINED_VOCAB = {
    'Solvent': [
        '1,4-Dioxane', 'Acetonitrile', 'Benzene', 'Chloroform', 'Cyclohexane',
        'Dichloromethane', 'Dimethylformamide', 'Dimethylsulfoxide', 'Ethanol',
        'Ethylacetate', 'Heptane', 'Hexane', 'Methanol', 'N-Methyl-2-pyrrolidone',
        'Tetrahydrofuran', 'Toluene', 'Water', 'DMSO', 'Acetone'
    ],
}

# 3. 가짜 실험 조건 생성
n = len(csv)
csv["Solvent"] = [random.choice(PREDEFINED_VOCAB['Solvent']) for _ in range(n)]
csv["Temperature"] = np.random.uniform(low=15.0, high=60.0, size=n).round(2)  # 예: 15–60℃
csv["Pressure"] = np.random.uniform(low=0.95, high=1.05, size=n).round(3)     # 예: 0.95–1.05 atm

# 4. 결과 확인 및 저장
print(csv[["Solvent", "Temperature", "Pressure"]].head())

# 5. 염 제거를 위한 함수 정의
def remove_salts(smiles: str) -> str:
    try:
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[Invalid SMILES] Cannot parse: {smiles}")
            return None
        largest = lfc.choose(mol)
        return Chem.MolToSmiles(largest)
    except Exception as e:
        print(f"[Error] {smiles} → {e}")
        return None

# 6. SMILES 열에서 염 제거 수행
csv["smiles_new"] = csv["smiles_new"].apply(remove_salts)

# 7. 확인 및 저장
print(csv[["smiles_new"]].head())
csv = csv[~csv["smiles_new"].isnull()].reset_index(drop=True)
# 필요 시 저장
csv.to_csv("NIST_with_fake_golbal.csv", index=False)