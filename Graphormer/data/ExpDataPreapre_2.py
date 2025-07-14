import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def missing_counter(df:pd.DataFrame):
    df_keys = df.keys()
    missing_dict = {}
    not_missing_dict = {}
    for key in df_keys:
        missing_count = df[key].isna().sum()
        non_missing_count = df[key].notna().sum()
        missing_dict[key] = missing_count
        not_missing_dict[key] = non_missing_count
    return missing_dict, not_missing_dict

def visualize_missing_values(missing_dict: dict, tick_interval=10):
    columns = list(missing_dict.keys())
    missing_counts = list(missing_dict.values())

    # 막대 그래프 생성
    plt.figure(figsize=(20, 12))
    plt.bar(columns, missing_counts, color='skyblue')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.title('Missing Values Per Column')

    # x축 눈금 설정 (간격 조절)
    plt.xticks(
        ticks=np.arange(0, len(columns), tick_interval),  # 일정 간격마다 표시
        labels=[columns[i] for i in range(0, len(columns), tick_interval)],
        rotation=45, ha='right'
    )

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

#nist=pd.read_csv("NIST.csv")
#miss_dict, yes_dict = missing_counter(nist)
#visualize_missing_values(miss_dict)
#visualize_missing_values(yes_dict, tick_interval=100)

#Photochemcad_only=pd.read_csv("Photochemcad_only.csv")
#miss_dict, yes_dict = missing_counter(Photochemcad_only)
#visualize_missing_values(miss_dict)
#visualize_missing_values(yes_dict, tick_interval=50)

USGS=pd.read_csv("USGS_완_smiles.csv",)
print(USGS.columns)
USGS=USGS.iloc[:,3:]
USGS.drop(columns=['ID','splib', 'SA', 'NIC', 'AREF/RREF', 'K', 'ASD', 'percent', 'mol (right)', 'etc'], inplace=True)
print(USGS.columns)
USGS.to_csv('USGS.csv')
miss_dict, yes_dict = missing_counter(USGS)
visualize_missing_values(yes_dict, tick_interval=100)


