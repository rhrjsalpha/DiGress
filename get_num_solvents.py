import pandas as pd

df = pd.read_csv('/Graphormer/graphormer_data/train_50_with_features.csv')
num_unique_solvents = len(df['Solvent'].unique())
print(f"NUM_UNIQUE_SOLVENTS: {num_unique_solvents}")