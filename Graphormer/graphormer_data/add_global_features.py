
import pandas as pd
import numpy as np

def add_global_features(input_csv_path, output_csv_path):
    """
    Reads a CSV file, adds global features (Solvent, Temperature, Pressure),
    and saves the modified DataFrame to a new CSV file.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the output CSV file.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
        return

    # Add 'Solvent' column (nominal)
    solvents = ['Water', 'Ethanol', 'Methanol', 'Acetone', 'DMSO']
    df['Solvent'] = np.random.choice(solvents, size=len(df))

    # Add 'Temperature' column (numerical, e.g., in Kelvin)
    df['Temperature'] = np.random.uniform(273.15, 373.15, size=len(df)).round(2) # 0-100 Celsius

    # Add 'Pressure' column (numerical, e.g., in atm)
    df['Pressure'] = np.random.uniform(0.5, 2.0, size=len(df)).round(2) # 0.5-2.0 atm

    df.to_csv(output_csv_path, index=False)
    print(f"Successfully added global features to {input_csv_path} and saved to {output_csv_path}")

if __name__ == "__main__":
    # Define input and output paths
    base_path = "/Graphormer/graphormer_data/"
    
    train_input = base_path + "train_50.csv"
    train_output = base_path + "train_50_with_features.csv"
    add_global_features(train_input, train_output)

    test_input = base_path + "test_10.csv"
    test_output = base_path + "test_10_with_features.csv"
    add_global_features(test_input, test_output)
