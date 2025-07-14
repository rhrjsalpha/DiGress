from Graphormer.GP5.data_prepare.Dataloader_QMData import get_global_feature_info

global_feature_names = ['Solvent', 'Temperature', 'Pressure']
temp_dataset_path = "/Graphormer/graphormer_data/train_50_with_features.csv"

try:
    global_dim, nominal_dims = get_global_feature_info(temp_dataset_path, global_feature_names)
    print(f"Calculated Global Feature Dimension: {global_dim}")
    print(f"Nominal Feature Dimensions: {nominal_dims}")
except Exception as e:
    print(f"Error: {e}")