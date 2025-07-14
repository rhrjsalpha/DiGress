import pandas as pd

nist_data = pd.read_csv("NIST_완성_smiles.csv")
print(nist_data.keys())
nist_data_ori = pd.read_csv("jdx_정리_all_wavelength_2.csv")
print(nist_data_ori.keys())

CAS_list = []
for one_name in nist_data_ori['name']:
    splited_name = one_name.split('|')
    CAS_num = splited_name[2].replace("-UVVis.jdx", "")
    CAS_list.append(CAS_num)
nist_data_ori['CAS'] = CAS_list
print(nist_data_ori.keys())

result = pd.merge(nist_data, nist_data_ori, on='CAS', how='inner')
print(len(result), result.shape)
print(len(nist_data_ori), nist_data_ori.shape)
print(len(nist_data), nist_data_ori.shape)
print(result.keys())
new_result = result.iloc[:, 104:]
new_result.drop(columns=['Unnamed: 0', 'name'], inplace=True)
new_result = new_result.rename(columns={'100_y': '100.0'})
print(new_result.keys())
new_result.to_csv('NIST.csv', index=False)



