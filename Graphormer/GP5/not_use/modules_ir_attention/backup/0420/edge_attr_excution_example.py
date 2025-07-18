from ogb.utils.mol import smiles2graph

smiles = "CCO"  # Ethanol
graph = smiles2graph(smiles)

# edge_attr 추출
edge_attr = graph.get("edge_feat", None)

# 출력
print("edge_attr =", edge_attr)

