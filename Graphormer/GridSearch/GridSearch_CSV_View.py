import pandas as pd

Result = pd.read_csv(r'C:\Users\analcheminfo\Documents\UV-Vis NNs\Graphormer\GridSearch/GredSearch.csv')
print(Result.head(3))
print(Result.shape)
print(Result.keys())
loss_history = Result['loss_history']
print(loss_history)