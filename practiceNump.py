import numpy as np
import pandas as pd
ih = pd.read_csv('input_hidden.csv')
ho = pd.read_csv('hidden_output.csv')
x = np.array([(0, 0, 0, 0, 0),
              (0, 0, 0, 0, 0),
              (0, 0, 0, 0, 0),
              (0, 0, 0, 0, 0),
              (0, 0, 0, 0, 0),
              (0, 0, 0, 0, 0),
              (0, 0, 0, 0, 0),
              (0, 0, 0, 0, 0),
              (0, 0, 0, 0, 0),
              (0, 0, 0, 0, 0)],dtype=float)
y = np.array([0,0,0,0,0,0,0,0,0,0],dtype=float)
for i in range(10):
    for j in range(5):
        x[i][j]=ih.iloc[i][j]

for k in range(10):
    y[k]=ho.iloc[k][0]
print(x[0][0:4])
print(y)