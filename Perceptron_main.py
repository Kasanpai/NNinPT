
import pandas as pd
import numpy as np
from neural import Perceptron


df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = [10, 5, 5] # задаем число нейронов скрытого (А) слоя 
outputSize = 1 if len(y.shape) else y.shape[1] 

num_hidden_layers = 3


NN = Perceptron(inputSize, hiddenSizes, outputSize)

NN.train(X, y, n_iter=5, eta = 0.01)

y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values
out, hidden_predict = NN.predict(X)

print(sum(out-y.reshape(-1, 1)))
