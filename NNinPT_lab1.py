

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Для 3D-графиков

# Считываем данные
df = pd.read_csv('data.csv')

print(df.head())

# Целевая переменная
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)


X = df.iloc[:, [0, 1, 2]].values

def neuron(w, x):
    value = w[0] + w[1]*x[0] + w[2]*x[1] + w[3]*x[2]
    return 1 if value >= 0 else -1


w = np.array([0.0, 0.1, 0.2, 0.3])  # смещение + 3 признака
print("Пример работы нейрона:", neuron(w, X[0]))

w = np.random.random(4)
eta = 0.01
w_iter = []

for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w, xi)
    error = target - predict
    w[1:] += eta * error * xi  # корректировка весов для признаков
    w[0] += eta * error        # корректировка смещения
    if j % 10 == 0:
        w_iter.append(w.tolist())


sum_err = sum((target - neuron(w, xi)) / 2 for xi, target in zip(X, y))
print("Всего ошибок: ", sum_err)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Строим точки для классов
ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], color='red', marker='o', label='Iris-setosa')
ax.scatter(X[y==-1, 0], X[y==-1, 1], X[y==-1, 2], color='blue', marker='x', label='Other')

# Построим разделяющую гиперплоскость
x1_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
x2_range = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

# Для каждого значения x1, x2 на сетке вычислим x3, чтобы линия была разделяющей
x3_grid = -(w[0] + w[1]*x1_grid + w[2]*x2_grid) / w[3]


ax.plot_surface(x1_grid, x2_grid, x3_grid, color='yellow', alpha=0.5)


ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_zlabel('Признак 3')
ax.legend()

plt.show()
