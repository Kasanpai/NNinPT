
import numpy as np 
class Perceptron:
    def __init__(self, inputSize, hiddenSizes, outputSize):
        # Инициализация слоев
        self.Win = np.zeros((1+inputSize, hiddenSizes[0]))  # Входной слой в первый скрытый слой
        self.Win[0,:] = np.random.randint(0, 3, size=(hiddenSizes[0]))  
        self.Win[1:,:] = np.random.randint(-1, 2, size=(inputSize, hiddenSizes[0]))

        self.hidden_weights = []  # Список для весов между скрытыми слоями
        for i in range(1, len(hiddenSizes)):
            Whide = np.zeros((1 + hiddenSizes[i-1], hiddenSizes[i]))  # весовые коэффициенты между слоями
            Whide[0,:] = np.random.randint(0, 3, size=(hiddenSizes[i]))  
            Whide[1:,:] = np.random.randint(-1, 2, size=(hiddenSizes[i-1], hiddenSizes[i]))
            self.hidden_weights.append(Whide)
        
        self.Wout = np.random.randint(0, 2, size = (1+hiddenSizes[-1],outputSize)).astype(np.float64)
        

    def predict(self, Xp):
        # Проход через первый скрытый слой
        hidden = np.where((np.dot(Xp, self.Win[1:,:]) + self.Win[0,:]) >= 0.0, 1, -1).astype(np.float64)
        
        # Проход через все скрытые слои
        for Whide in self.hidden_weights:
            hidden = np.where((np.dot(hidden, Whide[1:,:]) + Whide[0,:]) >= 0.0, 1, -1).astype(np.float64)
        
        # Выходной слой
        out = np.where((np.dot(hidden, self.Wout[1:,:]) + self.Wout[0,:]) >= 0.0, 1, -1).astype(np.float64)
        
        return out, hidden

    def train(self, X, y, n_iter=5, eta = 0.01):
        for i in range(n_iter):
            print(self.Wout.reshape(1, -1))
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                self.Wout[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)
        return self


