import numpy as np

class MLP:
    
    def __init__(self, inputSize, outputSize, learning_rate=0.1, hiddenSizes=5):
        # Инициализация нейронной сети
        self.weights = [
            np.random.uniform(-2, 2, size=(inputSize, hiddenSizes)),  # Веса скрытого слоя
            np.random.uniform(-2, 2, size=(hiddenSizes, outputSize))  # Веса выходного слоя
        ]
        self.learning_rate = learning_rate
        self.layers = None

    # Сигмоида
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Производная от сигмоиды
    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
     
    # Прямой проход
    def feed_forward(self, x):
        input_ = x  # Входные сигналы
        hidden_ = self.sigmoid(np.dot(input_, self.weights[0]))  # Выход скрытого слоя
        output_ = self.sigmoid(np.dot(hidden_, self.weights[1]))  # Выход сети (выходной слой)
        
        self.layers = [input_, hidden_, output_]
        return self.layers[-1]
    
    # Обратный проход
    def backward(self, target):
        # Считаем производную ошибки сети
        err = (target - self.layers[-1])
    
        # Прогоняем производную ошибки обратно ко входу, считая градиенты и корректируя веса
        for i in range(len(self.layers)-1, 0, -1):
            # Градиент слоя = ошибка слоя * производную функции активации
            err_delta = err * self.derivative_sigmoid(self.layers[i])       
            err = np.dot(err_delta, self.weights[i - 1].T)
            dw = np.dot(self.layers[i - 1].T, err_delta)
            
            # Обновляем веса слоя
            self.weights[i - 1] += self.learning_rate * dw
    
    # Функция обучения с использованием стохастического градиентного спуска (SGD)
    def train(self, x_values, target):
        
        n = np.shape(x_values)[0]
        idx = np.arange(n)
        print(idx)
        np.random.shuffle(idx)
        print(idx)
        for i in idx:
            self.feed_forward(x_values[i:i+1])
            self.backward(target[i:i+1])
    
    # Функция предсказания
    def predict(self, x_values):
        return self.feed_forward(x_values)
