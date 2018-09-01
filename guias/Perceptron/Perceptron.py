import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """Implementa um Perceptron"""
    def __init__(self, input_size, epochs = 100, learning_rate = 0.01):
        # Adiciona o bias no vetor de pesos
        self.weights = np.zeros(input_size + 1)
        self.lr = learning_rate
        self.epochs = epochs
    
    # Função de ativação
    def sgn_activation(self, input):
        return 1 if input >= 0 else 0

    def predict(self, inputs):
        summ = np.dot(inputs, self.weights[1:]) + self.weights[0]
        
        return self.sgn_activation(summ)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # Atualiza os pesos: w <- w + α(y — f(x))x
                self.weights[1:] += self.lr * (label - prediction) * inputs
                # Atualiza o bias
                self.weights[0] += self.lr * (label - prediction)

def printData(data):
    plt.title("Iris Dataset")
    plt.scatter(data[:50, 0], data[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(data[50:100, 0], data[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('Petal length (cm)')
    plt.ylabel('Sepal length (cm)')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    # Importa o iris dataset
    dataset = pd.read_csv('./iris_data.csv', header=None)

    # Extrai os 100 primeiras rotulos 
    labels = dataset.iloc[0:100, 4].values

    # Substitui o nome dos rotulos por 0 ou 1
    labels = np.where(labels == 'Iris-setosa', 0, 1)

    # Extrai duas caracteristicas das 100 primeiras amostras
    inputs = dataset.iloc[0:100, [0, 2]].values

    # Instacia um percetron
    perceptron = Perceptron(2)
    perceptron.train(inputs, labels)

    # Testes
    test_inputs = np.array([
        [5.0, 1.4],
        [4.9, 1.4],
        [6.0, 4.9],
        [6.3, 4.5]
    ])

    test_label = np.array([0, 0, 1, 1])

    for i, item in enumerate(test_inputs):
        print("Prediction for {} is {}, expected were {}".format(
                item, perceptron.predict(item), test_label[i] 
            )
        )
    
    # Mostra um gráfico com as amostras
    printData(inputs)

    
    