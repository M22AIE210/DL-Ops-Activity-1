import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

def plot_function(x, plots):
    for plot in plots:
        if plot == 'sigmoid':
            y = sigmoid(x)
            plt.plot(x, y, label='Sigmoid')
        elif plot == 'relu':
            y = relu(x)
            plt.plot(x, y, label='ReLU')
        elif plot == 'leaky_relu':
            y = leaky_relu(x)
            plt.plot(x, y, label='Leaky ReLU')
        elif plot == 'tanh':
            y = tanh(x)
            plt.plot(x, y, label='Tanh')

    plt.xlabel('x')
    plt.ylabel('Activation')
    plt.title('Activation function plot')
    plt.grid(True)
    plt.legend()
    plt.savefig('activation_functions.png')
    plt.show()


x = np.linspace(-10, 10, 100)
plot_function(x, ['sigmoid', 'relu', 'leaky_relu', 'tanh'])