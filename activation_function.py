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
    file_sufix = ""
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
        file_sufix = file_sufix+'_'+plot
    plt.xlabel('x')
    plt.ylabel('Activation')
    plt.title('Activation function plot')
    plt.grid(True)
    plt.legend()
    file_sufix = str(plots)
    plt.savefig(f'activation_functions_{file_sufix}.png')
    plt.show()


x  = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
#plot_function(x, ['sigmoid', 'relu', 'leaky_relu', 'tanh'])
#ploting only relu, tanh and leaky_relu
plot_function(np.array(x), ['relu', 'leaky_relu', 'tanh'])
