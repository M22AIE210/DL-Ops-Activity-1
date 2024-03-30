import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    plt.plot(x, y, label='Sigmoid')
    plt.xlabel('x')
    plt.ylabel('Activation')
    plt.title('Sigmoid function plot')
    plt.grid(True)
    plt.legend()
    plt.savefig('sigmoid.png')
    plt.show()

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
sigmoid(np.array(random_values))
