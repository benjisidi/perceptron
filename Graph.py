import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from Neuron import Neuron

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
# We're going to use a neuron as a convenient way to calculate results, rather than to train it
neuron = Neuron(2)
neuron.threshold = 0.5
neuron.gain = 0


def f(x, y):
    (rows, cols) = x.shape
    zs = np.zeros([100, 100])
    inputSpace = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    expected = np.array([0, 0, 0, 1])
    for i in range(0, rows):
        for j in range(0, cols):
            neuron.weights = [x[i][j], y[i][j]]
            results = np.array([neuron.compute(x) for x in inputSpace])
            loss = np.mean((expected - results)**2)
            zs[i][j] = loss
    return zs


fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
ax.plot_surface(X, Y, Z, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('loss')
plt.show()
