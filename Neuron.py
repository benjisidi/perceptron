import random
import numpy as np
from tqdm import tqdm


class Neuron:

    def __init__(self, nInputs):
        self.weights = np.array([random.random() for x in range(nInputs)])
        self.threshold = random.random()
        self.gain = random.random()
        self.deltas = []

    def __repr__(self):
        return f'Gain:\t{self.gain}\nThreshold:\t{self.threshold}\nWeights:\t{self.weights}\nCurrentError:\t{self.deltas[-1]}'

    def compute(self, inputs):
        total = np.sum(inputs * self.weights)
        return np.heaviside(total - self.threshold, 0)

    # Widrow-Hoff delta rule
    def adapt(self, inputs, desired, actual):
        delta = desired - actual
        self.deltas.append(delta)
        self.weights = self.weights + self.gain*delta*inputs
        loss = np.mean(delta**2)
        self.threshold += self.gain*loss
        self.gain += self.gain*loss
        return

    def train(self, inputs, expected):
        for i in range(len(inputs)):
            result = self.compute(inputs[i])
            self.adapt(inputs[i], expected[i], result)
            test = [
                self.compute([0, 0]),
                self.compute([1, 0]),
                self.compute([0, 1]),
                self.compute([1, 1])
            ]
            if (test == [0.0, 0.0, 0.0, 1.0]):
                break
        return i


if __name__ == "__main__":
    failed = 0
    its = tqdm(range(1000), postfix={"Success Rate": 0})
    failures = []
    trainingTimes = []
    for i in its:
        trainingData = np.array([[random.randint(0, 1), random.randint(0, 1)]
                                 for x in range(100)])
        trainingResults = np.array([x[0] and x[1] for x in trainingData])
        neuron = Neuron(2)
        cycles = neuron.train(trainingData, trainingResults)
        trainingTimes.append(cycles)
        if (cycles == 99):
            failed += 1
            failures.append([neuron.threshold, neuron.gain, neuron.weights])
        its.set_postfix({"Success Rate": (1 - failed/(i+1))*100})
    print(f'Failed: {failed}\nSuccess Rate: {(1 - failed/1000)*100:.2f}%\nAverage number of cycles: {np.mean(trainingTimes)}\nMax number of cycles: {np.max(trainingTimes)}')
