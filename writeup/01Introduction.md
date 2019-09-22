### Introduction

I have recently been reading _Neural Computing: An Introduction_ (R Beale and T Jackson), an old but very clearly written description of the basic principles of the field. It is, however, very light on mathematics. I decided that a fun way to round out my knowledge would be to build a very simple perceptron, completely from scratch.

### Method

As with most large projects, I figure a sensible way to proceed is to build something simple and then slowly add complexity. So, we shall begin by creating a very simple, single neuron perceptron, attempting to train it using the *Widrow-Hoff delta rule* (see below), and once we've finished playing with that, we'll add some layers and implement backpropagation.

Our first foray into the world of neural computing is going to be *extremely* simple: a few inputs, one neuron, and an output. All our perceptron has to do is multiply the inputs by their respective weights, sum them and then threshold them to create an output. When training, we are going to compare the output to the desired result, and adjust the weights as follows:

```katex
\begin{aligned}
\Delta =& d(t) - a(t)
\\w_i(t+1) =& w_i(t) + \eta\Delta x_i(t)
\\d(t) =&  \left\{
        \begin{array}{ll}
            +1 & \! \textrm{if input from class A}\\
            0 & \!  \textrm{if input from class B}
        \end{array}
    \right.
\end{aligned}
```

Easy enough:
```python
import random
import numpy as np


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
        return

    def train(self, inputs, expected):
        for i in range(len(inputs)):
            result = self.compute(inputs[i])
            self.adapt(inputs[i], expected[i], result)
```
This type of neuron unit is called an ADALINE (adaptive linear neuron). It's simple enough that the code is concise and easily understandable (which is why I chose it as a starting point), but it's not going to do much without some data and a pattern to train it on. I decided to see if it could learn the boolean operation AND as a starting point.

```python
if __name__ == "__main__":
    failed = 0
    its = tqdm(range(1000), postfix={"Success Rate": 0})
    for i in its:
        trainingData = np.array([[random.randint(0, 1), random.randint(0, 1)]
                                 for x in range(1000)])
        trainingResults = np.array([x[0] and x[1] for x in trainingData])
        neuron = Neuron(2)
        neuron.train(trainingData, trainingResults)
        test = [
            neuron.compute([0, 0]),
            neuron.compute([1, 0]),
            neuron.compute([0, 1]),
            neuron.compute([1, 1])
        ]
        if (test != [0.0, 0.0, 0.0, 1.0]):
            failed += 1
        its.set_postfix({"Success Rate": (1 - failed/(i+1))*100})
    print(f'Failed: {failed}\nSuccess Rate: {(1 - failed/1000)*100:.2f}%')

```

To measure the effectiveness of my neuron, I ran 1000 iterations training a neuron on 1000 data points, and then testing it against the entire input space (for a boolean operation with two inputs this is only 4 points), to see if it represented the operation successfully. I used `tqdm` to give me a nice progress bar while training:

```
λ python Neuron.py
100%|███████████████████████████| 1000/1000 [00:20<00:00, 47.69it/s, Success Rate=53.1] Failed: 469
Success Rate: 53.10%
```

A 53.1% success rate given 1000 data points (covering the entire input space an average of 250 times) is very poor for a pattern this simple. After some investigation, I deduced one reason for this was the randomly initialized gain being far larger than my threshold - this makes it almost impossible for my weights to converge, as they overshoot the "correct" value by a large margin each time they are adjusted.

I altered my constructor to make the gain 1/5 of the threshold

```python
    def __init__(self, nInputs):
        self.weights = np.array([random.random() for x in range(nInputs)])
        self.threshold = random.random()
        self.gain = self.threshold/5
        self.deltas = []
```

```
λ python Neuron.py
100%|███████████████████████████| 1000/1000 [00:19<00:00, 50.03it/s, Success Rate=99.1] Failed: 9
Success Rate: 99.10%
```

That's quite a bit better - there's still plenty of room for improvement, though. The first thing I decided to change now that I have something more or less functional is the training methodology. Up until now we have been executing 1000 training cycles and only then checking whether our weights have converged to their desired values. This results in a huge amount of unnecessary training.

Modifying the `train` method to test the input space each iteration and stop training if the perceptron has successfully learned the pattern more than doubles the speed, from 50it/s to 113it/s (meaning each perceptron is executing enough iterations to learn the pattern in an average of 1/113s):

```python
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
```

```
λ python Neuron.py
100%|██████████████████████████| 1000/1000 [00:08<00:00, 113.46it/s, Success Rate=99] Failed: 10
Success Rate: 99.00%
```

Note that we can only get away with crudely testing the entire input space every iteration because both our neuron and our inputs are so small - when we get to a larger network and much larger inputs, we'll batch our inputs up and train until the average error in a batch is below a desired threshold.

The fact that 1% of my neurons are failing to learn `AND` after 1000 iterations is not not ideal. This is an incredibly simple pattern: my implementation should learn it in well under 1000 iterations without fail. Taking a look at the final threshold, gain and weights for the ones that failed quickly revealed the issue: At least one of the weights was initialised at a far greater value than the threshold, and, since the gain is proportional to the threshold, it would take a very, very long time for that weight to reach an appropriate value. This highlights a problem in our naïve solution of making the gain proportional to the threshold. What we *actually* want to do is leave them both as randomly initialized, and tweak them along with our weights (my textbook doesn't mention adjusting the gain or threshold at all, so I simply opted for a reasonable solution):

```python
    # Widrow-Hoff delta rule
    def adapt(self, inputs, desired, actual):
        delta = desired - actual
        self.deltas.append(delta)
        self.weights = self.weights + self.gain*delta*inputs
        loss = np.mean(delta**2)
        self.threshold += self.gain*loss
        self.gain += self.gain*loss
        return
```

This gets us the expected performance, even when dropping the size of the training set to 100 points. All 1000 test neurons learn the pattern reasonably quickly.

```
λ python Neuron.py
100%|██████████████████████████| 1000/1000 [00:01<00:00, 642.89it/s, Success Rate=100] Failed: 0
Success Rate: 100.00%
Average number of cycles: 5.376
Max number of cycles: 74
```

The huge increase in performance (113it/s to 642it/s) is due to the fact that dropping the size of the training data means we have to generate *far* fewer random inputs.

Just for kicks, I graphed the loss with a threshold of 0.5 for weights between 0 and 1. As we're using the heaviside function when computing our result, you end up with a funky looking surface:

![inputSpace](C:\Users\benji.sidi\Documents\personal\perceptron\writeup\inputSpace.png)

You can see the loss (computed as the mean squared error over the input space) is highest when both weights are too high, resulting in the neuron firing when either of the inputs is 1. Then there's a region where one of the two weights is too high and the other is appropriate, a "sweet spot" where both are correct, and lastly a region in the bottom right corner where they are both too low.

My final code for the neuron class and graph are as follows

```python
#Neuron.py
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

```



```python
# Graph.py
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
```

That's it for now, next post will be on extending our neuron implementation to become a fully fledged multilayer perceptron.