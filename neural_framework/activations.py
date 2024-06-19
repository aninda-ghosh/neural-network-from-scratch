import numpy as np

class ActivationLayer:
    def __init__(self, name):
        self.name = name

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError


class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__('sigmoid')

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return x * (1 - x)


class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__('relu')

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return np.where(x > 0, 1, 0)

