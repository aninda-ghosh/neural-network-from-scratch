import numpy as np
from .activations import Sigmoid, ReLU


class Neuron:
    def __init__(self, input_size, activation=None):
        assert activation in ['sigmoid', 'relu'], "Unsupported activation function"

        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'relu':
            self.activation = ReLU()

        self.weights = np.random.randn(input_size)
        self.bias = 0.0

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        self.linear_output = np.dot(inputs, self.weights) + self.bias
        if self.activation is not None:
            self.output = self.activation(self.linear_output)
        else:
            self.output = self.linear_output
        return self.output

    def backward(self, d_output, learning_rate):
        d_activation = d_output * self.activation.backward(self.linear_output)
        d_weights = np.dot(d_output.T, d_activation)
        d_bias = np.sum(d_activation, axis=0)
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        d_inputs = np.dot(d_activation, self.weights.T) 
        return d_inputs



# To be used for extension for other layers later.
class NeuralLayer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, error_gradient, learning_rate):
        raise NotImplementedError



# Fully connected layer with fully connected neurons
class FullyConnectedLayer(NeuralLayer):
    def __init__(self, input_size, output_size, activation):
        self.neurons = [Neuron(input_size, activation) for _ in range(output_size)]

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        self.inputs = inputs
        outputs = np.array([neuron(inputs) for neuron in self.neurons]).T
        return outputs

    def backward(self, error_gradient, learning_rate):
        input_gradients = np.zeros_like(self.inputs)
        for i, neuron in enumerate(self.neurons):
            input_gradients += neuron.backward(error_gradient[i], learning_rate)
        return input_gradients