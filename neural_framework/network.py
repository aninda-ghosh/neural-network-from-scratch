import numpy as np
from neural_framework.activations import Sigmoid, ReLU


"""
A single neuron in a neural network.

Attributes:
  weights (ndarray): The weight vector of the neuron.
  bias (float): The bias term of the neuron.
"""

class Neuron:
  def __init__(self, input_size, activation=None):
    """
    Initializes the neuron.

    Args:
      input_size (int): The number of inputs to the neuron.
      activation (str, optional): The type of activation function to use. Supported options are 'sigmoid' and 'relu'. Defaults to None.
    """
    assert activation in ['sigmoid', 'relu'], "Unsupported activation function"

    if activation == 'sigmoid':
      self.activation = Sigmoid()
    elif activation == 'relu':
      self.activation = ReLU()

    self.weights = np.random.randn(input_size)
    self.bias = 0.0

  def __call__(self, inputs):
    """
    Forward pass of the neuron.

    Args:
      inputs (ndarray): The input vector to the neuron.

    Returns:
      ndarray: The output of the neuron after applying activation (if available).
    """
    return self.forward(inputs)

  def forward(self, inputs):
    """
    Calculates the output of the neuron.

    Args:
      inputs (ndarray): The input vector to the neuron.

    Returns:
      ndarray: The output of the neuron.
    """
    self.linear_output = np.dot(inputs, self.weights) + self.bias
    if self.activation is not None:
      self.output = self.activation(self.linear_output)
    else:
      self.output = self.linear_output
    return self.output

  def backward(self, d_output, learning_rate):
    """
    Backpropagation through the neuron.

    Args:
      d_output (ndarray): The gradient of the loss function with respect to the output of this neuron.
      learning_rate (float): The learning rate used to update weights and bias.

    Returns:
      ndarray: The gradient of the loss function with respect to the input of this neuron.
    """
    d_activation = d_output * self.activation.backward(self.linear_output)
    d_weights = np.dot(d_output.T, d_activation)
    d_bias = np.sum(d_activation, axis=0)
    self.weights -= learning_rate * d_weights
    self.bias -= learning_rate * d_bias
    d_inputs = np.dot(d_activation, self.weights.T) 
    return d_inputs



"""
Base class for neural network layers.

This class serves as an abstract base class for different types of layers 
in a neural network. It defines the basic interface for forward and backward 
propagation.

"""

class NeuralLayer:
  def forward(self, inputs):
    """
    Forward pass of the layer.

    This method must be implemented by subclasses to define the specific forward propagation behavior.

    Args:
      inputs (ndarray): The input to the layer.

    Raises:
      NotImplementedError: Since this is an abstract method.
    """
    raise NotImplementedError

  def backward(self, error_gradient, learning_rate):
    """
    Backpropagation through the layer.

    This method must be implemented by subclasses to define the specific 
    backward propagation behavior for updating layer parameters.

    Args:
      error_gradient (ndarray): The gradient of the loss function with respect 
          to the output of the layer.
      learning_rate (float): The learning rate used to update layer parameters.

    Raises:
      NotImplementedError: Since this is an abstract method.
    """
    raise NotImplementedError


"""
Fully-connected layer in a neural network.

This class implements a fully-connected layer composed of multiple neurons. 
Each neuron in the layer has weights, bias, and an optional activation 
function.

Attributes:
  neurons (list[Neuron]): A list of neurons in the layer.

"""
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
        input_gradients = np.zeros_like(self.inputs) # Initialize input gradients to zero
        for i, neuron in enumerate(self.neurons):
            input_gradients += neuron.backward(error_gradient[i], learning_rate)
        return input_gradients