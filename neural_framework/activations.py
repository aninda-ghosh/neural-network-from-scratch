import numpy as np

"""Base class for activation layers in a neural network.

Attributes:
  name (str): The name of the activation layer.
"""

class ActivationLayer:
  def __init__(self, name):
    """Initializes the activation layer.

    Args:
      name (str): The name of the activation layer.
    """
    self.name = name

  def __call__(self, x):
    """Forward pass of the activation layer.

    This method wraps the `forward` method for convenient usage.

    Args:
      x (ndarray): The input to the activation layer.

    Returns:
      ndarray: The output of the activation layer.
    """
    return self.forward(x)

  def forward(self, x):
    """Applies the activation function to the input.

    This method must be implemented by subclasses to define the specific
    activation function.

    Args:
      x (ndarray): The input to the activation layer.

    Raises:
      NotImplementedError: Since this is an abstract method.
    """
    raise NotImplementedError

  def backward(self, dy):
    """Backpropagation through the activation layer.

    This method must be implemented by subclasses to compute the gradient
    of the loss function with respect to the input of the activation layer.

    Args:
      dy (ndarray): The gradient of the loss function with respect to the
        output of the activation layer.

    Raises:
      NotImplementedError: Since this is an abstract method.
    """
    raise NotImplementedError


"""Sigmoid class for activation layers in a neural network.

Attributes:
  name (str): The name of the activation layer.
"""
class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__('sigmoid')

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return x * (1 - x)


"""Relu class for activation layers in a neural network.

Attributes:
  name (str): The name of the activation layer.
"""
class ReLU(ActivationLayer):
    def __init__(self):
        super().__init__('relu')

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return np.where(x > 0, 1, 0)

