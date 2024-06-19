import numpy as np

"""Base class for loss functions used in training neural networks.

Attributes:
  name (str): The name of the loss function.
"""

class Losses:
  def __init__(self, name):
    """Initializes the loss function.

    Args:
      name (str): The name of the loss function.
    """
    self.name = name

  def __call__(self, y_true, y_pred):
    """Forward pass of the loss function.

    Args:
      y_true (ndarray): The ground truth labels.
      y_pred (ndarray): The predicted values by the model.

    Returns:
      float: The calculated loss value.
    """
    return self.forward(y_true, y_pred)

  def forward(self, y_true, y_pred):
    """Calculates the loss between predicted and true values.

    This method must be implemented by subclasses to define the specific
    loss function (e.g., Mean Squared Error, Cross-Entropy).

    Args:
      y_true (ndarray): The ground truth labels.
      y_pred (ndarray): The predicted values by the model.

    Raises:
      NotImplementedError: Since this is an abstract method.
    """
    raise NotImplementedError

  def backward(self, y_true, y_pred):
    """Backpropagation through the loss function.

    This method must be implemented by subclasses to compute the gradient
    of the loss function with respect to the predicted values (y_pred).

    Args:
      y_true (ndarray): The ground truth labels.
      y_pred (ndarray): The predicted values by the model.

    Raises:
      NotImplementedError: Since this is an abstract method.
    """
    raise NotImplementedError



"""mean Square Error class.

Attributes:
  name (str): The name of the loss function.
"""
class MSE(Losses):
    def __init__(self):
        super().__init__('mse')

    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
