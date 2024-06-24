import numpy as np
import json
from neural_framework.network import FullyConnectedLayer, InputLayer
from neural_framework.activations import Sigmoid, ReLU
from neural_framework.loss_func import MSE


class ANN:
    def __init__(self, input_size, hidden_layers, output_size, activation):

        #! SANITY CHECKS    
        assert input_size > 0, "Input size must be greater than 0"
        assert output_size > 0, "Output size must be greater than 0"
    
        if len(hidden_layers) > 0:
            for layer in hidden_layers:
                assert layer > 0, "Hidden layer size must be greater than 0"
    
        assert activation in ['sigmoid', 'relu', 'leaky_relu', 'tanh'], "Unsupported activation function"
        

        self.layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        self.layers.append(InputLayer(input_size, activation))
        for i in range(1, len(layer_sizes)):
            self.layers.append(FullyConnectedLayer(layer_sizes[i-1], layer_sizes[i], activation))
    
    def __call__(self, X):
        return self.forward(X)

    def __repr__(self):
        output = ""
        for i, layer in enumerate(self.layers):
            output += f"Layer {i}:\n"
            for j, neuron in enumerate(layer.neurons):
                output += f"  Neuron {j}:\n"
                output += f"    Weights: {neuron.weights}\n"
                output += f"    Bias: {neuron.bias}\n"
        return output


    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        self.output = output
        return output

    def backward(self, error_gradient, learning_rate):
        for layer in reversed(self.layers): # Backpropagate the error
            error_gradient = layer.backward(error_gradient, learning_rate)
        return error_gradient

    def save(self, path):
        parameters_dict = {}
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                parameters_dict[f"layer_{i}_neuron_{j}"] = {
                    "weights": neuron.weights.tolist(),
                    "bias": neuron.bias
                }
        
        # Store in a JSON file
        with open(path, "w") as file:
            json.dump(parameters_dict, file)

    def load(self, path):
        parameters_dict = {}
        with open(path, "r") as file:
            parameters_dict = json.load(file)

        assert len(parameters_dict) == sum([len(layer.neurons) for layer in self.layers]), "Model architecture mismatch"

        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                neuron.weights = np.array(parameters_dict[f"layer_{i}_neuron_{j}"]["weights"])
                neuron.bias = parameters_dict[f"layer_{i}_neuron_{j}"]["bias"]
