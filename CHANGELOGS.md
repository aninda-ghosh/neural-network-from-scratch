# Neural Network From Scratch

## Changelogs

### v0.3.0 (feature + bugfix)

- Fixed the issue with 1 hidden neuron edge case. The dimensionality squuezing was effecting the backward gradient flow.
- Decouples loss backward function and model backward function call.
- Added Leaky Relu activation to tackle the vanishing gradient problem.
- Added Tanh activation.
- Added a script to generate data sources e.g. Spiral Data, XOR Data `generate_data.ipynb`
- Added Binary Cross Entropy Loss function.

### v0.2.1 (feature)

- Added visualization support for the training losses.

### v0.2.0 (feature + bugfix)

- Added suuport for Input layer separately, Initial implementation of the Neuron class was a bit buggy, now each neuron has `n inputs`, `1 Activation` and `1 Output`.
- Added a small value int he bias term to stabilise the backpropagation, relu activation can lead to 0 backward gradient sometimes.
- Fixed the backprop algorithm for a single neuron. 

### v0.1.1 (bugfix)

- Fixed model loading issue, previously the model loading was local and was not effecting the true weights in the context of the model.
- Removed support for None activation in the network.

### v0.1.0 (feature)

- Added setup script for local installation.
- Added model save and load feature.
- Config file explicitly added to modify the parameters globally.
- Added Docstrings for the codebase.

### v0.0.1 (feature)

- Defined components for a fully connected network. (Modularized)
    - Neuron
    - Layers
- Added definitions of the Loss Functions.
    - Mean Square Error
- Added definitions of the Activation Functions
    - Sigmoid [Link](https://en.wikipedia.org/wiki/Sigmoid_function)
    - Relu [Link](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- A basic training script with sample usage.