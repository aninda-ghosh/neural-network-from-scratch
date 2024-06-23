# Neural Network From Scratch

## Changelogs

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