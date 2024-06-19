import numpy as np
import tqdm
from neural_framework.loss_func import MSE
from neural_framework.ann import ANN


class CustomModel:
    def __init__(self, config, learning_rate=0.01):
        
        assert config is not None, "Configuration is required"

        self.model = ANN(config['input_size'], config['hidden_layers'], config['output_size'], config['activation'])
        self.learning_rate = learning_rate
        self.loss = MSE()

        print(f"Model: {self.model}")

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in tqdm.tqdm(range(len(X))):
                output = self.model(X[i])
                error = self.loss(y[i], output)
                epoch_loss += error
                accumulated_grad = self.model.backward(y[i], self.learning_rate)

            print(f"Epoch {epoch + 1}/{epochs}, Mean Loss: {epoch_loss/len(X)}")

    def predict(self, X):
        outputs = []
        
        for i in tqdm.tqdm(range(len(X))):
            output = self.model(X[i])
            outputs.append(output)
        
        return outputs


if __name__ == "__main__":
    np.random.seed(0)  # For reproducibility

    config = {
        "input_size": 3,
        "hidden_layers": [3],
        "output_size": 2,
        "activation": "relu"
    }

    # Generate some random data for training
    X_train = np.array([
        [0.1, 0.2, 0.3],
        [0.5, 0.4, 0.3],
        [0.9, 0.8, 0.7],
        [0.3, 0.2, 0.1],
        [0.6, 0.5, 0.4],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2],
        [0.8, 0.7, 0.6],
        [0.3, 0.4, 0.5]
    ])

    y_train = np.array([
        [0.4, 0.5],
        [0.2, 0.1],
        [0.6, 0.5],
        [0.4, 0.5],
        [0.3, 0.2],
        [0.6, 0.5],
        [0.5, 0.6],
        [0.1, 0.2],
        [0.3, 0.4],
        [0.6, 0.7]
    ])

    # Train the model
    model = CustomModel(config)

    # Train the model
    model.train(X_train, y_train, epochs=10)

    # Generate some random data for testing
    X_test = np.array([
        [0.1, 0.3, 0.5],
        [0.5, 0.7, 0.9],
        [0.6, 0.2, 0.4],
        [0.3, 0.5, 0.7],
        [0.2, 0.6, 0.8]
    ])

    y_test = np.array([
        [0.4, 0.6],
        [0.8, 0.7],
        [0.5, 0.3],
        [0.6, 0.8],
        [0.7, 0.9]
    ])

    y_pred = model.predict(X_test)
    print(f"\nPredictions: {y_pred}")

    # Calculate the mean squared error
    mse = MSE()
    error = mse(y_test, y_pred)
    print(f"Test Loss: {error}")