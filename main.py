import numpy as np
import tqdm
import json
import argparse
import os
from neural_framework.loss_func import MSE
from neural_framework.ann import ANN
import matplotlib.pyplot as plt


class CustomModel:
    def __init__(self, config):
        
        assert config is not None, "Configuration is required"

        self.model = ANN(config['input_size'], config['hidden_layers'], config['output_size'], config['activation'])
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        self.loss = MSE()

    def train(self, X, y):
        losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            for i in tqdm.tqdm(range(len(X))):
                output = self.model(X[i])
                error = self.loss(y[i], output)
                epoch_loss += error
                accumulated_grad = self.model.backward(y[i], self.learning_rate)
            losses.append(epoch_loss/len(X))

            print(f"Epoch {epoch + 1}/{self.epochs}, Mean Loss: {epoch_loss/len(X)}")
        
        return losses

    def predict(self, X):
        outputs = []
        
        for i in tqdm.tqdm(range(len(X))):
            output = self.model(X[i])
            outputs.append(output)
        
        return outputs

    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        if os.path.exists(path):
            self.model.load(path)
        else:
            raise FileNotFoundError(f"Model file not found at {path}, please run the training script first")




if __name__ == "__main__":
    np.random.seed(21)  #? For reproducibility

    parser = argparse.ArgumentParser(description="Neural Framework, Custom Model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--action", type=str, default="train", help="Action to perform: train or test")
    parser.add_argument("--visualize", type=bool, default=False, help="Visualize the training losses")

    args = parser.parse_args()

    # Read the configuration file
    with open(args.config, "r") as file:
        config = json.load(file)
    

    model = CustomModel(config)


    if args.action == "train":
        print("Training the model...")

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
        train_losses = model.train(X_train, y_train)
        model.save(config['model_path'])

        print(f"Model Arch: \n{model.model}")

        if args.visualize:
            # Plot the training losses with respect to epochs
            fig = plt.figure()
            plt.plot(np.arange(0, config['epochs']), train_losses)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.show()

    else:
        print("Testing the model...")
        
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

        # Load the model
        model.load(config['model_path'])

        print(f"Loaded Model Arch: \n{model.model}")

        y_pred = model.predict(X_test)
        print(f"\nPredictions: {y_pred}")

        # Calculate the mean squared error
        mse = MSE()
        error = mse(y_test, y_pred)
        print(f"Test Loss: {error}")