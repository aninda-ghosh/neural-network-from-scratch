import numpy as np
import tqdm
import json
import argparse
import os
from neural_framework.loss_func import MSE
from neural_framework.ann import ANN
import matplotlib.pyplot as plt
import csv


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
            batch_loss = 0
            outputs = []
            for i in tqdm.tqdm(range(len(X))):
                output = self.model(X[i])
                outputs.append(output)
                error = self.loss(y[i], output)
                batch_loss += error

            y = np.array(y)
            outputs = np.array(outputs)

            loss_gradient = self.loss.backward(y, outputs)
            loss_gradient = np.mean(loss_gradient, axis=0)

            self.model.backward(loss_gradient, self.learning_rate)
            losses.append(batch_loss/len(X))

            print(f"Epoch {epoch + 1}/{self.epochs}, Mean Loss: {batch_loss/len(X)}")
        
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

    parser = argparse.ArgumentParser(description="Neural Framework, Custom Model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--action", type=str, default="train", help="Action to perform: train or test")
    parser.add_argument("--visualize", type=bool, default=False, help="Visualize the training losses")

    args = parser.parse_args()

    # Read the configuration file
    with open(args.config, "r") as file:
        config = json.load(file)
    
    np.random.seed(config['seed'])  #? For reproducibility

    model = CustomModel(config)


    if args.action == "train":
        print("Training the model...")

        print(f"Model Arch: \n{model.model}")

        # load training data
        train_data_path = config['train_data_path']
        with open(train_data_path, 'r') as file:
            reader = csv.reader(file)
            data = list(reader)
        
        X_train = np.array([list(map(float, row[:-1])) for row in data])
        #generate X_train with varied number of features x1, x2, x1_square, x2_square, x1_x2, sin(x1), sin(x2)
        X_train = np.array([[row[0], row[1], row[0]**2, row[1]**2, row[0]*row[1], np.sin(row[0]), np.sin(row[1])] for row in X_train])

        y_train = np.array([list(map(float, row[-1:])) for row in data])

        
        # # Generate some random data for training
        # X_train = np.array([
        #     [0, 0],
        #     [1, 1],
        #     [1, 0],
        #     [0, 1],
        # ])

        # y_train = np.array([
        #     [0],
        #     [0],
        #     [1],
        #     [1]
        # ])

    
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
        
        # Load test data
        test_data_path = config['test_data_path']
        with open(test_data_path, 'r') as file:
            reader = csv.reader(file)
            data = list(reader)

        X_test = np.array([list(map(float, row[:-1])) for row in data])
        
        #generate X_test with varied number of features x1, x2, x1_square, x2_square, x1_x2, sin(x1), sin(x2)
        X_test = np.array([[row[0], row[1], row[0]**2, row[1]**2, row[0]*row[1], np.sin(row[0]), np.sin(row[1])] for row in X_test])
        y_test = np.array([list(map(float, row[-1:])) for row in data])

        # # Generate some random data for testing
        # X_test = np.array([
        #     [0, 0],
        #     [1, 1],
        #     [1, 0],
        #     [0, 1],
        # ])

        # y_test = np.array([
        #     [0],
        #     [0],
        #     [1],
        #     [1]
        # ])

        # Load the model
        model.load(config['model_path'])

        print(f"Loaded Model Arch: \n{model.model}")

        y_pred = model.predict(X_test)
        print(f"\nPredictions: {y_pred}")

        #plot the predictions in a scatter plot with different colors and legend
        fig = plt.figure()
        plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, marker='o', cmap='autumn')
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Predictions")
        plt.show()

        # Calculate the mean squared error
        mse = MSE()
        error = mse(y_test, y_pred)
        print(f"Test Loss: {error}")