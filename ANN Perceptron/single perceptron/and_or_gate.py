import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate = 0.1, epochs = 10):
        self.weights = np.zeros(input_size+1)
        self.lr = learning_rate
        self.epochs = epochs
    
    def activation(self, x):
        return 1 if x>=0 else 0
    
    def predict(self, x):
        op = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation(op)
    
    def train(self, x, y):
        for _ in range(self.epochs):
            for input, label in zip(x, y):
                prediction = self.predict(input)
                error = label - prediction
                
                self.weights[1:] += error*self.lr*input
                self.weights[0] += error*self.lr


if __name__ == "__main__":


    # ----- AND gate ------
    
    data_and = pd.read_csv('and_truth_table.csv')


    X_and = data_and.iloc[:, :-1].values
    y_and = data_and.iloc[:, -1].values


    perceptron_and = Perceptron(input_size=X_and.shape[1], learning_rate=0.1, epochs=10)
    perceptron_and.train(X_and, y_and)


    print("Final Weights AND gate:", perceptron_and.weights)
    print("Testing Predictions AND gate:")
    for inputs in X_and:
        print(f"Input: {inputs}, Prediction: {perceptron_and.predict(inputs)}")
        
    # ----- OR gate ------
    
    data_or = pd.read_csv('or_truth_table.csv')


    X_or = data_or.iloc[:, :-1].values
    y_or = data_or.iloc[:, -1].values


    perceptron_or = Perceptron(input_size=X_or.shape[1], learning_rate=0.1, epochs=10)
    perceptron_or.train(X_or, y_or)


    print("Final Weights OR gate:", perceptron_or.weights)
    print("Testing Predictions OR gate:")
    for inputs in X_or:
        print(f"Input: {inputs}, Prediction: {perceptron_or.predict(inputs)}")