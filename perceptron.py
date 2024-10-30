import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def read_data(filename: str) -> pd.DataFrame:
    d = pd.read_csv(filename)
    df = pd.DataFrame(data=d)
    return df

def get_df_shape(df: pd.DataFrame) -> Tuple[int, int]:
    return df.shape

def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=test_size)
    return X_train, y_train, X_test, y_test


def extract_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Filter the dataframe to include only Setosa and Virginica rows
    # Extract the required features and labels from the filtered dataframe
    options = ['Setosa', 'Virginica']
    filteredData = df[df['variety'].isin(options)]
    features = filteredData[["sepal.length", "sepal.width"]]
    label = filteredData["variety"].map({'Setosa': 0, 'Virginica': 1})
    print(filteredData)
    return (features, label)


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the perceptron on the given input features and target labels.
        You need to do following steps:
        1. Initialize the weights and bias (you can initialize both to 0)
        2. Calculate the linear output (Z) of the perceptron for all the points in X
        3. Apply the activation function to Z and get the predictions (Y_hat)
        4. Calculate the weight update rule for the perceptron and update the weights and bias
        5. Repeat steps 2-4 for 'epochs' number of times
        6. Return the final weights and bias
        Args:
            X (array-like): The input features.
            y (array-like): The target labels.

        Returns:
            weights (array-like): Learned weights.
            bias (float): Learned bias.
        """
        weights = [0, 0]
        bias = 0
        epochErrors = []
        for j in range(self.epochs):
            fittingErrors = 0
            for i in range(len(X)):
                net = weights[0] * X[i][0] + weights[1] * X[i][1] + bias
                yhat = self.activation(net)
                desired = y[i]
                error = desired - yhat
                if error:
                    fittingErrors += 1
                newW0 = weights[0] + error * self.learning_rate * X[i][0]
                newW1 = weights[1] + error * self.learning_rate * X[i][1]
                newBias = bias + self.learning_rate * error
                weights[0], weights[1], bias = newW0, newW1, newBias
            epochErrors.append(fittingErrors)
        plt.plot(range(1, self.epochs + 1), epochErrors)
        plt.xlabel('Epochs')
        plt.ylabel('Errors')
        plt.show()
            
        self.weights, self.bias = weights, bias
        return weights, bias
        
        

    def predict(self, X):
        """
        Predict the labels for the given input features.

        Args:
            X (array-like): The input features.

        Returns:
            array-like: The predicted labels.
        """
        res = []
        for i in range(len(X)):  
            net = self.weights[0] * X[i][0] + self.weights[1] * X[i][1] + self.bias
            yhat = self.activation(net)
            res.append(yhat)
        return res
        

    def _unit_step_func(self, x):
        """
        The unit step function, also known as the Heaviside step function.
        It returns 1 if the input is greater than or equal to zero, otherwise 0.

        Args:
            x (float or array-like): Input value(s) to the function.

        Returns:
            int or array-like: Result of the unit step function applied to the input(s).
        """
        #determine if x is a number or a list
        #and then for each number based on the sign will assign 0 or 1
        
        if isinstance(x, (np.ndarray, list, tuple)):
            return [1 if num >= 0 else 0 for num in x]
        return 1 if x >= 0 else 0


# Testing
if __name__ == "__main__":
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    df=read_data("./iris.csv")
    shape = get_df_shape(df)
    features, label = extract_features_label(df)
    X_train, y_train, X_test, y_test = data_split(features, label, 0.2)
    X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values

    p = Perceptron(learning_rate=0.1, epochs=100)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # plt.scatter(X_train[:, 0], X_train[:, 1], marker="o")

    # x0_1 = np.amin(X_train[:, 0])
    # x0_2 = np.amax(X_train[:, 0])

    # x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    # x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    # ax.plot([x0_1, x0_2], [x1_1, x1_2], 'g-.', label='epochs 10')
    

    # p.epochs = 50
    # p.fit(X_train, y_train)

    # x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    # x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    # ax.plot([x0_1, x0_2], [x1_1, x1_2], color='red', label='epochs 50')
    
    # p.epochs = 100
    # p.fit(X_train, y_train)

    # x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    # x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    # ax.plot([x0_1, x0_2], [x1_1, x1_2], 'b--', label='epochs 100')
    
    

    # ymin = np.amin(X_train[:, 1])
    # ymax = np.amax(X_train[:, 1])
    # ax.set_ylim([ymin, ymax])
    # ax.legend()

    # plt.show()
