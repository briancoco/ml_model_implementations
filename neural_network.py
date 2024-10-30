import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def read_data(filename: str) -> pd.DataFrame:
    d = pd.read_csv(filename)
    df = pd.DataFrame(data=d)
    return df

def extract_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Filter the dataframe to include only Setosa and Virginica rows
    filtered_df = df[df['variety'].isin(['Versicolor', 'Virginica'])]
    # Extract the required features and labels from the filtered dataframe
    features = filtered_df[['sepal.length', 'sepal.width']]
    labels = filtered_df['variety'].map({'Versicolor': 0, 'Virginica': 1})
    return features, labels

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def function_derivative(z):
    fd = sigmoid(z)
    return fd * (1 - fd)


class NN:
    def __init__(self, features, hidden_neurons, output_neurons, learning_rate):
        self.features = features
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        
        # initialize weights
        self.V = np.random.randn(self.features, self.hidden_neurons)
        self.Z = np.random.randn(self.hidden_neurons, self.hidden_neurons)
        self.W = np.random.randn(self.hidden_neurons, self.output_neurons)
        
        # initialize biases: 0
        self.V0 = np.zeros((self.hidden_neurons))
        self.Z0 = np.zeros((self.hidden_neurons))
        self.W0 = np.zeros((self.output_neurons))
    
    def train(self, X, t, epochs=1000):
        costs = []
        for epoch in range(epochs):
            # forward pass
            net_u = X.dot(self.V) + self.V0 #vector consisting of the X values for each of our hidden layer neurons
            H = sigmoid(net_u) #vector of the y values (output) of each of our hidden layer neuron
            
            net_j = H.dot(self.Z) + self.Z0 #vector consisting of the x values of our 2nd hidden layer
            Z = sigmoid(net_j) #vector consisting of the y vals of our 2nd hidden layer
            
            net_z = Z.dot(self.W) + self.W0 #vector consisting of the X values for our output neuron
            O = sigmoid(net_z) #vector consisting of our model's output
            
            # backpropagation pass
            error_output = O - t
          
            d_W = Z.T.dot(error_output * function_derivative(net_z)) #gradient for the weights of our output neuron
            d_W0 = np.sum(error_output * function_derivative(net_z), axis=0) #gradient for our output neuron bias
            
            error_hidden_layer_2 = error_output.dot(self.W.T) * function_derivative(net_j)
            d_Z = H.T.dot(error_hidden_layer_2)
            d_Z0 = np.sum(error_hidden_layer_2, axis=0)
           
            error_hidden_layer = error_output.dot(self.W.T) * function_derivative(net_u)
            error_hidden_layer = error_hidden_layer.dot(self.Z.T)
            d_V = X.T.dot(error_hidden_layer)
            d_V0 = np.sum(error_hidden_layer, axis=0)
            
            # update weights and biases
            self.W -= self.learning_rate * d_W
            self.W0 -= self.learning_rate * d_W0
            self.Z -= self.learning_rate * d_Z
            self.Z0 -= self.learning_rate * d_Z0
            self.V -= self.learning_rate * d_V
            self.V0 -= self.learning_rate * d_V0
            
            #find the cost function
            if epoch % 10 == 0:
                loss =  np.square(np.subtract(t,O)).mean() 
                costs.append(loss)
                
        return costs
    
    def predict(self, X):
        net_u = X.dot(self.V) + self.V0
        H = sigmoid(net_u)
        net_j = H.dot(self.Z) + self.Z0
        Z = sigmoid(net_j)
        net_z = Z.dot(self.W) + self.W0
        O = sigmoid(net_z)
     
        return (O > 0.5).astype(int)
    
if __name__ == "__main__":
    def accuracy(t, y_pred):
        accuracy = np.sum(t == y_pred) / len(t)
        return accuracy

    # Read the data
    train_df = read_data("./iris_training_data.csv")
    test_df = read_data("./iris_testing_data.csv")
    
    X, t =  extract_features_label(train_df)
    X_test, t_test = extract_features_label(test_df)

    t = t.values.reshape([len(t),1])
    t_test = t_test.values.reshape([len(t_test),1])
    
# create Neural Network class
nn = NN(features=2, hidden_neurons=5, output_neurons=1, learning_rate=0.01)

# train the network
cost = nn.train(X, t)

# make predictions on the test data
y_pred = nn.predict(X_test)

# evaluate the accuracy
acc = accuracy(t_test,y_pred)
print(acc)
  
plt.plot(cost)
plt.show()