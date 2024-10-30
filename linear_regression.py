import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from typing import Tuple, List
import sklearn.linear_model

# Download and read the data.
def read_train_data(filename: str) -> pd.DataFrame:
    '''
        read train data and return dataframe
    '''
    data = pd.read_csv(filename)
    return data

def read_test_data(filename: str) -> pd.DataFrame:
    '''
        read test data and return dataframe
    '''
    data = pd.read_csv(filename)
    return data


# Prepare your input data and labels
def prepare_data( df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    '''
        Separate input data and labels, remove NaN values. 
        Execute this for both dataframes.
        return tuple of numpy arrays(train_data, train_label, test_data, test_label).
        may use .dropna,
    '''
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    n = df_train['x'].shape[1] if len(df_train['x'].shape) > 1 else 1

    x_train = df_train['x'].to_numpy()
    x_train = x_train.reshape(x_train.shape[0], n)
    
    y_train = df_train['y'].to_numpy()
    y_train = y_train.reshape(y_train.shape[0], 1)
    
    x_test = df_test['x'].to_numpy()
    x_test = x_test.reshape(x_test.shape[0], n)
    
    y_test = df_test['y'].to_numpy()
    y_test = y_test.reshape(y_test.shape[0], 1)
    
    return (x_train, y_train, x_test, y_test)


# Implement LinearRegression class
class LinearRegression:   
    def __init__(self, learning_rate=0.01, epoches=1000):        
        self.learning_rate = learning_rate
        self.iterations    = epoches
        self.W = None
        self.b = None
          
    # Function for model training         
    def fit(self, X, Y):
        #1. initialize weights
        #2. have a for loop that iterates epoches time
        #3. for each epoch, we want to iterate thru the data set and find the res of our gradient descent fns
        #4. use the learning rule to improve weight and bias
        #5. save weight and bias
        self.N, self.n = X.shape  
        self.W = np.zeros((self.n, 1))
        self.b = 0
        
        for i in range(self.iterations):
            y_pred = np.dot(X, self.W) + self.b
            
            dw = (1 / self.N) * np.dot(X.T, (y_pred - Y))
            db = (1 / self.N) * np.sum(y_pred - Y)
            
            self.W = self.W - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
        
        # for i in range(self.iterations):
        #     new_w = [0] * self.n
        #     new_b = 0
        #     for j in range(len(X)):
        #         y_pred = X[j].dot(self.W)[0]

        #         for z in range(self.n):
        #             #need to rework the equation for calculating the gradient
        #             #to account for dimension spaces greater than 2
        #             pderiv_weight = (Y[j] - y_pred - self.b) * (-2 * X[j][z])
        #             new_w[z] += pderiv_weight[0]
                    

        #         pderiv_bias = (Y[j] - y_pred - self.b) * -2
        #         new_b += pderiv_bias
                
        #     new_w = [w / len(X) for w in new_w]
        #     new_b = new_b / len(X)
            
        #     self.b -= (new_b * self.learning_rate)
            
        #     for z in range(self.n):
        #         self.W[z][0] -= new_w[z] * self.learning_rate
        
                


      
    # output      
    def predict(self, X):
        #takes in an array of independent variable values X and computes the predicted values Y
        # predicted_Y = []
        # for i in range(len(X)):
        #     predicted_Y.append([X[i], X[i].dot(self.W)[0] + self.b])
        # return np.array(predicted_Y).reshape(len(X), 2)
        return np.dot(X, self.W) + self.b
        
        

# Calculate and print the mean square error of your prediction
def MSE(y_test, pred):
    #calculates the mean sqaured error for the given model output
    result = 0
    for i in range(len(y_test)):
        result += (y_test[i] - pred[i]) ** 2
    return result / len(y_test)



if __name__ == "__main__":
   
    data_path_train   = "./train2.csv"
    data_path_test    = "./test.csv"
    df_train, df_test = read_train_data(data_path_train), read_test_data(data_path_test)


    train_X, train_y, test_X, test_y = prepare_data(df_train, df_test)
    r = LinearRegression(learning_rate=0.0001, epoches=10)
    r.fit(train_X, train_y)

    #print?
    #print(df_train.head())
    #print(df_test.head())

    # Make prediction with test set
    preds = r.predict(test_X)

    # Calculate and print the mean square error of your prediction
    mean_square_error = MSE(test_y, preds)
    print(mean_square_error) # I added this

    # plot your prediction and labels, you can save the plot and add in the report
    plt.scatter(test_X,test_y, label='data')
    plt.plot(test_X, preds)
    plt.legend()
    plt.show()

    
