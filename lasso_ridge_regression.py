import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from typing import Tuple, List

def MSE(y_test, pred):
    '''
        return the mean square error
    '''
    return metrics.mean_squared_error(y_test, pred)

def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    '''
        Separate input data and labels, remove NaN values. 
        Execute this for both dataframes.
        return tuple of numpy arrays(train_data, train_label, test_data, test_label).
    '''
    df_train_na_dropped = df_train.dropna()
    df_test_na_dropped = df_test.dropna()

    x_train = df_train_na_dropped['x'].to_numpy()
    x_train = x_train.reshape(x_train.shape[0], 1)

    y_train = df_train_na_dropped['y'].to_numpy()
    y_train = y_train.reshape(y_train.shape[0], 1)

    x_test  = df_test_na_dropped['x'].to_numpy()
    x_test = x_test.reshape(x_test.shape[0], 1)

    y_test  = df_test_na_dropped['y'].to_numpy()
    y_test = y_test.reshape(y_test.shape[0], 1)

    return x_train, y_train, x_test, y_test

# Download and read the data.
def split_data(filename: str, percent_train: float) -> pd.DataFrame:
    '''
        Given the data filename and percentage of train data, split the data
        into training and test data. 
    '''
    #read in the data
    #split the data into training and testing sets using the percent_train which defines how much of the data is for training
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    
    #reads in data
    data = pd.read_csv(filename)
    
    #splits data into training and testing sets
    train_size = int(len(data) * percent_train)
    df_train = data.iloc[0 : train_size]
    df_test = data.iloc[train_size : ]
    
    return df_train, df_test

# Implement LinearRegression class
class LinearRegression:   
    def __init__(self, learning_rate=0.01, epoches=1000):        
        self.learning_rate = learning_rate
        self.iterations    = epoches
        self.W = None
        self.b = None
          
    # Function for model training         
    def fit(self, X, Y):
        self.N, self.n = X.shape  
        self.W = np.zeros((self.n, 1))
        self.b = 0
        
        for i in range(self.iterations):
            y_pred = np.dot(X, self.W) + self.b
            
            dw = (1 / self.N) * np.dot(X.T, (y_pred - Y))
            db = (1 / self.N) * np.sum(y_pred - Y)
            
            self.W = self.W - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
        

      
    # output      
    def predict(self, X):
        predictions = np.zeros([np.shape(X)[0], np.shape(X)[0]])

        return np.dot(X, self.W) + self.b

class RidgeRegression(): 
      
    def __init__(self, learning_rate=.00001, iterations=1000, penalty=1) : 
          
        self.learning_rate = learning_rate         
        self.iterations = iterations         
        self.penalty = penalty 
          
    # Function for model training             
    def fit(self, X, Y) :       
        # weight initialization         
        self.N, self.n = X.shape  
        self.W = np.zeros((self.n, 1))
        self.b = 0
        
        # gradient descent learning   
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.W) + self.b
           
           # calculate gradients
            dw = (1 / self.N) * np.dot(X.T, (y_pred - Y)) + (2 * self.penalty / self.N) * np.sum(self.W ** 2)
            db = (1 / self.N) * np.sum(y_pred - Y)       
  
            # update weights     
            self.W -= dw * self.learning_rate
            self.b -= db * self.learning_rate

    def predict(self, X):     
        predictions = np.zeros([np.shape(X)[0], np.shape(X)[0]])

        return np.dot(X, self.W) + self.b

def kFold(folds: int, data: pd.DataFrame):
    '''
        Given the training data, iterate through 10 folds and validate 
        10 different Ridge Regression models. 

        Returns:
            mse_avg - Float value of the average MSE between the models. 
            min_model - Integer index of the model with the minimum MSE in models[].
            models - List containing each RidgeRegression() object.
            min_mse - Float value of the minimum MSE. 
            YOU MAY USE kFold() function.
    '''   
    
    models = []

    #initalize a kFold object passing in the number of folds
    #iterate thru fold
    #for each iteration, we want to construct a training and testing set for the given data
    #pass in the training and testing datasets into the prepare_data fn to seperate x and y
    #construct RidgeRegression obj(model) and fit using the training data
    #calculate predictions using the testing data
    #calculate the MSE
    
    kf = KFold(folds)
    mse_avg = 0
    min_model = -1
    min_mse = float('inf')
    
    
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        X_train, y_train, X_test, y_test = prepare_data(train_data, test_data)
        rr = RidgeRegression()
        rr.fit(X_train, y_train)
        preds = rr.predict(X_test)
        mse = MSE(y_test, preds)
        mse_avg += mse
        if mse < min_mse:
            min_mse, min_model = mse, i
        models.append(rr)
    
    mse_avg /= folds
        
        

    return mse_avg, min_model, models, min_mse


if __name__ == "__main__":
    
    '''
    PART ONE: Linear Regression
    '''

    data_path = "./data.csv"
    df_train, df_test = split_data(data_path, .75)
    
    train_X, train_y, test_X, test_y = prepare_data(df_train, df_test)
    r = LinearRegression(learning_rate=0.0001, epoches=10)
    r.fit(train_X, train_y)

    # Make prediction with test set
    preds = r.predict(test_X)

    # Calculate and print the mean square error of your prediction
    mean_square_error = MSE(test_y, preds)
    print("Linear Regression MSE:")
    print(mean_square_error) # I added this

    #plot your prediction and labels, you can save the plot and add in the report
    plt.scatter(test_X,test_y, label='data')
    plt.plot(test_X, preds)
    plt.legend()
    plt.show()

    '''
    PART TWO: Ridge Regression and K-Fold Cross Validation
    '''

    df_train, df_test = split_data(data_path, .80)
    kFold_train_X, kFold_train_y, kFold_test_X, kFold_test_y = prepare_data(df_train, df_test)

    mse_avg, min_model, models, mse_min = kFold(10, df_train)
    best_model = models[min_model]

    print("Ridge Regression MSE - Average:")
    print(mse_avg)
    
    print("Ridge Regression MSE - Best Model:")
    print(mse_min)

    kFold_preds = best_model.predict(kFold_test_X)
    kFold_mean_square_error = MSE(kFold_test_y, kFold_preds[:, :1])

    print("Ridge Regression MSE - Best Model Final Test Predictions:")
    print(kFold_mean_square_error)

    plt.scatter(kFold_test_X,kFold_test_y, label='data')
    plt.plot(kFold_test_X, kFold_preds)
    plt.legend()
    plt.show()