
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def read_data(filename: str) -> pd.DataFrame:
    d = pd.read_csv(filename)
    df = pd.DataFrame(data=d)
    return df

class GaussianNaiveBayes:
    def __init__(self, eps=1e-6):
        self.classes = ['Setosa', 'Veriscolor', 'Virginica']
        self.mean = None
        self.var = None
        self.priors = None
        self.eps = eps  # small constant to avoid division by zero

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Gaussian Naive Bayes model. Calculate the mean, variance, and prior probabilities for each class.

        Parameters:
        X : np.ndarray
            The training features
        y : np.ndarray
            The training labels
            
        Returns:
        mean : np.ndarray
              The mean for each class
        var : np.ndarray
              The variance for each class
        priors : np.ndarray
              The prior probabilities for each class
        """
        #given then training features and labels
        #we want to determine the mean, variance for a given feature for a given flower type
        
        #first thing we need to do is split up the features array based on flower type
        #this will allow us to easily calculate the mean, variance, and prior probability
        
        #once we seperate the data we can then perform our computations
        
        setosa = []
        virginica = []
        versicolor = []
        
        for i in range(len(y)):
            if y[i] == 'Setosa':
                setosa.append(X[i])
            elif y[i] == 'Virginica':
                virginica.append(X[i])
            else:
                versicolor.append(X[i])
        
        #for each flower type's features will have its own mean and variance
        #which will be used to calculate the probability of a test feature belonging to a given class
        setosaMeans = self.calcMeans(setosa)
        setosaVariances = self.calcVariance(setosa, setosaMeans)
        setosaPrior = len(setosa) / len(X)
        
        virginicaMeans = self.calcMeans(virginica)
        virginicaVariances = self.calcVariance(virginica, virginicaMeans)
        virginicaPrior = len(virginica) / len(X)
        
        versicolorMeans = self.calcMeans(versicolor)
        versicolorVariances = self.calcVariance(versicolor, versicolorMeans)
        versicolorPrior = len(versicolor) / len(X)
        
        self.mean = [setosaMeans, versicolorMeans, virginicaMeans]
        self.mean = np.reshape(self.mean, (3, 4))
        
        self.var = [setosaVariances, versicolorVariances, virginicaVariances]
        self.var = np.reshape(self.var, (3, 4))
        
        self.priors = [setosaPrior, versicolorPrior, virginicaPrior]
        self.priors = np.reshape(self.priors, -1)
        
        return [self.mean, self.var, self.priors]
        
    
    def calcMeans(self, flowers):
        res = [0, 0, 0, 0]
        for i in range(len(flowers)):
            for j in range(4):
                res[j] += flowers[i][j]
        
        res = [total / len(flowers) for total in res]
        return res
    
    def calcVariance(self, flowers, means):
        #the variance of a given feature is equal to the summation of differences between data point and mean squared
        #divied by total number of data points
        res = [0, 0, 0, 0]
        for flower in flowers:
            for i in range(4):
                res[i] += (flower[i] - means[i])**2
        return [total / len(flowers) for total in res]
                
        
        

    def gaussian_probability(self, x: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
        """
        Compute the Gaussian probability.

        Parameters:
        x : np.ndarray
            The input features
        mu : np.ndarray
            The mean values
        sigma2 : np.ndarray
            The variance values

        Returns:
        np.ndarray
            The Gaussian probabilities
        """
       #given our features, along with their means and variances
       #calculate the probabilities for each feature
        res = []
        for i in range(len(x)):
            featureRes = []
            for j in range(4):
                featureProb = (1/((2 * np.pi * sigma2[j])**(1/2))) * np.exp((-(x[i][j] - mu[j])**2) / (2 * sigma2[j]))
                featureRes.append(featureProb)
            res.append(featureRes)
        return np.reshape(res, (len(x), 4))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for given features using the trained Gaussian Naive Bayes model.

        Parameters:
        X : np.ndarray
            The features to predict

        Returns:
        np.ndarray
            The predicted classes
        """
        res = []
        setosaProb = [sum(probability) + np.log(self.priors[0]) for probability in self.gaussian_probability(X, self.mean[0], self.var[0])]
        virginicaProb = [sum(probability) + np.log(self.priors[2]) for probability in self.gaussian_probability(X, self.mean[2], self.var[2])]
        vericolorProb = [sum(probability) + np.log(self.priors[1]) for probability in self.gaussian_probability(X, self.mean[1], self.var[1])]
        for i in range(len(X)):
            bestProb = max(setosaProb[i], virginicaProb[i], vericolorProb[i])
            if bestProb == setosaProb[i]:
                res.append('Setosa')
            elif bestProb == virginicaProb[i]:
                res.append('Virginica')
            else:
                res.append('Versicolor')
        return np.reshape(res, -1)
        
        


def visualization(df: pd.DataFrame):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(35, 10))
    fig.suptitle("Data Visualization", fontsize=15)

    X = np.asarray(df.drop('variety', axis=1))
    x1, x2, x3, x4 = X.T[0], X.T[1], X.T[2], X.T[3]

    ax1.scatter(x1[:50], x2[:50], c='red')
    ax1.scatter(x1[50:100], x2[50:100], c='blue')
    ax1.scatter(x1[100:150], x2[100:150], c='green')
    ax1.set(xlabel='Sepal Length', ylabel='Sepal Width')

    ax2.scatter(x3[0:50], x4[0:50], c='red')
    ax2.scatter(x3[50:100], x4[50:100], c='blue')
    ax2.scatter(x3[100:150], x4[100:150], c='green')
    ax2.set(xlabel='Petal Length', ylabel="Petal Width")

    ax3.scatter(x1[:50]/x2[:50], x3[0:50]/x4[0:50], c='red')
    ax3.scatter(x1[50:100]/x2[50:100], x3[50:100]/x4[50:100], c='blue')
    ax3.scatter(x1[100:150]/x2[100:150], x3[100:150]/x4[100:150], c='green')
    ax3.set(xlabel='Sepal Length/Width', ylabel='Petal Length/Width')

    ax1.legend(['Iris-Setosa', 'Iris-Versicolor',
               'Iris-Virginica'], fontsize=15)
    ax2.legend(['Iris-Setosa', 'Iris-Versicolor',
               'Iris-Virginica'], fontsize=15)
    ax3.legend(['Iris-Setosa', 'Iris-Versicolor',
               'Iris-Virginica'], fontsize=15)

    plt.show()


# Testing
if __name__ == "__main__":
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Read the data
    train_df = read_data("./iris_training_data.csv")
    test_df = read_data("./iris_testing_data.csv")

    # Visualization for you to understand the data, you can comment it out
    # visualization(df)

    # Data preprocessing
    train_features = train_df.drop('variety', axis=1).values
    train_labels = train_df['variety'].values

    test_features = test_df.drop('variety', axis=1).values
    test_labels = test_df['variety'].values

    # Get the number of features
    num_features = train_features.shape[1]

    # Initialize and train the Naive Bayes classifier
    nb = GaussianNaiveBayes()
    nb.fit(train_features, train_labels)

    # Make predictions on the test set
    test_predictions = nb.predict(test_features)

    # Calculate accuracy
    accuracy = accuracy(test_labels, test_predictions)
    print("Naive Bayes Classification Accuracy: {:.2f}%".format(
        accuracy * 100))