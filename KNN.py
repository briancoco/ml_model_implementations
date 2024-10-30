import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List





def read_data(filename: str) -> pd.DataFrame:
    # read in the given file and formulate a pandas data frame using the data
    # we can use the pd.read_csv fn to read in the csv and return out the dataframe
    return pd.read_csv(filename)

def get_df_shape(df: pd.DataFrame) -> Tuple[int, int]:
    return df.shape

def extract_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = df[["sepal.length", "sepal.width"]]
    label = df["variety"]
    return (features, label)

def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=test_size
    )
    return x_train, y_train, x_test, y_test


def knn_test_score(
    n_neighbors: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    # takes in a value of k, alongside the training and testing data
    # and returns the accuracy of our knn model with the given testing data
    # we can calculate accuracy as the number of datapoints it got right divided by
    # number of data points
    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(x_train, y_train)
    modelResults = classifier.predict(x_test)
    totalCorrect = 0

    for predicted, actual in zip(modelResults, y_test):
        if predicted == actual:
            totalCorrect += 1
    return totalCorrect / len(y_test)


# ## Apply k-NN to a list of data
# Let Variable accu denote a list of accuracy corresponding to k[1,2,..,10]. You can use previously used functions (if they are correct)


def knn_evaluate_with_neighbours(
    n_neighbors_min: int,
    n_neighbors_max: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> List[float]:
    res = []
    for i in range(n_neighbors_min, n_neighbors_max + 1):
        res.append(knn_test_score(i, x_train, y_train, x_test, y_test))
    return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = read_data("./iris.csv")
    print(df)
    # assert on df
    shape = get_df_shape(df)
    # assert on shape
    features, label = extract_features_label(df)
    x_train, y_train, x_test, y_test = data_split(features, label, 0.33)
    print(knn_test_score(1, x_train, y_train, x_test, y_test))
    acc = knn_evaluate_with_neighbours(1, 10, x_train, y_train, x_test, y_test)
    plt.plot(range(1, 11), acc)
    plt.show()
