from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

def entropy(x):

  hist  = np.bincount(x)
  ps = hist / len(x)
  res = 0
  for p in ps:
    if p > 0:
      res += p * np.log(p)
  return -res

class Node:
  def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value

  def is_leaf(self):
    return self.value is not None

class TreeRegressor:
  def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    self.root = None
    self.n_feats = n_feats
    print(max_depth)

  def fit(self, X, y):
    self.n_feats = X.shape[1] if not self.n_feats else min(X.shape[1], self.n_feats)
    self.root = self.build_tree(X, y)

  def predict(self, X):

    return np.array([self.traverse_tree(x, self.root) for x in X])

  def build_tree(self, X, y, depth=0):

    ### YOUR CODE HERE
    ##stopping criteria
    ##CAN USE common_thing()
    ##TO DO
    ##get_best_split()
    ##
    ##split()

    n_samples, n_features = X.shape
    n_labels = len(np.unique(y))
    
    if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
      leaf_value = self.common_thing(y)
      return Node(value=leaf_value)
    
    feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
    
    best_feature, best_thresh = self.get_best_split(X, y, feat_idxs)
    left_idxs, right_idxs = self.split(X[:, best_feature], best_thresh)
    left = self.build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
    right = self.build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
    return Node(best_feature, best_thresh, left, right)
    

  def get_best_split(self, X, y, feat_idxs):

    ### YOUR CODE HERE
    ##
    ##information_gain()
    
    best_gain = -1
    split_idx, split_threshold = None, None
    
    for feat_idx in feat_idxs:
      X_column = X[:, feat_idx]
      thresholds = np.unique(X_column)
      
      for thr in thresholds:
        gain = self.information_gain(y, X_column, thr)
        
        if gain > best_gain:
          best_gain = gain
          split_idx = feat_idx
          split_threshold = thr
    
    return split_idx, split_threshold

    

  def information_gain(self, y, X_column, split_thresh):
    ### YOUR CODE HERE

    # parent loss
    ##entropy()
    ##
    # generate split
    ##spit()
    ##
    # compute the weighted avg. of the loss for the children

    # information gain is difference in loss before vs. after split

    parent_entropy = entropy(y)
    
    left_idxs, right_idxs = self.split(X_column, split_thresh)
    
    if len(left_idxs) == 0 or len(right_idxs) == 0:
      return 0

    n = len(y)
    n_l, n_r = len(left_idxs), len(right_idxs)
    e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
    child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
    
    information_gain = parent_entropy - child_entropy
    return information_gain

  def split(self, X_column, split_thresh):

    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
    right_idxs = np.argwhere(X_column > split_thresh).flatten()
    return left_idxs, right_idxs

    

  def traverse_tree(self, x, node):

    if node.is_leaf():
      return node.value
  
    if x[node.feature] <= node.threshold:
      return self.traverse_tree(x, node.left)
    return self.traverse_tree(x, node.right)

  def common_thing(self, y):
    count = Counter(y)
    common_thing = count.most_common(1)[0][0]
    return common_thing


if __name__ == "__main__":

  data = datasets.load_iris()
  X, y = data.data, data.target

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  accuracy_depths = []
  for depth in range(1, 6):   
    clf = TreeRegressor(max_depth=depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)
    accuracy_depths.append(acc)
    print("Accuracy at depth %d: %f" % (depth, acc))

  plt.figure()
  plt.plot(accuracy_depths)
  plt.xlabel("Depth")
  plt.ylabel("Accuracy")
  plt.show()