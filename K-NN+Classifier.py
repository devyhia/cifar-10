
# coding: utf-8

# ## K-NN Classifier

# In[1]:

from init import *

print("LOADED DATA ...")

# In[2]:

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

print("LOADED LIBS ...")

# In[4]:

X = train_features.reshape(50000, 3*32*32)
Xt = test_features.reshape(10000, 3*32*32)
y = train_labels.flatten()
yt = test_labels.flatten()

print("SHAPED DATA ...")

# K-NN Class
class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
      print("\rClassifiying {} ...".format(i),end="")

    return Ypred

print("LOADED CLASS ...")

# K-NN Classification

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(X, y) # train the classifier on the training images and labels
print("TRAINED CLASSIFIER ...")
Xt_predict = nn.predict(Xt) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print("PREDICTING CLASSIFIER ...")
print('accuracy: %f' % ( metrics.accuracy_score(yt, Xt_predict)))
