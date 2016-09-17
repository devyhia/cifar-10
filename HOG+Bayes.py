from sklearn import datasets
import os
import h5py
import six
import cv2
from six.moves import range, cPickle
import tarfile
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import datasets
from scipy import interpolate
import sklearn.svm as svm
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# ### Load Training Data
tar_file = tarfile.open("/home/hagar13/Downloads/cifar-10-python.tar.gz", 'r:gz')
train_batches = []
for batch in range(1, 6):
    file = tar_file.extractfile(
        'cifar-10-batches-py/data_batch_%d' % batch)
    try:
        if six.PY3:
            array = cPickle.load(file, encoding='latin1')
        else:
            array = cPickle.load(file)
        train_batches.append(array)
    finally:
        file.close()

train_features = np.concatenate(
    [batch['data'].reshape(batch['data'].shape[0], 3, 32, 32)
        for batch in train_batches])
train_labels = np.concatenate(
    [np.array(batch['labels'], dtype=np.uint8)
        for batch in train_batches])
train_labels = np.expand_dims(train_labels, 1)
# ### Load Testing Dat
file = tar_file.extractfile('cifar-10-batches-py/test_batch')
try:
    if six.PY3:
        test = cPickle.load(file, encoding='latin1')
    else:
        test = cPickle.load(file)
finally:
    file.close()

test_features = test['data'].reshape(test['data'].shape[0], 3, 32, 32)
test_labels = np.array(test['labels'], dtype=np.uint8)
test_labels = np.expand_dims(test_labels, 1)
LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']
# * 10,000 testing image
print(train_features.shape)
print(test_features.shape)
# ### Data Visualization
# Here, you need to note that the image size for `plt.imshow` should be ($width \times height \times RGB$) i.e. ($32 \times 32 \times 3$). The default is ($3 \times 32 \times 32$), that's why I use: `.T` (which is the transpose of $32 \times 32 \times 3$ = $3 \times 32 \times 32$).
# `np.rot90` does 90-degree rotation on the image. `k` is the a multiplier (i.e. if `k=1`, we have 90 deg rotation. If `k=2`, we have 180 deg rotation. If `k=3`, we have 270 deg rotation
train_images = np.array([np.rot90(train_features[i].T, k=3) for i in range(0,50000)])           # Train Images
test_images = np.array([np.rot90(test_features[i].T, k=3) for i in range(0,10000)])             # Test Images

winSize = (2,2)
blockSize = (2,2)
blockStride = (2,2)
cellSize = (2,2)
nbins = 3
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 2
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#sift = cv2.HOGDescriptor()

ImageSize = 128
features = []
#r_image = cv2.resize(image, (ImageSize, ImageSize))
#r_image = cv2.resize(image,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
for image in train_images:
    r_image = cv2.resize(image,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(r_image, cv2.COLOR_BGR2GRAY)
    dsc = hog.compute(gray)
    #print len(dsc)
    features.append(dsc)

print 'Features Done'
print np.shape(features)

#print type(features)

T = np.asarray(features)

nsamples, nx, ny = np.shape(T)
x = T.reshape((nsamples,nx*ny))
y = np.ndarray.ravel(train_labels)

np.savetxt('HOGfeatures.txt', x)
dtc = GaussianNB()
dtc.fit(x, y)

X_pred = dtc.predict(x)

print metrics.accuracy_score(y, X_pred)                  #38.87%
