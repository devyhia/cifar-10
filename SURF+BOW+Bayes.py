from sklearn import datasets
import os
import h5py
import six
import cv2
from six.moves import range, cPickle
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import datasets
from scipy import interpolate

iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# ### Load Training Data
tar_file = tarfile.open("cifar-10-python.tar.gz", 'r:gz')
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

dictionarySize = 1500
sift = cv2.SURF()
BOW = cv2.BOWKMeansTrainer(dictionarySize)
ImageSize = 256
#r_image = cv2.resize(image, (ImageSize, ImageSize))
for image in train_images:
    r_image = cv2.resize(image,None,fx=5, fy=5, interpolation = cv2.INTER_CUBIC)
    kp, dsc= sift.detectAndCompute(r_image, None)
    #print len(dsc)
    BOW.add(dsc)

#dictionary created
dictionary = BOW.cluster()
print len(dictionary)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
sift2 = cv2.DescriptorExtractor_create("SURF")
bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)
print "bow dictionary", np.shape(dictionary)

#returns descriptor of image at pth
def feature_extract(pth):
    r_image = cv2.resize(pth, (ImageSize, ImageSize))
    return bowDiction.compute(r_image, sift.detect(r_image))

train_desc = []

i = 0
for p in train_images:
    train_desc.extend(feature_extract(p))
    i = i+1

x = train_desc
y = np.ndarray.ravel(train_labels)

print "Train_desc", np.shape(x)
np.savetxt('SURFfeatures.txt', x)
dtc = GaussianNB()
dtc.fit(x, y)

X_pred = dtc.predict(x)

print metrics.accuracy_score(y, X_pred)                # Accuracy = 35.65%
