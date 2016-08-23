import os
import h5py
import six
import opencv2 as cv2
from six.moves import range, cPickle
import tarfile
import numpy as np
import matplotlib.pyplot as plt

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

# Feature Extraction part
Alldescriptors = [];
AllKeypoints = [];
ImageSize = 128
detector = cv2.SURF()
for image in train_images:
    resized_image = cv2.resize(image, (ImageSize, ImageSize))             # Images have been resized up to get more features ( a parameter to play with)
    keypoints,descriptors = detector.detectAndCompute(resized_image,None)
    Out_image = cv2.drawKeypoints(resized_image,keypoints,None,(255,0,0),2)

    Alldescriptors.append(descriptors)
    AllKeypoints.append(keypoints)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

# Feature matching part
N = 10;                     # number of classes we haves
BestFeatures = {}           # a dictionary to save best features per class
MaxDist = 0.1               # max distance that needs to be there to consider a feature close
MaxCount = 1               # MaxCount of features that needs to be there to consider a feature good
print("Enters the features exploration part")
for label in range(N):
    indices = np.where(train_labels == label)
    #des1 = Alldescriptors[indices[0][0]]
    # get the image that best represents the class ( the one with the Maximum number of features). The other approach was to choose the first image
    MaxLength = 0
    MaxIndex = 0
    for j in range(len(indices[0])):
        if len(Alldescriptors[indices[0][j]]) > MaxLength:
            MaxLength = len(Alldescriptors[indices[0][j]])
            MaxIndex = indices[0][j]
    #print MaxLength
    #print MaxIndex
    des1 = Alldescriptors[MaxIndex]
    # Loop over all other images to compare the descriptors
    for j in range(len(indices[0])):
        if j != MaxIndex:
            index = indices[0][j]
            des2 = Alldescriptors[index]
            counter = 0
            AllDesc = []
            for x in des1:
                counter = 0
                for y in des2:
                    t = np.subtract(x,y)
                    dist = np.sqrt(np.sum(t)**2)           # calculate the distance between two descriptors
                    if dist < MaxDist:
                        counter = counter + 1
                if counter > MaxCount:
                    AllDesc.append(x)
    BestFeatures[label] = AllDesc

# Dictionary for the best features we got; key represents the label, value represents the descriptors
print(len(BestFeatures))
print(len(BestFeatures[0]))
print(len(BestFeatures[1]))
print(len(BestFeatures[2]))
print(len(BestFeatures[3]))
print(len(BestFeatures[4]))
print(len(BestFeatures[5]))
print(len(BestFeatures[6]))
print(len(BestFeatures[7]))
print(len(BestFeatures[8]))
print(len(BestFeatures[9]))
