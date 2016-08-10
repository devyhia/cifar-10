

# ## Random Forrest Classifier

# In[7]:

from init import *


# In[2]:

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import datasets

# In[4]:

X = train_features.reshape(50000, 3*32*32)
Xt = test_features.reshape(10000, 3*32*32)
y = train_labels.flatten()
yt = test_labels.flatten()


# In[17]:

def RFC(n=16):
    msg("[RFC/%d] Training" % n)
    rfc = RandomForestClassifier(n_estimators=16, verbose=True)
    rfc.fit(X, y)
    done()
    
    msg("[RFC/%d] Training Accuracy" % n)
    X_pred = rfc.predict(X)
    acc_train = metrics.accuracy_score(y, X_pred)
    done()
    
    msg("[RFC/%d] Testing Accuracy" % n)
    Xt_pred = rfc.predict(Xt)
    acc_test = metrics.accuracy_score(yt, Xt_pred)
    done()
    
    print("==== Training Accuracy ====")
    print(acc_train)
    print("==== Testing Accuracy ====")
    print(acc_test)
    print("")


# ### Experiments

# In[19]:

for x in [8,16,24,32,48,64,96,128,150,200,256]:
    RFC(x)


# In[ ]:



