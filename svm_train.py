from init import *

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import sklearn.svm as svm
from sklearn import metrics
from sklearn import datasets
from sklearn.decomposition import PCA

X = train_features.reshape(50000, 3*32*32)
Xt = test_features.reshape(10000, 3*32*32)
y = train_labels.flatten()
yt = test_labels.flatten()

def SVM_SVC(itr=1, _X=None, _Xt=None):
    if _X is None:
        _X = X

    if _Xt is None:
        _Xt = Xt

    msg("[SVM POLY %d] Training" % itr)
    svc = svm.SVC(max_iter=itr, kernel='poly')
    svc.fit(X, y)
    done()

    msg("[SVM POLY %d] Training Accuracy" % itr)
    X_pred = svc.predict(X)
    msg(metrics.accuracy_score(y, X_pred))
    done()

    msg("[SVM POLY %d] Testing Accuracy" % itr)
    Xt_pred = svc.predict(Xt)
    msg(metrics.accuracy_score(yt, Xt_pred))
    done()

def SVM_SVC_SIG(_X=None, _Xt=None, I=2):
    if _X is None:
        _X = X

    if _Xt is None:
        _Xt = Xt

    msg("[SVM SIG %d] Training" %I)
    svc = svm.SVC(kernel='sigmoid', max_iter=I)
    svc.fit(X, y)
    done()

    msg("[SVM SIG %d] Training Accuracy"%I)
    X_pred = svc.predict(X)
    msg(metrics.accuracy_score(y, X_pred))
    done()

    msg("[SVM SIG %d] Testing Accuracy"%I)
    Xt_pred = svc.predict(Xt)
    msg(metrics.accuracy_score(yt, Xt_pred))
    done()

for i in [500,1000,2000,3000,-1]:
    SVM_SVC_SIG(I=i)
