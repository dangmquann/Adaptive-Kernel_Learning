import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

class KernelClassifier:
    def __init__(self):
        self.clf = None

    def fit(self, K_X, y_):
        y = np.copy(y_)
        if y.shape[0] > 1 and y.shape[1] > 1: # kernel matrix
            y = y[0,:]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            y = 1- y

        self.clf = SVC(kernel='precomputed', probability=True)
        self.N_train = K_X.shape[0]
        self.clf.fit(K_X, y)
        return self

    def predict(self, K_X):
        if K_X.shape[0] != self.N_train and K_X.shape[1] == self.N_train:
            K_X = K_X
        elif K_X.shape[0] == self.N_train:
            K_X = K_X.T
        else:
            raise ValueError("Input kernel matrix has wrong shape")
        return self.clf.predict(K_X)
        # return self.clf.predict_proba(K_X)[:, 1]

    
    def predict_proba(self, K_X):
        if K_X.shape[0] != self.N_train and K_X.shape[1] == self.N_train:
            pass
        elif K_X.shape[0] == self.N_train:
            K_X = K_X.T
        else:
            raise ValueError("Input kernel matrix has wrong shape")
        return self.clf.predict_proba(K_X)[:,1]

    def score(self, X, y):
        return self.clf.score(X, y)




