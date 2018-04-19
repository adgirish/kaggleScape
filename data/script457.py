
# coding: utf-8

# Hello everyone! In this kernel is represented 1 dimensional convolutional neural network. The idea is simple, without tuning of model's hyperparameters. The submission file is provided.

# Feature binarization and scaling created by our team

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

SEED = 42
np.random.seed(SEED)

from sklearn.preprocessing import StandardScaler, LabelBinarizer

#binarization of features
class FeatureBinarizatorAndScaler:
    """ This class needed for scales and factorize features
    """
    NUMERICAL_FEATURES = list()
    CATEGORICAL_FEATURES = list()
    BIN_FEATURES = list()
    binarizers = dict()
    scalers = dict()

    def __init__(self, numerical=list(), categorical=list(), binfeatures = list(), binarizers=dict(), scalers=dict()):
        self.NUMERICAL_FEATURES = numerical
        self.CATEGORICAL_FEATURES = categorical
        self.BIN_FEATURES = binfeatures
        self.binarizers = binarizers
        self.scalers = scalers

    def fit(self, train_set):
        for feature in train_set.columns:

            if feature.split('_')[-1] == 'cat':
                self.CATEGORICAL_FEATURES.append(feature)
            elif feature.split('_')[-1] != 'bin':
                self.NUMERICAL_FEATURES.append(feature)
            else:
                self.BIN_FEATURES.append(feature)
        for feature in self.NUMERICAL_FEATURES:
            scaler = StandardScaler()
            self.scalers[feature] = scaler.fit(np.float64(train_set[feature]).reshape((len(train_set[feature]), 1)))
        for feature in self.CATEGORICAL_FEATURES:
            binarizer = LabelBinarizer()
            self.binarizers[feature] = binarizer.fit(train_set[feature])

    def transform(self, data):
        binarizedAndScaledFeatures = np.empty((0, 0))
        for feature in self.NUMERICAL_FEATURES:
            if feature == self.NUMERICAL_FEATURES[0]:
                binarizedAndScaledFeatures = self.scalers[feature].transform(np.float64(data[feature]).reshape(
                    (len(data[feature]), 1)))
            else:
                binarizedAndScaledFeatures = np.concatenate((
                    binarizedAndScaledFeatures,
                    self.scalers[feature].transform(np.float64(data[feature]).reshape((len(data[feature]),
                                                                                       1)))), axis=1)
        for feature in self.CATEGORICAL_FEATURES:

            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures,
                                                         self.binarizers[feature].transform(data[feature])), axis=1)

        for feature in self.BIN_FEATURES:
            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures, np.array(data[feature]).reshape((len(data[feature]),
                                                                                       1))), axis=1)

        print(binarizedAndScaledFeatures.shape )

        return binarizedAndScaledFeatures


# Convolutional Neural Network implementation

# In[ ]:



from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.initializers import random_uniform

import pandas as pd

X_train = pd.read_csv('../input/train.csv')
y_train = X_train['target']
X_test = pd.read_csv('../input/test.csv')
test_id = X_test['id']
X_test = X_test.drop(['id'], axis=1)
X_train = X_train.drop(['id', 'target'], axis = 1)
y_train1 = abs(-1+y_train)
y_train = pd.concat([y_train, y_train1], axis=1)
binarizerandscaler = FeatureBinarizatorAndScaler()
binarizerandscaler.fit(X_train)
X_train = binarizerandscaler.transform(X_train)
X_test = binarizerandscaler.transform(X_test)
y_train = y_train.as_matrix()


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#hyperparameters
input_dimension = 226
learning_rate = 0.0025
momentum = 0.85
hidden_initializer = random_uniform(seed=SEED)
dropout_rate = 0.2


# create model
model = Sequential()
model.add(Convolution1D(nb_filter=32, filter_length=3, input_shape=X_train.shape[1:3], activation='relu'))
model.add(Convolution1D(nb_filter=16, filter_length=1, activation='relu'))
model.add(Flatten())
model.add(Dropout(dropout_rate))
model.add(Dense(128, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(64, kernel_initializer=hidden_initializer, activation='relu'))
model.add(Dense(2, kernel_initializer=hidden_initializer, activation='softmax'))

sgd = SGD(lr=learning_rate, momentum=momentum)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])
model.fit(X_train, y_train, epochs=5, batch_size=128)
predictions = model.predict_proba(X_test)

ans = pd.DataFrame(predictions)
ans = ans[0]


# In[ ]:


# Create submission file
sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = ans
sub.to_csv('submission.csv', float_format='%.6f', index=False)

