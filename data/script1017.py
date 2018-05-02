
# coding: utf-8

# ## Using Neural Networks through Keras

# Copied from Kaggle itself - see if I make it better !

# In[ ]:


## Measure execution time, becaus Kaggle cloud fluctuates  
import time
start = time.time()


# In[ ]:


## Importing standard libraries
get_ipython().run_line_magic('pylab', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


## Importing sklearn libraries

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


# In[ ]:


## Keras Libraries for Neural Networks

from keras.models import Sequential
from keras.layers import Merge
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping


# In[ ]:


## Read data from the CSV file
data = pd.read_csv('../input/train.csv')
parent_data = data.copy()    ## Always a good idea to keep a copy of original data
ID = data.pop('id')


# In[ ]:


data.shape
data.describe()


# In[ ]:


## Since the labels are textual, so we encode them categorically

y = data.pop('species')
y = LabelEncoder().fit(y).transform(y)
print(y.shape)


# In[ ]:


## Most of the learning algorithms are prone to feature scaling
## Standardising the data to give zero mean =)
from sklearn import preprocessing
X = preprocessing.MinMaxScaler().fit(data).transform(data)
X = StandardScaler().fit(data).transform(data)
## normalizing does not help here; l1 and l2 allowed
## X = preprocessing.normalize(data, norm='l1')
print(X.shape)
X


# In[ ]:


## We will be working with categorical crossentropy function
## It is required to further convert the labels into "one-hot" representation

y_cat = to_categorical(y)
print(y_cat.shape)


# In[ ]:


## retain class balances
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2,random_state=12345)
train_index, val_index = next(iter(sss.split(X, y)))
x_train, x_val = X[train_index], X[val_index]
y_train, y_val = y_cat[train_index], y_cat[val_index]
print("x_train dim: ",x_train.shape)
print("x_val dim:   ",x_val.shape)


# In[ ]:


## Developing a layered model for Neural Networks
## Input dimensions should be equal to the number of features
## We used softmax layer to predict a uniform probabilistic distribution of outcomes
## https://keras.io/initializations/ ;glorot_uniform, glorot_normal, lecun_uniform, orthogonal,he_normal

model = Sequential()
model.add(Dense(768,input_dim=192,  init='glorot_normal', activation='tanh'))
model.add(Dropout(0.4))

model.add(Dense(768, activation='tanh'))
model.add(Dropout(0.4))

model.add(Dense(99, activation='softmax'))


# In[ ]:


## Error is measured as categorical crossentropy or multiclass logloss
## Adagrad, rmsprop, SGD, Adadelta, Adam, Adamax, Nadam

model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])


# In[ ]:


## Fitting the model on the whole training data with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=300)

history = model.fit(x_train, y_train,batch_size=192,nb_epoch=2500 ,verbose=0,
                    validation_data=(x_val, y_val),callbacks=[early_stopping])


# In[ ]:


## we need to consider the loss for final submission to leaderboard
## print(history.history.keys())
print('val_acc: ',max(history.history['val_acc']))
print('val_loss: ',min(history.history['val_loss']))
print('train_acc: ',max(history.history['acc']))
print('train_loss: ',min(history.history['loss']))

print()
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))


# In[ ]:


## summarize history for loss
## Plotting the loss with the number of iterations
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


## Plotting the error with the number of iterations
## With each iteration the error reduces smoothly
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


## read test file
test = pd.read_csv('../input/test.csv')
index = test.pop('id')

## we need to perform the same transformations from the training set to the test set
test = preprocessing.MinMaxScaler().fit(test).transform(test)
test = StandardScaler().fit(test).transform(test)
yPred = model.predict_proba(test)


# In[ ]:


## Converting the test predictions in a dataframe as depicted by sample submission
yPred = pd.DataFrame(yPred,index=index,columns=sort(parent_data.species.unique()))


# In[ ]:


## write submission to file
fp = open('submission_nn_kernel.csv','w')
fp.write(yPred.to_csv())

## print run time
end = time.time()
print(round((end-start),2), "seconds")


# ---------
# 
# Earlier` we used a 4 layer network but the result came out to be overfitting the test set. We dropped the count of neurones in the network and also restricted the number of layers to 3 so as to keep it simple.
# Instead of submitting each test sample as a one hot vector we submitted each samples as a probabilistic distribution over all the possible outcomes. This "may" help reduce the penalty being exercised by the multiclass logloss thus producing low error on the leaderboard! ;)
# Any suggestions are welcome!
