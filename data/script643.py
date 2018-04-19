
# coding: utf-8

# # Lets try to predict CRA of alzheimer disease

# ![](http://yourcooladviser.in/wp-content/uploads/2017/06/stages-of-alzheimers-disease-21.jpg)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import keras
import glob
import seaborn as sns
import pandas as pd
import numpy as np
import timeit
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils


# In[ ]:


cross1=pd.read_csv('../input/oasis_longitudinal.csv') 
cross1 = cross1.fillna(method='ffill')
cross2=pd.read_csv('../input/oasis_cross-sectional.csv')
cross2 = cross2.fillna(method='ffill')


# In[ ]:


cross1.head()


# In[ ]:


cross2.head()


# In[ ]:


cross1.info()


# In[ ]:


cross2.head()


# In[ ]:


cross2.info()


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
#lets plot some graphics from the first dataset

from pylab import rcParams
rcParams['figure.figsize'] = 8, 5
cols = ['Age','MR Delay', 'EDUC', 'SES', 'MMSE', 'CDR','eTIV','nWBV','ASF']
x=cross1.fillna('')
sns_plot = sns.pairplot(x[cols])


# In[ ]:


#lets plot correleation matrix
corr_matrix =cross1.corr()
rcParams['figure.figsize'] = 15, 10
sns.heatmap(corr_matrix)


# In[ ]:


cross1.drop(['MRI ID'], axis=1, inplace=True)
cross1.drop(['Visit'], axis=1, inplace=True)


# In[ ]:


#cdr=cross1["CDR"]
cross1['CDR'].replace(to_replace=0.0, value='A', inplace=True)
cross1['CDR'].replace(to_replace=0.5, value='B', inplace=True)
cross1['CDR'].replace(to_replace=1.0, value='C', inplace=True)
cross1['CDR'].replace(to_replace=2.0, value='D', inplace=True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
for x in cross1.columns:
    f = LabelEncoder()
    cross1[x] = f.fit_transform(cross1[x])


# In[ ]:


cross1.head()


# In[ ]:


#cdr.replace(to_replace=0.0, value='A', inplace=True)
#cdr.replace(to_replace=0.5, value='B', inplace=True)
#cdr.replace(to_replace=1.0, value='C', inplace=True)
#cdr.replace(to_replace=2.0, value='D', inplace=True)


# In[ ]:


#from sklearn.preprocessing import LabelBinarizer
#encoder=LabelBinarizer()
#z1=encoder.fit_transform(cdr)


# In[ ]:


#print(z1)


# # Lets begin some machine learning

# In[ ]:


train, test = train_test_split(cross1, test_size=0.3)


# In[ ]:


X_train = train[['M/F', 'Age', 'EDUC', 'SES',  'eTIV', 'ASF']]
y_train = train.CDR
X_test = test[['M/F', 'Age', 'EDUC', 'SES',  'eTIV',  'ASF']]
y_test = test.CDR


# In[ ]:


# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)


# In[ ]:


y_train=np.ravel(y_train)
X_train=np.asarray(X_train)

y_test=np.ravel(y_test)
X_test=np.asarray(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)


# In[ ]:


classifier.score(X_test, y_test)


# In[ ]:


classifier.score(X_train, y_train)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=12)
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
print (classifier.score(X_train, y_train))
print (classifier.score(X_test, y_test))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))
prediction = knn.predict(X_test)
print(knn.score(X_test, y_test))


# In[ ]:


from sklearn.svm import SVC
svc=SVC(kernel="linear", C=0.01)
svc.fit(X_train, y_train)
prediction = svc.predict(X_test)


# In[ ]:


svc.score(X_test, y_test)


# In[ ]:


svc.score(X_train, y_train)


# In[ ]:


X_train.shape


# ## Neural net tensorflow

# ![](http://it-nowosti.ru/wp-content/uploads/2015/11/google-otkryvaet-isxodnyj-kod-si.jpg)

# In[ ]:


import tensorflow as tf
from sklearn import metrics
X_FEATURE = 'x'  # Name of the input feature.
feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(X_train).shape[1:])]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[35,70, 35], n_classes=4)

  # Train.
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_train}, y=y_train, num_epochs=100, shuffle=False)
classifier.train(input_fn=train_input_fn, steps=1000)

  # Predict.
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_test}, y=y_test, num_epochs=1, shuffle=False)
predictions = classifier.predict(input_fn=test_input_fn)
y_predicted = np.array(list(p['class_ids'] for p in predictions))
y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
scores = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


#if __name__ == '__main__':
   # tf.app.run()


# In[ ]:


y_train


# # We need to concat both datasets because we have insufficient data

# In[ ]:


cross1.head()


# In[ ]:


cross2.head()


# In[ ]:


#lets encode second dataset
for x in cross2.columns:
    f = LabelEncoder()
    cross2[x] = f.fit_transform(cross2[x])


# In[ ]:


#concanting both datasets
df = pd.concat([cross1,cross2])


# In[ ]:


df = df.fillna(method='ffill')
df.head()


# In[ ]:


train, test = train_test_split(cross1, test_size=0.3)
X_train1 = train[['ASF', 'Age', 'EDUC', 'Group',  'Hand', 'M/F','MMSE','MR Delay','SES','eTIV','nWBV']]
y_train1 = train.CDR
X_test1 = test[['ASF', 'Age', 'EDUC', 'Group',  'Hand', 'M/F','MMSE','MR Delay','SES','eTIV','nWBV']]
y_test1 = test.CDR


# In[ ]:


# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train1)

# Scale the train set
X_train1 = scaler.transform(X_train1)

# Scale the test set
X_test1 = scaler.transform(X_test1)


# In[ ]:


y_train1=np.ravel(y_train1)
X_train1=np.asarray(X_train1)

y_test1=np.ravel(y_test1)
X_test1=np.asarray(X_test1)


# In[ ]:


X_train1


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train1, y_train1)
prediction = classifier.predict(X_test1)
print(classifier.score(X_train1, y_train1))
print(classifier.score(X_test1, y_test1))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=5)
classifier.fit(X_train1, y_train1)
prediction = classifier.predict(X_test1)
print (classifier.score(X_train1, y_train1))
print (classifier.score(X_test1, y_test1))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train1, y_train1)
print(knn.score(X_train1, y_train1))
prediction = knn.predict(X_test1)
print(knn.score(X_test1, y_test1))


# In[ ]:


import tensorflow as tf
from sklearn import metrics
X_FEATURE = 'x'  # Name of the input feature.
feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(X_train1).shape[1:])]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[35,70,35], n_classes=4)

  # Train.
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_train1}, y=y_train1, num_epochs=100, shuffle=False)
classifier.train(input_fn=train_input_fn, steps=1000)

  # Predict.
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_test1}, y=y_test1, num_epochs=1, shuffle=False)
predictions = classifier.predict(input_fn=test_input_fn)
y_predicted = np.array(list(p['class_ids'] for p in predictions))
y_predicted = y_predicted.reshape(np.array(y_test1).shape)

  # Score with sklearn.
score = metrics.accuracy_score(y_test1, y_predicted)
print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
scores = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


# In[ ]:


import tensorflow as tf
from sklearn import metrics
X_FEATURE = 'x'  # Name of the input feature.
feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(X_train1).shape[1:])]

classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=4)

  # Train.
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_train1}, y=y_train1, num_epochs=100, shuffle=False)
classifier.train(input_fn=train_input_fn, steps=1000)

  # Predict.
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_test1}, y=y_test1, num_epochs=1, shuffle=False)
predictions = classifier.predict(input_fn=test_input_fn)
#y_predicted = np.array(list(p['class_ids'] for p in predictions))
y_predicted = y_predicted.reshape(np.array(y_test1).shape)

  # Score with sklearn.
score = metrics.accuracy_score(y_test1, y_predicted)
print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
scores = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


# ### And the winner is  DecisionTreeClassifier! 

# ### Conclusion: we need more data for more precise analysis
