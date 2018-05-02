
# coding: utf-8

# <h1>Lower Back Pain Classification Algorithm </h1>
# 
# <p>This dataset contains the anthropometric measurements of the curvature of the spine to support the model towards a more accurate classification.
# <br />
# Lower back pain affects around 80% of individuals at some point in their life. If this model becomes robust enough, then these measurements may soon become predictive and treatable measures. 
# <br /> 
# <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.471.4845&rep=rep1&type=pdf">This study</a> asserts the validity of the manual goniometer measurements as a valid clinical tool. </p>

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns

# read data into dataset variable
data = pd.read_csv("../input/Dataset_spine.csv")

# Drop the unnamed column in place (not a copy of the original)#
data.drop('Unnamed: 13', axis=1, inplace=True)

# Concatenate the original df with the dummy variables
data = pd.concat([data, pd.get_dummies(data['Class_att'])], axis=1)

# Drop unnecessary label column in place. 
data.drop(['Class_att','Normal'], axis=1, inplace=True)


# In[3]:


data.head()


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <h1>Exploratory Data Analysis </h1>

# In[5]:


data.columns = ['Pelvic Incidence','Pelvic Tilt','Lumbar Lordosis Angle','Sacral Slope','Pelvic Radius', 
                'Spondylolisthesis Degree', 'Pelvic Slope', 'Direct Tilt', 'Thoracic Slope', 
                'Cervical Tilt','Sacrum Angle', 'Scoliosis Slope','Outcome']

corr = data.corr()

# Set up the matplot figure
f, ax = plt.subplots(figsize=(12,9))

#Draw the heatmap using seaborn
sns.heatmap(corr, cmap='inferno', annot=True)


# In[6]:


data.describe()


# In[7]:


from pylab import *
import copy
outlier = data[["Spondylolisthesis Degree", "Outcome"]]
#print(outlier[outlier >200])
abspond = outlier[outlier["Spondylolisthesis Degree"]>15]
print("1= Abnormal, 0=Normal\n",abspond["Outcome"].value_counts())


# In[8]:


#   Dropping Outlier
data = data.drop(115,0)
colr = copy.copy(data["Outcome"])
co = colr.map({1:0.44, 0:0.83})

#   Plot scatter
plt.scatter(data["Cervical Tilt"], data["Spondylolisthesis Degree"], c=co, cmap=plt.cm.RdYlGn)
plt.xlabel("Cervical Tilt")
plt.ylabel("Spondylolisthesis Degree")

colors=[ 'c', 'y', 'm',]
ab =data["Outcome"].where(data["Outcome"]==1)
no = data["Outcome"].where(data["Outcome"]==0)
plt.show()
# UNFINISHED ----- OBJECTIVE: Color visual by Outcome - 0 for green, 1 for Red (example)


# In[9]:


#   Create the training dataset
training = data.drop('Outcome', axis=1)
testing = data['Outcome']


# In[10]:


#   Import necessary ML packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

#   Split into training/testing datasets using Train_test_split
X_train, X_test, y_train, y_test = train_test_split(training, testing, test_size=0.33, random_state=22, stratify=testing)


# <h1> Convert DataFrame Object to a numpy array due to faster computation in modelling</h1>

# In[11]:


import numpy as np

# convert to numpy.ndarray and dtype=float64 for optimal
array_train = np.asarray(training)
array_test = np.asarray(testing)
print(array_train.shape)
print(array_test.shape)

#   Convert each pandas DataFrame object into a numpy array object. 
array_XTrain, array_XTest, array_ytrain, array_ytest = np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)


# <h1> Employing Support Vector Machine as a Classifier - 85% </h1>

# In[12]:


#    Import Necessary Packages
from sklearn import svm
from sklearn.metrics import accuracy_score

#   Instantiate the classifier
clf = svm.SVC(kernel='linear')

#   Fit the model to the training data
clf.fit(array_XTrain, array_ytrain)

#   Generate a prediction and store it in 'pred'
pred = clf.predict(array_XTest)

#   Print the accuracy score/percent correct
svmscore = accuracy_score(array_ytest, pred)
print("Support Vector Machines are ", svmscore*100, "accurate")


# <h1> That's it! </h1>
# <p>~85% prediction accuracy with Support Vector Machines!  To increase the accuracy of the model, feature engineering is a suitable solution - as well as creating new variables based on domain knowledge.</p>

# <h2> Next Steps</h2>
# <li> Since we've done no feature engineering or any parameter tuning, there is a lot of room for improvement. </li>
# <li>Specifically, an ANN has been shown to acheive a 93% accuracy score when predicting low back pain from this study</li>
# 

# In[13]:


from keras.models import Sequential
from keras.layers import Dense, Activation
import keras


# In[14]:


print(array_XTrain.shape)
print(array_ytrain.shape)


# In[15]:


#  Define our model
model = Sequential()
model.add(Dense(32, activation='tanh', input_dim=12))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

one_hot_labels = keras.utils.to_categorical(array_ytrain, num_classes=10)

history = model.fit(array_XTrain, one_hot_labels,epochs=1000, batch_size=30)
weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]


# In[16]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(history.history['loss'])
plt.xlabel("Epochs (Batches)")
plt.ylabel("Loss")
plt.title("Training an Artificial Neural Net")


# <h3>A Little Note About Input Shapes</h3>
# 
# The input dimension on the input layer of a Neural Net (NN) seems to always cause me issues. It begins with me thinking that dimensions refer to the length of the input data. Although I still want to confirm, this is a misconception on my part. Input dimension (or shape) refers to the number of fields in the input data. 
# 
# "Fields", in this case, refers to the variables within the input data. The number of Fields, Features or Dimension should be the value of the input shape. 

# <p> After we've set up the model's parameters we must choose both a loss function and define the learning rate. An analogy to understand learning rate is to imagine a bowl. To find the lowest (or highest) point of the bowl, you take a small 'leap' from your current position to a random location. 
# 
# * If the jump is too large, you risk making too large a step and stepping over the most optimal position. 
# * If the jump is too small, you risk increasing the computational cost. I believe the standard is to set it to 0.1 as a 'default'. 
