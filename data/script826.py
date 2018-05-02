
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import xgboost
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/mushrooms.csv")
data.head(10)


# In[ ]:


labelEncoder = preprocessing.LabelEncoder()
for col in data.columns:
    data[col] = labelEncoder.fit_transform(data[col])

# Splitting test train set, with 20% of the data as the validation set
train, test = train_test_split(data, test_size = 0.2) 


# In[ ]:


# Train set
train_y = train['class']
train_x = train[[x for x in train.columns if 'class' not in x]]
# Test/Validation set
test_y = test['class']
test_x = test[[x for x in test.columns if 'class' not in x]]

models = [SVC(kernel='rbf', random_state=0), SVC(kernel='linear', random_state=0), XGBClassifier(), LogisticRegression()]
model_names = ['SVC_rbf', 'SVC_linear', 'xgboost', 'Logistic Regression']
for i, model in enumerate(models):
    model.fit(train_x, train_y)
    print ('The accurancy of ' + model_names[i] + ' is ' + str(accuracy_score(test_y, model.predict(test_x))) )


# In[ ]:


ax = xgboost.plot_importance(models[2])


# The dataset seems really straight forward, with the problem being very much linear in nature as both Logistic regression and SVM with a linear kernel had accuracy of more than 90%. 
