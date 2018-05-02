
# coding: utf-8

# # Introduction

# Often while working on a machine learning problem we come across situations where our model is working well on our validation data but it performs poorly on the test data. One of the possible reasons can be that the validation set is not a true representative of the test data. That means the input features (covariates) in our test data follow a different distribution compared to our validation/training data. This situation is also called ***Covariate Shift***. Here I'm discussing a method to identify whether covariate shift exist between the 2 datasets that we are comparing. I have also stated some ideas related to how we can adjust for this shift.
# 
# This kernel is inspired by the following discussion: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43453  and the code is borrowed from the following repository: https://github.com/erlendd/covariate-shift-adaption . The method discussed here comes from refs [1-2].
# @tilli your posts are amazing!!
# 
# Any improvements or possible ideas are welcome.
# 
# ##### References 
# [1] Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference, 90, 227–244.<br/>
# [2] Bickel, S. *et al.* (2009). Discriminative Learning Under Covariate Shift. Journal of Machine Learning Research, 10, 2137-2155<br/>
# 
# 

# ***Note: In this notebook I have compared train and test data but we can use the same method to compare any validation set to the test data***

# # Theory

# The covariates are some features (often represented by $X$) with an associated response variable $y$, and our goal in machine-learning is to compute $p(y|x,model)$ using some model.
# 
# The expected loss (l) of a model on the test data is written, $E[l(f(X_{test}), y)]$ where $f(x)$ maps X to $\hat{y}$. It is shown in [1] that this is related to the expected training loss by a factor $\beta = \frac{p(X_{te})}{p(X_{train})}$, i.e.
# 
# $E\left[l(f(X_{test}), y)\right] = E\left[\frac{p(X_{test})}{p(X_{train})}\,l(f(X_{train}), y)\right]$.
# 
# What does this mean? First, $p(X_{test})$ being on the numerator of $\beta$ means that points in the training data that are close to high-density regions of the test data will be weighted-up. This makes a lot of sense intuitively. Second, the $p(X_{train})$ term on the demoninator of $\beta$ actually reduces the contribution from training points that are in a high-density region of the training data. Therefore in a region of feature-space where the training and test distributions are very similar $\beta$ will be close to one, and in more imbalanced regions $\beta$ should vary compensate.
# 
# How to actually use this? There are different ways to make use of this discussed in the literature. One method is using kernel means matching, to match the differing distributions. Another simpler method is by training a discriminative classifier to learn $p(X_{train}|D)$, for some data $D$.
# 
# For an example of this in action, see below!
# 
# 

# # Porto Seguro’s Safe Driver Prediction
# I'm using data from this kaggle competition for exposition.
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

# In[11]:


import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import train_test_split
import pylab as plt
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import forest
from sklearn.tree import export_graphviz


# ## Data Loading 

# In[12]:


#loading test and train data
train = pd.read_csv('../input/unzippedraw/train.csv',low_memory=True)
test = pd.read_csv('../input/unzippedraw/test.csv',low_memory=True)


# In[13]:


#making copy
trn =  train.copy()
tst =  test.copy()


# ## Data Preparation

# In[14]:


#adding a column to identify whether a row comes from train or not
tst['is_train'] = 0
trn['is_train'] = 1 #1 for train


# In[15]:


tst.shape,trn.shape


# Test data has 892816 rows and train has 595212 rows. We have 59 independent variables.

# In[16]:


#combining test and train data
df_combine = pd.concat([trn, tst], axis=0, ignore_index=True)
#dropping 'target' column as it is not present in the test
df_combine = df_combine.drop('target', axis=1)


# For our current dataset **df_combine**, 'is_train' is the label to predict

# In[17]:


y = df_combine['is_train'].values #labels
x = df_combine.drop('is_train', axis=1).values #covariates or our dependent variables


# In[18]:


tst, trn = tst.values, trn.values


# ## Covariate Shift Analysis

# We are using Random Forest Classifier to predict the labels for each observation whether it is coming from train or not. You can also use a different classifier like logistic classifier.

# In[19]:


def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))


# In[20]:


def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))


# **Tip: If your dataset is large then you can use set_rf_samples() to pass a randomly subset of data to each tree instead of passsing all the rows. This makes a huge difference in computation time and reduces overfitting. These functions are taken from fastai library. Checkout this awesome library for deep learning and machine learning.
# https://github.com/fastai/fastai**
# 
# <br>

# In[21]:


set_rf_samples(60000) 
# reset_rf_samples() to revert back to default behavior


# ### Building a classifier

# In[23]:


m = RandomForestClassifier(n_jobs=-1,max_depth=5)
predictions = np.zeros(y.shape)


# We are using stratified 4 fold to ensure that percentage for each class is preserved and we cover the whole data once. 
# For each row the classifier will calculate the probability of it belonging to train.

# In[24]:


skf = SKF(n_splits=20, shuffle=True, random_state=100)
for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
    
    X_train, X_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
        
    m.fit(X_train, y_train)
    probs = m.predict_proba(X_test)[:, 1]
    predictions[test_idx] = probs


# ### Results

# We'll output the ROC-AUC metric for our classifier as an estimate how much covariate shift this data has. As we can see that value for AUC is very close to .5. It implies that our classifier is not able to distinguish the rows whether it is belonging to train or test. This implies that majority of the observations comes from a feature space which is not particular to test or train.

# In[25]:


print('ROC-AUC for X and Z distributions:', AUC(y, predictions))


# In the *predictions* array, we just computed the probability of a sample in the full dataset being sample taken from the training distribution ($train$). We'll call this $p(train|Data)$. Next we'll use the relationship that $p(train|Data) = 1 - p(test|Data)$ to estimate $\beta$ for our training samples,
# 
# $\beta_i = \frac{p_i(test|Data)}{p_i(train|Data)} = \frac{1 - p_i(train|Data)}{p_i(train|Data)} = \frac{1}{p(train|Data)} - 1$.
# 
# So we now have a method to convert the probability of each point belonging to the training distribution into our sample weights $\beta$. Let's see the distribution of these weights for the training samples

# In[27]:


plt.figure(figsize=(20,10))
predictions_train = predictions[len(tst):] #filtering the actual training rows
weights = (1./predictions_train) - 1. 
weights /= np.mean(weights) # Normalizing the weights
plt.xlabel('Computed sample weight')
plt.ylabel('# Samples')
sns.distplot(weights, kde=False)


# * A lot of the training samples have a weight equal to 1 and is in line with the AUC value
# * Almost 70% of training samples have sample weight of close to 1 and hence comes from a feature space which is not very specific to train or test high density region
# * This implies that train and test dataset are not very different
# 

# ## Application

# * When we are sampling from our training data to create a validation set, we can compare that validation set to the test data for similarity or dissimilarity. This can be a good check if we are evaluating different models on our validation
# 
# * The weights calculated can be used as sample_weights for any classifier to weigh in those observations which are closer to the test data
