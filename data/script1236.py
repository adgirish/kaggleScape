
# coding: utf-8

# # Predicting Allstate insurance claim severity
# ## Exploratory Data Analysis
# This notebook contains the exploratory data analysis performed on the training dataset provided by Allstate to predict insurance claim severity, analysis is performed on the following topics
# 
# 1. Training and testing dataset statistics
# 2. missing values
# 3. distribution of continuous features and transformation if necessary
# 
# Based on the analysis of feature transformation, best possible feature is picked for preprocessing.

# In[ ]:


# loading required modules
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load train and test dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


# printing train dataset information
train.info()


# In[ ]:


# printing test dataset information
test.info()


# In[ ]:


train.columns[train.isnull().sum() > 0 ]


# In[ ]:


# setting pandas env variables to display max rows and columns
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows',1000)


# In[ ]:


# describing statistics of continuous variables
train.describe()


# In[ ]:


# describing statistics of categorical variables
train.describe(include = ['object'])


# In[ ]:


# sepearte the categorical and continous features
cont_columns = []
cat_columns = []

for i in train.columns:
    if train[i].dtype == 'float':
        cont_columns.append(i)
    elif train[i].dtype == 'object':
        cat_columns.append(i)


# In[ ]:


# log transform the label variable
train['loss'] = np.log1p(train['loss'])


# In[ ]:


sns.pairplot(train[cont_columns], vars=['cont1','cont2','cont3','cont4','loss'], kind = 'scatter',diag_kind='kde')


# In[ ]:


sns.pairplot(train[cont_columns], vars=['cont5','cont6','cont7','cont8','loss'], kind = 'scatter',diag_kind='kde')


# In[ ]:


sns.pairplot(train[cont_columns], vars=['cont9','cont10','cont11','cont12','loss'], kind = 'scatter',diag_kind='kde')


# In[ ]:


sns.pairplot(train[cont_columns], vars=['cont13','cont14','loss'], kind = 'scatter',diag_kind='kde')


# In[ ]:


sns.pairplot(train[cont_columns], kind = 'scatter',diag_kind='kde')


# In[ ]:


plt.figure(figsize=(10,10))
plt.hist(train.loss, bins=100)
plt.title("histogram of target variable")
plt.show()


# In[ ]:


# Compute the correlation matrix
corr = train[cont_columns].corr()


# In[ ]:


sns.set(style="white")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


# In[ ]:


cont_columns.remove('loss')


# Let's look at the probability plot of continuous variables

# In[ ]:


import matplotlib.gridspec as gridspec
from scipy import stats

plt.figure(figsize=(15,25))
gs = gridspec.GridSpec(7, 2)
for i, cn in enumerate(train[cont_columns].columns):
    ax = plt.subplot(gs[i])
    stats.probplot(train[cn], dist = stats.norm, plot = ax)
    ax.set_xlabel('')
    ax.set_title('Probplot of feature: cont' + str(i+1))
plt.show()


# Let's look at the skewness and plot them.

# In[ ]:


skewness_list = []
for cn in train[cont_columns].columns:
    skewness_list.append(stats.skew(train[cn]))

plt.figure(figsize=(10,7))
plt.plot(skewness_list, 'bo-')
plt.xlabel("continous features")
plt.ylabel("skewness")
plt.title("plotting skewness of the continous features")
plt.xticks(range(15), range(1,15,1))
plt.plot([(0.25) for i in range(0,14)], 'r--')
plt.text(6, .1, 'threshold = 0.25')
plt.show()


# Except couple of features, all of them have skewness above 0.25, let's look at their distribution and experiment with different transformations.

# In[ ]:


skewed_cont_columns = []
for i, cn in enumerate(cont_columns):
    if skewness_list[i] >= 0.25:
        skewed_cont_columns.append(cn)


# In[ ]:


plt.figure(figsize=(15,25))
gs = gridspec.GridSpec(6, 2)
for i, cn in enumerate(skewed_cont_columns):
    ax = plt.subplot(gs[i])
    sns.distplot(train[cn], bins=50)
    ax.set_xlabel('')
    ax.set_title('hist plot of feature: ' + str(cn))
plt.show()


# Below function comes in handy in plotting the distribution and probability plot side by side and we look at 
# 
#  - original feature
#  - custom transformed feature
#  - boxcox transformed feature
# 
# in some cases custom transformation might be better than boxcox transformation, let's analyze.

# In[ ]:


def examine_transform(original, transformed):
    plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(3,2, width_ratios=(1,2))
    
    ax = plt.subplot(gs[0])
    sns.distplot(original, bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of orignal feature')
    
    ax = plt.subplot(gs[1])
    prob = stats.probplot(original, dist = stats.norm, plot = ax)
    ax.set_xlabel('')
    ax.set_title('Probplot of original feature')
    
    ax = plt.subplot(gs[2])
    sns.distplot(transformed, bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of transformed feature')
    
    ax = plt.subplot(gs[3])
    prob = stats.probplot(transformed, dist = stats.norm, plot = ax)
    ax.set_xlabel('')
    ax.set_title('Probplot of transformed feature')
    
    # apply boxcox transformation
    xt, _ = stats.boxcox(original)
    ax = plt.subplot(gs[4])
    sns.distplot(xt, bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of boxcox transformed feature')
    
    ax = plt.subplot(gs[5])
    prob = stats.probplot(xt, dist = stats.norm, plot = ax)
    ax.set_xlabel('')
    ax.set_title('Probplot of boxcox transformed feature')
    
    
    plt.show()


# In[ ]:


examine_transform(train.cont1, np.power(train.cont1,0.5))


# In[ ]:


examine_transform(train.cont2, np.tan(train.cont2))


# In[ ]:


examine_transform(train.cont4, np.power(train.cont4,0.2))


# In[ ]:


examine_transform(train.cont5, np.log(train.cont5))


# In[ ]:


examine_transform(train.cont6, np.power(train.cont6,0.5))


# In[ ]:


examine_transform(train.cont7, np.log(train.cont7))


# In[ ]:


examine_transform(train.cont8, np.power(train.cont8,0.4))


# In[ ]:


examine_transform(train.cont9, np.power(train.cont9,0.4))


# In[ ]:


examine_transform(train.cont10+1, np.tanh(train.cont10+1))


# In[ ]:


examine_transform(train.cont11, np.power(train.cont11,0.4))


# In[ ]:


examine_transform(train.cont12, np.power(train.cont12,0.4))


# In[ ]:


examine_transform(train.cont13, np.abs(train.cont13 - np.mean(train.cont13)))


# In[ ]:


examine_transform(train.cont14, np.abs(train.cont14 - np.mean(train.cont14)))


# In[ ]:


# here is the pick of the transformation based on the above analysis
feature_transformation = {  'cont1': 'boxcox'
                          , 'cont2': 'np.tan'
                          , 'cont3': 'none'
                          , 'cont4': 'boxcox'
                          , 'cont5': 'boxcox'
                          , 'cont6': 'boxcox'
                          , 'cont7': 'boxcox'
                          , 'cont8': 'boxcox'
                          , 'cont9': 'boxcox'
                          , 'cont10': 'boxcox'
                          , 'cont11': 'boxcox'
                          , 'cont12': 'boxcox'
                          , 'cont13': 'abs_mean_shift'
                          , 'cont14': 'abs_mean_shift'
                         }

