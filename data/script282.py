
# coding: utf-8

# # Safe Driver Prediction Explotory Data Analysis
# 
# `Kueip- Sept 2017`
# 
# ---
# ## Outline:
# -   ** Intoduction** ([completed]())
#     - Packages Loading
#     - Check Memory Usage
# -  ** Multii-Variables Analysis**  ([non-complete]())
# -  ** Bi-Variables Analysis**  ([non-complete]())
#         - Feature Values Distribution
# -  ** Target Value Analysis** ([completed]())
# -  ** Missing Values Analysis** ([completed]())
#     - Matrix
#     - HeatMap
#     - Bar
# 
# -  ** Feature Important** ([non-complete]())
#         - Decision Tree
#         - RandomForest
#         - XGB
#         - LGB

# # Introduction
# Porto Seguro, one of Brazil’s largest auto and homeowner insurance companies, completely agrees. Inaccuracies in car insurance company’s claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones.

# ### Packages Loading

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # visualization
from subprocess import check_output
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier 

import xgboost as xgb # Gradeint Boosting
from xgboost import XGBClassifier # Gradeint Boosting
import lightgbm as lgb # Gradeint Boosting
import gc
import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ", train.shape)
print("Test shape : ", test.shape)


# - No. of rows are large with 58 columns. 
# 
# From VC dimension theory, we dont worry about overfitting too much, if we could cover the function set, choose the proper number of features.
# 

# In[ ]:


train.head()


# ### Check Memory Usage

# In[ ]:


train.info(verbose=False),test.info(verbose=False)


# ### Convert Type

# In[ ]:


for c, dtype in zip(train.columns, train.dtypes):
    if dtype == np.float64:
        train[c] = train[c].astype(np.float32) 
    elif dtype == np.int64:
        train[c] = train[c].astype(np.int32) 
gc.collect()
for c, dtype in zip(test.columns, test.dtypes):
    if dtype == np.float64:
        test[c] = test[c].astype(np.float32) 
    elif dtype == np.int64:
        test[c] = test[c].astype(np.int32) 


# ## Multi-Variable Analysis

# In[ ]:


from collections import Counter
count = Counter()
unique_values_dict = {}
for col in train.columns:
    unique_values_dict[col] = np.sort(train[col].unique())
    count[col] = len(np.sort(train[col].unique()))   


# In[ ]:


cat_cols = [ col for col , val in count.items() if(val==10)]
plt.figure(figsize=(20,70))
for i in range(len(cat_cols)):
    c = cat_cols[i]
    
    means = train.groupby(c).target.mean()
    stds = train.groupby(c).target.std()#.fillna(0)
    means_astds = train.groupby(c).target.mean() + train.groupby(c).target.std()
    means_sstds = train.groupby(c).target.mean() - train.groupby(c).target.std()
    
    ddd = pd.concat([means, stds, means_astds, means_sstds], axis=1); 
    ddd.columns = ['means', 'stds', 'means + stds', 'means - stds']
    ddd.sort_values('means', inplace=True)
    
    plt.subplot(len(cat_cols), 2, 2*i+1)
    ax = sns.countplot(train[c], order=ddd.index.values)
    plt.xticks(rotation=90)
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.0f}'.format(y), (x.mean(), y), ha='center', va='bottom')
    
    plt.subplot(len(cat_cols ), 2, 2*i+2)
    plt.fill_between(range(len(train[c].unique())), 
                     ddd.means.values - ddd.stds.values,
                     ddd.means.values + ddd.stds.values,
                     alpha=0.3
                    )
    plt.xticks(range(len(train[c].unique())), ddd.index.values, rotation=90,fontsize=18)
    plt.plot(ddd.means.values, color='b', marker='.', linestyle='dashed', linewidth=0.7)
    plt.plot(ddd['means + stds'].values, color='g', linestyle='dashed', linewidth=0.7)
    plt.plot(ddd['means - stds'].values, color='r', linestyle='dashed', linewidth=0.7)
    plt.xlabel(c + ': Means, STDs and +- STDs',fontsize=18)
    #plt.ylim(80, 270)
plt.show()


# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

a=[column for column in train]
trace = go.Heatmap(z=train.corr().values,
                   x=a,
                   y=a)
data=[trace]
py.iplot(data, filename='backorders heatmap')


# ## Binary Variables:
# 

# In[ ]:


cat_cols = [ col for col , val in count.items() if(val==2)]
plt.figure(figsize=(25,100))
for i in range(len(cat_cols)):
    c = cat_cols[i]
    
    means = train.groupby(c).target.mean()
    stds = train.groupby(c).target.std()#.fillna(0)
    means_astds = train.groupby(c).target.mean() + train.groupby(c).target.std()
    means_sstds = train.groupby(c).target.mean() - train.groupby(c).target.std()
    
    ddd = pd.concat([means, stds, means_astds, means_sstds], axis=1); 
    ddd.columns = ['means', 'stds', 'means + stds', 'means - stds']
    ddd.sort_values('means', inplace=True)
    
    plt.subplot(len(cat_cols), 2, 2*i+1)
    ax = sns.countplot(train[c], order=ddd.index.values)
    plt.xticks(rotation=90)
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.0f}'.format(y), (x.mean(), y), ha='center', va='bottom')
    
    plt.subplot(len(cat_cols ), 2, 2*i+2)
    plt.fill_between(range(len(train[c].unique())), 
                     ddd.means.values - ddd.stds.values,
                     ddd.means.values + ddd.stds.values,
                     alpha=0.3
                    )
    plt.xticks(range(len(train[c].unique())), ddd.index.values, rotation=90,fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(ddd.means.values, color='b', marker='.', linestyle='dashed', linewidth=0.7)
    plt.plot(ddd['means + stds'].values, color='g', linestyle='dashed', linewidth=0.7)
    plt.plot(ddd['means - stds'].values, color='r', linestyle='dashed', linewidth=0.7)
    plt.xlabel(c + ': Means, STDs and +- STDs',fontsize=16)
    #plt.ylim(80, 270)
plt.show()


# - **ps_ind_10_bin**
# - **ps_ind_11_bin**
# - **ps_ind_12_bin**
# - **ps_ind_13_bin** 
# 
# have obvious imbalanced distribution.

# ## Target Variable Analysis

# In[ ]:


labels = '1', '0'
sizes = [train[train.target==1].shape[0],train[train.target==0].shape[0]]
colors = ['gold', 'lightskyblue']
explode = (0.1, 0)  # explode 1st slice
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.2f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# - imblalanced data
# - Scikit-Learn provide [StratifiedShuffleSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit) API by preserving the percentage of samples for each class.

# ### A Review on Imbalanced Learning Methods
# 
# Imbalanced classification is a supervised learning problem where one class outnumbers other class by a large proportion. This problem is faced more frequently in binary classification problems than multi-level classification problems. The reasons which leads to reduction in accuracy of ML algorithms on imbalanced data sets:
#     1. ML algorithms struggle with accuracy because of the unequal distribution in dependent variable.
#     2. This causes the performance of existing classifiers to get biased towards majority class.
#     3. The algorithms are accuracy driven i.e. they aim to minimize the overall error to which the minority class contributes very little.
#     4. ML algorithms assume that the data set has balanced class distributions.
#     5. They also assume that errors obtained from different classes have same cost

# ### How to use imbalanced data to cheat your boss ? Let's conduct an experiment!

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
X = train.drop(['id','target'], axis=1).values
y = train.target.values
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    break
    
from sklearn.dummy import DummyClassifier
# Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
# Therefore the dummy 'most_frequent' classifier always predicts class 0
dummy_majority.score(X_test, y_test)


# 
# Hey Boss, I design a bullshit classifier with accuracy 96.35%.
# 
# .
# 
# .
# 
# 
# Now, you should know why **Normalized Gini** is the metric in this case, instead of **accuracy**. 
# (If  we just used a majority class to assign values to all records, we will still be having a high accuracy.)
# 
# One Specific example, if this bullshiter recognize a terrorist isnt a terroist, it will become a disaster.

# ## Missing Values Analysis
# - Thanks **Pedro Schoen** for pointing missing are encoded as **-1**
# - Let's encode **-1** as `np.nan`

# In[ ]:


for col in np.intersect1d(train.columns,test.columns):
    train.loc[train[col]==-1,col] = np.nan
    test.loc[test[col]==-1,col] = np.nan


# ## Train Set Missing Values

# In[ ]:


missing_df = train.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['ratio'] = round(missing_df['missing_count'] / train.shape[0],4)
missing_df[missing_df['ratio']>0][['column_name', 'missing_count','ratio']].sort_values(by='ratio',ascending=False)


# In[ ]:


def missingno_matrix(df):
    missingValueColumns = df.columns[df.isnull().any()].tolist()
    msno.matrix(df[missingValueColumns],width_ratios=(10,1),            figsize=(20,8),color=(0,0, 0),fontsize=12,sparkline=True,labels=True)
    plt.show()
missingno_matrix(train)


# In[ ]:


def missingno_heatmap(df):
    missingValueColumns = df.columns[df.isnull().any()].tolist()
    msno.heatmap(df[missingValueColumns],figsize=(20,20))
    plt.show()
missingno_heatmap(train)


# In[ ]:


def missing_bar(df):

    missingValueColumns = df.columns[df.isnull().any()].tolist()
    msno.bar(df[missingValueColumns],figsize=(20,8),color="#34495e",fontsize=12,labels=True)
    plt.show()
missing_bar(train)


# ## Test Set Missing Values

# In[ ]:


missing_df = test.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['ratio'] = round(missing_df['missing_count'] / test.shape[0],4)
missing_df[missing_df['ratio']>0][['column_name', 'missing_count','ratio']].sort_values(by='ratio',ascending=False)


# In[ ]:


missingno_matrix(test)


# In[ ]:


missingno_heatmap(test)


# In[ ]:


missing_bar(test)


# - Good News. Both dataset get the same NaN ratio/distribution

# ## Feature Importance

# - **Decision Tree Classifier**
# 
# A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes). The paths from root to leaf represent classification rules.

# In[ ]:


matplotlib.style.use('fivethirtyeight')
matplotlib.rcParams['figure.figsize'] = (12,6)
model = DecisionTreeClassifier(max_depth=6 ,random_state=87)
model.fit(X_train, y_train)
feat_names = train.drop(['id','target'],axis=1).columns
## plot the importances ##
importances = model.feature_importances_

indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by DecisionTreeClassifier")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()


# In[ ]:


from sklearn.tree import export_graphviz
import graphviz
treegraph = export_graphviz(model, out_file=None, 
                         feature_names=train.drop(['id','target'],axis=1).columns,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(treegraph)  
graph


# - **RandomForest Classifier**
# 
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.

# In[ ]:


model = RandomForestClassifier(max_depth=8)
model.fit(X_train, y_train)
feat_names = train.drop(['id','target'],axis=1).columns
## plot the importances ##
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
plt.title("Feature importances by Random Forest")
plt.bar(range(len(indices)), importances[indices], color='lightblue', yerr=std[indices], align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()


# - **XGB Classifier**
# 
# XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework

# In[ ]:


model = XGBClassifier(eta = 0.01, max_depth = 8, subsample = 0.8, colsample_bytree= 0.8)
model.fit(X_train, y_train)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by XGB") # Thanks Oscar Takeshita's kindly remind
plt.bar(range(len(indices)), importances[indices], color='lightblue', align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()


# - Create Tree digraph by using 
# 
# `xgb.to_graphviz`

# In[ ]:


xgb.to_graphviz(model, fmap='', rankdir='UT', num_trees=6,
                yes_color='#0000FF', no_color='#FF0000')


# - **LightGBM Classifier**
# 
# LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages

# In[ ]:


lgb_params = {}
lgb_params['objective'] = 'binary'
lgb_params['sub_feature'] = 0.80 
lgb_params['max_depth'] = 7
lgb_params['feature_fraction'] = 0.7
lgb_params['bagging_fraction'] = 0.7
lgb_params['bagging_freq'] = 10
lgb_params['learning_rate'] = 0.01

lgb_train = lgb.Dataset(X_train, y_train)
lightgbm = lgb.train(lgb_params, lgb_train, feature_name=[ i for i in feat_names])


# In[ ]:


plt.figure(figsize=(12,6))
lgb.plot_importance(lightgbm,max_num_features=30)
plt.title("Feature importances by LightGBM")
plt.show()


# In[ ]:


ax = lgb.plot_tree(lightgbm, tree_index=83, figsize=(20, 8), show_info=['split_gain'])
plt.show()


# # Acknowledgement:
# 1. Pedro Schoen
# 
# 
# ## Stay Tuned
# 
