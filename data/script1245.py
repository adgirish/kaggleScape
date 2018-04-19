
# coding: utf-8

# 
# Based on work done on this notebook, https://www.kaggle.com/sudosudoohio/stratified-kfold-xgboost-eda-tutorial-0-281.
# Filled in missing values with Median and  the public score increased from 0.281 to 0.285 after training with new data. 
# Also, custom weights are given to the labels to tackle imbalanced labels in the data. However, performance hasn't improved.
# 
# Hope you will find the notebook helpful! .
# 

# In[ ]:


## Importing Requried Libraries
import numpy as np
import subprocess
import pandas as pd

from IPython.display import Image

from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, accuracy_score

# classifiers
from sklearn.ensemble import GradientBoostingClassifier

# reproducibility
seed = 123


# In[ ]:


#### LOADING DATA ####
### TRAIN DATA
train_data = pd.read_csv("../input/train.csv", na_values='-1')
cor_data = train_data.copy()
                        
## Filling the missing data NAN with median of the column
train_data_nato_median = pd.DataFrame()
for column in train_data.columns:
    train_data_nato_median[column] = train_data[column].fillna(train_data[column].median())

train_data = train_data_nato_median.copy()

### TEST DATA
test_data = pd.read_csv("../input/test.csv", na_values='-1')
## Filling the missing data NAN with mean of the column
test_data_nato_median = pd.DataFrame()
for column in test_data.columns:
    test_data_nato_median[column] = test_data[column].fillna(test_data[column].median())
    
test_data = test_data_nato_median.copy()
test_data_id = test_data.pop('id')


# In[ ]:


## Identifying Categorical data
column_names = train_data.columns
categorical_column = column_names[column_names.str[10] == 'c']

## Changing categorical columns to category data type
def int_to_categorical(data):
    """ 
    changing columns to catgorical data type
    """
    for column in categorical_column:
        data[column] =  data[column].astype('category')


# In[ ]:


## Creating list of train and test data and converting columns of interest to categorical type
datas = [train_data,test_data]

for data in datas:
    int_to_categorical(data)

#print(test_data.dtypes)


# In[ ]:


## Decribing categorical variables
# def decribe_Categorical(x):
#     """ 
#     Function to decribe Categorical data
#     """
#     from IPython.display import display, HTML
#     display(HTML(x[x.columns[x.dtypes =="category"]].describe().to_html))

# decribe_Categorical(train_data)


# In[ ]:


### FUNCTION TO CREATE DUMMIES COLUMNS FOR CATEGORICAL VARIABLES
def creating_dummies(data):
    """creating dummies columns categorical varibles
    """
    for column in categorical_column:
        dummies = pd.get_dummies(data[column],prefix=column)
        data = pd.concat([data,dummies],axis =1)
        ## dropping the original columns ##
        data.drop([column],axis=1,inplace= True)


# In[ ]:


### CREATING DUMMIES FOR CATEGORICAL VARIABLES  
for column in categorical_column:
        dummies = pd.get_dummies(train_data[column],prefix=column)
        train_data = pd.concat([train_data,dummies],axis =1)
        train_data.drop([column],axis=1,inplace= True)


for column in categorical_column:
        dummies = pd.get_dummies(test_data[column],prefix=column)
        test_data = pd.concat([test_data,dummies],axis =1)
        test_data.drop([column],axis=1,inplace= True)

print(train_data.shape)
print(test_data.shape)


# In[ ]:


#Define covariates in X and dependent variable in y
features = train_data.iloc[:,2:] ## FEATURE DATA
targets= train_data.target ### LABEL DATA

### CHECKING DIMENSIONS
print(features.shape)
print(targets.shape)


# In[ ]:


ax = sns.countplot(x = targets ,palette="Set2")
sns.set(font_scale=1.5)
ax.set_xlabel(' ')
ax.set_ylabel(' ')
fig = plt.gcf()
fig.set_size_inches(10,5)
ax.set_ylim(top=700000)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))

plt.title('Distribution of 595212 Targets')
plt.xlabel('Initiation of Auto Insurance Claim Next Year')
plt.ylabel('Frequency [%]')
plt.show()


# In[ ]:


sns.set(style="white")


# Compute the correlation matrix
corr = cor_data.corr()


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# In[ ]:


# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score


# In[ ]:


unwanted = features.columns[features.columns.str.startswith('ps_calc_')]
print(unwanted)


# In[ ]:


train = features.drop(unwanted, axis=1)  
test = test_data.drop(unwanted, axis=1)  

print(train.shape)
print(test.shape)


# In[ ]:


## Spliting train data 
kfold = 2 ## I used 5 Kfolds. In interest of computational time using 2
skf = StratifiedKFold(n_splits=kfold, random_state=42)


# In[ ]:


## Specifiying parameters
params = {
    'min_child_weight': 10.0,
    'objective': 'binary:logistic',
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round' : 700
    }


# In[ ]:


# Converting pandas series to numpy array to be fed to XGBoost package
X= train.values
y = targets.values

type(X),type(y)


# In[ ]:


# Submission data frame
sub = pd.DataFrame()
sub['id'] = test_data_id
sub['target'] = np.zeros_like(test_data_id)
sub.shape


# We are providing custom weights to the labels in the data. Here I used two ways to do it,
# 1. Manually assigning weights to the label as by numpy array 'weights' in the code
# 2. using 'scale_pos_weights' parameter

# In[ ]:


## Model fitting
# THis is where the magic happens

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    ### Custom weights: To deal with imbalanced label
    #weights = np.zeros(len(y_train))
    #weights[y_train == 0] = 1
    #weights[y_train == 1] = 9            
    #d_train = xgb.DMatrix(X_train, label = y_train, weight = weights)
    
    
    d_train = xgb.DMatrix(X_train, label = y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(test.values)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    ### USing Scale_pos_weights parameter
    ## In this case we are splitting giving weights to the label according 
    # to their relative ratio's in the data
    
   # train_labels = d_train.get_label()
   # ratio = float(np.sum(train_labels == 0))/ np.sum(train_labels == 1)
   # params['scale_pos_weight'] = ratio

    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
    # and the custom metric (maximize=True tells xgb that higher metric is better)
    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, 
                    feval=gini_xgb, maximize=True, verbose_eval=100)

    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    # Predict on our test data
    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
    sub['target'] += p_test/kfold


# *OUTPUT*
# Adding each layer of topic of the other:
# 1. Base XGBoost: 0.281
# 2. XGBoost with missing values filled with Median(equal class weights): 0.285
# 3. XGbosot with custom weight: 0.282
# 4. XGboost with scale_pos_weights parameter: 0.280

# In[ ]:


#sub.to_csv("ModifiedXGBOOST.csv",index =  False)


# **FUTURE WORK:**
# 1. Planning on using Gridsearch 
# 2. Dealing with Imbalanced data such as this,
#     For class imbalance, we have many methods that could deal with the problem. There is no one method that could essentially solve the issue every single time. Here are few methods that we could use to deal with Class imbalance.
# 
# 1. Weighted loss functions: Giving higher weights to minority class(in this case, class 1). For example, Logistic regression model has inbuilt parameter 'class_weights' which can used to assign weights to the class labels as per choice.
# 2. Random Undersampling: As the name suggests, randomly undersample examples from majority class
# 3. NEARMISS-1 : Retain points from majority class whose mean distance to the K nearest points in S is lowest
# 4. NEARMISS-2: Keep points from majority class whose mean distance to the K farthest points in minority class is lowest
# 5. NEARMISS-3: Select K nearest neighbours In majority class for every point in minority class
# 6. Condensed Nearest Neighbour(CNN)
# 7. Edited Nearest Neighbour(ENN)
# 8. Repeated Edited Nearest Neighbour
# 9. TOMEK LINK REMOVAL
# 10. Random Oversampling
# 11. Synthetic Minority Oversampling Technique(SMOTE)
# 12. SMOTE + Tomek Link Removal
# 13. SMOTE + ENN
# 14. EASYENSEMBLE
# 15. BALANCECASCADE and many more
# 
# Reference: 
# 1. https://www.youtube.com/watch?v=-Z1PaqYKC1w
# 2. https://www.youtube.com/watch?v=32tXCJP0HYc&list=PLZnYQQzkMilqTC12LmnN4WpQexB9raKQG&index=13
