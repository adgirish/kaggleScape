
# coding: utf-8

# Hello,  
# This is my first post on Kaggle. I am participating in this challenge as part of a course project on optimisation at Ohio State University. This kernel gets a 0.283 on LB. Any suggestions on how to improve further would be much appreciated. (Note: The public kernels claiming 0.284 did not give me any improvement over the score achieved using this kernel)
# 
# TL;DR
# Basic idea - 3 significantly different LightGBM trees - 4x upsampling - bayesian encoding of categorical features - dropping calculated features - 10 fold CV  
# Jump to last section to directly view the kernel
# 
# This code is based on [Oliver's kernel](https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283/code) and many public contributors. Thanks to them

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000 # Jupyter notebook backend restricts number of points in plot
import pandas as pd
import scipy as scp
import csv
import seaborn as sns

train_master = pd.read_csv('../input/train.csv')
test_master = pd.read_csv('../input/test.csv')
train_master.describe()


# # Visual Data Exploration
# 
# There are 3 types of variables - Binary, Categorical and Continuous. Lets start with target variable by visualizing their distribution.

# In[ ]:


binary_columns = [s for s in list(train_master.columns.values) if '_bin' in s]
categorical_columns = [s for s in list(train_master.columns.values) if '_cat' in s]
non_continuous_feature_subs = ['_cat', '_bin', 'target', 'id']
continuous_columns = [s for s in list(train_master.columns.values) 
                      if all(x not in s for x in non_continuous_feature_subs)]
target_column = 'target'

ind_columns = [s for s in list(train_master.columns.values) if '_ind' in s]
car_columns = [s for s in list(train_master.columns.values) if '_car' in s]
calc_columns = [s for s in list(train_master.columns.values) if '_calc' in s]
reg_columns = [s for s in list(train_master.columns.values) if '_reg' in s]


# ## Target Variable
# 
# Lets check the distribution of target classes

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

init_notebook_mode()

labels = ['1','0']
values = [(train_master[target_column]==1).sum(),(train_master[target_column]==0).sum()]
colors = ['#FEBFB3', '#E1396C']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

iplot([trace])


# We see that the target is unevenly distributed such that the number of insurance claims are << non-claims. To overcome this problem, we will either upsample the data by duplicating rows with positive target values or downsample data by deleting rows with negative target values. Do note that upsampling has to be done DURING cross validation instead of BEFORE. Check out the upsampling section in ensemble and CV notebook to see why. Downsampling leads to loss of information and therefore we will use the upsampling technique.

# # Binary Features
# 
# Lets check the distribution of 1s and 0s in binary features

# In[ ]:


zero_list = []
one_list = []
for col in binary_columns:
    zero_list.append((train_master[col]==0).sum())
    one_list.append((train_master[col]==1).sum())

trace1 = go.Bar(
    x=binary_columns,
    y=zero_list ,
    name='0s count'
)
trace2 = go.Bar(
    x=binary_columns,
    y=one_list,
    name='1s count'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title='Count of 1s and 0s in binary variables'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')


# We see that variables ```ps_ind_10_bin```, ```ps_ind_11_bin```, ```ps_ind_12_bin``` and ```ps_ind_13_bin``` have almost all 0s and therefore may not be of much use in prediction. A feature selection step in pipeline will determine whether to include them or not. It is important to note that feature selection should be performed DURING cross validation and not before. In short, feature selection should not use the data from validation set in CV. For more information check out the upsampling section in ensemble and CV notebook.
# 
# Still, to get a feel for data, lets check for similarity between features. To check similarity between 2 binary variables, we will XOR each's row element and count the percentage of 0s and 1s

# In[ ]:


binary_corr_data = []
r = 0
for i in binary_columns:
    binary_corr_data.append([])
    for j in binary_columns:
        s = sum(train_master[i]^train_master[j])/float(len(train_master[i]))
        binary_corr_data[r].append(s)
    r+=1


# In[ ]:


trace = go.Heatmap(z=binary_corr_data, x=binary_columns, y=binary_columns, colorscale='Greys')
data=[trace]
iplot(data)


# The heatmap gives us some insights into the most important variables. For example, lightly colored columns are most uncorrelated fromall other variables. These are features like - ```ps_ind_06_bin```, ```ps_ind_16_bin```, ```ps_calc_16_bin```, ```ps_calc_17_bin```, ```ps_calc_19_bin```
# 
# In the same way, lets check similarity of each feature with the target variable to visualise each features prediction power.

# In[ ]:


binary_target_corr_data = []
for i in binary_columns:
    s = sum(train_master[i]^train_master[target_column])/float(len(train_master[i]))
    binary_target_corr_data.append(s)


# In[ ]:


binary_target_corr_chart = [go.Bar(
    x=binary_columns,
    y=binary_target_corr_data
)]
iplot(binary_target_corr_chart)


# We again observe that features -  ```ps_ind_06_bin```, ```ps_ind_16_bin```, ```ps_calc_16_bin```, ```ps_calc_17_bin```, ```ps_calc_19_bin``` are most insightful
# 
# We will perform a more formal feature selection during the cross validation stage.

# # Continuous Features
# 
# First lets check if there are missing values in the data set (For binary variables it was specified that it had no missing values).

# In[ ]:


value_list = []
missing_list = []
for col in continuous_columns:
    value_list.append((train_master[col]!=-1).sum())
    missing_list.append((train_master[col]==-1).sum())

trace1 = go.Bar(
    x=continuous_columns,
    y=value_list ,
    name='Actual Values'
)
trace2 = go.Bar(
    x=continuous_columns,
    y=missing_list,
    name='Missing Values'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    title='Count of missing values in continuous variables'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')


# We see that only ```ps_reg_03``` and ```ps_car_14``` have significant number of missing values. Apart from that, ```ps_car_11``` has 5 and ```ps_car_12``` has 1 missing value. Lets evaluate the chi squared test between each of continuous variables and the target variable

# In[ ]:


from sklearn.feature_selection import chi2, mutual_info_classif

minfo_target_to_continuous_features = mutual_info_classif(
    train_master[continuous_columns],train_master[target_column])

minfo_target_to_continuous_chart = [go.Bar(
    x=continuous_columns,
    y=minfo_target_to_continuous_features
)]
iplot(minfo_target_to_continuous_chart)


# It seems like ```ps_reg_03``` and ```ps_car_14``` are fairly independant of the target variable. Again, a more formal feature selection will be performed during cross validation
# 
# Lets evaluate the pearson correlation between between each of continuous variables to see if two features are highly correlated and therefore present redundant information.

# In[ ]:


continuous_corr_data = train_master[continuous_columns].corr(method='pearson').as_matrix()

trace = go.Heatmap(z=continuous_corr_data, x=continuous_columns, 
                   y=continuous_columns, colorscale='Greys')
data=[trace]
iplot(data)


# It seems like ```ps_reg_03``` and ```ps_reg_01``` maybe linearly related. Same for ```ps_car_12``` and ```ps_car_13```

# # Categorical Features
# 
# Categorical features are a tricky business in classification, especially while using trees. Since tree algorithms use binary trees, they need to find an appropriate split. But categorical variables have no inherent order in them. Various techniques are used to overcome this problem. One such technique is encoding of categorical variables such that the encoded variables have an order. But what encoding scheme to follow?
# 
# A 2001 [paper](https://www.researchgate.net/publication/220520258_A_Preprocessing_Scheme_for_High-Cardinality_Categorical_Attributes_in_Classification_and_Prediction_Problems) by Daniele Micci-Barreca illustrates one approach.

# # Evaluation Criteria for Predictions
# 
# According to competition details, the challenge uses normalized gini coefficient to evaluate the predictions. This is the same function we should be using for our cross validation step (Gini is linearly related to AUC-ROC so we will use AUC for evaluation). Lets define the function.

# In[ ]:


def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g


# # Cross Validation
# 
# To evaluate performance of our model and to ensure that our it generalizes well over new data, we do a 10 fold cross-validation. Folds will be stratified to ensure equal proportions of target variable in each. In each fold 
# 1. We upsample the positive target data from training fold
# 2. Choose best features using training fold
# 3. Train the model over training fold and evaluate it over the hold out validation fold.

# In[ ]:


from sklearn.model_selection import StratifiedKFold

n_splits = 10
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)


# # Single Model
# 
# We will start with XgBoost. To evaluate optimal parameters for XgBoost would require a search like grid search. Parameter tuning is needed to avoid overfitting and there are two basic ways to do that -   
# 1. By controlling complexity of tree (Regularization)
# 2. By adding randomness via sub-sampling data and columns
# 
# Suppose we want to find the optimal values from - 
# 
# ```max_depth``` = {3, 4, 5} (Regularization)  
# ```gamma``` = {1, 5, 9} (Regularization)  
# ```colsample_bytree``` = {0.7, 0.8, 0.9}  
# ```subsample``` = {0.7, 0.8, 0.9}  
# ```learning_rate``` = {0.1, 0.05, 0.005}
# 
# A grid search over these parameters means evaluating model at 243 parameter combinations. And with 10-fold CV, it means that you train the model 2430 times. Training 1 model on one core of an c4.4xlarge EC2 instance takes anywhere between 10 to 15 minutes depending on the depth of the tree and learning rates. With 16 cores running parallely will take approximately 38 hours. c4.4xlarge charges ~ \$0.8/hr which leads to a total cost of $30 and two days of time. And this is just to evaluate XgBoost over a small set of parameter range.
# 
# If you want to do it yourself, I have listed the code in appendix notebook with details of setting up an EC2 instance with jupyter notebook interface.
# 
# Combining a short grid search costing me $10 and prior experience, I decided to use the following parameters - 
# 
# ```max_depth``` = 4  
# ```gamma``` = 9  
# ```colsample_bytree``` = 0.8  
# ```subsample``` = 0.8  
# ```learning_rate``` = 0.05
# 
# The objective function is (gives the probabilities)  
# ```objective``` = ```"binary:logistic"```
# 
# The other option which gave equally good performance was evaluation of pair-wise ranks  
# ```objective``` = ```"rank:pairwise"```
# 
# Parameters for cross validation -   
# ```num_rounds``` = 1000 with early stopping window of 10 epochs (And use all trees to predict)  
# ```folds``` = 10
# 
# (For brevity, I have excluded the code but the it gave a LB of 0.281)

# # Average of 3 boostes trees with LightGBM
# 
# (Note: LightGBM proved to be extremely fast compared to XGBoost)
# 
# Trees have this property that when we change the training dataset, we may end up with drastically different trees. Averaging over all such different trees should better generalise our CV score over the test data. Look at it as a forest of gradient boosted trees.
# 
# Each tree is differentiated by its complexity and randomness. We will generate 3 trees as follows -
# 1. A deeper tree (```max_depth``` = 5) that randomly chooses only a small (30%) subset of the available features and 0.7 bagging fraction
# 2. An average tree (```max_depth``` = 4) that randomly chooses 90% of features and 0.9 bagging fraction
# 2. An shallow tree (```max_depth``` = 3) that chooses all features and all rows
# 
# At each fold, we evaluate the arithmatic mean of predictions of the 3 trees on the validation set. The overall score over training data is then taken as the average score over all folds.
# 
# This model leads to 0.287 CV and 0.283 on LB (Seems like a drastic overfit!)
# 
# # Parameter Selection
# 
# The main parameter to control complexity of tree will be ```max_depth``` instead of ```num_leaves```. Since LightGBM grows tree leaf wise, for same number of leaves, LightGBM will give a much deeper tree than depth wise. ```max_depth``` is a much more intuitive limit on how deep the tree is going to grow.
# 
# We will use ```feature_fraction``` and ```bagging_fraction``` and ```bagging_freq``` to control randomness.
# 
# We will again experiment with LightGBM's internal handling of categorical features and our encoding scheme defined previously. For this model, we use our previous encoding scheme.
# 
# Lets first define the encoding scheme.
# 

# In[ ]:


import numpy as np
from sklearn import metrics

def encode_cat_features(train_df, test_df, cat_cols, target_col_name, smoothing=1):
    prior = train_df[target_col_name].mean()
    probs_dict = {}
    for c in cat_cols:
        probs = train_df.groupby(c, as_index=False)[target_col_name].mean()
        probs['counts'] = train_df.groupby(c, as_index=False)[target_col_name].count()[[target_col_name]]
        probs['smoothing'] = 1 / (1 + np.exp(-(probs['counts'] - 1) / smoothing))
        probs['enc'] = prior * (1 - probs['smoothing']) + probs['target'] * probs['smoothing']
        probs_dict[c] = probs[[c,'enc']]
    return probs_dict


# Lets start building the model. Uncomment the code (commented because it exceeds time limit)

# In[ ]:


'''
import lightgbm as lgb
import numpy as np
import pandas as pd

np.random.seed(3)
model_scores = {}

# Drop binary columns with almost all zeros. 
# Why now? Just follow along for now. We have a lot of experimentation to be done
train = train_master.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin'],axis=1)
test = test_master.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin'],axis=1)

# Drop calculated features
# But WHY??? 
# Because we are assuming that tree can generate any complicated function 
# of base features and calculated features add no more information
# Is this assumption valid? Results will tell
calc_columns = [s for s in list(train_master.columns.values) if '_calc' in s]
train = train.drop(calc_columns, axis=1)  
test = test.drop(calc_columns, axis=1)

# Get categorical columns for encoding later
categorical_columns = [s for s in list(train_master.columns.values) if '_cat' in s]
target_column = 'target'

# Replace missing values with NaN
train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

# Initialize DS to store validation fold predictions
y_val_fold = np.empty(len(train))

# Initialize DS to store test predictions with aggregate model and individual models
y_test = np.zeros(len(test))
y_test_model_1 = np.zeros(len(test))
y_test_model_2 = np.zeros(len(test))
y_test_model_3 = np.zeros(len(test))

for fold_number, (train_ids, val_ids) in enumerate(
    folds.split(train.drop(['id',target_column], axis=1), 
                train[target_column])):
    
    X = train.iloc[train_ids]
    X_val = train.iloc[val_ids]
    X_test = test
    
    # Encode categorical variables using training fold
    encoding_dict = encode_cat_features(X, X_val, categorical_columns, target_column)
    
    for c, encoding in encoding_dict.items():
        X = pd.merge(X, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))
        X = X.drop(c, axis = 1)
        X = X.rename(columns = {'enc':'enc_'+c})
        
        X_test = pd.merge(X_test, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))
        X_test = X_test.drop(c, axis = 1)
        X_test = X_test.rename(columns = {'enc':'enc_'+c})
        
        X_val = pd.merge(X_val, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))
        X_val = X_val.drop(c, axis = 1)
        X_val = X_val.rename(columns = {'enc':'enc_'+c})
        
    # Seperate target column and remove id column from all
    y = X[target_column]
    X = X.drop(['id',target_column], axis=1)
    X_test = X_test.drop('id', axis=1)
    y_val = X_val[target_column]
    X_val = X_val.drop(['id',target_column], axis=1)
    
    # Upsample data in training folds
    ids_to_duplicate = pd.Series(y == 1)
    X = pd.concat([X, X.loc[ids_to_duplicate]], axis=0)
    y = pd.concat([y, y.loc[ids_to_duplicate]], axis=0)
    # Again Upsample (total increase becomes 4 times)
    X = pd.concat([X, X.loc[ids_to_duplicate]], axis=0)
    y = pd.concat([y, y.loc[ids_to_duplicate]], axis=0)
    
    # Shuffle after concatenating duplicate rows
    # We cannot use inbuilt shuffles since both dataframes have to be shuffled in sync
    shuffled_ids = np.arange(len(X))
    np.random.shuffle(shuffled_ids)
    X = X.iloc[shuffled_ids]
    y = y.iloc[shuffled_ids]
    
    # Feature Selection goes here
    # TODO
    
    # Define parameters of GBM as explained before for 3 trees
    params_1 = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 3,
        'learning_rate': 0.05,
        'feature_fraction': 1,
        'bagging_fraction': 1,
        'bagging_freq': 10,
        'verbose': 0
    }
    params_2 = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,
        'verbose': 0
    }
    params_3 = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 5,
        'learning_rate': 0.05,
        'feature_fraction': 0.3,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': 0
    }
    
    # Create appropriate format for training and evaluation data
    lgb_train = lgb.Dataset(X, y)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    # Create the 3 classifiers with 1000 rounds and a window of 100 for early stopping
    clf_1 = lgb.train(params_1,lgb_train, num_boost_round=1000,
                      valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=50)
    clf_2 = lgb.train(params_2,lgb_train, num_boost_round=1000,
                      valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=50)
    clf_3 = lgb.train(params_3,lgb_train, num_boost_round=1000,
                      valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=50)
    
    # Predict raw scores for validation ids
    # At each fold, 1/10th of the training data get scores
    y_val_fold[val_ids] = (clf_1.predict(X_val, raw_score=True)+
                           clf_2.predict(X_val, raw_score=True)+
                           clf_3.predict(X_val, raw_score=True)) / 3

    # Predict and average over folds, raw scores for test data
    y_test += (clf_1.predict(X_test, raw_score=True)+
               clf_2.predict(X_test, raw_score=True)+
               clf_3.predict(X_test, raw_score=True)) / (3*n_splits)
    y_test_model_1 += clf_1.predict(X_test, raw_score=True) / n_splits
    y_test_model_2 += clf_2.predict(X_test, raw_score=True) / n_splits
    y_test_model_3 += clf_3.predict(X_test, raw_score=True) / n_splits
    
    # Display fold predictions
    # Gini requires only order and therefore raw scores need not be scaled
    print("Fold %2d : %.9f" % (fold_number + 1, gini(y_val, y_val_fold[val_ids])))
    
# Display aggregate predictions
# Gini requires only order and therefore raw scores need not be scaled
print("Average score over all folds: %.9f" % gini(train_master[target_column], y_val_fold))
'''


# Scale scores and save predictions for submission

# In[ ]:


'''
temp = y_test
# Scale the raw scores to range [0.0, 1.0]
temp = np.add(temp,abs(min(temp)))/max(np.add(temp,abs(min(temp))))

df = pd.DataFrame(columns=['id', 'target'])
df['id']=test_master['id']
df['target']=temp
df.to_csv('benchmark__0_283.csv', index=False, float_format="%.9f")
df.shape
'''


# I have purposefully kept aside all extra analysis and messy experiments. I have tried most of the tricks mentioned in the discussions - using HM instead of AM or Logistic regression for ensembling, recursive feature selection, custom features like sum of binary variables, even downgrading LightGBM version, etc, but none has helped me improve the score. I guess I just have to experiment adding more classifiers and averaging them appropriately.
# 
# Some current known drawbacks - 
# 1. Encoding is based on training set and there are some categories in hold out that do not occur in training. This results in NaN values on the join
# 2. Literally no feature engineering
# 3. Overfitting
# 
# Please do let me know your opinions and suggestions and thanks to public contributors for the invaluable help!
# 
# P.S. I know this is off topic but I will be shameless and ask - I am looking for internships for summer 2018. Any help in that direction will also be appreciated :)
