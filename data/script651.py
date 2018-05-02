
# coding: utf-8

# In this edited version of my kernel, I have included some new features and some others are under progress. Some have been influenced from [THIS KERNEL](https://www.kaggle.com/sudosudoohio/stratified-kfold-xgboost-eda-tutorial-0-281).
# 
# # What lies ahead of you?
# 
# *  **Data Exploration**
#     * Analyzing Datatypes
#     * Analyzing Missing Values
#     * Visualizing missing values
#     * Memory Usage Analysis
#     
# *  **Data Analysis** (visualizing each and every type of feature in the data set)
#     * Splitting columns based on types
#     * Binary Features
#     * Categorical Features
#     * Continuous/Ordinal Features
#     * Correlation (**ps_calc** have an outrageous attitude!!!)
# * **Feature Engineering** 
#     * New Binary features
#     * New Continuous/Ordinal features (*in progress*)
# *  **Modeling**
#     * Gradient Boosting
#     * XGBoost

# # Importing Libraries and Loading Data

# In[ ]:


import numpy as np # linear algebra
import seaborn as sns
import missingno as msno
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from subprocess import check_output
from sklearn import *
import xgboost as xgb
from multiprocessing import *
from ggplot import *

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_sample = pd.read_csv('../input/sample_submission.csv')
# Any results you write to the current directory are saved as output.


# In[ ]:


df_train.shape


# In[ ]:


print(len(df_train.columns))
#new_cont_ord_cols = [c for c in df_train.columns if not c.startswith('ps_calc_')]
new_cont_ord_cols = [c for c in df_train.columns if not c.endswith('bin')]
no_bin_cat_cols = [c for c in new_cont_ord_cols if not c.endswith('cat')][2:]


# In[ ]:


''' 
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
col = [c for c in train.columns if c not in ['id','target']]
print(len(col))
col = [c for c in col if not c.startswith('ps_calc_')]
print(len(col))

train = train.replace(-1, np.NaN)
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}

'''


# In[ ]:


'''
def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c: #standard arithmetic
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
            #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
            #df[c+str('_exp')] = np.exp(df[c].values) - 1
    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred)

params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
x1, x2, y1, y2 = model_selection.train_test_split(train, train['target'], test_size=0.25, random_state=99)

x1 = multi_transform(x1)
x2 = multi_transform(x2)
test = multi_transform(test)

col = [c for c in x1.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]
print(x1.values.shape, x2.values.shape)

#remove duplicates just in case
tdups = multi_transform(train)
dups = tdups[tdups.duplicated(subset=col, keep=False)]

x1 = x1[~(x1['id'].isin(dups['id'].values))]
x2 = x2[~(x2['id'].isin(dups['id'].values))]
print(x1.values.shape, x2.values.shape)

y1 = x1['target']
y2 = x2['target']
x1 = x1[col]
x2 = x2[col]

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=200)
test['target'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit+45)
test['target'] = (np.exp(test['target'].values) - 1.0).clip(0,1)

sub = pd.DataFrame()
sub['id'] = test['id']
sub['target'] = test['target']
sub.to_csv('xgb1.csv', index=False)

#test[['id','target']].to_csv('xgb_submission.csv', index=False, float_format='%.5f')
'''


# # Data Exploration
# 
# First things first, let us explore what we have!

# In[ ]:


df_train.head()


# Saving the **target** variable separately and dropping it from the training set.

# In[ ]:


target = df_train['target']
#df_train = df_train.drop('target', 1)


# ## Analyzing Datatypes
# 
# We only have two datatypes in our dataset: **int** and **float**.

# In[ ]:


print(df_train.dtypes.unique())
print(df_train.dtypes.nunique())

print(df_test.dtypes.unique())
print(df_test.dtypes.nunique())


# In[ ]:


pp = pd.value_counts(df_train.dtypes)
pp.plot.bar()
plt.show()


# ## Analyzing Missing Values

# In[ ]:


print (df_train.isnull().values.any())
print (df_test.isnull().values.any())


# However, as mentioned by someone in the comments, "This isn't true!" The missing values have been replaced by -1.
# 
# We will replace them using np.nan and see how it is distributed.
# 
# 

# In[ ]:


#df_train.replace(-1, np.nan)
#df_test.replace(-1, np.nan)
df_train[(df_train == -1)] = np.nan
df_test[(df_test == -1)] = np.nan

print('done') 


# Checking for missing values again

# In[ ]:


print (df_train.isnull().values.any())
print (df_test.isnull().values.any())   


# Printing list of columns with missing values in both the train and test dataframe:

# In[ ]:


cols_missing_val_train = df_train.columns[df_train.isnull().any()].tolist()
print(cols_missing_val_train)
print('\n')

cols_missing_val_test = df_test.columns[df_test.isnull().any()].tolist()
print(cols_missing_val_test)


# We see that the train dataframe has an extra column with missing values (**ps_car_12**).
# 
# ## Visualizing missing values

# In[ ]:


#--- Train dataframe ---
msno.bar(df_train[cols_missing_val_train],figsize=(20,8),color="#19455e",fontsize=18,labels=True,)


# In[ ]:


#--- Test dataframe ---
msno.bar(df_test[cols_missing_val_test],figsize=(20,8),color="#50085e",fontsize=18,labels=True,)


# We can see that the missing values a proportional in both the test and train dataframes.

# In[ ]:


#--- Train dataframe ---
msno.matrix(df_train[cols_missing_val_train],width_ratios=(10,1),            figsize=(20,8),color=(0.2,0.2,0.2),fontsize=18,sparkline=True,labels=True)


# In[ ]:


#--- Test dataframe ---
msno.matrix(df_test[cols_missing_val_test],width_ratios=(10,1),            figsize=(20,8),color=(0.2,0.2,0.2),fontsize=18,sparkline=True,labels=True)


# We see a similar resemblance of proportional missing values in the train and test dataframes!

# Replacing the missing values to -1.

# In[ ]:


df_train.replace(np.nan, -1, inplace=True)
df_test.replace(np.nan, -1, inplace=True)


# ## Memory Usage

# In[ ]:


#--- memory consumed by train dataframe ---
mem = df_train.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
print('\n')
#--- memory consumed by test dataframe ---
mem = df_test.memory_usage(index=True).sum()
print("Memory consumed by test set      :   {} MB" .format(mem/ 1024**2))


# By altering the datatypes we can reduce memory usage:

# In[ ]:


def change_datatype(df):
    float_cols = list(df.select_dtypes(include=['int']).columns)
    for col in float_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

change_datatype(df_train)
change_datatype(df_test) 


# In[ ]:


#--- Converting columns from 'float64' to 'float32' ---
def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
        
change_datatype_float(df_train)
change_datatype_float(df_test)


# Let us check the memory consumed again:

# In[ ]:


#--- memory consumed by train dataframe ---
mem = df_train.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
print('\n') 
#--- memory consumed by test dataframe ---
mem = df_test.memory_usage(index=True).sum()
print("Memory consumed by test set      :   {} MB" .format(mem/ 1024**2))


# That is memory consumption reduced by **greater than 50%** !!!

# In[ ]:


print(len(df_test.columns))
print(len(df_train.columns))
#print(len(target.columns))


# # Quick Modeling (without any analysis)

# Quick check to make sure the columns are the same in both `train` and `test` data.

# In[ ]:


len(set(df_test.columns) and set(df_train.columns))


# ## Random Forest

# In[ ]:


df_train = df_train.replace(-1, np.NaN)
d_median = df_train.median(axis=0)
d_mean = df_train.mean(axis=0)
df_train = df_train.fillna(-1)

dcol = [c for c in df_train.columns if c not in ['id','target']]
df_train['ps_car_13_x_ps_reg_03'] = df_train['ps_car_13'] * df_train['ps_reg_03']
#df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
for c in dcol:
        if '_bin' not in c: #standard arithmetic
            df_train[c+str('_median_range')] = (df_train[c].values > d_median[c]).astype(np.int)
            df_train[c+str('_mean_range')] = (df_train[c].values > d_mean[c]).astype(np.int)
            df_train[c+str('_sq')] = np.power(df_train[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            df_train[c+str('_log')] = np.log(np.abs(df_train[c].values) + 1)
            df_train[c+str('_exp')] = np.exp(df_train[c].values) - 1


# In[ ]:


change_datatype(df_train)


# In[ ]:


df_train.head()


# In[ ]:


from sklearn.model_selection import train_test_split

features= [c for c in df_train.columns.values if c  not in ['id', 'target']]
#numeric_features= [c for c in df.columns.values if c  not in ['id','text','author','processed']]
#target = 'author'

X_train, X_test, y_train, y_test = train_test_split(df_train[features], df_train['target'], test_size=0.33, random_state=42)
X_train.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

pipeline = Pipeline([
    #('features',feats),
    ('classifier', RandomForestClassifier(random_state = 42))
    #('classifier', GradientBoostingClassifier(random_state = 42))
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
np.mean(preds == y_test)


# In[ ]:


pipeline.get_params().keys()


# In[ ]:


from sklearn.model_selection import GridSearchCV

hyperparameters = { #'features__text__tfidf__max_df': [0.9, 0.95],
                    #'features__text__tfidf__ngram_range': [(1,1), (1,2)],
                    #'classifier__learning_rate': [0.1, 0.2],
                    'classifier__n_estimators': [20, 30, 50],
                    'classifier__max_depth': [2, 4],
                    'classifier__min_samples_leaf': [2, 4]
                  }
clf = GridSearchCV(pipeline, hyperparameters, cv = 3)
 
# Fit and tune model
clf.fit(X_train, y_train)


# In[ ]:


clf.best_params_


# In[ ]:


#refitting on entire training data using best settings
clf.refit

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)

np.mean(preds == y_test)


# In[ ]:


df_test = df_test.replace(-1, np.NaN)
dt_median = df_test.median(axis=0)
dt_mean = df_test.mean(axis=0)
df_test = df_test.fillna(-1)

dtcol = [c for c in df_test.columns if c not in ['id']]
df_test['ps_car_13_x_ps_reg_03'] = df_test['ps_car_13'] * df_test['ps_reg_03']
#df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
for c in dtcol:
        if '_bin' not in c: #standard arithmetic
            df_test[c+str('_median_range')] = (df_test[c].values > dt_median[c]).astype(np.int)
            df_test[c+str('_mean_range')] = (df_test[c].values > dt_mean[c]).astype(np.int)
            df_test[c+str('_sq')] = np.power(df_test[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            df_test[c+str('_log')] = np.log(np.abs(df_test[c].values) + 1)
            df_test[c+str('_exp')] = np.exp(df_test[c].values) - 1


# In[ ]:


change_datatype(df_test)


# In[ ]:


submission = pd.read_csv('../input/test.csv')

#preprocessing
#test_features= [c for c in submission.columns.values if c  not in ['id']]
test_features= [c for c in df_test.columns.values if c  not in ['id']]
#submission = processing(submission)
predictions = clf.predict_proba(df_test[test_features])

preds = pd.DataFrame(data = predictions, columns = clf.best_estimator_.named_steps['classifier'].classes_)

#generating a submission file
result = pd.concat([submission[['id']], preds], axis=1)
result = result.drop(0, axis=1)
result.columns = ['id', 'target']
result.head()

result.to_csv('random_forest.csv', index=False)


# In[ ]:


'''  

from sklearn.cross_validation import train_test_split
import xgboost as xgb

X_train = df_train.drop(['id'],axis = 1)
X_id_train = df_train['id'].values
Y_train = target.values

X_test = df_test.drop(['id'], axis=1)
X_id_test = df_test['id'].values

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size = 0.4, random_state = 1000)
print('Train samples: {} Validation samples: {}'.format(len(x_train), len(x_valid)))

d_train = xgb.DMatrix(x_train, y_train)
d_valid = xgb.DMatrix(x_valid, y_valid)
d_test = xgb.DMatrix(X_test)

params = {}
params['min_child_weight'] = 10.0
params['objective'] = 'binary:logistic'
params['eta'] = 0.02
params['silent'] = True
params['max_depth'] = 9
params['subsample'] = 0.9
params['colsample_bytree'] = 0.9

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

# Create an XGBoost-compatible metric from Gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model = xgb.train(params, d_train, 100, watchlist, early_stopping_rounds=100, feval=gini_xgb, maximize=True, verbose_eval=10)

xgb.plot_importance(model)
fig, ax = plt.subplots(figsize=(12,18))
plt.show()

p_test = model.predict(d_test)

#--- Submission file ---

sub = pd.DataFrame()
sub['id'] = X_id_test
sub['target'] = p_test
sub.to_csv('xgb.csv', index=False)


importance = model.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
plt.gcf().savefig('features_importance.png')

''' 


# # Data Analysis
# ## Splitting columns based on types
# According to the data given to us:
# * features that belong to similar groupings are tagged as such in the feature names (e.g., **ind**, **reg**, **car**, **calc**). 
# * feature names include the postfix **bin** to indicate binary features and **cat** to indicate categorical features.
# * feature names without **boon** or **cat** are grouped as** continuous/ordinal** features.

# In[ ]:


#-- List of all columns --
train_cols = df_train.columns.tolist()

#--- binary and categorical features list ---
bin_cols = []
cat_cols = []

#--- continous/ordinal features list ---
cont_ord_cols = []

#--- different feature groupings ---
ind_cols = []
reg_cols = []
car_cols = []
calc_cols = []

for col in train_cols:
    if (('ps' in str(col)) & ('bin' not in str(col)) & ('cat' not in str(col))):
        cont_ord_cols.append(col)
    
for col in train_cols:
    if ('bin' in str(col)):
        bin_cols.append(col)
    if ('cat' in str(col)):
        cat_cols.append(col)
        
    if ('ind' in str(col)):
        ind_cols.append(col)
    if ('reg' in str(col)):
        reg_cols.append(col)
    if ('car' in str(col)):
        car_cols.append(col)
    if ('calc' in str(col)):
        calc_cols.append(col)
        


# Columns present in `cont_ord_cols` list have a collection of different types of columns.
# 
# So we can divide them into **continuous** and **ordinal** variables based on their data types.

# In[ ]:


float_cols = []
int_cols = []
for col in cont_ord_cols:
    if (df_train[col].dtype == np.float32):
          float_cols.append(col)        #--- continuous variables ---
    elif ((df_train[col].dtype == np.int8) or (df_train[col].dtype == np.int16)):
          int_cols.append(col)          #--- ordinal variables ---


# The following snippet is confirmation that all the variables are **ordinal** beacuse they have more than 2 unique values.

# In[ ]:


for col in int_cols:
    print (df_train[col].nunique())


# Exploring each of the above extracted grouped features individually:
# 
# ## Binary features:
# 
# Binary features whose single attribute is less than 10% will be collected in a separate list

# In[ ]:


cols_to_delete = []
th = 0.1
for col in range(0, len(bin_cols)):
    print (bin_cols[col])
    print (df_train[bin_cols[col]].unique())
    pp = pd.value_counts(df_train[bin_cols[col]])
    
    for i in range(0, len(pp)):
        if((pp[i]/float(len(df_train))) <= th):
            cols_to_delete.append(bin_cols[col])
            
    pp.plot.bar()
    plt.show()


# In[ ]:


print(cols_to_delete)


# The above mentioned columns have highly skewed values hence can be dropped from both the training and test set.

# In[ ]:



for col in cols_to_delete:
   df_train.drop([col], axis=1, inplace=True)
   df_test.drop([col], axis=1, inplace=True)
   


# ## Categorical Features
# 
# Exploring the categorical variables:

# In[ ]:


for col in range(0, len(cat_cols)):
    print (cat_cols[col])
    print (df_train[cat_cols[col]].unique())
    pp = pd.value_counts(df_train[cat_cols[col]])      
    pp.plot.bar()
    plt.show()


# From the graphs, only **ps_car_10_cat** is highly skewed hence can be removed from training and test set.

# In[ ]:


'''

cat_cols_to_delete = [ 'ps_car_10_cat']

for col in cat_cols_to_delete:
    df_train.drop([col], axis=1, inplace=True)
    df_test.drop([col], axis=1, inplace=True) 

''' 


# ## Continuous/Ordinal Features
# 
# Features having different prefixes such as **ind**, **reg**, **car** and **calc**; excluding binary and categorical features.

# In[ ]:


ind_cols_no_bin_cat = []
reg_cols_no_bin_cat = []
car_cols_no_bin_cat = []
calc_cols_no_bin_cat = []

for col in train_cols:
    if (('ind' in str(col)) and ('bin' not in str(col)) and ('cat' not in str(col))):
        ind_cols_no_bin_cat.append(col)
    if (('reg' in str(col)) and ('bin' not in str(col)) and ('cat' not in str(col))):
        reg_cols_no_bin_cat.append(col)
    if (('car' in str(col)) and ('bin' not in str(col)) and ('cat' not in str(col))):
        car_cols_no_bin_cat.append(col)
    if (('calc' in str(col)) and ('bin' not in str(col)) and ('cat' not in str(col))):
        calc_cols_no_bin_cat.append(col)


# ### Visualizing **ind** features
# 
# (Uncomment the following snippets of code to visualzie the various grouped features. They take a long time to load hence I have commented them out)

# In[ ]:


'''
what_col = ind_cols_no_bin_cat
for col in range(0, len(what_col)):
    print (what_col[col])
    print (df_train[what_col[col]].unique())
    pp = pd.value_counts(df_train[what_col[col]])      
    pp.plot.bar()
    plt.show()
'''   


# Column **ps_ind_14** is heavily skewed hence can be removed.

# ### Visualizing **reg** features

# In[ ]:


'''
what_col = reg_cols_no_bin_cat
for col in range(0, len(what_col)):
    print (what_col[col])
    print (df_train[what_col[col]].unique())
    pp = pd.value_counts(df_train[what_col[col]])      
    pp.plot.bar()
    plt.show()
 '''  


# Column **ps_reg_03** does not seem to show anything at all, hence can be removed.
# 
# ### Visualizing **car** features

# In[ ]:


''' 
what_col = car_cols_no_bin_cat
for col in range(0, len(what_col)):
    print (what_col[col])
    print (df_train[what_col[col]].unique())
    pp = pd.value_counts(df_train[what_col[col]])      
    pp.plot.bar()
    plt.show()
'''


# ### Visualizing **calc** features

# In[ ]:


''' 
what_col = calc_cols_no_bin_cat
for col in range(0, len(what_col)):
    print (what_col[col])
    print (df_train[what_col[col]].unique())
    pp = pd.value_counts(df_train[what_col[col]])      
    pp.plot.bar()
    plt.show()
'''


# Colukmns belonging to type ***calc***: 
# * **ps_calc_01**,
# * **ps_calc_02**,
# * **ps_calc_03** 
# 
# have a uniform distribution, which do not offer anything significant. Hence these can also be removed.

# In[ ]:


''' other_cols_to_delete = ['ps_ind_14', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_reg_03']

for col in other_cols_to_delete:
    df_train.drop([col], axis=1, inplace=True)
    df_test.drop([col], axis=1, inplace=True)''' 


# # Feature Engineering
# 
# ### NOTE: ALWAYS REMEMBER TO INCLUDE SAME SET OF FEATURES FOR THE TEST DATA ALSO!!

# In[ ]:


'''        
for col1 in int_cols:
    for col2 in float_cols:
        l_mean = 
        df_train[col1 + '_' + col2] = 
''' 


# ## New Binary Features
# 
# Here I have included logical AND, OR and XOR operation between every binary feature.

# In[ ]:


train_cols = df_train.columns
bin_cols = df_train.columns[df_train.columns.str.endswith('bin')]
''' 
for i in ["X1","X2"]:
    for j in ["X2","X3"]:
        if i != j:
            col_name = i + j
            k[col_name + '_OR'] = k[i]|k[j] 
            k[col_name + '_AND'] = k[i]&k[j] 
            k[col_name + '_XOR'] = k[i]^k[j] 
           
def second_order(df, c_names):
    names_col=[]
    pp=0
    for i in c_names[:c_names.size-1]:
        for j in c_names[pp:c_names.size]:
            if i != j:
                col_name = i + str('_') + j
                df[col_name + '_OR'] = df[i]|df[j] 
                df[col_name + '_AND'] = df[i]&df[j] 
                df[col_name + '_XOR'] = df[i]^df[j]
            
                #col_name = ii + str('_and_') + jj
                #names_col.append(col_name)
                #df[col_name] = df[ii]&df[jj]
        pp+=1
    return df, names_col   

df_train, train_new_cols = second_order(df_train, bin_cols)
df_test, test_new_cols = second_order(df_test, bin_cols)

print(len(df_train.columns))
print(len(df_test.columns))
'''


# ## New Continuous/Ordinal Features (*in progress*)

# In[ ]:


''' 
print(len(df_train.columns))
#new_cont_ord_cols = [c for c in df_train.columns if not c.startswith('ps_calc_')]
#new_cont_ord_cols = [c for c in df_train.columns if not c.endswith('bin') ]
for col in no_bin_cat_cols:
    #df_train[col + str('_greater_median')] = (df_train[col].values > df_train[col].median()).astype(np.int)
    #df_train[col + str('_greater_mean')] = (df_train[col].values > df_train[col].mean()).astype(np.int)
    df_train[col + str('_sq')] = np.power(df_train[col].values,2).astype(np.float32)
    df_train[col + str('_sqr')] = np.square(df_train[col].values).astype(np.float32)
    df_train[col + str('_log')] = np.log(np.abs(df_train[col].values) + 1)
    #df_train[col + str('_exp')] = np.exp(df_train[col].values) - 1
    
#new_cont_ord_test_cols = [c for c in df_test.columns if not c.startswith('ps_calc_')]
for col in no_bin_cat_cols:
    #df_test[col + str('_greater_median')] = (df_test[col].values > df_test[col].median()).astype(np.int)
    #df_test[col + str('_greater_mean')] = (df_test[col].values > df_test[col].mean()).astype(np.int)
    df_test[col + str('_sq')] = np.power(df_test[col].values,2).astype(np.float32)
    df_test[col + str('_sqr')] = np.square(df_test[col].values).astype(np.float32)
    df_test[col + str('_log')] = np.log(np.abs(df_test[col].values) + 1)
    #df_test[col + str('_exp')] = np.exp(df_test[col].values) - 1    
'''    


# ## New Second Order Continuous/Ordinal Features (*based on Gradient Boosting feature importance*)

# In[ ]:


'''
new_col =['ps_car_12', 'ps_car_14', 'ps_car_15', 'ps_car_13', 'ps_reg_03', 'ps_ind_03', 'ps_ind_15', 'ps_reg_02', 'ps_reg_01', 'ps_calc_02', 'ps_calc_11', 'ps_calc_10']


def new_second_order(df, c_names):
    names_col=[]
    pp=0
    for i in c_names[:len(c_names)-1]:
        for j in c_names[pp:len(c_names)]:
            if i != j:
                col_name = i + str('_*_') + j
                df[col_name] = df[i] * df[j] 
                
            
                #col_name = ii + str('_and_') + jj
                #names_col.append(col_name)
                #df[col_name] = df[ii]&df[jj]
        pp+=1
    return df, names_col   

df_train, train_new_cols = new_second_order(df_train, new_col)
df_test, test_new_cols = new_second_order(df_test, new_col)
'''


# In[ ]:


print(len(df_train.columns))
print(len(df_test.columns))


# ## Correlation

# In[ ]:


''' 
sns.set(style="white")
corr = df_train.corr()
f, ax = plt.subplots(figsize=(18, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
'''


# Outrageous! Not even a single **calc** feature seems to have any interest in indulging themselves with anything!! It is better to remove them all!!

# In[ ]:


'''
removed_calc_cols = []
for col in df_train.columns:
    if ('calc' in str(col)):
        removed_calc_cols.append(col)
    
#unwanted = train.columns[train.columns.str.startswith('ps_calc_')]

df_train = df_train.drop(removed_calc_cols, axis=1)  
df_test = df_test.drop(removed_calc_cols, axis=1)  
''' 


# In[ ]:


df_train.replace(np.nan, -1, inplace=True)
df_test.replace(np.nan, -1, inplace=True)
print('Done')


# # Modeling
# ## Gradient Boosting

# In[ ]:


''' 
X_train = df_train.drop(['id'],axis = 1)
X_id_train = df_train['id'].values
Y_train = target.values

X_test = df_test.drop(['id'], axis=1)
X_id_test = df_test['id'].values
'''


# In[ ]:


''' 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.02, max_depth=7, random_state = 0, loss='ls')
#GBR = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 500, max_depth = 9, min_samples_split = 2, min_samples_leaf = 2, max_features = 10, random_state=123)
    
GBR.fit(X_train, Y_train)

print (GBR)
'''


# In[ ]:


#--- List of important features for Gradient Boosting Regressor ---
''' 
features_list = X_train.columns.values
feature_importance = GBR.feature_importances_
sorted_idx = np.argsort(feature_importance)

print(sorted_idx)
''' 


# In[ ]:


''' 
plt.figure(figsize=(15, 15))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()
''' 


# In[ ]:


#--- Predicting Gradient boost result for test data ---
# y_GBR = GBR.predict(X_test)


# In[ ]:


''' 
final = pd.DataFrame()
final['id'] = X_id_test
final['target'] = y_GBR
final.to_csv('Gradient_Boost_1.csv', index=False)
print('DONE!!')
'''


# ## XGBoost

# In[ ]:


import xgboost as xgb


# In[ ]:


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


'''
from sklearn.model_selection import StratifiedKFold

kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=42)
'''


# In[ ]:


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


'''
for i, (train_index, test_index) in enumerate(skf.split(X_train, Y_train)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X_train[train_index], X_train[test_index]
    y_train, y_valid = Y_train[train_index], Y_train[test_index]
    # Convert our data into XGBoost format
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(X_test.values)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
    # and the custom metric (maximize=True tells xgb that higher metric is better)
    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=100)

    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    # Predict on our test data
    p_test = mdl.predict(d_test)
    sub['target'] += p_test/kfold
'''    


# ## Random Forest

# In[ ]:


''' 
from sklearn.ensemble import RandomForestClassifier  

RF = RandomForestClassifier(n_estimators=100, max_depth=8, criterion='entropy', min_samples_split=10, max_features=120, n_jobs=-1, random_state=123, verbose=1, class_weight = "balanced")
RF.fit(X_train, Y_train)

print(RF)

#--- List of important features ---

features_list = X_train.columns.values
feature_importance = RF.feature_importances_
sorted_idx = np.argsort(feature_importance)

print(sorted_idx)

plt.figure(figsize=(15, 15))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()

 
Y_pred = RF.predict(X_test)

final = pd.DataFrame()
final['id'] = X_id_test
final['target'] = Y_pred
final.to_csv('RF.csv', index=False)
print('DONE!!')

'''


# In[ ]:


#-- Adaboost ---
'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

Ada_R = AdaBoostRegressor(DecisionTreeRegressor(max_depth=7), n_estimators = 400, random_state = 99)

Ada_R.fit(X_train, Y_train)

print (Ada_R)

features_list = X_train.columns.values
feature_importance = Ada_R.feature_importances_
sorted_idx = np.argsort(feature_importance)

print(sorted_idx)

plt.figure(figsize=(15, 15))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()

#--- Predicting Ada boost result for test data ---
y_Ada = Ada_R.predict(X_test)


final = pd.DataFrame()
final['id'] = X_id_test
final['target'] = y_Ada
final.to_csv('Ada_Boost_1.csv', index=False)
print('DONE!!')
''' 


# In[ ]:


'''  

from sklearn.cross_validation import train_test_split
import xgboost as xgb

X_train = df_train.drop(['id'],axis = 1)
X_id_train = df_train['id'].values
Y_train = target.values

X_test = df_test.drop(['id'], axis=1)
X_id_test = df_test['id'].values

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=4242)
print('Train samples: {} Validation samples: {}'.format(len(x_train), len(x_valid)))

d_train = xgb.DMatrix(x_train, y_train)
d_valid = xgb.DMatrix(x_valid, y_valid)
d_test = xgb.DMatrix(X_test)

params = {}
params['min_child_weight'] = 10.0
params['objective'] = 'binary:logistic'
params['eta'] = 0.02
params['silent'] = True
params['max_depth'] = 9
params['subsample'] = 0.9
params['colsample_bytree'] = 0.9

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

# Create an XGBoost-compatible metric from Gini

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

model = xgb.train(params, d_train, 100, watchlist, early_stopping_rounds=100, feval=gini_xgb, maximize=True, verbose_eval=10)

xgb.plot_importance(model)
fig, ax = plt.subplots(figsize=(12,18))
plt.show()

p_test = model.predict(d_test)

#--- Submission file ---

sub = pd.DataFrame()
sub['id'] = X_id_test
sub['target'] = p_test
sub.to_csv('xgb2.csv', index=False)


importance = model.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
plt.gcf().savefig('features_importance.png')
''' 


# ### Can you think of more features? Let me know in the comments!
# 
# # STAY TUNED FOR MORE UPDATES !!!
