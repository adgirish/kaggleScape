
# coding: utf-8

# ## Here we'll try to encode categorical features...

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import xgboost as xgb 
from sklearn.metrics import r2_score

from IPython.display import display, HTML
# Shows all columns of a dataframe
def show_dataframe(X, rows = 2):
    display(HTML(X.to_html(max_rows=rows)))


# In[ ]:


# Datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# Categorical features
cat_cols = []
for c in train.columns:
    if train[c].dtype == 'object':
        cat_cols.append(c)
print('Categorical columns:', cat_cols)

# Dublicate features
d = {}; done = []
cols = train.columns.values
for c in cols: d[c]=[]
for i in range(len(cols)):
    if i not in done:
        for j in range(i+1, len(cols)):
            if all(train[cols[i]] == train[cols[j]]):
                done.append(j)
                d[cols[i]].append(cols[j])
dub_cols = []
for k in d.keys():
    if len(d[k]) > 0: 
        # print k, d[k]
        dub_cols += d[k]        
print('Dublicates:', dub_cols)

# Constant columns
const_cols = []
for c in cols:
    if len(train[c].unique()) == 1:
        const_cols.append(c)
print('Constant cols:', const_cols)


# Figures below show categorical features (on the left) sorted by means of **y**'s grouped by labels. On the right there are corresponding **mean**'s, **std**'s (filled blue), **max**'s (green line) and **min**'s (red line).

# In[ ]:


plt.figure(figsize=(20,32))
for i in range(len(cat_cols)):
    c = cat_cols[i]
    
    means = train.groupby(c).y.mean()
    stds = train.groupby(c).y.std().fillna(0)
    maxs = train.groupby(c).y.max()
    mins = train.groupby(c).y.min()
    
    ddd = pd.concat([means, stds, maxs, mins], axis=1); 
    ddd.columns = ['means', 'stds', 'maxs', 'mins']
    ddd.sort_values('means', inplace=True)
    
    plt.subplot(8,2,2*i+1)
    ax = sns.countplot(train[c], order=ddd.index.values)
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.0f}'.format(y), (x.mean(), y), ha='center', va='bottom')
    
    plt.subplot(8,2,2*i+2)
    plt.fill_between(range(len(train[c].unique())), 
                     ddd.means.values - ddd.stds.values,
                     ddd.means.values + ddd.stds.values,
                     alpha=0.3
                    )
    plt.xticks(range(len(train[c].unique())), ddd.index.values)
    plt.plot(ddd.means.values, color='b', marker='.', linestyle='dashed', linewidth=0.7)
    plt.plot(ddd.maxs.values, color='g', linestyle='dashed', linewidth=0.7)
    plt.plot(ddd.mins.values, color='r', linestyle='dashed', linewidth=0.7)
    plt.xlabel(c + ': Maxs, Means, Mins and +- STDs')
    plt.ylim(55, 270)


# In[ ]:


# Glue train + test
train['eval_set'] = 0; test['eval_set'] = 1
df = pd.concat([train, test], axis=0, copy=True)
# Reset index
df.reset_index(drop=True, inplace=True)


# ### Categorical feature encoding
# In the next cell for every categorical column from **cat_cols** we'll find **mean** of **y's** for every label using **.groupby()**. Then we sort labels by values of **means**. Now, when labels are sorted, they can be encoded by numbers from *0* to *numbers of labels - 1*.

# In[ ]:


def add_new_col(x):
    if x not in new_col.keys(): 
        # set n/2 x if is contained in test, but not in train 
        # (n is the number of unique labels in train)
        # or an alternative could be -100 (something out of range [0; n-1]
        return int(len(new_col.keys())/2)
    return new_col[x] # rank of the label

for c in cat_cols:
    # get labels and corresponding means
    new_col = train.groupby(c).y.mean().sort_values().reset_index()
    # make a dictionary, where key is a label and value is the rank of that label
    new_col = new_col.reset_index().set_index(c).drop('y', axis=1)['index'].to_dict()
    # add new column to the dataframe
    df[c + '_new'] = df[c].apply(add_new_col)

# drop old categorical columns
df_new = df.drop(cat_cols, axis=1)

# show the result
show_dataframe(df_new, 5)


# ### Train-test split

# In[ ]:


X = df.drop(list((set(const_cols) | set(dub_cols) | set(cat_cols))), axis=1)

# Train
X_train = X[X.eval_set == 0]
y_train = X_train.pop('y'); 
X_train = X_train.drop(['eval_set', 'ID'], axis=1)

# Test
X_test = X[X.eval_set == 1]
X_test = X_test.drop(['y', 'eval_set', 'ID'], axis=1)

# Base score
y_mean = y_train.mean()
# Shapes

print('Shape X_train: {}\nShape X_test: {}'.format(X_train.shape, X_test.shape))


# ### Model (XGBoost)

# In[ ]:


### Regressor

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 100, 
    'eta': 0.005,
    'max_depth': 3,
    'subsample': 0.95,
    'colsample_bytree': 0.6,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.log(y_mean),
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(X_train, np.log(y_train))
dtest = xgb.DMatrix(X_test)

# evaluation metric
def the_metric(y_pred, y):
    y_true = y.get_label()
    return 'r2', r2_score(y_true, y_pred)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=2000, 
                   nfold = 3,
                   early_stopping_rounds=50,
                   feval=the_metric,
                   verbose_eval=100, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print('num_boost_rounds=' + str(num_boost_rounds))

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

# Predict on trian and test
y_train_pred = np.exp(model.predict(dtrain))
y_pred = np.exp(model.predict(dtest))

print('First 5 predicted test values:', y_pred[:5])


# In[ ]:


plt.figure(figsize=(16,4))

plt.subplot(1,4,1)
train_scores = cv_result['train-r2-mean']
train_stds = cv_result['train-r2-std']
plt.plot(train_scores, color='green')
plt.fill_between(range(len(cv_result)), train_scores - train_stds, 
                 train_scores + train_stds, alpha=0.1, color='green')
test_scores = cv_result['test-r2-mean']
test_stds = cv_result['test-r2-std']
plt.plot(test_scores, color='red')
plt.fill_between(range(len(cv_result)), test_scores - test_stds, 
                 test_scores + test_stds, alpha=0.1, color='red')
plt.title('Train and test cv scores (R2)')

plt.subplot(1,4,2)
plt.title('True vs. Pred. train')
plt.plot([80,265], [80,265], color='g', alpha=0.3)
plt.scatter(x=y_train, y=y_train_pred, marker='.', alpha=0.5)
plt.scatter(x=[np.mean(y_train)], y=[np.mean(y_train_pred)], marker='o', color='red')
plt.xlabel('Real train'); plt.ylabel('Pred. train')

plt.subplot(1,4,3)
sns.distplot(y_train, kde=False, color='g')
sns.distplot(y_train_pred, kde=False, color='r')
plt.title('Distr. of train and pred. train')

plt.subplot(1,4,4)
sns.distplot(y_train, kde=False, color='g')
sns.distplot(y_pred, kde=False, color='b')
plt.title('Distr. of train and pred. test')



plt.figure(figsize=(18,1))
plt.plot(y_train_pred[:200], color='r', linewidth=0.7)
plt.plot(y_train[:200], color='g', linewidth=0.7)
plt.title('First 200 true and pred. trains')

print('Mean error =', np.mean(y_train - y_train_pred))
print('Train r2 =', r2_score(y_train, y_train_pred))


# ### Feature importance

# In[ ]:


# First 50 features
features_score = pd.Series(model.get_fscore()).sort_values(ascending=False)[:50]
plt.figure(figsize=(7,10))
sns.barplot(x=features_score.values, y=features_score.index.values, orient='h')


# In[ ]:


# output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
# output.to_csv('subm.csv', index=False)

