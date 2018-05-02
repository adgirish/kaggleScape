
# coding: utf-8

# Load the required libraries and data. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
macro = pd.read_csv('../input/macro.csv')

train.sample(3)
# Any results you write to the current directory are saved as output.


# Let's look at our target variable

# In[ ]:


plt.figure(figsize=(10,8))
sns.distplot(train.price_doc.values, bins=60, kde=True)
plt.xlabel('Price', fontsize=12)
plt.show()


# The distribution is right skewed. Let's log transform the variable.

# In[ ]:


plt.figure(figsize=(10,8))
sns.distplot(np.log(train.price_doc.values), bins=60, kde=True)
plt.xlabel('Price', fontsize=12)
plt.show()


# The peaks at points between 13 and 14 and 14 and 15 requires attention. I wi
# Let's look at the missing values 

# In[ ]:


def missing_plot(dataframe, figure_x, figure_y):
    df = dataframe.isnull().sum().reset_index()
    df.columns = ['column_name', 'na_count']
    df = df[df.na_count > 0]
    df = df.sort_values(by=['na_count'], ascending = [False])
    plt.figure(figsize=(figure_x, figure_y))
    sns.barplot(x="na_count", y ="column_name",data = df, orient="h")
    plt.xlabel('Missing count', fontsize=12)
    plt.show()
missing_plot(train, 10,45)


# In[ ]:


def corr_plot(dataframe, top_n, target, fig_x, fig_y):
    corrmat = dataframe.corr()
    #top_n - top n correlations +1 since price is included
    top_n = top_n + 1 
    cols = corrmat.nlargest(top_n, target)[target].index
    cm = np.corrcoef(train[cols].values.T)
    f, ax = plt.subplots(figsize=(fig_x,fig_y))
    sns.set(font_scale=1.25)
    cmap = plt.cm.viridis
    hm = sns.heatmap(cm, cbar=False, annot=True, square=True,cmap = cmap, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    return cols
corr_20 = corr_plot(train, 20, 'price_doc', 10,10)


# In[ ]:


len(train)


# In[ ]:


features_imp = list(corr_20[1:22])
data_train = train[features_imp]
data_test = test[features_imp]
data_train.head()


# Let's just build a model using the mean num of rooms for the missing values

# In[ ]:


data_train['num_room'].fillna(data_train["num_room"].mean(), inplace=True)
data_test['num_room'].fillna(data_train["num_room"].mean(), inplace = True)
data_train.head()
data_test.head()


# In[ ]:


target = list(corr_20[0:1])
id_test = test['id']


# In[ ]:


df_columns = data_train.columns
x_train = data_train.values
y_train = train[target].values
x_test = data_test.values


# In[ ]:


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(x_test, feature_names=df_columns)


# In[ ]:


cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()


# In[ ]:


num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)


# In[ ]:


y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()


# In[ ]:


output.to_csv('submission.csv', index=False)

