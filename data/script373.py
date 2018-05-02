
# coding: utf-8

# In this exploration notebook, we shall try to uncover the basic information about the dataset which will help us build our models / features.
# 
# Let us start with importing the necessary modules.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)


# First let us import the train file and get some idea about the data.

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
train_df.shape


# In[ ]:


train_df.head()


# There are quite a few variables in this dataset. 
# 
# Let us start with target variable exploration - 'price_doc'. First let us do a scatter plot to see if there are any outliers in the data.

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.show()


# Looks okay to me. Also since the metric is RMSLE, I think it is okay to have it as such. However if needed, one can truncate the high values. 
# 
# We can now bin the 'price_doc' and plot it.

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train_df.price_doc.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()


# Certainly a very long right tail. Since our metric is Root Mean Square **Logarithmic** error, let us plot the log of price_doc variable.

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(np.log(train_df.price_doc.values), bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()


# This looks much better than the previous one. 
# 
# Now let us see how the median housing price change with time. 

# In[ ]:


train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4]+x[5:7])
grouped_df = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()


# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(grouped_df.yearmonth.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Year Month', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# There are some variations in the median price with respect to time. Towards the end, there seems to be some linear increase in the price values.
# 
# Now let us dive into other variables and see. Let us first start with getting the count of different data types. 

# In[ ]:


train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# So majority of them are numerical variables with 15 factor variables and 1 date variable.
# 
# Let us explore the number of missing values in each column.

# In[ ]:


missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# Seems variables are found to missing as groups.
# 
# Since there are 292 variables, let us build a basic xgboost model and then explore only the important variables.

# In[ ]:


for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
        
train_y = train_df.price_doc.values
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


# So the top 5 variables and their description from the data dictionary are:
# 
#  1. full_sq - total area in square meters, including loggias, balconies and other non-residential areas
#  2. life_sq - living area in square meters, excluding loggias, balconies and other non-residential areas
#  3. floor - for apartments, floor of the building
#  4. max_floor - number of floors in the building
#  5. build_year - year built
# 
# Now let us see how these important variables are distributed with respect to target variable.
# 
# **Total area in square meters:**

# In[ ]:


ulimit = np.percentile(train_df.price_doc.values, 99.5)
llimit = np.percentile(train_df.price_doc.values, 0.5)
train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit

col = "full_sq"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=np.log1p(train_df.full_sq.values), y=np.log1p(train_df.price_doc.values), size=10)
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of Total area in square metre', fontsize=12)
plt.show()


# **Living area in square meters:**

# In[ ]:


col = "life_sq"
train_df[col].fillna(0, inplace=True)
ulimit = np.percentile(train_df[col].values, 95)
llimit = np.percentile(train_df[col].values, 5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=np.log1p(train_df.life_sq.values), y=np.log1p(train_df.price_doc.values), 
              kind='kde', size=10)
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of living area in square metre', fontsize=12)
plt.show()


# **Floor:**
# 
# We will see the count plot of floor variable.

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="floor", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# The distribution is right skewed. There are some good drops in between (5 to 6, 9 to 10, 12 to 13, 17 to 18). Now let us see how the price changes with respect to floors.

# In[ ]:


grouped_df = train_df.groupby('floor')['price_doc'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,8))
sns.pointplot(grouped_df.floor.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# This shows an overall increasing trend (individual houses seems to be costlier as well - check price of 0 floor houses). 
# A sudden increase in the house price is also observed at floor 18.
# 
# **Max floor:**
# 
# Total number of floors in the building is one another important variable. So let us plot that one and see.

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="max_floor", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Max floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# We could see that there are few tall bars in between (at 5,9,12,17 - similar to drop in floors in the previous graph). May be there are some norms / restrictions on the number of maximum floors present(?). 
# 
# Now let us see how the median prices vary with the max floors. 

# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x="max_floor", y="price_doc", data=train_df)
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Max Floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# More to come. Stay tuned.!
