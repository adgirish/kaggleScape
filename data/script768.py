
# coding: utf-8

# # Mercari Price Suggestion Challenge
# 
# This is first of its kind and kernel only competition, and stage2 files will not be downloaded and will be available only in kernels. In this notebook, I am doing basic data exploration and reporting my findings. I will be making a simple model at the end to just as to make it complete. Val loss will be reported at the end of this notebook.
# 
# **NOTE - If you find this kernel useful, please upvote, and if you have any suggestion or if anything is not clear please comment, I will try to explain my work**
# 
# In this notebook we will extract many features and make a simple model which runs faster than **Bojan** :P I will be extracting following features - 
# - **Yes/No features ** - if description present, if brand name present
# - ** Category and brand encoding** - category has three levels and we will be seperating each level and then provide encoding for all three levels.
# - ** SVD comp on if-idf calculated over item description** - *self explanatory*
# - **SVD comp on if-itf calculated over product name** - *self explanatory*
# - **item_condition_id** - use it without processing
# - **shipping** - use it without processing
# 
# *Only very few features in first round of feature extraction*
# 
# *** We will train a XGB model to check how the extracted features are performing***
# 
# **PS_1:  These features are working very well, please change n_comp in SVD and change the number of iterations in XGB from 80 to 1000, you will get very high accuracies.**
# 
# **PS_2: I am learning markov models for predicting probability of author given sentence ( with history of last 3 words), I have taken a part of code from internet and I have written a probability function, but that function is faulty and I am having difficulty fixing it. If you know markov models, please contact or have a look at [notebook](https://www.kaggle.com/maheshdadhich/creative-feature-engineering-lb-0-35) **
# 
# 
# 
# 

# In[ ]:


import pandas as pd  #pandas for using dataframe and reading csv 
import numpy as np   #numpy for vector operations and basic maths 
import urllib        #for url stuff
import re            #for processing regular expressions
import datetime      #for datetime operations
import calendar      #for calendar for datetime operations
import time          #to get the system time
import scipy         #for other dependancies
from sklearn.cluster import KMeans # for doing K-means clustering
from haversine import haversine # for calculating haversine distance
import math          #for basic maths operations
import seaborn as sns #for making plots
import matplotlib.pyplot as plt # for plotting
import os                # for os commands
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes


# ## Reading in the files..

# In[ ]:


train_df = pd.read_csv('../input/train.tsv', sep='\t')
train_df.head(10)


# In[ ]:


# checking test file.. 
test_df = pd.read_csv('../input/test.tsv', sep='\t')
test_df.head()
# its clear that we are supposed to predict the price, given other variables.


# In[ ]:


# Lets check the basic price histogram and see if what is the range of prices 
get_ipython().run_line_magic('matplotlib', 'inline')
start = time.time()
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.despine(left=True)
sns.distplot(np.log(train_df['price'].values+1), axlabel = 'Log(price)', label = 'log(trip_duration)', bins = 50, color="y")
plt.setp(axes, yticks=[])
plt.tight_layout()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
plt.show()


# ** Findings ** - This is the plot of log(price), I expected that the prices variation will be there a lot, so I checked the log(price) histogram instead of normal histogram. Its clear from above plot that most of the products have price greater than 1 dollars , and highest number of products have prices between \$ 100  to \$ 1000. 

# In[ ]:


# Lets check if shipping has any impact on prices 
start = time.time()
fig, ax = plt.subplots(figsize=(11, 7), nrows=2, sharex=True, sharey=True)
sns.distplot(np.log(train_df.loc[train_df['shipping']==1]['price'].values+1), ax=ax[0], color='blue', label='shipping')
sns.distplot(np.log(train_df.loc[train_df['shipping']==0]['price'].values+1), ax=ax[1], color='green', label='No shipping')
ax[0].legend(loc=0)
ax[1].legend(loc=0)
plt.show()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))


# **findings** - Its clear that the No shipping has slightly narrow distribution of prices and starting from 100 dollors while shipping products are starting from 10 dollars 

# In[ ]:


# Lets check the basic price histogram and see if what is the range of prices 
get_ipython().run_line_magic('matplotlib', 'inline')
start = time.time()
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
sns.despine(left=True)
sns.distplot(train_df['item_condition_id'], axlabel = 'item_condition_id', label = 'item_condition_id', bins = 12, color="g", kde = False)
plt.setp(axes, yticks=[])
plt.tight_layout()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))
plt.show()


# # Basic feature engineering 
# Lets start with features that we have mentioned in introduction in FE round 1, we will add other features but first check how these features are performing in for prediction. For finding category label, we need to first devide item category into three categories, primary, secondary and tirtiary ( I have named them _1/_2/_3 here). After that we will make a dictionary and give labels to different category and make cat labels features.
# 
# - **1. Category label features - **

# In[ ]:


# 1. Extract 3 category related features 
def cat_split(row):
    try:
        text = row
        txt1, txt2, txt3 = text.split('/')
        return txt1, txt2, txt3
    except:
        return np.nan, np.nan, np.nan


train_df["cat_1"], train_df["cat_2"], train_df["cat_3"] = zip(*train_df.category_name.apply(lambda val: cat_split(val)))
test_df["cat_1"], test_df["cat_2"], test_df["cat_3"] = zip(*test_df.category_name.apply(lambda val: cat_split(val)))
train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


# making dictionaries for different categories 
keys = train_df.cat_1.unique().tolist() + test_df.cat_1.unique().tolist()
keys = list(set(keys))
values = list(range(keys.__len__()))
cat1_dict = dict(zip(keys, values))

keys2 = train_df.cat_2.unique().tolist() + test_df.cat_2.unique().tolist()
keys2 = list(set(keys2))
values2 = list(range(keys2.__len__()))
cat2_dict = dict(zip(keys2, values2))

keys3 = train_df.cat_3.unique().tolist() + test_df.cat_3.unique().tolist()
keys3 = list(set(keys3))
values3 = list(range(keys3.__len__()))
cat3_dict = dict(zip(keys3, values3))


# In[ ]:


# function to assign category label
def cat_lab(row,cat1_dict = cat1_dict, cat2_dict = cat2_dict, cat3_dict = cat3_dict):
    """function to give cat label for cat1/2/3"""
    txt1 = row['cat_1']
    txt2 = row['cat_2']
    txt3 = row['cat_3']
    return cat1_dict[txt1], cat2_dict[txt2], cat3_dict[txt3]

train_df["cat_1_label"], train_df["cat_2_label"], train_df["cat_3_lable"] = zip(*train_df.apply(lambda val: cat_lab(val), axis =1))
test_df["cat_1_label"], test_df["cat_2_label"], test_df["cat_3_lable"] = zip(*test_df.apply(lambda val: cat_lab(val), axis =1))
train_df.head(10)


# ** 2. if category present -yes/no features -**

# In[ ]:


def if_catname(row):
    """function to give if brand name is there or not"""
    if row == row:
        return 1
    else:
        return 0
    
train_df['if_cat'] = train_df.category_name.apply(lambda row : if_catname(row))
test_df['if_cat'] = test_df.category_name.apply(lambda row : if_catname(row))
train_df.head()


# **3. If brand name is present - yes/no features -**

# In[ ]:


# brand name related features 
def if_brand(row):
    """function to give if brand name is there or not"""
    if row == row:
        return 1
    else:
        return 0
    
train_df['if_brand'] = train_df.brand_name.apply(lambda row : if_brand(row))
test_df['if_brand'] = test_df.brand_name.apply(lambda row : if_brand(row))
train_df.head()


# ** 4. Brand name label features -** 

# In[ ]:


# makinfg brand name dict features 
keys = train_df.brand_name.dropna().unique()
values = list(range(keys.__len__()))
brand_dict = dict(zip(keys, values))

def brand_label(row):
    """function to assign brand label"""
    try:
        return brand_dict[row]
    except:
        return np.nan

train_df['brand_label'] = train_df.brand_name.apply(lambda row: brand_label(row))
test_df['brand_label'] = test_df.brand_name.apply(lambda row: brand_label(row))
train_df.head()


# ** 5. if item_description present - yes/no feature -**

# In[ ]:


# item description related features 
print("Description of item is not present in {}".format(train_df.loc[train_df.item_description == 'No description yet'].shape[0]))
print("while the shape of train_df is {}".format(train_df.shape[0]))

def if_description(row):
    """function to say if description is present or not"""
    if row == 'No description yet':
        a = 0
    else:
        a = 1
    return a

train_df['is_description'] = train_df.item_description.apply(lambda row : if_description(row))
test_df['is_description'] = test_df.item_description.apply(lambda row : if_description(row))
train_df.head()


# In[ ]:


# Nulls in item description in train or test as tf-idf is not defined on nan
print(train_df.item_description.isnull().sum())
print(test_df.item_description.isnull().sum())
# lets drop these 4 items 
print(train_df.shape[0])
train_df = train_df.loc[train_df.item_description == train_df.item_description]
test_df = test_df.loc[test_df.item_description == test_df.item_description]
train_df = train_df.loc[train_df.name == train_df.name]
test_df = test_df.loc[test_df.name == test_df.name]
print(train_df.shape[0])
print("Dropped records where item description was nan")


# ** 6. SVD on tf-idf on unigrams for iten_description -**

# In[ ]:


# description related tf-idf features 
# I guess "No dscription present won't affact these features ... So, I am not removing them.
import time
start = time.time()
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train_df['item_description'].values.tolist() + test_df['item_description'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['item_description'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['item_description'].values.tolist())

n_comp = 40
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)
end = time.time()
print("time taken {}".format(end - start))


# In[ ]:


print(train_df.shape[0])
train_df = train_df.loc[train_df.item_description == train_df.item_description]
test_df = test_df.loc[test_df.item_description == test_df.item_description]
train_df = train_df.loc[train_df.name == train_df.name]
test_df = test_df.loc[test_df.name == test_df.name]
print(train_df.shape[0])
print("Dropped records where item description was nan")


# **7. SVD on tf-idf of unigram of product name features -**
# 

# In[ ]:


print(train_df.shape[0])
train_df = train_df.loc[train_df.item_description == train_df.item_description]
test_df = test_df.loc[test_df.item_description == test_df.item_description]
train_df = train_df.loc[train_df.name == train_df.name]
test_df = test_df.loc[test_df.name == test_df.name]
print(train_df.shape[0])
print("Dropped records where item description was nan")


# In[ ]:


# product name related features 

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train_df['name'].values.tolist() + test_df['name'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['name'].values.tolist())
test_tfidf = tfidf_vec.transform(test_df['name'].values.tolist())

n_comp = 40
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    
train_svd.columns = ['svd_name_'+str(i) for i in range(n_comp)]
test_svd.columns = ['svd_name_'+str(i) for i in range(n_comp)]
train_df = pd.concat([train_df, train_svd], axis=1)
test_df = pd.concat([test_df, test_svd], axis=1)


# In[ ]:


# test check for dimensions before model 
print("Train should have one columns more than test")
print(train_df.shape[1])
print(test_df.shape[1])
print("perfect The data is fine")


#  # XGboost regressor ...
#  Now we have 49 features which could be used in price prediction and let's use them and see how they are performing 

# In[ ]:


# XGboost regressor ...
# replace all nan with -1 
print(train_df.isnull().sum())
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)
print(train_df.isnull().sum())


# In[ ]:


train = train_df.copy()
test = test_df.copy()
print("Difference of features in train and test are {}".format(np.setdiff1d(train.columns, test.columns)))
print("")
do_not_use_for_training = ['cat_1','test_id','cat_2','cat_3','train_id','name', 'category_name', 'brand_name', 'price', 'item_description']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
print("We will be using following features for training {}.".format(feature_names))
print("")
print("Total number of features are {}.".format(len(feature_names)))


# In[ ]:


y = np.log(train['price'].values + 1)


# In[ ]:


from sklearn.model_selection import train_test_split
Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(test[feature_names].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

start = time.time()
xgb_par = {'min_child_weight': 20, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 15,
            'subsample': 0.9, 'lambda': 2.0, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}

model_1 = xgb.train(xgb_par, dtrain, 80, watchlist, early_stopping_rounds=20, maximize=False, verbose_eval=20)
print('Modeling RMSLE %.5f' % model_1.best_score)
end = time.time()
print("Time taken in training is {}.".format(end - start))


# In[ ]:


start = time.time()
yvalid = model_1.predict(dvalid)
ytest = model_1.predict(dtest)
end = time.time()
print("Time taken in prediction is {}.".format(end - start))


# In[ ]:


# Lets check how the distribution of test and vaidation set looks like ...
start = time.time()
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
sns.distplot(yvalid, ax=ax[0], color='blue', label='Validation')
sns.distplot(ytest, ax=ax[1], color='green', label='Test')
ax[0].legend(loc=0)
ax[1].legend(loc=0)
plt.show()
end = time.time()
print("Time taken by above cell is {}.".format((end-start)))


# In[ ]:


start = time.time()
if test.shape[0] == ytest.shape[0]:
    print('Test shape OK.') 
test['price'] = np.exp(ytest) - 1
test[['test_id', 'price']].to_csv('mahesh_xgb_submission_mercari.csv', index=False)
end = time.time()
print("Time taken in training is {}.".format(end - start))


# **Sleep time zzzzz......**
# # Upvote if you find this analysis useful... Thanks (y)
