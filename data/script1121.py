
# coding: utf-8

# # Mean-Encoding High Cardinality Categoricals without Seeing into the future.
# 
# When we want to build models that predict `y ~ x` and `x` is a high cardinality categorical variable we often try to do one-hot encoding with a sparse matrix and hope that everything works. However it may not be best to so in terms of model complexity, overfitting, and leakage. Instead we may want to try mean encoding features without seeing into the future. 
# 
# ## Mean Encoding  w/ a empirical prior 
# 
# Often times we may want to mean encode features. Instead of fitting a model like `y ~ (1 if X=x else 0)` we may want to replace it with a estimate of `p(y | X=x)`. While the MLE may work well if all categories had sufficient sample size to do an estimate, in this data set it is not the case.
# 
# > For example if we flipped a coin once and got heads. our MLE suggests
# > that` P(heads|coin1) = 1/1 = 1 `and our model would think that this
# > coin is definitly biases. however if we flipped a join 100 times and
# > got 90 heads, `P(heads| coin2) = 90/100 = .9` although the MLE is
# > smaller we have an idea that our estimate is more confident.
# 
# Turns out we can think of adding a prior as the same thing as adding fake data, so say we add 10 fake unbiased coin flips we would expect 5 heads, that would mean our new estimate would look like 
# 
# P(heads | coin1) = (1 + 5 / 1 + 10) = 55% 
# P(heads | coin2) = (90 + 5 / 100 + 10) = 86% 
# 
# ## Without seeing into the future
# 
# If we took just the estimates for each id we could come into situations where future events could be predicting the pass, therefore, we should also make sure that on any given date, we only use historical data.
# 
# 
# # Method
# 
# that being said our method is quite simple.
# 
# 
# 1. Estimate a empirical prior using MLE of the whole data.
# 2. On any given day, for a building_id, we use the historical data to estimate the counts, add in the fake data and compute the posterior.
# 
# # Contents
# 
# This notebooks goes over a step by step example of how things work and presents the strip in the last few cells. I hope you've learning something either about pandas or math. Let me know if there are more efficient ways of doing what I did!
# 
# Thanks!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_df = pd.read_json("../input/train.json")
test_df = pd.read_json("../input/test.json")
train_test = pd.concat([train_df, test_df], 0).set_index("listing_id")


# Here we will use building_id but feel free to fork and run it with manager_id, the script will output a CSV indexed on listing_id so you can join that to the rest of your data and fit a model! 

# In[ ]:


# ARGUMENTS

category = "building_id"
n_fake_data = 20


# In[ ]:


df = train_test[[category, "created", "interest_level"]].copy()

df["low"] = (df["interest_level"] == "low").astype(int)
df["medium"] = (df["interest_level"] == "medium").astype(int)
df["high"] = (df["interest_level"] == "high").astype(int)

df["created"] = pd.to_datetime(df["created"]).dt.dayofyear
train_test["created"] = df["created"]

del df["interest_level"]

df.head()


# ## Compute Empirical Prior

# In[ ]:


interests = ["low", "medium", "high"]
priors = df[interests].sum() / df[interests].sum().sum()
priors


# In[ ]:


df = (df.
      sort_values("created").
      groupby([category, "created"]).
      agg(sum)[interests].
      reset_index("created"))


# ## Example for single building_id

# In[ ]:


# sort on created in order to make sure we are always using historical data
# see that each day has its own count data for low/medium/high

temp = df.loc[["0"]].copy().sort_values("created")
temp.head()


# In[ ]:


# by using cumsum we essentually compute the best counts using historical data

temp[interests] = temp[interests].cumsum(0)
temp.head()


# In[ ]:


# we then shift in order to make sure that one any given day, 
# we do not know what is happening on that day

temp[interests] = temp[interests].shift().fillna(0)
temp.head()


# In[ ]:


# add the prior aka 'fake data' into the mix 

temp[interests] = temp[interests] + n_fake_data * priors
temp.head()


# In[ ]:


# then we compute the MLE with the fake data, (not sure if this counts as MAP)

n = temp[interests].sum(1)
temp[interests] = temp[interests].apply(lambda _: _/n)
temp.head()


# Now we see that each day has a reasonable probability distribution and it has been smoothed out with the prior. also note that as we collect more data, the  influence of the fake samples will disappear. changing `n_fake_data` will change how that influence spreads out.
# 
# # Computing Final Results
# 
# The for loop takes a bit, about 2 minutes. if there is a way to do this more efficiently please let me know!

# In[ ]:


nd1 = 1 + n_fake_data 
npriors = n_fake_data * priors 


# In[ ]:


idxs = set(df.index)
total = len(idxs)

for i, idx in enumerate(idxs):
    temp = df.loc[[idx]].copy()
    if len(temp) == 1:
        temp.loc[:,interests] = temp.loc[:,interests].fillna(0) + npriors
        temp.loc[:,interests] /= nd1
    else:
        temp.loc[:,interests] = temp.loc[:,interests].cumsum(0).shift().fillna(0) + npriors
        n = temp.loc[:,interests].sum(1)
        temp.loc[:,interests] = temp.loc[:,interests].apply(lambda _: _/n)
    df.loc[[idx]] = temp
    
    if i % 1000 == 0:
        print("completed {}/{}".format(i, total))
        
df.reset_index(category, inplace=1)


# In[ ]:


features = train_test[[category, "created"]].copy()
features["listing_id"] = train_test.index
features = pd.merge(df, features, left_on=[category, "created"], right_on=[category, "created"])
features = features.set_index("listing_id")[interests]
features.columns = [category + "_" + c for c in features.columns]


# In[ ]:


features.sample(10)


# In[ ]:


features.to_csv("meancoded_{}.csv".format(category))

