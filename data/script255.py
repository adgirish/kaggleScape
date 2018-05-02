
# coding: utf-8

# # Shake-up or Shake-down?
# 
# Everybody is talking about shake-up at this competition ([here](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43144), [here](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43315) ,[here](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43547), [here](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43336)). Here is my 2 cents. Let's try to estimate shake-up numericaly somehow. This notebook based on [nice exploration](https://www.kaggle.com/vpaslay/is-your-small-gini-significant) of how many samples should be guessed additionally to get an improvement of 0.001 of gini score and a [discussion](https://www.kaggle.com/vpaslay/is-your-small-gini-significant#244525) below it. Another [interesting kernel](https://www.kaggle.com/alexfir/expected-gini-standard-error) on this topic estimated the standard error of simple model depending on test size.
# 
# The main question I want to explore here is:
# - How much can be the **difference between public and private test score**?
# 
# We will use simple and naive method to estimate aforementioned difference depending on the public score. We do not have labels for test dataset, but we have train labels, so let's assume that our train set can represent test set. We will use OOF predictions of train set, split them randomly with a same proportion as public and private leaderboard split (private is **70%** of all test). (OOF predictions were taken from [notebook v 38](https://www.kaggle.com/aharless/xgboost-cv-lb-284), feature "New kernel with this data" didn't work as I expected and I couldn't read the data =( so downloaded and uploaded the validation predictions).
# 
# ## Assumptions
# It should be noticed that here we assume several things:
# - Train and test datasets have similar class balances;
# - Difference of sample sizes of train and test can be ignored;
# - Generaly: OOF predictions of train set can represent test set.

# ## Load data
# 
# Let's load OOF predictions and train target (with ID field) and define gini calculating function.

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

# load the data
oof_preds = pd.read_csv('../input/xgb-valid-preds-public/xgb_valid.csv')
y = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv', 
                usecols = ['id', 'target'])

print('Shape of OOF preds: \t', oof_preds.shape)
print('Shape of train target:\t', y.shape)


# In[ ]:


# gini calculation from https://www.kaggle.com/tezdhar/faster-gini-calculation
def ginic(actual, pred):
    actual = np.asarray(actual) #In case, someone passes Series or list
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalizedc(a, p):
    if p.ndim == 2:#Required for sklearn wrapper
        p = p[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1
    return ginic(a, p) / ginic(a, a)


# ## Single split
# 
# Here we make one split of OOF predictions using *train_test_split* from *sklearn* (with fixed seed). As mentioned above proportion of test size is 70% from all test - so we will use he same share. 

# In[ ]:


PROPORTION_PRIVATE = 0.70
y_preds_public, y_preds_private, y_public, y_private = train_test_split(oof_preds.target.values, 
                                                                        y.target.values, 
                                                                        test_size=PROPORTION_PRIVATE, 
                                                                        random_state=42)

print('Proportion of private:\t',PROPORTION_PRIVATE)
print('Public score:\t', round(gini_normalizedc(y_public, y_preds_public), 6))
print('Private score:\t', round(gini_normalizedc(y_private, y_preds_private), 6))


# So, we splited OOF predictions somehow and got 0.275 gini score on small part (public) and 0.290 on big part (private). That was a lucky split=) Let's do it many times to collect statistics over scores.

# ## 10k splits
# 
# Here we will do the public-private split 10 000 times with different random seeds and collect gini scores from every split. (take some time - about 20 min)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'gini_public = []\ngini_private = []\n# do the split 10k times\nfor rs in range(10000):\n    y_preds_public, y_preds_private, y_public, y_private = train_test_split(oof_preds.target.values, \n                                                                            y.target.values, \n                                                                            test_size=PROPORTION_PRIVATE, \n                                                                            random_state=rs)\n    gini_public.append(gini_normalizedc(y_public, y_preds_public))\n    gini_private.append(gini_normalizedc(y_private, y_preds_private))\n\n# save results to numpy arrays\ngini_public_arr = np.array(gini_public)\ngini_private_arr = np.array(gini_private)')


# Let's plot a histogram of public-private difference of scores:

# In[ ]:


# 10000 random_states
plt.figure(figsize=(10,6))
plt.hist(gini_public_arr - gini_private_arr, bins=50)
plt.title('(Public - Private) scores')
plt.xlabel('Gini score difference')
plt.show()


# Looks much the same as in [aforementioned kernel](https://www.kaggle.com/vpaslay/is-your-small-gini-significant): we have deviation mostly between -0.02 and 0.02, it's realy huge range that leads to depression=(.
# 
# But wait! Here we use the OOF predictions of model which score on leaderboard **we know** (it's 0.284). So let's naively assume that this score represent our public-private split score on train and find in our array of public ginis (computed above) those splits, which score 0.284. Let's plot the public-private difference only for them:

# In[ ]:


#find indexies where public score was .284
my_indexies = np.where((gini_public_arr >= 0.284) &(gini_public_arr < 0.285))[0]

plt.figure(figsize=(10,6))
plt.hist(gini_public_arr[my_indexies] - gini_private_arr[my_indexies], bins=50)
plt.title('(Public - Private) scores, where public = .284')
plt.xlabel('Gini score difference')
plt.show()


# Hm... absolutely different picture: we have not so wide, uniform range between -0.003 and -0.0016, and most importantly that private score is a higher than public (all differences < 0).
# 
# For comparison let's look at differences in splits, which scores 0.286 on public part:

# In[ ]:


#find indexies where public score was .286
my_indexies = np.where((gini_public_arr >= 0.286) &(gini_public_arr < 0.287))[0]

plt.figure(figsize=(10,6))
plt.hist(gini_public_arr[my_indexies] - gini_private_arr[my_indexies], bins=50)
plt.title('(Public - Private) scores, where public = .286')
plt.xlabel('Gini score difference')
plt.show()


# On the plot above again we have uniform distribution in range between -0.0020 and 0.0012. Lets compare it with range of public between 0.284 and 0.287 (not including 0.287)

# In[ ]:


#find indexies where public score was .284-.287
my_indexies = np.where((gini_public_arr >= 0.284) &(gini_public_arr < 0.287))[0]

plt.figure(figsize=(10,6))
plt.hist(gini_public_arr[my_indexies] - gini_private_arr[my_indexies], bins=50)
plt.title('(Public - Private) scores, where public between .284 and .287')
plt.xlabel('Gini score difference')
plt.show()


# ## Summary
# 
# So if we take into consideration **all assumptions mentioned above** and assume that our model (which OOF we used here) quiet stable and scores 0.284 on public we can expect private score between 0.285 and 0.287 (from -0.001 to -0.003), which is literally speaking "shake UP", not "shake DOWN" of scores. 
# 
# So that is quite interesting conclusion and what needed to be mentioned that this method is truely naive (and maybe, misleading) and used several assumptions, which can be violated in real train-test setting.
# 
# Hope this notebook will help you guys. If you have any comments or remarks feel free to write them below.
