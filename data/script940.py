
# coding: utf-8

# #Inspiration
# This kernel is inspired by [David Thaler][1] idea to estimate the distribution of LB. My idea is to exploit the fact that log function exacerbates the value of x near 0 as opposite to its value at 1 (which is exactly zero). In order to deal with computational stability, log loss function utilizes a small coefficient epsilon = 1e-15. Fortunately, this small additive coefficient still reserve the great disparency between log(1-epsilon) and log(epsilon).
# 
# I will first present my estimation method first then do all the computation at the end.
# 
#   [1]: https://www.kaggle.com/davidthaler/quora-question-pairs/how-many-1-s-are-in-the-public-lb

# In[ ]:


import numpy as np
import pandas as pd
epsilon = 1e-15
print('log(epsilon):',-np.log(epsilon))
print('log(1-epsilon):',-np.log(1-epsilon))


# Here is a [quick description][1] of log loss function.
# 
# #Motivation
# 
# Firstly, what happened when we label all test data as 0. This obviously gives best loss score on negative label data and huge penalty on positive test data. In order word, this the log loss score would consist of many scores from positive label data (which is a constant **log(epsilon)** for each positive data point in LB). Thus we can ignore the log loss score from negative data point (**log(1-epsilon)** per negative data point).
# 
# [1]:https://www.kaggle.com/wiki/LogarithmicLoss

# In[ ]:


df_test = pd.read_csv('../input/test.csv')
n = len(df_test)
print('n:',n)

const = (n/n)*-np.log(1-epsilon)
print('const:',const)


# #Estimate portion of n1 (n1/n)
# 
# Let define n1, n2, n3 as the number of positive data point, negative data point, and computer generated data point. Let call alpha as the log loss of test data label all 0. We have:
# alpha = (n1/n)*-log(epsilon)+(n2/n)*-log(1-epsilon)
# 
# ##Lower bound of n1/n:
# 
# Since we only have n=2345796 data row in the testing data set, n2 < 2345796. Thus:
# 
# (n2/n)*-log(1-epsilon) < const, with const = (n/n)*-log(1-epsilon) = 9.99200722163e-16
# Remember that: -log(1-epsilon) > 0
# 
# Then alpha < (n1/n)*-log(epsilon)+ const. As a result: **(n1/n) > (alpha - const)/(-log(epsilon))**
# 
# ##Upper bound of n1/n:
# (n1/n)*-log(epsilon) < alpha Then: **(n1/n) < alpha/(-log(epsilon))**
# 
# Thus we have: **(alpha - const)/(-log(epsilon)) < (n1/n) < alpha/(-log(epsilon))**
# 
# #Estimating portion of n2 (n2/n):
# The same goes for n2. Let call beta = (n1/n)*-log(1-epsilon)+(n2/n)*-log(epsilon)
# 
# Then **(beta - const)/(-log(epsilon)) < (n2/n) < beta/(-log(epsilon))**
# 
# 
#   [1]: https://www.kaggle.com/wiki/LogarithmicLoss

# In[ ]:


test = pd.read_csv('../input/test.csv')
sub = test[['test_id']].copy()
sub['is_duplicate'] = 0
sub.to_csv('submission_alpha.csv', index=False)


# In[ ]:


test = pd.read_csv('../input/test.csv')
sub = test[['test_id']].copy()
sub['is_duplicate'] = 1
sub.to_csv('submission_beta.csv', index=False)


#  This result alpha = 6.0188 and beta = 28.52056

# In[ ]:


alpha = 6.0188 
beta = 28.52056

print ((alpha - const)/(-np.log(epsilon)),'< (n1/n) <',alpha/(-np.log(epsilon)))
print ((beta - const)/(-np.log(epsilon)),'< (n2/n) <',beta/(-np.log(epsilon)))


# Then we can conclude that Public leaderboard consist of 17.5% of positive data point and 82.5% of negative data point.

# Now can we estimate the exact number of positive/negative data point. In order word, can we know how many computer generated data point in test set ??
# To be continue.... 
# 
# Peace :)
