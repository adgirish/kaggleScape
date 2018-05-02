
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# Let's understand how to minimize smape with constant values, like what is done in the best public kernels.  Is the median the right value?
# First let us define the smape function.  It is quite straightforward, the only caveat is to treat nan correctly.  Thanks to the official answers on the forum, we know we can use this code.  It handles the case where there are nan in the y_true array, but it assumes there are no nan in the y_pred array.
# 

# In[ ]:


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)


# Let's start with a simple example where the y_true series to predict contains only one point, say 3.

# In[ ]:


y_true = np.array(3)
y_pred = np.ones(1)
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)


# We see that SMAPE is 0 when the predicted value is equal to the true value.  We also see that an under estimate is penalized more than an over estimate.  Last, we see that the function is not convex for values above the true value.  This may lead to many local minima.  Let's see a second example to check for this, with two values 1 and 9 in the series we try to predict.

# In[ ]:


y_true = np.array([1,9])
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])
print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 
      ' at median %0.2f' % np.nanmedian(y_true))


# In this case there are two global minima with SMAPE = 80 for the two cases where our constant prediction is equal to one of the value in the true series.  The function reaches a local maxima with SMAPE = 100 for y_pred = 3.  And the value of the median (y_pred = 5) is about 95.24, i.e. it is significantly higher than the global minima.

# Does this mean that SMAPE is impossible to optimize?  Let's see if we have more points in our y_true series, for instance a uniformly sampled series:

# In[ ]:


np.random.seed(0)
y_true = np.random.uniform(1, 9, 100)
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])
print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 
      ' at median %0.2f' % np.nanmedian(y_true))


# We see that the minimum of the smape function is met near the median which is good.  It could explain why the public kernels do well.  Let's see with a skewed distribution.

# In[ ]:


np.random.seed(0)
y_true = np.random.lognormal(1, 1, 100)
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])
print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 
      ' at median %0.2f' % np.nanmedian(y_true))


# Here again the median does well.

# Wait a minute.  What is one or more values in the series are 0?   Let's start with one zero only.

# In[ ]:


y_true = np.array([0])
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)


# The function is discontinue at 0.  It is equal to 200 everywhere except at 0 where it equals 0.  Let's now look at two values.

# In[ ]:


np.random.seed(0)
y_true = np.array([0,9])
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])
print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 
      ' at median %0.2f' % np.nanmedian(y_true))


# Here we have two local minima at 100 when y_pred equals one of the values in the y_true series.   There is a discontinuity at 0. The mathematical limit of the series SMAPE(0, x) when x tends to 0 is 200.  It means that there is no local maxima near 0 as the value 200 cannot be reached.  But we can get values as close as we want to 200.

# What if we have more values in the series, but still a proportion of 0?  

# In[ ]:


np.random.seed(0)
y_true = np.random.lognormal(1, 1, 100)
y_true[y_true < 3] = 0
print('There are %d zeros in the series' % np.sum(y_true == 0))
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])
print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 
      ' at median %0.2f' % np.nanmedian(y_true))


# We see that the median is really not a good choice here and that 0 would be way better.  Moreover, a gradient descent from anywhere except 0 will miss the global minima, by large.

# I hope this notebook shows that the discontinuity at 0 makes it tricky to optimize SMAPE with constant predictions.  If you like it then please upvote it (button at the top left).
