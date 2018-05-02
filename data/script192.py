
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


# This notebook was inspired by @kilian's excellent notebook: https://www.kaggle.com/batzner/gini-coefficient-an-intuitive-explanation, especially the alternative way of computing gini at the end.  I simply implemented the computation directly.  The code uses Numba to make it run fast.

# In[2]:


from numba import jit

@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


# How efficient is the above code?  Let's compare it with code taken from another excellent notebook by Mohsin Hasan: https://www.kaggle.com/tezdhar/faster-gini-calculation . His code is reproduced below.

# In[3]:


#Remove redundant calls
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


# Let's create a random test set roughly as large as the train data set.

# In[4]:


a = np.random.randint(0,2,600000)
p = np.random.rand(600000)
print(a[10:15], p[10:15])


# As sanity check, let's compare the output of the tow methods.

# In[5]:


gini_normalizedc(a, p)


# In[6]:


eval_gini(a, p)


# In[7]:


gini_normalizedc(a, p) - eval_gini(a, p)


# Looks fine, difference is negligible.  Let's time them now.

# In[8]:


get_ipython().run_cell_magic('timeit', '', 'gini_normalizedc(a,p)')


# In[9]:


get_ipython().run_cell_magic('timeit', '', 'eval_gini(a,p)')


# OK, the speedup is not that large, but there is a speedup still.  Note that my code only handles binary values for y_true while Mohsin's code is more general.

# The speedup looks better if we factor the sorting time out.  Let's measure it.

# In[10]:


get_ipython().run_line_magic('timeit', 'np.argsort(p)')


# Further improvements would have to come from the sort algorithm part.

#  Motivation for writing this code was to understand what gini really is, but I'll be happy if some readers find it useful too.  Please upvote (button at top right) if this is the case.
