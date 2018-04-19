
# coding: utf-8

# # OpenMP Performance Sensitivity to the Number of Threads Used
# 
# ## TL;DR  
# 
# *When using Kaggle Kernels, be aware that Kaggle's backend environment reports more CPUs than are available to your Kernel's execution. Some commonly used libraries default values for performance tuning parameters are **significantly sub-optimal** in this environment.  In particular, when using `xgboost` or other `OpenMP` based libraries, we suggest you start your Python kernels with `import os ; os.environ['OMP_NUM_THREADS'] = '4'`, prior to any other import.  Similarly for R, use `library(OpenMPController)` followed by `omp_set_num_threads(4)`.*
# 
# ## Full Version
# 
# As covered in our [blog](http://blog.kaggle.com/2017/09/21/product-launch-amped-up-kernels-resources-code-tips-hidden-cells/), an individual Kernel executing on Kaggle gets 4 logical CPUs worth of compute.
# 
# However, currently (and subject to change), a Kaggle Kernel actually runs on a Google Compute Engine (GCE) VM which has 32 logical CPUs.  That is the CPU count you will see when you query the system, e.g. using the `multiprocessing` Python library:

# In[ ]:


import multiprocessing
print(multiprocessing.cpu_count())


# While the system has 32 CPUs and your Kernel may use all 32 of these within a given Kernel session, the Docker container the Kernel runs in is limited to use only 4 CPUs worth of time over an arbitrary (but short) period of time.
# 
# **In summary, while your Kernel only effectively can use 4 CPUs, the system reports the presence of 32 CPUs.**
# 
# ## Why does this matter?
# 
# Many computation libraries allow you to  benefit from parallelism by making use of multiple logical CPUs when these are available to you.  When a CPU-bound computation can be executed in parallel (from the type of algorithm involved and the capabilities of the implementation) and can benefit from it (the inputs are large enough that the implied fixed-cost overhead is largely amortized), you would typically get the best performance (smallest running time) by using as many computation threads as you have logical CPUs available.  Most of these libraries will enable parallelism when these conditions are met, use that most optimal thread count, by default.
# 
# Let's take the example of `xgboost`, a popular library that provides gradient boosted decision trees, and in particular in this notebook, its Python interface.
# 
# Our `xgboost` installation is built with [GNU OpenMP](https://gcc.gnu.org/onlinedocs/libgomp) (a.k.a. GOMP) support to handle the parallelism aspects.  Let's find out what GOMP's configuration looks like in Kaggle Kernels' runtime:

# In[ ]:


# We can use the OMP_DISPLAY_ENV environment variable to have GOMP output its
# configuration to stderr. That is done from C++ (libgomp) by operating on the
# process's stderr file descriptor. Jupyter doesn't properly intercept operations
# on that file descriptor (it only intercepts the Python space `sys.stderr`).
# So let's do that here first:
import os
import sys

stderr_fileno = 2  # sys.stderr.fileno(), if Jupyter hadn't messed with it.
rpipe, wpipe = os.pipe()
os.dup2(wpipe, stderr_fileno)

os.environ['OMP_DISPLAY_ENV'] = 'TRUE'
import xgboost as xgb
os.close(stderr_fileno)
os.close(wpipe)
print(os.read(rpipe, 2**10).decode())


# You can find the meaning of each of these environment variables in [GOMPs documenation](https://gcc.gnu.org/onlinedocs/libgomp/Environment-Variables.html).
# 
# A relevant one to discuss though, is `OMP_NUM_THREADS`.  As the output shows, it defaulted to `32`, which is our system's CPU count.  Let's see how this setting fares in a simple training of the Mercari dataset:

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import xgboost as xgb


# In[ ]:


train_dataset = pd.read_csv('../input/train.tsv', sep='\t')
ignore_columns = (
    'name', 'item_description', 'brand_name', 'category_name', 'train_id', 'test_id', 'price')
train_columns = [c for c in train_dataset.columns if c not in ignore_columns]
dtrain = xgb.DMatrix(train_dataset[train_columns], train_dataset['price'])
params = {'silent': 1}
num_rounds = 100

def train(params, dtrain, num_rounds):
    """Returns the duration to train a model with given parameters."""
    start_time = time.time()
    xgb.train(params, dtrain, num_rounds, [(dtrain, 'train')], verbose_eval=False)
    return time.time() - start_time


# In[ ]:


print('duration: %.2fs' % train(params, dtrain, num_rounds))


# Let's see how this looks when we use only 4 threads, which is how many CPUs we actually can use in the Kaggle Kernels environment.
# 
# `OMP_NUM_THREADS` can no longer be used to influence the number of threads used at this point, as it is only read once by `xgboost` and/or GOMP.   We can however set the `xgboost` `nthread` parameter to have it reconfigure this at run-time:

# In[ ]:


params['nthread'] = 4
print('duration: %.2fs' % train(params, dtrain, num_rounds))


# The difference is very significant, and suprisingly [x] so:  While spawning more threads than there are CPUs available isn't helpful and causes multiple threads to be multiplexed on a single CPU, it is unclear why that overhead causes `xgboost` to perform slower by several multiples.
# 
# *[x] Not to you?  Please let me know in comments what you see the cause for this might be!*
# 
# Here is some more data:
# 

# In[ ]:


results = pd.DataFrame()
for nthread in (1, 2, 4, 8, 16, 32):
    params['nthread'] = nthread
    durs = []
    for i in range(16):
        durs.append(train(params, dtrain, num_rounds))
    results[nthread] = durs
    print('nthread = %d, durations = %s' % (nthread, ', '.join(['%.2fs' % d for d in durs])))


# In[ ]:


fig = plt.figure(figsize=(12, 6))
plt.errorbar(results.columns, results.mean(), yerr=results.std(), linestyle='-', fmt='o', ecolor='g', capthick=2)
plt.xlabel('nthread', fontsize=18)
plt.ylabel('duration (s)', fontsize=18)
plt.figsize=(12, 6)

