
# coding: utf-8

# # The Problem:
# #### The training data, `train.csv` is a large file (~5 GB) - this can be problematic if you have relatively low RAM (8 GB).
# # The Solution:
# ## Set `low_memory=True` in Pandas' `read_csv`
# On a machine with relatively low RAM, attempting to load the entire file in a pandas DataFrame can lead to failure caused by running out of memory.  One way of fixing this issue is to make use of the `low_memory=True` argument of `read_csv`.  With this method, the csv file is processed in chunks requiring lower memory usage, while at the same time reading the csv's contents into a single DataFrame.
# ## But the Jupyter kernel still keeps restarting even with `low_memory=True`....why?
# The dtypes of the columns of the DataFrame must be specified in `read_csv` if we wish to set `low_memory=True`.  This is because not specifying dtypes forces pandas to guess column dtypes - which is a memory-intensive task.  Please see this Stack Overflow answer for a additional explanation:
# https://stackoverflow.com/a/27232309
# 
# # The Complete Solution
# We first create a new file called `small_train.csv` using only the first row of data from `train.csv`:
# 

# In[ ]:


get_ipython().system('head -2 train.csv > small_train.csv')


# In[ ]:


import pandas as pd


# In[ ]:


small_train = pd.read_csv('../input/small-train/small_train.csv')
print(small_train)


# In[ ]:


types_dict = small_train.dtypes.to_dict()
types_dict


# Next, let's update types of some columns to make them more memory efficient.  This is based on information shared in the following kernel:
# 
# https://www.kaggle.com/jagangupta/memory-optimization-and-eda-on-entire-dataset
# 
# I highly recommend looking at the link above - it shows additonal steps for making your dataframe even more memory efficient.

# In[ ]:


types_dict = {'id': 'int32',
             'item_nbr': 'int32',
             'store_nbr': 'int8',
             'unit_sales': 'float32'}


# Now, we can use `types_dict` to specify the dtypes of each column of the DataFrame we are loading the `train.csv` file into:

# In[ ]:


grocery_train = pd.read_csv('train.csv', low_memory=True, dtype=types_dict)


# The steps above will let you load the entire 5 GB file in memory without crashing the Jupyter kernel.

# # Feather Format: Quickly Reloading Saved Training Dataframe

# Every time you reopen your Jupyter notebook, you need not rerun the steps shown in the previous section.  Instead, simply use the **feather** format to save the `grocery_train` dataframe after you load it in memory the first time.  The feather format enables very fast read and write access for working with dataframes, both in `R` and `Python` (read more here: https://blog.rstudio.com/2016/03/29/feather/).  Note that your pandas version must be 0.20.0 or newer for the code below to work.

# In[ ]:


os.makedirs('tmp', exist_ok=True)  # Make a temp dir for storing the feather file
# Save feather file, requires pandas 0.20.0 at least:
grocery_train.to_feather('./tmp/grocery_train_raw')


# ### Going forward, you can read the `grocery_train` dataframe directly from the feather file as shown below:

# In[ ]:


grocery_train = pd.read_feather('./tmp/train_sub_raw')

