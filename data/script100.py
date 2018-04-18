
# coding: utf-8

# # How to Work with BIG Datasets on Kaggle Kernels (16G RAM)
# 
# This particular competition is asking us to look and analyze some really big data sets.  In its given form, it won't even load into pandas on the kaggle kernels.  If you don't have a very fancy computer, it probably won't load on yours either.  At least on my laptop (macbook, 16G RAM) it won't.
# 
# I am assuming I'm not the only one with limited computing power, and limited budget for running paid cloud setup.  So I've been looking into different methods to work with big data on limited resources.
# 
# Below are some tips I  collected while learning to get by with mostly Kaggle kernels while not overloading its allocated RAM.  
# 
# These tips are probably more useful for  begginners and intermediate users, but if you are an expert and know of better ways, please share and I'll be happy to update the book!

# # OUTLINE
# (i do intend to make this a linked heading at some point...)
# 
# - ** TIP 1  - Deleting unused variables and gc.collect() **
# - **TIP 2 - Presetting the datatypes**
# - ** TIP 3 - Importing selected rows of the a file (including generating your own subsamples)**
# - **TIP 4 - Importing in batches and processing each individually**
# - **TIP 5 - Importing just selected columns**
# - ** TIP 6 - Creative data processing**
# - ** TIP 7 - Using Dask **
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import datetime
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import gc
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#make wider graphs
sns.set(rc={'figure.figsize':(12,5)});
plt.figure(figsize=(12,5));


# ## TIP # 1 Deleting unused variables and gc.collect() 
# 
# The thing about python is that once it loads something into RAM it doesn't really get rid of it effectively.  So if you load a huge dataframe into pandas, and then make a copy of it and never use it again, that original dataframe will still be in your RAM.  Eating away at your memory.   Same goes for any other variables you create.
# 
# Therefore if you used up a dataframe (or other variable), get in the habit of deleting it.  
# 
# For example, if you create a dataframe  `temp`, extract some features and merge results to your main training set, `temp` will still be eating up space.  You need to explicitely delete it by stating `del temp`.  You also need to make sure that nothing else is referring to `temp` (you don't have any other variables bound to it).
# 
# Even after doing so there may still be residual memory usage going on.
# 
# That's where the garbage collection module comes in.   `import gc` at the beginning of your project, and then each time you want to clear up space put command `gc.collect()` .  
# 
# It also helps to run `gc.collect()` after multiple transformations/functions/copying etc...  as all the little references/values accumulate.

# In[ ]:


# eg:
#import some file
temp = pd.read_csv('../input/train_sample.csv')

#do something to the file
temp['os'] = temp['os'].astype('str')


# In[ ]:


#delete when no longer needed
del temp
#collect residual garbage
gc.collect()


# ## TIP # 2   Presetting the datatypes
# If you import data into CSV, python will do it's best to guess the datatypes, but it will tend to error on the side of allocating more space than necessary.
# So if you know in advance that your numbers are integers, and don't get bigger than certain values, set the datatypes at minimum requirements before importing.

# In[ ]:


dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }

train = pd.read_csv('../input/train_sample.csv', dtype=dtypes)

#check datatypes:
train.info()


# ## TIP # 3 Importing selected rows of a csv file

# ### a) Select number of rows to import
# Instead of the default  `pd.read_csv('filename') ` you can use parameter `nrows` to specify number of rows to import.  For exampe:
# `train = pd.read_csv('../input/train.csv', nrows=10000)` will only read the first 10000 rows (including the heading)..

# In[ ]:


train = pd.read_csv('../input/train.csv', nrows=10000, dtype=dtypes)
train.head()


# ### b)  Simple row skip (with or without headings)
# You can also specify number of rows to skip (`skiprows`) , if you, for example want 1 million rows after the first 5 million:
# `train = pd.read_csv('../input/train.csv', skiprows=5000000, nrows=1000000)`.  This however will ignore the first line with headers.  Instead you can pass in range of rows to skip, that will not include the first row  (indexed `[0]`).

# In[ ]:


#plain skipping looses heading info.  It's OK for files that don't have headings, 
#or dataframes you'll be linking together, or where you make your own custom headings...
train = pd.read_csv('../input/train.csv', skiprows=5000000, nrows=1000000, header = None, dtype=dtypes)
train.head()


# In[ ]:


#but if you want to import the headings from the original file
#skip first 5mil rows, but use the first row for heading:
train = pd.read_csv('../input/train.csv', skiprows=range(1, 5000000), nrows=1000000, dtype=dtypes)
train.head()


# ### c) Picking wich rows to skip  (Make a list of what you DON'T want)
# 

# ** This is how you can do your own random sampling**
# 
# Since 'skiprows' can take in a list of rows you want to skip, you can make a list of random rows you want to input.   I.e. you can sample your data anyway you like!
# 
# Recall how many rows the train set in TalkingData has:

# In[ ]:


import subprocess
#from https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python , Olafur's answer
def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

lines = file_len('../input/train.csv')
print('Number of lines in "train.csv" is:', lines)


# Let's say you want to pull a random sample of 1 million lines out of the total dataset.  That means that you want a list of `lines - 1 - 1000000` random numbers ranging from 1 to 184903891. 
# 
# Note: generating such long list also takes a lot of space and  some time.  Be patient and make sure to use del and gc.collect() when done!

# In[ ]:


#generate list of lines to skip
skiplines = np.random.choice(np.arange(1, lines), size=lines-1-1000000, replace=False)

#sort the list
skiplines=np.sort(skiplines)


# In[ ]:


#check our list
print('lines to skip:', len(skiplines))
print('remaining lines in sample:', lines-len(skiplines), '(remember that it includes the heading!)')

###################SANITY CHECK###################
#find lines that weren't skipped by checking difference between each consecutive line
#how many out of first 100000 will be imported into the csv?
diff = skiplines[1:100000]-skiplines[2:100001]
remain = sum(diff!=-1)
print('Ratio of lines from first 100000 lines:',  '{0:.5f}'.format(remain/100000) ) 
print('Ratio imported from all lines:', '{0:.5f}'.format((lines-len(skiplines))/lines) )


# Now let's import the randomly selected 1million rows

# In[ ]:


train = pd.read_csv('../input/train.csv', skiprows=skiplines, dtype=dtypes)
train.head()


# Delete the now useless list and any other garbaged generated along the way

# In[ ]:


del skiplines
gc.collect()


# And inspect our resulting table.  But first convert time stamps to timeseries type.

# In[ ]:


train['click_time'] = pd.to_datetime(train['click_time'])
train['attributed_time'] = pd.to_datetime(train['attributed_time'])


# In[ ]:


train.describe(include='all')


# In my previous notebook (https://www.kaggle.com/yuliagm/talkingdata-eda-plus-time-patterns) we found that the data is organized by click time.  Therefore if our random sampling went according to plan, the resulting set should roughly span the full time period and mimick the click pattern. 
# 
# We see from above that first and last click span the 4 day period.
# 
# Let's try a chart to see if the pattern looks consistent:

# In[ ]:


#round the time to nearest hour
train['click_rnd']=train['click_time'].dt.round('H')  

#check for hourly patterns
train[['click_rnd','is_attributed']].groupby(['click_rnd'], as_index=True).count().plot()
plt.title('HOURLY CLICK FREQUENCY');
plt.ylabel('Number of Clicks');

train[['click_rnd','is_attributed']].groupby(['click_rnd'], as_index=True).mean().plot()
plt.title('HOURLY CONVERSION RATIO');
plt.ylabel('Converted Ratio');


# Looks all-right!  
# 
# Now you can analyze your own subsample and run models on it.

# ## TIP #4   Importing in batches and processing each individually

# We know that the proportion of clicks that was attributed is very low.  So let's say we want to look at all of them at the same time.  We don't know what rows they are, and we can't load the whole data and filter.  But we can load in chuncks, extract from each chunk what we need and get rid of everything else!
# 
# The idea is simple.  You specify size of chunk (number of lines) you want pandas to import at a time.  Then you do some kind of processing on it.  Then pandas imports the next chunk, untill there are no more lines left.
# 
# So below I import one million rows, extract only rows that have 'is_attributed'==1 (i.e. app was downloaded) and then merge these results into common dataframe for further inspection.

# In[ ]:


#set up an empty dataframe
df_converted = pd.DataFrame()

#we are going to work with chunks of size 1 million rows
chunksize = 10 ** 6

#in each chunk, filter for values that have 'is_attributed'==1, and merge these values into one dataframe
for chunk in pd.read_csv('../input/train.csv', chunksize=chunksize, dtype=dtypes):
    filtered = (chunk[(np.where(chunk['is_attributed']==1, True, False))])
    df_converted = pd.concat([df_converted, filtered], ignore_index=True, )


# Let's see what we've got:

# In[ ]:


df_converted.info()


# In[ ]:


df_converted.head()


# Perfect!  Now we know that in our entire dataset there are 456846 samples of conversions.  We can explore this subset  for patterns, anomalies, etc...  
# 
# Using analogous method you can explore specific ips, apps, devices, etc  combinations, devices, etc...   

# ## TIP #5 Importing just selected columns
# 
# If you want to analyze just some specific feature, you can import just the selected columns.
# 
# For example, lets say we want to analyze clicks by ips.  Or conversions by ips.
# 
# Importing just 2 fields as opposed to full table just may fit in our RAM

# In[ ]:


#wanted columns
columns = ['ip', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'is_attributed' : 'uint8',
        }

ips_df = pd.read_csv('../input/train.csv', usecols=columns, dtype=dtypes)


# Let's see what we've got.

# In[ ]:


print(ips_df.info())
ips_df.head()


# The dataframe is big (over 2G, but manageable).
# 
# Now let's say you want to generate counts of ips frequencies (maybe to use as a feature in a model).
# 
# The regular groupby method will crush the system.  So you'd have to be creative about how to do it.  
# 
# Which is what the next section is about:

# ## Tip #6  Creative data processing
# 
# The kernel cannot handle groupby on the whole dataframe.  But it can do it in sections.  For example:

# In[ ]:


#processing part of the table is not a problem
ips_df[0:100][['ip', 'is_attributed']].groupby('ip', as_index=False).count()[:10]


# So you can calculate counts in batches, merge them and sum up to total counts.
# 
# (Takes a bit of  time, but works)

# In[ ]:


size=100000
all_rows = len(ips_df)
num_parts = all_rows//size

#generate the first batch
ip_counts = ips_df[0:size][['ip', 'is_attributed']].groupby('ip', as_index=False).count()

#add remaining batches
for p in range(1,num_parts):
    start = p*size
    end = p*size + size
    if end < all_rows:
        group = ips_df[start:end][['ip', 'is_attributed']].groupby('ip', as_index=False).count()
    else:
        group = ips_df[start:][['ip', 'is_attributed']].groupby('ip', as_index=False).count()
    ip_counts = ip_counts.merge(group, on='ip', how='outer')
    ip_counts.columns = ['ip', 'count1','count2']
    ip_counts['counts'] = np.nansum((ip_counts['count1'], ip_counts['count2']), axis = 0)
    ip_counts.drop(columns=['count1', 'count2'], axis = 0, inplace=True)


# In[ ]:


#see what we've got:
ip_counts.head()


# Sort highest to lowest

# In[ ]:


ip_counts.sort_values('counts', ascending=False)[:20]


# Check that the sum of counts adds up to the total number of values:

# In[ ]:


np.sum(ip_counts['counts'])


# Let's say you wanted proportion of conversions by ip, get the sums of conversions and then devide them by counts...

# In[ ]:


size=100000
all_rows = len(ips_df)
num_parts = all_rows//size

#generate the first batch
ip_sums = ips_df[0:size][['ip', 'is_attributed']].groupby('ip', as_index=False).sum()

#add remaining batches
for p in range(1,num_parts):
    start = p*size
    end = p*size + size
    if end < all_rows:
        group = ips_df[start:end][['ip', 'is_attributed']].groupby('ip', as_index=False).sum()
    else:
        group = ips_df[start:][['ip', 'is_attributed']].groupby('ip', as_index=False).sum()
    ip_sums = ip_sums.merge(group, on='ip', how='outer')
    ip_sums.columns = ['ip', 'sum1','sum2']
    ip_sums['conversions_per_ip'] = np.nansum((ip_sums['sum1'], ip_sums['sum2']), axis = 0)
    ip_sums.drop(columns=['sum1', 'sum2'], axis = 0, inplace=True)


# In[ ]:


ip_sums.head(10)


# In[ ]:


#check proportion (we calculated earlier how many rows of data had conversions)
np.sum(ip_sums['conversions_per_ip'])/184900000


# What if we want proportion of conversions to click per ip?  

# In[ ]:


ip_conversions=ip_counts.merge(ip_sums, on='ip', how='left')
ip_conversions.head()


# In[ ]:


ip_conversions['converted_ratio']=ip_conversions['conversions_per_ip']/ip_conversions['counts']
ip_conversions[:10]


# In[ ]:


#some cleanup
del ip_conversions
del ip_sums
del ips_df
del df_converted
del train
gc.collect()


# ## TIP #7 Using Dask
# 
# ### DASK
# 
# **What is it?**
# - A python library for parallel computing that can work on a single notebook or large cluster.
# 
# **What does it do?**
# - it parallizes NumPy and Pandas
# - makes it possible to work on larger-than-memory datasets
# - in case of single machine uses its own task scheduler to get the most out of your machine  (kaggle kernels are multicore, so there is definetly room for improvement)
# - BASICALLY IT WILL MAKE SOME COMPUTATIONS FIT RAM, AND WILL DO IT FASTER
# 
# **Its limitations?**
# - it's still relativelty early in development
# - it doesn't have all of Panda's options/functions/features. Only the most common ones.
# - many operations that require setting the index are still computationally expensive 
# 
# First you'll need to import the library.

# In[ ]:


import dask
import dask.dataframe as dd


# There are different sections to Dask, but for this case you'll likely just use** Dask DataFrames**.
# 
# Here are some basics from the developers:
# 
# > A Dask DataFrame is a large parallel dataframe composed of many smaller Pandas dataframes, split along the index. These pandas dataframes may live on disk for larger-than-memory computing on a single machine, or on many different machines in a cluster. One Dask dataframe operation triggers many operations on the constituent Pandas dataframes.
# 
# (https://dask.pydata.org/en/latest/dataframe.html)
# 
# For convenience and Dask.dataframe copies the Pandas API.  Thus commands look and feel familiar.
# 
# **What DaskDataframes can do?**
# -they are very fast on most commonly used set of Pandas API
# <br><br>
# *below is taken directly from: https://dask.pydata.org/en/latest/dataframe.html *
# <br>
# 
# ### Trivially parallelizable operations (fast):
# - Elementwise operations: `df.x + df.y, df * df`
# - Row-wise selections: `df[df.x > 0]`
# - Loc: `df.loc[4.0:10.5]`
# - Common aggregations: `df.x.max(), df.max()`
# - Is in: `df[df.x.isin([1, 2, 3])]`
# - Datetime/string accessors: `df.timestamp.month`
#  
# <br>
# ### Cleverly parallelizable operations(fast):
# - groupby-aggregate (with common aggregations): `df.groupby(df.x).y.max(), df.groupby('x').max()`
# - groupby-apply on index: `df.groupby(['idx', 'x']).apply(myfunc)`, where `idx` is the index level name
# - value_counts: `df.x.value_counts()`
# - Drop duplicates: `df.x.drop_duplicates()`
# - Join on index: `dd.merge(df1, df2, left_index=True, right_index=True`) or `dd.merge(df1, df2, on=['idx', 'x'])` where `idx` is the index name for both `df1` and `df2`
# - Join with Pandas DataFrames: `dd.merge(df1, df2, on='id')`
# - Elementwise operations with different partitions / divisions:` df1.x + df2.y`
# - Datetime resampling: `df.resample(...)`
# - Rolling averages: `df.rolling(...)`
# - Pearson Correlations: `df[['col1', 'col2']].corr()`
# 
# <br>
# ### Notes/observations:
# - To actually get results of many of the above functions you have to add `.compute()` at the end.  eg, for value_counts would be: `df.x.value_counts().compute()`.  This hikes up RAM use a lot.  I believe it's because `.compute()` gets the data into pandas format, with all the accompanying overhead.  (Please correct me if wrong).
# 
# - I've been playing with dask for the past little while here on Kaggle Kernels, and while they can load full data and do some nice filtering, many actual operations do hike up RAM to extreme and even crush the system.  For example, after loading 'train' dataframe, just getting `len(train)` hiked RAM up to 9GB.  So be careful...  Use a lot of `gc.collect()` and other techniques for making data smaller.  So far I find dask most useful for filtering (selecting rows with specified features).
# 
# <br><br>**Now let's see some examples.**
# 
# First, let's load the big train data:

# In[ ]:


# Loading in the train data
dtypes = {'ip':'uint32',
          'app': 'uint16',
          'device': 'uint16',
          'os': 'uint16',
          'channel': 'uint16',
          'is_attributed': 'uint8'}

train = dd.read_csv("../input/train.csv", dtype=dtypes, parse_dates=['click_time', 'attributed_time'])
train.head()


# In[ ]:


train.info()


# In[ ]:


len(train)


# In[ ]:


train.columns


# Let's see how it works for selecting data subsets:

# In[ ]:


#select only rows 'is_attributed'==1
train[train['is_attributed']==1].head()


# In[ ]:


#select only data attributed after 2017-11-06 
train[train['attributed_time']>='2017-11-07 00:00:00'].head()


# Now we'll do some heavier lifting.<br>
# 
# Let's get counts by ip

# In[ ]:


ip_counts = train.ip.value_counts().compute()
ip_counts[:20]


# In[ ]:


#clean up to free up space
#for future work, you can export data you generated to CSVs so you don't have to make it
#all over again
del ip_counts
gc.collect()


# Let's now try to get mean conversion by channel:

# In[ ]:


channel_means = train[['channel','is_attributed']].groupby('channel').mean().compute()
channel_means[:20]


# Unfortunately `as_index=False` does not appear to be implemented in Dask yet...  You'd have to do manual manipulation to get the channel into a column...  For example like this:
# 

# In[ ]:


channel_means=channel_means.reset_index()
channel_means[:20]


# *** To be continued....***

# As you can see, not all is lost if your computing resources are limited..  By combining/manipulating these and other methods, you can get quite a lot out of this data.  You can run models on subsamples, generate features for categories, etc...
# 
# You can also do analysis and preprocessing prep here, and then run the real big models on a cloud.  It will save you money, not having to do all the work on a dollar counter.
# 
# Of course all these tricks are time consuming and require lots of extra effort.  Naturally life would be easier if you had a fancy computer, or endless free cloud access.  But the reality for some of us is that we just don't...   
# 
# I hope this helps those of you who are limited in resources, but still want to learn and explore the data.
