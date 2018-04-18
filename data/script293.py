
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[6]:


import pytz
import gc
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


tr_s = 300000
te_s = 100000
nrows=None

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

tr = pd.read_csv('../input/train.csv', dtype=dtypes, usecols=['ip', 'is_attributed', 'click_time'], nrows=nrows).sample(tr_s)
gc.collect()
te = pd.read_csv('../input/test_supplement.csv', dtype=dtypes, usecols=['ip', 'click_time'], nrows=nrows).sample(te_s)
all_df = tr.append(te)
gc.collect()

cst = pytz.timezone('Asia/Shanghai')
all_df['click_time'] = pd.to_datetime(all_df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
all_df['day'] = all_df.click_time.dt.day.astype('uint8')

all_df['day7'] = all_df.day == 7 # 1'st day
all_df['day8'] = all_df.day == 8 # 2'nd day
all_df['day9'] = all_df.day == 9 # 3'rd day
all_df['day_test'] = all_df.day == 10 # 4'th day(test)


# In[7]:


def print_count(df, tgt):
    df = df[['ip', tgt]].groupby('ip')[tgt].sum().to_frame().reset_index()
    df[tgt+'_count'] = df[tgt].rolling(window=1000).mean()
    plt.plot(df.ip, df[tgt+'_count'])


# In[8]:


print_count(all_df,'day_test')


# In[9]:


print_count(all_df,'day7')


# In[10]:


print_count(all_df,'day8')


# In[11]:


print_count(all_df,'day9')


# Endoding should be done by day10(test) -> day7 -> day8 -> day9.
# 
# IPs between 130,000 and 220,000 did not appear on the 10th (test), but it appeared on the 7th.
# 
# So, it is no wonder that these IPs are different from the behavior of IPs that appeared on the 10th.
# 
# Also, it is not surprising that CVR differs in these groups as follows.

# In[13]:


def print_attr(df):
    df = df[['ip', 'is_attributed']].groupby('ip').is_attributed.mean().to_frame().reset_index()
    df['roll'] = df.is_attributed.rolling(window=1000).mean()
    plt.plot(df.ip, df.roll)

print_attr(all_df[all_df.day == 7])
print_attr(all_df[all_df.day == 8])
print_attr(all_df[all_df.day == 9])

