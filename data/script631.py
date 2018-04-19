
# coding: utf-8

# This notebook was motivated by this great [notebook](https://www.kaggle.com/yuliagm/be-careful-about-ips-as-a-signal/notebook) from @yulia .  She discuss there a potential difference between train and test data.    Here is a simple way to see that difference.

# In[12]:


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

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's load some data.

# In[2]:


dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

train = pd.read_csv('../input/train.csv', dtype=dtypes, usecols=['ip', 'is_attributed'])
test = pd.read_csv('../input/test.csv', dtype=dtypes, usecols=['ip'])


# Let's now look at the downlad rate per ip in the train data.

# In[3]:


df = train.groupby('ip').is_attributed.mean().to_frame().reset_index()

df.head()


# One way to display numeric data and spot trend is to use a moving average.  Let's try it here.

# In[6]:


df['roll'] = df.is_attributed.rolling(window=1000).mean()
plt.plot(df.ip, df.roll)


# There is a clear cut split around 130,000, and a lighter split around 220,000.  Below the first split ip have about 0.02 app download rate, then the rate climbs above 0.25.  This is a 10x increase, worth eploring further.
# 
# A little trial and error leads to an identification of where the split is.

# In[14]:


df1 = df[(df.ip >= 120000) & (df.ip <= 130000)]
plt.plot(df1.ip, df1.roll)


# In[15]:


df1 = df[(df.ip >= 126000) & (df.ip <= 126700)]
plt.plot(df1.ip, df1.roll)


# In[16]:


df1 = df[(df.ip >= 126400) & (df.ip <= 126500)]
plt.plot(df1.ip, df1.roll)


# In[17]:


df1 = df[(df.ip >= 126415) & (df.ip <= 126425)]
plt.plot(df1.ip, df1.roll)


# The rate starts to climb at 126420.
# 
# Let's now look at where the ips present in test are compared to that split:

# In[9]:


test.ip.max()


# All the test ip are below the split!
# 
# I thought this was significant, which is why I share it.
