
# coding: utf-8

# **Introduction**
# This kernel is a simple playing around with the quora datasets

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn'

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print(train_df.shape)

print(test_df.shape)

# Any results you write to the current directory are saved as output.


# let's view the top 5 lines of train_df

# In[ ]:


train_df.head()


# Now let's view the top few lines of test_df

# In[ ]:


test_df.head()


# **Target Variable Exploration**

# In[ ]:


is_dup = train_df['is_duplicate'].value_counts()

plt.figure(figsize=(8,4))

sns.barplot(is_dup.index, is_dup.values, alpha=0.8, color=color[1])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Is Duplicate', fontsize=12)

plt.show()


# Let's check the number of non duplicates vs the number of duplicates

# In[ ]:


is_dup


# In percentage:

# In[ ]:


is_dup / is_dup.sum()

