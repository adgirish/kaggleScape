
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dtypes = {'store_nbr': np.dtype('int64'),
          'item_nbr': np.dtype('int64'),
          'unit_sales': np.dtype('float64'),
          'onpromotion': np.dtype('O')}

train = pd.read_csv('../input/train.csv', index_col='id', parse_dates=['date'], dtype=dtypes)

# If done on all train data, results in 367m rows. So, we're taking a small sample:
date_mask = (train['date'] >= '2016-08-28') & (train['date'] <= '2016-08-31')
print(train.shape)
train = train[date_mask]
print(train.shape)


# In[ ]:


orig_train_index = train.index # will need later

# Bracket the dates
max_date = train['date'].max()  
min_date = train['date'].min()  
days = (max_date - min_date).days + 1 

# Master list of dates
dates = [min_date + datetime.timedelta(days=x) for x in range(days)]
dates.sort()

# Master list of stores
unique_stores = list(set(train['store_nbr'].unique())) # | set(test['Store'].unique()))
unique_stores.sort()
num_unique_stores = len(unique_stores)

# Master list of Items
unique_items = list(set(train['item_nbr'].unique())) # | set(test['item_nbr'].unique()))
unique_items.sort()
num_unique_items = len(unique_items)

# Unique Date / Store index
date_index = np.repeat(dates, num_unique_stores * num_unique_items) # num dates * num stores * num items
store_index = np.concatenate([np.repeat(unique_stores, num_unique_items)] * days)
item_index = np.concatenate([unique_items] * days * num_unique_stores)

print(len(date_index))
print(len(store_index))
print(len(item_index))

start = train.index.tolist()[0]
new_train_index = list(range(len(item_index)))
new_train_index = new_train_index + start

train_new = pd.DataFrame(index=new_train_index, columns=train.columns)

train_new['date'] = date_index
train_new['store_nbr'] = store_index
train_new['item_nbr'] = item_index

train_new.index.name = 'id'

# Set the indexes (makes it easy to insert data into new)
train_new.set_index(['date', 'store_nbr', 'item_nbr'], drop=True, inplace=True)
train.set_index(['date', 'store_nbr', 'item_nbr'], drop=True, inplace=True)

# Update the master index with and train
train_new.update(train)
train_new.reset_index(inplace=True)

# Return the original train back to normal
train.reset_index(inplace=True)
train.set_index(orig_train_index, inplace=True)

# Fill the created unit sales with zero
train_new['unit_sales'].fillna(0, inplace=True)

print(train.shape)
print(train_new.shape)


# In[ ]:


train.head()


# In[ ]:


train_new.head()

