
# coding: utf-8

# # 1. Introduction
# This will be the longest EDA you've ever seen...
# 
# Let's load some libraries and the data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # 2. Adding song year to our datasets

# In[ ]:


songs_extra = pd.read_csv('../input/song_extra_info.csv')

def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
        
songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)

train = train.merge(songs_extra, on = 'song_id', how = 'left')
test = test.merge(songs_extra, on = 'song_id', how = 'left')


# # 3. Let's count what fraction of songs was released in 2017
# We will use rolling mean with a window of 50000 for this purpose.

# In[ ]:


train['2017_songs_frac'] = (train['song_year'] == 2017).rolling(window = 50000, center = True).mean()
test['2017_songs_frac'] = (test['song_year'] == 2017).rolling(window = 50000, center = True).mean()


# # 4. Let's plot it against train and test index values

# In[ ]:


plt.figure()
plt.plot(train.index.values, train['2017_songs_frac'], '-',
        train.shape[0] + test.index.values, test['2017_songs_frac'], '-');


# # 5. Yes! Data is chronologically ordered!
# I think everyone should be aware of this, maybe even organizers should confirm this. It does help to establish a pretty good (as for the time series problem) validation set - you just leave last 2.5 mln (length of the test data) rows of the training data for the validation. It helped me to get 0.69 score without even taking a time series approach to this problem.
