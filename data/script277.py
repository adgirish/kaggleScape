
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train = pd.read_json('../input/train.json')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# load the training dataset.
train = pd.read_json('../input/train.json')


# In[ ]:


# Isolation function.
def iso(arr):
    p = np.reshape(np.array(arr), [75,75]) >(np.mean(np.array(arr))+2*np.std(np.array(arr)))
    return p * np.reshape(np.array(arr), [75,75])


# In[ ]:


# Size in number of pixels of every isolated object.
def size(arr):     
    return np.sum(arr<-5)


# In[ ]:


# Feature engineering iso1 and iso2.
train['iso1'] = train.iloc[:, 0].apply(iso)
train['iso2'] = train.iloc[:, 1].apply(iso)


# In[ ]:


# Feature engineering s1 s2 and size.
train['s1'] = train.iloc[:,5].apply(size)
train['s2'] = train.iloc[:,6].apply(size)
train['size'] = train.s1+train.s2


# In[ ]:


# How works s1 on the discrimination
print(train.groupby('is_iceberg')['s1'].mean())


# In[ ]:


# Hist comparison
train.groupby('is_iceberg')['s1'].hist(bins=60, alpha=.6)


# In[ ]:


# How works s2 on the discrimination
print(train.groupby('is_iceberg')['s2'].mean())


# In[ ]:


# Hist comparison
train.groupby('is_iceberg')['s2'].hist(bins=60, alpha=.6)


# In[ ]:


# How works size on the discrimination
print(train.groupby('is_iceberg')['size'].mean())


# In[ ]:


# Hist comparison
train.groupby('is_iceberg')['size'].hist(bins=60, alpha=.6)


# In[ ]:


# Indexes for ships or icebergs.
index_ship=np.where(train['is_iceberg']==0)
index_ice=np.where(train['is_iceberg']==1)


# In[ ]:


# For ploting
def plots(band,index,title):
    plt.figure(figsize=(12,10))
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.xticks(())
        plt.yticks(())
        plt.xlabel((title))
        plt.imshow(np.reshape(train[band][index[0][i]], (75,75)),cmap='gist_heat')
    plt.show()  


# In[ ]:


plots('band_1',index_ship,'band1 ship')


# In[ ]:


plots('band_1',index_ice,'band1 iceberg')


# In[ ]:


plots('iso1',index_ship,'iso1 ship')


# In[ ]:


plots('iso1',index_ice,'iso1 iceberg')


# **NOTE :** It looks like images with incidence angles having less than or equal to 4 decimal are the naturally captured images, and those with greater precision are machine generated, as 'brassmonkey' describes very well. 
# In the data description of the competition is also refered as : 
# "Please note that we have included machine-generated images in the test set to prevent hand labeling. They are excluded in scoring."
# This is an important point to be in mind. 
# 

# **Conclusion.** Size matters. These features could be improved :
# -Tuning the times std to take for the isolation. 
# -Applying some filters to help on the discrimination. 
# -The size can be categorized in order to help on the accuracy of the Classifier.
# 
# I hope these lines be useful for your. Please vote up.
# 
