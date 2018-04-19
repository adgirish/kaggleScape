
# coding: utf-8

# In[1]:


import pyLDAvis.gensim


# **This kernel is for beginners only.**
# 
# **Please upvote this kernel which motivates me to do more.**
# 
# ![](https://78.media.tumblr.com/0a56b418334765ec595a0982fe25aac3/tumblr_ouloa3CUT41wq17fxo3_400.gif)
# 
# 

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[3]:


train = pd.read_csv('../input/en_train.csv')
test = pd.read_csv('../input/en_test.csv')


# In[4]:


train.tail()


# In[5]:


test.tail()


# In[6]:


train['class'].unique()


# In[7]:


train['class'].value_counts().sort_values(ascending = False)


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


sns.countplot(y = train['class'],data = train)


# In[10]:


train[train['class']=='PUNCT'].head()


# In[11]:


train[train['class']=='DATE'].head()


# In[12]:


train[train['class']=='LETTERS'].head()


# In[13]:


train[train['class']=='CARDINAL'].head()


# In[14]:


train[train['class']=='VERBATIM'].head()


# In[15]:


train[train['class']=='DECIMAL'].head()


# In[16]:


train[train['class']=='MEASURE'].head()


# In[17]:


train[train['class']=='MONEY'].head()


# In[18]:


train[train['class']=='ORDINAL'].head()


# In[19]:


train[train['class']=='TIME'].head()


# Ok till now we saw how each class  wrods are represented before and after and now we got some idea on whole data. isnt it..?
# 
# Now lets looks how the sample submission file should be and we'll move in that direction ok.
# 
# 

# In[20]:


sample_submission = pd.read_csv("../input/en_sample_submission.csv")
sample_submission.head()


# If you observ carefully there are actually sentances in the train and test file. At first even i thought theses are chunks of words which represnted before and after convetions.
# 
# I found sentances by grouping sentance ids together. I'll show you some images which will be easier to understand the data.

# ![image.png](attachment:image.png)
# 
# This is ho3 it looks the original train data set.

# ![image.png](attachment:image.png)
# 
# If you club all the senteance-id 4 there is a sentance in it. 
# 

# ![](https://media2.giphy.com/media/l46CkATpdyLwLI7vi/200.webp#5-grid1)
# 
# Woooow!

# In[21]:


train['sentences'] = train.groupby("sentence_id")["sentence_id"].count()
train['sentences'].describe()


# so here what we got is, minimum words in a sentance are 2 and maximum words in a sentance are 256 and on an avg every sentance consists of 13 words.
# 
# i mean every setance_id consists of 13 token_ids and max is 256 and min is 2 only.

# In[22]:


test['sentences'] = test.groupby("sentence_id")["sentence_id"].count()
test['sentences'].describe()


# same applies to test set as said in train set. Now we'll see in visual way.

# In[23]:


plt.figure(figsize=(30,8))
sns.countplot(x = train['sentences'],data = train)
plt.xticks(rotation = 90)
plt.show()


# In[24]:


train[train['sentences'] ==256]


# In[25]:


train[train['sentence_id']==520453]


# we'll form this as sentance ok

# In[26]:


max_sentance = train[train['sentence_id']==520453].before.values.tolist()
max_sentance = ' '.join(max_sentance)
max_sentance


# Now we'll see how small sentance looks like.

# In[27]:


train[train['sentences']==2]


# these are the values which have only 2 words. lets check.

# In[28]:


min_sentance = train[train['sentence_id']==41].before.values.tolist()
min_sentance = ' '.join(min_sentance)
min_sentance #you can replace the above number to get different sentances which have 2 words only.


# I knwo this is lengthy explanation. But i want to make it clear the concept finally.

# ![](https://media.giphy.com/media/cyoN6pC6kek2A/giphy.gif)

# **More to come ...**
# 
# **Please encourage me to do more by upvoting**
# 
# **Thank you :)**
