
# coding: utf-8

# ![](https://img-aws.ehowcdn.com/560x560/photos.demandstudios.com/getty/article/151/36/87689206.jpg)

# # More To Come. Stay Tuned. !!
# In this Notrebook, I did  Google Landmark Retrieval Exploratory Analysis.
# If there are any suggestions/changes you would like to see in the Kernel please let me know :). Appreciate every ounce of help!
# 
# This notebook will always be a work in progress. Please leave any comments about further improvements to the notebook! Any feedback or constructive criticism is greatly appreciated!.
# ** If you like it or it helps you , you can upvote and/or leave a comment :).**
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data = pd.read_csv('../input/index.csv')
test_data = pd.read_csv('../input/test.csv')
submission = pd.read_csv("../input/sample_submission.csv")


# In[3]:


print("Training data size",train_data.shape)
print("test data size",test_data.shape)


# In[4]:


train_data.head()


# In[5]:


test_data.head()


# In[6]:


submission.head()


# In[7]:


# now open the URL
temp = 4444
print('id', train_data['id'][temp])
print('url:', train_data['url'][temp])


# In[8]:


# missing data in training data 
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)
missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()


# In[9]:


# missing data in test data 
total = test_data.isnull().sum().sort_values(ascending = False)
percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending = False)
missing_test_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_test_data.head()


# ## Lets display some images from URLs

# In[26]:


from IPython.display import Image
from IPython.core.display import HTML 

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(20).iteritems()])

    display(HTML(images_list))


# In[27]:


urls = train_data['url']
display_category(urls, "")


# In[29]:


urls = train_data['url']
display_category(urls, "")


# ## Lets see unique URL

# In[30]:


# Unique URL's
train_data.nunique()


# ## All URLs are unique.

# ## Now Lets extract the website names and see their occurances

# In[14]:


# Extract site_names for train data
temp_list = list()
for path in train_data['url']:
    temp_list.append((path.split('//', 1)[1]).split('/', 1)[0])
train_data['site_name'] = temp_list
# Extract site_names for test data
temp_list = list()
for path in test_data['url']:
    temp_list.append((path.split('//', 1)[1]).split('/', 1)[0])
test_data['site_name'] = temp_list


# ### We have added one new column "site_name". lets see

# In[15]:


print("Training data size",train_data.shape)
print("test data size",test_data.shape)


# In[16]:


train_data.head(8)


# In[17]:


test_data.head()


# ### occurances of sites in train_data

# In[18]:


# Occurance of site in decreasing order(Top categories)
temp = pd.DataFrame(train_data.site_name.value_counts())
temp.reset_index(inplace=True)
temp.columns = ['site_name','count']
temp


# ### As we can see there are total 17 unique sites.

# In[19]:


# Plot the Sites with their count
plt.figure(figsize = (9, 8))
plt.title('Sites with their count')
sns.set_color_codes("pastel")
sns.barplot(x="site_name", y="count", data=temp,
            label="Count")
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()


# ### occurances of sites in test_data

# In[20]:


# Occurance of site in decreasing order(Top categories)
temp = pd.DataFrame(test_data.site_name.value_counts())
temp.reset_index(inplace=True)
temp.columns = ['site_name','count']
temp


# ### Total unique sites are 25 in test data and some are different from train_data

# In[ ]:


# Plot the Sites with their count
plt.figure(figsize = (9, 8))
plt.title('Sites with their count')
sns.set_color_codes("pastel")
sns.barplot(x="site_name", y="count", data=temp,
            label="Count")
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()


# ### As we can see that most of the images are taken from one site only.
