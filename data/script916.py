
# coding: utf-8

# ![](https://www.listchallenges.com/f/lists/6c536c33-489b-4bee-b259-00074fa53a0d.jpg)

# # More To Come. Stay Tuned. !!
# If there are any suggestions/changes you would like to see in the Kernel please let me know :). Appreciate every ounce of help!
# 
# **This notebook will always be a work in progress**. Please leave any comments about further improvements to the notebook! Any feedback or constructive criticism is greatly appreciated!.** If you like it or it helps you , you can upvote and/or leave a comment :).**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




# # More is coming Soon

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
submission = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


print("Training data size",train_data.shape)
print("test data size",test_data.shape)
submission.head()


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


# now open the URL
temp = 4444
print('id', train_data['id'][temp])
print('url:', train_data['url'][temp])
print('landmark id:', train_data['landmark_id'][temp])


# In[ ]:


train_data['landmark_id'].value_counts().hist()


# In[ ]:


# missing data in training data 
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)
missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()


# In[ ]:


# missing data in test data 
total = test_data.isnull().sum().sort_values(ascending = False)
percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending = False)
missing_test_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_test_data.head()


# ## We can see there is no missing data
# ## Now Lets see most frequent Landmarks

# In[ ]:


# Occurance of landmark_id in decreasing order(Top categories)
temp = pd.DataFrame(train_data.landmark_id.value_counts().head(8))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']
temp


#  ### The most frequent landmark_id is 9633  and the count is 50337.

# In[ ]:


# Plot the most frequent landmark_ids
plt.figure(figsize = (9, 8))
plt.title('Most frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp,
            label="Count")
plt.show()


# ## Lets see least frequent landmarks

# In[ ]:


# Occurance of landmark_id in increasing order
temp = pd.DataFrame(train_data.landmark_id.value_counts().tail(8))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']
temp


# ### There are many least frequent landmarks whose count is 1

# In[ ]:


# Plot the least frequent landmark_ids
plt.figure(figsize = (9, 8))
plt.title('Least frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp,
            label="Count")
plt.show()


# ## lets see unique URLs

# In[ ]:


# Unique URL's
train_data.nunique()


# In[ ]:


#Class distribution
plt.figure(figsize = (10, 8))
plt.title('Category Distribuition')
sns.distplot(train_data['landmark_id'])

plt.show()


# In[ ]:


print("Number of classes under 20 occurences",(train_data['landmark_id'].value_counts() <= 20).sum(),'out of total number of categories',len(train_data['landmark_id'].unique()))


# ## Lets display some images from URLs / Some URLs Visulization 

# In[ ]:


from IPython.display import Image
from IPython.core.display import HTML 

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(12).iteritems()])

    display(HTML(images_list))


# In[ ]:


category = train_data['landmark_id'].value_counts().keys()[0]
urls = train_data[train_data['landmark_id'] == category]['url']
display_category(urls, "")


# In[ ]:


category = train_data['landmark_id'].value_counts().keys()[1]
urls = train_data[train_data['landmark_id'] == category]['url']
display_category(urls, "")


# # Now Lets extract the website name and see their occurances

# In[ ]:


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

# In[ ]:


print("Training data size",train_data.shape)
print("test data size",test_data.shape)


# In[ ]:


train_data.head(8)


# In[ ]:


test_data.head()


# ## occurances of sites in train_data

# In[ ]:


# Occurance of site in decreasing order(Top categories)
temp = pd.DataFrame(train_data.site_name.value_counts())
temp.reset_index(inplace=True)
temp.columns = ['site_name','count']
temp


# ### As we can see there are total 16 unique sites.

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


# ## occurances of sites in test_data

# In[ ]:


# Occurance of site in decreasing order(Top categories)
temp = pd.DataFrame(test_data.site_name.value_counts())
temp.reset_index(inplace=True)
temp.columns = ['site_name','count']
temp


# ### Total unique sites are 25 in test data and some are different from train_data
# 

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

# ##### Referances :
#  * https://www.kaggle.com/mxdbld/yadv-simple-exploration-of-google-recognition 
#  * https://www.kaggle.com/gpreda/google-landmark-recognition-challenge-eda
