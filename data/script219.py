
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# Looks we have features named **ps_ind_xx** which are mostly hidden for security reasons. But this aint stopping a data scientist analyse the dataset. Let's layout our steps to analyse EDA
# 
# 1. ** Retrive information on all the column types.**
# 2. **Hunt down the NaN or Null values in the dataset. **
# 3. **Visualizing Binary and Categorical Features seperately**
# 4. **Study continously varying features**

# ## Step 1: Retrive information on all the column types.
#  

# In[ ]:


# Get information on names of columns
# Only taking till 5 index,
# as names might populate the code output and is not intuitive for this simple EDA
train.columns[:5]


# In[ ]:


# Get those dtypes of those named columns
train.dtypes[:5] 


# To be honest. I honestly can't interpret the information clearly with two plain black outputs! Let's build a panda data frame of names of columns and their types.

# In[ ]:


df_dtypes = pd.DataFrame({'Feature': train.columns , 'Data Type': train.dtypes.values})


# In[ ]:


df_dtypes.head(15)


# Looks good to me. And when I checked the full dataframe there are no **strings**, **object**, and **time-type** as they are already proccesed as given in the Description, 
# 
# > **Data Description**
# > In this competition, you will predict the probability that an auto insurance policy holder files a claim.
# > 
# > In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.

# In[ ]:


## Fixing -1 with NaN values
train_v1 = train.replace(-1, np.NaN)
test_v1 = test.replace(-1, np.NaN)


# ## Step 2 : Hunt down the NaN or Null values in the dataset
# 
#  

# In[ ]:


plt.figure(figsize=(18,7))
sns.heatmap(train_v1.head(100).isnull() == True, cmap='viridis')


# **Explaning the code :**
# 1. **train.isnull() == True** - Returns a matrix of True and False values. **True** if a cell is Null, **False** if not null.
# 2. **cmap='viridis'**   -  Color scheme for the heatmap. This one is my favorite. 

# The yellow bars indicate the presence of NaN values. It seems like few processed categorical have missing values.

# In[ ]:


have_null_df = pd.DataFrame(train_v1.isnull().any(), columns=['Have Null?']).reset_index()


# In[ ]:


have_null_df[have_null_df['Have Null?'] == True]['index']


# Looks like we are having the complete list of columns to be **busted**. Filling these Null values can have a greater significance in your score.  

# **Dropping all the NaN values for initial EDA**

# In[ ]:


train_v1.dropna(inplace=True)


# In[ ]:


plt.figure(figsize=(16,11))
sns.heatmap(train_v1.head(100).corr(), cmap='viridis')


# **We are having these empty white spaces because the heatmap is plotted with correlation between the binary, interger and categorical feature. This form of representation is hard to interpret. Let's divide our columns into seperate groups to carry our further analysis.
# **

# ## Step 3: Visualizing Binary and Categorical Features seperately

# In[ ]:


binary_feat = [c for c in train_v1.columns if c.endswith("bin")]
categorical_feat = [c for c in train_v1.columns if c.endswith("cat")]


# ### Binary features  countplots

# In[ ]:


plt.figure(figsize=(17,20))
for i, c in enumerate(binary_feat):
    ax = plt.subplot(6,3,i+1)
    sns.countplot(train_v1[c])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ### Categorical Countplots

# In[ ]:


plt.figure(figsize=(17,20))
for i, c in enumerate(categorical_feat):
    ax = plt.subplot(6,3,i+1)
    sns.countplot(train_v1[c])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# The count plot of ps_car_11_cat seems to have lot of categories. Let's check out the zoomed version of the graph

# In[ ]:


plt.figure(figsize=(17,6))
sns.countplot(train_v1['ps_car_11_cat'])


# We can see spike in the right end of the plot. Let's check the number of unique valies in **ps_car_11_cat**

# In[ ]:


print ("There are {} unique values for ps_car_11_cat" .format(train_v1['ps_car_11_cat'].nunique()))


# Now, that are too many number categories. Let's see the top 10 highest counted categories...

# In[ ]:


train_v1['ps_car_11_cat'].value_counts().head(10)


# ## Step 4: Study continously varying features

# In[ ]:


continuous_feat= [i for i in train_v1.columns if 
                    ((i not in binary_feat) and (i not in categorical_feat) and (i not in ["target", "id"]))]


# In[ ]:


train_v1[continuous_feat].head(5)


# In[ ]:


ind_feat = [c for c in continuous_feat if c.startswith("ps_ind")]
reg_feat = [c for c in continuous_feat if c.startswith("ps_reg")]
car_feat = [c for c in continuous_feat if c.startswith("ps_car")]
calc_feat = [c for c in continuous_feat if c.startswith("ps_calc")]
target = ['target']


# Time to check the correlation with our **target value**.

# In[ ]:


plt.figure(figsize=(17,11))
sns.heatmap(train_v1[ind_feat+ calc_feat + car_feat + reg_feat + target].corr(), cmap= plt.cm.inferno)


# Hmm.. Something clicked at top left and bottom right portion. Let's remove calc_feat and observe the heatmap

# In[ ]:


plt.figure(figsize=(17,11))
sns.heatmap(train_v1[ind_feat+ car_feat + reg_feat + target].corr(), cmap= 'viridis', annot=True)


# **That's all for today. This notebook shall be updated biweekly with new information and dept EDA.**
# 
# **Next update - 3 Oct, 2017**

# **#TODO** list for this notebook
# 1. Detailed EDA
# 2. Feature Importance through various models and their ensemble importance.
# 3. Explaining how to choose the best ML model / ensemble methods.
# 4. Parameters tuning and Hyper parameters setting.

# Feel free to brainstrom ideas and other queries in the comment section.
# 
# ***P.S - This is my first Kaggle EDA Kernel. Upvotes are extremly appreciated! ***
