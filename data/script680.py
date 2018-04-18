
# coding: utf-8

# # Exploring the Dataset

# In[ ]:


import numpy as np 
import pandas as pd 


# In[ ]:


df = pd.read_csv("../input/en_train.csv")


# In[ ]:


df.head(10)


# we are dealing with 16 classes

# In[ ]:


df['class'].unique()


# In[ ]:


tdf = pd.read_csv("../input/en_test.csv")
tdf.head(20)


# Let's see how they are distributed

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
f,axarray = plt.subplots(2,1,figsize=(15,10))
hist = df.groupby('class',as_index=False).count()
hist = hist[hist['class']!='PLAIN']
g= sns.barplot(x=hist['class'],y=hist['before'],ax=axarray[0])
for item in g.get_xticklabels():
    item.set_rotation(45)
hist = hist[hist['class']!='PUNCT']
g= sns.barplot(x=hist['class'],y=hist['before'],ax=axarray[1])
for item in g.get_xticklabels():
    item.set_rotation(45)
plt.show()


# Plain and Punctuation are the two most present classes followed by Date, Letters, Cardinal and Verbatim.

# In[ ]:


length = df.groupby(['sentence_id'],as_index=False).count()
length = length.groupby(['before'],as_index=False).count()


# In[ ]:


length['before'].describe()


# In[ ]:


f,axarray = plt.subplots(1,1,figsize=(15,10))
length = length[0:40]
sns.barplot(x = length['before'],y=length['after'])


# ## Punctuation

# In[ ]:


df[df['class']=='PUNCT'].head()


# In[ ]:


len(df[df['class']=='DATE'])


# ## Dates

# In[ ]:


df[df['class']=='DATE'].head(10)


# In[ ]:


len(df[df['class']=='PUNCT'])


# ## Letters

# In[ ]:


df[df['class']=='LETTERS'].head()


# In[ ]:


len(df[df['class']=='LETTERS'])


# ## Cardinals

# In[ ]:


df[df['class']=='CARDINAL'].head()


# In[ ]:


len(df[df['class']=='CARDINAL'])


# ## Verbatim

# In[ ]:


df[df['class']=='VERBATIM'].head()


# In[ ]:


len(df[df['class']=='VERBATIM'])


# ## DECIMAL

# In[ ]:


df[df['class']=='DECIMAL'].head()


# In[ ]:


len(df[df['class']=='DECIMAL'])


# ## MEASURE

# In[ ]:


df[df['class']=='MEASURE'].head()


# In[ ]:


len(df[df['class']=='MEASURE'])


# ## MONEY

# In[ ]:


df[df['class']=='MONEY'].head()


# In[ ]:


len(df[df['class']=='MONEY'])


# ## ORDINAL

# In[ ]:


df[df['class']=='ORDINAL'].head()


# In[ ]:


len(df[df['class']=='ORDINAL'])


# ## TIME

# In[ ]:


df[df['class']=='TIME'].head()


# In[ ]:


len(df[df['class']=='TIME'])


# ## ELECTRONIC

# In[ ]:


df[df['class']=='ELECTRONIC'].head()


# In[ ]:


len(df[df['class']=='ELECTRONIC'])


# ## DIGIT

# In[ ]:


df[df['class']=='DIGIT'].head()


# ## FRACTION

# In[ ]:


df[df['class']=='FRACTION'].head()


# In[ ]:


len(df[df['class']=='FRACTION'])


# ## TELEPHONE

# In[ ]:


df[df['class']=='TELEPHONE'].head()


# In[ ]:


len(df[df['class']=='TELEPHONE'])


# ## ADDRESS

# In[ ]:


df[df['class']=='ADDRESS'].head()


# In[ ]:


len(df[df['class']=='ADDRESS'])

