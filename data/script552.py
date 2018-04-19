
# coding: utf-8

# Hey there!
# 
# I am not sure whether someone has already brought this up, but I think it's safe to assume that "ps_car_06_cat" identifies **car makers** while "ps_car_11_cat" refers to specific **car models** (maybe the most common?).  
# 
# See the code below for explanation and let me know whate you think about it!

# In[ ]:


# libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# load data
print('Loading data...')
train = pd.read_csv('../input/train.csv', na_values=-1).drop(['target'], axis = 1)
test = pd.read_csv('../input/test.csv', na_values=-1)
df = pd.concat([train,test])


# A simple boxplot shows how, apart from  "ps_car_11_cat == 104", every value of this column **corresponds to one and only one value** of "ps_car_06_cat"

# In[ ]:


f, ax = plt.subplots(1,figsize = (15,5))
sns.boxplot(x="ps_car_11_cat", y="ps_car_06_cat", data=df, ax = ax )
plt.xticks(rotation=90);


# More rigorously, we can show this with a single line of code:

# In[ ]:


df.groupby('ps_car_11_cat')['ps_car_06_cat'].nunique().tail(10)


# These are the car makers whose models are individually labeled using "ps_car_11_cat"

# In[ ]:


df[df['ps_car_11_cat']!=104].ps_car_06_cat.value_counts().sort_index()


# "ps_car_11_cat == 104" identifies other models of these carmakers, plus some models of other carmakers which are not labeled individually

# In[ ]:


df[df['ps_car_11_cat']==104].ps_car_06_cat.value_counts().sort_index()


# If, as reasonably suggested by many (see [here](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41488) for example), "ps_car_12" refers to something like "vehicle engine cylinder capacity (cc)", we can see that each individual model is not always associated with a fixed car configuration.  

# In[ ]:


f, ax = plt.subplots(1,figsize = (15,5))
sns.boxplot(x="ps_car_11_cat", y="ps_car_12", data=df, ax = ax )
plt.xticks(rotation=90);


# Some "variations" within these models seem more prominent than others... see for instance "ps_car_14", which has been associated to [weight](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/40599) ...

# In[ ]:


f, ax = plt.subplots(1,figsize = (15,5))
sns.boxplot(x="ps_car_11_cat", y="ps_car_14", data=df, ax = ax )
plt.xticks(rotation=90);


# ...or "ps_car_15", which has been associated to [manufacture year](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/40599) 

# In[ ]:


f, ax = plt.subplots(1,figsize = (15,5))
sns.boxplot(x="ps_car_11_cat", y="ps_car_15", data=df, ax = ax )
plt.xticks(rotation=90);


# Again, let me know what you think about it... and gimme a like if you think this was intersting :D
