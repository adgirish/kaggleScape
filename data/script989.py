
# coding: utf-8

# <h1> Welcome to my Kernel </h1><br>
# I will do some explorations though the data of Financial Hedging just to better understand the pattern of variables
# <br>
# 
# 
# <h3> Introduction to the data</h3> 

# <b>Background:</b>
# The underlying concept behind hedging strategies is simple, create a model, and make money doing it. The hardest part is finding the features that matter. For a more in-depth look at hedging strategies, I have attached one of my graduate papers to get you started.
#  <br>
# - <b>Mortgage-Backed Securities</b> <br>
# - <b>Geographic Business Investment</b> <br>
# - <b> Real Estate Analysis </b><br>
# 
# 
# <b>Statistical Fields:</b> <br>
# Note: All interpolated statistical data include Mean, Median, and Standard Deviation Statistics. For more information view the variable definitions document.
# 
# <b>Monthly Mortgage & Owner Costs: </b>Sum of mortgage payments, home equity loans, utilities, property taxes <br>
# <b>Monthly Owner Costs:</b> Sum of utilities, property taxes <br>
# <b>Gross Rent:</b> contract rent plus the estimated average monthly cost of utilities <br>
# <b>Household Income: </b>sum of the householder and all other individuals +15 years who reside in the household <br>
# <b>Family Income:</b> Sum of incomes of all members +15 years of age related to the householder.

# In[ ]:


#Load the librarys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

get_ipython().run_line_magic('matplotlib', 'inline')

# figure size in inches
rcParams['figure.figsize'] = 8,6


# In[ ]:


df_features = pd.read_csv("../input/kaggle_ACS_Financial_Features.csv", encoding='ISO-8859-1' )


# In[ ]:


#looking the shape of data
print(df_features.shape)

#in my local notebook I also have plotted describe(), info() functions
print(df_features.nunique())


# In[ ]:


#Looking the data
df_features.head()


# Ps: In this dataset we dont have any Nan's 

# <h1>First, let's known the Type variable that will be a first guide to the exploration</h1><br>
# We have just 2 categorical variables, so lets explore them
# 

# In[ ]:


print(df_features.Type.value_counts())

g = sns.factorplot(x='Type', data=df_features, kind="count",size=6,aspect=2)
g.set_titles("Count by Type")
plt.show()


# <h1>Primary Key</h1>

# In[ ]:


print(df_features.Primary.value_counts())

sns.factorplot(x="Primary", data=df_features, kind="count",size=5,aspect=1)
plt.show()


# <h2>What if we cross Type and Primary? 

# In[ ]:


g = sns.factorplot(x="Type", data=df_features, 
               kind="count", hue="Primary",
               size=5, aspect=2)
g.set_axis_labels(x_var="Types",
                  y_var="Counting")
plt.show()


# <h1>We can see that the Track is represented by themself</h1><br>
# We don't need to color by Primary
# 

# <h2>I will explore all the continuous variables through our  categorical datas</h2>
# 
# <h1> Starting by <i>family_income</i></h1>

# In[ ]:


fig, ax = plt.subplots(5,1, figsize=(12,8*3))

sns.distplot(df_features['family_income_mean'], 
             ax=ax[0],bins=50)
sns.boxplot(x='Type', y='family_income_mean', data=df_features,
            ax=ax[1])
sns.boxplot(x='Type', y='family_income_median', data=df_features, 
            ax=ax[2])
sns.boxplot(x='Type', y='family_income_stdev', data=df_features, 
            ax=ax[3])
sns.boxplot(x='Type', y='family_income_families', data=df_features, 
            ax=ax[4])

plt.show()


# <h1> <i>gross_rent</i></h1>

# In[ ]:


fig, ax = plt.subplots(5,1, figsize=(12,8*3))

sns.distplot(df_features['gross_rent_mean'], 
             ax=ax[0],bins=50)
sns.boxplot(x='Type', y='gross_rent_mean', data=df_features, 
            ax=ax[1])
sns.boxplot(x='Type', y='gross_rent_median', data=df_features, 
            ax=ax[2])
sns.boxplot(x='Type', y='gross_rent_stdev', data=df_features, 
            ax=ax[3])
sns.boxplot(x='Type', y='gross_rent_samples', data=df_features, 
            ax=ax[4])

plt.show()


# <h1> <i>morgages_ocsts</i></h1>

# In[ ]:


fig, ax = plt.subplots(5,1, figsize=(12,8*3))

sns.distplot(df_features['morgages_ocsts_mean'], 
             ax=ax[0],bins=50)
sns.boxplot(x='Type', y='morgages_ocsts_mean', data=df_features, 
            ax=ax[1])
sns.boxplot(x='Type', y='morgages_ocsts_median', data=df_features, 
            ax=ax[2])
sns.boxplot(x='Type', y='morgages_ocsts_stdev', data=df_features, 
            ax=ax[3])
sns.boxplot(x='Type', y='morgages_csts_samples', data=df_features, 
            ax=ax[4])

plt.show()


# <h1> <i>owner_cost</i></h1>

# In[ ]:


fig, ax = plt.subplots(5,1, figsize=(12,8*3))

sns.distplot(df_features['owner_cost_mean'], 
             ax=ax[0],bins=50)
sns.boxplot(x='Type', y='owner_cost_mean', data=df_features, 
            ax=ax[1])
sns.boxplot(x='Type', y='owner_cost_median', data=df_features, 
            ax=ax[2])
sns.boxplot(x='Type', y='owner_cost_stdev', data=df_features, 
            ax=ax[3])
sns.boxplot(x='Type', y='owner_cost_samples', data=df_features, 
            ax=ax[4])
plt.show()


# <h1> <i>household_income</i></h1>

# In[ ]:


fig, ax = plt.subplots(5,1, figsize=(12,8*3))

sns.distplot(df_features['household_income_mean'], 
             ax=ax[0],bins=50)
sns.boxplot(x='Type', y='household_income_mean', data=df_features, 
            ax=ax[1])
sns.boxplot(x='Type', y='household_income_median', data=df_features, 
            ax=ax[2])
sns.boxplot(x='Type', y='household_income_stdev', data=df_features, 
            ax=ax[3])
sns.boxplot(x='Type', y='household_income_wsum', data=df_features, 
            ax=ax[4])
plt.show()


# <h1> <i>Exploring the States</i></h1>

# In[ ]:


print("States with frequency greatest than 500: ")
print(df_features.State_Name.value_counts()[:13])

g = sns.factorplot(x="State_Name", data=df_features, 
                   kind="count", size = 6, aspect=2,  
                   orient='v')
g.set_titles(template="State count")
g.set_axis_labels(x_var="States Name",
                  y_var="Counting ")
g.set_xticklabels(rotation=90)
plt.show()

