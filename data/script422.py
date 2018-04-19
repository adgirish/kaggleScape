
# coding: utf-8

# **Basic Navigation of the FIFA 18 Complete Dataset**

# In[19]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
# Next, we'll load the Iris flower dataset, which is in the "../input/" directory
fifa = pd.read_csv("../input/CompleteDataset.csv") # the iris dataset is now a Pandas DataFrame
# Any results you write to the current directory are saved as output.


# In[20]:


fifa.head()


# Listing out the column names. 

# In[21]:


fifa.columns


# Lets take a few columns for preliminary analysis.

# In[22]:


fifa = fifa[['Name', 'Age', 'Nationality', 'Overall', 'Wage', 'Potential', 'Club', 'Value', 'Preferred Positions']]
fifa.head(10)


# Crafting a new parameter for Growth = Potential - Overall

# In[23]:


fifa['Growth'] = fifa['Potential'] - fifa['Overall']
fifa.head()


# Plotting growth potential and overall against age

# In[24]:


fifa_growth = fifa.groupby(['Age'])['Growth'].mean()
fifa_overall = fifa.groupby(['Age'])['Overall'].mean()
fifa_potential = fifa.groupby(['Age'])['Potential'].mean()

summary = pd.concat([fifa_growth, fifa_overall, fifa_potential], axis=1)

axis = summary.plot()
axis.set_ylabel('Rating Points')
axis.set_title('Average Growth Potential by Age')


# Let's find clubs with most players rated over 85

# In[25]:


cutoff = 85
players = fifa[fifa['Overall']>cutoff]
grouped_players = fifa[fifa['Overall']>cutoff].groupby('Club')
number_of_players = grouped_players.count()['Name'].sort_values(ascending = False)

ax = sns.countplot(x = 'Club', data = players, order = number_of_players.index)

ax.set_xticklabels(labels = number_of_players.index, rotation='vertical')
ax.set_ylabel('Number of players (Over 90)')
ax.set_xlabel('Club')
ax.set_title('Top players (Overall > %.i)' %cutoff)


# Let's clean wage and value first. Using the function from [this](https://www.kaggle.com/chomi87/eda-on-fifa-2018) kernel.

# In[27]:


def extract_value_from(value):
    out = value.replace('€', '')
    if 'M' in out:
        out = float(out.replace('M', ''))*1000000
    elif 'K' in value:
        out = float(out.replace('K', ''))*1000
    return float(out)


# In[28]:


fifa['Value'] = fifa['Value'].apply(lambda x: extract_value_from(x))
fifa['Wage'] = fifa['Wage'].apply(lambda x: extract_value_from(x))


# Let's explore Value against Overall rating

# In[36]:


fifa_wage = fifa.groupby(['Overall'])['Wage'].mean()
fifa_value = fifa.groupby(['Overall'])['Value'].mean()
fifa_wage = fifa_wage.apply(lambda x: x/1000)
fifa_value = fifa_value.apply(lambda x: x/1000000)
fifa["Wage(by Potential)"] = fifa["Wage"]
fifa["Value(by Potential)"] = fifa["Value"]
fifa_wage_p = fifa.groupby(['Potential'])['Wage(by Potential)'].mean()
fifa_value_p = fifa.groupby(['Potential'])['Value(by Potential)'].mean()
fifa_wage_p = fifa_wage_p.apply(lambda x: x/1000)
fifa_value_p = fifa_value_p.apply(lambda x: x/1000000)
summary = pd.concat([fifa_wage, fifa_value, fifa_wage_p, fifa_value_p], axis=1)

axis = summary.plot()
axis.set_ylabel('Wage / Value')
axis.set_title('Average Wage / Value by Rating')


# Doing the same against age.

# In[37]:


fifa_wage_a = fifa.groupby(['Age'])['Wage'].mean()
fifa_value_a = fifa.groupby(['Age'])['Value'].mean()
fifa_wage_a = fifa_wage_a.apply(lambda x: x/1000)
fifa_value_a = fifa_value_a.apply(lambda x: x/1000000)
summary = pd.concat([fifa_wage_a, fifa_value_a], axis=1)

axis = summary.plot()
axis.set_ylabel('Wage / Value')
axis.set_title('Average Age')

