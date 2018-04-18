
# coding: utf-8

# Hello there, everyone.  I did a brief analysis on the "managers" since at first glance the average "interest level" seemed to differ substantially from one to another . 
# 
# Anyway, let me know what you think about it and like this notebook if you enjoyed reading it (it's my 1st one, be nice :D)

# In[ ]:


# let's load the usual packages first
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ... and get the data...

# In[ ]:


train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')


# First of all, let's see how many different managers we have on both datasets.

# In[ ]:


man_train_list = train_df.manager_id.unique()
man_test_list = test_df.manager_id.unique()
print("Train: {0}".format(len(man_train_list)))
print("Test: {0}".format(len(man_test_list)))


# There are more managers in the test dataset, which also features more records.
# 
# Let's create a dataframe with all the train and test managers, including the number of entries they are responsible for.

# In[ ]:


temp1 = train_df.groupby('manager_id').count().iloc[:,-1]
temp2 = test_df.groupby('manager_id').count().iloc[:,-1]
df_managers = pd.concat([temp1,temp2], axis = 1, join = 'outer')
df_managers.columns = ['train_count','test_count']
print(df_managers.head(20))


# Some managers have entries only in one of the two datasets. But as we will see later, these managers have only very few entries.
# 
# Indeed, a minority of managers are responsible for most of the entries of both dataset

# In[ ]:


print(df_managers.sort_values(by = 'train_count', ascending = False).head(10))


# This is more clear if one looks at the plots for the cumulative distributions.

# In[ ]:


fig, axes = plt.subplots(1,2, figsize = (12,5))
temp = df_managers['train_count'].dropna().sort_values(ascending = False).reset_index(drop = True)
axes[0].plot(temp.index+1, temp.cumsum()/temp.sum())
axes[0].set_title('cumulative train_count')

temp = df_managers['test_count'].dropna().sort_values(ascending = False).reset_index(drop = True)
axes[1].plot(temp.index+1, temp.cumsum()/temp.sum())
axes[1].set_title('cumulative test_count')


# The Pareto principle, i.e. the 80/20 rule, seems to apply here. As 20% of the managers are roughly responsible for roughly 80% of the entries.

# In[ ]:


ix20 = int(len(df_managers['train_count'].dropna())*0.2)
print("TRAIN: 20% of managers ({0}) responsible for {1:2.2f}% of entries".format(ix20,df_managers['train_count'].sort_values(ascending = False).cumsum().iloc[ix20]/df_managers['train_count'].sum()*100))

ix20 = int(len(df_managers['test_count'].dropna())*0.2)
print("TEST: 20% of managers ({0}) responsible for {1:2.2f}% of entries".format(ix20, df_managers['test_count'].sort_values(ascending = False).cumsum().iloc[ix20]/df_managers['test_count'].sum()*100))


# As mentioned before, fortunately, these top contributors are the same for both datasets. The managers featuring in only one of the two datasets usually have very few entries.

# In[ ]:


man_not_in_test = set(man_train_list) - set(man_test_list)
man_not_in_train = set(man_test_list) - set(man_train_list)

print("{} managers are featured in train.json but not in test.json".format(len(man_not_in_test)))
print("{} managers are featured in test.json but not in train.json".format(len(man_not_in_train)))


# In[ ]:


print(df_managers.loc[list(man_not_in_test)]['train_count'].describe())
print(df_managers.loc[list(man_not_in_train)]['test_count'].describe())


# Besides, it looks like there is a strong correlation between the number of entries of the contributors in both datasets.

# In[ ]:


df_managers.sort_values(by = 'train_count', ascending = False).head(1000).corr()


# In[ ]:


df_managers.sort_values(by = 'train_count', ascending = False).head(100).plot.scatter(x = 'train_count', y = 'test_count')


# Now let's focus on the training dataset and on the "interest_level" of its top 100 contributors.
# These folks account for a whopping 35% of the entries. The 1st alone for over 5% of them! That's quite a lot. 
# 
# According to the discussion above, similar figures are expected for the test dataset.

# In[ ]:


temp = df_managers['train_count'].sort_values(ascending = False).head(100)
temp = pd.concat([temp,temp.cumsum()/df_managers['train_count'].sum()*100], axis = 1).reset_index()
temp.columns = ['manager_id','count','percentage']
print(temp)


# Let's isolate the entries relative to these 100 managers with the "interest_level" column as well. We create dummies from this latter column as they are easier to work with.

# In[ ]:


man_list = df_managers['train_count'].sort_values(ascending = False).head(100).index
ixes = train_df.manager_id.isin(man_list)
df100 = train_df[ixes][['manager_id','interest_level']]
interest_dummies = pd.get_dummies(df100.interest_level)
df100 = pd.concat([df100,interest_dummies[['low','medium','high']]], axis = 1).drop('interest_level', axis = 1)

print("The top100 contributors account for {} entries\n".format(len(df100)))

print(df100.head(10))


# Before continuing, let's give them some fake identities based on the most common first and last names in the US.

# In[ ]:


import itertools

# 50 most common surnames in the 90s (http://surnames.behindthename.com/top/lists/united-states/1990)
last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 
 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 
 'Martinez', 'Robinson', 'Clark', 'Rodriguez', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 'Young',
 'Hernandez', 'King', 'Wright', 'Lopez', 'Hill', 'Scott', 'Green', 'Adams', 'Baker', 'Gonzalez', 'Nelson', 
 'Carter', 'Mitchell', 'Perez', 'Roberts', 'Turner', 'Phillips', 'Campbell', 'Parker', 'Evans', 'Edwards', 'Collins']

# 10 most common first names for females and males (names.mongabay.com) 
first_names = ['Mary',  'Patricia',  'Linda',  'Barbara',  'Elizabeth',  
               'Jennifer',  'Maria',  'Susan',  'Margaret',  'Dorothy',
               'James', 'John', 'Robert', 'Michael', 'William', 'David',
               'Richard', 'Charles', 'Joseph', 'Thomas']

names = [first + ' ' + last for first,last in (itertools.product(first_names, last_names))]

# shuffle them
np.random.seed(12345)
np.random.shuffle(names)

dictionary = dict(zip(man_list, names))
df100.loc[df100.manager_id.isin(dictionary), 'manager_id' ] = df100['manager_id'].map(dictionary)
print(df100.head())


# In[ ]:


# see if the name coincides
print(names[:10])
print(df100.groupby('manager_id').count().sort_values(by = 'low', ascending = False).head(10))


# Splendid... we have their names now, so let's proceed and compute their average performances in terms of "interest level" so we can spot who's a pro and who's not. 

# In[ ]:


gby = pd.concat([df100.groupby('manager_id').mean(),df100.groupby('manager_id').count()], axis = 1).iloc[:,:-2]
gby.columns = ['low','medium','high','count']
gby.sort_values(by = 'count', ascending = False).head(10)


# Their performances seem very different, even for people with similar number of entries.
# 
# Indeed they are..

# In[ ]:


gby.sort_values(by = 'count', ascending = False).drop('count', axis = 1).plot(kind = 'bar', stacked = True, figsize = (15,5))
plt.figure()
gby.sort_values(by = 'count', ascending = False)['count'].plot(kind = 'bar', figsize = (15,5))


# I think this high diversity should be accounted for when building our predictive model! 
# 
# It would be interesting to rank the managers based on their intereset levels. For instance, we could compute their "skill" by assigning 0 points for "lows", 1 for "mediums" and 2 for "highs". Since they have different number of entries, let's quickly do so by multiplying the average results.

# In[ ]:


gby['skill'] = gby['medium']*1 + gby['high']*2 

print("Top performers")
print(gby.sort_values(by = 'skill', ascending = False).reset_index().head())
print("\nWorst performers")
print(gby.sort_values(by = 'skill', ascending = False).reset_index().tail())


# Dorothy Turner and Dorothy Lopez are rocking it! Poor Dorothy Martinez instead should consider moving to another industry... 402 entries, all of them uninspiring (btw I did not pick the random seed to have all the Dorothies here...).
# 
# I won't go deeper to try to explain why these performances are so different. It seems though like most of the managers do a poor job (I am sure it ain't their fault, is just that the properties they handle are not that cool after all...).
# 
# Cheers!
# 
# p.s.: I did a similar analysis on "building_id" here --> https://www.kaggle.com/den3b81/two-sigma-connect-rental-listing-inquiries/some-insights-on-building-id

# In[ ]:


gby.skill.plot(kind = 'hist')
print(gby.mean())

