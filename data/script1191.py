
# coding: utf-8

# Food that tastes good, makes me fat. I don't want to get fat(ter). So even though a steady diet of Taco Bell is tempting, probably won't help me accomplish my 'be less fat' goal.
# 
# So as an alternative to Taco Bell, I want to find low calorie recipes that 1) taste good & 2) have relatively high amounts of protein.
# 
# Most people would just search for healthy recipes, but it's more fun to use a data set like this.
# 
# Disclaimer: As it will become painfully clear, I'm definitely not a nutritionist. Please forgive the use of low calorie as an (incorrect) proxy for 'healthy' and any other nutrition-based errors.

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[ ]:


#load data
df1 = pd.read_csv('../input/epi_r.csv')


# In[ ]:


#quick look at data
df1.head(2)


# 680 is a lot of columns, looks like a majority are ingredient types. Let's assume that I don't care about ingredients, only basic nutritional info & quick/simple metrics

# In[ ]:


#narrowing dataset & looking at summary stats
recipes = df1.iloc[:,:10]
recipes.drop(['#cakeweek','#wasteless'], axis=1, inplace=True)

recipes.describe()


# Few observations
# * Lots of 4.375 ratings, could be issue with data, let's assume there isn't a problem.
# * Definitely need to pull outliers from 'calories' column (should handle the high values for 'protein' - 'sodium'
# * Will need to fill missing values

# In[ ]:


#checking nan values in each column
for i in recipes.columns:
    print(i, sum(recipes[i].isnull()))


# In[ ]:


#filling nan values with pseudo-average & removing outliers ---- may not be best method, alternative is to drop nan rows
cal_clean = recipes.loc[recipes['calories'].notnull()]

q1  = cal_clean['calories'].quantile(.25)
q3  = cal_clean['calories'].quantile(.75)
iqr = q3 - q1

for i in recipes.columns[1:6]:
    recipes[i].fillna(cal_clean.loc[(cal_clean['calories'] > q1) & (recipes['calories'] < q3)][i].mean(), inplace=True)
    
recipes = recipes.loc[(recipes['calories'] > q1-(iqr*3)) & (recipes['calories'] < q3+(iqr*3))]


# In[ ]:


#check summary stats after cleaning data
recipes.describe()


# After cleaning up the data & removing outliers; this looks much better

# In[ ]:


#plotting health metrics against recipe rating
dict_plt = {0:'calories',1:'protein',2:'fat',3:'sodium'}

sns.set(font_scale=.7)

fig, ax = plt.subplots(1,4, figsize=(10,3))

for i in range(4):
    sns.barplot(x='rating',y=dict_plt[i], data=recipes, ax=ax[i], errwidth=1)
    ax[i].set_title('rating by {}'.format(dict_plt[i]), size=15)
    ax[i].set_ylabel('')


# Who would have guessed? Higher rated recipes have more calories & more fat! This genious discovery has uncovered the source of all obesity problems, people like 'bad' food! Again this was discovered by me & not a previously known, obvious fact.
# 
# Interestingly, the 5-star ratings see a decrease in both. Maybe I can find a few recipes that taste good without getting fat(ter). I'm looking for recipes with low calorie count but with decent amount of protein.
# 
# 
# Note: Totally realize that low calorie doesn't mean healthy, just using the 2,000 - 2,500 daily calorie intake rule as a proxy for 'healthy.'

# In[ ]:


five_star = recipes.loc[recipes['rating'] == 5]

print('We have {:,} 5-star recipes to choose from'.format(len(recipes.loc[recipes['rating'] == 5])))


# In[ ]:


a = pd.qcut(five_star['calories'], [0,.33,.66,1], labels=['low cal','med cal', 'high cal']).rename('cal_bin')

five_star = five_star.join(a)


# In[ ]:


low_cal = five_star.loc[five_star['cal_bin'] == 'low cal']

plt.scatter(x='calories', y='protein', s=low_cal['fat']*5, data=low_cal)

plt.xlabel('Calories')
plt.ylabel('Protein')
plt.axhspan(ymin=20, ymax=25, xmin=.48, xmax=.6, alpha=.2, color='r')
plt.axhspan(ymin=27, ymax=34, xmin=.7, xmax=.9, alpha=.4, color='r')


# In[ ]:


#light red box from chart above
low_cal.loc[(low_cal['protein'] > 20) & (low_cal['calories'] < 160)]


# In[ ]:


#dark red box from chart above
low_cal.loc[low_cal['protein'] > 27]


# With the two tables above, I've found some alternatives to Taco Bell. Reading through, not all are options (i.e Turkey Stock) but there are some options.
# 
# Now I have a list of recipes that I can look up the recipes on http://www.epicurious.com/recipes-menus 
# 
# And if all of these options fail, Taco Bell is only a 5-minute drive away.
# 
# Note: Going through this data, it's clear that there are some abnormalities. Like duplicate entries (Giblet or Turkey Stock) or abnormally high values (sodium in Roast Turkey). Instead of trying to prune/clean further, I'll just trust the 'eye-test' when looking at actual recipes.
