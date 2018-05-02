
# coding: utf-8

# In[ ]:


## imports
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot, iplot, init_notebook_mode
import warnings
from subprocess import check_output
from IPython.core.display import display, HTML
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('precision', 4)
warnings.simplefilter('ignore')
init_notebook_mode()
display(HTML("<style>.container { width:100% !important; }</style>"))

get_ipython().run_line_magic('matplotlib', 'inline')


# ![](https://sc01.alicdn.com/kf/UT8Y1g1XKRXXXagOFbXU/cocoa-beans.jpg_350x350.jpg)
# ## Chocolate Bar Ratings
# Chocolate is one of the most popular candies in the world. Each year, residents of the United States collectively eat more than 2.8 billions pounds. However, not all chocolate bars are created equal! This dataset contains expert ratings of over 1,700 individual chocolate bars, along with information on their regional origin, percentage of cocoa, the variety of chocolate bean used and where the beans were grown.
# 
# ### Flavors of Cacao Rating System:
# - 5= Elite (Transcending beyond the ordinary limits)
# - 4= Premium (Superior flavor development, character and style)
# - 3= Satisfactory(3.0) to praiseworthy(3.75) (well made with special qualities)
# - 2= Disappointing (Passable but contains at least one significant flaw)
# - 1= Unpleasant (mostly unpalatable)
# 
# ### Data description
# - __Company  (Maker-if known)__ - Name of the company manufacturing the bar.
# - __Specific Bean Origin or Bar Name__ - The specific geo-region of origin for the bar.
# - __REF__ - <font color='red'>Help us describe this column...</font> __What is it?__
# - __Review Date__ - Date of publication of the review.
# - __Cocoa Percent__ - Cocoa percentage (darkness) of the chocolate bar being reviewed.
# - __Company Location__ - Manufacturer base country.
# - __Rating __- Expert rating for the bar.
# - __Bean Type__ - The variety (breed) of bean used, if provided.
# - __Broad Bean Origin__ - The broad geo-region of origin for the bean.
# 
# ### Table of content
# 1. [Data preparation + EDA](#eda)
# 2. [Feature engineering](#fe)
# 3. [Data visualization](#dv)
# 4. [WHAT IS REF?!](#ref)

# ## Author's summary
# Hello everyone! This is my very first kernel in Kaggle platform. I hope that you will enjoy my work.  Thx.

# <a id="eda">
# #### 1. Data preparation + EDA

# In[ ]:


## Load data
choko = pd.read_csv('../input/flavors_of_cacao.csv')
choko.shape # How many revies we have


# In[ ]:


# Explore first 5 rows
choko.head().T


# In[ ]:


# Explore description
choko.describe(include='all').T


# In[ ]:


# Explore datatypes
choko.dtypes


# In[ ]:


## Before we continue - rename some columns, 
original_colnames = choko.columns
new_colnames = ['company', 'species', 'REF', 'review_year', 'cocoa_p',
                'company_location', 'rating', 'bean_typ', 'country']
choko = choko.rename(columns=dict(zip(original_colnames, new_colnames)))
## And modify data types
choko['cocoa_p'] = choko['cocoa_p'].str.replace('%','').astype(float)/100
choko.head()


# In[ ]:


# Explore description
choko.describe(include='all').T


# In[ ]:


## Look at most frequent species
choko['species'].value_counts().head(10)


# In[ ]:


## Is where any N/A values in origin country?
choko['country'].isnull().value_counts()


# In[ ]:


## Replace origin country
choko['country'] = choko['country'].fillna(choko['species'])
choko['country'].isnull().value_counts()


# In[ ]:


## Look at most frequent origin countries
choko['country'].value_counts().head(10)


# In[ ]:


## Wee see that a lot of countries have ' ' value - means that this is 100% blend. Let's look at this
choko[choko['country'].str.len()==1]['species'].unique()


# In[ ]:


## Is there another way to determine blends?
choko[choko['species'].str.contains(',')]['species'].nunique()


# In[ ]:


## Is there any misspelling/reduction?
choko['country'].sort_values().unique()


# In[ ]:


## Text preparation (correction) func
def txt_prep(text):
    replacements = [
        ['-', ', '], ['/ ', ', '], ['/', ', '], ['\(', ', '], [' and', ', '], [' &', ', '], ['\)', ''],
        ['Dom Rep|DR|Domin Rep|Dominican Rep,|Domincan Republic', 'Dominican Republic'],
        ['Mad,|Mad$', 'Madagascar, '],
        ['PNG', 'Papua New Guinea, '],
        ['Guat,|Guat$', 'Guatemala, '],
        ['Ven,|Ven$|Venez,|Venez$', 'Venezuela, '],
        ['Ecu,|Ecu$|Ecuad,|Ecuad$', 'Ecuador, '],
        ['Nic,|Nic$', 'Nicaragua, '],
        ['Cost Rica', 'Costa Rica'],
        ['Mex,|Mex$', 'Mexico, '],
        ['Jam,|Jam$', 'Jamaica, '],
        ['Haw,|Haw$', 'Hawaii, '],
        ['Gre,|Gre$', 'Grenada, '],
        ['Tri,|Tri$', 'Trinidad, '],
        ['C Am', 'Central America'],
        ['S America', 'South America'],
        [', $', ''], [',  ', ', '], [', ,', ', '], ['\xa0', ' '],[',\s+', ','],
        [' Bali', ',Bali']
    ]
    for i, j in replacements:
        text = re.sub(i, j, text)
    return text


# In[ ]:


choko['country'].str.replace('.', '').apply(txt_prep).unique()


# In[ ]:


## Replace country feature
choko['country'] = choko['country'].str.replace('.', '').apply(txt_prep)


# In[ ]:


## Looks better
choko['country'].value_counts().tail(10)


# In[ ]:


## How many countries may contain in Blend?
(choko['country'].str.count(',')+1).value_counts()


# In[ ]:


## Is there any misspelling/reduction in company location?
choko['company_location'].sort_values().unique()


# In[ ]:


## We need to make some replacements
choko['company_location'] = choko['company_location'].str.replace('Amsterdam', 'Holland').str.replace('U.K.', 'England').str.replace('Niacragua', 'Nicaragua').str.replace('Domincan Republic', 'Dominican Republic')

choko['company_location'].sort_values().unique()


# In[ ]:


## Is there any misspelling/reduction in company name?
choko['company'].str.lower().sort_values().nunique() == choko['company'].sort_values().nunique()


# <a id="fe">
# #### 2. Feature engineering

# In[ ]:


## Let's define blend feature
choko['is_blend'] = np.where(
    np.logical_or(
        np.logical_or(choko['species'].str.lower().str.contains(',|(blend)|;'),
                      choko['country'].str.len() == 1),
        choko['country'].str.lower().str.contains(',')
    )
    , 1
    , 0
)
## How many blends/pure cocoa?
choko['is_blend'].value_counts()


# In[ ]:


## Look at 5 blends/pure rows
choko.groupby('is_blend').head(5)


# In[ ]:


## Define domestic feature
choko['is_domestic'] = np.where(choko['country'] == choko['company_location'], 1, 0)
choko['is_domestic'].value_counts()


# <a id="dv">
# #### 3. Data Visualization

# In[ ]:


## Look at distribution of Cocoa %
fig, ax = plt.subplots(figsize=[16,4])
sns.distplot(choko['cocoa_p'], ax=ax)
ax.set_title('Cocoa %, Distribution')
plt.show()


# In[ ]:


## Look at distribution of rating
fig, ax = plt.subplots(figsize=[16,4])
for i, c in choko.groupby('is_domestic'):
    sns.distplot(c['cocoa_p'], ax=ax, label=['Not Domestic', 'Domestic'][i])
ax.set_title('Cocoa %, Distribution, hue=Domestic')
ax.legend()
plt.show()


# In[ ]:


## Look at distribution of rating
fig, ax = plt.subplots(figsize=[16,4])
sns.distplot(choko['rating'], ax=ax)
ax.set_title('Rating, Distribution')
plt.show()


# In[ ]:


## Look at distribution of rating
fig, ax = plt.subplots(figsize=[16,4])
for i, c in choko.groupby('is_domestic'):
    sns.distplot(c['rating'], ax=ax, label=['Not Domestic', 'Domestic'][i])
ax.set_title('Rating, Distribution, hue=Domestic')
ax.legend()
plt.show()


# In[ ]:


## Look at boxplot over the countries, even Blends
fig, ax = plt.subplots(figsize=[6, 16])
sns.boxplot(
    data=choko,
    y='country',
    x='rating'
)
ax.set_title('Boxplot, Rating for countries (+blends)')


# In[ ]:


## But hot we can see what country is biggest contributor in rating?
choko_ = pd.concat([pd.Series(row['rating'], row['country'].split(',')) for _, row in choko.iterrows()]
         ).reset_index()
choko_.columns = ['country', 'rating']
choko_['mean_rating'] = choko_.groupby(['country'])['rating'].transform('mean')

## Look at boxplot over the countries (contributors in blends)
fig, ax = plt.subplots(figsize=[6, 16])
sns.boxplot(
    data=choko_.sort_values('mean_rating', ascending=False),
    y='country',
    x='rating'
)
ax.set_title('Boxplot, Rating for countries (contributors)')


# In[ ]:


choko_.groupby(['country'])['rating'].mean().sort_values(ascending=False).head(10)


# In[ ]:


choko_ = pd.concat([pd.Series(row['cocoa_p'],
                              row['country'].split(',')) for _, row in choko.iterrows()]
         ).reset_index()
choko_.columns = ['country', 'rating']
choko_['mean_rating'] = choko_.groupby(['country'])['rating'].transform('mean')


# In[ ]:


## Look at boxplot over the countries (contributors in blends)
choko_ = pd.concat([pd.Series(row['cocoa_p'], row['country'].split(',')) for _, row in choko.iterrows()]
         ).reset_index()
choko_.columns = ['country', 'cocoa_p']

## Look at boxplot over the countries (contributors in blends)
fig, ax = plt.subplots(figsize=[6, 16])
sns.boxplot(
    data=choko_,
    y='country',
    x='cocoa_p'
)
ax.set_title('Boxplot, Cocoa %, for countries (contributors)')


# In[ ]:


## Prepare full tidy choko_ dataframe
def choko_tidy(choko):
    data = []
    for i in choko.itertuples():
        for c in i.country.split(','):
            data.append({
                'company': i.company,
                'species': i.species,
                'REF': i.REF,
                'review_year': i.review_year,
                'cocoa_p': i.cocoa_p,
                'company_location': i.company_location,
                'rating': i.rating,
                'bean_typ': i.bean_typ,
                'country': c,
                'is_blend': i.is_blend,
                'is_domestic': i.is_domestic
            })
    return pd.DataFrame(data)
        
choko_ = choko_tidy(choko)
print(choko_.shape, choko.shape)
choko_.head()


# In[ ]:


## Look at rating by company location
fig, ax = plt.subplots(figsize=[6, 16])
sns.boxplot(
    data=choko,
    y='company_location',
    x='rating'
)
ax.set_title('Boxplot, Rating by Company location')


# In[ ]:


## What better? Domestic Or not?
fig, ax = plt.subplots(figsize=[6, 6])
sns.boxplot(
    data=choko,
    x='is_domestic',
    y='rating',
)
ax.set_title('Boxplot, Rating by Domestic')


# In[ ]:


## What better? Pure or blend?
fig, ax = plt.subplots(figsize=[6, 6])
sns.boxplot(
    data=choko,
    x='is_blend',
    y='rating',
)
ax.set_title('Boxplot, Rating by Blend/Pure')


# ### Hmmm
# - Blend is better
# - Domestic is worse 

# In[ ]:


choko_.head()


# In[ ]:


## Look at goodsflow
flow = pd.crosstab(
    choko_['company_location'],
    choko_['country']
)
flow['tot'] = flow.sum(axis=1)
flow = flow.sort_values('tot', ascending=False)
flow = flow.drop('tot', axis=1)

fig, ax = plt.subplots(figsize=[10,6])
sns.heatmap(flow.head(20), cmap='Reds', linewidths=.5)
ax.set_title('Goods Flow from origin to Company location')


# - Biggest manufactorer country - __U.S.A__
# - Biggest ofigin coutry is __Equador__, and also biggest domestic manufactorer
# - Also a lot of domestics from Colombia, Brazil, Madagascar, Venezuela

# In[ ]:


## What about quality(rating)
## Look at goodsflow
flow = pd.crosstab(
    choko_['company_location'],
    choko_['country'],
    choko_['rating'], aggfunc='mean'
)
flow['tot'] = flow.sum(axis=1)
flow = flow.sort_values('tot', ascending=False)
flow = flow.drop('tot', axis=1)

fig, ax = plt.subplots(figsize=[10,6])
sns.heatmap(flow.head(20), cmap='RdBu_r', linewidths=.5)
ax.set_title('Goods Flow from origin to Company location, mean rating')


# - __USA__ have the biggest flow of manufactory but mean quality 3-3.5
# - In __England__ and __Canada__ very good mean rating

# In[ ]:


## What about quality(rating) is case of years
## Look at goodsflow
flow = pd.crosstab(
    choko_['company_location'],
    choko_['review_year'],
    choko_['rating'], aggfunc='mean'
)
flow['tot'] = flow.sum(axis=1)
flow = flow.sort_values('tot', ascending=False)
flow = flow.drop('tot', axis=1)

fig, ax = plt.subplots(figsize=[10,6])
sns.heatmap(flow.head(20), cmap='RdBu_r', linewidths=.5)
ax.set_title('Goods Flow from Company location, mean rating by years')


# #### Wow!
# - __USA__, __Australia__ mean rating is getting better and better
# - __ Canada__ rating is always good

# In[ ]:


## Look the same data at the tsplot
flow.T.head()


# In[ ]:


flow_ = flow.T
## Preprocess
# for c in flow_.columns:
#     flow_[c] = flow_[c] - flow_[c].dropna().iloc[0]

fig, ax = plt.subplots(figsize=[16,8])
for c in choko_['company_location'].value_counts().head(10).index:
    ax.plot(flow_.index, flow_[c], label=c)
ax.legend(ncol=1, loc=4)
ax.set_title('Timeline of Cocoa Rating by Company location')
plt.show()


# In[ ]:


## What country manufacture the best blend|pure?
blends = pd.crosstab(
    choko_['company_location'],
    choko_['is_blend'],
    choko_['rating'], aggfunc='mean'
)
blends['tot'] = blends.max(axis=1)
blends = blends.sort_values('tot', ascending=False)
blends = blends.drop('tot', axis=1)

fig, ax = plt.subplots(figsize=[10,6])
sns.heatmap(blends.head(25), cmap='RdBu_r', linewidths=.5)
ax.set_title('Best Manufactorer from Company location, mean rating blend/pureness')


# - __Lithuania, And Bolivia__ blends better than pure
# - __Poland, Venezuela and Scotland__ blends are sucks
# - __Iceland__ dont produce blends
# - __Chile__ the best

# In[ ]:


## What country manufacture the best blend|pure?
dom = pd.crosstab(
    choko_['company_location'],
    choko_['is_domestic'],
    choko_['rating'], aggfunc='mean'
)
dom['tot'] = dom.max(axis=1)
dom = dom.sort_values('tot', ascending=False)
dom = dom.drop('tot', axis=1)

fig, ax = plt.subplots(figsize=[10,6])
sns.heatmap(dom.head(25), cmap='RdBu_r', linewidths=.5)
ax.set_title('Best Manufactorer from Company location, mean rating by Domestic or not')


# - __Equador, Nicaragua__ better not Domestic.
# - Etc

# <a id="ref">
# ### 4. What is REF?!

# In[ ]:


## What is REF?
sns.heatmap(choko_.corr(), cmap='coolwarm')


# Hmmm.. REF is highli correlated with review_year

# In[ ]:


## Look at REF distribution
sns.distplot(choko_['REF'])


# In[ ]:


## Look at YEAR distribution
sns.distplot(choko_['review_year'])


# In[ ]:


fig, [ax1, ax2] = plt.subplots(2, 1, figsize=[16,8])
ax1.plot(choko_['review_year'])
ax2.plot(choko_['REF'])
plt.show()


# In[ ]:


sns.boxplot(
    data=choko_,
    x='review_year',
    y='REF'
)


# In[ ]:


choko_.groupby('review_year').agg({'REF': ['min', 'max', 'mean', 'prod', 'nunique', 'count']})


# In[ ]:


choko['REF'].nunique(), choko.shape


# - REF is highly correlated to review_year
# - Sometimes intersects over the year, but never overlaps year cuts
# - I think that REF is increment id of  bunch of reviews
# 
# Thx,
