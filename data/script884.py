
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import geojson

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors

import seaborn as sns

cmap = sns.diverging_palette(220, 15, as_cmap=True)


# In[ ]:


# Reading Data
df_train = pd.read_csv('../input/favorita-grocery-sales-forecasting/train.csv')
df_items = pd.read_csv('../input/favorita-grocery-sales-forecasting/items.csv')
df_stores = pd.read_csv('../input/favorita-grocery-sales-forecasting/stores.csv')
df_train['date'] = pd.to_datetime(df_train['date'])
df_train.set_index('date', inplace=True)


# In[ ]:


# To make it more manageable, I'm just taking the data from the current year for the a further analysis
df_2017 = df_train[df_train.index>'2017-1-1']
df_2017 = pd.merge(df_2017, df_items, on='item_nbr', right_index=True)
df_2017 = pd.merge(df_2017, df_stores, on='store_nbr', right_index=True)


# ## Introduction

# This kernel is not intended to perform data analysis from the perspective of its numerical nature, but attempts to get insights and information about the data by observing the Corporación Favorita business. As the title might suggest, here I'm trying to use all the senses (at least the most part) to create intuition about the data. My only focus is to foment new ideias, discussion and try to create a more rich atmosphere for the Feature Engineering part.
# 
# Because of the size of the training set, I performed a simplification and considered only the portion of the data related to 2017. I thought that this would be the correct approach for two main  reasons:
# * The year 2017 has the most complete and largest portfolio of products;
# * The most recent year can tell us more about the consumption habits of clients.
# 
# To plot the map of Ecuador, I used the file ecuardor_geojson ([https://www.kaggle.com/victorgrobberio/ecuardorgeojson](http://)).

# ## Business Analysis

# Check the portfolio and number of stores over time. This is a very interesting proxy for the size of the series. Newer products or products from newer stores are shorter than the oldest ones. This "lack of information" might have a great impact to our modeling algorithms.
# The major part of products has its beginning dating 2015-2016, which implies that we may have the most part of the series with "2 years long".

# In[ ]:


sns.set_style("whitegrid")

func = lambda df_grouped: len(df_grouped.unique())
df_monthly = df_train.groupby(pd.TimeGrouper('MS')).agg({'item_nbr': func,
                                                         'store_nbr': func})

ax = df_monthly.plot(subplots=True, layout=(1,2), figsize=(13,5), legend=False, linewidth=3, colormap=cmap)
ax[0][0].set_title("Favorita's Portfolio size Evolution")
ax[0][0].set_ylabel("Qtd. of different SKUs")
ax[0][1].set_title("Favorita's Growth")
ax[0][1].set_ylabel("Number of stores")
plt.show()


# To see the characteristics of sales data, here follows to histograms. The first one indicates the length os the series of each "product-store". Its possible to observe that the major part of the "product-store" has long series. Actually many product-store have almost 5 years of data records.
# 
# Another import point is how intermitent are the series. This is illustrated by the second histogram. Note that the major part of the series is quite complete (which means that the major part of the "product-store" sells everyday and we have this information). There are just a few intermitent series.

# In[ ]:


def func_tag(df):
    try:
        last_day = df.index.max()
        first_day = df.index.min()

        period_len = (last_day - first_day).days
        fill_rate = len(df) / period_len

        return pd.Series({'period_len': period_len, 'fill_rate': fill_rate})
    except:
        return pd.Series({'period_len': np.nan, 'fill_rate': np.nan})
    
df_train.drop(['id', 'unit_sales', 'onpromotion'], axis=1, inplace=True)
df_train = df_train.groupby(['store_nbr', 'item_nbr']).apply(func_tag).reset_index()
df_train['fill_rate'] = 100 * df_train['fill_rate']
df_train[df_train['fill_rate']>100] = 100

sns.set_style("whitegrid")
fig, axes = plt.subplots(nrows=2, figsize=(7, 12), sharex=False)

df_train[['period_len']].hist(ec='black', ax=axes[0])
axes[0].set_title('Distribution of the series length')
axes[0].set_xlabel('Days')

df_train[['fill_rate']].hist(ec='black', ax=axes[1])
axes[1].set_title('Distribution of the series fill rate')
axes[1].set_xlabel('Fill rate (%)')
plt.show()

del df_train


# Altought it's a big corporation, the Corporación Favorita isn't present all over the country. Here we can see it's total unit sales per state. It's mandatory to say that this is a very simplified analysis, because I'm not considering negative sales and I'm using the unit sales ignoring the fact that some products aren't sold in "units" but in kilograms (for example). 
# Because of the sales from the capital (Quito) were much higher than the other cities, I used the logarithm to create a more heterogeneous heatmap.

# In[ ]:


sns.set_style("white")
df_grouped_state = df_2017.groupby([pd.TimeGrouper('D'), 'state']).agg({'unit_sales':sum}).reset_index()

x = df_grouped_state.groupby('state').agg({'unit_sales': sum}).reset_index()
x['state'] = x['state'].apply(lambda state_name: state_name.upper())

with open("../input/ecuardorgeojson/ecuador.geojson") as json_file:
    json_data = geojson.load(json_file)

fig, ax = plt.subplots(figsize=(10,9))

patches = []
for feature in json_data['features']:
    state = feature['properties']['DPA_DESPRO'].upper()
    city = feature['properties']['DPA_DESCAN'].upper()
    poly = Polygon(np.array(feature['geometry']['coordinates'][0][0]), closed=True)
    patches.append({'state': state, 'city': city, 'poly': poly})
df_patches = pd.DataFrame.from_records(patches)
df_patches = pd.merge(df_patches, x, on='state', how='left')
df_patches = df_patches.fillna(1)

# cmap=matplotlib.cm.RdBu_r
p = PatchCollection(df_patches['poly'])#, cmap=matplotlib.cm.RdBu_r)

norm = colors.Normalize()
p.set_facecolor(cmap(norm(np.log(df_patches['unit_sales']))))

mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array(df_patches['unit_sales'])
plt.colorbar(mapper, shrink=0.4, label="2017's Sales Units (log)")

ax.add_collection(p)
ax.set_xlim(-81.2, -75)
ax.set_ylim(-5.1, 1.3)
plt.title('Heatmap - Total unit sales (Log Scale) per State', fontsize=14)
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelleft='off', labelbottom='off')
plt.show()


# Here we can see how the sales was distributed in the states and cities. This might indicate that Corporación Favorita is trying to compete for the big markets and could have a good market share there. To validate this idea, we have to check the most economic data from the cities from Ecuador.

# In[ ]:


sns.set_style("whitegrid")
fig, axes = plt.subplots(nrows=2, figsize=(7, 12), sharex=False)

df = df_grouped_state.pivot(index='date', columns='state', values='unit_sales').sum()
df.sort_values(ascending=True).plot(kind='barh', colormap=cmap, ax=axes[0])
axes[0].set_title('Total sales in 2017 - per State')
plt.xlabel('Total units sold')

df_grouped_city = df_2017.groupby('city').agg({'unit_sales':sum}).sort_values('unit_sales', ascending=True)
df_grouped_city.plot(kind='barh', colormap=cmap, ax=axes[1], legend=False)
axes[1].set_title('Total sales in 2017 - per City')
plt.xlabel('Total units sold')
plt.show()


# To give us a more intuitive feeling about the sales behavior, I'm comparing the sales behavior of the corporation per state. Two interesting insights come out: 
# * The corporation isn't growing their market share or improving sales force in 2017. The total sales units have a static behavior all over the year (for the engineers, the sales are in their steady state);
# * Visually one might realize that sales have a very interesing seasonality.

# In[ ]:


ax = np.log(df_grouped_state.pivot(index='date', columns='state', values='unit_sales')).plot(figsize=(14,7), colormap=cmap, linewidth=2)
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)
plt.title('Sales evolution - 2017', fontsize=14)
plt.ylabel('Total units sold')
plt.grid(which='minor')
plt.show()


# In the next figure we can verify how the observed seasonality affects the sales throughout the week. Here is very clear that:
# * Saturday and Sunday are the most important days for the major part of the stores.
# 
# Two questions that we need to answer now are:
# * Why is Thursday the less significant day for sales for almost all stores?
# * What do the stores with big sales on the beginning of the week (Mon, Tue, Wed) have in common?
# 

# In[ ]:


df_2017['weekday'] = [item.weekday() for item in df_2017.index]
df_stores_weekday = df_2017.groupby(['store_nbr', 'weekday']).sum().reset_index()
df_stores_weekday = df_stores_weekday.pivot(index='weekday', columns='store_nbr', values='unit_sales')

plt.figure(figsize=(15,4))
ax = sns.heatmap(df_stores_weekday.apply(lambda col: (col-min(col))/(max(col)-min(col)), axis=0), 
                 cmap=cmap, cbar_kws={'label': 'Normalized Sale'})
ax.set_ylabel('Weekday')
ax.set_xlabel('Store number')
ax.set_yticklabels(['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'])
ax.set_title('Sales intensity of each store - per weekday')
plt.yticks(rotation=0)
plt.show()


# I've selected the stores that have the most significant day on Monday, Tuesday or Wednesday. (I've got no insight here)

# In[ ]:


different_stores = df_stores_weekday.apply(lambda col: col.idxmax() if col.idxmax()<3 else None, axis=0).dropna().index
df_stores[df_stores['store_nbr'].isin(different_stores)].set_index('store_nbr')


# The same approach is applied to verify the seasonality of products (in this case, data from 2016 was also used)

# In[ ]:


df_train = pd.read_csv('../input/favorita-grocery-sales-forecasting/train.csv');
df_train['date'] = pd.to_datetime(df_train['date'])
df_train.set_index('date', inplace=True)

df = df_train[df_train.index>='2016-1-1'].drop(['id', 'store_nbr', 'onpromotion'], axis=1)
df = pd.merge(df, df_items[['item_nbr', 'class']], on='item_nbr', right_index=True).drop(['item_nbr'], axis=1)
df['month'] = [item.month for item in df.index]
df_item_month = df.groupby(['class', 'month']).mean().reset_index()
df_item_month = df_item_month.pivot(index='month', columns='class', values='unit_sales')

plt.figure(figsize=(10,15))
ax = sns.heatmap(df_item_month.apply(lambda col: (col-min(col))/(max(col)-min(col)), axis=0).T, 
                 cmap=cmap, cbar_kws={'label': 'Normalized Sale'})
ax.set_ylabel('Product Class')
ax.set_xlabel('Month')
ax.set_title("Sales intensity of each product's class - per month")
plt.show()


# Product classes and family market share can show us information about offer and demand. In other words: what does the corporation is good at selling. Observe that "food" (this could be used as a feature in the future) represent the most part of the units sold by the corp. Personal care articles might have their importance, but doesn't represent a major share.

# In[ ]:


sns.set_style("whitegrid")
df = df_2017.groupby('family').agg({'unit_sales': sum}).sort_values('unit_sales', ascending=False)

OTHERS = df.iloc[10:].sum()
df.drop(df.iloc[10:].index.tolist(), inplace=True)
df.loc['OTHERS'] = OTHERS

ax = df.plot.pie(y='unit_sales', figsize=(6, 6), colormap=cmap)
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, bbox_to_anchor=(1.3, 0.8), loc=2, borderaxespad=0., fontsize=12)
plt.ylabel(' ')
plt.title('Market share per product family (2017)', fontsize=14)
plt.show()


# In[ ]:


sns.set_style("whitegrid")
df = df_2017.groupby('class').agg({'unit_sales': sum}).sort_values('unit_sales', ascending=False)

OTHERS = df.iloc[10:].sum()
df.drop(df.iloc[10:].index.tolist(), inplace=True)
df.loc['OTHERS'] = OTHERS

ax = df.plot.pie(y='unit_sales', figsize=(6, 6), colormap=cmap)
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, bbox_to_anchor=(1.3, 0.8), loc=2, borderaxespad=0., fontsize=12)
plt.ylabel(' ')
plt.title('Market share per product class (2017)', fontsize=14)
plt.show()


# To check the influence of promotion actions on sales, I've first normalized the sales data for each "product-store", fitting the values on the [0, 1] range (no negative value was considered here). For this analysis I've also considered just the "product-store"s that were on promotion at some moment.

# In[ ]:


func_norm = lambda df: (df['unit_sales'] - df['unit_sales'].min())/(df['unit_sales'].max() - df['unit_sales'].min())
df_2017 = df_2017[['unit_sales', 'item_nbr', 'store_nbr', 'onpromotion']].set_index(['onpromotion'])
df_2017 = df_2017[df_2017['unit_sales']>0] # Take just positive sales
df_2017 =  df_2017.groupby(['store_nbr', 'item_nbr']).apply(func_norm).reset_index() # Normalize sales item-store

def func(df):
    if len(df['onpromotion'].unique())==2:
        return df
    else:
        pass
    
teste = df_2017.groupby(['item_nbr', 'store_nbr']).apply(func).dropna()

sns.boxplot(x="onpromotion", y="unit_sales", data=teste)
plt.ylabel('unit_sales normalized')
plt.show()

