
# coding: utf-8

# This notebook tries to describe the trend of Japan trade statistics and relationship among each world areas. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# `year_latest` is a statistics which collects the value of the goods between Japan and the other countries. These records include 
# 
# * `VY` is a value of goods which was traded
# * `hs2` is a top category which was categorized hierarchically by [HS code](http://www.foreign-trade.com/reference/hscode.htm).
# * `hs4` is a second categorization as well as hs2
# * `hs6` is a third one
# * `Year` specifies a records is a statistics of the year.

# In[ ]:


# The trade statistics records for each year from 1988 to 2016
df = pd.read_csv('../input/year_latest.csv')

df.head()


# In[ ]:


# The area and country codes
country_df = pd.read_csv('../input/country_eng.csv')

country_df.head()


# # Trade with each area

# In[ ]:


# Joined trade records with Country name
joined_df = pd.merge(df, country_df, on=['Country'])

# Plot VY transition
def plot_vys(column_value, column_name):
    data = vys[vys[column_name] == column_value]
    plt.plot(data['Year'], data['VY'], label=column_value)
    
areas = np.unique(country_df['Area'].values)
    
grouped_by_area = joined_df[['Year', 'VY', 'Area']].groupby(['Area', 'Year'], as_index=False)
vys = grouped_by_area.aggregate(np.sum)

plt.figure(figsize=(7.5, 8))
for area in areas:
    plot_vys(area, 'Area')
    
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# This graph shows Asia area is the biggest trade partner for Japan and the most growing partner. One thing to note here is that the VY is decreasing at 2008 for all areas. This trend was caused by "Reman Shock" which hit all around the world. And the VY value does not come back at the same level to before "Reman Shock" except for Asia area. 
# 
# In addition there are two time span when trade value was decreased in the plot between Middle East.  Japan depends on the oil imported from mainly Middle East. So the trend was caused by crude oil weaker trend recently. Let's confirm the main goods from Middle East.
# 
# # Trade with Middle East

# In[ ]:


hs2_df = pd.read_csv('../input/hs2_eng.csv')
middle_east_df = joined_df[joined_df['Area'] == 'Middle_East']

grouped_by_hs2 = joined_df[['Year', 'VY', 'hs2']].groupby(['hs2','Year'], as_index=False)

vys = grouped_by_hs2.aggregate(np.sum)
vys = pd.merge(vys, hs2_df, on=['hs2'])

main_goods = vys[vys['VY'] >  0.4 * 1e10]
hs2_names = np.unique(main_goods['hs2_name'].values)

for hs2_name in hs2_names:
    plot_vys(hs2_name, 'hs2_name')

plt.legend()
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
plt.show()


# We can see several goods are traded between Japan and Middle East areas. As expected the most traded goods after 2010 is "Fuels". "Fuels" is the only good which trade was decreased from 2014's. 

# In[ ]:


areas = np.unique(country_df['Area'].values)
    
grouped_by_area = joined_df[['Year', 'VY', 'Area']].groupby(['Area', 'Year'], as_index=False)
vys = grouped_by_area.aggregate(np.sum)
plt.figure(figsize=(12.0, 8))
g = seaborn.boxplot(x="Area", y="VY", data=vys, palette="PRGn")
g.set_xticklabels(labels=areas, rotation=45)


# We can see the trade growth (the difference between mean value and high deviation) of Asia are is the most. On the other hand the trade between Japan and North America or Western Europe relatively stable though they are ranked as 2nd and 3rd position. The growth of Middle East is relatively high compared with North America and Western Europe.
# 
# # Trade with Asian countries

# In[ ]:


grouped_by_country = joined_df[['Year', 'VY', 'Country_name']].groupby(['Country_name','Year'], as_index=False)

vys = grouped_by_country.aggregate(np.sum)


# In[ ]:


plt.figure(figsize=(7.5, 8))

asia_countries = country_df[country_df['Area'] == 'Asia']['Country_name']
    
for country in asia_countries:
    plot_vys(country, 'Country_name')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# # Trade with China
# 
# China is the biggest trade partner for Japan. Let's take a look into detail of trade with this country.

# In[ ]:


# Trade values with China
china_df = joined_df[joined_df['Country_name'] == 'People\'s_Republic_of_China']
grouped_by_hs2 = china_df[['Year', 'VY', 'hs2']].groupby(['hs2','Year'], as_index=False)

vys = grouped_by_hs2.aggregate(np.sum)
vys = pd.merge(vys, hs2_df, on=['hs2'])
vys.head()


# In[ ]:


main_goods = vys[vys['VY'] >  0.15 * 1e10]
hs2_names = np.unique(main_goods['hs2_name'].values)

for hs2_name in hs2_names:
    plot_vys(hs2_name, 'hs2_name')

plt.legend(bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.)
plt.show()


# We can see the biggest goods traded with Chine is machinery type goods.

# In[ ]:


minor_goods = vys[vys['VY'] <  0.1 * 1e6]
hs2_names = np.unique(minor_goods['hs2_name'].values)

pivotted_vys = vys[vys['hs2_name'].isin(hs2_names)]
pivotted_vys = pivotted_vys.pivot('Year' ,'hs2_name', 'VY')
plt.figure(figsize=(12.0, 8))
g = seaborn.heatmap(pivotted_vys, annot=True, linewidths=.5)
g.set_xticklabels(labels=hs2_names, rotation=45)


# Minor goods traded with China such as Aircraft or Corks are not changing so much.

# # Conclusion
# 
# We can see some interesting points from these data set.
# 
# * The trade value was decreased at 2008 all over the world.
# * The main goods traded between Japan and Middle East is crude oil.
# * The trade value of crude oil was decreasing recently due to low price of oil.
# * China is the biggest trade partner for Japan.
# * The main goods traded between Japan and China is machinery.
