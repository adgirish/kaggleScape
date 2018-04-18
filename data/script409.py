
# coding: utf-8

# # Introduction
# 
# Thank you all for your support and time. My goal with this kernel is to provide a broad first look at the weather stations.
# 
# If you want a more exhaustive and detailed look at the data and how to use it, please check out my new kernel here: https://www.kaggle.com/huntermcgushion/exhaustive-weather-eda-file-overview
# <br>
# I would love to hear your thoughts on both this kernel and the one linked above. Thanks again for your time!

# In[9]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from subprocess import check_output
from IPython.display import display
from IPython.core.display import HTML
import plotly
import plotly.offline as py

py.init_notebook_mode(connected=False)
sns.set_style('whitegrid')

# Because the weather data directory is compressed, you can find it like this:
weather_dir = '../input/rrv-weather-data'

print('##### ../input/ Contents:')
print(os.listdir("../input"))

print('\n##### {} Contents:'.format(weather_dir))
print(os.listdir(weather_dir))
# Note that the contents of the below dir consist of another dir of the same name  
print('\n##### {} Contents:'.format('{}/1-1-16_5-31-17_Weather'.format(weather_dir)))
print(os.listdir('{}/1-1-16_5-31-17_Weather'.format(weather_dir)))
print('\n##### {} Contents:'.format('{}/1-1-16_5-31-17_Weather/1-1-16_5-31-17_Weather'.format(weather_dir)))
print(len(os.listdir('{}/1-1-16_5-31-17_Weather/1-1-16_5-31-17_Weather'.format(weather_dir))))


# In[10]:


# air_store_info = pd.read_csv('{}/air_store_info_with_nearest_active_station.csv', index_col=False)
air_store_info = pd.read_csv('{}/air_store_info_with_nearest_active_station.csv'.format(weather_dir))
# hpg_store_info = pd.read_csv('../input/rrv-weather-data/hpg_store_info_with_nearest_active_station.csv', index_col=False)
hpg_store_info = pd.read_csv('{}/hpg_store_info_with_nearest_active_station.csv'.format(weather_dir))

air_store_info['coordinate_count'] = air_store_info.groupby(
    ['latitude', 'longitude']
).latitude.transform('count').astype(int)
hpg_store_info['coordinate_count'] = hpg_store_info.groupby(
    ['latitude', 'longitude']
).latitude.transform('count').astype(int)


# In[11]:


display(air_store_info.head())
display(hpg_store_info.head())


# In[12]:


sns.distplot(air_store_info['station_vincenty'], rug=True, kde=True)
plt.title('AIR - Distances (km) to Stations Distribution')
plt.show()

# NOTE: This one might take a minute to show up
sns.distplot(hpg_store_info['station_vincenty'], rug=True, kde=True)
plt.title('HPG - Distances (km) to Stations Distribution')
plt.show()


# The above distibution plots help visualizing the distances between our stores and nearest stations, but we can do better. 
# <br>
# Below, we're looking at joint plots with scatter plot overlays for both the AIR and HPG stores.
# <br>
# They give a clearer picture of just how many stores (at unique coordinates) are a given distance away from their closest station.

# In[13]:


p = sns.jointplot(x='station_vincenty', y='coordinate_count', data=air_store_info, kind='kde')
plt.title('AIR KDE Joint Plot', loc='left')
p.plot_joint(plt.scatter, c='w', s=30, linewidth=1, marker='x')
plt.show()

p = sns.jointplot(x='station_vincenty', y='coordinate_count', data=hpg_store_info, kind='kde')
plt.title('HPG KDE Joint Plot', loc='left')
p.plot_joint(plt.scatter, c='w', s=30, linewidth=1, marker='x')
plt.show()


# **What Are We Looking At?**
# 
# As we can see from the joint plots, most of the closest weather stations are within about 8 km of the store we are looking at.
# 
# The AIR outliers are: 3 unique store coordinates whose nearest stations are between 10 and 12 km away.
# <br>
# The coordinate_count values seem low, so there are probably under 20 AIR stores that are more than 10 km away from a station.
# 
# The HPG outliers are more numerous and confusing, largely because there are 4,690 HPG stores, compared to AIR's 829.
# <br>
# They seem to be: a little over 100 stores (4 unique coordinates) that are between 12 and 15 km from their nearest station, along with about 10 more that are a whopping 17.5 km away.
# 
# But that's not really specific enough. Let's check the actual data...

# In[14]:


def view_distances(df, distance, cols=['station_vincenty', 'coordinate_count']):
    return df[cols].groupby(cols).filter(
         lambda _: _[cols[0]].mean() > distance
    ).drop_duplicates().sort_values(by=cols).reset_index(drop=True)

display(view_distances(air_store_info, 8))


# For the AIR stores, there are 16 stores (3 unique coordinates) that are between 8 and 8.5 km away from their closest station.
# <br>
# There are also 13 stores (3 unique coordinates) that are more than 10 km away from a station.
# <br>
# The furthest an AIR store is from an active weather station is 11.48 km.

# In[15]:


display(view_distances(hpg_store_info, 10))


# For the HPG stores, we see 116 stores (6 unique coordinates) between 10 and 13.14 km away from a station.
# <br>
# Then, there are 14 stores (1 unique coordinate pair) located 17.18 km from a station.
# 
# # Maps
# Below are a couple maps showing the locations of all AIR and HPG stores, as well as all active and terminated weather stations.
# The maps were originally created using Plot.ly, with MapBox, which made some beautiful maps that very clearly showed the positions of our stores and weather stations. Unfortunately, MapBox requires the use of an access token, which isn't supposed to be made public. So I had to jump through a few hoops to show the plot below without explicitly declaring my MapBox access token, and I'm still not very confident in its security... Nonetheless, I wanted you to see these:

# In[16]:


display(HTML("""<div>
    <a href="https://plot.ly/~hmcgushion/4/?share_key=pArkblYSXlgzZWvBrvqeFH" target="_blank" title="AllStoresAndStations_Plot" style="display: block; text-align: center;"><img src="https://plot.ly/~hmcgushion/4.png?share_key=pArkblYSXlgzZWvBrvqeFH" alt="AllStoresAndStations_Plot" style="max-width: 100%;width: 1000px;"  width="1000" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="hmcgushion:4" sharekey-plotly="pArkblYSXlgzZWvBrvqeFH" src="https://plot.ly/embed.js" async></script>
</div>"""))


# Above, we have all of the AIR store locations in red, and all of the HPG locations in blue.
# <br>
# We also have all of the active weather stations in green, and all of the terminated station in black.
# <br>
# You can zoom and pan and hover over any point to see its latidude, longitude coordinates, and the area (for stores), or the station_id for weather stations.
# 
# The next ScatterMapBox will show all of the stores, with only the nearest weather stations.
# <br>
# Additionally, it shows the distance from each store to its nearest station, along with a line between the store and station to better visualize the distance.
