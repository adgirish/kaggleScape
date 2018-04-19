
# coding: utf-8

# ## Global Air Pollution Measurements
# 
# * [Air Quality Index - Wiki](https://en.wikipedia.org/wiki/Air_quality_index)
# * [BigQuery - Wiki](https://en.wikipedia.org/wiki/BigQuery)
# 
# In this notebook data is extracted from *BigQuery Public Data* assesible exclusively only in *Kaggle*. The BigQurey Helper Object will convert data in cloud storage into *Pandas DataFrame* object. The query syntax is same as *SQL*. As size of data is very high convert entire data to DataFrame is cumbersome. So query is written such that will be readly available for Visualization.
# ***
# >**Baisc attributes of Air quality index** 
# * Measurement units
#     * $ug/m^3$: micro gram/cubic meter 
#     * $ppm$: Parts Per Million
# * Pollutant
#     * $O3$: Ozone gas
#     * $SO2$: Sulphur Dioxed
#     * $NO2$: Nitrogen Dioxed
#     * $PM 2.5$: Particles with an aerodynamic diameter less than $2.5 μm$
#     * $PM 10$: Particles with an aerodynamic diameter less than $10 μm$
#     * $CO$: Carbon monoxide
#  
# **Steps**
# 1. Load Packages
# 2. Bigquery Object
# 3. AQI range and Statistics 
# 4. Distribution of country listed in AQI
# 5. Location
# 6. Air Quality Index value distribution Map veiw
# 7. Pollutant Statistics
# 8. Distribution of pollutant and unit
# 9. Distribution of Source name
# 10. Sample AQI Averaged over in hours
# 11. AQI variation with time
# 12. Country Heatmap
# 13. Animation

# ###  Load packages

# In[134]:


# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins as plugins

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_rows =10
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Bigquery
# BigQuery is a RESTful web service that enables interactive analysis of massively large datasets working in conjunction with Google Storage. It is an Infrastructure as a Service that may be used complementarily with MapReduce.

# In[135]:


# Customized query helper function explosively in Kaggle
import bq_helper

# Helper object
openAQ = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                 dataset_name='openaq')
# List of table
openAQ.list_tables()


# In[136]:


#Schema 
openAQ.table_schema('global_air_quality')


# ### Table display

# In[137]:


openAQ.head('global_air_quality')


# In[138]:


# Summary statics
query = """SELECT value,averaged_over_in_hours
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³'
            """
p1 = openAQ.query_to_pandas(query)
p1.describe()


# # Air Quality Index Range
# * [AQI Range](http://aqicn.org/faq/2013-09-09/revised-pm25-aqi-breakpoints/)
# <center><img src = 'https://campuspress.yale.edu/datadriven/files/2012/03/AQI-1024x634-1ybtu6l.png '><center>
# 
# The range of AQI is 0 - 500, so lets limit data to that range, in previous kernel's these  outlier data points are not removed

# In[139]:


query = """SELECT value,country 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value < 0
            """
p1 = openAQ.query_to_pandas(query)
p1.describe().T


# There are more than 100 value having value less than 0. The lowest value is -999000, which is outlier data point. **Air Quality Meter** is digital a instruments, if meter is show error value then sensor is disconnected or faulty. 

# In[140]:


query2 = """SELECT value,country,pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value > 0
            """
p2 = openAQ.query_to_pandas(query2)
print('0.99 Quantile',p2['value'].quantile(0.99))
p2.describe().T


# In[141]:


p2[p2['value']>10000]


# Country 
# * MK is *Macedonia* [wiki](https://en.wikipedia.org/wiki/Republic_of_Macedonia)
# * CL is *Chile* [Wiki](https://en.wikipedia.org/wiki/Chile)
# >In both the countries some may some natural disaster happend so AQI is very high. 
#  We will disgrad value more than 10000, which are outlier data point 

# ### Distribution of country listed in AQI

# In[142]:


query = """SELECT country,COUNT(country) as `count`
    FROM `bigquery-public-data.openaq.global_air_quality`
    GROUP BY country
    HAVING COUNT(country) >10
    ORDER BY `count`
    """
cnt = openAQ.query_to_pandas_safe(query)
cnt.tail()

plt.style.use('bmh')
plt.figure(figsize=(14,4))
sns.barplot(cnt['country'], cnt['count'], palette='magma')
plt.xticks(rotation=45)
plt.title('Distribution of country listed in data');


# ## Location
# We find find different location where air quality is taken. This location data consist of latitude and logitude, city.

# In[143]:


#Average polution of air by countries
query = """SELECT AVG(value) as `Average`,country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY country
            ORDER BY Average DESC
            """
cnt = openAQ.query_to_pandas(query)


# In[144]:


plt.figure(figsize=(14,4))
sns.barplot(cnt['country'],cnt['Average'], palette= sns.color_palette('gist_heat',len(cnt)))
plt.xticks(rotation=90)
plt.title('Average polution of air by countries in unit $ug/m^3$')
plt.ylabel('Average AQI in $ug/m^3$');


# * Country PL ( Poland) and IN (India) are top pollutor of air
# ***
# ### AQI measurement center

# In[145]:


query = """SELECT city,latitude,longitude,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY latitude,city,longitude   
            """
location = openAQ.query_to_pandas_safe(query)


# In[146]:


#Location AQI measurement center
m = folium.Map(location = [20,10],tiles='Mapbox Bright',zoom_start=2)

# add marker one by on map
for i in range(0,500):
    folium.Marker(location = [location.iloc[i]['latitude'],location.iloc[i]['longitude']],                 popup=location.iloc[i]['city']).add_to(m)
    
m #  DRAW MAP


# We find that thier are many air qulity index measurement unit across -US- and -Europe-. Thier are few measurement center in -African- continent. We are hardly find any measuring center in Mid East, Russia.

# ### Air Quality Index value distribution Map veiw

# In[147]:


query = """SELECT city,latitude,longitude,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY latitude,city,longitude   
            """
location = openAQ.query_to_pandas_safe(query)
location.dropna(axis=0, inplace=True)


# In[148]:


plt.style.use('ggplot')
f,ax = plt.subplots(figsize=(14,10))
m1 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180, llcrnrlat=-90, urcrnrlat=90,
            resolution='c',lat_ts=True)

m1.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m1.fillcontinents(color='grey', alpha=0.3)
m1.drawcoastlines(linewidth=0.1, color="white")
m1.shadedrelief()
m1.bluemarble(alpha=0.4)
avg = location['Average']
m1loc = m1(location['latitude'].tolist(),location['longitude'])
m1.scatter(m1loc[1],m1loc[0],lw=3,alpha=0.5,zorder=3,cmap='coolwarm', c=avg)
plt.title('Average Air qulity index in unit $ug/m^3$ value')
m1.colorbar(label=' Average AQI value in unit $ug/m^3$');


# ### US

# In[149]:


#USA location
query = """SELECT 
            MAX(latitude) as `max_lat`,
            MIN(latitude) as `min_lat`,
            MAX(longitude) as `max_lon`,
            MIN(longitude) as `min_lon`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US' """
us_loc = openAQ.query_to_pandas_safe(query)
us_loc


# In[ ]:


query = """ SELECT city,latitude,longitude,averaged_over_in_hours,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US' AND unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY latitude,city,longitude,averaged_over_in_hours,country """
us_aqi = openAQ.query_to_pandas_safe(query)


# In[ ]:


# USA
min_lat = us_loc['min_lat']
max_lat = us_loc['max_lat']
min_lon = us_loc['min_lon']
max_lon = us_loc['max_lon']

plt.figure(figsize=(14,8))
m2 = Basemap(projection='cyl', llcrnrlon=min_lon, urcrnrlon=max_lon, llcrnrlat=min_lat, urcrnrlat=max_lat,
            resolution='c',lat_ts=True)
m2.drawcounties()
m2.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m2.fillcontinents(color='grey', alpha=0.3)
m2.drawcoastlines(linewidth=0.1, color="white")
m2.drawstates()
m2.bluemarble(alpha=0.4)
avg = (us_aqi['Average'])
m2loc = m2(us_aqi['latitude'].tolist(),us_aqi['longitude'])
m2.scatter(m2loc[1],m2loc[0],c = avg,lw=3,alpha=0.5,zorder=3,cmap='rainbow')
m1.colorbar(label = 'Average AQI value in unit $ug/m^3$')
plt.title('Average Air qulity index in unit $ug/m^3$ of US');


# AQI of US range 0 to 400, most of city data points are within 100
# ### India

# In[ ]:


#INDIA location
query = """SELECT 
            MAX(latitude) as `max_lat`,
            MIN(latitude) as `min_lat`,
            MAX(longitude) as `max_lon`,
            MIN(longitude) as `min_lon`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'IN' """
in_loc = openAQ.query_to_pandas_safe(query)
in_loc


# In[ ]:


query = """ SELECT city,latitude,longitude,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'IN' AND unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY latitude,city,longitude,country """
in_aqi = openAQ.query_to_pandas_safe(query)


# In[ ]:


# INDIA
min_lat = in_loc['min_lat']-5
max_lat = in_loc['max_lat']+5
min_lon = in_loc['min_lon']-5
max_lon = in_loc['max_lon']+5

plt.figure(figsize=(14,8))
m3 = Basemap(projection='cyl', llcrnrlon=min_lon, urcrnrlon=max_lon, llcrnrlat=min_lat, urcrnrlat=max_lat,
            resolution='c',lat_ts=True)
m3.drawcounties()
m3.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m3.fillcontinents(color='grey', alpha=0.3)
m3.drawcoastlines(linewidth=0.1, color="white")
m3.drawstates()
avg = in_aqi['Average']
m3loc = m3(in_aqi['latitude'].tolist(),in_aqi['longitude'])
m3.scatter(m3loc[1],m3loc[0],c = avg,alpha=0.5,zorder=5,cmap='rainbow')
m1.colorbar(label = 'Average AQI value in unit $ug/m^3$')
plt.title('Average Air qulity index in unit $ug/m^3$ of India');


# ### Distribution of pollutant and unit

# In[ ]:


# Unit query
query = """SELECT  unit,COUNT(unit) as `count`
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY unit
        """
unit = openAQ.query_to_pandas(query)
# Pollutant query
query = """SELECT  pollutant,COUNT(pollutant) as `count`
        FROM `bigquery-public-data.openaq.global_air_quality`
        GROUP BY pollutant
        """
poll_count = openAQ.query_to_pandas_safe(query)


# In[ ]:


plt.style.use('fivethirtyeight')
plt.style.use('bmh')
f, ax = plt.subplots(1,2,figsize = (14,5))
ax1,ax2= ax.flatten()
ax1.pie(x=unit['count'],labels=unit['unit'],shadow=True,autopct='%1.1f%%',explode=[0,0.1],       colors=sns.color_palette('hot',2),startangle=90,)
ax1.set_title('Distribution of measurement unit')
explode = np.arange(0,0.1)
ax2.pie(x=poll_count['count'],labels=poll_count['pollutant'], shadow=True, autopct='%1.1f%%',        colors=sns.color_palette('Set2',5),startangle=60,)
ax2.set_title('Distribution of pollutants in air');


# * The most polular unit of mesurement of air quality is $ug/m^3$
# * $O^3$ is share 23% pollution in air.
# ***
# ### Pollutant Statistics

# In[ ]:


query = """ SELECT pollutant,
                AVG(value) as `Average`,
                COUNT(value) as `Count`,
                MIN(value) as `Min`,
                MAX(value) as `Max`,
                SUM(value) as `Sum`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY pollutant
            """
cnt = openAQ.query_to_pandas_safe(query)
cnt 


#  We find
# * The CO (carbon monoxide) having very wide range of value.
# * Look at sum of CO which is highest in list.
# * Except Average AQI of CO, all are below 54 $ug/m^3$

# ### Pollutants by Country

# In[ ]:


query = """SELECT AVG(value) as`Average`,country, pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³'AND value BETWEEN 0 AND 10000
            GROUP BY country,pollutant"""
p1 = openAQ.query_to_pandas_safe(query)


# In[ ]:


# By country
p1_pivot = p1.pivot(index = 'country',values='Average', columns= 'pollutant')

plt.figure(figsize=(14,15))
ax = sns.heatmap(p1_pivot, lw=0.01, cmap=sns.color_palette('Reds',500))
plt.yticks(rotation=30)
plt.title('Heatmap average AQI by Pollutant');


# In[ ]:


f,ax = plt.subplots(figsize=(14,6))
sns.barplot(p1[p1['pollutant']=='co']['country'],p1[p1['pollutant']=='co']['Average'],)
plt.title('Co AQI in diffrent country')
plt.xticks(rotation=90);


# In[ ]:


f,ax = plt.subplots(figsize=(14,6))
sns.barplot(p1[p1['pollutant']=='pm25']['country'],p1[p1['pollutant']=='pm25']['Average'])
plt.title('pm25 AQI in diffrent country')
plt.xticks(rotation=90);


# ### Distribution of Source name
# The institution where AQI is measure 

# In[ ]:


#source_name 
query = """ SELECT source_name, COUNT(source_name) as `count`
    FROM `bigquery-public-data.openaq.global_air_quality`
    GROUP BY source_name
    ORDER BY count DESC
    """
source_name = openAQ.query_to_pandas_safe(query)


# In[ ]:


plt.figure(figsize=(14,10))
sns.barplot(source_name['count'][:20], source_name['source_name'][:20],palette = sns.color_palette('YlOrBr'))
plt.title('Distribution of Top 20 source_name')
#plt.axvline(source_name['count'].median())
plt.xticks(rotation=90);


# We find 
# * Airnow is top source unit in list
# * Europian country are top in the list, the instition name is starts with 'EEA country'.
# ***
# 
# ### Sample AQI Averaged over in hours
# The sample of AQI value taken in different hour

# In[ ]:


query = """SELECT averaged_over_in_hours, COUNT(*) as `count`
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY averaged_over_in_hours
            ORDER BY count DESC """
cnt = openAQ.query_to_pandas(query)


# In[ ]:


#cnt['averaged_over_in_hours'] = cnt['averaged_over_in_hours'].astype('category')
plt.figure(figsize=(14,5))
sns.barplot( cnt['averaged_over_in_hours'],cnt['count'], palette= sns.color_palette('brg'))
plt.title('Distibution of quality measurement per hour ');


# we find that air quality is measured every hour
# ***
# ### AQI in ppm

# In[ ]:


query = """SELECT AVG(value) as`Average`,country,
            EXTRACT(YEAR FROM timestamp) as `Year`,
            pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'ppm' 
            GROUP BY country,Year,pollutant"""
pol_aqi = openAQ.query_to_pandas_safe(query)


# In[ ]:


# By month in year
plt.figure(figsize=(14,8))
sns.barplot(pol_aqi['country'], pol_aqi['Average'])
plt.title('Distribution of average AQI by country $ppm$');


#  ### AQI variation with time

# In[ ]:


query = """SELECT EXTRACT(YEAR FROM timestamp) as `Year`,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY EXTRACT(YEAR FROM timestamp)
            """
quality = openAQ.query_to_pandas(query)

query = """SELECT EXTRACT(MONTH FROM timestamp) as `Month`,
            AVG(value) as `Average`
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY EXTRACT(MONTH FROM timestamp)
            """
quality1 = openAQ.query_to_pandas(query)


# In[ ]:


# plot
f,ax = plt.subplots(1,2, figsize= (14,6),sharey=True)
ax1,ax2 = ax.flatten()
sns.barplot(quality['Year'],quality['Average'],ax=ax1)
ax1.set_title('Distribution of average AQI by year')
sns.barplot(quality1['Month'],quality['Average'], ax=ax2 )
ax2.set_title('Distribution of average AQI by month')
ax2.set_ylabel('');


# In[ ]:


# by year & month
query = """SELECT EXTRACT(YEAR from timestamp) as `Year`,
            EXTRACT(MONTH FROM timestamp) as `Month`,
            AVG(value) as `Average`
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
        GROUP BY year,Month"""
aqi_year = openAQ.query_to_pandas_safe(query)


# In[ ]:


# By month in year
plt.figure(figsize=(14,8))
sns.pointplot(aqi_year['Month'],aqi_year['Average'],hue = aqi_year['Year'])
plt.title('Distribution of average AQI by month');


# We find 
# * the data available for perticular year is incomplete
# * the year 2016, 2017 data is availabel completely

# ### Country Heatmap

# In[ ]:


# Heatmap by country 
query = """SELECT AVG(value) as `Average`,
            EXTRACT(YEAR FROM timestamp) as `Year`,
            country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
            GROUP BY country,Year
            """
coun_aqi = openAQ.query_to_pandas_safe(query)


# In[ ]:


coun_pivot = coun_aqi.pivot(index='country', columns='Year', values='Average').fillna(0)
# By month in year
plt.figure(figsize=(14,15))
sns.heatmap(coun_pivot, lw=0.01, cmap=sns.color_palette('Reds',len(coun_pivot)))
plt.yticks(rotation=30)
plt.title('Heatmap average AQI by YEAR');


# ### Animation

# In[ ]:


query = """SELECT EXTRACT(YEAR FROM timestamp) as `Year`,AVG(value) as `Average`,
            latitude,longitude
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit = 'µg/m³' AND value BETWEEN 0 AND 10000
        GROUP BY Year, latitude,longitude
        """
p1 = openAQ.query_to_pandas_safe(query)


# In[ ]:


from matplotlib import animation,rc
import io
import base64
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
fig = plt.figure(figsize=(14,10))
plt.style.use('ggplot')

def animate(Year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('Average AQI in Year: '+str(Year))
    m4 = Basemap(llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180,urcrnrlon=180,projection='cyl')
    m4.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
    m4.fillcontinents(color='grey', alpha=0.3)
    m4.drawcoastlines(linewidth=0.1, color="white")
    m4.shadedrelief()
    
    lat_y = list(p1[p1['Year'] == Year]['latitude'])
    lon_y = list(p1[p1['Year'] == Year]['longitude'])
    lat,lon = m4(lat_y,lon_y) 
    avg = p1[p1['Year'] == Year]['Average']
    m4.scatter(lon,lat,c = avg,lw=2, alpha=0.3,cmap='hot_r')
    
   
ani = animation.FuncAnimation(fig,animate,list(p1['Year'].unique()), interval = 1500)    
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# In[ ]:


# Continued


# >>>>>> ### Thank you for visiting, please upvote if you like it. 
