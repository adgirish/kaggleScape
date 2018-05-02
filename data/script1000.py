
# coding: utf-8

# Exploratory Data Analysis on Indian states
# ==========================================
# 
# <a href="#dk24">Click here for final insights</a>

# I am exploring this data set for any insights. I am going to plot bar graphs and actual India map against various attributes. Lets explore India :)
# 
# This is going to be a long notebook. So if you want to check out the results quickly, follow the below links :
# 
# <a href="#dk1">Which State have most number of top cities ?</a><br/>
# 
# <a href="#dk2">Top 10 Populous cities in India</a><br/>
# <a href="#dk3">Which state have most of its population in urban areas ?</a><br/>
# 
# <a href="#dk4">Which state have most of its male population in urban areas ?</a><br/>
# <a href="#dk5">Top 10 cities with high male population</a><br/>
# 
# <a href="#dk6">Which state have most of its male population in urban areas ?</a><br/>
# <a href="#dk7">Top 10 cities with high female population</a><br/>
# 
# <a href="#dk8">Which state have most of its kids population in urban areas ?</a><br/>
# <a href="#dk9">Top 10 cities with high kids population</a><br/>
# 
# <a href="#dk10">Which state have most of its male kids population in urban areas ?</a><br/>
# <a href="#dk11">Top 10 cities with high male kids population</a><br/>
# 
# <a href="#dk12">Which state have most of its female kids population in urban areas ?</a><br/>
# <a href="#dk13">Top 10 cities with high female kids population</a><br/>
# 
# <a href="#dk14">Analysing Litracy rate of the states</a><br/>
# <a href="#dk15">Top 10 cities with most number of literates live</a><br/>
# 
# <a href="#dk16">Analysing Male Litracy rate of the states</a><br/>
# <a href="#dk17">Top 10 cities with most number of male literates live</a><br/>
# 
# <a href="#dk18">Analysing Female Litracy rate of the states</a><br/>
# <a href="#dk19">Top 10 cities with most number of female literates live</a><br/>
# 
# <a href="#dk20">Analyzing effective literacy rate</a><br/>
# 
# <a href="#dk21">Analyzing Graduates</a><br/>
# 
# <a href="#dk22">Analyzing Sex ratio</a><br/>
# 
# <a href="#dk23">Analyzing Sex ratio for children below 6</a><br/>

# Importing all the required packages
# -----------------------------------

# In[ ]:


# importing packages
import pandas as pd
import numpy as np
from scipy.interpolate import spline
from numpy import array
import matplotlib as mpl

# for plots
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.dates import date2num
from mpl_toolkits.basemap import Basemap

# for date and time processing
import datetime

# for statistical graphs
import seaborn as sns


# Importing all the Data into notebook
# -----------------------------------

# In[ ]:


cities = pd.read_csv ("../input/cities_r2.csv")


# Viewing Data and Verifying
# --------------------------

# In[ ]:


cities.head ()


# In[ ]:


cities.info ()
# there is no null values anywhere in the dataset


# In[ ]:


cities.describe ()


# In[ ]:


print (cities.describe(include=['O']))
# from the below output, we can learn that there is two Aurangabad's. One is in Maharashtra 
# and one is in Bihar
# most of the cities are selected from Uttar Pradesh


# <a id="dk1">Plotting state wise cities to check which state have most number of cities in it </a>
# ------------------------------------------------------------------------
# 

# In[ ]:


# A bar chart to show from which states, how many cities are taken for examination.
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['name_of_city'].count().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('No of cities taken for analysis', fontsize = 20)
plt.show ()
# we can see states like UP and WB are given high priority by taking more than 60 cities.


# <a id="dk2">Top 10 populous cities</a>
# ----------------------
# 

# In[ ]:


# Extracting Co-ordinates details from the provided data
cities['latitude'] = cities['location'].apply(lambda x: x.split(',')[0])
cities['longitude'] = cities['location'].apply(lambda x: x.split(',')[1])
cities.head(1)


# In[ ]:


# A table to show top 10 cities with most population
print("The Top 10 Cities sorted according to the Total Population (Descending Order)")
top_pop_cities = cities.sort_values(by='population_total',ascending=False)
top10_pop_cities=top_pop_cities.head(10)
top10_pop_cities


# In[ ]:


# Plotting these top 10 populous cities on India map. Circles are sized according to the 
# population of the respective city

plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(top10_pop_cities['longitude'])
lt=array(top10_pop_cities['latitude'])
pt=array(top10_pop_cities['population_total'])
nc=array(top10_pop_cities['name_of_city'])

x, y = map(lg, lt)
population_sizes = top10_pop_cities["population_total"].apply(lambda x: int(x / 5000))
plt.scatter(x, y, s=population_sizes, marker="o", c=population_sizes, cmap=cm.Dark2, alpha=0.7)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 Populated Cities in India',fontsize=20)


# <a id='dk3'>Plotting Statewise cities to check which state have most population living in urban areas</a>
# ------------------------------------------------------------------------
# 

# In[ ]:


# A bar chart to show the population of the states
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['population_total'].sum().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('No of cities taken for analysis', fontsize = 20)
plt.show ()
# we can see states like Maharashtra and UP have huge urban population


# Plotting every city on India map according to population
# --------------------------------------------------------

# In[ ]:


# Creating a function to plot the population data on real India map

def plot_map(sizes, colorbarValue):

    plt.figure(figsize=(19,20))
    f, ax = plt.subplots(figsize=(19, 20))

    # Setting up Basemap
    map = Basemap(width=5000000, height=3500000, resolution='l', projection='aea', llcrnrlon=69,
                  llcrnrlat=6, urcrnrlon=99, urcrnrlat=36, lon_0=78, lat_0=20, ax=ax)
                  
    # draw map boundaries
    map.drawmapboundary()
    map.drawcountries()
    map.drawcoastlines()

    # plotting cities on map using previously derived coordinates
    x, y = map(array(cities["longitude"]), array(cities["latitude"]))
    cs = map.scatter(x, y, s=sizes, marker="o", c=sizes, cmap=cm.Dark2, alpha=0.5)

    # adding colorbar
    cbar = map.colorbar(cs, location='right',pad="5%")
    cbar.ax.set_yticklabels(colorbarValue)

    plt.show()


# In[ ]:


# Using the function created in the previous cell, we are plotting the population data

population_sizes = cities["population_total"].apply(lambda x: int(x / 5000))
colorbarValue = np.linspace(cities["population_total"].min(), cities["population_total"].max(), 
                            num=10)
colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)


# <a id="dk4">Plotting Statewise cities to check which state have most male population</a>
# ------------------------------------------------------------------------
# 

# In[ ]:


# A bar chart to show the male population of the states
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['population_male'].sum().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('No of cities taken for analysis', fontsize = 20)
plt.show ()
# we can see states like Maharashtra and UP have huge male population


# In[ ]:


# Plotting the same on the map
population_sizes = cities["population_male"].apply(lambda x: int(x / 5000))
colorbarValue = np.linspace(cities["population_male"].min(), cities["population_male"].max(), 
                            num=10)
colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)


# <a id="dk5"> These are the top 10 cities with high male population</a>
# -----------------------------------------------------
# 

# In[ ]:


# A table to show top 10 cities with most male population
print("The Top 10 Cities sorted according to the Total Male Population (Descending Order)")
top_male_cities = cities.sort_values(by='population_male',ascending=False)
top10_male_pop_cities=top_male_cities.head(10)
top10_male_pop_cities


# In[ ]:


# Plotting these top 10 male populous cities on India map. Circles are sized according to the 
# male population of the respective city

plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(top10_male_pop_cities['longitude'])
lt=array(top10_male_pop_cities['latitude'])
pt=array(top10_male_pop_cities['population_male'])
nc=array(top10_male_pop_cities['name_of_city'])

x, y = map(lg, lt)
population_sizes_male = top10_male_pop_cities["population_male"].apply(lambda x: int(x / 5000))
plt.scatter(x, y, s=population_sizes_male, marker="o", c=population_sizes_male, cmap=cm.Dark2, alpha=0.7)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 Male Populated Cities in India',fontsize=20)


# <a id="dk6">Plotting Statewise cities to check which state have most female population</a>
# ------------------------------------------------------------------------
# 

# In[ ]:


# A bar chart to show the female population of the states
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['population_female'].sum().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('No of cities taken for analysis', fontsize = 20)
plt.show ()
# we can see again states like Maharashtra and UP have huge female population


# In[ ]:


# Plotting the same on the map
population_sizes = cities["population_female"].apply(lambda x: int(x / 5000))
colorbarValue = np.linspace(cities["population_female"].min(), cities["population_female"].max(), 
                            num=10)
colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)


# <a id="dk7">These are the top 10 cities with high female population</a>
# -----------------------------------------------------
# 

# In[ ]:


# A table to show top 10 cities with most female population
print("The Top 10 Cities sorted according to the Total Female Population (Descending Order)")
top_female_cities = cities.sort_values(by='population_female',ascending=False)
top10_female_pop_cities=top_female_cities.head(10)
top10_female_pop_cities


# In[ ]:


# Plotting these top 10 female populous cities on India map. Circles are sized according to the 
# female population of the respective city

plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(top10_female_pop_cities['longitude'])
lt=array(top10_female_pop_cities['latitude'])
pt=array(top10_female_pop_cities['population_female'])
nc=array(top10_female_pop_cities['name_of_city'])

x, y = map(lg, lt)
population_sizes_female = top10_female_pop_cities["population_female"].apply(lambda x: int(x / 5000))
plt.scatter(x, y, s=population_sizes_female, marker="o", c=population_sizes_female, cmap=cm.Dark2, alpha=0.7)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 Female Populated Cities in India',fontsize=20)


# <a id="dk8">Plotting Statewise cities to check which state have most Kids (aged between 0 to 6) population</a>
# ------------------------------------------------------------------------
# 

# In[ ]:


# A bar chart to show the kids population of the states
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['0-6_population_total'].sum().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('No of kids', fontsize = 20)
plt.show ()
# we can see again states like Maharashtra and UP have huge kids population living in cities


# In[ ]:


# Plotting the same on the map
population_sizes = cities["0-6_population_total"].apply(lambda x: int(x / 5000))
colorbarValue = np.linspace(cities["0-6_population_total"].min(), cities["0-6_population_total"].max(), 
                            num=10)
colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
# Kids population is obviously smaller than the overall population and bigger cities like Delhi,
# Mumbai, Banglore, Kolkata, Hyderabad, Chennai have vast number of kids living in cities


# <a id="dk9">Top 10 cities where most of the kids live</a>
# -----------------------------------------
# 

# In[ ]:


# Lets find the top ten cities in which large number of kids live
print("The Top 10 Cities sorted according to the Total Kids Population (Descending Order)")
top_kids_cities = cities.sort_values(by='0-6_population_total',ascending=False)
top10_kids_pop_cities=top_kids_cities.head(10)
top10_kids_pop_cities


# In[ ]:


# Lets find the top ten cities in which large number of kids live

plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(top10_kids_pop_cities['longitude'])
lt=array(top10_kids_pop_cities['latitude'])
pt=array(top10_kids_pop_cities['0-6_population_total'])
nc=array(top10_kids_pop_cities['name_of_city'])

x, y = map(lg, lt)
population_sizes_kids = top10_kids_pop_cities["0-6_population_total"].apply(lambda x: int(x / 5000))
plt.scatter(x, y, s=population_sizes_kids, marker="o", c=population_sizes_kids, cmap=cm.Dark2, alpha=0.7)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 Kids Populated Cities in India',fontsize=20)


# <a id="dk10">Plotting Statewise cities to check which state have most Male Kids (aged between 0 to 6) population</a>
# ------------------------------------------------------------------------
# 

# In[ ]:


# A bar chart to show the male kids population of the states
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['0-6_population_male'].sum().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('No of male kids', fontsize = 20)
plt.show ()
# we can see again states like Maharashtra and UP have huge male kids population living in cities


# In[ ]:


# Plotting the same on the map
population_sizes = cities["0-6_population_male"].apply(lambda x: int(x / 5000))
colorbarValue = np.linspace(cities["0-6_population_male"].min(), cities["0-6_population_male"].max(), 
                            num=10)
colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
# Kids population is obviously smaller than the overall population and bigger cities like Delhi,
# Mumbai, Banglore, Kolkata, Hyderabad, Chennai have vast number of kids living in cities


# <a id="dk11">Top 10 cities where most of the male kids live</a>
# ----------------------------------------------
# 

# In[ ]:


# Lets find the top ten cities in which large number of male kids live
print("The Top 10 Cities sorted according to the Total Male Kids Population (Descending Order)")
top10_male_kids_cities = cities.sort_values(by='0-6_population_male',ascending=False)
top10_male_kids_pop_cities=top10_male_kids_cities.head(10)
top10_male_kids_pop_cities


# In[ ]:


# Lets find the top ten cities in which large number of male kids live

plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(top10_male_kids_pop_cities['longitude'])
lt=array(top10_male_kids_pop_cities['latitude'])
pt=array(top10_male_kids_pop_cities['0-6_population_male'])
nc=array(top10_male_kids_pop_cities['name_of_city'])

x, y = map(lg, lt)
population_sizes_male_kids = top10_male_kids_pop_cities["0-6_population_male"].apply(lambda x: int(x / 5000))
plt.scatter(x, y, s=population_sizes_male_kids, marker="o", c=population_sizes_male_kids, cmap=cm.Dark2, alpha=0.7)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 Male Kids Populated Cities in India',fontsize=20)


# <a id="dk12">Plotting Statewise cities to check which state have most Female Kids (aged between 0 to 6) population</a>
# ------------------------------------------------------------------------
# 

# In[ ]:


# A bar chart to show the female kids population of the states
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['0-6_population_female'].sum().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('No of female kids', fontsize = 20)
plt.show ()
# we can see again states like Maharashtra and UP have huge male kids population living in cities


# In[ ]:


# Plotting the same on the map
population_sizes = cities["0-6_population_female"].apply(lambda x: int(x / 5000))
colorbarValue = np.linspace(cities["0-6_population_female"].min(), cities["0-6_population_female"].max(), 
                            num=10)
colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
# Kids population is obviously smaller than the overall population and bigger cities like Delhi,
# Mumbai, Banglore, Kolkata, Hyderabad, Chennai have vast number of kids living in cities


# <a id="dk13">Top 10 cities where most of the female children live</a>
# ----------------------------------------------------
# 

# In[ ]:


# Lets find the top ten cities in which large number of female kids live
print("The Top 10 Cities sorted according to the Total Female Kids Population (Descending Order)")
top10_female_kids_cities = cities.sort_values(by='0-6_population_female',ascending=False)
top10_female_kids_pop_cities=top10_female_kids_cities.head(10)
top10_female_kids_pop_cities


# In[ ]:


# Lets find the top ten cities in which large number of female kids live

plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(top10_female_kids_pop_cities['longitude'])
lt=array(top10_female_kids_pop_cities['latitude'])
pt=array(top10_female_kids_pop_cities['0-6_population_female'])
nc=array(top10_female_kids_pop_cities['name_of_city'])

x, y = map(lg, lt)
population_sizes_female_kids = top10_female_kids_pop_cities["0-6_population_female"].apply(lambda x: int(x / 5000))
plt.scatter(x, y, s=population_sizes_female_kids, marker="o", c=population_sizes_female_kids, cmap=cm.Dark2, alpha=0.7)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 Female Kids Populated Cities in India',fontsize=20)


# Analysing Litracy rate of the states
# ------------------------------------
# <a id="dk14"></a>

# In[ ]:


# A bar chart to show the total literates of the states
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['literates_total'].sum().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('Total litracy rate of states', fontsize = 20)
plt.show ()
# we can see again states like Maharashtra and UP have huge litrate population living in cities


# In[ ]:


# Plotting the same on the map
population_sizes = cities["literates_total"].apply(lambda x: int(x / 5000))
colorbarValue = np.linspace(cities["literates_total"].min(), cities["literates_total"].max(), 
                            num=10)
colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
# Major metro cities again shows higher litracy rates


# Top 10 cities where most of the literates live
# ----------------------------------------------
# <a id="dk15"></a>

# In[ ]:


# Lets find the top ten cities in which large number of literates live
print("The Top 10 Cities sorted according to the Total litrate Population (Descending Order)")
top10_literate_cities = cities.sort_values(by='literates_total',ascending=False)
top10_literate_cities=top10_literate_cities.head(10)
top10_literate_cities


# In[ ]:


# lets plot the top 10 literate cities on India map
plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(top10_female_kids_pop_cities['longitude'])
lt=array(top10_female_kids_pop_cities['latitude'])
pt=array(top10_female_kids_pop_cities['literates_total'])
nc=array(top10_female_kids_pop_cities['name_of_city'])

x, y = map(lg, lt)
population_sizes_female_kids = top10_female_kids_pop_cities["literates_total"].apply(lambda x: int(x / 5000))
plt.scatter(x, y, s=population_sizes_female_kids, marker="o", c=population_sizes_female_kids, cmap=cm.Dark2, alpha=0.7)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 most literate Cities in India',fontsize=20)


# Analysing Male Litracy rate of the states
# -----------------------------------------
# <a id="dk16"></a>

# In[ ]:


# # A bar chart to show the total male literates of the states
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['literates_male'].sum().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('No of total male literates of the states', fontsize = 20)
plt.show ()
# we can see again states like Maharashtra and UP have huge male literate population living in cities


# In[ ]:


# Plotting the same on the map
population_sizes = cities["literates_male"].apply(lambda x: int(x / 5000))
colorbarValue = np.linspace(cities["literates_male"].min(), cities["literates_male"].max(), 
                            num=10)
colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
# Major metro cities again shows higher male litracy rates


# Top 10 cities where most of the male literates live
# ---------------------------------------------------
# <a id="dk17"></a>

# In[ ]:


# Lets find the top ten cities in which large number of males are literate
print("The Top 10 Cities sorted according to the male literate Population (Descending Order)")
top10_male_literate_cities = cities.sort_values(by='literates_male',ascending=False)
top10_male_literate_cities=top10_male_literate_cities.head(10)
top10_male_literate_cities


# In[ ]:


# Lets find the top ten cities in which large number of males are literate on the map of India

plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(top10_female_kids_pop_cities['longitude'])
lt=array(top10_female_kids_pop_cities['latitude'])
pt=array(top10_female_kids_pop_cities['literates_male'])
nc=array(top10_female_kids_pop_cities['name_of_city'])

x, y = map(lg, lt)
population_sizes_female_kids = top10_female_kids_pop_cities["literates_male"].apply(lambda x: int(x / 5000))
plt.scatter(x, y, s=population_sizes_female_kids, marker="o", c=population_sizes_female_kids, cmap=cm.Dark2, alpha=0.7)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 male litracy cities in India',fontsize=20)


# Analysing Female Litracy rate of the states
# -----------------------------------------
# <a id="dk18"></a>

# In[ ]:


# A bar chart to show the female litracy population of the states
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['literates_female'].sum().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('No of female literates', fontsize = 20)
plt.show ()
# we can see again states like Maharashtra and UP have huge female literate population living in cities


# In[ ]:


# Plotting the same on the map
population_sizes = cities["literates_female"].apply(lambda x: int(x / 5000))
colorbarValue = np.linspace(cities["literates_female"].min(), cities["literates_female"].max(), 
                            num=10)
colorbarValue = colorbarValue.astype(int)

plot_map(population_sizes, colorbarValue)
# Major metro cities again shows higher female litracy rates


# In[ ]:


# Lets find the top ten cities in which large number of female literates live
print("The Top 10 Cities sorted according to the Total Female literates Population (Descending Order)")
top10_female_literates_cities = cities.sort_values(by='literates_female',ascending=False)
top10_female_literates_cities = top10_female_literates_cities.head(10)
top10_female_literates_cities


# Top 10 cities in which most of the female literates live
# --------------------------------------------------------
# <a id="dk19"></a>

# In[ ]:


# Lets find the top ten cities in which large number of female literates live

plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(top10_female_kids_pop_cities['longitude'])
lt=array(top10_female_kids_pop_cities['latitude'])
pt=array(top10_female_kids_pop_cities['literates_female'])
nc=array(top10_female_kids_pop_cities['name_of_city'])

x, y = map(lg, lt)
population_sizes_female_kids = top10_female_kids_pop_cities["literates_female"].apply(lambda x: int(x / 5000))
plt.scatter(x, y, s=population_sizes_female_kids, marker="o", c=population_sizes_female_kids, cmap=cm.Dark2, alpha=0.7)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 Female literates Populated Cities in India',fontsize=20)


# Analyzing effective literacy rate
# ---------------------------------
# <a id="dk20"></a>

# In[ ]:


# seperating effective literacy rate from the main dataset and sorting then in descending order
state_literacy_effective = cities[["state_name","effective_literacy_rate_total","effective_literacy_rate_male","effective_literacy_rate_female"]].groupby("state_name").agg({"effective_literacy_rate_total":np.average,
                                                                                                "effective_literacy_rate_male":np.average,
                                                                                                "effective_literacy_rate_female":np.average})
state_literacy_effective.sort_values("effective_literacy_rate_total", ascending=True).plot(kind="barh",
                      grid=True,
                      figsize=(16,15),
                      alpha = 0.6,
                      width=0.6,
                      stacked = False,
                      edgecolor="g",
                      fontsize = 20)
plt.grid(b=True, which='both', color='lightGreen',linestyle='-')
plt.show ()
# from the below chart, Mizoram, Kerala and HP have highest effective literacy rate across India


# Analyzing graduates
# -------------------
# <a id="dk21"></a>

# In[ ]:


# seperating Graduates from the main dataset and sorting then in descending order
state_graduates  = cities[["state_name",
                                  "total_graduates",
                                  "male_graduates",
                                  "female_graduates"]].groupby("state_name").agg({"total_graduates":np.average,
                                                                                  "male_graduates":np.average,
                                                                                  "female_graduates":np.average})
# Plotting the bar chart 
state_graduates.sort_values("total_graduates", ascending=True).plot(kind="barh",
                      grid=True,
                      figsize=(16,15),
                      alpha = 0.6,
                      width=0.6,
                      stacked = False,
                      edgecolor="g",
                      fontsize = 20)
plt.grid(b=True, which='both', color='lightGreen',linestyle='-')
plt.show ()
# from the below Chandigarh, NCT of Delhi, Maharashta have most of their graduates living in cities.
# we can note that Kerala and Meghalaya are the only states that have more number of female graduates than 
# male graduates


# <a id="dk22"></a>
# 
# Analyzing Sex ratio across states
# ---------------------------------

# In[ ]:


# A bar chart to show how many females are there for per 1000 males.
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['sex_ratio'].mean().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('No of females available for every 1000 males', fontsize = 20)
plt.show ()
# We can see that states of Kerala, Manipur, Meghalaya, Puducherry, Mizoram are having more females per 1000 males


# <a id="dk23"></a>Analyzing Sex ratio across states for children below 6
# ------------------------------------------------------

# In[ ]:


# A bar chart to show how many females are there for per 1000 males.
fig = plt.figure(figsize=(20,20))
states = cities.groupby('state_name')['child_sex_ratio'].mean().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20)
plt.grid(b=True, which='both', color='Black',linestyle='-')
plt.xlabel('No of girls available for every 1000 boys', fontsize = 20)
plt.show ()
# Not even a single state have 1000 girls for every 1000 boys


# Final Insights (Please remember that these insights are only for urban population. 70% of Indian population lives in Non-urban, Semi-urban and Village areas)
# --------------
# <a id="dk24"></a>

# > Uttar Pradesh and West Bengal are the two states from which most of the cities were taken, followed by Maharashtra and Andhra Pradesh.
# > 
# > Greater Mumbai, Delhi, Bengaluru are the top most populous cities of India in that order.
# > 
# > Maharashtra, Uttar Pradesh and Andhra Pradesh are the top 3 states, where most of the people lives in urban areas.
# > 
# > Maharashtra is the only state that have most of the male, female, male kids, female kids population living in urban areas.
# > 
# > Greater Mumbai is the most populated city in India (both men and women)
# > 
# > Delhi is the top most city that have high kids population
# > 
# > Maharashtra is the only state which have huge literate population living in urban areas.
# > 
# > Again Greater Mumbai have highest number of literates (both men and women), very closely followed by Delhi.
# > 
# > Mizoram is the state that have very high effective literacy rate, very closely followed by Kerala and Himachal Pradesh.
# > 
# > Not even a single state have more femal literates than male literates. Worst case is Rajasthan, where difference between effective
# > literacy rate of men and women is very high.
# > 
# > Good thing is that almost all the states have effective literacy rate of more than 80 % (remeber this data is only for cities. 80% of
# > Indian population lives in non-urban, semi-urban and village areas)
# > 
# > Interesting thing to note is, Kerala and Meghalaya are the only states where more female graduates are seen than male graduates in urban areas. Worst case is Bihar and Jharkand, where difference between men and women graduates is very high in urban areas itself.
# > 
# > This numbers might go up drastically in rural areas.
# > 
# > Sex ratio is defined as how many females are there for every 1000 males (atleast in India it is calculated in this way). 
# >
# > Kerala,Manipur, Meghalaya, Puduchery, Mizoram are the states where more than 1000 females are there for every      > 1000 males. It means there are more females than males.
# > 
# > Worst case is Chandigarh and Himachal Pradesh where there are around 800 females for every 1000 males (which is clearly a bad sign)
# > 
# > When children below 6 are taken into account, not even a single state have 1000 girls for 1000 boys. This may change when these kids become adults, because age limit for this calculation is very narrow (6 yeras).
# 
# 
