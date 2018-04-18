
# coding: utf-8

# In[6]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
import math
import datetime

# using Basemap for map visualization. Installed it with "conda install basemap"
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import FastMarkerCluster
from folium.plugins import MarkerCluster

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import folium
from folium import plugins
from folium.plugins import HeatMap
from folium.plugins import FastMarkerCluster
from folium.plugins import MarkerCluster


# ## About Kiva.org
# 
# **Kiva envisions a world where all people hold the power to create opportunity for themselves and others.**
# 
# Kiva is an international nonprofit, founded in 2005 and based in San Francisco, with a mission to connect people through lending to alleviate poverty. 
# 
# In Kaggle Datasets' inaugural Data Science for Good challenge, Kiva is inviting the Kaggle community to help them build more localized models to estimate the poverty levels of residents in the regions where Kiva has active loans. 
# 
# This notebook tries to explore the ways to achieve that.
# 
# Part 1: EDA

# In[7]:


kiva_loans=pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv",parse_dates=['funded_time','date'])
#kiva_loans.shape

kiva_loans["funded_time"]=pd.to_datetime(kiva_loans.funded_time)
kiva_loans.dropna()
kiva_loans["funded_year"]=kiva_loans["date"].dt.year
kiva_loans["funded_month"]=kiva_loans["date"].dt.month


# In[8]:


color_dict = {'Food': 'red', 'Transportation': 'pink', 'Arts': 'yellow','Services': 'blue', 'Agriculture': 'green',
'Manufacturing': 'cyan', 'Wholesale': 'brown', 'Retail': 'Gold', 'Clothing': 'black', 'Construction': 'magenta', 'Health': 'lightgreen',
'Education': 'crimson', 'Personal Use': 'purple', 'Housing': 'orange', 'Entertainment': 'lightblue'}

loc=pd.read_csv('../input/additional-kiva-snapshot/loan_coords.csv')
loc.rename(index=str, columns={"loan_id": "id"},inplace=True)
#loc.head()


# In[9]:


kiva=pd.merge(kiva_loans, loc,on='id',how="left")
#kiva.shape


# ### Loan locations across the world

# In[10]:


p=kiva.plot(kind='scatter', x='longitude', y='latitude',
                color='green',figsize=(15,10), 
                title='Loan locations for World Map')
p.grid(False)
plt.savefig('Loan-location.png');


# In[11]:


countries_by_sectors_yearly_funded_amount_mean = kiva.groupby(['country','sector', 'funded_year'])['funded_amount'].mean().unstack()
#print(countries_by_sectors_yearly_funded_amount_mean.shape)
#countries_by_sectors_yearly_funded_amount_mean.head()


# In[12]:


Funded_Regions_BySectors = kiva.groupby(['country','sector']).first()[['latitude', 'longitude']]
#print(Funded_Regions_BySectors.shape)
#Funded_Regions_BySectors.head()


# In[13]:


#code credit: https://www.kaggle.com/pavelevap/global-warming-confirmed-basemap-animation?scriptVersionId=485498
def get_temp_markers(countries, year):
    
    k=0
    points = np.zeros(990, dtype=[('lon', float, 1),
                                      ('lat', float, 1),
                                      ('size',  float, 1),
                                      ('color', object, '')])
    cmap = plt.get_cmap('viridis')
    for i, country in enumerate(random_countries):
        country=country[0]
        funds = countries_by_sectors_yearly_funded_amount_mean.loc[country]
        sectors=funds.index
        for j , sector in enumerate(sectors):
            amount = funds.loc[sector].loc[year]
            if(math.isnan(amount)):
                break;
            coords = Funded_Regions_BySectors.loc[country].loc[sector][['latitude', 'longitude']].values
            lat = float(coords[0])
            lon = float(coords[1])
            if(math.isnan(lat)):
                break;
            points['lat'][k] = lat
            if(math.isnan(lon)):
                break;
            points['lon'][k] = lon
            points['size'][k] = amount/5
            points['color'][k] = color_dict[sector]
            k=k+1
            #print(k," ",amount," ",lat," ",lon," ",color_dict[sector])
    points=points[points['lat']!=0]
    return points


# ### Mean Funded Amount According to scetors in year 2014

# In[14]:


fig = plt.figure(figsize=(18, 15))
cmap = plt.get_cmap('viridis')
map = Basemap(projection='cyl')
map.drawmapboundary()
map.drawcoastlines(color='black')
map.fillcontinents(color='beige',lake_color='lightblue', zorder=3);

START_YEAR = 2014
LAST_YEAR = 2017
n_countries = 65
random_countries = countries_by_sectors_yearly_funded_amount_mean.sample(n_countries).index
year_text = plt.text(-170, 80, str(START_YEAR),fontsize=15)
temp_markers = get_temp_markers(random_countries, START_YEAR)


xs, ys = map(temp_markers['lon'], temp_markers['lat'])
scat = map.scatter(xs, ys, s=temp_markers['size'], c=temp_markers['color'], cmap=cmap, marker='o', 
                   alpha=0.3, zorder=10)
plt.title('Mean fundings by sectors for year 2014 ',fontsize=19)
labels=['Agriculture', 'Food', 'Retail', 'Services', 'Personal Use', 'Housing', 'Clothing', 'Education', 'Transportation',
        'Arts', 'Health', 'Construction', 'Manufacturing', 'Entertainment', 'Wholesale']
handles=[scat,scat,scat,scat,scat,scat,scat,scat,scat,scat,scat,scat,scat]
plt.legend(handles, labels,  loc = 6
           , title='Sectors', markerscale=0.3,labelspacing=0.3)

ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('green')
leg.legendHandles[1].set_color('red')
leg.legendHandles[2].set_color('gold')
leg.legendHandles[3].set_color('blue')
leg.legendHandles[4].set_color('purple')
leg.legendHandles[5].set_color('orange')
leg.legendHandles[6].set_color('black')
leg.legendHandles[7].set_color('crimson')
leg.legendHandles[8].set_color('pink')
leg.legendHandles[9].set_color('yellow')
leg.legendHandles[10].set_color('lightgreen')
leg.legendHandles[11].set_color('magenta')
leg.legendHandles[12].set_color('cyan');
plt.savefig('Mean-fundings-by-sectors-for-year-2014.png');


# Above plot shows mean Funded amount granted to borrowers and higher the amount higher the size of markers.
# Colours are according to sectors.
# 
# ### Animated Story
# If animation is taking time to load, kindly check output tab.

# In[15]:


get_ipython().run_line_magic('matplotlib', 'nbagg')

# Create new map 
fig = plt.figure(figsize=(18, 15))
cmap = plt.get_cmap('viridis')
map = Basemap(projection='cyl')
map.drawmapboundary()
map.drawcoastlines(color='black')
map.fillcontinents(color='beige',lake_color='lightblue', zorder=3);


# Create  data
START_YEAR = 2014
LAST_YEAR = 2017
n_countries = 80
random_countries = countries_by_sectors_yearly_funded_amount_mean.sample(n_countries).index


# Initialize the map in base position
temp_markers = get_temp_markers(random_countries, START_YEAR)
xs, ys = map(temp_markers['lon'], temp_markers['lat'])

# Construct the scatter which we will update during animation
# as the years change.
scat = map.scatter(xs, ys, s=temp_markers['size'], c=temp_markers['color'], cmap=cmap, marker='o', 
                   alpha=0.3, zorder=10)
year_text = plt.text(-170, 80, str(START_YEAR),fontsize=15)
text="Mean funded Amount According to Sectors"
title_text = plt.text(-170, -85, text,fontsize=15)
labels=['Agriculture', 'Food', 'Retail', 'Services', 'Personal Use', 'Housing', 'Clothing', 'Education', 'Transportation',
        'Arts', 'Health', 'Construction', 'Manufacturing', 'Entertainment', 'Wholesale']
handles=[scat,scat,scat,scat,scat,scat,scat,scat,scat,scat,scat,scat,scat]
plt.legend(handles, labels,  loc = 6
           , title='Sectors', markerscale=0.3,labelspacing=0.3)

ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('green')
leg.legendHandles[1].set_color('red')
leg.legendHandles[2].set_color('gold')
leg.legendHandles[3].set_color('blue')
leg.legendHandles[4].set_color('purple')
leg.legendHandles[5].set_color('orange')
leg.legendHandles[6].set_color('black')
leg.legendHandles[7].set_color('crimson')
leg.legendHandles[8].set_color('pink')
leg.legendHandles[9].set_color('yellow')
leg.legendHandles[10].set_color('lightgreen')
leg.legendHandles[11].set_color('magenta')
leg.legendHandles[12].set_color('#eeefff');



def update(frame_number):
    # Get an index which we can use to re-spawn the oldest year.
    current_year = START_YEAR + (frame_number % 4)

    temp_markers = get_temp_markers(random_countries, current_year)
    xs, ys = map(temp_markers['lon'], temp_markers['lat'])

    # Update the scatter collection, with the new colors, sizes and positions.
    scat.set_offsets(np.c_[xs, ys])
    scat.set_color(temp_markers['color'])
    scat.set_sizes(temp_markers['size'])
    year_text.set_text(str(current_year))
    text="Kiva - Mean Funded Amount to Borrowers According to Sectors"
    title_text.set_text(text)



# Construct the animation, using the update function as the animation
# director.
plt.title('Kiva - Mean fundings by sectors for years 2014-2017 ',fontsize=19)
ani = FuncAnimation(fig, update, interval=1000,repeat=False,blit=True)
#plt.show()


# In[ ]:


ani.save('anim.gif', writer='imagemagick', fps=2)
import io
import base64
filename = 'anim.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# In[16]:


kiva_mpi_region_locations=pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
#kiva_mpi_region_locations.shape


# ### World Regions with MPI 
# OPHI  calculates the Global Multidimensional Poverty Index MPI, which has been published since 2010 in the United Nations Development Programme’s Human Development Report. 
# 
# Let's plot Kiva’s estimates as to the geolocation of subnational MPI regions.

# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(12,8))
sns.barplot(x=kiva_mpi_region_locations.world_region.value_counts().values,y=kiva_mpi_region_locations.world_region.value_counts().index)
plt.title("World Regions")
plt.savefig('world regions.png');


# ### Let's zoom on african countries with MPI region locations

# In[18]:


african_countries = kiva_mpi_region_locations[kiva_mpi_region_locations['world_region']== 'Sub-Saharan Africa']
plt.figure(figsize=(12,15))
sns.barplot(x=african_countries.country.value_counts().values,y=african_countries.country.value_counts().index,palette="viridis")
plt.title("African Countries")
plt.savefig('african countries.png');


# ## Heatmap for Multi-Dimentional Poverty index for world

# In[19]:


#remove NANs
kiva_mpi_region_locations = kiva_mpi_region_locations.dropna(axis=0)

# Create weight column, using date
kiva_mpi_region_locations['weight'] = kiva_mpi_region_locations.MPI.multiply(15).astype(int)
#kiva_mpi_region_locations.weight.unique()


# In[20]:


kiva_loactions_on_heatmap = folium.Map(location=[kiva_mpi_region_locations.lat.mean(), kiva_mpi_region_locations.lon.mean() ],tiles= "Stamen Terrain",
                    zoom_start = 2) 

# List comprehension to make out list of lists
heat_data = [[[row['lat'],row['lon']] 
                for index, row in kiva_mpi_region_locations[kiva_mpi_region_locations['weight'] == i].iterrows()] 
                 for i in range(0,11)]
#print(heat_data)
# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(kiva_loactions_on_heatmap)

hm.save('world MPI heatmap.html')

# Display the map
kiva_loactions_on_heatmap


# 
# Looks like Africa has got highest number of MPI Locations.
# 
# ### Let's zoom on Africa

# In[21]:


heat_df =kiva_mpi_region_locations[kiva_mpi_region_locations['world_region']== 'Sub-Saharan Africa']

#remove NANs
heat_df = heat_df.dropna(axis=0)

# Create weight column, using date
heat_df['weight'] = heat_df.MPI.multiply(15).astype(int)
heat_df = heat_df.dropna(axis=0,subset=['lat','lon', 'weight','LocationName'])
#heat_df.weight.unique()


# In[22]:


kiva_loactions_on_heatmap_africa = folium.Map(location=[heat_df.lat.mean(), heat_df.lon.mean() ],tiles= "Stamen Terrain",
                    zoom_start = 3) 

# List comprehension to make out list of lists
heat_data = [[[row['lat'],row['lon']] 
                for index, row in heat_df[heat_df['weight'] == i].iterrows()] 
                 for i in range(0,11)]
#print(heat_data)
# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(kiva_loactions_on_heatmap_africa)
hm.save('africa MPI heatmap.html')

# Display the map
kiva_loactions_on_heatmap_africa


# ### Poverty locations for South Asia as per OPHI's MPI
# 
# Click on cluster circle to see clustered points

# In[23]:


kiva_mpi_region_locations_africa = kiva_mpi_region_locations[kiva_mpi_region_locations['world_region'] == 'South Asia']
kiva_mpi_region_locations_africa.dropna(axis=0, inplace=True)
m = folium.Map(
    location=[kiva_mpi_region_locations_africa.lat.mean(), kiva_mpi_region_locations_africa.lon.mean()],
    tiles='Cartodb Positron',
    zoom_start=4
)

marker_cluster = MarkerCluster(
    name='African Locations',
    overlay=True,
    control=False,
    icon_create_function=None
)

for k in range(kiva_mpi_region_locations_africa.shape[0]):
    location = kiva_mpi_region_locations_africa.lat.values[k], kiva_mpi_region_locations_africa.lon.values[k]
    marker = folium.Marker(location=location,icon=folium.Icon(color='green', icon='ok-sign'))
    popup = kiva_mpi_region_locations_africa.LocationName.values[k]
    folium.Popup(popup).add_to(marker)
    marker_cluster.add_child(marker)

marker_cluster.add_to(m)

folium.LayerControl().add_to(m)

m.save("marker cluster south asia.html")
m


# ### Clustering  locations in Africa
# 
# Click on cluster circle to see clustered points

# In[24]:


#%%time

m = folium.Map(
    location=[kiva_mpi_region_locations_africa.lat.mean(), kiva_mpi_region_locations_africa.lon.mean() ],
    tiles='Cartodb Positron',
    zoom_start=4
)

FastMarkerCluster(data=list(zip(kiva_mpi_region_locations_africa.lat.values, kiva_mpi_region_locations_africa.lon.values))).add_to(m)

folium.LayerControl().add_to(m)
m.save('africa loc cluster.html')

m


# to be continued...
