
# coding: utf-8

# <p>Note :<br>
# If you have any comments/suggestions about the notebook and how to improve or optimize anything I'd be very happy to hear them!
# Although I've been using Python for 8+ years I keep learning new things.<p>
# 
# <h1>Meteorite basic analysis</h1>
# <p> This is just a couple of plots and basic level analysis about the dataset to see what it looks like</p>
# <p>I have separated the study in two parts : geographical analysis and historical analysis</p>

# <h2>Importing and Cleaning</h2>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np                       # linear algebra
import pandas as pd                      # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt          # plotting library
from mpl_toolkits.basemap import Basemap # library to plot maps
import seaborn as sns                    # this one is just for the fancy look

from ipywidgets import *                 # the library to include widgets

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

plt.rcParams['figure.figsize']=(9,7)


# In[ ]:


#Let's first clean the dataset and keep valid entries with known latitude and longitude
data = pd.read_csv('../input/meteorite-landings.csv')
data = data[(data.reclat != 0.0) & (data.reclong != 0.0)]
data.info()


# In[ ]:


# hereafter I look only at the valid entries
valids = data.groupby('nametype').get_group('Valid').copy()
valids.dropna(inplace=True)
valids.info()


# <h2>Geographical Study</h2>

# In[ ]:


#Here comes the funny part, let's focus first on the geographical information
map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines(linewidth=0.5)
map.drawmeridians(range(0, 360, 20),linewidth=0.1)
map.drawparallels([-66.56083,-23.5,0.0,23.5,66.56083], linewidth=0.6) #plotting equator and tropics and polar circles

x, y = map(valids.reclong,valids.reclat)

map.scatter(x, y, marker='.',alpha=0.25,c='green',edgecolor='None')
plt.title('Map of all valid impacts', fontsize=15)


# In[ ]:


#Let's create a heatmap which might be more interesting
h = plt.hist2d(valids.reclong,valids.reclat,bins=(np.arange(-180,182,2),np.arange(-90,92,2)))
X,Y = np.meshgrid(h[1][:-1]+1.0,h[2][:-1]+1.0)

map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines(linewidth=0.6)
map.drawmeridians(range(0, 360, 20),linewidth=0.1)
map.drawparallels([-66.56083,-23.5,0.0,23.5,66.56083], linewidth=0.6)

data_interp, x, y = map.transform_scalar(np.log10(h[0].T+0.1), X[0], Y[:,0], 360, 360, returnxy=True)
map.pcolormesh(x, y, data_interp,cmap='hot_r')
map.colorbar()
plt.title('Heatmap of all meteorite impacts', fontsize=15)


# In[ ]:


#We can easily focus on a specific area of the map, let's look at Oman :
map = Basemap(projection='cyl',llcrnrlat=10.0,urcrnrlat=30.0,llcrnrlon=40.0,urcrnrlon=70.0,resolution='i')
map.drawmapboundary(fill_color='w')
map.drawcoastlines(linewidth=0.6)
map.drawcountries()

h = plt.hist2d(valids.reclong,valids.reclat,bins=(np.arange(40,71,1.0),np.arange(10,31,1)))
X,Y = np.meshgrid(h[1][:-1]+0.5,h[2][:-1]+0.5)
data_interp, x, y = map.transform_scalar(np.log10(h[0].T+0.1), X[0], Y[:,0], 101, 101, returnxy=True)
map.pcolormesh(x, y, data_interp,cmap='hot_r')
map.scatter(valids.reclong,valids.reclat,marker='.', alpha=0.5, edgecolor='None', color='g')


# <p>I don't think meteorite choose in which country they land. There's very very likely a bias linked to geopolitics and population density </p>
# <p>I recommend the reader to download the notebook and the dataset and look at different regions of the map, it is pretty easy to do using Basemap, simply change : <br><i>llcrnrlat=lower_left_corner_latitude <br >urcrnrlat=upper_right_corner_latitude <br> llcrnrlon= lower_left_corner_longitude <br> urcrnrlon=upper_right_corner_longitude</i></p>

# <h2>Comparing Meteorites found and seen falling</h2>

# In[ ]:


# What about the difference between meteorites seen falling and ones found ???
v_fell = valids.groupby('fall').get_group('Fell')
v_found = valids.groupby('fall').get_group('Found')


# In[ ]:


plt.figure(figsize=(10,10))

plt.subplot(211)
h = plt.hist2d(v_fell.reclong,v_fell.reclat,bins=(np.arange(-180,182,2),np.arange(-90,92,2)))
X,Y = np.meshgrid(h[1][:-1]+1.0,h[2][:-1]+1.0)

map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines(linewidth=0.6)
map.drawmeridians(range(0, 360, 20),linewidth=0.1)
map.drawparallels([-66.56083,-23.5,0.0,23.5,66.56083], linewidth=0.6)

data_interp, x, y = map.transform_scalar(np.log10(h[0].T+0.1), X[0], Y[:,0], 360, 360, returnxy=True)
map.pcolormesh(x, y, data_interp,cmap='hot_r')
map.colorbar()
plt.title('Heatmap of meteorites seen falling', fontsize=15)

plt.subplot(212)
h = plt.hist2d(v_found.reclong,v_found.reclat,bins=(np.arange(-180,182,2),np.arange(-90,92,2)))
X,Y = np.meshgrid(h[1][:-1]+1.0,h[2][:-1]+1.0)

map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines(linewidth=0.6)
map.drawmeridians(range(0, 360, 20),linewidth=0.1)
map.drawparallels([-66.56083,-23.5,0.0,23.5,66.56083], linewidth=0.6)

data_interp, x, y = map.transform_scalar(np.log10(h[0].T+0.1), X[0], Y[:,0], 360, 360, returnxy=True)
map.pcolormesh(x, y, data_interp,cmap='hot_r')
map.colorbar()
plt.title('Heatmap of meteorites founds', fontsize=15)


# <h3>Comment</h3>
# <p>Meteorites seen falling seem to correlate with densely populated area (make sense) while meteorites are found in more desertic areas.
# There is probably a deviation to this "first order" correlation due to the following factors :
#  <ul>
# <li>Light Polution : observation is obviously more difficult in inner cities or near sources of bright light
# <li>Land cover type : It is probably easier to find a meteorite in the desert than in the middle of the rainforest. Add to this the local geological activity that can "destroy" fallen meteorites (though this effect is likely very localized).
# </ul>
# </p>
# 
# <p>This would need to be quantified!!! Will try to include this later...</p>

# In[ ]:


#Another view of the previous statement
map = Basemap(projection='cyl',llcrnrlat=10,llcrnrlon=-20,urcrnrlat=50,urcrnrlon=40,resolution='c')
map.etopo()
map.drawcountries()

map.scatter(v_fell.reclong,v_fell.reclat,edgecolor='none',color='k',alpha=0.6)
map.scatter(v_found.reclong,v_found.reclat,edgecolor='none',color='m',alpha=0.6)

plt.title('Comparison between meteorites found (magenta) and meteorites seen fallin (black)', fontsize=15)


# <h3> The following three cells allow to superimpose world population density maps and the meteorites discoveries. Nonetheless it does not work here because of a problem reaching the image url (still trying to figure out how to fix this).<br>
# It does work on my personal laptop though so if you download the notebook and run it from your machine it should be fine.</h3>

# In[ ]:


map = Basemap(projection='cyl',resolution='i')
# uncomment the line to visualize the plot
#map.warpimage(image='http://neo.sci.gsfc.nasa.gov/servlet/RenderData?si=875430&cs=rgb&format=JPEG&width=1440&height=720')
map.scatter(valids.reclong,valids.reclat,marker='.',alpha=0.5,edgecolor='None',color='b')

plt.title('Map of all meteorites detected compared to population density', fontsize=15)


# In[ ]:


map = Basemap(projection='cyl',resolution='i')
#map.warpimage(image='http://neo.sci.gsfc.nasa.gov/servlet/RenderData?si=875430&cs=rgb&format=JPEG&width=1440&height=720')
map.scatter(v_found.reclong,v_found.reclat,marker='.',alpha=0.5,edgecolor='None',color='m')

plt.title('Map of meteorites found compared to population density', fontsize=15)


# In[ ]:


map = Basemap(projection='cyl',resolution='i')
#map.warpimage(image='http://neo.sci.gsfc.nasa.gov/servlet/RenderData?si=875430&cs=rgb&format=JPEG&width=1440&height=720')
map.scatter(v_fell.reclong,v_fell.reclat,marker='.',alpha=0.5,edgecolor='None',color='k')

plt.title('Map of meteorites seen falling compared to population density', fontsize=15)


# In[ ]:


# NOTE : This cell does not work properly either because of some external configuration
# You can download the notebook and if you have ipywidgets installed it should run smoothly

# Now an interactive figure showing the two types of meteorites ("seen falling" and "found")
# as a function of the year
def fscat(x):
    plt.figure(figsize=(10,8))
    map = Basemap(projection='cyl',resolution='c')
    map.drawmapboundary(fill_color='w')
    map.drawcoastlines(linewidth=0.5)
    map.scatter(v_fell.reclong[v_fell.year==x],v_fell.reclat[v_fell.year==x],
                color='k',alpha=0.7)
    map.scatter(v_found.reclong[v_found.year==x],v_found.reclat[v_found.year==x],
                color='m',alpha=0.7)
    return 

wid = IntSlider(min=1900,max=2013,step=1,description='Year',layout=Layout(width='75%'))
_ = interact(fscat,x=wid)


# <h2>Quick study of the mass distribution</h2>

# In[ ]:


map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines(linewidth=0.6)
map.drawmeridians(range(0, 360, 20),linewidth=0.1)
map.drawparallels([-66.56083,-23.5,0.0,23.5,66.56083], linewidth=0.6)

map.scatter(valids.reclong,valids.reclat,s=np.sqrt(valids.mass/150),alpha=0.4,color='g')
plt.title('Location of all meteorites with their mass', fontsize=15)


# In[ ]:


plt.figure(figsize=(10,10))

plt.subplot(211)

map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines(linewidth=0.6)
map.drawmeridians(range(0, 360, 20),linewidth=0.1)
map.drawparallels([-66.56083,-23.5,0.0,23.5,66.56083], linewidth=0.6)

map.scatter(v_fell.reclong,v_fell.reclat,s=np.sqrt(v_fell.mass/150),
            alpha=0.4,color='k')
plt.title('Location and mass of meteorites seen falling')


plt.subplot(212)

map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines(linewidth=0.6)
map.drawmeridians(range(0, 360, 20),linewidth=0.1)
map.drawparallels([-66.56083,-23.5,0.0,23.5,66.56083], linewidth=0.6)

map.scatter(v_found.reclong,v_found.reclat,s=np.sqrt(v_found.mass/150),
            alpha=0.4,color='m')
plt.title('Location and mass of meteorites found')


# <h3>Comment</h3>
# <p>According to the previous, there is a difference in mass between meteorites seen falling and meteorites found. A simple histogram should be able to reveal this.</p>

# In[ ]:


plt.figure(figsize=(8,6))
sns.distplot(np.log10(v_found.mass),color='k')
sns.distplot(np.log10(v_fell.mass),color='m',axlabel='$\log[Mass]$')
plt.legend(['Meteorites Found','Meteorites Seen Falling'])


# <h2>Historical Analysis</h2>

# In[ ]:


plt.subplot(211)
valids.year.hist(bins=np.arange(1900,2014,1),figsize=(8,7))
plt.title('Discoveries per year 1900-2013')
plt.xlim(1900,2014)

plt.subplot(212)
valids.year.hist(bins=np.arange(1900,2023,10),figsize=(8,7))
plt.title('Discoveries per decade 1900-2013')
plt.xlim(1900,2014)


# <p> I would not be surprised to find a correlation with the world population growth... although communication and technological improvements can explain the trend</p>

# In[ ]:


# Againg it is interesting to separate "seen" and "found"
plt.subplot(211)
v_found.year.hist(bins=np.arange(1900,2014,1),figsize=(8,7))
plt.title('Meteorites found per year 1900-2013')
plt.xlim(1900,2014)

plt.subplot(212)
v_found.year.hist(bins=np.arange(1900,2023,10),figsize=(8,7))
plt.title('Meteorites found per decade 1900-2013')
plt.xlim(1900,2014)


# In[ ]:


plt.subplot(211)
v_fell.year.hist(bins=np.arange(1900,2014,1),figsize=(8,7))
plt.title('Meteorites seen falling per year 1900-2013')
plt.xlim(1900,2014)

plt.subplot(212)
v_fell.year.hist(bins=np.arange(1900,2023,10),figsize=(8,7))
plt.title('Meteorites seen falling per decade 1900-2013')
plt.xlim(1900,2014)


# <h3>Comments</h3>
# <p>The rate of meteorites seen falling is surprisingly constant over the last century considering the increase of population worldwide. Nonetheless it could be explained by an increase in density and not in spread.</p>
# <p>We can also have a look at the number of discoveries as a function of the latitude... </p>

# In[ ]:


plt.scatter(valids.year,valids.reclat,color='g',alpha=0.4)
plt.xlim(1800,2010)
plt.ylim(-90,90)
plt.ylabel('Latitude')
plt.xlabel('Year')
plt.title('Meteorite recorded latitude vs year')


# <h2>Class Analysis</h2>
# 
# <p>We can first check how many recorded class (recclass) of meteorites we have in our data set and their frequency</p>
# 

# In[ ]:


#The following line returns a pandas Series with recclass as index
# and the number of occurences as attached value
v_class = valids['recclass'].value_counts()
v_class


# In[ ]:


#If we want to know the number of recclass just type :
len(v_class)

#Note : you can find the same answer using :
# len(np.unique(valids.recclass))


# <p>This is quite a lot of different classes and there is a high chance that figures will be quite messy if we plot everything at the same time<br>
# Let's first look at a pie chart excluding classes with less than 50 members.</p>

# In[ ]:


v_class[v_class > 50.0].plot.pie(autopct='%.2f')


# <p>Doesn't look bad... but maybe we should group meteorites by family (using the link https://en.wikipedia.org/wiki/Meteorite_classification ) <br>
# The following is bit tedious but sometimes you have to do it...</p>
# 

# In[ ]:


#First make a copy of the dataset
tmp_df1 = valids.copy()

#this prevents modification on tmp_df1 to affect v_class


# In[ ]:


tmp_df1.recclass.replace(to_replace=['Acapulcoite', 'Acapulcoite/Lodranite', 'Acapulcoite/lodranite',
           'Lodranite','Lodranite-an','Winonaite','Achondrite-prim'],value='Achondrite_prim',inplace=True)

tmp_df1.recclass.replace(to_replace=['Angrite', 'Aubrite','Aubrite-an','Ureilite', 'Ureilite-an','Ureilite-pmict',
           'Brachinite','Diogenite', 'Diogenite-an', 'Diogenite-olivine', 'Diogenite-pm',
           'Eucrite', 'Eucrite-Mg rich', 'Eucrite-an', 'Eucrite-br','Eucrite-cm',
           'Eucrite-mmict', 'Eucrite-pmict', 'Eucrite-unbr','Howardite'],value='Achondrite_aste',inplace=True)

tmp_df1.recclass.replace(to_replace=['Lunar', 'Lunar (anorth)', 'Lunar (bas. breccia)',
           'Lunar (bas/anor)', 'Lunar (bas/gab brec)', 'Lunar (basalt)',
           'Lunar (feldsp. breccia)', 'Lunar (gabbro)', 'Lunar (norite)'],value='Lunar',inplace=True)

tmp_df1.recclass.replace(to_replace=['Martian', 'Martian (OPX)','Martian (chassignite)', 'Martian (nakhlite)',
           'Martian (shergottite)'],value='Martian',inplace=True)

tmp_df1.recclass.replace(to_replace=['C','C2','C4','C4/5','C6','C1-ung', 'C1/2-ung','C2-ung',
           'C3-ung', 'C3/4-ung','C4-ung','C5/6-ung',
           'CB', 'CBa', 'CBb', 'CH/CBb', 'CH3', 'CH3 ', 'CI1', 'CK', 'CK3',
           'CK3-an', 'CK3.8', 'CK3/4', 'CK4', 'CK4-an', 'CK4/5', 'CK5',
           'CK5/6', 'CK6', 'CM', 'CM-an', 'CM1', 'CM1/2', 'CM2', 'CM2-an',
           'CO3', 'CO3 ', 'CO3.0', 'CO3.1', 'CO3.2', 'CO3.3', 'CO3.4', 'CO3.5',
           'CO3.6', 'CO3.7', 'CO3.8', 'CR', 'CR-an', 'CR1', 'CR2', 'CR2-an',
           'CV2', 'CV3', 'CV3-an'],value='Chondrite_carbon',inplace=True)

tmp_df1.recclass.replace(to_replace=['OC', 'OC3','H', 'H(5?)', 'H(?)4', 'H(L)3', 'H(L)3-an', 'H-an','H-imp melt',
           'H-melt rock', 'H-metal', 'H/L3', 'H/L3-4', 'H/L3.5',
           'H/L3.6', 'H/L3.7', 'H/L3.9', 'H/L4', 'H/L4-5', 'H/L4/5', 'H/L5',
           'H/L6', 'H/L6-melt rock', 'H/L~4', 'H3', 'H3 ', 'H3-4', 'H3-5',
           'H3-6', 'H3-an', 'H3.0', 'H3.0-3.4', 'H3.1', 'H3.10', 'H3.2',
           'H3.2-3.7', 'H3.2-6', 'H3.2-an', 'H3.3', 'H3.4', 'H3.4-5',
           'H3.4/3.5', 'H3.5', 'H3.5-4', 'H3.6', 'H3.6-6', 'H3.7', 'H3.7-5',
           'H3.7-6', 'H3.7/3.8', 'H3.8', 'H3.8-4', 'H3.8-5', 'H3.8-6',
           'H3.8-an', 'H3.8/3.9', 'H3.8/4', 'H3.9', 'H3.9-5', 'H3.9-6',
           'H3.9/4', 'H3/4', 'H4', 'H4 ', 'H4(?)', 'H4-5', 'H4-6', 'H4-an',
           'H4/5', 'H4/6', 'H5', 'H5 ', 'H5-6', 'H5-7', 'H5-an',
           'H5-melt breccia', 'H5/6', 'H6', 'H6 ', 'H6-melt breccia', 'H6/7',
           'H7', 'H?','H~4', 'H~4/5', 'H~5', 'H~6','L', 'L(?)3',
           'L(H)3', 'L(LL)3', 'L(LL)3.05', 'L(LL)3.5-3.7', 'L(LL)5', 'L(LL)6',
           'L(LL)~4', 'L-imp melt', 'L-melt breccia', 'L-melt rock', 'L-metal',
           'L/LL', 'L/LL(?)3', 'L/LL-melt rock', 'L/LL3', 'L/LL3-5', 'L/LL3-6',
           'L/LL3.10', 'L/LL3.2', 'L/LL3.4', 'L/LL3.5', 'L/LL3.6/3.7', 'L/LL4',
           'L/LL4-6', 'L/LL4/5', 'L/LL5', 'L/LL5-6', 'L/LL5/6', 'L/LL6',
           'L/LL6-an', 'L/LL~4', 'L/LL~5', 'L/LL~6', 'L3', 'L3-4', 'L3-5',
           'L3-6', 'L3-7', 'L3.0', 'L3.0-3.7', 'L3.0-3.9', 'L3.05', 'L3.1',
           'L3.10', 'L3.2', 'L3.2-3.5', 'L3.2-3.6', 'L3.3', 'L3.3-3.5',
           'L3.3-3.6', 'L3.3-3.7', 'L3.4', 'L3.4-3.7', 'L3.5', 'L3.5-3.7',
           'L3.5-3.8', 'L3.5-3.9', 'L3.5-5', 'L3.6', 'L3.6-4', 'L3.7',
           'L3.7-3.9', 'L3.7-4', 'L3.7-6', 'L3.7/3.8', 'L3.8', 'L3.8-5',
           'L3.8-6', 'L3.8-an', 'L3.9', 'L3.9-5', 'L3.9-6', 'L3.9/4', 'L3/4',
           'L4', 'L4 ', 'L4-5', 'L4-6', 'L4-an', 'L4-melt rock', 'L4/5', 'L5',
           'L5 ', 'L5-6', 'L5-7', 'L5/6', 'L6', 'L6 ', 'L6-melt breccia',
           'L6-melt rock', 'L6/7', 'L7', 'LL', 'LL(L)3', 'LL-melt rock', 'LL3',
           'LL3-4', 'LL3-5', 'LL3-6', 'LL3.0', 'LL3.00', 'LL3.1', 'LL3.1-3.5',
           'LL3.10', 'LL3.15', 'LL3.2', 'LL3.3', 'LL3.4', 'LL3.5', 'LL3.6',
           'LL3.7', 'LL3.7-6', 'LL3.8', 'LL3.8-6', 'LL3.9', 'LL3.9/4', 'LL3/4',
           'LL4', 'LL4-5', 'LL4-6', 'LL4/5', 'LL4/6', 'LL5', 'LL5-6', 'LL5-7',
           'LL5/6', 'LL6', 'LL6 ', 'LL6(?)', 'LL6/7', 'LL7', 'LL7(?)',
           'LL<3.5', 'LL~3', 'LL~4', 'LL~4/5', 'LL~5', 'LL~6',
           'L~3', 'L~4', 'L~5', 'L~6','Relict H','Relict OC'],value='Chondrite_ordin',inplace=True)

tmp_df1.recclass.replace(to_replace=['EH','EH-imp melt', 'EH3', 'EH3/4-an', 'EH4', 'EH4/5', 'EH5', 'EH6',
           'EH6-an', 'EH7', 'EH7-an', 'EL3', 'EL3/4', 'EL4', 'EL4/5', 'EL5',
           'EL6', 'EL6 ', 'EL6/7', 'EL7','E','E3','E4', 'E5','E6'],value='Chondrite_enst',inplace=True)

tmp_df1.recclass.replace(to_replace=['K', 'K3','R', 'R3', 'R3-4', 'R3-5', 'R3-6', 'R3.4', 'R3.5-6',
           'R3.6', 'R3.7', 'R3.8', 'R3.8-5', 'R3.8-6', 'R3.9', 'R3/4', 'R4',
           'R4/5', 'R5', 'R6'],value='Chondrite_other',inplace=True)

tmp_df1.recclass.replace(to_replace=['Pallasite', 'Pallasite, PES','Pallasite, PMG',
           'Pallasite, PMG-an', 'Pallasite, ungrouped',
           'Pallasite?'],value='Pallasite',inplace=True)

tmp_df1.recclass.replace(to_replace=['Mesosiderite', 'Mesosiderite-A','Mesosiderite-A1',
           'Mesosiderite-A2', 'Mesosiderite-A3','Mesosiderite-A3/4',
           'Mesosiderite-A4', 'Mesosiderite-B','Mesosiderite-B1',
           'Mesosiderite-B2', 'Mesosiderite-B4','Mesosiderite-C',
           'Mesosiderite-C2', 'Mesosiderite-an','Mesosiderite?'],value='Mesosiderite',inplace=True)

tmp_df1.recclass.replace(to_replace=['Iron, IC', 'Iron, IC-an', 'Iron, IIAB', 'Iron, IIAB-an',
           'Iron, IIC', 'Iron, IID', 'Iron, IID-an','Iron, IIF', 'Iron, IIG',
           'Iron, IIIAB', 'Iron, IIIAB-an', 'Iron, IIIAB?', 'Iron, IIIE',
           'Iron, IIIE-an', 'Iron, IIIF', 'Iron, IVA', 'Iron, IVA-an',
           'Iron, IVB'],value='Magmatic',inplace=True)

tmp_df1.recclass.replace(to_replace=['Iron, IAB complex', 'Iron, IAB-MG','Iron, IAB-an', 'Iron, IAB-sHH',
           'Iron, IAB-sHL', 'Iron, IAB-sLH','Iron, IAB-sLL', 'Iron, IAB-sLM',
           'Iron, IAB-ung', 'Iron, IAB?','Iron, IIE',
           'Iron, IIE-an', 'Iron, IIE?'],value='Non_magmatic',inplace=True)

tmp_df1.recclass.replace(to_replace=['Iron','Iron?','Relict iron','Chondrite-fusion crust',
           'Fusion crust','Impact melt breccia',
           'Enst achon-ung','Iron, ungrouped','Stone-uncl', 'Stone-ung',
           'Unknown','Achondrite-ung','Chondrite-ung',
           'Enst achon','E-an',  'E3-an',  'E5-an'],value='Unknown-Ungrouped',inplace=True)


# In[ ]:


#Counting again occurences ...
tmp_df1['recclass'].value_counts()


# In[ ]:


# ... and visualizing it
tmp_df1['recclass'].value_counts().plot.pie(autopct='%.2f')


# <p>From the previous graph we see that the class is highly dominated by ordinary chondrite meteorites (nearly 90%).
# If you plot the number of discoveries of this class as a function of time it will look like the general trend and we won't gain much information.<br>
# Going back to the first pie chart then it seems that "H", "L" and "LL" are more frequent than others then let's group meteorites in four classes : "H", "L", "LL" and "Other".</p>

# In[ ]:


#Let's make another copy
tmp_df2 = valids.copy()


# In[ ]:


tmp_df2.recclass.replace(to_replace=['Acapulcoite', 'Acapulcoite/Lodranite', 'Acapulcoite/lodranite',
           'Lodranite','Lodranite-an','Winonaite','Achondrite-prim','Angrite',
           'Aubrite','Aubrite-an','Ureilite', 'Ureilite-an','Ureilite-pmict',
           'Brachinite','Diogenite', 'Diogenite-an', 'Diogenite-olivine', 'Diogenite-pm',
           'Eucrite', 'Eucrite-Mg rich', 'Eucrite-an', 'Eucrite-br','Eucrite-cm',
           'Eucrite-mmict', 'Eucrite-pmict', 'Eucrite-unbr','Howardite',
           'Lunar', 'Lunar (anorth)', 'Lunar (bas. breccia)',
           'Lunar (bas/anor)', 'Lunar (bas/gab brec)', 'Lunar (basalt)',
           'Lunar (feldsp. breccia)', 'Lunar (gabbro)', 'Lunar (norite)',
           'Martian', 'Martian (OPX)','Martian (chassignite)', 'Martian (nakhlite)',
           'Martian (shergottite)','C','C2','C4','C4/5','C6','C1-ung', 'C1/2-ung','C2-ung',
           'C3-ung', 'C3/4-ung','C4-ung','C5/6-ung', 'CB', 'CBa', 'CBb', 'CH/CBb',
           'CH3', 'CH3 ', 'CI1', 'CK', 'CK3','CK3-an', 'CK3.8', 'CK3/4', 'CK4', 'CK4-an', 'CK4/5', 'CK5',
           'CK5/6', 'CK6', 'CM', 'CM-an', 'CM1', 'CM1/2', 'CM2', 'CM2-an',
           'CO3', 'CO3 ', 'CO3.0', 'CO3.1', 'CO3.2', 'CO3.3', 'CO3.4', 'CO3.5',
           'CO3.6', 'CO3.7', 'CO3.8', 'CR', 'CR-an', 'CR1', 'CR2', 'CR2-an',
           'CV2', 'CV3', 'CV3-an', 'OC', 'OC3','Relict H','Relict OC',
           'EH','EH-imp melt', 'EH3', 'EH3/4-an', 'EH4', 'EH4/5', 'EH5', 'EH6',
           'EH6-an', 'EH7', 'EH7-an', 'EL3', 'EL3/4', 'EL4', 'EL4/5', 'EL5',
           'EL6', 'EL6 ', 'EL6/7', 'EL7','E','E3','E4', 'E5','E6',
           'K', 'K3','R', 'R3', 'R3-4', 'R3-5', 'R3-6', 'R3.4', 'R3.5-6',
           'R3.6', 'R3.7', 'R3.8', 'R3.8-5', 'R3.8-6', 'R3.9', 'R3/4', 'R4',
           'R4/5', 'R5', 'R6','Pallasite', 'Pallasite, PES','Pallasite, PMG',
           'Pallasite, PMG-an', 'Pallasite, ungrouped',
           'Pallasite?','Mesosiderite', 'Mesosiderite-A','Mesosiderite-A1',
           'Mesosiderite-A2', 'Mesosiderite-A3','Mesosiderite-A3/4',
           'Mesosiderite-A4', 'Mesosiderite-B','Mesosiderite-B1',
           'Mesosiderite-B2', 'Mesosiderite-B4','Mesosiderite-C',
           'Mesosiderite-C2', 'Mesosiderite-an','Mesosiderite?',
           'Iron, IC', 'Iron, IC-an', 'Iron, IIAB', 'Iron, IIAB-an',
           'Iron, IIC', 'Iron, IID', 'Iron, IID-an','Iron, IIF', 'Iron, IIG',
           'Iron, IIIAB', 'Iron, IIIAB-an', 'Iron, IIIAB?', 'Iron, IIIE',
           'Iron, IIIE-an', 'Iron, IIIF', 'Iron, IVA', 'Iron, IVA-an',
           'Iron, IVB','Iron, IAB complex', 'Iron, IAB-MG','Iron, IAB-an', 'Iron, IAB-sHH',
           'Iron, IAB-sHL', 'Iron, IAB-sLH','Iron, IAB-sLL', 'Iron, IAB-sLM',
           'Iron, IAB-ung', 'Iron, IAB?','Iron, IIE',
           'Iron, IIE-an', 'Iron, IIE?','Iron','Iron?','Relict iron','Chondrite-fusion crust',
           'Fusion crust','Impact melt breccia',
           'Enst achon-ung','Iron, ungrouped','Stone-uncl', 'Stone-ung',
           'Unknown','Achondrite-ung','Chondrite-ung',
           'Enst achon','E-an',  'E3-an',  'E5-an'],value='Others',inplace=True)

tmp_df2.recclass.replace(to_replace=['H', 'H(5?)', 'H(?)4', 'H(L)3', 'H(L)3-an', 'H-an','H-imp melt',
           'H-melt rock', 'H-metal', 'H/L3', 'H/L3-4', 'H/L3.5',
           'H/L3.6', 'H/L3.7', 'H/L3.9', 'H/L4', 'H/L4-5', 'H/L4/5', 'H/L5',
           'H/L6', 'H/L6-melt rock', 'H/L~4', 'H3', 'H3 ', 'H3-4', 'H3-5',
           'H3-6', 'H3-an', 'H3.0', 'H3.0-3.4', 'H3.1', 'H3.10', 'H3.2',
           'H3.2-3.7', 'H3.2-6', 'H3.2-an', 'H3.3', 'H3.4', 'H3.4-5',
           'H3.4/3.5', 'H3.5', 'H3.5-4', 'H3.6', 'H3.6-6', 'H3.7', 'H3.7-5',
           'H3.7-6', 'H3.7/3.8', 'H3.8', 'H3.8-4', 'H3.8-5', 'H3.8-6',
           'H3.8-an', 'H3.8/3.9', 'H3.8/4', 'H3.9', 'H3.9-5', 'H3.9-6',
           'H3.9/4', 'H3/4', 'H4', 'H4 ', 'H4(?)', 'H4-5', 'H4-6', 'H4-an',
           'H4/5', 'H4/6', 'H5', 'H5 ', 'H5-6', 'H5-7', 'H5-an',
           'H5-melt breccia', 'H5/6', 'H6', 'H6 ', 'H6-melt breccia', 'H6/7',
           'H7', 'H?','H~4', 'H~4/5', 'H~5', 'H~6'],value='H',inplace=True)

tmp_df2.recclass.replace(to_replace=['L', 'L(?)3','L(H)3', 'L(LL)3', 'L(LL)3.05', 'L(LL)3.5-3.7', 'L(LL)5', 'L(LL)6',
           'L(LL)~4', 'L-imp melt', 'L-melt breccia', 'L-melt rock', 'L-metal',
           'L/LL', 'L/LL(?)3', 'L/LL-melt rock', 'L/LL3', 'L/LL3-5', 'L/LL3-6',
           'L/LL3.10', 'L/LL3.2', 'L/LL3.4', 'L/LL3.5', 'L/LL3.6/3.7', 'L/LL4',
           'L/LL4-6', 'L/LL4/5', 'L/LL5', 'L/LL5-6', 'L/LL5/6', 'L/LL6',
           'L/LL6-an', 'L/LL~4', 'L/LL~5', 'L/LL~6', 'L3', 'L3-4', 'L3-5',
           'L3-6', 'L3-7', 'L3.0', 'L3.0-3.7', 'L3.0-3.9', 'L3.05', 'L3.1',
           'L3.10', 'L3.2', 'L3.2-3.5', 'L3.2-3.6', 'L3.3', 'L3.3-3.5',
           'L3.3-3.6', 'L3.3-3.7', 'L3.4', 'L3.4-3.7', 'L3.5', 'L3.5-3.7',
           'L3.5-3.8', 'L3.5-3.9', 'L3.5-5', 'L3.6', 'L3.6-4', 'L3.7',
           'L3.7-3.9', 'L3.7-4', 'L3.7-6', 'L3.7/3.8', 'L3.8', 'L3.8-5',
           'L3.8-6', 'L3.8-an', 'L3.9', 'L3.9-5', 'L3.9-6', 'L3.9/4', 'L3/4',
           'L4', 'L4 ', 'L4-5', 'L4-6', 'L4-an', 'L4-melt rock', 'L4/5', 'L5',
           'L5 ', 'L5-6', 'L5-7', 'L5/6', 'L6', 'L6 ', 'L6-melt breccia',
           'L6-melt rock', 'L6/7', 'L7','L~3', 'L~4', 'L~5', 'L~6'],value='L',inplace=True)

tmp_df2.recclass.replace(to_replace=['LL', 'LL(L)3', 'LL-melt rock', 'LL3',
           'LL3-4', 'LL3-5', 'LL3-6', 'LL3.0', 'LL3.00', 'LL3.1', 'LL3.1-3.5',
           'LL3.10', 'LL3.15', 'LL3.2', 'LL3.3', 'LL3.4', 'LL3.5', 'LL3.6',
           'LL3.7', 'LL3.7-6', 'LL3.8', 'LL3.8-6', 'LL3.9', 'LL3.9/4', 'LL3/4',
           'LL4', 'LL4-5', 'LL4-6', 'LL4/5', 'LL4/6', 'LL5', 'LL5-6', 'LL5-7',
           'LL5/6', 'LL6', 'LL6 ', 'LL6(?)', 'LL6/7', 'LL7', 'LL7(?)',
           'LL<3.5', 'LL~3', 'LL~4', 'LL~4/5', 'LL~5', 'LL~6'],value='LL',inplace=True)


# In[ ]:


# Count
tmp_df2['recclass'].value_counts()


# In[ ]:


# Visualize
tmp_df2['recclass'].value_counts().plot.pie(autopct='%.2f')


# <p>From here we can look at the frequency of discoveries as a function of time.</p>
# 

# In[ ]:


# Here I just create "shortcuts" to each class
Hs = tmp_df2[tmp_df2.recclass=='H']
Ls = tmp_df2[tmp_df2.recclass=='L']
LLs = tmp_df2[tmp_df2.recclass=='LL']
Oth = tmp_df2[tmp_df2.recclass=='Others']


# In[ ]:


plt.figure(figsize=(7,10))

plt.subplot(411)
_=plt.hist(Hs.year.values,bins=np.arange(1900,2014,2),lw=2,histtype='step')
plt.ylabel('H class')
_=plt.xticks(np.arange(1900,2030,10))

plt.subplot(412)
_=plt.hist(Ls.year.values,bins=np.arange(1900,2014,2),lw=2,histtype='step')
plt.ylabel('L class')
_=plt.xticks(np.arange(1900,2030,10))

plt.subplot(413)
_=plt.hist(LLs.year.values,bins=np.arange(1900,2014,2),lw=2,histtype='step')
plt.ylabel('LL class')
_=plt.xticks(np.arange(1900,2030,10))

plt.subplot(414)
_=plt.hist(Oth.year.values,bins=np.arange(1900,2014,2),lw=2,histtype='step')
plt.ylabel('Others')
_=plt.xticks(np.arange(1900,2030,10))


# In[ ]:


#To be continued

