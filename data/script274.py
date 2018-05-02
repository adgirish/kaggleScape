
# coding: utf-8

# # Global Terrorism (1970 - 2015)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')

try:
    t_file = pd.read_csv('../input/globalterrorismdb_0616dist.csv', encoding='ISO-8859-1')
    print('File load: Success')
except:
    print('File load: Failed')


# In[ ]:


t_file = t_file[np.isfinite(t_file.latitude)]


# In[ ]:


t_file.head()


# In[ ]:


regions = list(set(t_file.region_txt))
colors = ['yellow', 'red', 'lightblue', 'purple', 'green', 'orange', 'brown',          'aqua', 'lightpink', 'lightsage', 'lightgray', 'navy']


# In[ ]:


plt.figure(figsize=(15,8))
m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='burlywood',lake_color='lightblue', zorder = 1)
m.drawmapboundary(fill_color='lightblue')

def pltpoints(region, color = None, label = None):
    x, y = m(list(t_file.longitude[t_file.region_txt == region].astype("float")),            (list(t_file.latitude[t_file.region_txt == region].astype("float"))))
    points = m.plot(x, y, "o", markersize = 4, color = color, label = label, alpha = .5)
    return(points)

for i, region in enumerate(regions):
    pltpoints(region, color = colors[i], label = region)  
    
plt.title("Global Terrorism (1970 - 2015)")
plt.legend(loc ='lower left', prop= {'size':11})
plt.show()    


# **From the graph above, we can see, that terrorism is widespread, but judging by where the points are located, and quite obviously, it mostly affects areas that are more densley populated.**

# In[ ]:


count_year = t_file.groupby(['iyear']).count()
mean_year = t_file.groupby(['iyear']).mean()

fig = plt.figure(figsize = (10,8))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.set(title = 'Total acts of terrorism', ylabel = 'Act Count', xlabel = 'Year')
ax1.plot(count_year.index, count_year.eventid)
ax2.set(title = 'Average Number of Deaths per Act', ylabel = 'Death Count', xlabel = 'Year')
ax2.plot(mean_year.index, mean_year.nkill)
fig.autofmt_xdate()


# **As we can see from the above graphs, not only has the number of terroristic acts increased, but also the number of deaths per act hs been on the rise. This could possible be due to there being more densely populated areas over time.**

# In[ ]:


region_mean_kills = []
for region in regions:
    region_mean_kills.append(t_file.nkill[t_file.region_txt == region].mean())

print('Average number of people killed per attack by Region\n')
for i, region in enumerate(regions):
    print('{}:{}'.format(region, round(region_mean_kills[i],2)))


# **We can also note, that on average, every terror attack in Sub-Saharan Africa claims over 5 lives.**
# 

# In[ ]:


def mapmean(row):
    for i, region in enumerate(regions):
        return region_mean_kills[i]


# In[ ]:


t_file['region_mean'] = t_file.apply(mapmean, axis = 1)
t_file['nkill-mean'] = t_file['nkill'] - t_file['region_mean']
t_file['absnkill-mean'] = abs(t_file['nkill-mean'])


# In[ ]:


def get_points(year, region = regions):
    points = t_file[['iyear', 'latitude', 'longitude', 'nkill', 'region_mean', 'nkill-mean', 'absnkill-mean']][t_file.iyear == year]
    return(points)


# # Lastly:
#     
# **Here is an animation of how terrorism has progressed from 1970 through 2015**

# In[ ]:


fig = plt.figure(figsize=(10, 10))
fig.text(.8, .3, 'R. Troncoso', ha='right')
fig.suptitle('Global Terrorism (1970 - 2015)')
cmap = plt.get_cmap('coolwarm')

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='burlywood',lake_color='lightblue', zorder = 1)
m.drawmapboundary(fill_color='lightblue')

START_YEAR = 1970
LAST_YEAR = 2015

points = get_points(START_YEAR)
x, y= m(list(points['longitude']), list(points['latitude']))
scat = m.scatter(x, y, s = points['absnkill-mean']*2, marker='o', alpha=0.3, zorder=10, c = points['nkill-mean'], cmap = cmap)
year_text = plt.text(-170, 80, str(START_YEAR),fontsize=15)
plt.close()

def update(frame_number):
    current_year = START_YEAR + (frame_number % (LAST_YEAR - START_YEAR + 1))
    points = get_points(current_year)
    color = list(points['nkill-mean'])
    x, y = m(list(points['longitude']), list(points['latitude']))
    scat.set_offsets(np.dstack((x, y)))
    scat.set_color(cmap(points['nkill-mean']))
    scat.set_sizes(points['absnkill-mean']*1.5)
    year_text.set_text(str(current_year))
    
ani = animation.FuncAnimation(fig, update, interval=750, frames=LAST_YEAR - START_YEAR + 1)
ani.save('animation.gif', writer='imagemagick', fps=2)


# In[ ]:


import io
import base64

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# **The points above represent the all terrorist attacks.**
# 
# **The color and size represent the number of people killed during that particular attack.**
# 
# **As you can see, there has been an increase in attacks, as well as the number of deaths per attack has increased, particularly in Sub-Sahararan Africa and the Middle East.**
