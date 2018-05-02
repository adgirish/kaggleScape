
# coding: utf-8

# Visualizing earthquakes (1965-2016)
# -----------------------------------

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
    t_file = pd.read_csv('../input/database.csv', encoding='ISO-8859-1')
    print('File load: Success')
except:
    print('File load: Failed')


# In[ ]:


t_file['Year']= t_file['Date'].str[6:]


# In[ ]:


fig = plt.figure(figsize=(10, 10))
fig.text(.8, .3, 'Soumitra', ha='right')
cmap = plt.get_cmap('coolwarm')

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='burlywood',lake_color='lightblue', zorder = 1)
m.drawmapboundary(fill_color='lightblue')


START_YEAR = 1965
LAST_YEAR = 2016

points = t_file[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']][t_file['Year']==str(START_YEAR)]

x, y= m(list(points['Longitude']), list(points['Latitude']))
scat = m.scatter(x, y, s = points['Magnitude']*points['Depth']*0.3, marker='o', alpha=0.3, zorder=10, cmap = cmap)
year_text = plt.text(-170, 80, str(START_YEAR),fontsize=15)
plt.title("Earthquake visualisation (1965 - 2016)")
plt.close()


def update(frame_number):
    current_year = START_YEAR + (frame_number % (LAST_YEAR - START_YEAR + 1))
    year_text.set_text(str(current_year))
    points = t_file[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']][t_file['Year']==str(current_year)]
    x, y= m(list(points['Longitude']), list(points['Latitude']))
    color = points['Depth']*points['Magnitude'];
    scat.set_offsets(np.dstack((x, y)))
    scat.set_sizes(points['Magnitude']*points['Depth']*0.3)
    
ani = animation.FuncAnimation(fig, update, interval=750, frames=LAST_YEAR - START_YEAR + 1)
ani.save('animation.gif', writer='imagemagick', fps=5)


# In[ ]:


import io
import base64

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# **The points above represent the all the earthquakes in the given timeframe.**
# <br>
# The size represents the approximate amount of strength/sensation on the surface, taking in account the depth and magnitude. We can see that the west coast of Southern America and South East Asia experience the worst earthquakes. Most earthquakes near the Sumatran region are underwater thus resulting in frequent tsunamis. 

# The script is inspired from Ronald Troncoso's work on the Global terrorist dataset. (https://www.kaggle.com/ronaldtroncoso20/d/START-UMD/gtd/global-terrorism-trends-animation)
