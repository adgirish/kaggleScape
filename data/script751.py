
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


df_events = pd.read_csv("../input/events.csv")
df_events.shape


# In[ ]:


df_events.head()


# In[ ]:


fig = plt.figure(figsize=(16, 12))
markersize = 1
markertype = ',' # pixel
markercolor = '#444444'
markeralpha = .8 #  a bit of transparency

# http://isithackday.com/geoplanet-explorer/index.php?woeid=23424781
# Location (lat/lon): 36.894402, 104.166
# Bounding Box:
# NE 53.5606, 134.773605
# SW 15.77539, 73.557701
m = Basemap(
    projection='mill', lon_0=104.166, lat_0=36.894402,
    llcrnrlon=73.557701, llcrnrlat=15.77539,
    urcrnrlon=134.773605, urcrnrlat=53.5606)

# avoid border around map
m.drawmapboundary(fill_color='#ffffff', linewidth=.0)

# draw event locations
x, y = m(df_events.longitude.values, df_events.latitude.values)
m.scatter(x, y, markersize, marker=markertype, color=markercolor, alpha=markeralpha)

# annotations
plt.annotate('TalkingData Event Locations', xy=(0.02, .96), size=20, xycoords='axes fraction')
footer = 'Author: Ramiro Gomez - ramiro.org | Data: TalkingData provided by kaggle.com'
plt.annotate(footer, xy=(0.02, -0.02), size=14, xycoords='axes fraction')

plt.show()

