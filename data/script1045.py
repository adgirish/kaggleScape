
# coding: utf-8

# # What is Folium
# Folium is a tool that makes you look like a mapping God while all the work is done in the back end.
# 
# It's a Python wrapper for a tool called leaflet.js. We basically give it minimal instructions, JS does loads of work in the background and we get some very, very cool maps. It's great stuff.
# 
# For clarity, the map is technically called a 'Leaflet Map'. The tool that let's you call them in Python is called 'Folium'. 
# 
# # The other cool stuff?
# It gives you interactive functionality. Want to let users drop markers on the map? Can do. Build heatmaps? Can do. Build heatmaps that change with time? Can do.
# 
# Funk yeah! Let's do this.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import folium

#df_traffic = pd.read_csv('../input/ukTrafficAADF.csv')
df_acc = pd.read_csv('../input/accidents_2005_to_2007.csv', dtype=object)


# In[ ]:


map_hooray = folium.Map(location=[51.5074, 0.1278],
                    zoom_start = 11) # Uses lat then lon. The bigger the zoom number, the closer in you get
map_hooray # Calls the map to display

#map_osm = folium.Map(location=[54.7, -4.36])
#This 2nd set of coordinates will drop you down right in the middle of the UK, which is actually the seas because it's between mainland and N.Ireland, 


# # Fun visual styles

# In[ ]:


t_list = ["Stamen Terrain", "Stamen Toner", "Mapbox Bright"]
map_hooray = folium.Map(location=[51.5074, 0.1278],
                        tiles = "Stamen Terrain",
                        zoom_start = 12)
map_hooray


# In[ ]:


map_hooray = folium.Map(location=[51.5074, 0.1278],
                        tiles = "Stamen Toner",
                        zoom_start = 12)
map_hooray


# # Markers
# 
# These are defined outside the map. This is similar to a basmap. Once you've set the location, zoom, style, i.e. the place, everything else is an addition that's placed over the top, so it's called and added to (.add_to) the map.
# 
# Note the 'popup attribute. This text appears on clicking the map.

# In[ ]:


map_hooray = folium.Map(location=[51.5074, 0.1278],
                        tiles = "Stamen Toner",
                        zoom_start = 12)
# 'width=int' and 'height=int' can also be added to the map

folium.Marker([51.5079, 0.0877], popup='London Bridge').add_to(map_hooray)
map_hooray


# ## More complex markers
# **The London Bridge marker** is the same as above, I just added a little colour with that extra line of green
# 
# **The second marker** is more interesting. First, it's not a marker like the pin. The pin is an icon type marker. The circle is essentially just a coloured overlay, so we use a different colour command. 
# 
# CircleMarker radius is set in pixels so if you change the zoom you need to change the pixels. It can also take  a fill_color that's semi-transparent.
# 
# ### Interactive markers
# Let your users add marker with
# > markers = map_hooray.add_child(folium.ClickForMarker(popup="pop_up_name"))
# 
# Literally add that code to your map and then users can click anywhere to add their own marker.

# In[ ]:


# Set the map up
map_hooray = folium.Map(location=[51.5074, 0.1278],
                        tiles = "Stamen Toner",
                        zoom_start = 9)
# Simple marker
folium.Marker([51.5079, 0.0877],
              popup='London Bridge',
              icon=folium.Icon(color='green')
             ).add_to(map_hooray)

# Circle marker
folium.CircleMarker([51.4183, 0.2206],
                    radius=30,
                    popup='East London',
                    color='red',
                    ).add_to(map_hooray)

# Interactive marker
map_hooray.add_child(folium.ClickForMarker(popup="Dave is awesome"))

map_hooray


# # Interaction with the map

# In[ ]:


map_hooray = folium.Map(location=[51.5074, 0.1278],
                    zoom_start = 11) # Uses lat then lon. The bigger the zoom number, the closer in you get

from folium import plugins

# Adds tool to the top right
from folium.plugins import MeasureControl
map_hooray.add_child(MeasureControl())

# Fairly obvious I imagine - works best with transparent backgrounds
from folium.plugins import FloatImage
url = ('https://media.licdn.com/mpr/mpr/shrinknp_100_100/AAEAAQAAAAAAAAlgAAAAJGE3OTA4YTdlLTkzZjUtNDFjYy1iZThlLWQ5OTNkYzlhNzM4OQ.jpg')
FloatImage(url, bottom=5, left=85).add_to(map_hooray)

map_hooray


# # Other marker types
# 
# I'm skipping over a few markers to move to more interetsing analysis but it's worth knowing that you can also employ. Polygons are markers that let you choose the shape.
# 
# folium.RegularPolygonMarker(
#     [lat, lon],
#     popup='name',
#     fill_color='color name',
#     number_of_sides= integer,
#     radius=pixels
#     ).add_to(map_name)
#     
# ### You can also use Vincent/Vega markers
# These are clickable effects. So far, we've just seen text pop-ups. Vincent markers use additional JS to pull in graphical overlays, e.g. click on a pop-up to see the timeline of it's history. You can see an example of them in the Folium documentation, https://folium.readthedocs.io/en/latest/quickstart.html#vincent-vega-markers
# 

# ## Add icons from fontawesome.io
# 
# Reference the "prefix='fa'" to pull icons from fontawesome.io
# 
# Run help(folium.Icon) to get the full documentation on what you can do with icons

# In[ ]:


map_hooray = folium.Map(location=[51.5074, 0.1278],
                        tiles = "Stamen Toner",
                        zoom_start = 9)

folium.Marker([51.5079, 0.0877],
              popup='London Bridge',
              icon=folium.Icon(color='green')
             ).add_to(map_hooray)
folium.Marker([51.5183, 0.5206], 
              popup='East London',
              icon=folium.Icon(color='red',icon='university', prefix='fa') 
             ).add_to(map_hooray)

folium.Marker([51.5183, 0.3206], 
              popup='East London',
              icon=folium.Icon(color='blue',icon='bar-chart', prefix='fa') 
             ).add_to(map_hooray)
             # icon=folium.Icon(color='red',icon='bicycle', prefix='fa')

map_hooray.add_child(folium.ClickForMarker(popup="Dave is awesome"))

map_hooray


# # Heatmaps, boo-ya!
# Definitely one of the best functions in Folium. This does not take Dataframes. You'll need to give it a list of lat, lons, i.e. a list of lists. It should be like this. NaNs will also trip it up,
# > [[lat, lon],[lat, lon],[lat, lon],[lat, lon],[lat, lon]]

# In[ ]:


from folium import plugins
from folium.plugins import HeatMap


map_hooray = folium.Map(location=[51.5074, 0.1278],
                    zoom_start = 13) 

# Ensure you're handing it floats
df_acc['Latitude'] = df_acc['Latitude'].astype(float)
df_acc['Longitude'] = df_acc['Longitude'].astype(float)

# Filter the DF for rows, then columns, then remove NaNs
heat_df = df_acc[df_acc['Speed_limit']=='30'] # Reducing data size so it runs faster
heat_df = heat_df[heat_df['Year']=='2007'] # Reducing data size so it runs faster
heat_df = heat_df[['Latitude', 'Longitude']]
heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude'])

# List comprehension to make out list of lists
heat_data = [[row['Latitude'],row['Longitude']] for index, row in heat_df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(map_hooray)

# Display the map
map_hooray



# # Heatmap with time series
# 
# This is very similat to Heatmap, just one touch more complicated. It takes a list of list OF LISTS! Yep, another layer deep.
# 
# In this example we organise it by month. So we have 12 lists of lists, e.g.
# 
# Jen = [[lat,lon],[lat,lon],[lat,lon]]
# Feb = [[lat,lon],[lat,lon],[lat,lon]]
# March = [[lat,lon],[lat,lon],[lat,lon]]
# 
# list of lists of lists = [Jan, Feb, March]
# that looks like [[[lat,lon],[lat,lon],[lat,lon]],[[lat,lon],[lat,lon],[lat,lon]],[[lat,lon],[lat,lon],[lat,lon]]]
# 
# To understand that better you should use Ctrl+F to spot the double brackets, '[[', and fint eh sub lists.
# 
# To make that happen you use a list comprehesntion within a list comprehension. You can see that below where I declare 'heat_data = '
# 
# I break this down a little further below the map.
# 
# **For reasons I don't understand the play, forward, backward buttons are missing their logos but they do work on the bottom left.**

# In[ ]:


from folium import plugins

map_hooray = folium.Map(location=[51.5074, 0.1278],
                    zoom_start = 13) 

# Ensure you're handing it floats
df_acc['Latitude'] = df_acc['Latitude'].astype(float)
df_acc['Longitude'] = df_acc['Longitude'].astype(float)

# Filter the DF for rows, then columns, then remove NaNs
heat_df = df_acc[df_acc['Speed_limit']=='40'] # Reducing data size so it runs faster
heat_df = heat_df[heat_df['Year']=='2007'] # Reducing data size so it runs faster
heat_df = heat_df[['Latitude', 'Longitude']]

# Create weight column, using date
heat_df['Weight'] = df_acc['Date'].str[3:5]
heat_df['Weight'] = heat_df['Weight'].astype(float)
heat_df = heat_df.dropna(axis=0, subset=['Latitude','Longitude', 'Weight'])

# List comprehension to make out list of lists
heat_data = [[[row['Latitude'],row['Longitude']] for index, row in heat_df[heat_df['Weight'] == i].iterrows()] for i in range(0,13)]

# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(map_hooray)
# Display the map
map_hooray



# # The lists of lists of lists 
# ### a.k.a. list comprehension within a list comprehension)
# 
# **Here's the line we used.**
# 
# > heat_data = [[[row['Latitude'],row['Longitude']] for index, row in heat_df[heat_df['Weight'] == i].iterrows()] for i in range(0,13)]
# 
#    
# **Rewriting list comprehensions as regular Python can be helpful**
# > heat_data1 = []
# 
# > for i in range(0,13):
# 
# >     heat_data2 = []
# 
# >     heat_data1.append(heat_data2)
# 
# >     for index, row in heat_df[heat_df['Weight'] == i].iterrows():
# 
# >         lt_lon = [row['Latitude'],row['Longitude']]
# 
# >         heat_data2.append(lat_lon)

# # Plugins
# There are too many to demo them all but check out this notebook to see the additional plugins you can use. Likely to be of interest are MarkerCluster and Fullscreen.
# 
# http://nbviewer.jupyter.org/github/python-visualization/folium/blob/master/examples/Plugins.ipynb 
# 
# 
