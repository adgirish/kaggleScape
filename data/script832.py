
# coding: utf-8

# This small notebook outputs a single .gpx file with all the inquiries that have received a high interest level. Feel free to use it and deepen your exploratory data analysis with the help of Google Earth.
# 
# First, import the required packages.

# In[ ]:


#Author Justin Neumann

import json
import pandas as pd
import gpxpy as gpx
import gpxpy.gpx


# Then import the training set and create a new data frame with latitude, longitude and interest_level. Note: There is space for improvement in this code section.

# In[ ]:


#import training data
with open('../input/train.json') as data_file:
    data = json.load(data_file)
train = pd.DataFrame(data)

train.head(1)


# Now create the structure of the .gpx file.

# In[ ]:


# create gpx file as from https://pypi.python.org/pypi/gpxpy/0.8.8
gpx = gpxpy.gpx.GPX()

for index, row in train.iterrows():
    #print (row['latitude'], row['longitude'])

    if row['interest_level'] == 'high': #opting for all nominals results in poor performance of Google Earth
        gps_waypoint = gpxpy.gpx.GPXWaypoint(row['latitude'],row['longitude'],elevation=10)
        gpx.waypoints.append(gps_waypoint)

# You can add routes and waypoints, too...


# Finally, export the .gpx file and feel free to use it in Google Earth via Tools->GPS->Import from File. I do not recommend to include all inquiries because of poor performance of Google Earth.

# In[ ]:


filename = "test.gpx"
FILE = open(filename,"w")
FILE.writelines(gpx.to_xml())
FILE.close()
print ('Created GPX:')


# The result looks like ![enter image description here][1]
# 
# 
#   [1]: http://i.imgur.com/46EuCPC.jpg
