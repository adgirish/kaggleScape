
# coding: utf-8

# Introduction
# ------------
# 
# The purpose of this kernel is to give you a quick introduction to this Dataset.  You’ll probably want to start with the **accident.csv** table.  To get information about each table and their fields, consult  the [FARS User’s Manual.](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812315) 
# 
# After getting a feel for the data, you might want to read the [Traffic Safety Facts Research Note](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812318).  A page from this report is shown below.
# 
# ![Traffic Safety Facts Research Note][1]
# 
# 
# You might be interested in reading the [White House Blog](https://www.whitehouse.gov/blog/2016/08/29/2015-traffic-fatalities-data-has-just-been-released-call-action-download-and-analyze) on getting Data Scientist to share their finds on this data.
# 
# 
#   [1]: https://storage.googleapis.com/montco-stats/kaggleImages/trafficDeathSummary.png

# ## Listing the Files ##
# 
# Note: Some files are under **../input/extra**

# In[ ]:


# Listing the files
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

print("../input/extra")
print(check_output(["ls", "../input/extra"]).decode("utf8"))


# accident.csv
# ------------
# 

# In[ ]:


import pandas as pd
import numpy as np
import datetime


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)


# Good for interactive plots
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()


# You might want to get started with accident.csv 

# Read data accident.csv

FILE="../input/accident.csv"
d=pd.read_csv(FILE)


# crashTime
# ---------
# 
# You might want crashTime in datetime format.  Careful, since sometimes the HOUR and MINUTE could be unknown (has a value of 99).  
# 
# Note these are local times at the crash site.

# In[ ]:



def f(x):
    year = x[0]
    month = x[1]
    day = x[2]
    hour = x[3]
    minute = x[4]
    # Sometimes they don't know hour and minute
    if hour == 99:
        hour = 0
    if minute == 99:
        minute = 0
    s = "%02d-%02d-%02d %02d:%02d:00" % (year,month,day,hour,minute)
    c = datetime.datetime.strptime(s,'%Y-%m-%d %H:%M:%S')
    return c
 
d['crashTime']   = d[['YEAR','MONTH','DAY','HOUR','MINUTE']].apply(f, axis=1)
d['crashDay']    = d['crashTime'].apply(lambda x: x.date())
d['crashMonth']  = d['crashTime'].apply(lambda x: x.strftime("%B") )
d['crashMonthN'] = d['crashTime'].apply(lambda x: x.strftime("%d") ) # sorting
d['crashTime'].head()


# In[ ]:


d.head()


# In[ ]:


d.count()[0]


# Motorists in the crash
# ----------------------

# In[ ]:


# Take a look at breakdown by PERSONS (Motorists in the crash - don't assume killed)
d["PERSONS"].value_counts()


# In[ ]:


# Total
d["PERSONS"].sum()


# FATALS
# ------

# In[ ]:


# Broken down by FATALS
d["FATALS"].value_counts()


# In[ ]:


# Total
d["FATALS"].sum()


# Google Map (3 or more FATALS for incident)
# ------------------------------------------

# In[ ]:


# Where are the 3 or more FATALS per incident?
#  Reference:
#    https://www.kaggle.com/mchirico/d/nhtsa/2015-traffic-fatalities/fatalities-3-or-more
import IPython
url = 'https://www.kaggle.io/svf/474808/8194665eeb7f21d92f65ffa5f17c285f/output.html'
iframe = '<iframe src=' + url + ' width=700 height=525></iframe>'
IPython.display.HTML(iframe)


# The above map shows a single Traffic Incident (row in the accident.csv table), having 3 or more fatalities. If you mouse over an icon, you'll see the details on the crash.  This can be a quick way to pan around and find a particular devastating accident.  
# 
# You can reference the Kernel above [here](https://www.kaggle.com/mchirico/d/nhtsa/2015-traffic-fatalities/fatalities-3-or-more)

# In[ ]:



# School Bus Fatalities 

import IPython
url = 'https://www.kaggle.io/svf/474975/0021b5e39cead137f450588c873eae28/output.html'
iframe = '<iframe src=' + url + ' width=700 height=525></iframe>'
IPython.display.HTML(iframe)


# If you mouse hover over the bus icons, you'll see there are not a large number of fatalities.

# In[ ]:


# Bicycle Fatalities  
import IPython
url = 'https://www.kaggle.io/svf/473865/4ba7ff04c62cb2b89155f486e0393ae7/output.html'
iframe = '<iframe src=' + url + ' width=700 height=525></iframe>'
IPython.display.HTML(iframe)


# The dark green lines above are bike routes.  Vehicle color traffic on the map is live.  So, depending on when you're viewing this, you might see heavy or light traffic.  Yes.  The above map is a Kaggle Kernel.  The code can be found [here](https://www.kaggle.com/mchirico/d/nhtsa/2015-traffic-fatalities/bike-zoom-chicago-map/code).

# Decoding States
# ---------------
# 
# Reference [FARS User’s Manual.](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812315) 

# In[ ]:


states = {1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 
          6: 'California', 8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 
          11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 
          16: 'Idaho', 17: 'Illinois', 18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 
          21: 'Kentucky', 22: 'Louisiana', 23: 'Maine', 24: 'Maryland', 
          25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota', 
          28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska', 
          32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico', 
          36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 
          40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 43: 'Puerto Rico', 
          44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee', 
          48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia', 52: 'Virgin Islands', 
          53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'}

d['state']=d['STATE'].apply(lambda x: states[x])


# In[ ]:



# Incident by state
d['state'].value_counts().to_frame()



# In[ ]:


# You can check these values against
#   https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812318
d.groupby(['state']).agg({'FATALS':sum})


# In[ ]:


# Deadlist Week by State?
# Deadlist Month by State?
p = pd.pivot_table(d, values='FATALS', 
                   index=['crashTime'], columns=['state'], aggfunc=np.sum)

# Sample by W Week
#pp=p.resample('W', how=[np.sum]).reset_index()

# Sample by M Month
pp=p.resample('M', how=[np.sum]).reset_index()
pp.sort_values(by='crashTime',ascending=False,inplace=True)

# Let's flatten the columns 
pp.columns = pp.columns.get_level_values(0)

# Show values
# Note, last week might not be a full week
pp


# In[ ]:


# Pick a particular state
pp[['crashTime','Pennsylvania']].sort_values(by=['Pennsylvania'],
                                             ascending=False,inplace=False).head(15)


# Pennsylvania (all fatalities)
# -----------------------------
# 
# The following Google map shows all fatalities for Pennsylvania.  This is an example of pulling one Kaggle Kernel into another. The Kernel code for the Google map can be found [here](https://www.kaggle.com/mchirico/d/nhtsa/2015-traffic-fatalities/pennsylvania-fatalities)

# In[ ]:


import IPython
url = 'https://www.kaggle.io/svf/490446/e51617de25c0084fdc51496ce476d947/output.html'
iframe = '<iframe src=' + url + ' width=700 height=525></iframe>'
IPython.display.HTML(iframe)


# Weather
# -------
# 
# Reference [FARS User’s Manual.](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812315) 
# 
# Definition: The prevailing atmospheric conditions that existed at the
# time of the crash as indicated in the case material.
# 
#  - 0 No Additional Atmospheric Conditions
#  - 1 Clear
#  - 2 Rain
#  - 3 Sleet, Hail
#  - 4 Snow
#  - 5 Fog, Smog, Smoke
#  - 6 Severe Crosswinds
#  - 7  Blowing Sand, Soil, Dirt
#  - 8  Other
#  - 9  "Value Not Used"
#  - 10 Cloudy
#  - 11  Blowing Snow
#  - 12 Freezing Rain or Drizzle
#  -  98 Not Reported
#  -  99 Unknown

# In[ ]:


weather = {0: 'No Additional Atmospheric Conditions', 1: 'Clear', 
           2: 'Rain', 3: 'Sleet, Hail', 
           4: 'Snow', 5: 'Fog, Smog, Smoke', 6: 'Severe Crosswinds', 
           7: 'Blowing Sand, Soil, Dirt', 
           8: 'Other', 10: 'Cloudy', 11: 'Blowing Snow', 
           12: 'Freezing Rain or Drizzle', 
           98: 'Not Reported', 99: 'Unknown'}

d['weather']=d['WEATHER'].apply(lambda x: weather[x])
d['weather1']=d['WEATHER1'].apply(lambda x: weather[x])
d['weather2']=d['WEATHER2'].apply(lambda x: weather[x])


# In[ ]:


d[['WEATHER','WEATHER1','WEATHER2']].head()


# In[ ]:


d[['weather','weather1','weather2']].head()


# In[ ]:


# Interesting.  Clear weather is the worst
d['weather'].value_counts()


# ## Weather and Drunk Driving ##

# In[ ]:


drunk = d[d.DRUNK_DR == 1]
n_drunk = d[d.DRUNK_DR == 0]

# Careful, maintain order
drunk_dict = drunk['weather'].value_counts().to_dict()
not_drunk_dict = n_drunk['weather'].value_counts().to_dict()


fig = {
  "data": [
    {
      "values": list(drunk_dict.values()),
      "labels": list(drunk_dict.keys()),
      "domain": {"x": [0, .48]},
      "name": "Drunk",
      "hoverinfo":"label+percent+name",
      "hole": .48,
      "type": "pie"
    },     
    {
      "values": list(not_drunk_dict.values()),
      "labels": list(not_drunk_dict.keys()),
      "text":"Not Drunk",
      "textposition":"inside",
      "domain": {"x": [.52, 1]},
      "name": "Not Drunk",
      "hoverinfo":"label+percent+name",
      "hole": .48,
      "type": "pie"
    }],
  "layout": {
        "title":"Weather and Drunk Driving",
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "Drunk",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "  Not Drunk",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}
iplot(fig)


# The above pie graph seems to show roughly the same percentages on the type of weather for drunk vs not drunk fatalities.  However, it's still worth a closer look.  In particular, compare "Rain" between the two graphs.
# 

# VE_TOTAL  (number of contact motor vehicles)
# --------
# 
# Reference [FARS User’s Manual.](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812315) 
# 
# **Definition:** This data element is the number of contact motor vehicles that the officer reported
# on the PAR as a unit involved in the crash.
# 
# **Additional Information:** This number represents all of the vehicles in the crash. This includes
# the vehicles in-transport which are in the Vehicle data file and the vehicles not in-transport
# which are in the Parkwork data file (previously Vehnit). 

# In[ ]:


d['VE_TOTAL'].value_counts()


# ## Distraction  (distract.csv) ##

# In[ ]:


# Reading in the data
FILE = "../input/distract.csv"
dd = pd.read_csv(FILE, encoding = "ISO-8859-1")


# In[ ]:


dd.head()


# In[ ]:


distract = {0: 'Not Distracted', 1: 'Looked But Did Not See',
           3: 'By Other Occupant(s)', 4: 'By a Moving Object in Vehicle',
           5: 'While Talking or Listening to Cellular Phone',
           6: 'While Manipulating Cellular Phone',
           7: 'While Adjusting Audio or Climate Controls',
           9: 'While Using Other Component/Controls Integral to Vehicle',
           10: 'While Using or Reaching For Device/Object Brought Into Vehicle',
           12: 'Distracted by Outside Person, Object or Event',
           13: 'Eating or Drinking',
           14: 'Smoking Related',
           15: 'Other Cellular Phone Related',
           16: 'No Driver Present/Unknown if Driver Present',
           17: 'Distraction/Inattention',
           18: 'Distraction/Careless',
           19: 'Careless/Inattentive',
           92: 'Distraction (Distracted), Details Unknown',
           93: 'Inattention (Inattentive), Details Unknown',
           96: 'Not Reported',
           97: 'Lost In Thought/Day Dreaming',
           98: 'Other Distraction',
           99: 'Unknown if Distracted'}

dd['mdrdstrd'] = dd['MDRDSTRD'].apply(lambda x: distract[x])


# In[ ]:


dd['mdrdstrd'].value_counts()


# In[ ]:


# Careful, maintain order
table = dd['mdrdstrd'].value_counts().to_dict()



fig = {
  "data": [
    {
      "values": list(table.values()),
      "labels": list(table.keys()),
      "domain": {"x": [0, .58]},
      "name": "Distracted",
      "hoverinfo":"label+percent+name",
      "hole": .48,
      "type": "pie"
    }],
  "layout": {
        "title":"Distracted",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "",
                "x": 0.80,
                "y": 0.5
            }
        ]
    }
}
iplot(fig)


# Interesting on the above graph.  Cell phone distraction seems to be in the low percentages.  
# However, it is worth noting there are several categories for cell phone distractions.

# In[ ]:


# Let's combine cell phones
distract2 = {0: 'Not Distracted', 1: 'Looked But Did Not See',
           3: 'By Other Occupant(s)', 4: 'By a Moving Object in Vehicle',
           5: 'Cell Phone',
           6: 'Cell Phone',
           7: 'While Adjusting Audio or Climate Controls',
           9: 'While Using Other Component/Controls Integral to Vehicle',
           10: 'While Using or Reaching For Device/Object Brought Into Vehicle',
           12: 'Distracted by Outside Person, Object or Event',
           13: 'Eating or Drinking',
           14: 'Smoking Related',
           15: 'Cell Phone',
           16: 'No Driver Present/Unknown if Driver Present',
           17: 'Distraction/Inattention',
           18: 'Distraction/Careless',
           19: 'Careless/Inattentive',
           92: 'Distraction (Distracted), Details Unknown',
           93: 'Inattention (Inattentive), Details Unknown',
           96: 'Not Reported',
           97: 'Lost In Thought/Day Dreaming',
           98: 'Other Distraction',
           99: 'Unknown if Distracted'}


dd['mdrdstrd2'] = dd['MDRDSTRD'].apply(lambda x: distract2[x])


# In[ ]:


dd['mdrdstrd2'].value_counts()


# In[ ]:


# Careful, maintain order
table = dd['mdrdstrd2'].value_counts().to_dict()



fig = {
  "data": [
    {
      "values": list(table.values()),
      "labels": list(table.keys()),
      "domain": {"x": [0, .58]},
      "name": "Distracted",
      "hoverinfo":"label+percent+name",
      "hole": .48,
      "type": "pie"
    }],
  "layout": {
        "title":"Distracted - Cell Phone Groups Combined",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "",
                "x": 0.80,
                "y": 0.5
            }
        ]
    }
}
iplot(fig)


# In[ ]:


p = pd.pivot_table(dd, values='VEH_NO', 
                   index=['ST_CASE'], columns=['mdrdstrd2'], aggfunc=lambda x: len(x.unique()))


# In[ ]:


p.fillna(0, inplace=True)
# Let's flatten the columns 
p.columns = p.columns.get_level_values(0)

p.head()


# In[ ]:


# Interesting... so few "Cell Phone"
p['Cell Phone'].value_counts()


# In[ ]:


print("Total Crashes with at least one known Cell Phone distraction:",430+9+3)
print("Percent:", (430+9+3.0)/(31724+430+9+3)  * 100)


# Note the above might be worth some more research.

# **Checking our results...**

# In[ ]:


# Let's check this.  
# Here's the question we're asking: Is there a crash where distraction from 
# Cell Phone is listed for 3 vehicles involved?
p[p['Cell Phone']==3]


# In[ ]:


# Here we're going back to check our data 

dd[dd['ST_CASE'] == 470227]


# Distractions - VEH_NO 1
# -----------------------
# 
# It might be worth looking at other vehicles.

# In[ ]:


dd[dd['VEH_NO']==1]['mdrdstrd2'].value_counts()


# In[ ]:


table = dd[dd['VEH_NO']==1]['mdrdstrd2'].value_counts().to_dict()



fig = {
  "data": [
    {
      "values": list(table.values()),
      "labels": list(table.keys()),
      "domain": {"x": [0, .58]},
      "name": "Distracted",
      "hoverinfo":"label+percent+name",
      "hole": .48,
      "type": "pie"
    }],
  "layout": {
        "title":"Distracted VEH_NO 1 - Cell Phone Groups Combined",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "",
                "x": 0.80,
                "y": 0.5
            }
        ]
    }
}
iplot(fig)


# <br>

# ##Violations  (../input/extra/violatn.csv) ##
# 
# Definition: This data element identifies all violations charged to this driver.
# 
# Here are a few **Attribute Codes**:
# 
#  - 0 None
#  - 1 Manslaughter or Homicide
#  - 11 Driving While Intoxicated (Alcohol or Drugs) or BAC above Limit (Any Detectable BAC
# for CDLs)
#  - 46 Fail to Yield Generally
#  - 99 Unknown Violation(s)

# In[ ]:


# Reading in the data
FILE = "../input/extra/violatn.csv"
v = pd.read_csv(FILE, encoding = "ISO-8859-1")


# In[ ]:


v.head()


# In[ ]:


v['MVIOLATN'].value_counts().head(8)


# Sequence of Events  (../input/extra/vsoe.csv)
# ==================
# 
# This file tells the story of the crash -- the sequence of events that led to the fatality or fatalities. 
#  

# In[ ]:


# Reading in the data
FILE = "../input/extra/vsoe.csv"
vsoe = pd.read_csv(FILE, encoding = "ISO-8859-1")


# In[ ]:


vsoe.head()


# In[ ]:


soe = {1:"Rollover/Overturn",
2:"Fire/Explosion",
3:"Immersion or Partial Immersion",
4:"Gas Inhalation",
5:"Fell/Jumped from Vehicle",
6:"Injured in Vehicle (Non-Collision)",
7:"Other Non-Collision",
8:"Pedestrian",
9:"Pedalcyclist",
10:"Railway Vehicle",
11:"Live Animal",
12:"Motor Vehicle in Transport",
14:"Parked Motor Vehicle",
15:"Non-Motorist on Personal Conveyance",
16:"Thrown or Falling Object",
17:"Boulder",
18:"Other Object (Not Fixed)",
19:"Building",
20:"Impact Attenuator/Crash Cushion",
21:"Bridge Pier or Support",
23:"Bridge Rail (Includes Parapet)",
24:"Guardrail Face",
25:"Concrete Traffic Barrier",
26:"Other Traffic Barrier",
30:"Utility Pole/Light Support",
31:"Other Post",
32:"Culvert",
33:"Curb",
34:"Ditch",
35:"Embankment",
38:"Fence",
39:"Wall",
40:"Fire Hydrant",
41:"Shrubbery",
42:"Tree (Standing Only)",
43:"Other Fixed Object",
44:"Pavement Surface Irregularity (Ruts Potholes Grates etc.)",
45:"Working Motor Vehicle",
46:"Traffic Signal Support",
48:"Snow Bank",
49:"Ridden Animal or Animal-Drawn Conveyance",
50:"Bridge Overhead Structure",
51:"Jackknife (Harmful to This Vehicle)",
52:"Guardrail End",
53:"Mail Box",
54:"Motor Vehicle In-Transport Strikes or is Struck by Cargo Persons or Objects Set-in-Motion from/by Another Motor Vehicle In-Transport",
55:"Motor Vehicle in Motion Outside the Trafficway",
57:"Cable Barrier",
58:"Ground",
59:"Traffic Sign Support",
60:"Cargo/Equipment Loss or Shift (Non-Harmful)",
61:"Equipment Failure (Blown Tire",
62:"Separation of Units",
63:"Ran Off Road - Right",
64:"Ran Off Road - Left",
65:"Cross Median",
66:"Downhill Runaway",
67:"Vehicle Went Airborne",
68:"Cross Centerline",
69:"Re-Entering Highway",
70:"Jackknife (Non-Harmful)",
71:"End Departure",
72:"Cargo/Equipment Loss or Shift (Harmful To This Vehicle)",
73:"Object Fell From Motor Vehicle In-Transport",
79:"Ran Off Roadway - Direction Unknown",
99:"Unknown",}


# In[ ]:


vsoe['soe'] = vsoe['SOE'].apply(lambda x: soe[x])


# Note below how easy it is to read what happened:  One vehicle ran off to the road to the right, hit an embankment, went airborne and hit a tree.

# In[ ]:


vsoe[(vsoe['ST_CASE'] == 10001) & (vsoe['VEH_NO'] == 1)].sort_values(by='VEVENTNUM',ascending=True)


# In[ ]:


d[d['ST_CASE']==10001][['PERSONS','FATALS','DRUNK_DR','crashTime','state','weather']].head()


# In[ ]:


# Information on this person

# Reading in the data
FILE = "../input/person.csv"
person = pd.read_csv(FILE, encoding = "ISO-8859-1")
person[person['ST_CASE']==10001][['AGE','AIR_BAG','DRUGS','DRUG_DET',
                                  'DOA','RACE']]


#  - DOA: 7 Died at Scene
#  - DRUG_DET: 7 Test for Drugs, Results, Unknown
#  - RACE: 1, white
#  - AIR_BAG:  1 Deployed: Front
# 
#  

# 
# <br><br>
# 
# Vision Obstruction (vision.csv)
# ------------------
# 
# **MVISOBSC**
# 
# Attribute Codes
# 
#  - 00 No Obstruction Noted
#  - 01 Rain, Snow, Fog, Smoke, Sand, Dust
#  - 02 Reflected Glare, Bright Sunlight, Headlights
#  - ... (more values, but not listed)

# In[ ]:


# Reading in the data
FILE = "../input/vision.csv"
# watch name collusions use vision_df not vision
vision_df = pd.read_csv(FILE, encoding = "ISO-8859-1")


# In[ ]:


vis = {0:"No Obstruction Noted",
       1:"Rain, Snow, Fog, Smoke, Sand, Dust",
       2:"Reflected Glare, Bright Sunlight, Headlights",
       3:"Curve, Hill, or Other Roadway Design Features",
       4:"Building, Billboard, or Other Structure",
       5:"Trees, Crops, Vegetation",
       6:"In-Transport Motor Vehicle (Including Load)",
       7:"Not-in-Transport Motor Vehicle (Parked, Working)",
       8:"Splash or Spray of Passing Vehicle",
       9:"Inadequate Defrost or Defog System",
       10:"Inadequate Vehicle Lighting System",
       11:"Obstructing Interior to the Vehicle",
       12:"External Mirrors",
       13:"Broken or Improperly Cleaned Windshield",
       14:"Obstructing Angles on Vehicle",
       95:"No Driver Present/Unknown if Driver Present",
       97:"Vision Obscured – No Details",
       98:"Other Visual Obstruction",
       99:"Unknown"}


# In[ ]:


vision_df['mvisobsc'] = vision_df['MVISOBSC'].apply(lambda x: vis[x])


# In[ ]:


vision_df['mvisobsc'].value_counts()


# 
# <br><br>
# # Merge Two Tables

# In[ ]:


# 
FILE = "../input/vindecode.csv"
v = pd.read_csv(FILE, encoding = "ISO-8859-1")

FILE = "../input/maneuver.csv"
m = pd.read_csv(FILE, encoding = "ISO-8859-1")

dv = pd.merge(d, v, how='left',left_on='ST_CASE', right_on='ST_CASE')
dm = pd.merge(d, v, how='left',left_on='ST_CASE', right_on='ST_CASE')


# In[ ]:


v.head()


# In[ ]:


dv[['state','VINMAKE_T','VINMODEL_T']].head()


# maneuver.csv
# --------------------------
# 
# Reference [FARS User’s Manual](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812315) 
# 
# The Maneuver data file identifies each avoidance attempt (as a separate record). It contains the
# data elements ST_CASE, STATE, and VEH_NO, which are described in the beginning of the
# Data Element Definitions and Codes section. The data file also contains MDRMANAV which is
# described below.
# 
# ST_CASE, VEH_NO, and MDRMANAV are the unique identifiers for each record. ST_CASE
# and VEH_NO should be used to merge the Maneuver data file with the Vehicle data file.
# 
# 
# **MDRMANAV**
# 
# Definition: This data element identifies the thing(s) this driver attempted to avoid while the
# vehicle was on the road portion of the trafficway, just prior to the first harmful event for this
# vehicle.
# 
#  
#  **Attribute Codes**
# 
# 
#  - 00 Driver Did Not Maneuver To Avoid
#  - 01 Object
#  - 02 Poor Road Conditions (Puddle, Ice, Pothole, etc.)
#  - 03 Live Animal
#  - 04 Motor Vehicle
#  - 05 Pedestrian, Pedalcyclist or Other Non-Motorist
#  - 92 Phantom/Non-Contact Motor Vehicle
#  - 95 No Driver Present/Unknown if Driver Present
#  - 98 Not Reported
#  - 99 Unknown

# In[ ]:


m.head()


# In[ ]:


# Note, multiple Attribute Codes can be applied to a single ST_CASE
m['MDRMANAV'].value_counts().head()

