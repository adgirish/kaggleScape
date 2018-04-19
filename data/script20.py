
# coding: utf-8

# UFO Reports in United States
# ----------------------------
#  &nbsp;...by [Latitude/Longitude][1]<br>
#  &nbsp;...by [Year][2]<br>
#  &nbsp;...by [State][3]<br>
#  &nbsp;...per [Capita][4]<br>
#  &nbsp;...by [Area][5]
# 
# [1]: https://www.kaggle.io/svf/467568/f1648392162180ebab13a806a8d8b328/__results__.html#UFO-Reports-by-Latitude/Longitude
# [2]: https://www.kaggle.io/svf/467568/f1648392162180ebab13a806a8d8b328/__results__.html#UFO-Reports-by-Year
# [3]: https://www.kaggle.io/svf/467568/f1648392162180ebab13a806a8d8b328/__results__.html#UFO-Reports-by-State
# [4]: https://www.kaggle.io/svf/467568/f1648392162180ebab13a806a8d8b328/__results__.html#UFO-Reports-per-Capita
# [5]: https://www.kaggle.io/svf/467568/f1648392162180ebab13a806a8d8b328/__results__.html#UFO-Reports-per-Area

# Data Import
# -----------------

# In[ ]:


import numpy as np
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

ufo_data = pd.read_csv('../input/scrubbed.csv', usecols=[0, 1, 2, 9, 10], low_memory=False)
ufo_data['datetime'] = pd.to_datetime(ufo_data['datetime'], errors='coerce')
ufo_data.insert(1, 'year', ufo_data['datetime'].dt.year)
ufo_data['year'] = ufo_data['year'].fillna(0).astype(int)
ufo_data['city'] = ufo_data['city'].str.title()
ufo_data['state'] = ufo_data['state'].str.upper()
ufo_data['latitude'] = pd.to_numeric(ufo_data['latitude'], errors='coerce')
ufo_data = ufo_data.rename(columns={'longitude ':'longitude'})

us_states = np.asarray(['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                        'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                        'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                        'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                        'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'])

# UFO sightings in United States only (70,805 rows)
ufo_data = ufo_data[ufo_data['state'].isin(us_states)].sort_values('year')
ufo_data = ufo_data[(ufo_data.latitude > 15) & (ufo_data.longitude < -65)]
ufo_data = ufo_data[(ufo_data.latitude > 50) & (ufo_data.longitude > -125) == False]
ufo_data = ufo_data[ufo_data['city'].str.contains('\(Canada\)|\(Mexico\)') == False]


# UFO Reports by Latitude/Longitude
# ----------------------------

# In[ ]:


ufo_data['text'] = ufo_data[ufo_data.year > 0]['datetime'].dt.strftime('%B %-d, %Y')

data = [dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = ufo_data[ufo_data.year > 0]['longitude'],
        lat = ufo_data[ufo_data.year > 0]['latitude'],
        text = ufo_data[ufo_data.year > 0]['text'],
        mode = 'markers',
        marker = dict(
            size = 5.5,
            opacity = 0.75,
            color = 'rgb(0, 163, 81)',
            line = dict(color = 'rgb(255, 255, 255)', width = 1))
        )]

layout = dict(
         title = 'UFO Reports by Latitude/Longitude in United States (1910-2014)',
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             showland = True,
             landcolor = 'rgb(250, 250, 250)',
             subunitwidth = 1,
             subunitcolor = 'rgb(217, 217, 217)',
             countrywidth = 1,
             countrycolor = 'rgb(217, 217, 217)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
        )

figure = dict(data = data, layout = layout)
iplot(figure)


# UFO Reports by Year
# -------------------
# Have reports of UFO sightings increased, decreased, or remained constant in the past century?

# In[ ]:


# UFO sightings per year
ufo_peryear = np.asarray(ufo_data[ufo_data.year > 0].groupby('year').year.count())
# UFO sightings in 2014 estimated, data published in June 2014
ufo_peryear[-1] = ufo_peryear[-1] * 3

ufo_years = np.asarray(ufo_data[ufo_data.year > 0].year.unique())

trace = [go.Scatter(
         x = ufo_years,
         y = ufo_peryear,
         mode = 'lines',
         line = dict(
             color = 'rgb(0, 163, 81)',
             width = 3)
         )]

layout = go.Layout(
         title = 'UFO Reports by Year in United States (1910-2014)',
         xaxis = dict(
             rangeslider = dict(thickness = 0.05),
             showline = True,
             showgrid = False
         ),
         yaxis = dict(
             range = [0, 7000],
             showline = True,
             showgrid = False)
         )

figure = dict(data = trace, layout = layout)
iplot(figure)


# Reports of UFO sightings have skyrocketed in the past twenty years — rising from less than 1,000 in 1995 to more than 6,000 in 2012, 2013, and 2014 (estimated). Have more UFOs been sighted or have more UFO sightings been reported?
# 
# The NUFORC has collected UFO data from a phone hotline since 1974 and an online form since 1998. Could the availability of an online form be related to the dramatic increase in reported UFO sightings?

# UFO Reports by State
# -----------------------
# Which states have reported the most UFO sightings in the past century?

# In[ ]:


# UFO sightings per state
ufo_perstate = np.asarray(ufo_data.groupby('state').state.count())

ufo_scale = [[0, 'rgb(229, 249, 239)'], [1, 'rgb(0, 163, 81)']]

data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = ufo_scale,
        showscale = False,
        locations = us_states,
        locationmode = 'USA-states',
        z = ufo_perstate,
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            )
        )]

layout = dict(
         title = 'UFO Reports by State in United States (1910-2014)',
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             countrycolor = 'rgb(255, 255, 255)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
        )

figure = dict(data = data, layout = layout)
iplot(figure)


# The states with the highest populations — California, Texas, and Florida — reported the most UFO sightings.

# UFO Reports per Capita
# ----------------------
# Which states have reported the most UFO sightings given their population?

# In[ ]:


# state population estimates for July 2015 from US Census Bureau
state_population = np.asarray([738432, 4858979, 2978204, 6828065, 39144818, 5456574,
                               3590886, 672228, 945934, 20271272, 10214860, 1431603,
                               3123899, 1654930, 12859995, 6619680, 2911641, 4425092,
                               4670724, 6794422, 6006401, 1329328, 9922576, 5489594,
                               6083672, 2992333, 1032949, 10042802, 756927, 1896190,
                               1330608, 8958013, 2085109, 2890845, 19795791, 11613423,
                               3911338, 4028977, 12802503, 1056298, 4896146, 858469,
                               6600299, 27469114, 2995919, 8382993, 626042, 7170351,
                               5771337, 1844128, 586107])

# UFO sightings per 100,000 people in state
ufo_percapita = np.round(ufo_perstate / state_population * 100000, 2)

data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = ufo_scale,
        showscale = False,
        locations = us_states,
        locationmode = 'USA-states',
        z = ufo_percapita,
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            )
        )]

layout = dict(
         title = 'UFO Reports per 100,000 People in United States (1910-2014)',
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             countrycolor = 'rgb(255, 255, 255)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
        )

figure = dict(data = data, layout = layout)
iplot(figure)


# Why have states in the Northeast and West on the border with Canada reported the most UFO sightings per person?

# UFO Reports by Area
# --------------------
# Which states have reported the most UFO sightings given their size?

# In[ ]:


# state land areas in square kilometeres from US Census Bureau
state_area = np.asarray([1477953, 131171, 134771, 294207, 403466, 268431, 12542, 158,
                         5047, 138887, 148959, 16635, 144669, 214045, 143793, 92789, 
                         211754, 102269, 111898, 20202, 25142, 79883, 146435, 206232,
                         178040, 121531, 376962, 125920, 178711, 198974, 23187, 19047,
                         314161, 284332, 122057, 105829, 177660, 248608, 115883, 2678,
                         77857, 196350, 106798, 676587, 212818, 102279, 23871, 172119,
                         140268, 62259, 251470])

# UFO sightings per 1,000 square kilometers in state
ufo_perarea = np.round(ufo_perstate / state_area * 1000, 2)
# District of Columbia outlier (1 UFO sighting per square kilometer) adjusted
ufo_perarea[7] = round(ufo_perarea[7] / 6, 2)

data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = ufo_scale,
        showscale = False,
        locations = us_states,
        locationmode = 'USA-states',
        z = ufo_perarea,
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            )
        )]

layout = dict(
         title = 'UFO Reports per 1,000 Square Kilometers in United States (1910-2014)',
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             countrycolor = 'rgb(255, 255, 255)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
        )

figure = dict(data = data, layout = layout)
iplot(figure)


# The states with the highest population densities — New Jersey, Massachusetts, Connecticut, and Rhode Island — reported the most UFO sightings per square kilometer.
