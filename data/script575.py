
# coding: utf-8

# Traffic Fatalities in 2015
# --------------------------
# 
# &nbsp;...by [Latitude/Longitude][1]<br>
# &nbsp;...by [Date][2]<br>
# &nbsp;...by [State][3]/per [Capita][3]<br>
# &nbsp;...by [State][4]/per [Capita][3] (Sober and Drunk Drivers)
# 
# [1]: https://www.kaggle.io/svf/486873/972c1b4beb9083fd42d59b0f3e45b302/__results__.html#Traffic-Fatalities-by-Latitude/Longitude
# [2]: https://www.kaggle.io/svf/486873/972c1b4beb9083fd42d59b0f3e45b302/__results__.html#Traffic-Fatalities-by-Date
# [3]: https://www.kaggle.io/svf/486873/972c1b4beb9083fd42d59b0f3e45b302/__results__.html#Traffic-Fatalities-per-Capita
# [4]: https://www.kaggle.io/svf/486873/972c1b4beb9083fd42d59b0f3e45b302/__results__.html#Traffic-Fatalities-per-Capita:-Sober-and-Drunk-Drivers

# Data Import
# -----------------

# In[ ]:


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

traffic_data = pd.read_csv('../input/accident.csv',
                           usecols=[0, 1, 11, 12, 13, 25, 26, 50, 51])
traffic_data = traffic_data.rename(
    columns={'ST_CASE':'case_id', 'LONGITUD':'longitude',
             'DRUNK_DR':'drunk_drivers', 'FATALS':'fatalities'})
traffic_data.columns = traffic_data.columns.str.lower()
traffic_data['date'] = pd.to_datetime(traffic_data[['day', 'month', 'year']])
traffic_data = traffic_data[['case_id', 'date', 'state', 'latitude', 'longitude',
                             'drunk_drivers', 'fatalities']].sort_values('date')


# Traffic Fatalities by Latitude/Longitude
# ----------------------------------------

# In[ ]:


traffic_data['text'] = traffic_data['date'].dt.strftime('%B %-d'
                          ) + ', ' + traffic_data['fatalities'].astype(str) + ' Killed'

data = [dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = traffic_data[traffic_data.longitude < 0]['longitude'],
        lat = traffic_data[traffic_data.longitude < 0]['latitude'],
        text = traffic_data[traffic_data.longitude < 0]['text'],
        mode = 'markers',
        marker = dict( 
            size = traffic_data[traffic_data.longitude < 0]['fatalities'] ** 0.5 * 5,
            opacity = 0.75,
            color = 'rgb(215, 0, 0)')
        )]

layout = dict(
         title = 'Traffic Fatalities by Latitude/Longitude in United States (2015)<br>'
                 '<sub>Hover to View Collision Date and Deaths</sub>',
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


# Traffic Fatalities by Date
# ---------------------------

# In[ ]:


# traffic fatalities by date
traffic_perdate = np.asarray(traffic_data.groupby('date')['fatalities'].sum())

# thirty day moving average of traffic fatalites by date
traffic_average = pd.Series(traffic_perdate).rolling(window=30).mean()
traffic_average = np.asarray(traffic_average.drop(traffic_average.index[:29]))
traffic_average = np.round(traffic_average, 0)

traffic_dates = np.arange('2015-01', '2016-01', dtype='datetime64[D]')
traffic_range = traffic_dates[15:351]

trace_date = go.Scatter(
             x = traffic_dates,
             y = traffic_perdate,
             mode = 'lines',
             name = 'Fatalities',
             line = dict(
                 color = 'rgb(215, 0, 0)',
                 width = 3)
             )

trace_mean = go.Scatter(
             x = traffic_range,
             y = traffic_average,
             mode = 'lines',
             name = 'Average',
             line = dict(
                 color = 'rgb(215, 0, 0)',
                 width = 5),
             opacity = 0.33
             )

layout = go.Layout(
         title = 'Traffic Fatalities by Date in United States (2015)',
         showlegend = False,
         xaxis = dict(
             rangeslider = dict(thickness = 0.05),
             type = 'date',
             showline = True,
             showgrid = False
         ),
         yaxis = dict(
             range = [40.1, 165],
             autotick = False,
             tick0 = 20,
             dtick = 20,
             showline = True,
             showgrid = False)
         )

data = [trace_date, trace_mean]
figure = dict(data = data, layout = layout)
iplot(figure)


# Traffic Fatalities per Capita
# -----------------------------
# Which states reported the most traffic fatalities given their population?

# In[ ]:


us_states = np.asarray(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',
                        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA',
                        'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY',
                        'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                        'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'])

# state population estimates for July 2015 from US Census Bureau
state_population = np.asarray([4858979, 738432, 6828065, 2978204, 39144818, 5456574,
                               3590886, 945934, 646449, 20271272, 10214860, 1431603,
                               1654930, 12859995, 6619680, 3123899, 2911641, 4425092,
                               4670724, 1329328, 6006401, 6794422, 9922576, 5489594,
                               2992333, 6083672, 1032949, 1896190,2890845, 1330608,
                               8958013, 2085109, 19795791, 10042802, 756927, 11613423,
                               3911338, 4028977, 12802503, 1056298, 4896146, 858469,
                               6600299, 27469114, 2995919, 626042, 8382993, 7170351,
                               1844128, 5771337, 586107])

# traffic fatalities per 100,000 people in state
traffic_perstate = np.asarray(traffic_data.groupby('state')['fatalities'].sum())
traffic_percapita = np.round(traffic_perstate / state_population * 100000, 2)

traffic_scale = [[0, 'rgb(248, 213, 213)'], [1, 'rgb(215, 0, 0)']]

data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = traffic_scale,
        showscale = False,
        locations = us_states,
        locationmode = 'USA-states',
        z = traffic_percapita,
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            )
        )]

layout = dict(
         title = 'Traffic Fatalities per 100,000 People in United States (2015)',
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             countrycolor = 'rgb(255, 255, 255)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
         )

figure = dict(data = data, layout = layout)
iplot(figure)


# States in the Northwest and Southeast reported the most traffic fatalities per person. Wyoming had the highest fatality rate in the nation -- 1 in 4,000 residents was killed in a traffic collision in 2015!

# Traffic Fatalities per Capita: Sober and Drunk Drivers
# -----------------------------------
# Which states reported the highest proportion of traffic fatalities in collisions involving intoxicated drivers?

# In[ ]:


# traffic fatalities from sober drivers per 100,000 people in state
sober_perstate = np.asarray(traffic_data[traffic_data.drunk_drivers == 0].groupby(
                                                            'state')['fatalities'].sum())
sober_percapita = np.round(sober_perstate / state_population * 100000, 2)

# traffic fatalities from drunk drivers per 100,000 people in state
drunk_perstate = np.asarray(traffic_data[traffic_data.drunk_drivers > 0].groupby(
                                                            'state')['fatalities'].sum())
drunk_percapita = np.round(drunk_perstate / state_population * 100000, 2)

trace_dot = go.Scatter(
            x = sober_percapita,
            y = drunk_percapita,
            text = us_states,
            mode = 'markers+text',
            textposition = 'bottom',
            hoverinfo = 'x+y+text',
            marker = dict(
                color = 'rgb(215, 0, 0)',
                size = 8)
            )

trace_dash = go.Scatter(
             x = [0.1, 10.5],
             y = [0.1, 10.5],
             mode = 'lines',
             hoverinfo = 'none',
             line = dict(
                 color = 'rgb(68, 68, 68)',
                 width = 1.5,
                 dash = 'dot')
             )

layout = go.Layout(
         title = 'Traffic Fatalities per Capita by Driver Intoxication '
                 'in United States (2015)',
         showlegend = False,
         xaxis = dict(
             title = 'Fatalities per 100,000 People (Sober Drivers)',
             range = [0.1, 18.5],
             autotick = False,
             tick0 = 2,
             dtick = 2,
             showline = True,
             showgrid = False
         ),
         yaxis = dict(
             title = 'Fatalities per 100,000 People (Drunk Drivers)',
             range = [0.1, 10],
             autotick = False,
             tick0 = 2,
             dtick = 2,
             showline = True,
             showgrid = False)
         )

data = [trace_dot, trace_dash]
figure = dict(data = data, layout = layout)
iplot(figure)


# Most states reported twice as many traffic fatalities from collisions with no driver intoxication as from those involving one or more drunk drivers. Wyoming had the highest fatality rate in alcohol-related collisions, followed by Montana and North Dakota.
