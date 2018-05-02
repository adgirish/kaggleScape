
# coding: utf-8

# Data Import
# ------------------

# In[ ]:


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

terror_data = pd.read_csv('../input/globalterrorismdb_0616dist.csv', encoding='ISO-8859-1',
                          usecols=[0, 1, 2, 3, 8, 11, 13, 14, 35, 84, 100, 103])
terror_data = terror_data.rename(
    columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',
             'country_txt':'country', 'provstate':'state', 'targtype1_txt':'target',
             'weaptype1_txt':'weapon', 'nkill':'fatalities', 'nwound':'injuries'})
terror_data['fatalities'] = terror_data['fatalities'].fillna(0).astype(int)
terror_data['injuries'] = terror_data['injuries'].fillna(0).astype(int)

# terrorist attacks in United States only (2,198 rows)
terror_usa = terror_data[(terror_data.country == 'United States') &
                         (terror_data.state != 'Puerto Rico') &
                         (terror_data.longitude < 0)]
terror_usa['day'][terror_usa.day == 0] = 1
terror_usa['date'] = pd.to_datetime(terror_usa[['day', 'month', 'year']])
terror_usa = terror_usa[['id', 'date', 'year', 'state', 'latitude', 'longitude',
                         'target', 'weapon', 'fatalities', 'injuries']]
terror_usa = terror_usa.sort_values(['fatalities', 'injuries'], ascending = False)
terror_usa = terror_usa.drop_duplicates(['date', 'latitude', 'longitude', 'fatalities'])


# Terrorist Attacks by Latitude/Longitude
# ---------------------------------------

# In[ ]:


terror_usa['text'] = terror_usa['date'].dt.strftime('%B %-d, %Y') + '<br>' +                     terror_usa['fatalities'].astype(str) + ' Killed, ' +                     terror_usa['injuries'].astype(str) + ' Injured'

fatality = dict(
           type = 'scattergeo',
           locationmode = 'USA-states',
           lon = terror_usa[terror_usa.fatalities > 0]['longitude'],
           lat = terror_usa[terror_usa.fatalities > 0]['latitude'],
           text = terror_usa[terror_usa.fatalities > 0]['text'],
           mode = 'markers',
           name = 'Fatalities',
           hoverinfo = 'text+name',
           marker = dict(
               size = terror_usa[terror_usa.fatalities > 0]['fatalities'] ** 0.255 * 8,
               opacity = 0.95,
               color = 'rgb(240, 140, 45)')
           )
        
injury = dict(
         type = 'scattergeo',
         locationmode = 'USA-states',
         lon = terror_usa[terror_usa.fatalities == 0]['longitude'],
         lat = terror_usa[terror_usa.fatalities == 0]['latitude'],
         text = terror_usa[terror_usa.fatalities == 0]['text'],
         mode = 'markers',
         name = 'Injuries',
         hoverinfo = 'text+name',
         marker = dict(
             size = (terror_usa[terror_usa.fatalities == 0]['injuries'] + 1) ** 0.245 * 8,
             opacity = 0.85,
             color = 'rgb(20, 150, 187)')
         )

layout = dict(
         title = 'Terrorist Attacks by Latitude/Longitude in United States (1970-2015)',
         showlegend = True,
         legend = dict(
             x = 0.85, y = 0.4
         ),
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

data = [fatality, injury]
figure = dict(data = data, layout = layout)
iplot(figure)


# Terrorist Attacks by Year
# -------------------------

# In[ ]:


# terrorist attacks by year
terror_peryear = np.asarray(terror_usa.groupby('year').year.count())

terror_years = np.arange(1970, 2016)
# terrorist attacks in 1993 missing from database
terror_years = np.delete(terror_years, [23])

trace = [go.Scatter(
         x = terror_years,
         y = terror_peryear,
         mode = 'lines',
         line = dict(
             color = 'rgb(240, 140, 45)',
             width = 3)
         )]

layout = go.Layout(
         title = 'Terrorist Attacks by Year in United States (1970-2015)',
         xaxis = dict(
             rangeslider = dict(thickness = 0.05),
             showline = True,
             showgrid = False
         ),
         yaxis = dict(
             range = [0.1, 425],
             showline = True,
             showgrid = False)
         )

figure = dict(data = trace, layout = layout)
iplot(figure)


# Terrorist Attacks per Capita
# ----------------------------

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
                               2992333, 6083672, 1032949, 1896190, 2890845, 1330608,
                               8958013, 2085109, 19795791, 10042802, 756927, 11613423,
                               3911338, 4028977, 12802503, 1056298, 4896146, 858469,
                               6600299, 27469114, 2995919, 626042, 8382993, 7170351,
                               1844128, 5771337, 586107])

# terrorist attacks per 100,000 people in state
terror_perstate = np.asarray(terror_usa.groupby('state').state.count())
terror_percapita = np.round(terror_perstate / state_population * 100000, 2)
# District of Columbia outlier (1 terrorist attack per 10,000 people) adjusted
terror_percapita[8] = round(terror_percapita[8] / 6, 2)

terror_scale = [[0, 'rgb(252, 232, 213)'], [1, 'rgb(240, 140, 45)']]

data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = terror_scale,
        showscale = False,
        locations = us_states,
        locationmode = 'USA-states',
        z = terror_percapita,
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            )
        )]

layout = dict(
         title = 'Terrorist Attacks per 100,000 People in United States (1970-2015)',
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             countrycolor = 'rgb(255, 255, 255)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
         )

figure = dict(data = data, layout = layout)
iplot(figure)


# Terrorist Attacks by Target
# ---------------------------

# In[ ]:


# terrorist attack targets grouped in categories
target_codes = []

for attack in terror_usa['target'].values:
    if attack in ['Business', 'Journalists & Media', 'NGO']:
        target_codes.append(1)
    elif attack in ['Government (General)', 'Government (Diplomatic)']:
        target_codes.append(2)
    elif attack == 'Abortion Related':
        target_codes.append(4)
    elif attack == 'Educational Institution':
        target_codes.append(5)
    elif attack == 'Police':
        target_codes.append(6)
    elif attack == 'Military':
        target_codes.append(7)
    elif attack == 'Religious Figures/Institutions':
        target_codes.append(8)
    elif attack in ['Airports & Aircraft', 'Maritime', 'Transportation']:
        target_codes.append(9)
    elif attack in ['Food or Water Supply', 'Telecommunication', 'Utilities']:
        target_codes.append(10)
    else:
        target_codes.append(3)

terror_usa['target'] = target_codes
target_categories = ['Business', 'Government', 'Individuals', 'Healthcare', 'Education',
                     'Police', 'Military', 'Religion', 'Transportation', 'Infrastructure']

# terrorist attacks by target
target_count = np.asarray(terror_usa.groupby('target').target.count())
target_percent = np.round(target_count / sum(target_count) * 100, 2)

# terrorist attack fatalities by target
target_fatality = np.asarray(terror_usa.groupby('target')['fatalities'].sum())
target_yaxis = np.asarray([1.33, 2.36, 2.98, 0.81, 1.25, 1.71, 1.31, 1.53, 1.34, 0])

# terrorist attack injuries by target
target_injury = np.asarray(terror_usa.groupby('target')['injuries'].sum())
target_xaxis = np.log10(target_injury)

target_text = []
for i in range(0, 10):
    target_text.append(target_categories[i] + ' (' + target_percent[i].astype(str) 
                       + '%)<br>' + target_fatality[i].astype(str) + ' Killed, '
                       + target_injury[i].astype(str) + ' Injured')

data = [go.Scatter(
        x = target_injury,
        y = target_fatality,
        text = target_text,
        mode = 'markers',
        hoverinfo = 'text',
        marker = dict(
            size = target_count / 6.5,
            opacity = 0.9,
            color = 'rgb(240, 140, 45)')
        )]

layout = go.Layout(
         title = 'Terrorist Attacks by Target in United States (1970-2015)',
         xaxis = dict(
             title = 'Injuries',
             type = 'log',
             range = [1.36, 3.25],
             tickmode = 'auto',
             nticks = 2,
             showline = True,
             showgrid = False
         ),
         yaxis = dict(
             title = 'Fatalities',
             type = 'log',
             range = [0.59, 3.45],
             tickmode = 'auto',
             nticks = 4,
             showline = True,
             showgrid = False)
         )

annotations = []
for i in range(0, 10):
    annotations.append(dict(x=target_xaxis[i], y=target_yaxis[i],
                            xanchor='middle', yanchor='top',
                            text=target_categories[i], showarrow=False))
layout['annotations'] = annotations

figure = dict(data = data, layout = layout)
iplot(figure)


# Terrorist Attacks by Weapon
# ---------------------------

# In[ ]:


# terrorist attack weapons grouped in categories
weapon_codes = []

for attack in terror_usa['weapon'].values:
    if attack in ['Explosives/Bombs/Dynamite', 'Sabotage Equipment']:
        weapon_codes.append(1)
    elif attack == 'Incendiary':
        weapon_codes.append(2)
    elif attack in ['Firearms', 'Fake Weapons']:
        weapon_codes.append(3)
    elif attack == 'Melee':
        weapon_codes.append(5)
    elif attack == 'Biological':
        weapon_codes.append(6)
    elif attack in ['Chemical', 'Radiological']:
        weapon_codes.append(7)
    elif 'Vehicle' in attack:
        weapon_codes.append(8)
    else:
        weapon_codes.append(4)

terror_usa['weapon'] = weapon_codes
weapon_categories = ['Explosives', 'Flammables', 'Firearms', 'Miscellaneous',
                     'Knives', 'Bacteria/Viruses', 'Chemicals', 'Vehicles']

# terrorist attacks by weapon
weapon_count = np.asarray(terror_usa.groupby('weapon').weapon.count())
weapon_percent = np.round(weapon_count / sum(weapon_count) * 100, 2)

# terrorist attack fatalities by weapon
weapon_fatality = np.asarray(terror_usa.groupby('weapon')['fatalities'].sum())
weapon_yaxis = np.asarray([1.93, 1.02, 2.28, 0.875, 0.945, 0.83, 0.835, 3.2])

# terrorist attack injuries by weapon
weapon_injury = np.asarray(terror_usa.groupby('weapon')['injuries'].sum())
weapon_xaxis = np.log10(weapon_injury)

weapon_text = []
for i in range(0, 8):
    weapon_text.append(weapon_categories[i] + ' (' + weapon_percent[i].astype(str) 
                       + '%)<br>' + weapon_fatality[i].astype(str) + ' Killed, '
                       + weapon_injury[i].astype(str) + ' Injured')

weapon_fatality[6] = 7
    
data = [go.Scatter(
        x = weapon_injury,
        y = weapon_fatality,
        text = weapon_text,
        mode = 'markers',
        hoverinfo = 'text',
        marker = dict(
            size = (weapon_count + 50) / 10,
            opacity = 0.9,
            color = 'rgb(240, 140, 45)')
        )]

layout = go.Layout(
         title = 'Terrorist Attacks by Weapon in United States (1970-2015)',
         xaxis = dict(
             title = 'Injuries',
             type = 'log',
             range = [0.45, 3.51],
             tickmode = 'auto',
             nticks = 4,
             showline = True,
             showgrid = False
         ),
         yaxis = dict(
             title = 'Fatalities',
             type = 'log',
             range = [0.65, 3.33],
             tickmode = 'auto',
             nticks = 3,
             showline = True,
             showgrid = False)
         )

annotations = []
for i in range(0, 8):
    annotations.append(dict(x=weapon_xaxis[i], y=weapon_yaxis[i],
                            xanchor='middle', yanchor='top',
                            text=weapon_categories[i], showarrow=False))
layout['annotations'] = annotations

figure = dict(data = data, layout = layout)
iplot(figure)

