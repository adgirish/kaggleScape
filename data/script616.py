
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ### Is The Temperature Really Rising??

# In[ ]:


global1=pd.read_csv('../input/GlobalTemperatures.csv')
global1=global1[['dt','LandAverageTemperature']]
global1.dropna(inplace=True)
global1['dt']=pd.to_datetime(global1.dt).dt.strftime('%d/%m/%Y')
global1['dt']=global1['dt'].apply(lambda x:x[6:])
global1=global1.groupby(['dt'])['LandAverageTemperature'].mean().reset_index()
trace=go.Scatter(
    x=global1['dt'],
    y=global1['LandAverageTemperature'],
    mode='lines',
    )
data=[trace]

py.iplot(data, filename='line-mode')


# **Hover over the graph for interactivity **
# 
# So the answer is clearly visible---**THE MERCURY IS RISING**.  Matter of Concern!!!!
# 
# One thing to note is that is data for the years in the 1750's seems to be dirty as there is a huge drop in the temperature.

# ### Let's Compare for any 2 months

# In[ ]:


global2=pd.read_csv('../input/GlobalTemperatures.csv')
global2=global2[['dt','LandAverageTemperature']]
global2.dropna(inplace=True)
global2['dt']=pd.to_datetime(global2.dt).dt.strftime('%d/%m/%Y')
global2['month']=global2['dt'].apply(lambda x:x[3:5])
global2['year']=global2['dt'].apply(lambda x:x[6:])
global2=global2[['month','year','LandAverageTemperature']]
global2['month']=global2['month'].map({'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'})
def plot_month(month1,month2):
    a=global2[global2['month']==month1]
    b=global2[global2['month']==month2]
    trace0 = go.Scatter(
    x = a['year'],
    y = a['LandAverageTemperature'],
    mode = 'lines',
    name = month1
    )
    
    trace1 = go.Scatter(
    x = b['year'],
    y = b['LandAverageTemperature'],
    mode = 'lines',
    name = month2
    )
    data = [trace0,trace1]
    py.iplot(data, filename='line-mode')
plot_month('Aug','Nov')


# We see a similar trend for the months also. There is a continous increase in the temperatures in individual months too. We can check for any two months by just replacing the month names in the function.

# ### Average Temperature By Country (Interactive Map)

# In[ ]:


temp_country=pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')


# In[ ]:


temp_country['Country'].replace({'Denmark (Europe)':'Denmark','France (Europe)':'France','Netherlands (Europe)':'Netherlands','United Kingdom (Europe)':'Europe'},inplace=True)
temp_country.fillna(0,inplace=True)


# In[ ]:


temp_country1=temp_country.groupby(['Country'])['AverageTemperature'].mean().reset_index()


# In[ ]:


l1=list(['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra','Angola', 'Anguilla', 'Antigua and Barbuda', 'Argentina', 'Armenia','Aruba', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas, The',
       'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize','Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina','Botswana', 'Brazil', 'British Virgin Islands', 'Brunei',
       'Bulgaria', 'Burkina Faso', 'Burma', 'Burundi', 'Cabo Verde','Cambodia', 'Cameroon', 'Canada', 'Cayman Islands','Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',
       'Comoros', 'Congo, Democratic Republic of the','Congo, Republic of the', 'Cook Islands', 'Costa Rica',"Cote d'Ivoire", 'Croatia', 'Cuba', 'Curacao', 'Cyprus','Czech Republic', 'Denmark', 'Djibouti', 'Dominica','Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
       'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia','Falkland Islands (Islas Malvinas)', 'Faroe Islands', 'Fiji','Finland', 'France', 'French Polynesia', 'Gabon', 'Gambia, The',
       'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland','Grenada', 'Guam', 'Guatemala', 'Guernsey', 'Guinea-Bissau','Guinea', 'Guyana', 'Haiti', 'Honduras', 'Hong Kong', 'Hungary',
       'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland','Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey','Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Korea, North',
       'Korea, South', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia','Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macau', 'Macedonia', 'Madagascar','Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta',
       'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico','Micronesia, Federated States of', 'Moldova', 'Monaco', 'Mongolia',
       'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal','Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua','Nigeria', 'Niger', 'Niue', 'Northern Mariana Islands', 'Norway',
       'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea','Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal','Puerto Rico', 'Qatar', 'Romania', 'Russia', 'Rwanda','Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin','Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines','Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia','Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore',
       'Sint Maarten', 'Slovakia', 'Slovenia', 'Solomon Islands','Somalia', 'South Africa', 'South Sudan', 'Spain', 'Sri Lanka','Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syria',
       'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey','Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',
       'United Arab Emirates', 'United Kingdom', 'United States','Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam',
       'Virgin Islands', 'West Bank', 'Yemen', 'Zambia', 'Zimbabwe']) #Country names

l2=list(['AFG', 'ALB', 'DZA', 'ASM', 'AND', 'AGO', 'AIA', 'ATG', 'ARG','ARM', 'ABW', 'AUS', 'AUT', 'AZE', 'BHM', 'BHR', 'BGD', 'BRB','BLR', 'BEL', 'BLZ', 'BEN', 'BMU', 'BTN', 'BOL', 'BIH', 'BWA','BRA', 'VGB', 'BRN', 'BGR', 'BFA', 'MMR', 'BDI', 'CPV', 'KHM',
       'CMR', 'CAN', 'CYM', 'CAF', 'TCD', 'CHL', 'CHN', 'COL', 'COM','COD', 'COG', 'COK', 'CRI', 'CIV', 'HRV', 'CUB', 'CUW', 'CYP','CZE', 'DNK', 'DJI', 'DMA', 'DOM', 'ECU', 'EGY', 'SLV', 'GNQ',
       'ERI', 'EST', 'ETH', 'FLK', 'FRO', 'FJI', 'FIN', 'FRA', 'PYF','GAB', 'GMB', 'GEO', 'DEU', 'GHA', 'GIB', 'GRC', 'GRL', 'GRD','GUM', 'GTM', 'GGY', 'GNB', 'GIN', 'GUY', 'HTI', 'HND', 'HKG',
       'HUN', 'ISL', 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', 'IMN', 'ISR','ITA', 'JAM', 'JPN', 'JEY', 'JOR', 'KAZ', 'KEN', 'KIR', 'KOR',
       'PRK', 'KSV', 'KWT', 'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR','LBY', 'LIE', 'LTU', 'LUX', 'MAC', 'MKD', 'MDG', 'MWI', 'MYS','MDV', 'MLI', 'MLT', 'MHL', 'MRT', 'MUS', 'MEX', 'FSM', 'MDA',
       'MCO', 'MNG', 'MNE', 'MAR', 'MOZ', 'NAM', 'NPL', 'NLD', 'NCL','NZL', 'NIC', 'NGA', 'NER', 'NIU', 'MNP', 'NOR', 'OMN', 'PAK','PLW', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'POL', 'PRT', 'PRI',
       'QAT', 'ROU', 'RUS', 'RWA', 'KNA', 'LCA', 'MAF', 'SPM', 'VCT','WSM', 'SMR', 'STP', 'SAU', 'SEN', 'SRB', 'SYC', 'SLE', 'SGP',
       'SXM', 'SVK', 'SVN', 'SLB', 'SOM', 'ZAF', 'SSD', 'ESP', 'LKA','SDN', 'SUR', 'SWZ', 'SWE', 'CHE', 'SYR', 'TWN', 'TJK', 'TZA',
       'THA', 'TLS', 'TGO', 'TON', 'TTO', 'TUN', 'TUR', 'TKM', 'TUV','UGA', 'UKR', 'ARE', 'GBR', 'USA', 'URY', 'UZB', 'VUT', 'VEN',
       'VNM', 'VGB', 'WBG', 'YEM', 'ZMB', 'ZWE']) #Country Codes

df=pd.DataFrame(l1,l2)
df.reset_index(inplace=True)
df.columns=[['Code','Country']]
temp_country1=temp_country1.merge(df,left_on='Country',right_on='Country',how='left')
temp_country1.dropna(inplace=True)


# In[ ]:


data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'RdYlGn',
        reversescale = True,
        showscale = True,
        locations = temp_country1['Code'],
        z = temp_country1['AverageTemperature'],
        locationmode = 'Code',
        text = temp_country1['Country'].unique(),
        marker = dict(
            line = dict(color = 'rgb(200,200,200)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Temperature')
            )
       ]

layout = dict(
    title = 'Average Temperature By Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(0,255,255)',
        projection = dict(
        type = 'Mercator',
            
        ),
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap2010')


# ### Top 10 Hottest And Coldest Countries

# In[ ]:


hot=temp_country1.sort_values(by='AverageTemperature',ascending=False)[:10]
cold=temp_country1.sort_values(by='AverageTemperature',ascending=True)[:10]
top_countries=pd.concat([hot,cold])
top_countries.sort_values('AverageTemperature',ascending=False,inplace=True)
f,ax=plt.subplots(figsize=(12,8))
sns.barplot(y='Country',x='AverageTemperature',data=top_countries,palette='cubehelix',ax=ax).set_title('Top Hottest And Coldest Countries')
plt.xlabel('Mean Temperture')
plt.ylabel('Country')


# ### Trend In Temperatures for the Top Economies

# In[ ]:


countries=temp_country.copy()
countries['dt']=pd.to_datetime(countries.dt).dt.strftime('%d/%m/%Y')
countries['dt']=countries['dt'].apply(lambda x: x[6:])
countries=countries[countries['AverageTemperature']!=0]
countries.drop('AverageTemperatureUncertainty',axis=1,inplace=True)
li=['United States','China','India','Japan','Germany','United Kingdom']
countries=countries[countries['Country'].isin(li)]
countries=countries.groupby(['Country','dt'])['AverageTemperature'].mean().reset_index()
countries=countries[countries['dt'].astype(int)>1850]
abc=countries.pivot('dt','Country','AverageTemperature')
f,ax=plt.subplots(figsize=(20,10))
abc.plot(ax=ax)


# ### Maximum Temperature Differences

# In[ ]:


try1=temp_country.copy()
try1['dt']=try1['dt'].apply(lambda x:x[6:])
try2=try1[try1['dt']>'1850'].groupby('Country')['AverageTemperature'].max().reset_index()
try3=try1[try1['dt']>'1850'].groupby('Country')['AverageTemperature'].min().reset_index()
try2=try2.merge(try3,left_on='Country',right_on='Country',how='left')
try2.columns=[['Country','Max Temp','Mean Temp']]
try2['difference']=try2['Max Temp']-try2['Mean Temp']
try2=try2.sort_values(by='difference',ascending=False)
sns.barplot(x='difference',y='Country',data=try2[:10],palette='RdYlGn').set_title('Countries with Highest Difference between Max And Mean Temperture')
plt.xlabel('Temperature Difference')


# ### Temperature Difference By Country

# In[ ]:


try2=try2.merge(df,left_on='Country',right_on='Country',how='left')
try2.dropna(inplace=True)

data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'Viridis',
        reversescale = True,
        showscale = True,
        locations = try2['Code'],
        z = try2['difference'],
        locationmode = 'Code',
        text = try2['Country'].unique(),
        marker = dict(
            line = dict(color = 'rgb(200,200,200)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Temperature Difference')
            )
       ]

layout = dict(
    title = 'Temperature Difference By Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(0,255,255)',
        projection = dict(
        type = 'Mercator',
            
        ),
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap2010')


# The above geomap shows the difference between the maximum and minimum temperatures for each country.

# ### Temperatures By States

# In[ ]:


states=pd.read_csv('../input/GlobalLandTemperaturesByState.csv')
states.dropna(inplace=True)
states['dt']=pd.to_datetime(states.dt).dt.strftime('%d/%m/%Y')


# In[ ]:


f,ax=plt.subplots(figsize=(15,8))
top_states=states.groupby(['State','Country'])['AverageTemperature'].mean().reset_index().sort_values(by='AverageTemperature',ascending=False)
top_states=top_states.drop_duplicates(subset='Country',keep='first')
top_states.set_index('Country',inplace=True)
top_states['AverageTemperature']=top_states['AverageTemperature'].round(decimals=2)
top_states.plot.barh(width=0.8,color='#0154ff',ax=ax)
for i, p in enumerate(zip(top_states.State, top_states['AverageTemperature'])):
    plt.text(s=p,x=1,y=i,fontweight='bold',color='white')


# ### Temperature Trends in Hottest States

# In[ ]:


top_states1=states.copy()
top_states1['dt']=top_states1['dt'].apply(lambda x:x[6:])
top_states1=top_states1[top_states1['State'].isin(list(top_states.State))]
top_states1=top_states1.groupby(['State','dt'])['AverageTemperature'].mean().reset_index()
top_states1=top_states1[top_states1['dt'].astype(int)>1900]
f,ax=plt.subplots(figsize=(18,8))
top_states1.pivot('dt','State','AverageTemperature').plot(ax=ax)
plt.xlabel('Year')


# ### USA Map For State Temperatures

# In[ ]:


USA=states[states['Country']=='United States']
USA.dropna(inplace=True)
USA['State'].replace({'Georgia (State)':'Georgia','District Of Columbia':'Columbia'},inplace=True)
USA=USA[['AverageTemperature','State']]
USA=USA.groupby('State')['AverageTemperature'].mean().reset_index()
dummy=['Alabama', 'AL','Alaska', 'AK','American Samoa', 'AS','Arizona', 'AZ','Arkansas', 'AR','California', 'CA','Colorado', 'CO'
,'Connecticut', 'CT','Delaware', 'DE','Columbia', 'DC','Florida', 'FL','Georgia', 'GA','Guam', 'GU','Hawaii', 'HI'
,'Idaho', 'ID','Illinois', 'IL','Indiana', 'IN','Iowa', 'IA','Kansas', 'KS','Kentucky', 'KY'
,'Louisiana', 'LA','Maine', 'ME','Maryland', 'MD','Marshall Islands', 'MH','Massachusetts', 'MA','Michigan', 'MI','Micronesia', 'FM'
,'Minnesota', 'MN','Mississippi', 'MS','Missouri', 'MO','Montana', 'MT','Nebraska', 'NE','Nevada', 'NV','New Hampshire', 'NH','New Jersey', 'NJ'
,'New Mexico', 'NM','New York', 'NY','North Carolina', 'NC','North Dakota', 'ND','Northern Marianas', 'MP'
,'Ohio', 'OH','Oklahoma', 'OK','Oregon', 'OR','Palau', 'PW','Pennsylvania', 'PA','Puerto Rico', 'PR','Rhode Island', 'RI'
,'South Carolina', 'SC','South Dakota', 'SD','Tennessee', 'TN','Texas', 'TX'
,'Utah', 'UT','Vermont', 'VT','Virginia', 'VA','Virgin Islands', 'VI','Washington', 'WA'
,'West Virginia', 'WV','Wisconsin', 'WI','Wyoming', 'WY']
code=dummy[1::2]
del dummy[1::2]
usa=pd.DataFrame(dummy,code)
usa.reset_index(inplace=True)
usa.columns=[['Code','State']]
USA=USA.merge(usa,left_on='State',right_on='State',how='left')


# In[ ]:


data = [ dict(
        type='choropleth',
        colorscale = 'Viridis',
        autocolorscale = False,
        locations = USA['Code'],
        z = USA['AverageTemperature'].astype(float),
        locationmode = 'USA-states',
        text =USA['State'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Average Temperature")
        ) ]

layout = dict(
        title = 'Average Temperature for USA States',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map' )


# ### Temperatures By Cities

# In[ ]:


cities=pd.read_csv('../input/GlobalLandTemperaturesByCity.csv')
cities.dropna(inplace=True)
cities['year']=cities['dt'].apply(lambda x: x[:4])
cities['month']=cities['dt'].apply(lambda x: x[5:7])
cities.drop('dt',axis=1,inplace=True)
cities=cities[['year','month','AverageTemperature','City','Country','Latitude','Longitude']]
cities['Latitude']=cities['Latitude'].str.strip('N')
cities['Longitude']=cities['Longitude'].str.strip('E')
cities.head()


# ### Hottest Cities By Country

# In[ ]:


temp_city=cities.groupby(['City','Country'])['AverageTemperature'].mean().reset_index().sort_values(by='AverageTemperature',ascending=False)
temp_city=temp_city.drop_duplicates(subset='Country',keep='first')
temp_city=temp_city.set_index(['City','Country'])
plt.subplots(figsize=(8,30))
sns.barplot(y=temp_city.index,x='AverageTemperature',data=temp_city,palette='RdYlGn').set_title('Hottest Cities By Country')
plt.xlabel('Average Temperature')


# ### Average Temperature Of Major Indian Cities By Month

# In[ ]:


indian_cities=cities[cities['Country']=='India']
indian_cities=indian_cities[indian_cities['year']>'1850']
major_cities=indian_cities[indian_cities['City'].isin(['Bombay','New Delhi','Bangalore','Hyderabad','Calcutta','Pune','Madras','Ahmadabad'])]
heatmap=major_cities.groupby(['City','month'])['AverageTemperature'].mean().reset_index()
trace = go.Heatmap(z=heatmap['AverageTemperature'],
                   x=heatmap['month'],
                   y=heatmap['City'],
                  colorscale='Viridis')
data=[trace]
layout = go.Layout(
    title='Average Temperature Of Major Cities By Month',
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')


# ### Trends in Major Cities

# In[ ]:


graph=major_cities[major_cities['year']>'1900']
graph=graph.groupby(['City','year'])['AverageTemperature'].mean().reset_index()
graph=graph.pivot('year','City','AverageTemperature').fillna(0)
graph.plot()
fig=plt.gcf()
fig.set_size_inches(18,8)


# ### Indian Cities Temperatures

# In[ ]:


cities=indian_cities.groupby(['City'])[['AverageTemperature']].mean().reset_index()
cities=cities.merge(indian_cities,left_on='City',right_on='City',how='left')
cities=cities.drop_duplicates(subset=['City'],keep='first')
cities=cities[['City','AverageTemperature_x','Latitude','Longitude']]


# ## City Temperatures in India(Folium)

# In[ ]:


import folium
import folium.plugins
location=cities[['Latitude','Longitude']]
location=location.astype(float)
def color_point(x):
    if x>=28:
        color='red'
    elif ((x>15 and x<28)):
        color='blue'
    else:
        color='green'
    return color   
map1 = folium.Map(location=[20.59, 78.96],tiles='CartoDB dark_matter',zoom_start=4.5)
for point in location.index:
    folium.CircleMarker(list(location.loc[point].values),popup='<b>City: </b>'+str(cities.City[point]+'<br><b>Avg Temperature:</b>'+str(cities.AverageTemperature_x[point])),                        radius=1,color=color_point(cities.AverageTemperature_x[point])).add_to(map1)
map1


# Drag the map to find more cities. Click on each point to find more information about each point.

# ### Stay Tuned!!
# ### Do Upvote If You Like it
