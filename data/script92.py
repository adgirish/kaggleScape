
# coding: utf-8

# # INTRODUCTION 
# 
# *Youth is Wasted on the Young*
# 
# The famous phrase by George Bernard Shaw (or Oscar Wilde?), normally used with pejorative connotations when one sees a foolish youth whiling away his/her hours aimlessly, or perhaps with a hint of jealousy on the tongue of the deliverer. However, looking at the extremely high youth unemployment rates of today and the struggles of young professionals in securing a job, one would think apt that of inverting the aforementioned phrase. I personally know and have heard stories where the young people spend weeks and months working odd jobs and manual work because they cannot secure permanent, stable work.
# 
# And that is what I will be planning to do in this notebook, to explore the visualising capabilities of Plotly with regards to producing intuitive and interactive plots and hopefully convey a story that hopes to delve into the world of youth unemployment rates over the globe and whether over a period of half a decade, has anything changed for the better. 
# 
# Let's Go 

# In[ ]:


# Import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Let's load the data

# In[ ]:


country = pd.read_csv('../input/API_ILO_country_YU.csv')
country.head()


# In[ ]:


print(country.shape)


# Quite a handful of the 'countries' listed in the Country Name column are not actually countries. A quick glance reveals that we have some sub-continents listed in there (East Asia & Pacific, Europe & Central Asia) as well as other non-single country related things like (Upper middle income, IBRD only and IDA only). For the purposes of producing our visuals we will therefore whittle down our country list as follows

# In[ ]:


# Uncomment to view the total list of country names
print(country['Country Name'].unique())


# In[ ]:


# Create our new list of countries that we want to plot for. This was done manually as I was lazy
# to think of any clever tricks (eg text processing) to filter. 
country_list = ['Afghanistan','Angola','Albania','Argentina','Armenia','Australia'
,'Austria','Azerbaijan','Burundi','Belgium','Benin','Burkina Faso','Bangladesh','Bulgaria'
,'Bahrain','Bosnia and Herzegovina','Belarus','Belize','Bolivia','Brazil','Barbados','Brunei Darussalam'
,'Bhutan','Botswana','Central African Republic','Canada','Switzerland','Chile','China','Cameroon'
,'Congo','Colombia','Comoros','Cabo Verde','Costa Rica','Cuba','Cyprus','Czech Republic','Germany'
,'Denmark','Dominican Republic','Algeria','Ecuador','Egypt','Spain','Estonia','Ethiopia','Finland','Fiji'
,'France','Gabon','United Kingdom','Georgia','Ghana','Guinea','Greece','Guatemala','Guyana','Hong Kong'
,'Honduras','Croatia','Haiti','Hungary','Indonesia','India','Ireland','Iran','Iraq','Iceland','Israel'
,'Italy','Jamaica','Jordan','Japan','Kazakhstan','Kenya','Cambodia','Korea, Rep.','Kuwait','Lebanon','Liberia'
,'Libya','Sri Lanka','Lesotho','Lithuania','Luxembourg','Latvia','Macao','Morocco','Moldova','Madagascar'
,'Maldives','Mexico','Macedonia','Mali','Malta','Myanmar','Montenegro','Mongolia','Mozambique','Mauritania'
,'Mauritius','Malawi','Malaysia','North America','Namibia','Niger','Nigeria','Nicaragua','Netherlands'
,'Norway','Nepal','New Zealand   ','Oman','Pakistan','Panama','Peru','Philippines','Papua New Guinea'
,'Poland','Puerto Rico','Portugal','Paraguay','Qatar','Romania','Russian Federation','Rwanda','Saudi Arabia'
,'Sudan','Senegal','Singapore','Solomon Islands','Sierra Leone','El Salvador','Somalia','Serbia','Slovenia'
,'Sweden','Swaziland','Syrian Arab Republic','Chad','Togo','Thailand','Tajikistan','Turkmenistan','Timor-Leste'
,'Trinidad and Tobago','Tunisia','Turkey','Tanzania','Uganda','Ukraine','Uruguay','United States','Uzbekistan'
,'Venezuela, RB','Vietnam','Yemen, Rep.','South Africa','Congo, Dem. Rep.','Zambia','Zimbabwe'
]


# In[ ]:


# Create a new dataframe with our cleaned country list
country_clean = country[country['Country Name'].isin(country_list)]


# ## 3D GLOBE VISUALISATIONS WITH PLOTLY
# 
# Now onto the plots. I'm pretty new to Plotly so I derived my ideas and inspirations from these brilliant Kaggle notebooks. Please check them out:
# 
# [Map of temperatures and analysis of Global Warming][1] by Vladislav Amelin
# 
# [Master Painter Choices Throughout History][2] by brandnewpeterson
# 
# 
#   [1]: https://www.kaggle.com/amelinvladislav/d/berkeleyearth/climate-change-earth-surface-temperature-data/map-of-temperatures-and-analysis-of-global-warming
# 
#   [2]: https://plot.ly/~brandnewpeterson/487/master-painter-color-choices-throughout-history/?utm_source=mailchimp-jan-2015&utm_medium=email&utm_campaign=generalemail-jan2015&utm_term=master-painter
# 
# Now onto our own plots

# ## Has Global Youth Unemployment gotten better in these 4 years?

# In[ ]:


# Plotting 2010 and 2014 visuals
metricscale1=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'], 
              [0.15, 'rgb(171,221,164)'], [0.2, 'rgb(230,245,152)'], 
              [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'], 
              [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = metricscale1,
        showscale = True,
        locations = country_clean['Country Name'].values,
        z = country_clean['2010'].values,
        locationmode = 'country names',
        text = country_clean['Country Name'].values,
        marker = dict(
            line = dict(color = 'rgb(250,250,225)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Unemployment\nRate')
            )
       ]

layout = dict(
    title = 'World Map of Global Youth Unemployment in the Year 2010',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(28,107,160)',
        #oceancolor = 'rgb(222,243,246)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap2010')

metricscale2=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'], 
              [0.15, 'rgb(171,221,164)'], [0.2, 'rgb(230,245,152)'], 
              [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'], 
              [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = metricscale2,
        showscale = True,
        locations = country_clean['Country Name'].values,
        z = country_clean['2014'].values,
        locationmode = 'country names',
        text = country_clean['Country Name'].values,
        marker = dict(
            line = dict(color = 'rgb(250,250,200)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Unemployment\nRate')
            )
       ]

layout = dict(
    title = 'World Map of Global Youth Unemployment in the Year 2014',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(28,107,160)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(202, 202, 202)',
                width = '0.05'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap2014')


# *PLEASE CLICK AND SCROLL ABOVE. THESE GLOBE PLOTS ARE INTERACTIVE. DOUBLE-CLICK ON THE GLOBE IF YOU WANT TO GET BACK TO THE ORIGINAL VIEW.*

# ### Insights from the Visuals
# 
# **YEAR 2010** <br>
# Well first things first, in 2010 it is pretty obvious that Asia seems to be doing pretty well in terms of youth unemployment evident from the patches of green ( unemployment levels below 10% ).
# Africa and Southern/Eastern Europe on the other seems to be going through a pretty rough patch given their relatively higher occurrences of red ( unemployment levels above 30% ).
# 
# **YEAR 2014** <br>
# In 2014, just from a quick glance at visuals and colors on the plots tells us that the Russian Federation and the United States have improved quite a bit in terms of reducing their youth's unemployment rate. However, it is telling from the colors that things do not seem to have improved in Africa and Europe.

# ## SCATTER PLOTS OF UNEMPLOYMENT RATES
# 
# Let's now delve deeper into the data by taking a closer look at each individual country and its unemployment rate from the year 2010 to the year 2014. I am going to try to achieve this by using a scatter plot of the rates, plotting the visuals such that the higher the unemployment in a country the larger the scatter bubble. Hopefully this portrays the data more intuitively so let's see how it looks

# In[ ]:


# Scatter plot of 2010 unemployment rates
trace = go.Scatter(
    y = country_clean['2010'].values,
    mode='markers',
    marker=dict(
        size= country_clean['2010'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = country_clean['2010'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = country_clean['Country Name'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Scatter plot of unemployment rates in 2010',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Unemployment Rate',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot of 2014 unemployment rates
trace1 = go.Scatter(
    y = country_clean['2014'].values,
    mode='markers',
    marker=dict(
        size=country_clean['2014'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = country_clean['2014'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = country_clean['Country Name']
)
data = [trace1]

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2014',
    hovermode= 'closest',
    xaxis= dict(
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2014')


# *PLEASE CLICK AND MOVE THE SCATTER PLOTS. THEY ARE INTERACTIVE. DOUBLE CLICK TO GET BACK TO THE ORIGINAL VIEW*

# ### Insights from the Visuals
# 
# Well unfortunately it seems from 2010 to 2014 things have gone from bad to worse for quite a few of the European countries, the ones affected from the sub-prime crisis. For example, both Spain and Greece's youth unemployment has gone from their orange color (indicating about 30-40% rate) to a hazardous red color (about 40-50% unemployment).

# ## Net Movement of Unemployment Rate over the 5 years
# 
# Finally I investigate the movement of the unemployment rate of the top and bottom 15 countries to see if there was a net increase or a decrease in the number of youths unemployed from 2010 all the way to 2014. Doing so will quantitatively tell us which countries have had their rates increased or vice-versa.

# In[ ]:


# I first create an array containing the net movement in the unemployment rate.
diff = country_clean['2014'].values - country_clean['2010'].values


# Now I just want to visualise the top and bottom 15 countries with the most changes in their unemployment. Therefore the code to do this is as follows

# In[ ]:


x, y = (list(x) for x in zip(*sorted(zip(diff, country_clean['Country Name'].values), 
                                                            reverse = True)))

# Now I want to extract out the top 15 and bottom 15 countries 
Y = np.concatenate([y[0:15], y[-16:-1]])
X = np.concatenate([x[0:15], x[-16:-1]])


# **HEATMAP OF CHANGE IN UNEMPLOYMENT RATE FOR TOP & BOTTOM 15 COUNTRIES**
# 
# Let's now plot a Seaborn heatmap to visualise how the unemployment rates change over the years. The heatmap is sorted such that countries with the largest increases in unemployment are towards the top half of the plot while countries with the highest decreases in unemployment are found towards the bottom half. 

# In[ ]:


# Resize our dataframe first
keys = [c for c in country_clean if c.startswith('20')]
country_resize = pd.melt(country_clean, id_vars='Country Name', value_vars=keys, value_name='key')
country_resize['Year'] = country_resize['variable']

# Use boolean filtering to extract only our top 15 and bottom 15 moving countries
mask = country_resize['Country Name'].isin(Y)
country_final = country_resize[mask]

# Finally plot the seaborn heatmap
plt.figure(figsize=(12,10))
country_pivot = country_final.pivot("Country Name","Year",  "key")
country_pivot = country_pivot.sort_values('2014', ascending=False)
ax = sns.heatmap(country_pivot, cmap='jet', annot=False, linewidths=0, linecolor='white')
plt.title('Movement in Unemployment rate ( Warmer: Higher rate, Cooler: Lower rate )')


# **BARPLOT OF CHANGE IN UNEMPLOYMENT RATE FOR TOP 15 & BOTTOM 15 COUNTRIES**
# 
# And here we plot a Seaborn barplot to visualise the data in a different manner. The nice thing about seaborn barplots being that with any negative values ( the negative values being countries with unemployment rate decrease), it automatically plots the charts in the reverse x-axis order therefore making for intuitive visuals.

# In[ ]:



# Plot using Seaborn's barplot
sns.set(font_scale=1) 
f, ax = plt.subplots(figsize=(16, 14))
colors_cw = sns.color_palette('cubehelix_r', len(X))
sns.barplot(X, Y, palette = colors_cw[::-1])
Text = ax.set(xlabel='Decrease in Youth Unemployment Rates', 
              title='Net Increase in Youth Unemployment Rates')


# **BARPLOT OF CHANGE IN UNEMPLOYMENT RATE FOR ALL COUNTRIES**
# 
# Finally let's look at the barplot charts for youth unemployment rate changes in *ALL* countries present in the dataset for thoroughness, not just the top 15 & bottom 15 

# In[ ]:



# Plot using Seaborn's barplot
sns.set(font_scale=1) 
f, ax = plt.subplots(figsize=(16, 48))
colors_cw = sns.color_palette('cubehelix_r', len(x))
sns.barplot(x, y, palette = colors_cw[::-1])
Text = ax.set(xlabel='Decrease in Youth Unemployment Rates', 
              title='Net Increase in Youth Unemployment Rates')


# ## CONCLUDING REMARKS
# 
# The visuals portray a tale of both tragedy and hope. It is tragic when one can see that in these years after the Sub-prime mortgage crisis from 2010 to 2014, global youth unemployment has only gone from bad to worse shot up for the developed European countries of Greece, Cyprus, Spain, Italy, Portugal and Croatia. These are the top 6 countries which have had a net increase in their youth unemployment rates. One would have hoped that situations would have improved on average in these countries, but the data begs to differ.
# 
# On the other hand, it seems that the Eastern European countries of Estonia, Latvia, Lithuania, Moldova and Montenegro have seen the biggest decreases in their unemployment rates. The data seems to tell a story that is very Eurocentric, where European countries have experienced the wildest and most extreme swings in their youth unemployment rates, one where developed countries have fallen ground while the developing blocs are catching up. 
