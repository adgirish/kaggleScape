
# coding: utf-8

# I will **update** this notebook **every day**. The visualizations are made with Plotly, so they are interactive (hover and click). 
# Any suggestions/criticism/commentary is greatly appreciated. Also, thank you for your votes and taking the time to look at this notebook; I'm sure you will find something usefull/interesting here.

# # Table of contents<a id='toc'></a>
# ***
# 
# 1. [Intro](#INTRO)
# 1. [What is MPI](#MPI)
# 1. [My process](#process)
# 1. [Updates](#UPDATES)
# 1. [At a glance](#Ataglance)
#     * 5.1. [MPI at continent, country and region level](#MPIatcontinent,countryandregionlevel)
#     * 5.2. [Quick view of the financial partners](#Quickviewofthefinancialpartners)
#     * 5.3. [Loans quick view](#Loansquickview)
# * [Analysis of the regions based on MPI](#regionsMPI)
#     * 6.1. [Bottom 30 (least poor)](#bottom30)
#     * 6.2. [Poverty decomposition for bottom 10](#decompositionb10)
#     * 6.3. [Top 30 poorest regions](#top30)
#     * 6.4. [Poverty decomposition for top 10 (view: stacked bar)](#decompositiont10)
#     * 6.5. [Population ratio living here](#popliving)
#     * 6.6. [Main problems for top 20 regions (view: heatmap)](#problemst20)
#     * 6.7. [Investments made in Bolivia and Philippines](#BoPh)
#     * 6.8. [Kiva influence on loans diversity](#Kiva)
#     * 6.9. [Investment distribution per region (average amount per quantity)](#Investmentdistribution)
# * [Analysis of unfunded loans](#unfundedloans)
#     * 7.1. [Quick view](#Quickview)
#     * 7.2. [Distribution on activities and sectors](#activitiessectors)
#     * 7.3. [Funded vs not funded amount per expired loans](#expiredloans)
#     * 7.4 [Distribution of unfunded amounts](#distribunfunded)
#     
#     
# 

# ## 1. INTRO <a id="INTRO"></a>
# ***
# 
# Who is Kiva and what does it do?  https://www.kiva.org 
# 
# What do they need?  "..localized models to estimate the poverty levels of residents in the regions where Kiva has active loans." - direct quote. In the 'Problem Statement' section I found this:
# ''A good solution would connect the features of each loan or product to one of several poverty mapping datasets, which indicate the average level of welfare in a region on as granular a level as possible."
# 
# This is my take on the Data Science for Good: Kiva Crowdfunding dataset.
# 
# I started by looking for poverty datasets and I found Oxford Poverty & Human Development Initiative (http://ophi.org.uk/), from where I got the MPI for 120 countries and MPI data for 1017 regions, but only for 79 countries. 

# ## 2. What is MPI <a id="MPI"></a>
# ***
# 
# The main poverty line used in the OECD and the European Union is a relative poverty measure based on "economic distance", a level of income usually set at 60% of the median household income.
# 
# The United States, in contrast, uses an absolute poverty measure. The US poverty line was created in 1963–64 and was based on the dollar costs of the U.S. Department of Agriculture's "economy food plan" multiplied by a factor of three. The multiplier was based on research showing that food costs then accounted for about one-third of money income. This one-time calculation has since been annually updated for inflation.
# 
# Both absolute and relative poverty measures are usually based on a person's yearly income and frequently take no account of total wealth.This is because poverty often involves being deprived on several fronts, which do not necessarily correlate well with wealth.
# 
# MPI - was developed in 2010 by the Oxford Poverty & Human Development Initiative (OPHI) and the United Nations Development Programme.
# 
# MPI = H * A H: Percentage of people who are MPI poor (incidence of poverty) A: Average intensity of MPI poverty across the poor (%)
# 
# The following ten indicators are used to calculate the MPI:
# 
# 1. Education (each indicator is weighted equally at 1/6):
#  - Years of schooling: deprived if no household member has completed six years of schooling
#  - Child school attendance: deprived if any school-aged child is not attending school up to class 8
# 
# 2. Health (each indicator is weighted equally at 1/6):
#  - Child mortality: deprived if any child has died in the family in past 5 years
#  - Nutrition: deprived if any adult or child for whom there is nutritional information is stunted[4]
# 
# 3. Standard of Living (each indicator is weighted equally at 1/18):
#  - Electricity: deprived if the household has no electricity
#  - Sanitation: deprived if the household’s sanitation facility is not improved (according to MDG guidelines), or it is improved but shared with other households
#  - Drinking water: deprived if the household does not have access to safe drinking water (according to MDG guidelines) or safe drinking water is more than a 30-minute walk from home roundtrip
#  - Floor: deprived if the household has a dirt, sand or dung floor
#  - Cooking fuel: deprived if the household cooks with dung, wood or charcoal
#  - Assets ownership: deprived if the household does not own more than one of: radio, TV, telephone, bike, motorbike or refrigerator and does not own a car or truck
# 
# The MPI relies on two main datasets that are publicly available and comparable for most developing countries: the Demographic and Health Survey (DHS), and the Multiple Indicators Cluster Survey (MICS). Certain countries use special datasets. The MPI data tables list in full the surveys used for each country.
# 
# Source: Wikipedia - https://en.wikipedia.org/wiki/Measuring_poverty

# ## 3. My process <a id="process"></a>
# ***
# 
# My goal is to match the loans with the regions for which I have MPI data. 
# 
# First I joined the kiva_loans with Tables_5.3_Contribution_of_Deprivations (a.k.a. table5 for the rest of this notebook), on region names. I got around 66k entries with MPI indicators out of 675k. The reason for this is, mostly, the spelling and region names not matching across the databases. Checking the region column for kiva_loans, showed that the entries are names of town, villages, sub-divisions, administrative and geographical regions. (ex: Achacachi, Bolivia - *Achacachi is a town on the Altiplano plateau in the South American Andes in the La Paz Department in Bolivia. It is the capital of the Omasuyos Province.More at Wikipedia*) I took a look at loan_themes_by_region which has a LocationName column with info about the location of the loan. Randomly checking ten entries, I got misspelled names, a lot of entries that were illegible (ex: Vietnam, Peru - *mostly because of conversion to CSV format*).
# 
# Second, I used the unique coordinates from loan_themes_by_region with a batch geocoding script to get all the administrative region names from Google Maps. I linked the Google region names with the ones in table5 and joined on coordinates with original database; I ended up with over 9k entries out of 15k with MPI decomposition, inside the loan_themes_by_region database. 
# 
# Now I'm analyzing this dataset. It gives us insight about where the money end up, what is the situation in that area, what big problems need to be addressed, where can the loans have the greatest impact.
# My analysis is focused on where SHOULD the money go, where they are most needed. 
# 
# The next step is to join MPI data on regions and countries with kiva_loans dataset. There are a lot of good analyses already done on this dataset, so I will try to answer questions that were skipped or overlooked. 
# 
# ## 4. UPDATES<a id="UPDATES"></a>
# 
# *10 April 2018*
# 
# I got the loans.csv from beluga and I will be using this dataset to run some in-depth analysis instead of using kiva_loans.csv.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

import colorlover as cl
from IPython.display import HTML

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_excel('../input/mpi-on-regions/mpi_on_regions.xlsx', encoding='utf-8')
df_loan_theme = pd.read_excel('../input/mpi-on-regions/all_loan_theme_merged_with_geo_mpi_regions.xlsx', encoding = 'utf-8')
kiva_mpi_reg_loc = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')


# ## 5. At a glance <a id="Ataglance"></a>
# ***

# In[ ]:


df.shape


# The original loan_themes_by_region had over 15k entries. I kept all the entries where I could match the region with MPI indicators from Table 5.

# In[ ]:


print('Missing values: ')
df.isnull().sum()


# In[ ]:


print('Statistics of the database: ')
df.describe()


# In[ ]:


for x in df.loc[:, df.dtypes == 'object']:
    y = len(list(df[x].unique()))
    print('For column %s we have %d individual values' %(x,y))


# In[ ]:


for x in df.loc[:, df.dtypes == np.float64]:
    y = df[x].mean()
    print('For column %s the average is %.2f' % (x,y))


# Info about the dataframe: 231 Financial partners, activating in 11 sectors, helping with financing over $ 220 millions in 118 types of loans, in 484 administrative regions, in 51 countries, over 6 world regions with an average of 67.81 % population living in rural area.
# The loans go towards people who are poor and deprived, on average: 
# *  13.48 % in schooling and 11.59% in child school attendance when Education contributes with 21.14% to the region's MPI
# *  33.09 % in child mortality and 15.48% in nutrition when Health has an impact of 42.28% to the MPI of the region
# *  5.28% have no electricity, 6.80% lack proper sanitation, 4.45% have trouble finding drinking water, 5.42% have no floors, 10.37% lack cooking fuel, 4.61% have only one item from the following and no car/truck: radio, TV, telephone, bicycle, motorbike, or refrigerator. The Living standards indicator makes up for 36.58% of the everall poverty.
# 
# When I first saw the Kiva databases I wondered how much of an impact these loans have; the sum is considerable and the amount of people involved is nothing short of impressive. More important is where would these money have the greatest impact.
# 
# I intend to find out...

# ### 5.1. MPI at continent, country and region level <a id="MPIatcontinent,countryandregionlevel"></a>

# In[ ]:


w_reg = df['World region'].unique()
mpi_mean = df.groupby('World region')['country MPI'].mean()

trace = go.Scatter(x=mpi_mean.round(3), y=mpi_mean.round(3),
                   mode = 'markers',
                   marker = dict(size=mpi_mean.values*200, color = mpi_mean.values, 
                                 colorscale='YlOrRd', showscale=True, reversescale=True),
                   text = w_reg, line=dict(width = 2, color='k'),
                  )

axis_template = dict(showgrid=False, zeroline=False, nticks=10,
                    showline=True, title='MPI Scale', mirror='all')


layout = go.Layout(title="Average MPI of the World's Regions analyzed",
                  hovermode = 'closest', xaxis=axis_template, yaxis=axis_template)


data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


# need 51 colors or shades for my plot
cs12 = cl.scales['12']['qual']['Paired']
col = cl.interp(cs12, 60)

coun_mpi = df[['Country', 'country MPI']].drop_duplicates().reset_index(drop=True)

trace = go.Bar(x = coun_mpi['country MPI'].sort_values(ascending=True),
               y = coun_mpi['Country'], 
               orientation = 'h',
               marker = dict(color = col),
              )

layout = go.Layout(title='MPI of the Countries in the dataset',
                   width = 800, height = 1000,
                   margin = dict(l = 175),
                   yaxis=dict(#tickangle=45,
                              tickfont=dict(family='Old Standard TT, serif', size=13, color='black')
                             ),
                  )


data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='MPI of the Countries')


# Let's see how the globe looks when we plot the poor regions...

# In[ ]:


data = [ dict(type = 'scattergeo',
              lat = kiva_mpi_reg_loc['lat'],
              lon = kiva_mpi_reg_loc['lon'],
              text = kiva_mpi_reg_loc['LocationName'],
              marker = dict(size = 8,
                            line = dict(width=0.5, color='k'),
                            color = kiva_mpi_reg_loc['MPI'],
                            colorscale = 'YlOrRd',
                            reversescale = True,
                            colorbar=dict(title="MPI")
                            )
             )
       ]
layout = dict(title = 'Regional MPI across the globe',
             geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
             )
fig = dict( data=data, layout=layout )
py.iplot(fig)


# ### 5.2. Quick view of the financial partners <a id="Quickviewofthefinancialpartners"></a>

# Who are the most active partners and in which coutries they conduct business?

# In[ ]:


spec = cl.scales['10']['div']['Spectral']
spectral = cl.interp(spec, 20)

pc = df.groupby(['Field Partner Name'])['number'].sum().sort_values(ascending=False).head(20)
info = []
for i in pc.index:
    cn = str(df['Country'][df['Field Partner Name'] == i].unique())
    cn = cn.strip("['']").replace("' '", ', ')
    info.append(cn)

trace = go.Bar(x = pc.index,
               y = pc.values,
               text = info,
               orientation = 'v',
               marker = dict(color = spectral)
               )

layout = go.Layout(title='Number of loans facilitated by top 20 financial partners',
                   width = 800, height = 500,
                   margin = dict(b = 175),
                   )

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# As we can see there is a big gap between the top 3 and the rest of the partners. ( Kiva works with over 200 partners across the globe ). Negros Women for Tomorrow Foundation from Philippines atracts the most loans (over 70k), followed by One Acre Fund, from Kenya (more than 67k) and iDE Cambodia, Cambodia (close to 50k).
# 
# How about the gross amount?

# In[ ]:


pc2 = df.groupby(['Field Partner Name'])['amount'].sum().sort_values(ascending=False).head(20)
info2 = []
for i in pc2.index:
    cn = str(df['Country'][df['Field Partner Name'] == i].unique())
    cn = cn.strip("['']").replace("' '", ', ')
    info2.append(cn)


# In[ ]:


trace = go.Bar(x = pc2.index,
               y = pc2.values,
               text = info2,
               orientation = 'v',
               marker = dict(color = spectral),
               )

layout = go.Layout(title='Gross amount of the loans facilitated by financial partners (in $)',
                   width = 800, height = 500,
                   margin = dict(b = 175),
                   )

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Negros Women for Tomorrow Foundation from Philippines, contributes with over 22M in credits, followed by Asociacion Arariwa from Peru with over 10M and, at almost $9M , Urwego Opportunity Bank from Rwanda and CrediCampo from El Salvador.

# ### 5.3. Loans distribution per sectors and activities <a id='Loansquickview'></a>

# In[ ]:


gb2 = df.groupby(['Loan Theme Type', 'World region', 'sector'])
num2 = gb2['number'].agg(np.sum)
amo2 = gb2['amount'].agg(np.sum)
sumsdf2 = pd.DataFrame({'amount': gb2['amount'].agg(np.sum), 'number': gb2['number'].agg(np.sum)}).reset_index()

hover_text = []
for index, row in sumsdf2.iterrows():
    hover_text.append(('Number of loans: {a}<br>' + 
                       'Amount of loans: {b}<br>' +
                       'Loan theme type: {c}<br>' +
                       #'Country: {d}<br>' +
                       'World region: {e}<br>').format(a = row['number'],
                                                       b = row['amount'],
                                                       c = row['Loan Theme Type'],
                                                       #d = row['Country'],
                                                       e = row['World region']
                                                  )
                     )    

sumsdf2['text'] = hover_text

sectors = ['General Financial Inclusion', 'Other', 'Water and Sanitation', 'Mobile Money and ICT', 'Clean Energy', 'Education',
           'DSE Direct', 'Artisan', 'SME Financial Inclusion', 'Agriculture', 'Health']

data = []

for s in sorted(sectors):
    trace = go.Scatter(x = sumsdf2['amount'][sumsdf2['sector'] == s], 
                   y = sumsdf2['number'][sumsdf2['sector'] == s],
                   name = s,
                   mode = 'markers',
                   text = sumsdf2['text'][sumsdf2['sector'] == s],
                   hoverinfo = 'text',
                   hoveron = 'points+fills',    
                   marker = dict(size = np.sqrt(sumsdf2['amount'][sumsdf2['sector'] == s]),
                                 sizemode = 'area', 
                                 line=dict(width = 2),
                                 ),
                   )
    data.append(trace)

layout = go.Layout(title="Type of loans grouped on sectors and world regions",
                   hovermode = 'closest', 
                   xaxis=dict(title='Total amount of loans', type='log'),
                   yaxis=dict(title='Total number of loans', type='log'),
                   paper_bgcolor='rgb(243, 243, 243)',
                   plot_bgcolor='rgb(243, 243, 243)',
                   )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Let's see if a boxplot doesn't shine some light on what's going on: (using a logarithmic scale is the way to go) 

# In[ ]:


trace = []
for name, group in df_loan_theme.groupby(['sector']):
    trace.append(go.Box( x = group['amount'].values,
                         name = name
                       )
                )
layout = go.Layout( title = 'Amount of loans per sectors', 
                  margin = dict( l = 170 ),
                   xaxis = dict( type = 'log' )
                  )
fig = go.Figure(data = trace, layout = layout)
py.iplot(fig)


# If we deselect the General Financial Inclusion we gain a bit of perspective but not enough. A boxplot for Loan types is huge and a lot of themes keep repeating. A good idea would be to group together variations of a theme and maybe the plot will have more of a 'clean' look.

# In[ ]:


gb = df.groupby(['Sub-national region', 'Country', 'World region'])
num = gb['number'].agg(np.sum)
amo = gb['amount'].agg(np.sum)
sumsdf = pd.DataFrame({'amount': gb['amount'].agg(np.sum), 'number': gb['number'].agg(np.sum)}).reset_index()
sumsdf[30:35]


# In[ ]:


hover_text = []
for index, row in sumsdf.iterrows():
    hover_text.append(('Number of loans: {a}<br>' + 
                       'Amount of loans: {b}<br>' +
                       'Sub-national region: {c}<br>' +
                       'Country: {d}<br>').format(a = row['number'],
                                                  b = row['amount'],
                                                  c = row['Sub-national region'],
                                                  d = row['Country']
                                                  )
                     )    

sumsdf['text'] = hover_text
    
world = ['East Asia and the Pacific', 'Sub-Saharan Africa', 'Arab States', 'Latin America and Caribbean', 
         'Central Asia', 'South Asia']

data = []

for w in sorted(world):
    trace = go.Scatter(x = sumsdf['amount'][sumsdf['World region'] == w], 
                   y = sumsdf['number'][sumsdf['World region'] == w],
                   name = w,
                   mode = 'markers',
                   text = sumsdf['text'][sumsdf['World region'] == w],
                   hoverinfo = 'text',
                   marker = dict(size = np.sqrt(sumsdf['amount'][sumsdf['World region'] == w]),
                                 sizemode = 'area',                                 
                                 line=dict(width = 2),                                 
                                 ),
                   )
    data.append(trace)

layout = go.Layout(title="Loans across regions",
                   hovermode = 'closest', 
                   xaxis=dict(title='Total amount of loans', type='log'),
                   yaxis=dict(title='Total number of loans', type='log'),
                   paper_bgcolor='rgb(243, 243, 243)',
                   plot_bgcolor='rgb(243, 243, 243)',
                   )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ## 6. Analysis of the regions based on MPI<a id='regionsMPI'></a>
# ***

# ### 6.1. Bottom 30 (least poor)<a id='bottom30'></a>

# In[ ]:


top30 = df[['Country', 'Sub-national region', 'region MPI']].drop_duplicates().reset_index(drop=True)
small_mpi = top30.sort_values('region MPI').reset_index(drop=True)

c11s3 = cl.scales['11']['qual']['Set3']
col2 = cl.interp(c11s3, 30)

x = small_mpi['region MPI'].head(30)
y = small_mpi['Sub-national region'].head(30)
country = small_mpi['Country'].head(30)

trace = go.Bar(x = x[::-1],
               y = y[::-1],
               text = country[::-1],
               orientation = 'h',
               marker=dict(color = col2),
               )

layout = go.Layout(title='The 30 least poor regions of the globe',
                   width = 800, height = 800,
                   margin = dict(l = 195),
                   xaxis=dict(title='MPI'),
                   yaxis=dict(#tickangle=45, 
                              tickfont=dict(family='Old Standard TT, serif', size=13, color='black'),
                             )
                   )

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### 6.2. Poverty decomposition for bottom 10<a id='decompositionb10'></a>

# In[ ]:


t10_reg = df.sort_values('region MPI')
t10_reg.drop_duplicates(subset=['region MPI', 'Sub-national region'], inplace=True)
t10_reg = t10_reg.reset_index(drop=True)

labels = ['Education', 'Health', 'Living standards']
colors = ['rgb(11, 133, 215)', 'rgb(51,160,44)', 'rgb(240, 88, 0)']

for x in range(len(t10_reg[:10])):
    values = t10_reg.iloc[x, 17:20]
    trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent', marker=dict(colors=colors), sort=False)
    layout = go.Layout(title='Contribution to overall poverty for %s, %s' % (
                        t10_reg['Sub-national region'][x], t10_reg['Country'][x]))
    fig = go.Figure(data=[trace], layout=layout)
    py.iplot(fig)


# These regions have the lowest poverty rating; they have a good chance to be off the list soon. The main contribution to poverty is Health for 6 regions in 3 countries (Colombia, Jordan, Ecuador), and Living standards for 4 regions in 2 countries (Mongolia, Peru). For most of these regions, the main problem is their proximity to conflict zones.
# 
# Let's dig deeper:

# In[ ]:


labels = ['Schooling', 'Child school attendance', 'Child mortality', 'Nutrition', 'Electricity', 'Improved sanitation', 
          'Drinking water', 'Floor', 'Cooking fuel', 'Asset ownership']

x = list(t10_reg['Sub-national region'][:10])
text = list(t10_reg['Country'][:10])
data = []

for i in range(10):
    trace = go.Bar(x=x, y=t10_reg.iloc[:10, (20+i)], name=labels[i], text=text)
    data.append(trace)

    
layout = go.Layout(barmode='stack', title='Poverty decomposition for the 10 least poor regions', showlegend=True, 
                   margin = dict(b = 125),
                  xaxis=dict(title='Region', tickangle = 45), yaxis=dict(title='Percent %'))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# As we can see the main problems are child mortality, nutrition and education (both indicators have a strong presence in the chart)

# ### 6.3. Top 30 poorest regions<a id='top30'></a>

# In[ ]:


top30 = df[['Country', 'Sub-national region', 'region MPI']].drop_duplicates().reset_index(drop=True)
high_mpi = top30.sort_values('region MPI').reset_index(drop=True)

trace = go.Bar(x = high_mpi['region MPI'].tail(30),
               y = high_mpi.index[:30],
               text = high_mpi['Sub-national region'].tail(30) + ', ' + high_mpi['Country'].tail(30),
               orientation = 'h',
               marker = dict(color = col2),
               )

layout = go.Layout(title='The 30 poorest regions of the globe',
                   width = 800, height = 800,
                   xaxis=dict(title='MPI'),
                   yaxis=dict(tickangle=45, 
                              tickfont=dict(family='Old Standard TT, serif', size=11, color='black'),
                             )
                   )

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### 6.4. Poverty decomposition for top 10 (view: grouped and stacked bar)<a id='decompositiont10'></a>

# In[ ]:


t10_hi = df.sort_values('region MPI', ascending=False)
t10_hi.drop_duplicates(subset=['region MPI', 'Sub-national region'], inplace=True)
t10_hi = t10_hi.reset_index(drop=True)

labels = ['Education', 'Health', 'Living standards']
colors = ['rgb(11, 133, 215)', 'rgb(51,160,44)', 'rgb(240, 88, 0)']

x = list(t10_hi['Sub-national region'][:10])
text = list(t10_hi['Country'][:10])
data = []

for i in range(3):
    trace = go.Bar(x=x, y=t10_hi.iloc[:10, (17+i)], name=labels[i], text=text)
    data.append(trace)

    
layout = go.Layout(barmode='group', title='Contribution to overall poverty for the 10 poorest regions', showlegend=True, 
                   xaxis=dict(title='Region'))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# **Note**: On mouse hover you can see the name of the country for the region in the chart.

# In[ ]:


labels = ['Schooling', 'Child school attendance', 'Child mortality', 'Nutrition', 'Electricity', 'Improved sanitation', 
          'Drinking water', 'Floor', 'Cooking fuel', 'Asset ownership']

x = list(t10_hi['Sub-national region'][:10])
text = list(t10_hi['Country'][:10])
data = []

for i in range(10):
    trace = go.Bar(x=x, y=t10_hi.iloc[:10, (20+i)], name=labels[i], text=text)
    data.append(trace)

    
layout = go.Layout(barmode='stack', title='Poverty decomposition for top 10 poorest regions', showlegend=True, 
                  xaxis=dict(title='Region'), yaxis=dict(title='Percent %'))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# 9 out of 10 regions are in Sub-Saharan Africa; only Oecusse, Timor-Leste (a small island N of Australia) is part of East Asia and the Pacific.
# 
# We can see that more than 50% of the issues have to do with education and health; these will take time and concentrated effort to improve, as we noticed that even for the regions with the smallest MPI these are still ongoing issues. But for these regions, we can also notice a number of common problems:  scarcity of electricity, sanitation, floors, cooking fuels, which might be easier to tackle and improve the quality of life. The indicators for Living standards have a stronger presence for the poorest regions.

# ### 6.5. What's the percentage of population living in these poor regions?<a id='popliving'></a>

# In[ ]:


x = t10_hi['Population Share of the Region (%)'].head(50)
y = t10_hi['region MPI'].head(50)

hover_text = []
for index, row in t10_hi.head(50).iterrows():
    hover_text.append(('Population Share of the Region: {a}<br>' + 
                       'Region MPI: {b}<br>' +
                       'Sub-national region: {c}<br>' +
                       'Country: {d}<br>').format(a = str(row['Population Share of the Region (%)']*100)+'%',
                                                  b = row['region MPI'],
                                                  c = row['Sub-national region'],
                                                  d = row['Country']
                                                  )
                     )    


trace = go.Scatter(x=x, y=y, mode = 'markers',
                   text = hover_text,
                   hoverinfo = 'text',
                   line=dict(width = 2, color='k'),
                   marker = dict(size=x*200,
                                 color = x,
                                 colorscale='Rainbow', showscale=True, reversescale=False,
                                 ),
                   )

layout = go.Layout(title="Percent of the country's population living in the 50 poorest areas",
                   hovermode = 'closest',                  
                   xaxis=dict(title='Population Share of the Region (%)'),
                   yaxis=dict(title='MPI of the region')
                   )


data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# The plot shows us the regions that can benefit most from targeted loans and the population affected by this. The spread is high ranging from 3% all the way to almost 30%. Even for Androy, Madagascar, the smallest circle in our plot, 3% out of 25 mil is 750k people impacted by multidimensional poverty. At the opposite pole we have the North region of Burundi, with a population of 3.136.744 people.

# ### 6.6 What are the main problems of poverty and how big is the impact ?<a id='problemst20'></a>

# In[ ]:


df_mpi = df.drop_duplicates(subset=['Sub-national region', 'region MPI']).reset_index(drop=True)
df_mpi.sort_values(['region MPI'], ascending=False, inplace=True)
df_mpi = df_mpi.iloc[:, 10:].reset_index(drop=True)

x = list(df_mpi.columns.values)[10:20]
y = df_mpi['Sub-national region'].head(20)
z = df_mpi.iloc[:, 10:20].head(20).values

trace = go.Heatmap(x = x, y = y, z = z,
                  colorscale = 'Jet',
                  colorbar = dict(title = 'IN %', x = 'center')
                  )

layout = go.Layout(title='Decomposition of problems for 20 poorest regions',
                   margin = dict(l = 155, b = 100),
                    xaxis = dict(tickfont = dict(size=11)),
                    #yaxis = dict(tickangle = 45)
                  )

fig = dict(data = [trace], layout = layout)
py.iplot(fig)


# In[ ]:


df_loan = df.groupby(['Country', 'Sub-national region', 'sector', 'Loan Theme Type', 'region MPI'])
df_loan = pd.DataFrame({'number': df_loan['number'].sum(), 'amount': df_loan['amount'].sum()}).reset_index()


# In[ ]:


mycolors = ['#F81106','#FA726C','#F8C1BE',
            '#137503','#54B644','#B2F5A7',
            '#051E9B','#4358C0','#A6B3F9',
           '#9C06A0','#C34BC6','#F3A1F6',
           '#A07709','#CDA742','#F4DC9D',
           '#08A59E','#4DD5CE','#AAF7F3']
hover_text = []
for index, row in df_loan.iterrows():
    hover_text.append(('Loan type: {a}<br>' + 'Sector: {b}<br>' + 'Amount: {c}<br>').format(
                        a = row['Loan Theme Type'], b = row['sector'], 
                        c = '$' + str("{:,}".format(row['amount'])))
                     )
df_loan['text'] = hover_text


# ### 6.7. Investments made in Bolivia and Philippines<a id='BoPh'></a>
# 
# Let's see where all the money went for **Bolivia**. In the pie charts below, there is info regarding the amount per type of loan that got funded for every MPI region.

# In[ ]:


countries = ['Bolivia']
for c in countries:
    creg = df_loan[df_loan['Country'] == c]
    regions = pd.Series(creg['Sub-national region'].unique())
    for r in regions:
        selector = df_loan[(df_loan['Country'] == c) & (df_loan['Sub-national region'] == r)]
        trace = go.Pie(values = selector['amount'],
              labels = selector['Loan Theme Type'],
               text = selector['text'],
               hoverinfo = 'text',
               textinfo = 'percent',
               textfont = dict(size=15),
               marker = dict(colors = mycolors,
                            line = dict(color='k', width=0.75)),           
              )
        layout = go.Layout(title = 'Amount and type of loans for {}, {} (MPI: {})'.format(r,c, selector['region MPI'].median())
                          )
        fig = go.Figure(data = [trace], layout = layout)
        py.iplot(fig)


#  Now let's see how the **Philippines** fare:

# In[ ]:


countries = ['Philippines']

for c in countries:
    creg = df_loan[df_loan['Country'] == c]
    regions = pd.Series(creg['Sub-national region'].unique())
    for r in regions:
        selector = df_loan[(df_loan['Country'] == c) & (df_loan['Sub-national region'] == r)]
        trace = go.Pie(values = selector['amount'],
              labels = selector['Loan Theme Type'],
               text = selector['text'],
               hoverinfo = 'text',
               textinfo = 'percent',
               textfont = dict(size=15),
               marker = dict(colors = mycolors,
                            line = dict(color='k', width=0.75)),           
              )
        layout = go.Layout(title = 'Amount and type of loans for {}, {} (MPI: {})'.format(r,c, selector['region MPI'].median())
                          )
        fig = go.Figure(data = [trace], layout = layout)
        py.iplot(fig)


# If we could ignore Loan type = 'General', the charts would show a better picture regarding where the money went; not to mention that we gain almost no knowledge about what's really going on. Unfortunately there are regions that only got funded this type of loan; removing it completely would result in an incomplete image about the country. The best we can do is deselect it in the charts. 

# ### 6.8. Kiva influence on loans diversity<a id='Kiva'></a>

# In[ ]:


lt = df_loan_theme.groupby(['Loan Theme Type', 'forkiva'])['number'].sum()
lt = lt.to_frame().reset_index()
lt = lt.pivot(index = 'Loan Theme Type', columns = 'forkiva', values= 'number')
lt['No'] = lt['No'].fillna(0)
lt['Yes'] = lt['Yes'].fillna(0)
lt['total'] = lt['No'] + lt['Yes']
# get rid of General loan theme as is skewing the chart
lt = lt.loc[~(lt['No'] > 300000)]
lt = lt.sort_values('total', ascending = False).head(40)


# In[ ]:


trace0 = go.Bar(x = lt.No[::-1], y = lt.index[::-1], name = 'No',
              orientation = 'h')
trace1 = go.Bar(x = lt.Yes[::-1], y = lt.index[::-1], name = 'Yes',
              orientation = 'h')

data = [trace0, trace1]

layout = go.Layout(barmode = 'stack', title = 'Kiva influence on loan themes',
                   height = 900,
                   margin = dict(l = 155, t = 100),
                   xaxis = dict(tickfont = dict(size = 11)),
                   yaxis = dict(tickfont = dict(size = 10),
                               )
                  )

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# Looks like Kiva has a strong influence in creating and pushing loans towards critical areas like agriculture, water, green energy; also helping in conflict zones and with seniors. It seems some of these types were created specifically for Kiva lenders.

# ### 6.9. Investment distribution per region (average amount per quantity)<a id='Investmentdistribution'></a>

# In[ ]:


avg_am = df.groupby(['Sub-national region', 'Country', 'region MPI'])
lsum = avg_am['amount'].sum()
lcount = avg_am['number'].sum()
lavg = (lsum / lcount).round(2)
avg_am = pd.DataFrame({'loan amounts': lsum, 'total loans': lcount, 'average loan amount': lavg}).reset_index()
avg_am.head(10)


# In[ ]:


hover_text = []
for index, row in avg_am.iterrows():
    hover_text.append(('Total loans: {a}<br>' + 
                       'Average loan amount: {b}<br>' +
                       'Region MPI: {c}<br>' +
                       'Country: {d}<br>').format(a = row['total loans'], 
                                                  b = '$' + str("{:,}".format(row['average loan amount'])),
                                                  c = row['region MPI'],
                                                  d = row['Country'])
                     )

avg_am['info'] = hover_text


# In[ ]:


l100 = avg_am[avg_am['total loans'] <= 100].sort_values(['average loan amount'], ascending = False).head(20)
trace = go.Bar( x = l100['Sub-national region'],
                y = l100['average loan amount'],
               text = l100['total loans'],
                textposition = 'outside',
                hoverinfo = 'text',
                hovertext = l100['info'],
                marker=dict(color='#B2F5A7',
                            line=dict(color='rgb(8,48,107)',
                                      width=1.5)
                           )
              )
        
layout = go.Layout( title = 'Average amount per region (number of loans <= 100)')
fig = go.Figure(data = [trace], layout = layout)
py.iplot(fig)


# It makes sense to have large average amounts for regions where we have very small number of loans. Are these new regions on the Kiva platform ? Or just big projects that managed to get funded? The MPI is high (over 0.35) for these 3 regions that attracted $230k in 4 loans...
# - we notice that** Cibao Norte** and **Cibao Sur** from Dominican Republic obtained 46 loans with an average of over 5k per loan.
# - the average mpi for the plotted regions is 0.205.

# In[ ]:


l1k = avg_am[(avg_am['total loans'] > 100) & (avg_am['total loans'] <= 1000)].sort_values(['average loan amount'], ascending = False).head(20)
trace = go.Bar( x = l1k['Sub-national region'],
                y = l1k['average loan amount'],
                text = l1k['total loans'],
                textposition = 'outside',
                hoverinfo = 'text',
                hovertext = l1k['info'],
                marker=dict(color='#54B644',
                            line=dict(color='rgb(8,48,107)',
                                      width=1.5)
                           )
              )
        
layout = go.Layout( title = 'Average amount per region (number of loans <= 1000)',
                  margin = dict(b = 135)
                  )
fig = go.Figure(data = [trace], layout = layout)
py.iplot(fig)


# As expected, when the number of loans increases to hundreds, the average amount drops under $1500 for 80 % of the investigated regions.
# - the top 3 leaders for this plot, with MPI's at the lower end of the scale,  are all from Bolivia. 
# - most of the regions in this plot have got a number of loans between 100 and 350 with 3 outliers: the cluster of areas grouped as one region, **Boyacá, Cmarca, Meta** with 455 loans and **Bogota** with over 800 loans, both from *Columbia* and **Matagalpa** from *Nicaragua* with almost 900 loans. 
# - the mean MPI for these regions is 0.161 (the average MPI dropped 20%) 

# In[ ]:


l10k = avg_am[avg_am['total loans'] > 1000].sort_values(['average loan amount'], ascending = False).head(20)
trace = go.Bar( x = l10k['Sub-national region'],
                y = l10k['average loan amount'],
                text = l10k['total loans'],
                textposition = 'outside',
                hoverinfo = 'text',
                hovertext = l10k['info'],
                marker=dict(color='#137503',
                            line=dict(color='rgb(8,48,107)',
                                      width=1.5)
                           )
              )
        
layout = go.Layout( title = 'Average amount per region (number of loans > 1000)',
                   height = 600,
                   margin = dict(b = 155, r = 100)                  
                  )
fig = go.Figure(data = [trace], layout = layout)
py.iplot(fig)


# In this plot the number of loans increased to thousands, and the average dropped under $1200. 
#  - the leaders of this plot are the **North** and **Central** regions from *Jordan* with barely more than 1000 loans and the **Coast** region, from *Ecuador*, with more than 5000 loans. 
#  - a noticeable outlier is **La Paz**, *Bolivia* with over 10k loans.
#  - the regions in this plot have the lowest MPI from all three plots, with a mean of 0.048. (this is a drop of more than 75% than the first plot)
#  
#  In conclusion, we can see great improvement in regions where focused investments were made.

# ## 7. Analysis of unfunded loans (when values from funded amount are different than loan amount)<a id='unfundedloans'></a>
# ***

# Let's continue this analysis by investigating all the loans that Kiva has posted so far and also use the country stats database uploaded by beluga.

# In[2]:


coun_stats = pd.read_csv('../input/mpi-on-regions/country_stats.csv') # from beluga 
kl = pd.read_csv('../input/mpi-on-regions/all_kiva_loans.csv') # the 1.4 Mil entries downloaded from Kiva


# Applying a value count function on the 'distribution_model' column from the loans.csv, we find out that 1% of the loans (16790) don't go through a field partner. The countries that got direct loans are Kenya (with aprox 10k loans) and US (close to 6.7k). The top 5 sectors where these loans went are: 
# - Services: 4013; Food: 3231; Retail: 3067; Agriculture 2506; Clothing 2112.
# 
# The borrowers genders were: female    13027; male       3304, all 4 repayment intervals are present, and the length of the loans vary from 3 mo to 60 mo, with most of them being 3, 6, 12, 24, 36 months.
# (3.0mo - 5477 loans, 24.0 - 2976, 6.0 - 1907, 36.0 - 1624, 12.0 - 1063)
# 
# Not particular helpful for this analysis, but interesting nonetheless. 
# 

# Not all loans get funded (about 4.5% of our database). Let's get some insight into these...

# ### 7.1. Quick view<a id='Quickview'></a>

# In[3]:


kl['not_funded'] = kl['loan_amount'] - kl['funded_amount']


# In[4]:


nf = kl[kl['not_funded'] != 0].reset_index(drop=True)
nf.describe()


# Looks like we have a loan that was overfunded... On a closer inspection, seems like about a dozen of loans got overfunded by 25 or 50 ($), with one exception: a loan for farm supplies in a village in Armenia got 3400, instead of 3000. Interesting fact is that it's status is 'expired'...I wonder if it was a glitch, like the status didn't change when it was completly funded, and it allowed people to keep donating...
# Clicking on the 'Output' will show the 12 entries.
# 
# Anyhow , I will exclude them from this analysis.

# In[5]:


nf[nf.status == 'funded']
nf[nf.not_funded < 0]


# In[6]:


nf['not_funded_percent'] = nf['not_funded'] / nf['loan_amount'] * 100
nf['not_funded_percent'] = round(nf['not_funded_percent'], 2)


# In[7]:


nf['borrower_genders']=[elem if elem in ['female','male'] else 'group' for elem in nf['borrower_genders'] ]
borr = nf['borrower_genders'].value_counts()

pie1 = go.Pie( labels = ['Funded amount', 'Not funded amount'],
             values = [44457755, 45720020],
             hoverinfo = 'label+value+percent',
             textfont=dict(size=18, color='#000000'),
             name = "Loans that didn't get the funds",
             domain = dict( x=[0, 0.5] )
            )
pie2 = go.Pie( labels = borr.index,
             values = borr.values,
             hoverinfo = 'label+value+percent',
             textfont=dict(size=18, color='#000000'),
             text = "Distribution of genders",
             domain = dict( x=[0.5, 1] )
            )

layout = go.Layout(showlegend=True, title = 'Amount and gender distribution for expired loans')

fig = go.Figure(data=[pie1, pie2], layout=layout)
py.iplot(fig)


# In[8]:


sta = nf['status'].value_counts()
rep = nf['repayment_interval'].value_counts()

trace0 = go.Bar(x = sta.index, y = sta.values,
               marker = dict(color = sta.values,
                            colorscale = 'Viridis'
                            ),
                name = 'Status'
               )

trace1 = go.Bar(x = rep.index, y = rep.values,
               marker = dict(color = rep.values,
                            colorscale = 'Viridis'
                            ),
                name = 'Repayment interval'
               )

fig = tls.make_subplots(rows=1, cols=2, subplot_titles=( 'Status distribution for<br>partialy funded loans', 'Repayment interval for<br>partialy funded loans'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(showlegend = False, width = 800)

py.iplot(fig)


# When we look at the status of these loans, we see that we still have 2 of them labeled 'funded'.... even though are missing a small amount. I will focus on the 'expired' loans.

# In[9]:


nf_exp = nf[nf['status'] == 'expired']
country_sum = nf_exp.groupby(['country_name'])['not_funded'].sum().sort_values().tail(30)

trace = go.Bar(x = country_sum.values, 
               y = country_sum.index,
               orientation = 'h',
               marker = dict(color = country_sum.values,
                             colorscale = 'Viridis',
                             reversescale = True
                            )
              )
layout = go.Layout(title = 'Amount of unfunded loans per country',
                   margin = dict(l = 150), height = 750
                  )
fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)


# Even though US has it's own problems with economically challenged areas, according to MPI doesn't qualify for poverty loans, and it's leading the chart for unfunded amounts by a large margin. 

# ### 7. 2. Distribution on activities and sectors<a id='activitiessectors'></a>

# In[10]:


act = nf_exp['activity_name'].value_counts()[:30]
sec = nf_exp['sector_name'].value_counts()

activity = go.Bar(x = act.index, y = act.values,
               marker = dict(color = act.values,
                            colorscale = 'Portland'
                            ),
               )

sector = go.Bar(x = sec.index, y = sec.values,
               marker = dict(color = sec.values,
                            colorscale = 'Portland'
                            ),
               )

fig = tls.make_subplots(rows=2, cols=1, subplot_titles=( "Top 30 activities that didn't get loans", "Sectors that didn't get loans"))
fig.append_trace(activity, 1, 1)
fig.append_trace(sector, 2, 1)

fig['layout'].update(showlegend = False, height=900)

py.iplot(fig)


# I feel like these plot are very similar with the ones for the sectors\activities that got the most loans. I personally thought 'Personal Use' would be higher on the scale.

# In[11]:


S = nf_exp['sector_name'].unique()
N = len(list(S))
C = [ 'hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N) ] # create your own 'rainbow' palette (source: plotly website) 

trace = []
for s,c in zip(S,C):
    trace.append(go.Box(y = nf_exp[nf_exp['sector_name'] == s]['not_funded'],
                        marker = dict(color = c),
                        name = s
                        )
                 )
layout = go.Layout(title = 'Amount of unfunded loans per sectors',
                   yaxis = dict(type = 'log')
                  )

fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)


# Even though the y values are plotted on a logarithmic scale, the values displayed on mouse hover, are the entries from the data set. We are using this box plot to spot the outliers in every sector. 
# Info from this chart:
# - the minimum unfunded amount for an expired loan was $5 in Services and the biggest minimum is $150 in Wholesale
# - Retail and Agriculture have overfunded loans, so their minimum is in the negative (marked as expired loans despite having funded amount > requested amount - bug?)
# - Entertainment has the highest median and Personal Use has the lowest.
# - Entertainment has the highest maximum and no outliers; Personal Use has the lowest maximum without outliers (called upper fence in plotly)
# - 5 sectors have loans with unfunded amounts bigger than 10k
# - Agriculture and Food sectors have the biggest outliers (> $ 40k)

# ### 7.3. Funded vs not funded amount per expired loans<a id='expiredloans'></a>

# In[12]:


sec_f = nf_exp.groupby(['sector_name'])['funded_amount'].sum()
sec_nf = nf_exp.groupby(['sector_name'])['not_funded'].sum()

first = go.Bar( x = sec_f.index, y = sec_f.values,
               name = 'Funded amount of loan'              
              )

second = go.Bar( x = sec_nf.index, y = sec_nf.values,
               name = 'Not funded amount of loan'              
              )

data = [first, second]
layout = go.Layout( barmode = 'group', 
                  title = 'Funded vs not funded amounts per sector'
                  )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[13]:


big_act = nf_exp.groupby(['activity_name'])['loan_amount'].sum().sort_values(ascending=False).head(50)
act_f = [ nf_exp[nf_exp['activity_name'] == i]['funded_amount'].sum() for i in big_act.index ]
act_nf = [ nf_exp[nf_exp['activity_name'] == i]['not_funded'].sum() for i in big_act.index ]

first = go.Bar( x = big_act[:30].index, y = act_f[:30],
               name = 'Funded amount of loan'              
              )

second = go.Bar( x = big_act[:30].index, y = act_nf[:30],
               name = 'Not funded amount of loan'              
              )

data = [first, second]
layout = go.Layout( barmode = 'group', 
                  title = 'Funded vs not funded amounts per activity<br>(top 30 by amount of loans requested)',
                   showlegend=False,
                   margin = dict(b=175),
                   #bargap = 0.35
                  )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### 7.4. Distribution of unfunded amounts <a id='distribunfunded'></a>

# In[14]:


fund = np.log(nf_exp['funded_amount']) + 1
nfund = np.log(nf_exp['not_funded']) + 1

trace1 = go.Histogram(x=nf_exp['funded_amount'], nbinsx=50, opacity=0.75, name='Funded amount')
trace2 = go.Histogram(x=nf_exp['not_funded'], nbinsx=50, opacity=0.75, name='Not funded amount')

trace3 = go.Histogram(x=fund, nbinsx=50, opacity=0.75, name='Funded amount')
trace4 = go.Histogram(x=nfund, nbinsx=50, opacity=0.75, name='Not funded amount')

fig = tls.make_subplots(rows=1, cols=2, subplot_titles = ('Normal distribution', 'Logarithmic distribution'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(title='Funded vs Unfunded amount', showlegend=False)
py.iplot(fig)


# Our data has a strong right skewed; let's try and improve the distribution by getting rid of outliers. That loan that got overfunded by ($) 400 is marked as expired so it appears in our analysis. Also the gap in values is huge for 25 percent of our data. We need to take care of this. 

# In[20]:


nf_exp['not_funded'].describe()


# In[32]:


fun = np.array(nf_exp['funded_amount'])
mean = np.mean(fun, axis=0)
sd = np.std(fun, axis=0)

fun_noq = [x for x in fun if (x > mean - sd)]
fun_noq = [x for x in fun_noq if (x < mean + sd)]


# In[33]:


nofun = np.array(nf_exp['not_funded'])
mean = np.mean(nofun, axis=0)
sd = np.std(nofun, axis=0)

nofun_noq = [x for x in nofun if (x > mean - sd)]
nofun_noq = [x for x in nofun_noq if (x < mean + sd)]


# In[34]:


fun_noq_log = np.log(fun_noq) + 1
nofun_noq_log = np.log(nofun_noq) + 1

trace1 = go.Histogram(x=fun_noq, nbinsx=50, opacity=0.75, name='Funded amount')
trace2 = go.Histogram(x=nofun_noq, nbinsx=50, opacity=0.75, name='Not funded amount')

trace3 = go.Histogram(x=fun_noq_log, nbinsx=50, opacity=0.75, name='Funded amount')
trace4 = go.Histogram(x=nofun_noq_log, nbinsx=50, opacity=0.75, name='Not funded amount')

fig = tls.make_subplots(rows=1, cols=2, subplot_titles = ('Normal distribution', 'Logarithmic distribution'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(title='Funded vs Unfunded amount<br>(expired loans with removed quartiles)', showlegend=False)
py.iplot(fig)


# Since 75 percent of the entries are less than 1000, I chose the values located at one standard deviation over the mean.

# In[36]:


hist_data = [fun_noq, nofun_noq]
group_labels = ['Funded amount', 'Amount not funded']
colors = ['#3A4750', '#F64E8B']

fig = ff.create_distplot(hist_data, group_labels=group_labels, bin_size = 50, curve_type='normal', colors=colors, show_rug=False)
fig['layout'].update(title='Funded vs Unfunded amount distribution plot')
py.iplot(fig)


# In[ ]:


# not fund > 50% count / activities, 
# histogram for not_funded


# **-- To Be Continued -- ** ([Back to top](#toc))
