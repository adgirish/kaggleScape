
# coding: utf-8

# What is Zillow?
#      ---
# A few year old Web site that provides an automated home-valuation service offering "Zestimates" - estimated market values - for 67 million homes nationwide. Zillow has data on 110 million homes in 48 states, or 88 percent of all homes in the country. <br>
# 
# How does it work? Zillow collects data from a wide variety of sources, such as county recorder and assessor offices and real estate listings, then uses proprietary algorithms to arrive at automated models for estimating home values. Users can tweak property information to fine-tune the valuations. , <br>
# 
# How does it differ from other pricing reports? Most real estate reports provide data on homes that recently changed hands, which means median price can be skewed depending on whether the mix of homes tilted toward more-expensive or less-expensive homes. Zillow gives information on all homes, not just those recently sold. <br>
# 
# 
# 

# ![](https://www.housingwire.com/ext/resources/images/editorial/A-New-Big-Images/houses/For-Sale/Computer-keyboard-house-for-sale-copy.jpg?1459374825)

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
plt.style.use('bmh')
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
from plotly.graph_objs import *
from mpl_toolkits.mplot3d import axes3d
import folium 
from folium import plugins
from folium.plugins import HeatMap


# 110 million homes from North America listed on Zillow website
#      ---

# In[ ]:


m = folium.Map(location=[38,-10], tiles="Mapbox Bright", zoom_start=1.5)
folium.Circle(
      location=[40, -106],
      popup='North America',
      radius=3000000,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)
m


# **In this notebook i will be analysing Zillow Economics Data .
#  As mentioned in the overview of this dataset ,  Actionable information and Unique insight is what we need to derive from below datasets.  Showing raw data as it is thru graph wont help much <br><br>
# **  **Data mining** <br>
#  Instead of relying on dataframes given in this dataste , I will be preparing intermediate dataframes by summarizing data on Year , month , state , region for visualizing the data using various technique. To maintain the readibility i have hiddent the code , you can fork this notebook to take a look at the code

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Lets first start our analysis with small datasets (State_time_series.csv)
# >  First lets build some EDA models on small datasets , as you move ahead you can reproduce the similar graphs by using  bigger datasets. <br><br>
# > Use model built above on big datasets like (City_time_series.csv) . We need to take help of crosswalk table here to add few more columns for better visualization.

# In[ ]:


State_house = pd.read_csv("../input/State_time_series.csv", parse_dates=['Date'])
State_house['Century'] = State_house.Date.dt.year


# Lets Quickly check the benefit of having your house listed on Zillow and then move ahead with our further analysis.
#         ---
#  > I am expecting  reduced number of days on zillow over the years
# 

# In[ ]:


State_raw_house = State_house.groupby(['RegionName', State_house.Date.dt.year])['DaysOnZillow_AllHomes'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['RegionName',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['RegionName']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)
fig, ax = plt.subplots(figsize=(13, 13))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('no of days on Zillow')
ax.set_ylabel('States')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (180,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 no of days', (60,50),fontsize=12,rotation=0,color='Green')
ax.annotate('2017 no of days', (60,51),fontsize=12,rotation=0,color='Blue');

ax.annotate('with % +/-', (60,49),fontsize=12,rotation=0,color='Brown');


# **Look at the massive drop in the Number of days on Zillow for almost all the regions here except Vermont**
#     ---
# Zillow is the perfect destination to market your house and to get maximum exposure.

# Lets check sqft Area of different type of Homes listed on Zillow <br>
#      ---
# note:- This data is not provided here. I have divided Median listed price by Median sqft price 

# In[ ]:


flat_type = []
flat_area = []
v1_features = State_house.iloc[:,14:23].columns
v2_features = State_house.iloc[:,5:14].columns
f, ax = plt.subplots(figsize=(15, 10))
for cn1, cn2 in zip(v1_features,v2_features):
    area = (State_house[cn1].dropna().mean()/State_house[cn2].dropna().mean())
    flat_type.append(str(cn2)[26:])
    flat_area.append(area)
g = sns.barplot( y = flat_type,
            x = flat_area,
                palette="GnBu_d")
ax.set_xlabel('SqFt')
plt.title("SqFt Area of different type of Homes listed on Zillow")
plt.show()


# Lets plot  below columns in one graph for comparison and later plot each one of them seperately to analyse region wise spread
#        ---
# 1. PctOfHomesDecreasingInValues_AllHomes <br>
# 2. PctOfHomesIncreasingInValues_AllHomes<br>
# 3. PctOfHomesSellingForGain_AllHomes<br>
# 4. PctOfHomesSellingForLoss_AllHomes<br>
# 

# In[ ]:


trace0 = go.Scatter(
    x = State_house.groupby(State_house.Date.dt.year).PctOfHomesIncreasingInValues_AllHomes.mean().keys(),
    y = State_house.groupby(State_house.Date.dt.year).PctOfHomesIncreasingInValues_AllHomes.mean().values,
    mode = 'lines',
    name = 'PctOfHomesIncreasingInValues_AllHomes'
)
trace1 = go.Scatter(
    x = State_house.groupby(State_house.Date.dt.year).PctOfHomesDecreasingInValues_AllHomes.mean().keys(),
    y = State_house.groupby(State_house.Date.dt.year).PctOfHomesDecreasingInValues_AllHomes.mean().values,
    mode = 'lines',
    name = 'PctOfHomesDecreasingInValues_AllHomes'
)
trace2 = go.Scatter(
    x = State_house.groupby(State_house.Date.dt.year).PctOfHomesSellingForGain_AllHomes.mean().keys(),
    y = State_house.groupby(State_house.Date.dt.year).PctOfHomesSellingForGain_AllHomes.mean().values,
    mode = 'lines',
    name = 'PctOfHomesSellingForGain_AllHomes'
)
trace3 = go.Scatter(
    x = State_house.groupby(State_house.Date.dt.year).PctOfHomesSellingForLoss_AllHomes.mean().keys(),
    y = State_house.groupby(State_house.Date.dt.year).PctOfHomesSellingForLoss_AllHomes.mean().values,
    mode = 'lines',
    name = 'PctOfHomesSellingForLoss_AllHomes'
)
data = [trace0, trace1,trace2,trace3]

layout = go.Layout(
      xaxis=dict(title='year'),
      yaxis=dict(title='Median listing PricePerSqft$$'),
      title=('Pct of Homes sold for loss/gain and Homes Decreasing/increasing in values '))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# The percentage of homes in an given region with values that have decreased in the past year
#      ---

# In[ ]:


years = State_house.groupby(State_house.Date.dt.year).PctOfHomesDecreasingInValues_AllHomes.mean().keys() 
years = np.array(years)
plt.figure(figsize=(12, 6))
plt.plot(years, State_house.groupby(State_house.Date.dt.year).PctOfHomesDecreasingInValues_AllHomes.mean().values, color="#3F5D7D", lw=10)  ;


# In[ ]:


State_raw_house = State_house.groupby(['RegionName', State_house.Date.dt.year])['PctOfHomesDecreasingInValues_AllHomes'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['RegionName',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['RegionName']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)


fig, ax = plt.subplots(figsize=(13, 8))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('PctOfHomesDecreasingInValues')
ax.set_ylabel('States')
ax.set_title('')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (92,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 PctOfHomesDecreasingInValues', (50,6),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 PctOfHomesDecreasingInValues', (50,7),fontsize=14,rotation=0,color='Blue');

ax.annotate('with % +/-', (60,5),fontsize=14,rotation=0,color='Brown');


# The percentage of homes in an given region with values that have increased in the past year
#      ---

# In[ ]:


years = State_house.groupby(State_house.Date.dt.year).PctOfHomesIncreasingInValues_AllHomes.mean().keys() 
years = np.array(years)
plt.figure(figsize=(12, 6))
plt.plot(years, State_house.groupby(State_house.Date.dt.year).PctOfHomesIncreasingInValues_AllHomes.mean().values, color="#3F5D7D", lw=10)  ;


# In[ ]:


State_raw_house = State_house.groupby(['RegionName', State_house.Date.dt.year])['PctOfHomesIncreasingInValues_AllHomes'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['RegionName',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['RegionName']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)


fig, ax = plt.subplots(figsize=(13, 8))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('PctOfHomesIncreasingInValues')
ax.set_ylabel('States')
ax.set_title('')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (92,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 PctOfHomesIncreasingInValues', (10,34),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 PctOfHomesIncreasingInValues', (10,35),fontsize=14,rotation=0,color='Blue');

ax.annotate('with % +/-', (10,33),fontsize=14,rotation=0,color='Brown');


# The percentage of homes in an area that sold for a price higher than the previous sale price
#      ----

# In[ ]:


years = State_house.groupby(State_house.Date.dt.year).PctOfHomesSellingForGain_AllHomes.mean().keys() 
years = np.array(years)
plt.figure(figsize=(12, 6))
plt.plot(years, State_house.groupby(State_house.Date.dt.year).PctOfHomesSellingForGain_AllHomes.mean().values, color="#3F5D7D", lw=10)  ;


# In[ ]:


State_raw_house = State_house.groupby(['RegionName', State_house.Date.dt.year])['PctOfHomesSellingForGain_AllHomes'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['RegionName',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['RegionName']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)


fig, ax = plt.subplots(figsize=(13, 4))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('PctOfHomesSellingForGain')
ax.set_ylabel('States')
ax.set_title('')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (98,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 PctOfHomesSellingForGain', (58,3.75),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 PctOfHomesSellingForGain', (58,4),fontsize=14,rotation=0,color='Blue');

ax.annotate('with % +/-', (58,3.5),fontsize=14,rotation=0,color='Brown');


# The percentage of homes in an area that sold for a price lower than the previous sale price
#       ---

# In[ ]:


years = State_house.groupby(State_house.Date.dt.year).PctOfHomesSellingForLoss_AllHomes.mean().keys() 
years = np.array(years)
plt.figure(figsize=(12, 6))
plt.plot(years, State_house.groupby(State_house.Date.dt.year).PctOfHomesSellingForLoss_AllHomes.mean().values, color="#3F5D7D", lw=10)  ;


# In[ ]:


State_raw_house = State_house.groupby(['RegionName', State_house.Date.dt.year])['PctOfHomesSellingForLoss_AllHomes'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['RegionName',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['RegionName']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)


fig, ax = plt.subplots(figsize=(13,4))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('PctOfHomesSellingForLoss')
ax.set_ylabel('States')
ax.set_title('')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (43,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 PctOfHomesSellingForLoss', (30,2.50),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 PctOfHomesSellingForLoss', (30,2.75),fontsize=14,rotation=0,color='Blue');

ax.annotate('with % +/-', (30,2.25),fontsize=14,rotation=0,color='Brown');


# InventoryRaw AllHomes
#      ---
# > **Median of weekly snapshot of for-sale homes within a region for a given month**

# In[ ]:


years = State_house.groupby(State_house.Date.dt.year).InventoryRaw_AllHomes.mean().keys() 
years = np.array(years)
plt.figure(figsize=(12, 6))
plt.plot(years, State_house.groupby(State_house.Date.dt.year).InventoryRaw_AllHomes.mean().values, color="#3F5D7D", lw=10)  ;
#plt.plot(years, State_house.groupby(State_house.Date.dt.year).InventorySeasonallyAdjusted_AllHomes.mean().values, color="Red", lw=10)  ;


# In[ ]:


State_raw_house = State_house.groupby(['RegionName', State_house.Date.dt.year])['InventoryRaw_AllHomes'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['RegionName',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['RegionName']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)


fig, ax = plt.subplots(figsize=(13,12))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('InventoryRaw_AllHomes')
ax.set_ylabel('States')
ax.set_title('')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (250000,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 InventoryRaw_AllHomes', (150000,21),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 InventoryRaw_AllHomes', (150000,22),fontsize=14,rotation=0,color='Blue');

ax.annotate('with % +/-', (150000,20),fontsize=14,rotation=0,color='Brown');


# Lets explore PerSquareft price and total listed price for different types of home available in USA. 
#                      ----
# > This dataset contains values since 1996 to 2017 , We will first analyse how the price spread is through Boxplot

# Median Listing **PricePerSqft** for 
#      ----
# > 1Bedroom <br>
# 2Bedroom<br>
# 3Bedroom<br>
# 4Bedroom<br>
# 5BedroomOrMore<br>
# AllHomes<br>
# CondoCoop<br>
# DuplexTriplex<br>
# SingleFamilyResidence<br>

# Year wise Breakup
#       ----
#    > Boxplot <br>
#    Lineplot
# 

# In[ ]:


v_features = State_house.iloc[:,5:14].columns
plt.figure(figsize=(15,28))
gs = gridspec.GridSpec(9,1)
for i, cn in enumerate(State_house[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data = State_house)
    sns.boxplot(x='Century',y=cn,data=State_house[8243:])
    ax.set_xlabel('')
    ax.set_title(str(cn)[26:])
    ax.set_ylabel(' ')


# In[ ]:


State_scatter = State_house
State_scatter['Century'] = State_scatter.Date.dt.year
State_1Bedroom = State_scatter.groupby(['Century'])['MedianListingPricePerSqft_1Bedroom'].mean()
State_2Bedroom = State_scatter.groupby(['Century'])['MedianListingPricePerSqft_2Bedroom'].mean()
State_3Bedroom = State_scatter.groupby(['Century'])['MedianListingPricePerSqft_3Bedroom'].mean()
State_4Bedroom = State_scatter.groupby(['Century'])['MedianListingPricePerSqft_4Bedroom'].mean()
State_5BedroomOrMore = State_scatter.groupby(['Century'])['MedianListingPricePerSqft_5BedroomOrMore'].mean()
State_AllHomes = State_scatter.groupby(['Century'])['MedianListingPricePerSqft_AllHomes'].mean()
State_CondoCoop = State_scatter.groupby(['Century'])['MedianListingPricePerSqft_CondoCoop'].mean()
State_DuplexTriplex = State_scatter.groupby(['Century'])['MedianListingPricePerSqft_DuplexTriplex'].mean()
State_SingleFamilyResidence = State_scatter.groupby(['Century'])['MedianListingPricePerSqft_SingleFamilyResidence'].mean()

trace0 = go.Scatter(
    x = State_1Bedroom.index[14:],
    y = State_1Bedroom.values[14:],
    mode = 'lines',
    name = '1Bedroom'
)
trace1 = go.Scatter(
    x = State_2Bedroom.index[14:],
    y = State_2Bedroom.values[14:],
    mode = 'lines',
    name = '2Bedroom'
)
trace2 = go.Scatter(
    x = State_3Bedroom.index[14:],
    y = State_3Bedroom.values[14:],
    mode = 'lines',
    name = '3Bedroom'
)
trace3 = go.Scatter(
    x = State_4Bedroom.index[14:],
    y = State_4Bedroom.values[14:],
    mode = 'lines',
    name = '4Bedroom'
)
trace4 = go.Scatter(
    x = State_5BedroomOrMore.index[14:],
    y = State_5BedroomOrMore.values[14:],
    mode = 'lines',
    name = '5BedroomOrMore'
)
trace5 = go.Scatter(
    x = State_AllHomes.index[14:],
    y = State_AllHomes.values[14:],
    mode = 'lines',
    name = 'AllHomes'
)
trace6 = go.Scatter(
    x = State_CondoCoop.index[14:],
    y = State_CondoCoop.values[14:],
    mode = 'lines',
    name = 'CondoCoop'
)
trace7 = go.Scatter(
    x = State_DuplexTriplex.index[14:],
    y = State_DuplexTriplex.values[14:],
    mode = 'lines',
    name = 'DuplexTriplex'
)
trace8 = go.Scatter(
    x = State_SingleFamilyResidence.index[14:],
    y = State_SingleFamilyResidence.values[14:],
    mode = 'lines',
    name = 'SingleFamilyResidence'
)

data = [trace0, trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8]

layout = go.Layout(
      xaxis=dict(title='year'),
      yaxis=dict(title='Median listing PricePerSqft$$'),
      title=('Median Listing PricePerSqft over the years'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Overall
#     ---

# In[ ]:


v_features = State_house.iloc[:,5:14].columns
plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(2,5)
for i, cn in enumerate(State_house[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data = State_house)
    ax.set_title(str(cn)[26:])
    ax.set_ylabel(' ')


# Median Listed **Price** for 
#      ----
# > 1Bedroom <br>
# 2Bedroom<br>
# 3Bedroom<br>
# 4Bedroom<br>
# 5BedroomOrMore<br>
# AllHomes<br>
# CondoCoop<br>
# DuplexTriplex<br>
# SingleFamilyResidence<br>

# Year wise Breakup
#       ----
#    > Boxplot <br>
#    Lineplot
# 

# In[ ]:


v_features = State_house.iloc[:,14:23].columns
plt.figure(figsize=(15,28))
gs = gridspec.GridSpec(9,1)
for i, cn in enumerate(State_house[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data = State_house)
    sns.boxplot(x='Century',y=cn,data=State_house[8243:])
    ax.set_xlabel('')
    ax.set_title(str(cn)[19:])
    ax.set_ylabel(' ')


# In[ ]:


State_scatter = State_house
State_scatter['Century'] = State_scatter.Date.dt.year
State_1Bedroom = State_scatter.groupby(['Century'])['MedianListingPrice_1Bedroom'].mean()
State_2Bedroom = State_scatter.groupby(['Century'])['MedianListingPrice_2Bedroom'].mean()
State_3Bedroom = State_scatter.groupby(['Century'])['MedianListingPrice_3Bedroom'].mean()
State_4Bedroom = State_scatter.groupby(['Century'])['MedianListingPrice_4Bedroom'].mean()
State_5BedroomOrMore = State_scatter.groupby(['Century'])['MedianListingPrice_5BedroomOrMore'].mean()
State_AllHomes = State_scatter.groupby(['Century'])['MedianListingPrice_AllHomes'].mean()
State_CondoCoop = State_scatter.groupby(['Century'])['MedianListingPrice_CondoCoop'].mean()
State_DuplexTriplex = State_scatter.groupby(['Century'])['MedianListingPrice_DuplexTriplex'].mean()
State_SingleFamilyResidence = State_scatter.groupby(['Century'])['MedianListingPrice_SingleFamilyResidence'].mean()

trace0 = go.Scatter(
    x = State_1Bedroom.index[14:],
    y = State_1Bedroom.values[14:],
    mode = 'lines',
    name = '1Bedroom'
)
trace1 = go.Scatter(
    x = State_2Bedroom.index[14:],
    y = State_2Bedroom.values[14:],
    mode = 'lines',
    name = '2Bedroom'
)
trace2 = go.Scatter(
    x = State_3Bedroom.index[14:],
    y = State_3Bedroom.values[14:],
    mode = 'lines',
    name = '3Bedroom'
)
trace3 = go.Scatter(
    x = State_4Bedroom.index[14:],
    y = State_4Bedroom.values[14:],
    mode = 'lines',
    name = '4Bedroom'
)
trace4 = go.Scatter(
    x = State_5BedroomOrMore.index[14:],
    y = State_5BedroomOrMore.values[14:],
    mode = 'lines',
    name = '5BedroomOrMore'
)
trace5 = go.Scatter(
    x = State_AllHomes.index[14:],
    y = State_AllHomes.values[14:],
    mode = 'lines',
    name = 'AllHomes'
)
trace6 = go.Scatter(
    x = State_CondoCoop.index[14:],
    y = State_CondoCoop.values[14:],
    mode = 'lines',
    name = 'CondoCoop'
)
trace7 = go.Scatter(
    x = State_DuplexTriplex.index[14:],
    y = State_DuplexTriplex.values[14:],
    mode = 'lines',
    name = 'DuplexTriplex'
)
trace8 = go.Scatter(
    x = State_SingleFamilyResidence.index[14:],
    y = State_SingleFamilyResidence.values[14:],
    mode = 'lines',
    name = 'SingleFamilyResidence'
)

data = [trace0, trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8]

layout = go.Layout(
      xaxis=dict(title='year'),
      yaxis=dict(title='Median listing Price$$'),
      title=('Median Listing Price over the years'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Overall
#    ---

# In[ ]:


v_features = State_house.iloc[:,14:23].columns
plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(2,5)
for i, cn in enumerate(State_house[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data = State_house)
    ax.set_title(str(cn)[19:])
    ax.set_ylabel(' ')


# > **We see an Increasing Trend in the listed prices for all types of Home in USA**

# Median **Rental Price PerSqft** for 
#      ----
# > 1Bedroom <br>
# 2Bedroom<br>
# 3Bedroom<br>
# 4Bedroom<br>
# 5BedroomOrMore<br>
# AllHomes<br>
# CondoCoop<br>
# DuplexTriplex<br>
# SingleFamilyResidence<br>
# MultiFamilyResidence5PlusUnits<br>
# Studio<br>

# Year wise breakup
#      ----

# In[ ]:


v_features = State_house.iloc[:,29:40].columns
plt.figure(figsize=(15,32))
gs = gridspec.GridSpec(12,1)
for i, cn in enumerate(State_house[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data = State_house)
    sns.boxplot(x='Century',y=cn,data=State_house[8243:])
    ax.set_xlabel('')
    ax.set_title(str(cn)[25:])
    ax.set_ylabel(' ')


# Overall
#    ---

# In[ ]:


v_features = State_house.iloc[:,29:40].columns
plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(3,4)
for i, cn in enumerate(State_house[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data = State_house)
    ax.set_title(str(cn)[25:])
    ax.set_ylabel(' ')


# Median **Rental Price ** for 
#      ----
# > 1Bedroom <br>
# 2Bedroom<br>
# 3Bedroom<br>
# 4Bedroom<br>
# 5BedroomOrMore<br>
# AllHomes<br>
# CondoCoop<br>
# DuplexTriplex<br>
# SingleFamilyResidence<br>
# MultiFamilyResidence5PlusUnits<br>
# Studio<br>

# Year wise breakup
#    ---
#    > Boxplot <br>
#    Lineplot

# In[ ]:


v_features = State_house.iloc[:,40:51].columns
plt.figure(figsize=(15,32))
gs = gridspec.GridSpec(11,1)
for i, cn in enumerate(State_house[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data = State_house)
    sns.boxplot(x='Century',y=cn,data=State_house[8243:])
    ax.set_xlabel('')
    ax.set_title(str(cn)[18:])
    ax.set_ylabel(' ')


# In[ ]:


State_scatter = State_house
State_scatter['Century'] = State_scatter.Date.dt.year
State_1Bedroom = State_scatter.groupby(['Century'])['MedianRentalPrice_1Bedroom'].mean()
State_2Bedroom = State_scatter.groupby(['Century'])['MedianRentalPrice_2Bedroom'].mean()
State_3Bedroom = State_scatter.groupby(['Century'])['MedianRentalPrice_3Bedroom'].mean()
State_4Bedroom = State_scatter.groupby(['Century'])['MedianRentalPrice_4Bedroom'].mean()
State_5BedroomOrMore = State_scatter.groupby(['Century'])['MedianRentalPrice_5BedroomOrMore'].mean()
State_AllHomes = State_scatter.groupby(['Century'])['MedianRentalPrice_AllHomes'].mean()
State_CondoCoop = State_scatter.groupby(['Century'])['MedianRentalPrice_CondoCoop'].mean()
State_DuplexTriplex = State_scatter.groupby(['Century'])['MedianRentalPrice_DuplexTriplex'].mean()
State_SingleFamilyResidence = State_scatter.groupby(['Century'])['MedianRentalPrice_SingleFamilyResidence'].mean()
State_MultiFamilyResidence5PlusUnits  = State_scatter.groupby(['Century'])['MedianRentalPrice_MultiFamilyResidence5PlusUnits'].mean()
State_Studio  = State_scatter.groupby(['Century'])['MedianRentalPrice_Studio'].mean()

trace0 = go.Scatter(
    x = State_1Bedroom.index[14:],
    y = State_1Bedroom.values[14:],
    mode = 'lines',
    name = '1Bedroom'
)
trace1 = go.Scatter(
    x = State_2Bedroom.index[14:],
    y = State_2Bedroom.values[14:],
    mode = 'lines',
    name = '2Bedroom'
)
trace2 = go.Scatter(
    x = State_3Bedroom.index[14:],
    y = State_3Bedroom.values[14:],
    mode = 'lines',
    name = '3Bedroom'
)
trace3 = go.Scatter(
    x = State_4Bedroom.index[14:],
    y = State_4Bedroom.values[14:],
    mode = 'lines',
    name = '4Bedroom'
)
trace4 = go.Scatter(
    x = State_5BedroomOrMore.index[14:],
    y = State_5BedroomOrMore.values[14:],
    mode = 'lines',
    name = '5BedroomOrMore'
)
trace5 = go.Scatter(
    x = State_AllHomes.index[14:],
    y = State_AllHomes.values[14:],
    mode = 'lines',
    name = 'AllHomes'
)
trace6 = go.Scatter(
    x = State_CondoCoop.index[14:],
    y = State_CondoCoop.values[14:],
    mode = 'lines',
    name = 'CondoCoop'
)
trace7 = go.Scatter(
    x = State_DuplexTriplex.index[14:],
    y = State_DuplexTriplex.values[14:],
    mode = 'lines',
    name = 'DuplexTriplex'
)
trace8 = go.Scatter(
    x = State_SingleFamilyResidence.index[14:],
    y = State_SingleFamilyResidence.values[14:],
    mode = 'lines',
    name = 'SingleFamilyResidence'
)
trace9 = go.Scatter(
    x = State_MultiFamilyResidence5PlusUnits.index[14:],
    y = State_MultiFamilyResidence5PlusUnits.values[14:],
    mode = 'lines',
    name = 'MultiFamilyResidence5PlusUnits'
)
trace10 = go.Scatter(
    x = State_Studio.index[14:],
    y = State_Studio.values[14:],
    mode = 'lines',
    name = 'Studio'
)

data = [trace0, trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10]

layout = go.Layout(
      xaxis=dict(title='year'),
      yaxis=dict(title='Median Rental Price$$'),
      title=('Median Rental Price over the years'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Overall
#   ---

# In[ ]:


v_features = State_house.iloc[:,40:51].columns
plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(3,4)
for i, cn in enumerate(State_house[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data = State_house)
    ax.set_title(str(cn)[18:])
    ax.set_ylabel(' ')


# > **We see an Increasing Trend in the Rental prices for all types of Home in USA except for Studio flats**
#     

# In[ ]:


states = {
        'Alaska' :'AK',
'Alabama':'AL',
'Arkansas' :'AR',
'AmericanSamoa' :'AS',
'Arizona' :'AZ',
'California' :'CA',
'Colorado' :'CO',
'Connecticut' :'CT',
'DistrictofColumbia' :'DC',
'Delaware' :'DE',
'Florida' :'FL',
'Georgia' :'GA',
'Guam' :'GU',
'Hawaii' :'HI',
'Iowa' :'IA',
'Idaho' :'ID',
'Illinois' :'IL',
'Indiana' :'IN',
'Kansas' :'KS',
'Kentucky' :'KY',
'Louisiana' :'LA',
'Massachusetts' :'MA',
'Maryland' :'MD',
'Maine' :'ME',
'Michigan' :'MI',
'Minnesota' :'MN',
'Missouri' :'MO',
'NorthernMarianaIslands' :'MP',
'Mississippi' :'MS',
'Montana' :'MT',
'National' :'NA',
'NorthCarolina' :'NC',
'NorthDakota' :'ND',
'Nebraska' :'NE',
'NewHampshire' :'NH',
'NewJersey' :'NJ',
'NewMexico' :'NM',
'Nevada' :'NV',
'NewYork' :'NY',
'Ohio' :'OH',
'Oklahoma' :'OK',
'Oregon' :'OR',
'Pennsylvania' :'PA',
'PuertoRico' :'PR',
'RhodeIsland' :'RI',
'SouthCarolina' :'SC',
'SouthDakota' :'SD',
'Tennessee':'TN',
'Texas' :'TX',
'Utah' :'UT',
'Virginia' :'VA',
'VirginIslands' :'VI',
'Vermont' :'VT',
'Washington' :'WA',
'Wisconsin' :'WI',
'WestVirginia' :'WV',
'Wyoming':'WY'

}

Months = {
1 :'January',
2 :'February',
3 :'March',
 4 :'April',
5 :'May',
 6 :'June',
 7 :'July',
8 :'August',
9 :'September',
10 :'October',
11 :'November',
12 :'December',
}

Traces = {
1 :'trace0',
2 :'trace1',
3 :'trace2',
 4 :'trace3',
5 :'trace4',
 6 :'trace5',
 7 :'trace6',
8 :'trace7',
9 :'trace8',
10 :'trace9',
11 :'trace10',
12 :'trace11',
}


DF_singlefamily = {
1 : 'State_SingleFamilyResidence1',
2 : 'State_SingleFamilyResidence2',
3 : 'State_SingleFamilyResidence3',
4 : 'State_SingleFamilyResidence4',
5 : 'State_SingleFamilyResidence5',
6 : 'State_SingleFamilyResidence6',
7 : 'State_SingleFamilyResidence7',
8 : 'State_SingleFamilyResidence8',
9 : 'State_SingleFamilyResidence9',
10 : 'State_SingleFamilyResidence10',
11 : 'State_SingleFamilyResidence11',
12 : 'State_SingleFamilyResidence12'
}
    


# Are you looking for Single Family Residence in USA . Check below trend in prices for Year 2010 and 2017
#       -----
# > **Median of the listed price (or asking price) for homes listed on Zillow**
# 
# 

# In[ ]:


State_raw_house = State_house.groupby(['RegionName', State_house.Date.dt.year])['MedianListingPrice_SingleFamilyResidence'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['RegionName',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['RegionName']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)


# In[ ]:


fig, ax = plt.subplots(figsize=(13, 13))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('Home Value')
ax.set_ylabel('States')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (430000,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 Home Value', (150000,33),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 Home Value', (150000,34),fontsize=14,rotation=0,color='Blue');

ax.annotate('with Percent increase/Decrease', (150000,32),fontsize=14,rotation=0,color='Brown');


# In[ ]:


State_scatter = State_house
State_scatter['Century'] = State_scatter.Date.dt.year
j = 1
data = []
for i in range(12):
     family_month = State_scatter[State_scatter.Date.dt.month == j]
     DF_singlefamily[j] = family_month.groupby(['Century'])['MedianListingPrice_SingleFamilyResidence'].mean()
     Traces[j] = go.Scatter(
         x = DF_singlefamily[j].index[14:],
         y = DF_singlefamily[j].values[14:],
         mode = 'lines',
         name = Months[j]
     )
     data.append(Traces[j]) 
     j = j + 1

layout = go.Layout(
      xaxis=dict(title='year'),
      yaxis=dict(title='Median listing Price$$'),
      title=('Single Family Residence value over the years in every available month'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


State_name=[]
for pc in Feature :
    State_name.append(states[pc])

data = dict(type = 'choropleth',
           locations = State_name,
           locationmode = "USA-states" ,
           colorscale = 'YIOrRed',
            text = Feature,
            marker = dict (line = dict(color = 'rgb(255,255,255)',width=2)),
           z = total,
           colorbar = {'title':'Home Value'})

layout = dict(title = 'Median of the listed price (or asking price) for homes listed on Zillow for year 2017',
         geo=dict(scope = 'usa',showlakes = True,lakecolor='rgb(85,173,240)')) 

choromap2 = go.Figure(data = [data],layout=layout)
iplot(choromap2)


# > **A smoothed seasonally adjusted measure of the median estimated home value across a given region**

# In[ ]:


State_raw_house = State_house.groupby(['RegionName', State_house.Date.dt.year])['ZHVI_SingleFamilyResidence'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['RegionName',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['RegionName']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)


# In[ ]:


fig, ax = plt.subplots(figsize=(13, 13))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('Home Value')
ax.set_ylabel('States')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (720000,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 Home Value', (100000,50),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 Home Value', (100000,51),fontsize=14,rotation=0,color='Blue');

ax.annotate('with Percent increase/Decrease', (100000,49),fontsize=14,rotation=0,color='Brown');


# In[ ]:


State_scatter = State_house
State_scatter['Century'] = State_scatter.Date.dt.year
j = 1
data = []
for i in range(12):
     family_month = State_scatter[State_scatter.Date.dt.month == j]
     DF_singlefamily[j] = family_month.groupby(['Century'])['ZHVI_SingleFamilyResidence'].mean()
     Traces[j] = go.Scatter(
         x = DF_singlefamily[j].index[14:],
         y = DF_singlefamily[j].values[14:],
         mode = 'lines',
         name = Months[j]
     )
     data.append(Traces[j]) 
     j = j + 1

layout = go.Layout(
      xaxis=dict(title='year'),
      yaxis=dict(title='Median listing Price$$'),
      title=('Single Family Residence value over the years in every available month'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


State_name=[]
for pc in Feature :
    State_name.append(states[pc])

data = dict(type = 'choropleth',
           locations = State_name,
           locationmode = "USA-states" ,
           colorscale = 'YIOrRed',
            text = Feature,
            marker = dict (line = dict(color = 'rgb(255,255,255)',width=2)),
           z = total,
           colorbar = {'title':'Home Value'})

layout = dict(title = 'Median estimated Home value for year 2017',
         geo=dict(scope = 'usa',showlakes = True,lakecolor='rgb(85,173,240)')) 

choromap2 = go.Figure(data = [data],layout=layout)
iplot(choromap2)


# > **A smoothed seasonally adjusted measure of the median estimated market rate rent across a given region and housing type**

# In[ ]:


State_raw_house = State_house.groupby(['RegionName', State_house.Date.dt.year])['Zri_SingleFamilyResidenceRental'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['RegionName',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['RegionName']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)


# In[ ]:


fig, ax = plt.subplots(figsize=(13, 13))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('Market Rate rent')
ax.set_ylabel('States')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (2752,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 Market Rate Rent', (750,41),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 Market Rate Rent', (750,42),fontsize=14,rotation=0,color='Blue');

ax.annotate('with Percent increase/Decrease', (750,40),fontsize=14,rotation=0,color='Brown');


# In[ ]:


State_scatter = State_house
State_scatter['Century'] = State_scatter.Date.dt.year
j = 1
data = []
for i in range(12):
     family_month = State_scatter[State_scatter.Date.dt.month == j]
     DF_singlefamily[j] = family_month.groupby(['Century'])['Zri_SingleFamilyResidenceRental'].mean()
     Traces[j] = go.Scatter(
         x = DF_singlefamily[j].index[14:],
         y = DF_singlefamily[j].values[14:],
         mode = 'lines',
         name = Months[j]
     )
     data.append(Traces[j]) 
     j = j + 1

layout = go.Layout(
      xaxis=dict(title='year'),
      yaxis=dict(title='Median listing Price$$'),
      title=('Single Family Residence market rent over the years in every available month'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


State_name=[]
for pc in Feature :
    State_name.append(states[pc])

data = dict(type = 'choropleth',
           locations = State_name,
           locationmode = "USA-states" ,
           colorscale = 'YIOrRed',
            text = Feature,
            marker = dict (line = dict(color = 'rgb(255,255,255)',width=2)),
           z = total,
           colorbar = {'title':'Mareket Rate rent'})

layout = dict(title = 'Median estimated market rate rent 2017',
         geo=dict(scope = 'usa',showlakes = True,lakecolor='rgb(85,173,240)')) 

choromap2 = go.Figure(data = [data],layout=layout)
iplot(choromap2)


# House prices in Different Tiers 
#     ----
#    > Top Tier <br>
#    Middle Tier<br>
#    Bottom Tier
#    

# In[ ]:


State_scatter = State_house
State_scatter['Century'] = State_scatter.Date.dt.year
State_Bottom = State_scatter.groupby(['Century'])['ZHVI_BottomTier'].mean()
State_Middle = State_scatter.groupby(['Century'])['ZHVI_MiddleTier'].mean()
State_Top = State_scatter.groupby(['Century'])['ZHVI_TopTier'].mean()

trace2 = go.Scatter(
    x = State_Bottom.index[14:],
    y = State_Bottom.values[14:],
    mode = 'lines',
    name = 'Bottom Tier'
)
trace1 = go.Scatter(
    x = State_Middle.index[14:],
    y = State_Middle.values[14:],
    mode = 'lines',
    name = 'Middle Tier'
)
trace0 = go.Scatter(
    x = State_Top.index[14:],
    y = State_Top.values[14:],
    mode = 'lines',
    name = 'Top Tier'
)

data = [trace0, trace1,trace2]

layout = go.Layout(
      xaxis=dict(title='years'),
      yaxis=dict(title='Median Price$$'),
      title=('House prices in different Tiers over the years'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# **Gap between Top and Middle tier is huge and constant over the years .** <br>
# Lets find out the regions in which these Top Tier houses are found.

# In[ ]:


State_raw_house = State_house.groupby(['RegionName', State_house.Date.dt.year])['ZHVI_TopTier'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['RegionName',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['RegionName']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)


# In[ ]:


fig, ax = plt.subplots(figsize=(13, 13))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('House Prices')
ax.set_ylabel('States')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (1000000,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 House Prices', (200000,50),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 House Prices', (200000,51),fontsize=14,rotation=0,color='Blue');

ax.annotate('with Percent increase/Decrease', (200000,49),fontsize=14,rotation=0,color='Brown');


# In[ ]:


State_name=[]
for pc in Feature :
    State_name.append(states[pc])

data = dict(type = 'choropleth',
           locations = State_name,
           locationmode = "USA-states" ,
           colorscale = 'YIOrRed',
            text = Feature,
            marker = dict (line = dict(color = 'rgb(255,255,255)',width=2)),
           z = total,
           colorbar = {'title':'House Prices'})

layout = dict(title = 'House prices of Top tier category in 2017 from different regions',
         geo=dict(scope = 'usa',showlakes = True,lakecolor='rgb(85,173,240)')) 

choromap2 = go.Figure(data = [data],layout=layout)
iplot(choromap2)


# TIme to get our hand dirty with City_time_series dataset and to get the state codes we need load cities_crosswalk table here.
# 

# In[ ]:


City_house = pd.read_csv("../input/City_time_series.csv", parse_dates=['Date'])
City_crosswalk = pd.read_csv("../input/cities_crosswalk.csv")
City_house['Century'] = City_house.Date.dt.year


# Merge with Left join  (City_house dataframe at the left )

# In[ ]:


City_State_house = pd.merge(City_house, City_crosswalk, how='left', left_on='RegionName', right_on='Unique_City_ID')


# Quickly check whether you merge was successfull with crosswalk table 

# In[ ]:


City_State_house.head()


# In[ ]:


'''
columns = State_house.columns
n = 1
for x in columns : 
  print(x) 
  print (n)
  n = n + 1
''';


# Lets analyse the Listed prices for SingleFamilyResidence in California ,New York and Hawaii

# In[ ]:


City_State_house_filter = City_State_house[City_State_house.State == 'CA']
State_raw_house = City_State_house_filter.groupby(['City', City_State_house_filter.Date.dt.year])['MedianListingPrice_SingleFamilyResidence'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['City',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['City']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)


fig, ax = plt.subplots(figsize=(13,6))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('Home Value')
ax.set_ylabel('Cities')
ax.set_title('Cities in California')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (4900000,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 Home Value', (3000000,3.5),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 Home Value', (3000000,4.5),fontsize=14,rotation=0,color='Blue');

ax.annotate('with % increase/Decrease', (3000000,2.5),fontsize=14,rotation=0,color='Brown');


# In[ ]:


#Feature[55:]
#total[55:]
data = pd.DataFrame({
   'lon':[33.8, 33.74, 33.8, 33.02],
   'lat':[-118.0, -118.38, -118.39, -117.20],
   'name':['Los Alamitos', 'Rancho Palos Verdes', 'Palos Verdes Estates',
       'Rancho Santa Fe'],
   'value':[ 1042500.0 ,  1423173.5,  2260362.5,  3254937.5]
})
m = folium.Map(location=[36.77,-119.41], tiles="Mapbox Bright", zoom_start=6)
for i in range(0,len(data)):
   folium.Circle(
      location=[data.iloc[i]['lon'], data.iloc[i]['lat']],
      popup=data.iloc[i]['name'],
      radius=data.iloc[i]['value']*0.02,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)
m


# In[ ]:


City_State_house_filter = City_State_house[City_State_house.State == 'NY']
State_raw_house = City_State_house_filter.groupby(['City', City_State_house_filter.Date.dt.year])['MedianListingPrice_SingleFamilyResidence'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['City',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['City']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)


idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)

Feature = Feature[125:]
total = total[125:]
weightage = weightage[125:]
percent = percent[125:]
mid_pos  = mid_pos[125:]
size = size[125:]
    

fig, ax = plt.subplots(figsize=(13, 13))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('Home Value')
ax.set_ylabel('Cities')
ax.set_title('Top 50 Cities in New York')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (2500000,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 Home Value', (1500000,132),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 Home Value', (1500000,134),fontsize=14,rotation=0,color='Blue');

ax.annotate('with % increase/Decrease', (1500000,130),fontsize=14,rotation=0,color='Brown');


# In[ ]:


#Feature[55:]
#total[55:]
data = pd.DataFrame({
   'lon':[40.8245, 40.79, 40.8, 40.98, 40.87],
   'lat':[-72.66, -73.6, -72.56, -73.68, -73.51],
   'name':['Westhampton', 'East Hills', 'Southampton', 'Rye', 'Oyster Bay Cove'],
   'value':[1015937.1875,1323541.5, 2162875.0 , 2547375.0,1680062.5]
})
m = folium.Map(location=[40,-72], tiles="Mapbox Bright", zoom_start=6)
for i in range(0,len(data)):
   folium.Circle(
      location=[data.iloc[i]['lon'], data.iloc[i]['lat']],
      popup=data.iloc[i]['name'],
      radius=data.iloc[i]['value']*0.02,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)
m


# In[ ]:


City_State_house_filter = City_State_house[City_State_house.State == 'HI']
State_raw_house = City_State_house_filter.groupby(['City', City_State_house_filter.Date.dt.year])['MedianListingPrice_SingleFamilyResidence'].mean().unstack()
State_raw_house.columns.name = None      
State_raw_house = State_raw_house.reset_index()  
State_raw_house = State_raw_house[['City',2010,2011,2012,2013,2014,2015,2016,2017]]
State_raw_house = State_raw_house.dropna()
Feature = State_raw_house['City']
weightage = State_raw_house[2010]
total = State_raw_house[2017]
percent =  ((State_raw_house[2017] - State_raw_house[2010]) /State_raw_house[2010])*100
mid_pos = (State_raw_house[2010] + State_raw_house[2017]) / 2
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
percent = np.array(percent)
mid_pos  = np.array(mid_pos)

idx = weightage.argsort()
Feature, total, percent, mid_pos, weightage = [np.take(x, idx) for x in [Feature, total, percent, mid_pos , weightage]]

s = 1
size=[]
for i, cn in enumerate(weightage):
     s = s + 1        
     size.append(s)


fig, ax = plt.subplots(figsize=(13, 4))
ax.scatter(total,size,marker="o", color="lightBlue", s=size, linewidths=10)
ax.scatter(weightage,size,marker="o", color="LightGreen", s=size, linewidths=10)
ax.set_xlabel('Home Value')
ax.set_ylabel('Cities')
ax.set_title('Cities in Hawaii')
ax.spines['right'].set_visible(False)
ax.grid()

for i, txt in enumerate(Feature):
      ax.annotate(txt, (1000000,size[i]),fontsize=12,rotation=0,color='Red')
      ax.annotate('.', xy=(total[i], size[i]), xytext=(weightage[i], size[i]),
            arrowprops=dict(facecolor='LightGreen', shrink=0.06),
            )
for i, pct in enumerate(percent):
     ax.annotate(str(pct)[0:4], (mid_pos[i],size[i]),fontsize=12,rotation=0,color='Brown')

ax.annotate('2010 Home Value', (200000,5.2),fontsize=14,rotation=0,color='Green')
ax.annotate('2017 Home Value', (200000,5.5),fontsize=14,rotation=0,color='Blue');

ax.annotate('with Percent increase/Decrease', (200000,5),fontsize=14,rotation=0,color='Brown');


# In[ ]:


#Feature[55:]
#total[55:]
data = pd.DataFrame({
   'lon':[19.49, 21.33, 21.38,21.39 ,22.22],
   'lat':[-154.95, -158.0, -158.00,157.79 ,-159.48],
   'name':['Pahoa', 'Kapolei', 'Waipahu', 'Kaneohe', 'Princeville'],
   'value':[201650.0  ,   704618.75,   689125.0  ,   973625.0  ,  1174500.0 ]
})
m = folium.Map(location=[19.89,-155.58], tiles="Mapbox Bright", zoom_start=7)
for i in range(0,len(data)):
   folium.Circle(
      location=[data.iloc[i]['lon'], data.iloc[i]['lat']],
      popup=data.iloc[i]['name'],
      radius=data.iloc[i]['value']*0.02,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)
m

