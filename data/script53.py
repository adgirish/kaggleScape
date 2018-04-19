
# coding: utf-8

# ![Image](https://www.tilburguniversity.edu/upload/c67b70dd-9a4f-499b-82e0-0aee20dfe12a_jads%20logo.png)
# 
# _____________________________________________________________________________________________________
# <center> Problem statement </center> 
# <center> Preventing and reacting on terrorist attacks is a resource intense operation. Suboptimal resource allocation increases the chances of too little investment in risky areas. </center>
# <br>
#  <center> Introduction </center> 
# <center>In this notebook you can find descriptive statistics and predictive analysis we, a group of students, have made for the course Introduction to Data Science. We made descriptive statistics for attacks over time, the locations on maps, and the effectiveness and composition of different attacks.</center>

# ## Import data and packages

# In[ ]:


# import additional packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode()
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.svm       import SVC
from sklearn.ensemble  import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
import itertools

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# Import clean data

data_terrorism = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[1,2,7,8,9,13,14,16,26,27,28,34,68,69,71,81,98,100,101,103,104,105])
data_terrorism.info()


# ## Explore dataset

# In[ ]:


# Create copies of the original dataset for the predictive part

data1 = data_terrorism
data2 = data_terrorism
data3 = data_terrorism
data4 = data_terrorism


# In[ ]:


# Data head and drop NaN for nkill and nwound (variables of interest)

data_terrorism.dropna(subset=['nkill'])
data_terrorism.dropna(subset=['nwound'])
data_terrorism.head()


# ### **Which areas in the world have suffered the most from terrorist attacks?**

# In[ ]:


# Create plot

plt.figure(figsize=(30,16))

var1 = data_terrorism[(data_terrorism.nkill>=0)&(data_terrorism.nkill <=5)] 
var2 = data_terrorism[(data_terrorism.nkill>5)&data_terrorism.nkill <=10]
var3 = data_terrorism[data_terrorism['nkill'] >10]

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

# x, y = m(list(asia["longitude"].astype("float")), list(asia["latitude"].astype(float)))
# m.plot(x, y, "go", markersize = 6, alpha = 0.8, color = "#0000FF", label = "Asia")

x, y = m(list(var1["longitude"].astype(float)), list(var1["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.9, color = 'green')

x, y = m(list(var2["longitude"].astype(float)), list(var2["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.2, color = 'yellow')

x, y = m(list(var3["longitude"].astype(float)), list(var3["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.2, color = 'red')


plt.title('Global Terror Attacks (1970-2015) - number of people killed', fontsize=35)
plt.legend(handles=[mpatches.Patch(color='green', label = "< 6 kills"),
                    mpatches.Patch(color='yellow',label='6 - 10 kills'), mpatches.Patch(color='red',label='> 10 kills')],fontsize=30, markerscale = 5)
    
plt.show()


# In[ ]:


# Create plot

plt.figure(figsize=(30,16))

var4 = data_terrorism[(data_terrorism.nwound>0)&(data_terrorism.nwound <=10)] 
var5 = data_terrorism[(data_terrorism.nwound>10)&(data_terrorism.nwound <=25)]
var6 = data_terrorism[data_terrorism['nwound'] >25]

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

# x, y = m(list(asia["longitude"].astype("float")), list(asia["latitude"].astype(float)))
# m.plot(x, y, "go", markersize = 6, alpha = 0.8, color = "#0000FF", label = "Asia")

x, y = m(list(var4["longitude"].astype(float)), list(var4["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.5, color = 'green')

x, y = m(list(var5["longitude"].astype(float)), list(var5["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.9, color = 'yellow')

x, y = m(list(var6["longitude"].astype(float)), list(var6["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.2, color = 'red')

plt.title('Global Terror Attacks (1970-2015) - number of people wounded', fontsize=35)
plt.legend(handles=[mpatches.Patch(color='green', label = "1 - 10 wounded"),
                    mpatches.Patch(color='yellow',label='10 - 25 wounded'), mpatches.Patch(color='red',label='> 25 wounded')],fontsize=30, markerscale = 5, loc=3)
plt.show()


# In[ ]:


# Create plot

plt.figure(figsize=(30,16))

var7 = data_terrorism[(data_terrorism.propextent==3)] 
var8 = data_terrorism[(data_terrorism.propextent==2)]
var9 = data_terrorism[(data_terrorism.propextent==1)]

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()

# x, y = m(list(asia["longitude"].astype("float")), list(asia["latitude"].astype(float)))
# m.plot(x, y, "go", markersize = 6, alpha = 0.8, color = "#0000FF", label = "Asia")

x, y = m(list(var7["longitude"].astype(float)), list(var7["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = 'green', label = "1 - 3 kills")

x, y = m(list(var8["longitude"].astype(float)), list(var8["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = 'yellow', label = "4 - 5 kills")

x, y = m(list(var9["longitude"].astype(float)), list(var9["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 4, alpha = 0.8, color = 'red', label = "> 5 kills")


plt.title('Global Terror Attacks (1970-2015) - Property damage', fontsize=35)
plt.legend(handles=[mpatches.Patch(color='green', label = "< $1 million"),
                    mpatches.Patch(color='yellow',label='$ 1 million - $1 billion'), mpatches.Patch(color='red',label='> $ 1 billion')],fontsize=30, markerscale = 5, loc=3)
plt.show()


# The maps above shows all terrorist attacks per location, where the color is determined by the number of deaths, number of wounded and property damage of the attack.   
# <br>
# The knowledge gained is:
# * A high occurence of terror attacks with more than 10 fatalities is shown in the Middle East, pakistan/afghanistan, the center of Africa and the North of South America and Mexico.
# * A high occurence of terror attacks with more than 25 wounded is shown in the middle east and pakistan/afghanistan
# * Property damage with more than $1 billion of damages are reported only in the relative richest western countries
# * Attacks having damage between 1 million and 1 billion is in and near Colombia
# * Large countries that have been affected least by terrorism attacks are Russia, Greeenland,  Australia, Mongolia and Canada. These countries have in common that they are sparsely populated.

# ### **How has the number of terrorist attacks per region evolved over time ?**

# In[ ]:


# Group data per year for the world

terror_peryear_world = np.asarray(data_terrorism.groupby('iyear').iyear.count())
terror_years = np.arange(1970, 2016)


# In[ ]:


# Create plot

trace0 = [go.Scatter(
         x = terror_years,
         y = terror_peryear_world,
         mode = 'lines',
         line = dict(
             color = 'rgb(240, 140, 45)',
             width = 3),
        name = 'World'
         )]

layout = go.Layout(
         title = 'Terrorist Attacks by Year for the world (1970-2015)',
         xaxis = dict(
             rangeslider = dict(thickness = 0.05),
             showline = True,
             showgrid = False
         ),
        yaxis = dict(
             range = [0.1, 17500],
             showline = True,
             showgrid = False)
         )

figure = dict(data = trace0, layout = layout)
iplot(figure)


# In[ ]:


# Create distinct lines for the different regions

data_north_america = data_terrorism[(data_terrorism.region == 1) |  (data_terrorism.region == 2)]
data_asia = data_terrorism[(data_terrorism.region == 4) | (data_terrorism.region == 5) | (data_terrorism.region == 6) | (data_terrorism.region == 7)]
data_oceania = data_terrorism[(data_terrorism.region == 12)]
data_europe = data_terrorism[(data_terrorism.region == 8) | (data_terrorism.region == 9)]
data_south_america = data_terrorism[(data_terrorism.region == 3)]
data_middle_east_n_africa = data_terrorism[(data_terrorism.region == 10)]
data_sub_africa = data_terrorism[(data_terrorism.region == 11)]

peryear_north_america = np.asarray(data_north_america.groupby('iyear').iyear.count())
peryear_asia = np.asarray(data_asia.groupby('iyear').iyear.count())
peryear_oceania = np.asarray(data_oceania.groupby('iyear').iyear.count())
peryear_europe = np.asarray(data_europe.groupby('iyear').iyear.count())
peryear_south_america = np.asarray(data_south_america.groupby('iyear').iyear.count())
peryear_middle_east_n_africa = np.asarray(data_middle_east_n_africa.groupby('iyear').iyear.count())
peryear_sub_africa = np.asarray(data_sub_africa.groupby('iyear').iyear.count())


# In[ ]:


# Create plot

trace1 = go.Scatter(                             
         x = terror_years,
         y = peryear_north_america,
         mode = 'lines',
         line = dict(
             color = 'rgb(140, 140, 45)',
             width = 3),
        name = 'North- and Central America '
         )
trace2 = go.Scatter(                             
         x = terror_years,
         y = peryear_asia,
         mode = 'lines',
         line = dict(
             color = 'rgb(240, 40, 45)',
             width = 3),
        name = 'Asia'
         )
trace3 = go.Scatter(                             
         x = terror_years,
         y = peryear_oceania,
         mode = 'lines',
         line = dict(
             color = 'rgb(120, 120,120)',
             width = 3),
        name = 'Oceania'
         )
trace4 = go.Scatter(                             
         x = terror_years,
         y = peryear_europe,
         mode = 'lines',
         line = dict(
             color = 'rgb(0, 50, 72)',
             width = 3),
        name = 'Europe'
         )
trace5 = go.Scatter(                             
         x = terror_years,
         y = peryear_south_america,
         mode = 'lines',
         line = dict(
             color = 'rgb(27, 135 , 78)',
             width = 3),
        name = 'South America'
         )
trace6 = go.Scatter(                             
         x = terror_years,
         y = peryear_middle_east_n_africa,
         mode = 'lines',
         line = dict(
             color = 'rgb(230, 230, 230)',
             width = 3),
        name = 'Middle East and North Africa'
         )
trace7 = go.Scatter(                             
         x = terror_years,
         y = peryear_sub_africa,
         mode = 'lines',
         line = dict(
             color = 'rgb(238, 133, 26)',
             width = 3),
        name = 'Sub Saharan Africa'
         )

layout = go.Layout(
         title = 'Terrorist Attacks by Year per region (1970-2015)',
         xaxis = dict(
             rangeslider = dict(thickness = 0.05),
             showline = True,
             showgrid = False
         ),
         yaxis = dict(
             range = [0.1, 7500],
             showline = True,
             showgrid = False)
         )

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]

figure = dict(data = data, layout = layout)
iplot(figure)


# When looking at the first graph, it is interesting to see that from 1990 until around 2000 there was a low in terrorism in the world, and afterwards a huge increase. This might have something to do with 9/11 and the whole ware on terrorism from the USA.  
# 
# In the second graph, it is interesting to note that between 1970 and 1980 europe was the region with the most acts of terrorism, followed by North- and South america. After 2000 Asia, Middle east and Africa do have, by far, the most acts on terrorism. North- and South america have the lowest number of terrorist attacks, together with Oceania.

# ### **How many people died relatively per country?**

# In[ ]:


# Insert data about world population (average of a country between 1970 and 2015)

names_country = np.array(['Burundi', 'Comoros', 'Djibouti', 'Eritrea', 'Ethiopia', 'Kenya', 'Madagascar', 'Malawi', 
'Mauritius', 'Mozambique', 'Rwanda', 'Seychelles', 'Somalia', 'South Sudan', 'Uganda', 
'Tanzania', 'Zambia', 'Zimbabwe', 'Angola', 'Cameroon', 'Central African Republic', 'Chad', 
'Democratic Republic of the Congo', 'Equatorial Guinea', 'Gabon', 'Algeria', 'Egypt', 
'Libya', 'Morocco', 'Sudan', 'Tunisia', 'Western Sahara', 'Botswana', 'Lesotho', 'Namibia', 
'South Africa', 'Swaziland', 'Benin', 'Burkina Faso', 'Gambia', 'Ghana', 'Guinea', 
'Guinea-Bissau', 'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria', 'Senegal', 'Sierra Leone', 
'Togo', 'China', 'Hong Kong', 'Taiwan', 'North Korea', 'Japan', 'South Korea', 'Kazakhstan', 'Kyrgyzstan', 
'Tajikistan', 'Turkmenistan', 'Uzbekistan', 'Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Iran', 'Maldives', 
'Nepal', 'Pakistan', 'Sri Lanka', 'Brunei', 'Cambodia', 'Indonesia', 'Malaysia', 'Myanmar', 'Philippines', 
'Singapore', 'Thailand', 'Armenia', 'Azerbaijan', 'Bahrain', 'Cyprus', 'Georgia', 'Iraq', 'Israel', 'Jordan', 
'Kuwait', 'Lebanon', 'Qatar', 'Saudi Arabia', 'Syria', 'Turkey', 'United Arab Emirates', 'Yemen', 'Belarus', 
'Bulgaria', 'Czech Republic', 'Hungary', 'Poland', 'Moldova', 'Romania', 'Russia', 'Slovak Republic', 'Ukraine', 
'Denmark', 'Estonia', 'Finland', 'Iceland', 'Ireland', 'Latvia', 'Lithuania', 'Norway', 'Sweden', 'United Kingdom', 
'Albania', 'Andorra', 'Croatia', 'Greece', 'Italy', 'Malta', 'Montenegro', 'Portugal', 'Serbia', 'Slovenia', 'Spain', 
'Macedonia', 'Austria', 'Belgium', 'France', 'Germany', 'Luxembourg', 'Netherlands', 'Switzerland', 
'Antigua and Barbuda', 'Bahamas', 'Barbados', 'Cuba', 'Dominica', 'Dominican Republic', 'Grenada', 
'Guadeloupe', 'Haiti', 'Jamaica', 'Martinique', 'St. Kitts and Nevis', 'St. Lucia', 'Trinidad and Tobago',
'Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Mexico', 'Nicaragua', 'Panama', 'Argentina', 
'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Falkland Islands', 'French Guiana', 'Guyana', 'Paraguay', 
'Peru', 'Suriname', 'Uruguay', 'Venezuela', 'Canada', 'United States', 'Australia', 'New Zealand', 'Fiji', 
'New Caledonia', 'Papua New Guinea', 'Solomon Islands', 'Vanuatu', 'French Polynesia', 'Wallis and Futuna'])
 
avg_inhabitants = np.array([5929.10939130435, 461.982260869565, 568.661195652174, 3167.80902173913, 56660.6153478261, 26496.5380217391, 
13582.7968043478, 9863.82284782609, 1083.64836956522, 16245.7041956522, 7087.97323913044, 75.2548043478261, 
8030.06308695652, 6477.32982608696, 20852.8252608696, 29522.112326087, 9068.8527173913, 10345.934673913, 
14509.1303913043, 13196.8929347826, 3193.35939130435, 7325.28006521739, 41157.2444347826, 543.27952173913, 
1088.89243478261, 26797.3883043478, 61115.0571739131, 4508.86017391304, 25491.8166956522, 22802.6085217391, 
8361.28195652174, 268.547326086957, 1445.40223913043, 1638.97108695652, 1538.20356521739, 39382.7373695652, 
877.396826086957, 5890.58319565217, 10251.004326087, 1048.30447826087, 16443.0094782609, 7265.6877173913, 
1117.94847826087, 2553.27095652174, 10021.2752826087, 2348.29267391304, 9931.18865217391, 107011.648782609, 
8563.68869565217, 4493.72626086957, 4269.10167391304, 1163199.68545652, 5826.53819565218, 20129.2505652174, 
20663.6947391304, 122594.270434783, 43258.3279347826, 15532.5435869565, 4407.20291304348, 5482.9697826087, 
3862.84728260869, 21418.0874782609, 18216.3767391304, 112356.130934783, 531.21852173913, 921193.107956522, 
55578.0176956522, 243.8585, 20202.0932608696, 117190.784065217, 17287.3976304348, 277.343282608696, 10298.4438043478, 
187633.016391304, 19769.3171304348, 40896.353673913, 66855.7890652174, 3466.61245652174, 56205.7741304348, 
3069.41267391304, 7400.00323913044, 634.329739130435, 848.754391304348, 4804.98969565217, 20352.562, 5170.8377826087, 
4337.16436956522, 1951.00813043478, 3259.02689130435, 707.046130434783, 17364.4145434783, 13578.3613478261, 
56086.0281304348, 3142.34897826087, 14552.4723695652, 9718.86830434783, 8298.81045652174, 10303.6185869565, 
10318.2473913043, 37057.3202608696, 4111.80986956522, 21981.0171086957, 142224.572043478, 5193.58193478261, 
48784.3528913043, 5257.32304347826, 1425.63054347826, 5043.43595652174, 263.84202173913, 3770.52060869565, 
2406.43467391304, 3396.71195652174, 4381.9037826087, 8724.50784782609, 58607.5434782609, 2897.0577826087, 
56.5975217391304, 4516.65902173913, 10381.4900652174, 57283.226826087, 371.455673913043, 600.830913043478, 
10020.9863478261, 9132.0465, 1934.8962173913, 40429.5194130435, 1971.92363043478, 7937.46189130435, 10223.325673913, 
57694.2000434783, 79944.3875, 414.953630434783, 15191.1485, 6945.57619565217, 78.5750652173913, 270.924413043478, 
262.701282608696, 10518.807, 71.9289782608695, 7526.48147826087, 98.6522173913043, 389.146391304348, 7519.14154347826, 
2452.75043478261, 361.613913043478, 45.4052826086957, 141.484826086957, 1198.13836956522, 214.779152173913, 
3319.18826086957, 5281.94815217391, 10224.1756304348, 5526.89834782609, 89266.8574565218, 4320.28106521739, 
2656.76630434783, 33709.3748913043, 7361.63471739131, 153928.587804348, 13721.1711956522, 35549.3115434783, 
10898.9652391304, 2.35376086956522, 136.069326086956, 754.081152173913, 4498.26436956522, 22577.5376521739, 
434.704, 3136.78836956522, 21106.998673913, 28502.9505217391, 262097.535130435, 17685.9465652174, 3629.91191304348, 
735.87102173913, 186.23547826087, 4826.96276086956, 350.2555, 163.026043478261, 202.665760869565, 12.716347826087])    

inhabitants_data = pd.DataFrame({"country":names_country,"inhabitants":avg_inhabitants})


# In[ ]:


# merge inhabitants with grouped data

# Create nr. of attacks per country for the world
def create_df_grouped(data):
    dfout = pd.DataFrame({'country_txt':data['country_txt'].unique() ,
                          'region' : (sum(data['region'])/len(data['region'])),
                         'country': len(data['country']) })
    return dfout

attacks_world_grouped = data_terrorism.groupby('country_txt').apply(create_df_grouped)

# create extra country_txt variable for labeling purposes
attacks_world_grouped['country_txt2'] = attacks_world_grouped['country_txt']


# In[ ]:


# create arrays for the y values and change NaN to 0

data_terrorism['nkill'] = data_terrorism['nkill'].replace(np.nan, 0)
data_terrorism['nwound'] = data_terrorism['nwound'].replace(np.nan,0)


# In[ ]:


# create nr. of kills per country for the world

def create_df_grouped(data):
    dfout = pd.DataFrame({'country_txt':data['country_txt'].unique() , 
                          'region' : (sum(data['region'])/len(data['region'])), 
                         'country': sum(data['nkill']) })
    return dfout

nkill_world_grouped = data_terrorism.groupby('country_txt').apply(create_df_grouped)


# In[ ]:


# create nr. of wounded per country for the world

def create_df_grouped(data):
    dfout = pd.DataFrame({'country_txt':data['country_txt'].unique() , 
                          'region' : (sum(data['region'])/len(data['region'])), 
                         'country': sum(data['nwound']) })
    return dfout

nwound_world_grouped = data_terrorism.groupby('country_txt').apply(create_df_grouped)


# In[ ]:


# merge nr. of attacks and inhabitants

merged_data_nattacks = attacks_world_grouped.set_index('country_txt').join(inhabitants_data.set_index('country'))
merged_data_nattacks['nattacks_inhabitants'] = merged_data_nattacks.country / merged_data_nattacks.inhabitants 


# In[ ]:


# merge nr. of kills and inhabitants

merged_data_nkill = nkill_world_grouped.set_index('country_txt').join(inhabitants_data.set_index('country'))

# Create relative variable
merged_data_nkill['nkill_inhabitants'] = merged_data_nkill.country / merged_data_nkill.inhabitants


# In[ ]:


# merge nr. of woudned and inhabitants

merged_data_nwound = nwound_world_grouped.set_index('country_txt').join(inhabitants_data.set_index('country'))

# Create relative variable
merged_data_nwound['nwound_inhabitants'] = merged_data_nwound.country / merged_data_nwound.inhabitants


# In[ ]:


# Create subarrays for the different regions

# create x value: nr of attacks per country
merged_data_nattacks_north_america = merged_data_nattacks[(merged_data_nattacks.region == 1) |  (merged_data_nattacks.region == 2)]
merged_data_nattacks_asia = merged_data_nattacks[(merged_data_nattacks.region == 4) | (merged_data_nattacks.region == 5) | (merged_data_nattacks.region == 6) | (merged_data_nattacks.region == 7)]
merged_data_nattacks_oceania = merged_data_nattacks[(merged_data_nattacks.region == 12)]
merged_data_nattacks_europe = merged_data_nattacks[(merged_data_nattacks.region == 8) | (merged_data_nattacks.region == 9)]
merged_data_nattacks_south_america = merged_data_nattacks[(merged_data_nattacks.region == 3)]
merged_data_nattacks_east_n_africa = merged_data_nattacks[(merged_data_nattacks.region == 10)]
merged_data_nattacks_sub_africa = merged_data_nattacks[(merged_data_nattacks.region == 11)]

# create y value: succesfullness: nr. of kills, nr. of wounded
merged_data_nkill_north_america = merged_data_nkill[(merged_data_nkill.region == 1) |  (merged_data_nkill.region == 2)]
merged_data_nkill_asia = merged_data_nkill[(merged_data_nkill.region == 4) | (merged_data_nkill.region == 5) | (merged_data_nkill.region == 6) | (merged_data_nkill.region == 7)]
merged_data_nkill_oceania = merged_data_nkill[(merged_data_nkill.region == 12)]
merged_data_nkill_europe = merged_data_nkill[(merged_data_nkill.region == 8) | (merged_data_nkill.region == 9)]
merged_data_nkill_south_america = merged_data_nkill[(merged_data_nkill.region == 3)]
merged_data_nkill_east_n_africa = merged_data_nkill[(merged_data_nkill.region == 10)]
merged_data_nkill_sub_africa = merged_data_nkill[(merged_data_nkill.region == 11)]

merged_data_nwound_north_america = merged_data_nwound[(merged_data_nwound.region == 1) |  (merged_data_nwound.region == 2)]
merged_data_nwound_asia = merged_data_nwound[(merged_data_nwound.region == 4) | (merged_data_nwound.region == 5) | (merged_data_nwound.region == 6) | (merged_data_nwound.region == 7)]
merged_data_nwound_oceania = merged_data_nwound[(merged_data_nwound.region == 12)]
merged_data_nwound_europe = merged_data_nwound[(merged_data_nwound.region == 8) | (merged_data_nwound.region == 9)]
merged_data_nwound_south_america = merged_data_nwound[(merged_data_nwound.region == 3)]
merged_data_nwound_east_n_africa = merged_data_nwound[(merged_data_nwound.region == 10)]
merged_data_nwound_sub_africa = merged_data_nwound[(merged_data_nwound.region == 11)]


# In[ ]:


# Scatter plot with distinghuising between regions

trace1 = go.Scatter(                             
         x = (merged_data_nattacks_north_america.nattacks_inhabitants / 10),
         y = (merged_data_nkill_north_america.nkill_inhabitants / 10),
         mode = 'markers',
         marker = dict(
             color = 'rgb(140, 140, 45)',
             size = 5),
        text = merged_data_nattacks_north_america.country_txt2,
        name = 'North- and Central America '
         )
trace2 = go.Scatter(                             
         x = (merged_data_nattacks_asia.nattacks_inhabitants / 10),
         y = (merged_data_nkill_asia.nkill_inhabitants / 10),
         mode = 'markers',
         marker = dict(
             color = 'rgb(240, 40, 45)',
             size = 5),
        text = merged_data_nattacks_asia.country_txt2,
        name = 'Asia'
         )
trace3 = go.Scatter(                             
         x = (merged_data_nattacks_oceania.nattacks_inhabitants / 10),
         y = (merged_data_nkill_oceania.nkill_inhabitants / 10),
         mode = 'markers',
         marker = dict(
             color = 'rgb(120, 120,120)',
             size = 5),
        text = merged_data_nattacks_oceania.country_txt2,
        name = 'Oceania'
         )
trace4 = go.Scatter(                             
         x = (merged_data_nattacks_europe.nattacks_inhabitants / 10),
         y = (merged_data_nkill_europe.nkill_inhabitants / 10),
         mode = 'markers',
         marker = dict(
             color = 'rgb(0, 50, 72)',
             size = 5),
        text = merged_data_nattacks_europe.country_txt2,
        name = 'Europe'
         )
trace5 = go.Scatter(                             
         x = (merged_data_nattacks_south_america.nattacks_inhabitants / 10),
         y = (merged_data_nkill_south_america.nkill_inhabitants / 10),
         mode = 'markers',
         marker = dict(
             color = 'rgb(27, 135 , 78)',
             size = 5),
        text = merged_data_nattacks_south_america.country_txt2,
        name = 'South America'
         )
trace6 = go.Scatter(                             
         x = (merged_data_nattacks_east_n_africa.nattacks_inhabitants / 10),
         y = (merged_data_nkill_east_n_africa.nkill_inhabitants / 10),
         mode = 'markers',
         marker = dict(
             color = 'rgb(230, 230, 230)',
             size = 5),
        text = merged_data_nattacks_east_n_africa.country_txt2,
        name = 'Middle East and North Africa'
         )
trace7 = go.Scatter(                             
         x = (merged_data_nattacks_sub_africa.nattacks_inhabitants / 10),
         y = (merged_data_nkill_sub_africa.nkill_inhabitants / 10),
         mode = 'markers',
         marker = dict(
             color = 'rgb(238, 133, 26)',
             size = 5),
        text = merged_data_nattacks_sub_africa.country_txt2,
        name = 'Sub Saharan Africa'
         )

layout = dict(title = 'Number of attacks and number of kills per citizen',
              yaxis = dict(
                  zeroline = True,
                  title = 'Nr. of kills per citizen in percentages'),
              xaxis = dict(
                  zeroline = True,
                  title = 'Nr. of attacks per citizen in percentages'),
           )

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]

# Plot and embed in ipython notebook!
figure = dict(data = data, layout = layout)
iplot(figure)


# It is interesting to see, that when the number of attacks and the number of kills relative to the population is for the bulk of the countries almost the same. However, there are some countries with significantly higher numbers. 
# * The Falklan Islands do have a relatively high known number of attacks, but the *know* number of kills is zero, so either there are no kills known or nobody has been killed during the terrorist attacks.
# * Iraq is relatively the most dangerous country with the higher number of attacks and number of kills per citizen. On second and third place come Nicaragua and El Salvador.
# * When zooming into Europe, the most dangerous countries are United Kingdom, Croatia and Cyprus.

# ### **Fow which target type do terrorism attacks have the most impact?**

# In[ ]:


# Create subgroups

nr_targtype1 = np.asarray(data_terrorism.groupby('targtype1').targtype1.count())
nkill_targtype1 = np.asarray(data_terrorism.groupby('targtype1').nkill.sum())
average_nkill = np.divide(nkill_targtype1, nr_targtype1) 

targtype1_names = np.array(["Business","Gevernment","Police","Military","Abortion Related","Airports & Aircrafts", "Government (diplomaitc)", "Educational Institution", "Food or water supply","Journalists & Media","Maritime","NGO","Other","Private Citizens & property","Religious figures/institutions","Telecommunication","Terrorist/non-state militias","Tourists","Transportation","Unknown","Utilities","Violent politican parties"])
print(targtype1_names)

total_deaths = sum(nkill_targtype1)
average_nkill2 = np.divide(nkill_targtype1, total_deaths) 
average_nkill_kills = average_nkill2*100

nwound_targtype1 = np.asarray(data_terrorism.groupby('targtype1').nwound.sum())
average_nwound = np.divide(nwound_targtype1, nr_targtype1) 

total_wounded = sum(nwound_targtype1)
average_nwound2 = np.divide(nwound_targtype1, total_wounded) 
average_nwound_wounded = average_nwound2*100

propextent_targtype1 = np.asarray(data_terrorism.groupby('targtype1').propextent.sum())
average_propextent = np.divide(propextent_targtype1, nr_targtype1) 

total_property = sum(propextent_targtype1)
average_propextent2 = np.divide(propextent_targtype1, total_property) 
average_propextent_propextent = average_propextent2*100

# Create dataframe
targtype1_data = pd.DataFrame({"targtype1_names":targtype1_names,"nr_targtype1":nr_targtype1,"nkill_targtype1":nkill_targtype1,"average_nkill":average_nkill, "average_nkill_kills":average_nkill_kills,"average_nwound":average_nwound,"average_nwound_wounded":average_nwound_wounded,"average_propextent":average_propextent,"average_propextent_propextent":average_propextent_propextent})
targtype1_data.head()


# In[ ]:


#sort the dataframe from large to small

sorted_targtype1_data = targtype1_data.sort_values(by='average_nkill', ascending=0)
sorted1_targtype1_data = targtype1_data.sort_values(by='average_nkill_kills', ascending=0)
sorted2_targtype1_data = targtype1_data.sort_values(by='average_nwound', ascending=0)
sorted3_targtype1_data = targtype1_data.sort_values(by='average_nwound_wounded', ascending=0)
sorted4_targtype1_data = targtype1_data.sort_values(by='average_propextent', ascending=0)
sorted5_targtype1_data = targtype1_data.sort_values(by='average_propextent_propextent', ascending=0)


# In[ ]:


#Making barplots

ax = sns.barplot(y='targtype1_names',x='average_nkill', data=sorted_targtype1_data, color="#00035b", palette="Reds_r")
ax.set_xlabel("Average number deaths per target", size=10, alpha=1)
ax.set_ylabel("Targettype Names", size=10, alpha=1)
ax.set(xlim=(0, 5))
ax.set_title("The average number of deaths per attack given the target type", fontsize=12, alpha=1)
ax.tick_params(labelsize=10,labelcolor="black")
plt.show()

ax = sns.barplot(y='targtype1_names',x='average_nwound', data=sorted2_targtype1_data, color="#00035b", palette="Blues_r")
ax.set_xlabel("Average number of wounded people per attack", size=10, alpha=1)
ax.set_ylabel("Target type names", size=10, alpha=1)
ax.set(xlim=(0, 7))
ax.set_title("The average number of wounded people per attack given the target type", fontsize=12, alpha=1)
ax.tick_params(labelsize=10,labelcolor="black")
plt.show()

ax = sns.barplot(y='targtype1_names',x='average_propextent', data=sorted4_targtype1_data, color="#00035b", palette="Greens_r")
ax.set_xlabel("Average property damage per attack", size=10, alpha=1)
ax.set_ylabel("Targettype names", size=10, alpha=1)
ax.set(xlim=(0, 3))
ax.set_title("The average extent of property damage per target type", fontsize=12, alpha=1)
ax.tick_params(labelsize=10,labelcolor="black")
plt.show()

#Making donut charts
fig = {
  "data": [
    {
      "values": average_nkill_kills,
      "labels": targtype1_names
        ,
    "text":"Property Damage",
      "textposition":"inside",
      "domain": {"x": [0, .30]},
      "name": "",
      "hoverinfo":"label+percent+name",
          "hole": .4,
      "type": "pie"
    },     
      {
      "values": average_nwound_wounded,
      "labels": targtype1_names
        ,
    "text":"nkill",
      "textposition":"inside",
      "domain": {"x": [.35, .65]},
      "name": "",
      "hoverinfo":"label+percent+name",
          "hole": .4,
      "type": "pie"
    },     
    {
      "values": average_propextent_propextent,
      "labels":  targtype1_names
        ,
      "text":"Nwound",
      "textposition":"inside",
      "domain": {"x": [.70, 1]},
      "name": "",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Share of kills wounded and property per Target type ",
        "showlegend" : False,
        "annotations": [
            {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Killed",
                "x": 0.1275,
                "y": 0.5
            },
             {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Wounded",
                "x": 0.50,
                "y": 0.5
            },
            {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Property",
                "x": 0.885,
                "y": 0.5
            },
        ]
    }
}
iplot(fig, filename='donut')


# The Bar Charts show the average people killed/wounded/poroperty damage for each individual target type. In other words, it shows which targets come with the most amount of damage, which will later be used as a form of success of an attack.
# 
# Whereas the "donut" charts show the percentual contribution of each target type of the total kills/wounded/property damage. Which means for example that of all the people killed during terrorist attacks, 25,2% of those were casualties during an attack with a militarty target.
# 
# It can be seen that:
# * Private citizens, military and police combined have a total share in number of kills of nearly three quarters. However, the share in property damage is more evenly distributed over the different target types.
# * Military has the highest average number of deaths per attacks, but it has a relatively low number of wounded per attack compared to the other target types.
# * Terrorism attacks kill more defence personel (police and military) than private citizens.
# * Defence organizations suffer more property damage then private citizens.
# * When looking at the whole public sector, compared to the private sector. The public sector does suffer more than the private sector.

# ### **Fow which attack type do terrorism attacks have the most impact?**

# In[ ]:


# Create subgroups

nr_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').attacktype1.count())
nkill_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').nkill.sum())
average_nkill = np.divide(nkill_attacktype1, nr_attacktype1) 

attacktype1_names = np.array(['Assassination','Armed Attack','Bombing/Explosion','Hijacking','Hostage Taking barricade incident','Hostage Taking kidnapping','Facility/Infrastructure Attack','Unarmed Assault','Unknown'])
print(attacktype1_names)

total_deaths = sum(nkill_attacktype1)
average_nkill2 = np.divide(nkill_attacktype1, total_deaths) 
average_nkill_kills = average_nkill2*100

nwound_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').nwound.sum())
average_nwound = np.divide(nwound_attacktype1, nr_attacktype1) 

total_wounded = sum(nwound_attacktype1)
average_nwound2 = np.divide(nwound_attacktype1, total_wounded) 
average_nwound_wounded = average_nwound2*100

propextent_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').propextent.sum())
average_propextent = np.divide(propextent_attacktype1, nr_attacktype1) 
total_property = sum(propextent_attacktype1)
average_propextent2 = np.divide(propextent_attacktype1, total_property) 
average_propextent_propextent = average_propextent2*100


# In[ ]:


# Create DataFrame

attacktype_data = pd.DataFrame({"attacktype1_names":attacktype1_names,"nr_attacktype1":nr_attacktype1,"nkill_attacktype1":nkill_attacktype1,"average_nkill":average_nkill, "average_nkill_kills":average_nkill_kills,"average_nwound":average_nwound,"average_nwound_wounded":average_nwound_wounded,"average_propextent":average_propextent,"average_propextent_propextent":average_propextent_propextent})


# In[ ]:


# Sort the dataframes from large to small

sorted_attacktype_data = attacktype_data.sort_values(by='average_nkill', ascending=0)
sorted1_attacktype_data = attacktype_data.sort_values(by='average_nkill_kills', ascending=0)
sorted2_attacktype_data = attacktype_data.sort_values(by='average_nwound', ascending=0)
sorted3_attacktype_data = attacktype_data.sort_values(by='average_nwound_wounded', ascending=0)
sorted4_attacktype_data = attacktype_data.sort_values(by='average_propextent', ascending=0)
sorted5_attacktype_data = attacktype_data.sort_values(by='average_propextent_propextent', ascending=0)


# In[ ]:


# Make barplots 

ax = sns.barplot(y='attacktype1_names',x='average_nkill', data=sorted_attacktype_data, color="#00035b", palette="Reds_r")
ax.set_xlabel("Average number of killed people per attack", size=10, alpha=1)
ax.set_ylabel("Attacktype names", size=10, alpha=1)
ax.set(xlim=(0, 30))
ax.set_title("The average number of killed people per attack given the attack type", fontsize=12, alpha=1)
ax.tick_params(labelsize=10,labelcolor="black")
plt.show()

ax = sns.barplot(y='attacktype1_names',x='average_nwound', data=sorted2_attacktype_data, color="#00035b", palette="Blues_r")
ax.set_xlabel("Average number of wounded people per attack", size=10, alpha=1)
ax.set_ylabel("Attacktype names", size=10, alpha=1)
ax.set(xlim=(0, 30))
ax.set_title("The average number of wounded people per attack given the attack type", fontsize=12, alpha=1)
ax.tick_params(labelsize=10,labelcolor="black")
plt.show()

ax = sns.barplot(y='attacktype1_names',x='average_propextent', data=sorted4_attacktype_data, color="#00035b", palette="Greens_r")
ax.set_xlabel("Average property damage per attack", size=10, alpha=1)
ax.set_ylabel("Attacktype names", size=10, alpha=1)
ax.set(xlim=(0, 4))
ax.set_title("The average extent of property damage per attack type", fontsize=12, alpha=1)
ax.tick_params(labelsize=10,labelcolor="black")
plt.show()

# make donutcharts

fig = {
  "data": [
    {
      "values": average_nkill_kills,
      "labels": attacktype1_names
        ,
    "text":"Property Damage",
      "textposition":"inside",
      "domain": {"x": [0, .30]},
      "name": "",
      "hoverinfo":"label+percent+name",
          "hole": .4,
      "type": "pie"
    },     
      {
      "values": average_nwound_wounded,
      "labels": attacktype1_names
        ,
    "text":"nkill",
      "textposition":"inside",
      "domain": {"x": [.35, .65]},
      "name": "",
      "hoverinfo":"label+percent+name",
          "hole": .4,
      "type": "pie"
    },     
    {
      "values": average_propextent_propextent,
      "labels":  attacktype1_names
        ,
      "text":"Nwound",
      "textposition":"inside",
      "domain": {"x": [.70, 1]},
      "name": "",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Share of kills, wounded and property damage per attack type", "showlegend":False,
        "annotations": [
            {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Killed",
                "x": 0.13,
                "y": 0.5
            },
             {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Wounded",
                "x": 0.50,
                "y": 0.5
            },
            {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Property",
                "x": 0.885,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')


# The Bar Charts above show the average people killed/wounded/poroperty damage for each individual attack type. In other words, it shows which attack types come with the most amount of damage, which will later be used as a form of success of an attack. 
# 
# Whereas the "donut" charts show the percentual contribution of each attack type of the total kills/wounded/property damage. 
# 
# It can be seen that:
# * Bombing has the biggest share in number of wounded and the total amount of property damage, the biggest difference is for hijacking and unarmed assault.
# * An armerd attack has as many kills, but less wounded and property damage compared to bombing.
# * Infrastructure attacks have a lot of property damage with relative little kills and wounded.

# ### **How has the usage of weapons over time evolved?**

# In[ ]:


# Weapon type used in hystorical context

data_Biological = data_terrorism[(data_terrorism.weaptype1 == 1)]
data_Chemical = data_terrorism[(data_terrorism.weaptype1 == 2)]
data_Radiological = data_terrorism[(data_terrorism.weaptype1 == 3)]
data_Nuclear = data_terrorism[(data_terrorism.weaptype1 == 4)]
data_Firearms = data_terrorism[(data_terrorism.weaptype1 == 5)]
data_Explosives = data_terrorism[(data_terrorism.weaptype1 == 6)]
data_Fake = data_terrorism[(data_terrorism.weaptype1 == 7)]
data_Incendiary = data_terrorism[(data_terrorism.weaptype1 == 8)]
data_Melee = data_terrorism[(data_terrorism.weaptype1 == 9)]
data_Vehicle = data_terrorism[(data_terrorism.weaptype1 == 10)]
data_Sabotage = data_terrorism[(data_terrorism.weaptype1 == 11)]
data_Other = data_terrorism[(data_terrorism.weaptype1 == 12)]

peryear_Biological = np.asarray(data_Biological.groupby('iyear').iyear.count())
peryear_Chemical = np.asarray(data_Chemical.groupby('iyear').iyear.count())
peryear_Radiological = np.asarray(data_Radiological.groupby('iyear').iyear.count())
peryear_Nuclear = np.asarray(data_Nuclear.groupby('iyear').iyear.count())
peryear_firearms = np.asarray(data_Firearms.groupby('iyear').iyear.count())
peryear_Explosives = np.asarray(data_Explosives.groupby('iyear').iyear.count())
peryear_fake = np.asarray(data_Fake.groupby('iyear').iyear.count())
peryear_Incendiary = np.asarray(data_Incendiary.groupby('iyear').iyear.count())
peryear_Melee = np.asarray(data_Melee.groupby('iyear').iyear.count())
peryear_Vehicle = np.asarray(data_Vehicle.groupby('iyear').iyear.count())
peryear_Sabotage = np.asarray(data_Sabotage.groupby('iyear').iyear.count())
peryear_Other = np.asarray(data_Other.groupby('iyear').iyear.count())


# In[ ]:


# Group data per year for the world

terror_peryear_world = np.asarray(data_terrorism.groupby('iyear').iyear.count())
terror_years = np.arange(1970, 2016)

# Plot graph
trace1 = go.Scatter(                             
         x = terror_years,
         y = peryear_Biological,
         mode = 'lines',
         line = dict(
             color = 'rgb(140, 140, 45)',
             width = 3),
        name = 'Biological '
         )
trace2 = go.Scatter(                             
         x = terror_years,
         y = peryear_Chemical,
         mode = 'lines',
         line = dict(
             color = 'rgb(240, 40, 45)',
             width = 3),
        name = 'Chemical '
         )
trace3 = go.Scatter(                             
         x = terror_years,
         y = peryear_Radiological,
         mode = 'lines',
         line = dict(
             color = 'rgb(120, 120,120)',
             width = 3),
        name = 'Radiological'
         )
trace4 = go.Scatter(                             
         x = terror_years,
         y = peryear_Nuclear,
         mode = 'lines',
         line = dict(
             color = 'rgb(0, 50, 72)',
             width = 3),
        name = 'Nuclear'
         )
trace5 = go.Scatter(                             
         x = terror_years,
         y = peryear_firearms,
         mode = 'lines',
         line = dict(
             color = 'rgb(27, 135 , 78)',
             width = 3),
        name = 'firearms'
         )
trace6 = go.Scatter(                             
         x = terror_years,
         y = peryear_Explosives,
         mode = 'lines',
         line = dict(
             color = 'rgb(230, 230, 230)',
             width = 3),
        name = 'Explosives'
         )
trace7 = go.Scatter(                             
         x = terror_years,
         y = peryear_fake,
         mode = 'lines',
         line = dict(
             color = 'rgb(238, 133, 26)',
             width = 3),
        name = 'fake weapons'
         )

trace8 = go.Scatter(                             
         x = terror_years,
         y = peryear_Incendiary,
         mode = 'lines',
         line = dict(
             color = 'rgb(238, 133, 26)',
             width = 3),
        name = 'Incendiary'
         )

trace9 = go.Scatter(                             
         x = terror_years,
         y = peryear_Melee,
         mode = 'lines',
         line = dict(
             color = 'rgb(238, 133, 26)',
             width = 3),
        name = 'Melee'
         )

trace10 = go.Scatter(                             
         x = terror_years,
         y = peryear_Vehicle,
         mode = 'lines',
         line = dict(
             color = 'rgb(238, 133, 26)',
             width = 3),
        name = 'Vehicle'
         )

trace11 = go.Scatter(                             
         x = terror_years,
         y = peryear_Sabotage,
         mode = 'lines',
         line = dict(
             color = 'rgb(238, 133, 26)',
             width = 3),
        name = 'Sabotage'
         )

trace12 = go.Scatter(                             
         x = terror_years,
         y = peryear_Other,
         mode = 'lines',
         line = dict(
             color = 'rgb(238, 133, 26)',
             width = 3),
        name = 'Other'
         )

layout = go.Layout(
         title = 'Weapon type used (1970-2015)',
         xaxis = dict(
             rangeslider = dict(thickness = 0.05),
             showline = True,
             showgrid = False
         ),
         yaxis = dict(
             range = [0.1, 7500],
             showline = True,
             showgrid = False)
         )

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12]

figure = dict(data = data, layout = layout)
iplot(figure)


# The figure above shows the type of weapons used over time, worldwide. A significant finding from this graph is the sudden growth of 'Explosives' and 'Firearms', from around 2010. 
# 
# The findings are:
# * Over recent history explosives and firearms have always been the most popular attack type
# * Attacks with explosives and firearms have dropped after 1992
# * In 1997 there was an aboslute low in the number of attacks
# * Attacks with explosives and firearms have a high growth after 2002 and decreased after 2013

# ### ** How are the number of people killed distributed over the year? **
# 

# In[ ]:


# Average numbkills per month

width = 1/1.5
#ax.set_ylabel('Number of Killed')
#ax.set_xlabel('Month')
plt.bar(data_terrorism.imonth, ((data_terrorism.nkill)), width, color="blue")


# The barplot above shows the total number of kills per month over all years from 1970-2015, grouped by month. It provides the following insights:
# - The number of people killed are not evenly distributed over the year
# - April, June and September are months which generated relatively large amount of kills
# - The amount of people killed in the remaining months is relatively stable
# 
# *Disclaimer: whe should be careful when interpreting this figure. A few big attacks in the same month which killed a lot of people can fairly easily disturb the figure. However, the results we retrieved from the existing data are showing significant and interesting results worth interpreting.*
# 
# **How can the results be explained? **
# - June and September relate to the start and the end of summer holidays in the western world, respectively. In these months, many people are on the move and many events are heldm which indicates crowded places and easy targets.
# 
# 

# ### **What is the number of terrorists involved per weapon type in an attack?**

# In[ ]:


# Create first graph

ax = sns.regplot(x="weaptype1", y="nperps", data=data_terrorism)


# In[ ]:


# Create second graph

data_plot = data_terrorism[data_terrorism.nperps > 0]

plt.figure(figsize=(10,6))
plt.ylim(0, 100)
sns.set(style="ticks")
sns.boxplot(x="weaptype1", y="nperps", data=data_plot, palette="Reds")
plt.legend(['1. Biological','2. Chemical','3. Radiological','4. Nuclear', '5. Firearms', '6. Explosives', '7. Fake Weapons', '8. Incendiary', '9. Melee', '10. Vehicle', '11. Sabotage equipment', '12. Other', '13. Unknown'],loc = 2)
sns.despine(offset=10, trim=True)


# The first graph above tries to find a relation between the weapon type used and the amount of persons that participated in the terrorist attack. However, the graph is quite hard to read due to the outliers, and the large cluster of terrorist near 1. The second graph shows boxplots for the different attack types where the axis limit is set to 100 and is thus graph 1 differntly visualized.. 
# 
# This graph clearly shows:
# * Besides other and unknown, incendiary and sabotage equipment has the largest deviance in number of terrorists..
# * It also shows that independent of the weapontype, most attacks have been committed by less than 20 attackers.

# ### **How many people die or/and get wounded yearly by terror attacks?**

# In[ ]:


# Create arrays

peryear_nwound = np.asarray(data_terrorism.groupby('iyear').nwound.sum())
peryear_nkill = np.asarray(data_terrorism.groupby('iyear').nkill.sum())
terror_years = np.arange(1970, 2016)


# In[ ]:


# Plot graph

trace0 = go.Scatter(
         x = terror_years,
         y = peryear_nwound,
         mode = 'lines',
         line = dict(
             color = 'rgb(240, 140, 45)',
             width = 3),
        name = 'kills'
         )

trace1 = go.Scatter(
         x = terror_years,
         y = peryear_nkill,
         mode = 'lines',
         line = dict(
             color = 'rgb(0, 50, 72)',
             width = 3),
        name = 'wound'
         )

layout = go.Layout(
         title = 'kills & Wounded by Year for the world (1970-2015)',
         xaxis = dict(
             rangeslider = dict(thickness = 0.05),
             showline = True,
             showgrid = False
         ),
        yaxis = dict(
             range = [0.1, 50000],
             showline = True,
             showgrid = False)
         )
data = [trace0, trace1]
figure = dict(data = data, layout = layout)
iplot(figure)


# The above graph gives the following insights:
# * There is a positive trend line with the total deaths per year by terror attacks. 
# * There is a positive trend line with the total wounded per year by terror attacks. 
# * Since 1992 more people died than get wounded by terror attacks. 
# * Most people died due to terror attacks in the year 2014
# *  Most people get wounded due to terror attacks in the year 2013
# * There is a decline in the amount of wounded by terror attacks after the year 2013
# * There is a decline in the amount of deads by terror attacks after the year 2014 

# ### **Which weapons are most effective to use? **

# In[ ]:


# Create arrays

nr_weaptype1 = np.asarray(data_terrorism.groupby('weaptype1').weaptype1.count())
nkill_weaptype1 = np.asarray(data_terrorism.groupby('weaptype1').nkill.sum())
average_nkill_weap = np.divide(nkill_weaptype1, nr_weaptype1) 

weaptype1_names = np.array(['Biological','Chemical','Radiological','Firearms','Explosives/Bombs/Dynamite','Fake weapons','Indenciary','Melee','Vehicle','Sabotage Equipment','Other','Unkown'])
total_deaths_weap = sum(nkill_weaptype1)
average_nkill_weap_2 = np.divide(nkill_weaptype1, total_deaths_weap) 
average_nkill_weap_total = average_nkill_weap_2*100

nwound_weaptype1 = np.asarray(data_terrorism.groupby('weaptype1').nwound.sum())
average_nwound_weap = np.divide(nwound_weaptype1, nr_weaptype1) 

total_wounded_weap = sum(nwound_weaptype1)
average_nwound_weap_2 = np.divide(nwound_weaptype1, total_wounded_weap) 
average_nwound_weap_total = average_nwound_weap_2*100

total_nkill_nwound_weap = total_deaths_weap + total_wounded_weap
nkill_nwound_weap = [x + y for x, y in zip(nkill_weaptype1, nwound_weaptype1)]
share_nkill_nwound_weap = np.divide(nkill_nwound_weap,total_nkill_nwound_weap)
share_percent_nkill_nwound_weap = share_nkill_nwound_weap*100


# In[ ]:


# Make donutcharts

fig = {
  "data": [
    {
      "values": average_nkill_weap_total,
      "labels": weaptype1_names
        ,
    "text":"fatalities",
      "textposition":"inside",
      "domain": {"x": [0, .3]},
      "name": "",
      "hoverinfo":"label+percent+name",
          "hole": .4,
      "type": "pie"
    },     
      {
      "values": average_nwound_weap_total,
      "labels": weaptype1_names
        ,
    "text":"ijuries",
      "textposition":"inside",
      "domain": {"x": [.35, .65]},
      "name": "",
      "hoverinfo":"label+percent+name",
          "hole": .4,
      "type": "pie"
    },
       {
      "values": share_percent_nkill_nwound_weap,
      "labels": weaptype1_names
        ,
    "text":"fatalities + ijuries",
      "textposition":"inside",
      "domain": {"x": [.70, 1]},
      "name": "",
      "hoverinfo":"label+percent+name",
          "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Share of killed and wounded people per weapon type", "showlegend":False,
        "annotations": [
            {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Killed",
                "x": 0.13,
                "y": 0.5
            },
             {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Wounded",
                "x": 0.5,
                "y": 0.5
            },
            {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Casualties",
                "x": 0.89,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')


# The charts above show the percentual contribution of each weapon type in the total fatalities, injuries and total casualties caused by terrorist attacks.
# 
# * We can see that more than 85 percent of the people killed in terror attacks, have been killed by explosives or firearms.
# * We can see that explosives/ bombs /dynamite are by far the most effective weapons for maximizing the amount of casualties. 
# * Remarkably, firearms and explosives/ bombs /dynamite are equally effective in creating fatalities, whereas for creating injuries, explosives/ bombs /dynamite are 5x more effective. 

# ## **There is a second follow-up notebook containing the predictive analysis and the conclusions**
