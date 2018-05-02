
# coding: utf-8

# In[4]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# # **Air Pollution - Learn the basics**
# ***
# 
# **Mhamed Jabri — 02/19/2018**
# 
# Air pollution means exactly what you think it means : it's when the quality of the air you breathe drops. But how does that happen ? That's the real question. It's induced by the presence of harmful, unwanted substances in air (more precisely, into Earth's atmosphere). Those bad substances are the pollutants and most of the tables in this database focus on those pollutants and give information about their preseance in the air.  
# Also, note that [global warming and air pollution are two distinct phenomena](https://www.quora.com/What-is-the-difference-between-air-pollution-and-global-warming) but they do have a lot in common, mainly : **NEITHER ONE OF THEM IS A HOAX !!**
# 
# By including BigQuery to its kernels, Kaggle allows its users to explores HUGE databeses / datasets which means endless possibilities and discovers to be made. I've decided to create this notebook for two reasons : The first one is that this subject is highly interesting to everyone I think, at least to me, because I think we should all be concerned about the quality of the air we're breathing and how we are affecting it by our daily activities. The second one is that I couldn't wait to use BigQuery on a Jupyter Notebook and see how it goes because it is indeed very exciting !!
# 
# So I hope that when you're done with this notebook, you'll have learned a thing or two about Air Pollution. 
# 
# If you like the kernel, please leave me a comment / upvote, I would highly appreciate it :) ! 

# # Table of contents
# ***
# 
# * [What's air pollution / air quality ?](#introduction)
# * [1. Air pollution in the US in 2016](#counties)
# * [2. Yearly evolution of air quality : Rural vs Urban states](#yearly)
# * [3. Impact of Temperature and Humidity](#weather)
# * [4. Worldwide air pollution](#world)
# * [Conclusion](#conclusion)
# 

# ![](http://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Air_Pollution-Causes%26Effects.svg/1052px-Air_Pollution-Causes%26Effects.svg.png)

# # What's air pollution / air quality ? 
# <a id="introduction"></a> 
# ***
# 
# As said earlier, air pollution is due to the presence of some pollutants, it's time to learn more about the main pollutants that are present in our database : 
# * Sulphur dioxide ($SO_2$) : This contaminant is mainly emitted during the combustion of fossil fuels such as crude oil and coal.
# * Carbon monoxide ($CO$) : This gas consists during incomplete combustion of fuels example :  A car engine running in a closed room.
# * Nitrogen dioxide ($NO_2$) : These contaminants are emitted by traffic, combustion installations and the industries.
# * Ozone ($O_3$) : Ozone is created through the influence of ultra violet sunlight (UV) on pollutants in the outside air.
# * Particulate Matter ($PM$) : Particulate matter is the sum of all solid and liquid particles suspended in air. This complex mixture includes both organic and inorganic particles, such as dust, pollen, soot, smoke, and liquid droplets. These particles vary greatly in size, composition, and origin.
# 
# So how are those pollutants produced in our daily lives ? Well as you may have guessed, The main sources of air pollution are the industries, agriculture and traffic (held responsible for one-third of the greenhouse gas emissions). That being said, us consumers are also responsible of polluting the air through some of our activities such as smoking or heating houses ...   
# There's also the effect of weather (wind, temperature, ultra violet sunlight for the Ozone ...), the interactions of all those things provides the picture above which sums it up nicely actually.
# 
# ### Air Quality Index 
# 
# In severel tables in this databse, you'll find a column 'aqi' which stands for Air Quality Index. Basically the AQI is the measure of how air is polluted, with respect to some pollutant. That means that for a specific hour in a specific place you'll have different AQIs, one for each pollutant.  
# So one must know what values of AQI mean 'good' and what values mean 'bad', hence the table below.
# 
# ![](http://www.deq.idaho.gov/media/818444-aqi_496x260.jpg)
# 
# I feel like this introduction was important to give some context and explain what air pollution is about so that all that's coming would make sense for the reader. Let's code now !

# In[5]:


import pandas as pd
import numpy as np
from google.cloud import bigquery
from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
pollutants = ['o3','co','no2','so2','pm25_frm']


# # Air pollution in the US in 2016
# <a id="counties"></a>
# ***
# 
# In this first part, we'll try to get a grasp of how polluted was the air all over the US during 2016.  
# 
# To do so, I wanted to extract the average AQI for each pollutant for each county, which translates to a groupby(county) in SQL/pandas. In other words, since every table contains the information about one single pollutant, the use of JOIN was necessary. However, needing the information about 5 pollutants meant using 4 JOINs and it didn't respect the timeout limit so I needed another way around.    
# **This is where using Python combined to SQL becomes actually cool.** If you're writing such a query in MySQL for example, you won't have many possibilities to 'tweak' it. Here, since the query is actually a string for Python, I wrote a query that I could modify inside a for loop so that in each steap it gives me the information I want about one specific pollutant and at the end, I would just concatenate the dataframes ! 

# In[6]:


QUERY2016 = """
    SELECT
        pollutant.county_name AS County, AVG(pollutant.aqi) AS AvgAQI_pollutant
    FROM
      `bigquery-public-data.epa_historical_air_quality.pollutant_daily_summary` as pollutant
    WHERE
      pollutant.poc = 1
      AND EXTRACT(YEAR FROM pollutant.date_local) = 2016
    GROUP BY 
      pollutant.county_name
"""

df_2016 = None
for elem_g in pollutants : 
    query = QUERY2016.replace("pollutant", elem_g)
    temp = bq_assistant.query_to_pandas(query).set_index('County')
    df_2016 = pd.concat([df_2016, temp], axis=1, join='outer')
df_2016=df_2016.apply(lambda x: x.fillna(x.mean()),axis=0)

df_2016.sample(10,random_state=42)


# Okay so now we have the 2016 measures for every county in the US, what can we do about it to retrieve some useful insights ? Well, **clustering** is a natural answer in the context !
# Let's see what t-SNE will give us here : 

# In[ ]:


from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2,n_iter=2000,perplexity=35,random_state=5).fit_transform(df_2016)
df_tsne = pd.DataFrame(X_tsne)
df_tsne['County'] = list(df_2016.index)
df_tsne = df_tsne.set_index('County')
df_tsne.columns = ['ax1', 'ax2']

df_tsne.plot(kind='scatter', x='ax1', y='ax2',figsize=(10,8));

#c1 : ax1=[10 , 35]  ax2=[-20 , 10] 
#c2 : ax1=[-30 , -5]  ax2= [-10 , 10]
#c3 : ax1=[0 , 10]  ax2= [-30 , -2]


# Well, we end up with 4 clusters that are actually not that bad. We can see on the plot 4 distinct parts (basically top/bottom/left/right),  the cluster in the top has a somewhat complicated shape and it will be a bit trickier to extract it (we'll divide into two parts and join them).   
# Let's extract all those clusters, check the means for each pollutant in each cluster and see what properties do the counties of each cluster have in common. 

# In[ ]:


#Right part of the plot
msk_1 = ((df_tsne['ax1']>10) & (df_tsne['ax1']<35 )) & ((df_tsne['ax2']>-20) &(df_tsne['ax2']<10))
cluster_1 = df_tsne[msk_1]
indexes_1 = cluster_1.index 
ex_1 = df_2016.loc[indexes_1]

#left part of the plot
msk_2 = ((df_tsne['ax1']>-30) & (df_tsne['ax1']<-5 )) & ((df_tsne['ax2']>-10) &(df_tsne['ax2']<10))
cluster_2 = df_tsne[msk_2]
indexes_2 = cluster_2.index 
ex_2 = df_2016.loc[indexes_2]    

#bottom part of the plot
msk_3 = ((df_tsne['ax1']>0) & (df_tsne['ax1']<10)) & ((df_tsne['ax2']>-30) &(df_tsne['ax2']<-2))
cluster_3 = df_tsne[msk_3]
indexes_3 = cluster_3.index 
ex_3 = df_2016.loc[indexes_3]

#top part of the plot
msk_4_1 = ((df_tsne['ax1']>-18) & (df_tsne['ax1']<-3)) & ((df_tsne['ax2']>18) &(df_tsne['ax2']<30))
msk_4_2 = ((df_tsne['ax1']>0) & (df_tsne['ax1']<3.5)) & ((df_tsne['ax2']>0) &(df_tsne['ax2']<13))
cluster_4 = df_tsne[msk_4_1 | msk_4_2]
indexes_4 = cluster_4.index 
ex_4 = df_2016.loc[indexes_4]

means_c1 = ex_1.mean(axis=0)
means_c2 = ex_2.mean(axis=0)
means_c3 = ex_3.mean(axis=0)
means_c4 = ex_4.mean(axis=0)

means = pd.DataFrame([means_c1,means_c2,means_c3,means_c4], ['c1','c2','c3','c4'])
means


# The *second cluster (the left part of the plot) seems to be the most polluted one* : it has the highest average AQI for every pollutant except for the O3 where it's second to the fourth cluster.    
# The *first cluster (the right part of the plot) appears to be the cleanest one* with exception to the Ozone AQI, indeed it has a higher mean than the third cluster.     
# 
# It seems that moving from left to right on the plot, the air gets cleaner and the average AQI for PM2.5 drops greatly. On the other hand, moving from bottom to top on the plot, the biggest change appears for the average AQI for O3.
# 
# Let's get a look on some of the counties in each of those clusters : 

# In[ ]:


ex_counties = pd.concat([ex_1.sample(1,random_state=17), #countie from c1 17 27
                         ex_2.sample(1,random_state=21), #countie from c2
                         ex_3.sample(1,random_state=33), #countie from c3
                         ex_4.sample(1,random_state=57)], axis=0)

ex_counties


# Since I'm not American, I didn't actually know the state of pollution of any of those counties so I checked them up on the internet to see if our clustering conclusions were any good :   
# Sacremento is in the second bag (the most polluted one) and it seems that Sacremento area is indeed known for its bad air quality as ranks 8th in the nation for unhealthy air, according to the latest American Lung Association report ([source](http://www.sacbee.com/news/local/health-and-medicine/article147514699.html)).
# 

# # Rural VS Urban
# ***
# <a id="yearly"></a>
# ***
# 
# In the first part, we've taken 1 year of measure in all counties and compared between them.  
# In the following, we'll instead study the **evolution of air pollution throughout the years**. The way we'll do that is that we'll take 6 states : 
# - Urban (top three populated stated) :  California, Texas, Florida
# - Known to be Rural : Mississippi, Vermont, Alaska    
# 
# For each of those states, we'll study the evolution of the yearly AQI of several pollutants, namely : *o3, co, so2, pm2.5* and we'll try to give an answer to two questions : 
# 1. Is the air now cleaner or more polluted compared to what it was in the past ?
# 2. Which states are cleaner, the rural or the urbans ?

# In[ ]:


# A function that will be used later to put the name of state and name of pollutant in the query

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


# We write a query that takes a pollutant and a state and gives back the yearly average AQI for that pollutant. We'll use a 'for' loop with the function '*replace_all'* that's above to execute that query for different pollutant/states.  
# 
# We'll build a dictionary where the keys will be the pollutants and the values will be dataframes giving the evolution of the average AQI for that pollutant for the six gives states.

# In[ ]:


states = ['California','Texas','Florida',
          'Mississippi','Vermont','Alaska'] 
pollutants = ['o3','co','so2','pm25_frm']

QUERY = """
    SELECT EXTRACT(YEAR FROM pollutant.date_local) as Year , AVG(pollutant.aqi) as AvgAQI_State
    FROM
      `bigquery-public-data.epa_historical_air_quality.pollutant_daily_summary` as pollutant
      WHERE pollutant.poc = 1 AND  pollutant.state_name = 'State'
    GROUP BY Year
    ORDER BY Year ASC
        """

dict_pol={}
for elem_g in pollutants : 
    dict_pol[elem_g] = None 
    for elem_s in states :
        dic = {"State": elem_s, "pollutant": elem_g}
        query = replace_all(QUERY, dic)
        temp = bq_assistant.query_to_pandas(query).set_index('Year')
        dict_pol[elem_g] = pd.concat([dict_pol[elem_g], temp], axis=1, join='inner')
        
dict_pol['co'].head(10)


# This is a perfect example of how you can make interact SQL with Python thanks to BigQuery. Not only you use a query to build a dataframe but we actually run a loop on that query to modify it each time and retrieve different dataframes that we merged !     
# Indeed, in the previous example (query part 1), you could argue that a JOIN would have done the job.** Here the query wouldn't work with the use of JOIN**. Indeed, we want to group by YEAR for each state alone, something like 'state in('....') would take the average on all those states instead of giving you one column per state !
# 
# Now let's plot the results and see what we've got.

# In[ ]:


fig, axs = plt.subplots(figsize=(20,12),ncols=2,nrows=2 )
dict_pol['o3'].plot( y=['AvgAQI_California','AvgAQI_Texas','AvgAQI_Florida',
                        'AvgAQI_Alaska','AvgAQI_Mississippi','AvgAQI_Vermont'], ax=axs[0,0],
                    title='Evolution of o3')
dict_pol['pm25_frm'].plot( y=['AvgAQI_California','AvgAQI_Texas','AvgAQI_Florida',
                              'AvgAQI_Alaska','AvgAQI_Mississippi','AvgAQI_Vermont'], ax=axs[0,1],
                          title='Evolution of pm2.5')
dict_pol['so2'].plot( y=['AvgAQI_California','AvgAQI_Texas','AvgAQI_Florida',
                         'AvgAQI_Alaska','AvgAQI_Mississippi','AvgAQI_Vermont'], ax=axs[1,0],
                     title='Evolution of so2')
dict_pol['co'].plot( y=['AvgAQI_California','AvgAQI_Texas','AvgAQI_Florida',
                        'AvgAQI_Alaska','AvgAQI_Mississippi','AvgAQI_Vermont'], ax=axs[1,1],
                   title='Evolution of co')


plt.show();


# Now keep in mind that we want to use those plots to answer two questions.
# 
# We observe that both so2 and co have a better AQI now than years ago in all states, especially Alaska.  
# For particulate matter (PM_2.5), the AQI is better in all states except Alaska where it's actually increasing !   
# The ozone is the trickiest : there's no clear improvment in any state of those we chose.  
# Considering all those facts, the answer to our first question seems to be that**the air is cleaner now compared to years ago in the US except when it comes to O3**. I was actually a bit surprised by that so I googled it and found a pretty good [link](http://www.berkeleywellness.com/healthy-community/environmental-health/article/air-pollution-getting-worse-or-better). Basically, it confirms what these plots are telling : "Overall, air quality is improving nationally. But there is a problem with pollutants that are very difficult to controle, a good example is ozone (o3)".
# 
# Moving to our second point : For so2 and co, all states have approximately the same average AQI except Alaska which is a bit higher. For PM2.5, the rural states (Vermont and Missisipi) have a better AQI than the urban states. For o3, California comes first by a large margin and the rural states fair better but not with a considerale margin . So overall, **Air Quality seems to be better in the rural states considering Ozone and Particulate Matter**. I expected the Air Quality to be better by a larger margin in rural states actually so this was a little surprise for me. That being said, we should keep in mind that we're taking states not counties so they're not entirely rural. Comparing county to county should lead to more precise results.

# # Temperature and Humidity VS Air Pollution
# <a id="weather"></a>
# ***
# 
# This database also contains tables about temperature, humidity, pressure ... Naturally, we wonder how air quality affects / is affected by the weather and that's what we'll try to see in this part.  
# The state of california being the most populated but also one of the most polluted states, we'll focus on it for what will come. 
# Here's what we'll do : 
# - Extract the average temperature in the state of California for each day in 2016 and put in a dataframe
# - Extract the average relative humidity in the state of California for each day in 2016 and put in a dataframe
# - Extract the average AQI for* Ozone* in the state of California for each day in 2016 and put in a dataframe
# - Extract the average AQI for* PM_2.5* in the state of California for each day in 2016 and put in a dataframe   
# We'll concatenate all those dataframes which will give us a dataframe describing the daily evolution of those 4 quantities over the year 2016.
# 
# P.S : Here again I've written down 4 queries below instead of using INNER JOIN on 'date_local' because of timeout issues.

# In[ ]:


QUERYtemp = """
    SELECT
       EXTRACT(DAYOFYEAR FROM T.date_local) AS Day, AVG(T.arithmetic_mean) AS Temperature
    FROM
      `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary` as T
    WHERE
      T.state_name = 'California'
      AND EXTRACT(YEAR FROM T.date_local) = 2016
    GROUP BY Day
    ORDER BY Day
"""

QUERYrh = """
    SELECT
       EXTRACT(DAYOFYEAR FROM rh.date_local) AS Day, AVG(rh.arithmetic_mean) AS Humidity
    FROM
      `bigquery-public-data.epa_historical_air_quality.rh_and_dp_daily_summary` as rh
    WHERE
      rh.state_name = 'California'
      AND rh.parameter_name = 'Relative Humidity'
      AND EXTRACT(YEAR FROM rh.date_local) = 2016
    GROUP BY Day
    ORDER BY Day
"""

QUERYo3day = """
    SELECT
       EXTRACT(DAYOFYEAR FROM o3.date_local) AS Day, AVG(o3.aqi) AS o3_AQI
    FROM
      `bigquery-public-data.epa_historical_air_quality.o3_daily_summary` as o3
    WHERE
      o3.state_name = 'California'
      AND EXTRACT(YEAR FROM o3.date_local) = 2016
    GROUP BY Day
    ORDER BY Day
"""

QUERYpm25day = """
    SELECT
       EXTRACT(DAYOFYEAR FROM pm25.date_local) AS Day, AVG(pm25.aqi) AS pm25_AQI
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary` as pm25
    WHERE
      pm25.state_name = 'California'
      AND pm25.sample_duration = '24 HOUR'
      AND EXTRACT(YEAR FROM pm25.date_local) = 2016
    GROUP BY Day
    ORDER BY Day
"""

df_temp = bq_assistant.query_to_pandas(QUERYtemp).set_index('Day')
df_pres = bq_assistant.query_to_pandas(QUERYrh).set_index('Day')
df_o3daily = bq_assistant.query_to_pandas(QUERYo3day).set_index('Day')
df_pm25daily = bq_assistant.query_to_pandas(QUERYpm25day).set_index('Day')

df_daily = pd.concat([df_temp, df_pres, df_o3daily, df_pm25daily], axis=1, join='inner')

df_daily.sample(10,random_state = 42)


# We've got our dataset, before plotting the columns againt each other, let's plot a correlation matrix to get some insight about our features.

# In[ ]:


corr = df_daily.corr()

# plot the heatmap
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=False, linewidths=.5, cbar_kws={"shrink": .5});


# We don't really care about the correlation between Temperature and Humidity.   
# The Ozone AQI seems to be positively correlated to temperature and negatively correlated to humidity.  
# The PM_2.5 AQI doesn't seem to be correlated to any of the metrics.  
# 
# Let's see what the plots are saying !

# In[ ]:


plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
 
f, axs = plt.subplots(2,2,figsize=(15,10))
# multiple line plot
num=0
for column in df_daily :
    num+=1 # Find the right spot on the plot
    plt.subplot(2,2, num)
    # plot every groups, but discreet
    for v in df_daily : 
        plt.plot(df_daily.index, df_daily[v], marker='', color='grey', linewidth=0.6, alpha=0.3)
    # Plot the lineplot
    plt.plot(df_daily.index, df_daily[column], marker='',
             color=palette(num), linewidth=2.4, alpha=0.9, label=column)
    # Same limits for everybody!
    plt.xlim(0,370)
    plt.ylim(0,100)
    # Not ticks everywhere

    # Add title
    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )

plt.suptitle("Temperature and Humidity impact on Ozone and Particulate Matter", fontsize=17, fontweight=0, color='black', style='italic', y=1.0);
 


# Those plots are very telling, let's see what we've got here :   
# 
# **There's a clear impact of Temperature on the Ozone level**. Indeed, the hottest days seem to also be the ones where the Ozone level is at its highest, as suggested by the heatmap before those plots. We notice that the period between day #150 and day #250 (which means June, July and August) has the most peaks for the ozone level. Let's see what's the average AQI for those days : 

# In[ ]:


print('The average o3 AQI between day #150 and day #250 is' , round(df_daily['o3_AQI'].iloc[150:250].mean(),2))


# Wow ! This is obviously a high value for AQI and is definitely a bad thing for your health.  
# Now you must think *Well, it's definitely better to be in California during winter then* but slow down a little, it gets worse : The PM_2.5 AQI seems to be the highest between days #0 - #60 and #310 - #365, in months this means January, February, November and December which is basically late fall / early winter

# In[ ]:


temp = list(range(0, 60)) + list(range(310,366))
print('The average PM2.5 AQI during late fall / early winter is' , round(df_daily['pm25_AQI'].iloc[temp].mean(),2))


# So how come does this happen ? We've seen that High Temperature => High O3 level but PM 2.5 level doesn't seem to be correlated to neither temperature nor humidity (well a bit more to humidity actually) so why is it that high when the weather is cold ?  **It’s mostly from all the coal burning, both residential and commercial, that keeps us warm**, which says a lot about our responsabilities towards the environment.

# # Worldwide air pollution
# <a id="world"></a>
# ***
# 
# So far, we've been using a database that contains a lot of information about air pollution in the US. In this part, I'll use the *global_air_quality* table from the OpenAQ dataset and I'll try to see how different countries perform when it comes to air pollution.
# 
# Let's see first which year contains the largest number of measures because it doesn't make sens to average all the values over a decade for a country.   
# Also, the presence of the pollutants in the air isn't always measured using the same unit so we'll also check what's the dominant unit and stick with it.

# In[ ]:


bq_assistant_global = BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")

QUERY_year = """
    SELECT
       EXTRACT(YEAR FROM globalAQ.timestamp) AS Year,unit, count(*) AS Total_measures
    FROM
      `bigquery-public-data.openaq.global_air_quality` as globalAQ
    GROUP BY Year , unit
    ORDER BY Total_measures
"""

df_year = bq_assistant_global.query_to_pandas(QUERY_year).set_index('Year')
df_year


# 2018 is the year with the largest number of measures. The most used unit is $ug/m^3$.   
# We can't use all countries that appear in the data because some of them must have a very little number of measures, the same applies to pollutants who maybe weren't all measured the same amount of time, let's check.

# In[ ]:


QUERY_countries = """
    SELECT
       country, count(*) as nbmeasures, pollutant
    FROM
      `bigquery-public-data.openaq.global_air_quality` as globalAQ
    WHERE EXTRACT(YEAR FROM globalAQ.timestamp) = 2018 AND unit = 'µg/m³'
    GROUP BY country, pollutant
"""
df_countries = bq_assistant_global.query_to_pandas(QUERY_countries)

df_countries[df_countries['country']=='US']


# Going further, we'll only consider PM2.5 and we'll keep the countries that have at least 30 measures registered for that pollutant.

# In[ ]:


countries = list(df_countries[(df_countries['pollutant']=='pm25') & (df_countries['nbmeasures']>30 )]['country'])
countries


# In[ ]:


QUERY_global = """
    SELECT
       country, AVG(value) as Concentration
    FROM
      `bigquery-public-data.openaq.global_air_quality` as globalAQ
    WHERE 
       EXTRACT(YEAR FROM globalAQ.timestamp) = 2018 
       AND unit = 'µg/m³'
       AND country IN ('US','SK','IN','NO','BE','CL','TW','CZ','ES','NL','AU','CA',
                       'CN','GB','DE', 'FR')
       AND pollutant = 'pm25'
       AND value > 0
    GROUP BY country
"""

df_global = bq_assistant_global.query_to_pandas(QUERY_global)

f,ax = plt.subplots(figsize=(14,6))
sns.barplot(df_global['country'],df_global['Concentration'])
plt.title('Concentration of PM2.5 in the air over the world');


# Let's keep in mind that the conlusions we'll draw are based on PM2.5 alone (no Ozone or CO).  
# According to those values,** the most polluted air in 2018 is in India (IN) **and the values reached are critical and clearly unhealthy. India is nowhere near the other countries, in fact, the second country, Chile (CL), has a concentration of 58 while India's is 155 ... Czech Republic completes the top 3 of the polluest countries in the world.  
# On ther other hand, **Australia seems to have the cleanest air**, followed by Canada and the US.

# # Conclusion 
# <a id="conclusion"></a>
# ***
# I really enjoyed writing down this kernel !   
# 
# One of the greatest advantages of data science (which is also the main reason I chose to specialize in Data Science and Machine Learning) is that it gives you the opportunity to encounter so many new and meaningful subjects. Prior to this kernel, I didn't know that something called 'Particulate Matter' existed, I was convinced that air quality is far worse than what it was in the past ... And now, by analyzing some of this database and doing the necessary research to write down this notebook, I feel like I have a much better grasp of what air pollution really is (that's why I decided to go with this that title).
# 
# Kaggle is obviously getting better and better everyday as a learning platform. I had some prior experience using SQL but this is the first time I actually run queries to retrieve dataframes or modify my query with a for loop to do a lot of stuff simultaneously and I liked it very much. 
# 
# I hope this notebook helps readers on both aspects : 
# * The social aspect : understanding the challenges of air pollution, how it is measured, where it's more problematic, how it's affected by temperature ...
# * The data science aspect : how to use BigQuery to get the most out of our database and how to play with a query when you want to do a lot of different things.
# 
# Thank you for reading, hope you liked it and see you in another notebook ! 
