
# coding: utf-8

# # GPU EDA
# 
# **_2017-09-24_**
# 
# * [1.0 Introduction](#introduction)
#     * [1.1 Import packages](#import-packages)
#     * [1.2 Import data](#import-data)
#     * [1.3 Preprocessing and feature engineering](#preprocessing)
# * [2.0 Release dates](#release-dates)
# * [3.0 Resolution and 4k support](#resolution)
# * [4.0 Manufacturers](#manufacturers)
# * [5.0 Metrics by year](#metrics)
#     * [5.1 Core speed](#core-speed)
#     * [5.2 Memory](#memory)
#     * [5.3 Memory speed and bandwidth](#memory-speed)
#     * [5.4 Max power](#max-power)
#     * [5.5 Number of texture mapping units (TMU)](#tmu)
#     * [5.6 Texture Rate](#texture-rate)
#     * [5.7 Pixel Rate](#pixel-rate)
#     * [5.8 Process](#process)
#     * [5.9 Price](#price)
# * [6.0 Price/performance ratios](#ratios)
# * [7.0 GTX 1080](#gtx)
# * [8.0 Conclusion](#conclusion)

# ## 1.0 Introduction <a class="anchor" id="introduction"></a>
# Comprehensive analysis of GPU models over time, by price, max resolution, and various performance metrics. Please upvote if you like it, this notebook took time to compose. Any feedback a is also appreciated.

# ### 1.1 Import packages <a class="anchor" id="import-packages"></a>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ### 1.2 Import data <a class="anchor" id="import-data"></a>

# In[ ]:


df = pd.read_csv('../input/All_GPUs.csv')


# ### 1.3 Preprocessing and feature engineering <a class="anchor" id="preprocessing"></a>

# In[ ]:


#Convert release dates to useable format
df['Release_Date']=df['Release_Date'].str[1:-1]
df=df[df['Release_Date'].str.len()==11]
df['Release_Date']=pd.to_datetime(df['Release_Date'], format='%d-%b-%Y')


# In[ ]:


#Convert memory bandwidths to GB/s
s=df['Memory_Bandwidth']
s[(s.notnull()==True)&(s.str.contains('MB'))]=s[(s.notnull()==True)&(s.str.contains('MB'))].str[:-6].astype(int)/1000
s[(s.notnull()==True)&(s.str.contains('GB'))]=s[(s.notnull()==True)&(s.str.contains('GB'))].str[:-6].astype(float)
df['Memory_Bandwidth']=s


# In[ ]:


#Drop units from core_speed
df.Core_Speed=df.Core_Speed.str[:-4]
df.Core_Speed=df.Core_Speed.replace('',np.NaN)

# Create year/month/quarter features from release_date
df['Release_Price']=df['Release_Price'].str[1:].astype(float)
df['Release_Year']=df['Release_Date'].dt.year
df['Release_Quarter']=df['Release_Date'].dt.quarter
df['Release_Month']=df['Release_Date'].dt.month


# Brief look at the first few rows of the dataset:

# In[ ]:


df.head()


# ## 2.0 Release dates<a class="anchor" id="release-dates"></a>

# In[ ]:


plt.figure(figsize=(13, 6))
df['Release_Month'].groupby(df['Release_Month']).count().plot(kind='bar')
plt.title('Resolution counts')
plt.xlabel('Release Month')
plt.ylabel('Count of GPU releases')
plt.show()


# The modal month for releasing GPU models is June. December and July are low frequency release months.

# In[ ]:


plt.figure(figsize=(13, 6))
df['Release_Year'].groupby(df['Release_Year']).count().plot(kind='bar')
plt.title('Count of model releases')
plt.xlabel('Release Year')
plt.ylabel('GPU model releases')
plt.show()


# The modal year for number of models of GPU released was 2012. Remember this doesn't mean quantity of sales, only number of models released in a year. Anecdotally more GPUs have been sold in recent years with fewer models to choose from.
# 
# ## 3.0 Resolution and 4K Support<a class="anchor" id="resolution"></a>

# In[ ]:


res=['1920 x 1080', '1600 x 900','1366 x 768','2560 x 1440','2560 x 1600', '1024 x 768', '3840 x 2160']
plt.figure(figsize=(13,6))
for i in res:
        df[df['Best_Resolution']==i]['Best_Resolution'].groupby(df['Release_Year']).count().plot(kind='line')
plt.title('Resolution counts')
plt.xlabel('Release Year')
plt.ylabel('Count of GPU releases')
plt.legend(res)
plt.show()


# 1024 x 768 best resolution was most popular from 2002  until 2008 when 1366 x 768 took over. No new GPUs came out with a highest resolution capacity of 1024 x 768 after 2014. 4k support (3840 x 2160) was first introduced in 2013 and increased dramatically in 2017 as it replaced 2560x1600 as the highest resolution capacity.
# 
# ## 4.0 Manufacturers<a class="anchor" id="manufacturers"></a>

# In[ ]:


plt.figure(figsize=(12, 12))
df['Manufacturer'].value_counts().plot(kind='pie')


# Nvidia dominates in terms of number of models released

# In[ ]:


manufacturers=df['Manufacturer'].unique()
plt.figure(figsize=(13, 6))
for i in manufacturers:
      df[df['Manufacturer']==i]['Manufacturer'].groupby(df['Release_Year']).count().plot(kind='line')
plt.title('Manufacturer counts by release year')
plt.xlabel('Release Year')
plt.ylabel('Count of GPU releases')
plt.legend(manufacturers)
plt.show()


# Nvidia/AMD war in terms of models released (could be interpreted as market share) was close until around 2013 when Nvidia started to nudge ahead. Intel have fallen off in the last couple of years. AMD took a serious drop in 2016 but have released many GPUs in the first part of 2017. 

# In[ ]:


plt.figure(figsize=(13, 6))
sns.kdeplot(df[df['Release_Year']==2012]['Release_Price'])
sns.kdeplot(df[df['Release_Year']==2013]['Release_Price'])
sns.kdeplot(df[df['Release_Year']==2014]['Release_Price'])
sns.kdeplot(df[df['Release_Year']==2015]['Release_Price'])
sns.kdeplot(df[df['Release_Year']==2016]['Release_Price'])
plt.title('Price distributions by year')
plt.xlabel('Price')
plt.legend(['2012','2013','2014','2015','2016'])
plt.xlim(-100,1500)


# In[ ]:


plt.figure(figsize=(13, 6))
sns.kdeplot(df[(df['Manufacturer']=='Nvidia')&(df['Release_Price']<2000)]['Release_Price'])
#excluding expensive GPU from Nvidia, see section 5.9
sns.kdeplot(df[df['Manufacturer']=='AMD']['Release_Price'])
sns.kdeplot(df[df['Manufacturer']=='ATI']['Release_Price'])
plt.title('Price distributions by manufacturer')
plt.xlabel('Price')
plt.legend(['Nvidia','AMD','ATI'])


# Nvidia have a wider distribution than other manufacturers and have strong market share above 400 dollars. Nvidia is known for high quality, high end GPUs. AMD tends to cover the 50-300 dollar budget. ATI have a very specific niche in the market at approx 250 dollars.
# 
# ## 5.0 Performance by year<a class="anchor" id="performance"></a>
# ### 5.1 Core speed  <a class="anchor" id="core-speed"></a>

# In[ ]:


plt.figure(figsize=(13, 6))
df['Core_Speed'].fillna(0).astype(int).groupby(df['Release_Year']).mean().plot(kind='line')
df['Core_Speed'].fillna(0).astype(int).groupby(df['Release_Year']).max().plot(kind='line')
plt.title('Core speed in MHz by release year')
plt.xlabel('Release year')
plt.ylabel('Core speed MHz')
plt.legend(['Mean','Max'])
plt.xlim(2004,2016)
plt.show()


# In[ ]:


print(df.ix[df['Core_Speed'].fillna(0).astype(int).idxmax()][['Name','Core_Speed']])


# The most performant card in this dataset in terms of core speed is the GeForce GTX 1080 Asus ROG Strix Gaming OC 8GB, at 1784 MHz.
# 
# ### 5.2 Memory  <a class="anchor" id="memory"></a>

# In[ ]:


plt.figure(figsize=(13, 6))
df['Memory'].str[:-3].fillna(0).astype(int).groupby(df['Release_Year']).mean().plot(kind='line')
df['Memory'].str[:-3].fillna(0).astype(int).groupby(df['Release_Year']).median().plot(kind='line')
plt.title('Memory in MB by release year')
plt.xlabel('Release year')
plt.ylabel('Memory MB')
plt.legend(['Mean','Median'])
plt.show()


# ### 5.3 Memory speed and bandwidth  <a class="anchor" id="memory-speed"></a>

# In[ ]:


fig, ax1=plt.subplots(figsize=(13,6))
ax = df['Memory_Bandwidth'].fillna(0).astype(float).groupby(df['Release_Year']).mean().plot(kind='line', zorder=9999); 
df['Memory_Speed'].str[:-5].fillna(0).astype(float).groupby(df['Release_Year']).mean().plot(ax=ax, kind='line',secondary_y=True)
ax.set_ylabel('Memory Speed MHz', fontsize=10);

plt.title('Mean memory bandwidth and speed by release year')
plt.xlabel('Release year')
plt.ylabel('Memory bandwidth GB/sec')
plt.show()


# ### 5.4 Max power <a class="anchor" id="max-power"></a>

# In[ ]:


plt.figure(figsize=(13, 6))
df['Max_Power'].str[:-5].fillna(0).astype(float).groupby(df['Release_Year']).mean().plot(kind='line')
df['Max_Power'].str[:-5].fillna(0).astype(float).groupby(df['Release_Year']).median().plot(kind='line')
plt.title('Maximum power capacity of GPU in Watts by release year')
plt.xlabel('Release year')
plt.ylabel('Max power Watts')
plt.legend(['Mean','Median'])
plt.show()


# ### 5.5 Number of texture mapping units<a class="anchor" id="tmu"></a>

# In[ ]:


plt.figure(figsize=(13, 6))
df['TMUs'].groupby(df['Release_Year']).mean().plot(kind='line')
df['TMUs'].groupby(df['Release_Year']).max().plot(kind='line')
plt.title('TMU value by release year')
plt.legend(['Mean','Max'])
plt.xlabel('Release year')
plt.ylabel('TMU value')
plt.xlim(2001,)
plt.show()


# ### 5.6 Texture Rate<a class="anchor" id="texture-rate"></a>

# In[ ]:


plt.figure(figsize=(13, 6))
df['Texture_Rate'].str[:-9].astype(float).groupby(df['Release_Year']).mean().plot(kind='line')
df['Texture_Rate'].str[:-9].astype(float).groupby(df['Release_Year']).median().plot(kind='line')
plt.title('Texture rate by release year')
plt.legend(['Mean','Median'])
plt.xlabel('Release year')
plt.ylabel('Texture rate GTexel/s')
plt.xlim(2001,)
plt.show()


# ### 5.7 Pixel Rate<a class="anchor" id="pixel-rate"></a>

# In[ ]:


plt.figure(figsize=(13, 6))
df['Pixel_Rate'].str[:-9].astype(float).groupby(df['Release_Year']).mean().plot(kind='line')
df['Pixel_Rate'].str[:-9].astype(float).groupby(df['Release_Year']).median().plot(kind='line')
df['Pixel_Rate'].str[:-9].astype(float).groupby(df['Release_Year']).max().plot(kind='line')

plt.title('Pixel rate by release year')
plt.legend(['Mean','Median','Max'])
plt.xlabel('Release year')
plt.ylabel('Texture rate')
plt.xlim(2001,)
plt.show()


# ### 5.8 Process<a class="anchor" id="process"></a>

# In[ ]:


plt.figure(figsize=(13, 6))
df['Process'].str[:-2].astype(float).groupby(df['Release_Year']).mean().plot(kind='line')
df['Process'].str[:-2].astype(float).groupby(df['Release_Year']).min().plot(kind='line')
plt.title('Process by release year')
plt.legend(['Mean','Min'])
plt.xlabel('Release year')
plt.ylabel('Process Nm')
plt.xlim(2001,)
plt.show()


# ### 5.9 Price<a class="anchor" id="price"></a>

# In[ ]:


plt.figure(figsize=(13, 6))
df['Release_Price'].groupby(df['Release_Year']).mean().plot(kind='line')
df['Release_Price'].groupby(df['Release_Year']).median().plot(kind='line')
plt.title('Price by release year')
plt.legend(['Mean','Median'])
plt.xlabel('Release year')
plt.ylabel('Price $')
plt.xlim(2006,)
plt.show()


# In[ ]:


print(df.ix[df['Release_Price'].fillna(0).astype(int).idxmax()][['Name','Release_Price','Release_Year']])


# The most expensive card in the dataset is the Quadro Plex 7000, at $15k. This was one of the first GPUs with 4k support.
# 
# ## 6.0 Price/performance ratios<a class="anchor" id="ratios"></a>

# In[ ]:


plt.figure(figsize=(13, 6))
df['Ratio_Rate']=df['Release_Price']/(df['Texture_Rate'].str[:-9].fillna(0).astype(int))
df['Ratio_Speed']=df['Release_Price']/(df['Memory_Speed'].str[:-5].fillna(0).astype(int))
df['Ratio_BW']=df['Release_Price']/(df['Memory_Bandwidth'].fillna(0).astype(int))
df['Ratio_Memory']=df['Release_Price']/(df['Memory'].str[:-3].fillna(0).astype(int))

df['Ratio_Memory'].groupby(df['Release_Year']).median().plot(kind='line')
df['Ratio_BW'].groupby(df['Release_Year']).median().plot(kind='line')
df['Ratio_Speed'].groupby(df['Release_Year']).median().plot(kind='line')
df['Ratio_Rate'].groupby(df['Release_Year']).median().plot(kind='line')
plt.title('Price/performance ratio')
plt.legend(['Texture_Rate','Memory_Speed','Memory_Bandwidth','Memory'])
plt.xlabel('Release year')
plt.ylabel('Price to metric ratio')
plt.xlim(2005,)
plt.show()


# As we saw in section 5, the performance metrics have increased over time but the price has remained generally stable. This means price/performance ratios decrease over time and that you essentially get more performance for less money.
# 
# ## 7.0 GTX 1080<a class="anchor" id="gtx"></a>

# In[ ]:


print(len(df[df.Name.str.contains('GTX 1080')]['Name']))


# There are 76 model variations of the Nvidia GTX 1080.
# 
# ## 8.0 Conclusion<a class="anchor" id="conclusion"></a>
# 
# This EDA has shown the strong increase in GPU performance over recent years, but median prices stay approximately constant at ~250$, even despite inflation. Some metrics such as memory seem to be exponentially increasing. Price/performance ratios show that higher performance has generally gotten cheaper in the last few years and continues to decrease.
# 
# It seems the maximum values in 2017 for performance metrics are less than previous years. One reason for this could perhaps be the move to research grade GPUs and TPUs for very high performance which aren't included in this dataset e.g. Tesla K80, P100. An update to the dataset over time to include these for analysis would be interesting.
