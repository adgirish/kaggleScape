
# coding: utf-8

# 
# > "Of the 95% of Japanese that eat three meals a day, most people consider dinner to be the most important. More than 80% of them usually have dinner at home with their families. But as for what they actually eat, over 60% of Japanese rely on home meal replacement (ready-to-eat food bought elsewhere and taken home) at least once or twice a month. And more than 70% enjoy dining out at least once or twice monthly. This is the picture that emerged when Trends in Japan conducted an online survey concerning attitudes among Japanese people toward eating" [Japanese Online Survey] (http://)http://web-japan.org/trends01/article/020403fea_r.html 
# 
# 
# This notebook is exploratory data analysis and divided into the following:
# 
# Business Analysis:
# 1. Geographical distribution of the store 
# 2. Number of visitors  trend  
# 3. Visitors by genre
# 4. Visitors by weekdays trend and group size
# 5. Reservations trends
# 6. Productivity by air_store_id
# 
# Feature Engineering
# 
# Outliers
# 

# Basic information of the dataset:

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import ensemble, neighbors, linear_model, metrics, preprocessing
from datetime import datetime
import glob, re
import time, datetime
from datetime import timedelta

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

# from JdPaletto & the1owl1
# JdPaletto - https://www.kaggle.com/jdpaletto/surprised-yet-part2-lb-0-503?scriptVersionId=1867420
# the1owl1 - https://www.kaggle.com/the1owl/surprise-me
start1 =time.time()
data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])# bring air id to hpg reserve data
data['hs'] = pd.merge(data['hs'], data['id'], how='inner', on=['hpg_store_id'])# bring air id to hpg stores

print('Data structure.......................')
print('Training data....',data['tra'].shape)
print('Unique store id in training data',len(data['tra']['air_store_id'].unique()))
print('Id data....',data['id'].shape)
print('Air store data....',data['as'].shape,'& unique-',data['as']['air_store_id'].unique().shape)
print('Hpg store data....',data['hs'].shape,'& unique-',data['hs']['hpg_store_id'].unique().shape)
print('Air reserve data....',data['ar'].shape,'& unique-',data['ar']['air_store_id'].unique().shape)
print('Hpg reserve data....',data['hr'].shape,'& unique-',data['hr']['air_store_id'].unique().shape)
      
#converting datetime to date for reservation data
for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_hour'] = data[df]['visit_datetime'].dt.hour
    data[df]['visit_date'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_hour'] = data[df]['reserve_datetime'].dt.hour
    data[df]['reserve_date'] = data[df]['reserve_datetime'].dt.date
    
    data[df+'_hour'] = data[df]#keeping original
        
    #calculate reserve time difference and summarizing ar,hr to date
    data[df]['reserve_day_'+df] = data[df].apply(
        lambda r: (r['visit_date'] - r['reserve_date']).days, axis=1)
    data[df] = data[df].groupby(['air_store_id','visit_date'], as_index=False)[[
        'reserve_day_'+df, 'reserve_visitors']].sum().rename(columns={'reserve_visitors':'reserve_visitors_'+df})
    
#breaking down dates on training data & summarizing 
data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['day'] = data['tra']['visit_date'].dt.day
data['tra']['dow'] = data['tra']['visit_date'].dt.weekday
data['tra']['dow_name'] = data['tra']['visit_date'].dt.weekday_name
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['week'] = data['tra']['visit_date'].dt.week
data['tra']['quarter'] = data['tra']['visit_date'].dt.quarter
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date
data['tra']['year_mth'] = data['tra']['year'].astype(str)+'-'+data['tra']['month'].astype(str)


#extracting store id and date info from test data
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['day'] = data['tes']['visit_date'].dt.day
data['tes']['dow'] = data['tes']['visit_date'].dt.weekday
data['tes']['dow_name'] = data['tes']['visit_date'].dt.weekday_name
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['week'] = data['tes']['visit_date'].dt.week
data['tes']['quarter'] = data['tes']['visit_date'].dt.quarter
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date
data['tes']['year_mth'] = data['tes']['year'].astype(str)+'-'+data['tes']['month'].astype(str)

#extract unique stores based on test data and populate dow 1 to 6
unique_stores = data['tes']['air_store_id'].unique()#extract unique stores id from test data

store_7days = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) 
                    for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
store_sum = pd.DataFrame({'air_store_id': unique_stores})

# mapping train data dow to stores(test data) - min, mean, median, max, count 
tmp = data['tra'].groupby(['air_store_id'], as_index=False)[
    'visitors'].sum().rename(columns={'visitors':'total_visitors'})
store_7days = pd.merge(store_7days, tmp, how='left', on=['air_store_id']) 
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)[
    'visitors'].mean().rename(columns={'visitors':'mean_visitors'})
store_7days = pd.merge(store_7days, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)[
    'visitors'].median().rename(columns={'visitors':'median_visitors'})
store_7days = pd.merge(store_7days, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)[
    'visitors'].max().rename(columns={'visitors':'max_visitors'})
store_7days = pd.merge(store_7days, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)[
    'visitors'].count().rename(columns={'visitors':'count_observations'})
store_7days = pd.merge(store_7days, tmp, how='left', on=['air_store_id','dow']) 
# map stores(test) to store genre and location detail
store_7days = pd.merge(store_7days, data['as'], how='left', on=['air_store_id']) 
#map to hpg genre and area
store_7days = pd.merge(store_7days, data['hs'][['air_store_id','hpg_genre_name','hpg_area_name']], 
                       how='left', on=['air_store_id']) 

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

hf=data['hol']['holiday_flg']
dw=data['hol']['day_of_week']
data['hol']['long_wknd']=0

for i in range(len(data['hol'])):
    if (hf[i]==1)&(dw[i]=='Friday'):
        data['hol']['long_wknd'][i]=1
        data['hol']['long_wknd'][i+1]=1
        data['hol']['long_wknd'][i+2]=1
          
    if (hf[i]==1)&(dw[i]=='Monday'):
        data['hol']['long_wknd'][i]=1
        data['hol']['long_wknd'][i-1]=1
        data['hol']['long_wknd'][i-2]=1


train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 
train = pd.merge(train, store_7days, how='left', on=['air_store_id','dow']) 
test = pd.merge(test, store_7days, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])

#col = [c for c in train if c not in ['id', 'air_store_id','visit_date','visitors']]

#calculate qoq
qoq= train.groupby(['air_store_id','year','quarter'])['visitors'].sum()
qoq=qoq.unstack(0)
qoq=pd.DataFrame(qoq.to_records())
qoq=qoq.transpose()
qoq.drop(['year','quarter'],inplace=True)
qoq['2016Q2']=qoq[1]/qoq[0]*100
qoq['2016Q3']=qoq[2]/qoq[1]*100
qoq['2016Q4']=qoq[3]/qoq[2]*100
qoq['2017Q1']=qoq[4]/qoq[3]*100
lst=['2016Q2','2016Q3','2016Q4','2017Q1']
qoq=qoq[lst]
qoq['qoq_count']=qoq.apply(lambda x: x.count(), axis=1) 
qoq['qoq_growth']=qoq.apply(lambda x: x[x>100].count(), axis=1)
qoq['qoq_growth_pct'] = round(qoq['qoq_growth'] /qoq['qoq_count'],2)
qoq.index.names=['air_store_id']
qoq.reset_index(inplace=True)

train=pd.merge(train, qoq, how='left', on='air_store_id')

train = train.fillna(0) #change to one for algo training
test = test.fillna(0)
#df=df.rename(columns = {'two':'new_name'})
train['v_no_reservation']=train['visitors']-train['reserve_visitors_ar']-train['reserve_visitors_hr']
print(round(time.time()-start1,4))


# Time series of the data set

# In[ ]:


print('Dates................')
print('train date- ,',train['visit_date'].min(),' to ',train['visit_date'].max())
print('test date - ,',test['visit_date'].min(),' to ',test['visit_date'].max())
print('holiday df- ,',data['hol']['visit_date'].min(),' to ',data['hol']['visit_date'].max())


# **1. Geographical distribution of the store and holidays in Japan**

# Number of location of the stores- grouped by the same latitude + longitude

# In[ ]:


print(len(store_7days.groupby(['latitude','longitude'])['latitude','longitude'].size().reset_index()), 'physical stores')


# Stores location accross Japan in geographical heatmap

# In[ ]:


import folium
from folium import plugins

location =store_7days.groupby(['latitude', 'longitude']).size().reset_index()
locationheat = location[['latitude', 'longitude']]
locationheat = locationheat.values.tolist()

map1 = folium.Map(location=[39, 139], 
                        tiles = "Stamen Watercolor",# width=1000, height=500,
                        zoom_start = 5)
heatmap=plugins.HeatMap(locationheat).add_to(map1)
map1


# Store and their genres accross Japan. Probably a physical store has multiple genres with different air_store_id

# In[ ]:


location =store_7days.groupby(['air_store_id','air_genre_name'])['latitude','longitude'].mean().reset_index()
locationlist = location[['latitude', 'longitude']]
locationlist = locationlist.values.tolist()
map2 = folium.Map(location=[39, 139], 
                        tiles = "Stamen Toner",# width=1000, height=500,
                        zoom_start = 5)
marker_cluster=plugins.MarkerCluster().add_to(map2)
for point in range(0, len(location)):
    folium.Marker(locationlist[point], popup=location['air_genre_name'][point], 
    icon=folium.Icon(color='white', icon_color='red', 
                     #icon='fa fa-info-circle',
                     icon='fa fa-circle-o-notch fa-spin',
                     angle=0, 
                     prefix='fa')).add_to(marker_cluster)
map2


# Visualising holidays and weekend for the prediction period. There will be 3 consecutive holiday in May.

# In[ ]:


data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_month'] = data['hol']['visit_date'].dt.day
data['hol']['day'] = data['hol']['visit_date'].dt.weekday
data['hol']['week'] = data['hol']['visit_date'].dt.week
data['hol']['month'] = data['hol']['visit_date'].dt.month
data['hol']['quarter'] = data['hol']['visit_date'].dt.quarter
data['hol']['year'] = data['hol']['visit_date'].dt.year
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

def wkn(x):
    if x>4:
        return 1
    else:
        return 0

data['hol']['weekend']=data['hol']['day'].apply(wkn)

hols201704=data['hol'][(data['hol']['year']==2017)&(data['hol']['month']==4)]
hols=hols201704[['day_month','holiday_flg']].set_index('day_month')
wknd=hols201704[['day_month','weekend']].set_index('day_month')
hols201705=data['hol'][(data['hol']['year']==2017)&(data['hol']['month']==5)]
hols2=hols201705[['day_month','holiday_flg']].set_index('day_month')
wknd2=hols201705[['day_month','weekend']].set_index('day_month')

f, ax=plt.subplots(2,1, figsize=(15,4))
hols.plot(kind='bar', ax=ax[0], color='b')
wknd.plot(kind='bar', ax=ax[0], color='grey')
hols2.plot(kind='bar', ax=ax[1], color='b')
wknd2.plot(kind='bar', ax=ax[1], color='grey')
ax[0].set_title('April & May 2017 Holidays & Weekends')


# **2. Number of visitors and genres trend **

# Visitors and reservation time series. Walk in visitors are far more than customers with reservation. A drastic drop in number of visitors during new year due to store closure at certain places.
# 

# In[ ]:


#Visitor each day
f,ax = plt.subplots(1,1,figsize=(15,8))
plt1 = train.groupby(['visit_date'], as_index=False).agg({'visitors': np.sum})
plt2 = train.groupby(['visit_date'], as_index=False).agg({'reserve_visitors_ar': np.sum})
plt3 = train.groupby(['visit_date'], as_index=False).agg({'reserve_visitors_hr': np.sum})
plt1=plt1.set_index('visit_date')
plt2=plt2.set_index('visit_date')
plt3=plt3.set_index('visit_date')
plt1.plot(color='salmon', kind='area', ax=ax)
plt2.plot(color='cornflowerblue', kind='line', ax=ax)
plt3.plot(color='y', kind='line', ax=ax)
plt.ylabel("Sum of Visitors")
plt.title("Visitor and Reservations")


# Number of air_store_id  increased by 150% in mid 2016

# In[ ]:


f,ax = plt.subplots(1,1, figsize=(15,5))
genre= train.groupby(['visit_date'])['air_store_id'].size()
genre.plot(kind='area',  color= 'chocolate', grid=True, ax=ax, legend=True)
plt.ylabel("Number Stores")
plt.title("Number Unique Store ID")


# **3. Visitors by genre**

# Total visitor by air_genre_name. Izakaya & Cafe/Sweets are the two  most populr genres

# In[ ]:


f,ax=plt.subplots(1,1, figsize=(10,8))
genre=train.groupby(['air_genre_name'],as_index=False)['visitors'].sum()
genre.sort_values(by='visitors', ascending=True, inplace=True)
genre['air_genre'] =[i for i,x in enumerate(genre['air_genre_name'])] 
genre = genre.sort_values(by='visitors', ascending=False)#.reset_index()
my_range = genre['air_genre']
plt.hlines(y=my_range, xmin=0, xmax=genre['visitors'], color='goldenrod',alpha=0.8) #[‘solid’ | ‘dashed’ | ‘dashdot’ | ‘dotted’]
plt.plot(genre['visitors'], my_range, "o",markersize=25,label='visitors',color='orangered')

# Add titles and axis names
plt.yticks(my_range, genre['air_genre_name'],fontsize=15)
plt.title("Total visitors by air_genre_name", loc='center')
plt.xlabel('Score')
plt.ylabel('Features')
#plt.legend()


# 
# - some of genres are new addition like - international cuisine, karaoke/party
# - Yakiniku/korean genre is showing increasing demand
# - Numbers of visitors for karaoke/party frequently surge in numbers on weekends

# In[ ]:


ax = sns.FacetGrid(train, col="air_genre_name", col_wrap=4, size=3, hue='air_genre_name',margin_titles=True,
                  aspect=1.5, palette='husl', ylim=(0,150))
ax = ax.map(plt.plot, "visit_date", "visitors",  marker=".", linewidth = 0.5)


# **4. Visitors by weekdays trend and their group size**

# Friday, Saturday and Sunday are the busiest days of the week

# In[ ]:


pvt=pd.pivot_table(train, index=['year','week'], columns='dow',values='visitors',aggfunc=[np.mean],fill_value=0)
pvt=pd.DataFrame(pvt.to_records())
pvt.columns=[pvt.replace("('mean', ", "").replace(")", "") for pvt in pvt.columns]
pvt['year_week']=pvt['year'].astype(str) +'-'+ pvt['week'].astype(str)
pvt=pvt.set_index('year_week')
pvt.drop(['year','week'], axis=1,inplace=True)
f, ax=plt.subplots(1,1, figsize=(15,8))
pvt.plot(kind='line', ax=ax,cmap='inferno')
plt.ylabel("Sum of Visitors")
plt.xlabel("Week")
plt.title("Visitors by Day of the Week ")


# In[ ]:


print('Number of total visitors- ', train['visitors'].sum())
print('Number of stores- ', )
print('Number of average daily visitors per air_store_id-', round(train['visitors'].mean(),2))


# Visitors, group size and daily trend:
# Generally, highest number of visitors is on Friday, Saturday and Sunday. The peak day of the week is Saturday, while if it is holiday the peak day will be Thursday and Friday. 

# In[ ]:


max_date=max(train['visit_date'])
one_year = datetime.timedelta(days=364)
cmap='inferno'
year_ago= max_date - one_year
train2=train#[train['visit_date']>year_ago]
pvt=train2.groupby(['dow','dow_name'])['visitors'].mean().reset_index()

train2=train.loc[(train['day']<8)&(train['holiday_flg']==1)]
pvt2=train2.groupby(['dow','dow_name'])['visitors'].mean().reset_index()

train3=train.loc[train['holiday_flg']==1]
pvt3=train3.groupby(['dow','dow_name'])['visitors'].mean().reset_index()
train4=train.loc[(train['long_wknd']==1)]
pvt4=train4.groupby(['dow','dow_name'])['visitors'].mean().reset_index()

pvt5=pd.pivot_table(train, index=['dow'], columns='month',values='visitors',aggfunc=[np.mean],fill_value=0)#.reset_index()
pvt5=pd.DataFrame(pvt5.to_records())
pvt5.columns=[pvt5.replace("('mean', ", "").replace(")", "") for pvt5 in pvt5.columns]
pvt5=pvt5.set_index('dow')

f, ax=plt.subplots(2,2, figsize=(15,10), sharey=False)
ax[0,0].bar(pvt['dow'] ,pvt['visitors'],color='darkturquoise')
ax[0,1].bar(pvt2['dow'] ,pvt2['visitors'],color='slategrey')
ax[1,0].bar(pvt3['dow'] ,pvt3['visitors'],color='thistle')
sns.heatmap(pvt5, ax=ax[1,1],cmap=cmap)
ax[0,0].set_title('Mean Daily Visitors')
ax[0,1].set_title('Mean Daily Visitors on first 7 days of the Month')
ax[1,0].set_title('Mean Daily Visitors on holiday')
ax[1,1].set_title('DOW vs Month mean visitors')

ax[0,0].set_ylim(0,30)
ax[0,1].set_ylim(0,30)
ax[1,0].set_ylim(0,30)
#ax[1,1].set_xlim(0,100)
#plt.xlabel("Month")


# Larger mean visitors on holiday except day 5 - Saturday.

# In[ ]:


plt1=train['visitors'].value_counts().reset_index().sort_index()
fig, ax = plt.subplots(figsize=(15, 6), nrows=1, ncols=2, sharex=False, sharey=False)
ax[0].bar(plt1['index'] ,plt1['visitors'],color='limegreen')
ax[1]= sns.boxplot(y='visitors',x='dow', data=train,hue='holiday_flg',palette="Set3")
ax[1].set_title('Number of daily visitors by day of the week')
ax[0].bar(plt1['index'] ,plt1['visitors'],color='limegreen')
ax[0].set_title('Frequency')
ax[0].set_xlim(0,100)
ax[1].set_ylim(0,100)
ax[1].legend(loc=1)


# **5. Reservations trends**

# In[ ]:


print('Total air reserve visitors - ',data['ar_hour']['reserve_visitors'].sum())
print('Total hpg reserve visitors - ',data['hr_hour']['reserve_visitors'].sum())


# Most of the customers are making reservation for the period of one week or visited restaurant on day 4 (Friday) & day 5 (Saturday) the most. There is noticably different reservation behaviour between air and hpg customers, where hpg visitors tend to book reservation throughout the day but air visitors doing it at later part of the day

# In[ ]:


data['ar_hour']['dow_reserve'] = data['ar_hour']['reserve_datetime'].dt.weekday
data['ar_hour']['dow_visit'] = data['ar_hour']['visit_datetime'].dt.weekday
data['hr_hour']['dow_reserve'] = data['hr_hour']['reserve_datetime'].dt.weekday
data['hr_hour']['dow_visit'] = data['hr_hour']['visit_datetime'].dt.weekday
air_res= data['ar_hour'].groupby(['reserve_day_ar'],as_index=False)['reserve_visitors'].sum()[:40]
hpg_res= data['hr_hour'].groupby(['reserve_day_hr'],as_index=False)['reserve_visitors'].sum()[:40]
air_res2=data['ar_hour'].groupby(['dow_visit',],as_index=False)['reserve_visitors'].sum()
hpg_res2=data['hr_hour'].groupby(['dow_visit',],as_index=False)['reserve_visitors'].sum()
air_res3=data['ar_hour'].groupby(['reserve_hour','visit_hour'])['reserve_visitors'].sum().unstack()
hpg_res3=data['hr_hour'].groupby(['reserve_hour','visit_hour'])['reserve_visitors'].sum().unstack()

f, ax=plt.subplots(3,2, figsize=(15,12),sharey=False)
ax[0,0].bar(air_res['reserve_day_ar'] ,air_res['reserve_visitors'],color='royalblue')
ax[0,1].bar(hpg_res['reserve_day_hr'] ,hpg_res['reserve_visitors'],color='tomato')
ax[1,0].bar(air_res2['dow_visit'] ,air_res2['reserve_visitors'],color='royalblue')
ax[1,1].bar(hpg_res2['dow_visit'] ,hpg_res2['reserve_visitors'],color='tomato')
sns.heatmap(air_res3, ax=ax[2,0],cmap='inferno')
sns.heatmap(hpg_res3, ax=ax[2,1],cmap='inferno')
ax[0,0].set_title('Air Reservation in Number of Days')
ax[0,1].set_title('Hpg Reservation in Number of Days')
ax[1,0].set_title('Air reserve visitors by dow')
ax[1,1].set_title('Hpg reserve visitors by dow')
ax[2,0].set_title('Air Reserve Hour vs Visit hour')
ax[2,1].set_title('Hpg Reserve Hour vs Visit hour')


# **6. Productivity by air_store_id**

# In[ ]:


store_mean= train.groupby(['air_store_id'], as_index=False)['visitors'].mean().rename(columns={'visitors':'overall_mean'})
train=pd.merge(train, store_mean, how = 'left',on='air_store_id')

train['vis_qtl']=pd.qcut(train['overall_mean'], 4, labels=['Quartile 4','Quartile 3','Quartile 2','Quartile 1'])
quartile=train.groupby(['vis_qtl'],as_index=False).agg({'air_store_id':lambda x: len(x.unique()),
                                               'mean_visitors':lambda x: x.mean(),
                                               'visitors':lambda x: x.sum()})
quartile.rename(columns={'air_store_id':'stores', 'visitors':'total_visitors'},inplace=True)
quartile.sort_values(by='total_visitors', ascending=False,inplace=True)
quartile['cumulative_visitors'] = quartile['total_visitors'].cumsum()/quartile['total_visitors'].sum()
quartile


# We stacked and ordered the store according to average productiviy per day and arrange it form best to the lowest and cut the number of air_store_id into 4. "air_store_id" in Quartile 1 and Quartile 2 (412 which 50% of the total) contribute to 70% of the total visitors.

# In[ ]:


tot_visitors = quartile[['vis_qtl','total_visitors']]
tot_visitors2 = quartile[['vis_qtl','cumulative_visitors']]
tot_visitors.set_index('vis_qtl',inplace=True)
tot_visitors2.set_index('vis_qtl',inplace=True)
quartile.sort_values(by='total_visitors', ascending=False,inplace=True)
f, ax=plt.subplots(1,2, figsize=(12,4))
tot_visitors.plot(kind='bar',  ax=ax[0],color='y',width=0.8)
tot_visitors2.plot(kind='bar',  ax=ax[1],color='darkseagreen',width=0.8)
ax[0].set_title('Total visitors by Store Productivity Quartile')
ax[1].set_title('%Cumulative visitors  by Store Productivity Quartile')


# **Feature Engineering**
# 
# Visualising feature engineering. We are exploring options and will test all of them in our model. 

# 
# Outliers
# Identifying outliers and treating them accordingly in our model building. Intuitively, area and dow are probably a good grouping to programatically find outliers. I believe that area and dow are among two most important factors that determines business traffic as it has similar holiday, celebration, culture and daily traffic.
# note: ol_1 shows number of observations that have number visitors 1 standard deviation above the mean for air_are_name and dow grouping.

# In[ ]:


# outliers based on air_area_name & dow grouping
area_dow_std_df=train.groupby(['air_area_name','dow'])['visitors'].std().reset_index().rename(columns={'visitors':'std_area_dow'})
area_dow_mean_df=train.groupby(['air_area_name','dow'])['visitors'].mean().reset_index().rename(columns={'visitors':'mean_area_dow'})
train2=pd.merge(train, area_dow_std_df, how="left", on=['air_area_name','dow'])
train2=pd.merge(train2, area_dow_mean_df, how="left", on=['air_area_name','dow'])

x=train2['visitors']
y=train2['mean_area_dow']
z=train2['std_area_dow']
ol_df=[]
for n in range(10):
    train2['ol_{}'.format(n)]= [1 if (x>y+z*n) else 0 for x, y, z in zip(x,y,z)]
    ol_dfs=train2['ol_{}'.format(n)].value_counts()
    ol_df.append(ol_dfs)
    
ol_df=pd.DataFrame(ol_df)   
ol_df.index.name='outliers'

f, ax=plt.subplots(1,1, figsize=(8,5))
ol_df.plot(kind='barh',width=1,ax=ax, color=['dodgerblue','violet'])
ax.set_title('Count of Outliers Base on multiple std from mean - grouped by air_area_name & dow')
ax.set_ylabel('X* Standard Deviation From Mean')
# Adding a title and a subtitle
plt.text(x = 100000, y = 11, s = "Outliers",fontsize = 25, weight = 'bold', alpha = .75)


# **Please Upvote if you find it useful**
