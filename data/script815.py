
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import os
#print(os.listdir("../input"))

import time

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#make wider graphs
sns.set(rc={'figure.figsize':(12,5)});
plt.figure(figsize=(12,5));


# This notebook is built using some of ideas in the book by *anokas* at  https://www.kaggle.com/anokas/talkingdata-adtracking-eda  .  I suggest reviewing that book first.
# 
# For this part I use  the first 10 million rows from train ( as the complete sets are too big for this kernel) and the full test data.
# 
# Note:  in the notebook I sometimes refer to downloads as conversions.  i.e. is_attributed == 1  means the click converted.

# In[ ]:


#import first 10,000,000 rows of train and all test data
train = pd.read_csv('../input/train.csv', nrows=10000000)
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# ip, app, device, os and channel are actually categorical variables encoded as integers.   Set them as categories for analysis.

# In[ ]:


variables = ['ip', 'app', 'device', 'os', 'channel']
for v in variables:
    train[v] = train[v].astype('category')
    test[v]=test[v].astype('category')


# Convert date stamps to date/time type.

# In[ ]:


#set click_time and attributed_time as timeseries
train['click_time'] = pd.to_datetime(train['click_time'])
train['attributed_time'] = pd.to_datetime(train['attributed_time'])
test['click_time'] = pd.to_datetime(test['click_time'])

#set as_attributed in train as a categorical
train['is_attributed']=train['is_attributed'].astype('category')


# Now lets do a quick inspection of train and test data main statistics

# In[ ]:


train.describe()


# *this graph is adapted from https://www.kaggle.com/anokas/talkingdata-adtracking-eda*:

# In[ ]:


plt.figure(figsize=(10, 6))
cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [len(train[col].unique()) for col in cols]
sns.set(font_scale=1.2)
ax = sns.barplot(cols, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature (from 10,000,000 samples)')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
# for col, uniq in zip(cols, uniques):
#     ax.text(col, uniq, uniq, color='black', ha="center")


# Quick check to make sure that Nan values in 'attribute_time' are only for samples that did not convert.  Check that counts of 'attributed_time' values is same as count of converted clicks.

# In[ ]:


#double check that 'attributed_time' is not Null for all values that resulted in download (i.e. is_attributed == 1)
train[['attributed_time', 'is_attributed']][train['is_attributed']==1].describe()


# In[ ]:


#set click_id to categorical, for cleaner statistics view
test['click_id']=test['click_id'].astype('category')
test.describe()


# **Quick Notes/Observations** :
# - There are only 18717 attributed_time values.  This means only 18,717 out of 10,000,000 clicks resulted in a download.  That's less than 0.2% !
# - There are ip adresses that trigger  a click over 50 thousand times.  Seems strange that one ip address would click that often in a span of just 4 days.  Does that mean that ip address encoded is not device id, but network id?  (explore this below)
# - First click in train set is on 2017-11-06 14:32:21.  Test clicks start on  2017-11-10.  Based on data specifications, train coveres a 4 day period.  This means that the train and test data do not overlap, but test data is taken the day after train data ends.
# -Train data is ordered by timestamp.  (therefore batches pulled in order cover limited time span)
# - 2017-11-06 was a Monday.   2017-11-10 was a Friday.  i.e. Train is Mon-Thur, Test is Friday
# -There is no missing data in Test.  Missing values in train appear to be only for attributed_time, where there isn't any value due to no app download.

# Only a small proportion of clicks were followed by a download:

# In[ ]:


plt.figure(figsize=(6,6))
#sns.set(font_scale=1.2)
mean = (train.is_attributed.values == 1).mean()
ax = sns.barplot(['App Downloaded (1)', 'Not Downloaded (0)'], [mean, 1-mean])
ax.set(ylabel='Proportion', title='App Downloaded vs Not Downloaded')
for p, uniq in zip(ax.patches, [mean, 1-mean]):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+0.01,
            '{}%'.format(round(uniq * 100, 2)),
            ha="center")


# ### Explore ip counts.  Check if multiple ips have any downloads.
# 
# At this point I was trying to figure out what 'ip' were actually encoding.  My original understanding that ips were user specific did not hold up to scrutiny.
# If ip repeated too many times, was it a bot?  This does not appear to be true, as repeated ips do convert.  See below:

# In[ ]:


#temporary table to see ips with their associated count frequencies
temp = train['ip'].value_counts().reset_index(name='counts')
temp.columns = ['ip', 'counts']
temp[:10]


# In[ ]:


#add temporary counts of ip feature ('counts') to the train table, to see if IPs with high counts have conversions
train= train.merge(temp, on='ip', how='left')


# In[ ]:


#check top 10 values
train[train['is_attributed']==1].sort_values('counts', ascending=False)[:10]


# In[ ]:


train[train['is_attributed']==1].ip.describe()


# So high frequency ip counts do get conversions.   Up to 56 downloads for one ip.  Each IP must be for some network with many devices.

# In[ ]:


#convert 'is_attributed' back to numeric for proportion calculations
train['is_attributed']=train['is_attributed'].astype(int)


# ### Conversion rates over Counts of 300 most popular IPs

# In[ ]:


proportion = train[['ip', 'is_attributed']].groupby('ip', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = train[['ip', 'is_attributed']].groupby('ip', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='ip', how='left')
merge.columns = ['ip', 'click_count', 'prop_downloaded']

ax = merge[:300].plot(secondary_y='prop_downloaded')
plt.title('Conversion Rates over Counts of 300 Most Popular IPs')
ax.set(ylabel='Count of clicks')
plt.ylabel('Proportion Downloaded')
plt.show()

print('Counversion Rates over Counts of Most Popular IPs')
print(merge[:20])


# Conversions are noisy and do not appear to correlate with how popular an IP is.

# ### Conversions by App
# 
# Check 100 most popular apps by click count:

# In[ ]:


proportion = train[['app', 'is_attributed']].groupby('app', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = train[['app', 'is_attributed']].groupby('app', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='app', how='left')
merge.columns = ['app', 'click_count', 'prop_downloaded']

ax = merge[:100].plot(secondary_y='prop_downloaded')
plt.title('Conversion Rates over Counts of 100 Most Popular Apps')
ax.set(ylabel='Count of clicks')
plt.ylabel('Proportion Downloaded')
plt.show()

print('Counversion Rates over Counts of Most Popular Apps')
print(merge[:20])


# There is a again a huge difference in clicks per app, with minimum of one click on an app and max at almost 13 million.  The proportion flucuates more as the counts go down, since each additional click has larger impact on the proportion value.  In general, for apps with counts in the thousands the ratio stays within 0.0001 - 0.0015 boundary.  For less popular apps it fluxuates more widely.  

# ### Conversions by OS
# Look at top 100 operating systems by click count

# In[ ]:


proportion = train[['os', 'is_attributed']].groupby('os', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = train[['os', 'is_attributed']].groupby('os', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='os', how='left')
merge.columns = ['os', 'click_count', 'prop_downloaded']

ax = merge[:100].plot(secondary_y='prop_downloaded')
plt.title('Conversion Rates over Counts of 100 Most Popular Operating Systems')
ax.set(ylabel='Count of clicks')
plt.ylabel('Proportion Downloaded')
plt.show()

print('Counversion Rates over Counts of Most Popular Operating Systems')
print(merge[:20])


# Same story. For values in the thousands the boundary on the ratio is very low, roughly between 0.0006 and 0.003, but as counts on OS become lower, the ratio starts fluxuating more wildely.

# ### Conversions by Device
# 
# Devices are extremely disproportionately distributed, with number one device used almost 94% of time.  For that device proportion download was 0.001326. (0.13%)

# In[ ]:


proportion = train[['device', 'is_attributed']].groupby('device', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = train[['device', 'is_attributed']].groupby('device', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='device', how='left')
merge.columns = ['device', 'click_count', 'prop_downloaded']

print('Count of clicks and proportion of downloads by device:')
print(merge)


# ### Conversions by Channel
# 

# In[ ]:


proportion = train[['channel', 'is_attributed']].groupby('channel', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = train[['channel', 'is_attributed']].groupby('channel', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='channel', how='left')
merge.columns = ['channel', 'click_count', 'prop_downloaded']

ax = merge[:100].plot(secondary_y='prop_downloaded')
plt.title('Conversion Rates over Counts of 100 Most Popular Apps')
ax.set(ylabel='Count of clicks')
plt.ylabel('Proportion Downloaded')
plt.show()

print('Counversion Rates over Counts of Most Popular Channels')
print(merge[:20])


# There appear to be a few peaks for channels at reasonable click quantity, but overall the pattern holds same as for categories above.  

# # Checking for time patterns
# 
# Round the click time down to an hour of the day to see if there are any hourly patterns.
# 
# For this part cannot use the first n rows from train data, as it's organized by time.  To get a genral idea for the pattern, will use train data from the randomly sampled 100000 train set provided by organizers.

# In[ ]:


train_smp = pd.read_csv('../input/train_sample.csv')


# In[ ]:


train_smp.head(7)


# In[ ]:


#convert click_time and attributed_time to time series
train_smp['click_time'] = pd.to_datetime(train_smp['click_time'])
train_smp['attributed_time'] = pd.to_datetime(train_smp['attributed_time'])


# In[ ]:


#round the time to nearest hour
train_smp['click_rnd']=train_smp['click_time'].dt.round('H')  

#check for hourly patterns
train_smp[['click_rnd','is_attributed']].groupby(['click_rnd'], as_index=True).count().plot()
plt.title('HOURLY CLICK FREQUENCY');
plt.ylabel('Number of Clicks');

train_smp[['click_rnd','is_attributed']].groupby(['click_rnd'], as_index=True).mean().plot()
plt.title('HOURLY CONVERSION RATIO');
plt.ylabel('Converted Ratio');


# There is no clear hourly time pattern in ratios, however there is a definete pattern in frequency of clicks based on time of day.
# 
# Lets extract the hour of day from each day as a separate feature, and see combined trend (merge the 4 days together by hour).

# In[ ]:


#extract hour as a feature
train_smp['click_hour']=train_smp['click_time'].dt.hour


# In[ ]:


train_smp.head(7)


# Let's check number of clicks by hour:

# In[ ]:


train_smp[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).count().plot(kind='bar', color='#a675a1')
plt.title('HOURLY CLICK FREQUENCY Barplot');
plt.ylabel('Number of Clicks');

train_smp[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).count().plot(color='#a675a1')
plt.title('HOURLY CLICK FREQUENCY Lineplot');
plt.ylabel('Number of Clicks');


# And number of conversions by hours:

# In[ ]:


train_smp[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).mean().plot(kind='bar', color='#75a1a6')
plt.title('HOURLY CONVERSION RATIO Barplot');
plt.ylabel('Converted Ratio');

train_smp[['click_hour','is_attributed']].groupby(['click_hour'], as_index=True).mean().plot( color='#75a1a6')
plt.title('HOURLY CONVERSION RATIO Lineplot');
plt.ylabel('Converted Ratio');


# let's overlay the two graphs to see if patterns correlate in any way

# In[ ]:


#adapted from https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales
#smonek's answer


group = train_smp[['click_hour','is_attributed']].groupby(['click_hour'], as_index=False).mean()
x = group['click_hour']
ymean = group['is_attributed']
group = train_smp[['click_hour','is_attributed']].groupby(['click_hour'], as_index=False).count()
ycount = group['is_attributed']


fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()

host.set_xlabel("Time")
host.set_ylabel("Proportion Converted")
par1.set_ylabel("Click Count")

#color1 = plt.cm.viridis(0)
#color2 = plt.cm.viridis(0.5)
color1 = '#75a1a6'
color2 = '#a675a1'

p1, = host.plot(x, ymean, color=color1,label="Proportion Converted")
p2, = par1.plot(x, ycount, color=color2, label="Click Count")

lns = [p1, p2]
host.legend(handles=lns, loc='best')

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')


# The proportions may be more reliable if estimated on full data.  With the random sample it's  hard too tell because the variability is too high, especially for the hours with low click counts.   i.e. the fewer clicks/conversions, the wider margin of the estimated conversion ratio.  (see below)

# In[ ]:


sns.barplot('click_hour', 'is_attributed', data=train_smp)
plt.title('HOURLY CONVERSION RATIO');
plt.ylabel('Converted Ratio');


# ### Look into attributed_time
# It could be useful to learn more about conversions that did take place.
# Let's see how much time passed from clicking on the ad to downloading it.

# In[ ]:


train_smp['timePass']= train_smp['attributed_time']-train_smp['click_time']
#check:
train_smp[train_smp['is_attributed']==1][:15]


# In[ ]:


train_smp['timePass'].describe()


# It takes as long as (almost) 20 hours to go from click to purchase and as little as 4 seconds.  
# 
# The 4 seconds seems to low to make a decision.  This person would have either seen the ad before, or already been aware of the product some other way.
# 
# Does that mean the ad was clicked on multiple times, but only one click was counted as conversion?   Or did the person click on the ad specifically with the intent to download?  (eg, if channel is something like google search, the ad could be clicked during search results view and app downloaded immediately because that's what the person intended to do right away)
# 
# Raises questions to explore:
#    - How accurately are conversions tracked? How are clicks and downloads linked?  What happens if download after multiple clicks?  Is there a way to identify likely same users (same IP, Device, etc...)

# ### Check actual train data (the first 10,000,000)
# double check the same feature on the first 10 million rows of train data:

# In[ ]:


#check first 10,000,000 of actual train data
train['timePass']= train['attributed_time']-train['click_time']
train['timePass'].describe()


# Here minimum time from click to download is virtually instanteneous.  How is this possible?  It is clearly not a result of a human decision made from clicking on an ad seen for the first time.
