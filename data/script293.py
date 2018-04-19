
# coding: utf-8

# # Historical Bitcoin Data Analysis

# In[ ]:


import numpy as np
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import seaborn as sns
import datetime, pytz

init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## A Basic Hodler's Account
# 
# Simple analysis of returns based on $10 USD bitcoin buy at 00:00 each Monday morning.
# 
# Credit for data cleaning: https://www.kaggle.com/smitad/bitcoin-trading-strategy-simulation

# In[ ]:


#define a conversion function for the native timestamps in the csv file
def dateparse (time_in_secs):    
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))


# In[ ]:


data = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv', parse_dates=[0], date_parser=dateparse)


# In[ ]:


# First thing is to fix the data for bars/candles where there are no trades. 
# Volume/trades are a single event so fill na's with zeroes for relevant fields...
data['Volume_(BTC)'].fillna(value=0, inplace=True)
data['Volume_(Currency)'].fillna(value=0, inplace=True)
data['Weighted_Price'].fillna(value=0, inplace=True)

# next we need to fix the OHLC (open high low close) data which is a continuous timeseries so
# lets fill forwards those values...
data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)

data.tail()


# In[ ]:


# create valid date range
start = datetime.datetime(2015, 1, 1, 0, 0, 0, 0, pytz.UTC)
end = datetime.datetime(2017, 10, 17, 20, 0, 0, 0, pytz.UTC)

# find rows between start and end time and find the first row (00:00 monday morning)
weekly_rows = data[(data['Timestamp'] >= start) & (data['Timestamp'] <= end)].groupby([pd.Grouper(key='Timestamp', freq='W-MON')]).first().reset_index()
weekly_rows.tail()


# If anyone knows why this command creates days that don't exist (Oct 23) in the dataset, let me know.

# In[ ]:


# create time series plot of account value v. investment
trace0 = go.Scatter(
    x = weekly_rows['Timestamp'],
    y = (weekly_rows.index+1)*10,
    mode = 'lines',
    name = 'Investment'
)
trace1 = go.Scatter(
    x = weekly_rows['Timestamp'],
    y = ((10.0 / weekly_rows['Close'].astype(float)).cumsum()) * weekly_rows['Close'].astype(float),
    mode = 'lines',
    name = 'Account Value'
)
trace2 = go.Scatter(
    x = weekly_rows['Timestamp'],
    y = weekly_rows['Close'].astype(float),
    mode = 'lines',
    name = 'Bitcoin Price'
)
plot_data = [trace0, trace1, trace2]
iplot(plot_data)


# ## Optimal Buy Times

# In[ ]:


# Create 'Day of Week' and 'Time Decimal' column for later
dayOfWeek={0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
data['Day of Week'] = data['Timestamp'].dt.dayofweek.map(dayOfWeek)

data['Time Decimal'] = data['Timestamp'].dt.hour + data['Timestamp'].dt.minute/60


# ### Best Day of Week to Buy
# 
# Determine which day of the week most often has the lowest price.

# In[ ]:


# find indices with min value of that week
idx = data.groupby([pd.Grouper(key='Timestamp', freq='W-MON')])['Close'].transform(min) == data['Close']

# remove duplicate day rows
weekly_lows = data[idx].groupby([pd.Grouper(key='Timestamp', freq='D')]).first().reset_index()


# In[ ]:


# create histogram for day of week
sns.countplot(x='Day of Week',data=weekly_lows, order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])


# In[ ]:


x = []
y = []
for key, day in dayOfWeek.items():
    x.append(day)
    y.append(weekly_lows['Day of Week'].value_counts()[day])
bar_tracer = [go.Bar(x=x, y=y)]
iplot(bar_tracer)


# ### Best Time of Day to Buy
# 
# Determine what time of day most often has the lowest price.

# In[ ]:


# find indices with min value of that day
daily_lows = data[data.groupby([pd.Grouper(key='Timestamp', freq='D')])['Close'].transform(min) == data['Close']]


# In[ ]:


sns.boxplot(x="Day of Week", y="Time Decimal", data=daily_lows, palette='rainbow')


# In[ ]:


box_tracer = []
for key, day in dayOfWeek.items():
    box_tracer.append(
        go.Box(
            y = daily_lows[daily_lows['Day of Week'] == day]['Time Decimal'],
            name = day
        )
    )
iplot(box_tracer)


# ### Tuesday Low Histogram
# 
# Let's take a closer look into the distrobution of the lows on Tuesdays.

# In[ ]:


sns.distplot(daily_lows[daily_lows['Day of Week'] == 'Tue']['Time Decimal'], bins=24, kde=False);


# In[ ]:


histo_tracer = [go.Histogram(x=daily_lows[daily_lows['Day of Week'] == 'Tue']['Time Decimal'])]
iplot(histo_tracer)

