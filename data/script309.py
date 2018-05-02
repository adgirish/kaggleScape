
# coding: utf-8

# In this tutorial, we'll try to forecast transactions data and poke around open source [Prophet](https://facebook.github.io/prophet/) library released by Facebook. We'll learn  how to integrate holiday periods and optimize the model by playing parameters.
# 
# Prophet is a procedure for forecasting time series data. It is based on an additive model where non-linear trends are fit with yearly and weekly seasonality, plus holidays. It works best with daily periodicity data with at least one year of historical data. Prophet is robust to missing data, shifts in the trend, and large outliers. You can learn more about from the [docs](https://facebook.github.io/prophet/docs/quick_start.html).
# 
# Let's start loading libraries and data.

# In[ ]:


# Load libraries
import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 70)
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()

from fbprophet import Prophet

get_ipython().run_line_magic('time', "df_transactions = pd.read_csv('../input/transactions.csv')")
get_ipython().run_line_magic('time', "df_holidays_events = pd.read_csv('../input/holidays_events.csv')")

print('Data and libraries are loaded.')


# In[ ]:


df_transactions


# If we look at the transactions data, transactions are grouped by store no. We'll simplify this for now and group them by date. We can clearly see seasonality and holiday effect on total transactions.

# In[ ]:


transactions = df_transactions.groupby('date')['transactions'].sum()
py.iplot([go.Scatter(
    x=transactions.index,
    y=transactions
)])


# Now, let's try the prophet library and see how well it predicts. But before that, we should prepare the data. According to docs:
# 
# > Prophet follows the sklearn model API. We create an instance of the Prophet class and then call its fit and predict methods.
# > The input to Prophet is always a dataframe with two columns: **ds** and **y**. The ds (datestamp) column must contain a date or datetime (either is fine). The **y** column must be numeric, and represents the measurement we wish to forecast.
# 

# In[ ]:


transactions = pd.DataFrame(transactions).reset_index()
transactions.columns = ['ds', 'y']
transactions


# In[ ]:


m = Prophet()
m.fit(transactions)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast


# In[ ]:


py.iplot([
    go.Scatter(x=transactions['ds'], y=transactions['y'], name='y'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),
    go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')
])


# In[ ]:


# Calculate root mean squared error.
print('RMSE: %f' % np.sqrt(np.mean((forecast.loc[:1682, 'yhat']-transactions['y'])**2)) )


# As we see in the graph above, prediction is fairly well and aligns with the data's up and downs. You can zoom in the graph by selecting a zoom area with mouse.
# 
# But the trend is fairly rigid, it misses the sub trends in mid-years. The trend is rising at first half of the year and a little bit slowing down after that. Let's make the trend a little bit flexible. If the trend changes are being overfit (too much flexibility) or underfit (not enough flexibility), you can adjust the strength of the sparse prior using the input argument **changepoint_prior_scale**. By default, this parameter is set to 0.05. Increasing it will make the trend more flexible. (https://facebook.github.io/prophet/docs/trend_changepoints.html)

# In[ ]:


m = Prophet(changepoint_prior_scale=2.5)
m.fit(transactions)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[ ]:


# Calculate root mean squared error.
print('RMSE: %f' % np.sqrt(np.mean((forecast.loc[:1682, 'yhat']-transactions['y'])**2)) )
py.iplot([
    go.Scatter(x=transactions['ds'], y=transactions['y'], name='y'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),
    go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')
])


# Now we'll add more seasonality to the model. As we see, prophet calculates weekly and yearly seasonality. We don't need daily seasonality, because we don't have intra-day data for this tutorial. Just adding monthly seasonality should be enough.

# In[ ]:


m = Prophet(changepoint_prior_scale=2.5)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(transactions)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[ ]:


# Calculate root mean squared error.
print('RMSE: %f' % np.sqrt(np.mean((forecast.loc[:1682, 'yhat']-transactions['y'])**2)) )
py.iplot([
    go.Scatter(x=transactions['ds'], y=transactions['y'], name='y'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),
    go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')
])


# Now it's time to add effects of holiday events into our model. We need to adjust data format first. Prophet needs two columns (holiday and ds) and a row for each occurrence of the holiday. We could also include columns lower_window and upper_window which extend the holiday out to `[lower_window, upper_window]` days around the date. But, I think data is arranged well and we don't have to pay so much attention to that.

# In[ ]:


df_holidays_events


# In[ ]:


holidays = df_holidays_events[df_holidays_events['transferred'] == False][['description', 'date']]
holidays.columns = ['holiday', 'ds']
#holidays['lower_window'] = 0
#holidays['upper_window'] = 0
holidays


# In[ ]:


m = Prophet(changepoint_prior_scale=2.5, holidays=holidays)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.fit(transactions)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[ ]:


# Calculate root mean squared error.
print('RMSE: %f' % np.sqrt(np.mean((forecast.loc[:1682, 'yhat']-transactions['y'])**2)) )
py.iplot([
    go.Scatter(x=transactions['ds'], y=transactions['y'], name='y'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),
    go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')
])


# We managed to predict the spikes for new-year period. The model couldn't catch the downward spike for Jan 4th 2016, so it could not predict Jan 1st 2017 successfully. That's very reasonable, as there is no holiday for Jan 4th of 2016. But the model predicts the sales on the 24th Decembers nicely. And also forecasted period after Aug 15th 2017 looks good.
# 
# And, as a non-ecuadorian person, I can say from the graph that "Dia de la Madre" is an important event, and it boosts sales previous day.
# 
# And, that's it. Prophet is a fairly easy to use library to forecast time-series data, which only uses previous data and holidays for that. There are more features and parameters like saturating forecasts, uncertanity intervals etc. which we didn't cover here. You can read more from their paper https://peerj.com/preprints/3190.pdf.
# 
# I think it's not a full-featured regression or forecasting tool, as we cannot use other data to use correlations or effects of another variable etc. But, It could be a replacement for time-series models like ARMA, ARIMA etc. Please comment below if you think you know more ways to improve this model. Thank you!
