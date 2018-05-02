
# coding: utf-8

# To get a feel for the data beyond [this analysis](https://www.kaggle.com/headsortails/be-my-guest-recruit-restaurant-eda), we'll plot some data for several random restaurants individually.

# In[ ]:


import pandas as pd
import numpy as np

AIR_RESERVE = 'air_reserve'
AIR_STORE_INFO = 'air_store_info'
AIR_VISIT_DATA = 'air_visit_data'
DATE_INFO = 'date_info'
HPG_RESERVE = 'hpg_reserve'
HPG_STORE_INFO = 'hpg_store_info'
STORE_ID_RELATION = 'store_id_relation'
SAMPLE_SUBMISSION = 'sample_submission'

data = {
    AIR_VISIT_DATA: pd.read_csv('../input/air_visit_data.csv'),
    AIR_STORE_INFO: pd.read_csv('../input/air_store_info.csv'),
    HPG_STORE_INFO: pd.read_csv('../input/hpg_store_info.csv'),
    AIR_RESERVE: pd.read_csv('../input/air_reserve.csv'),
    HPG_RESERVE: pd.read_csv('../input/hpg_reserve.csv'),
    STORE_ID_RELATION: pd.read_csv('../input/store_id_relation.csv'),
    SAMPLE_SUBMISSION: pd.read_csv('../input/sample_submission.csv'),
    DATE_INFO: pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}


# In[ ]:


def plot_number_of_visitors_to_one_resaturant(daat, air_visit_data, air_id):
  """Plots the number of visitors to this restaurant over time."""
  x = air_visit_data['visit_date']
  y = air_visit_data['visitors'] 

  traces = []
  traces.append(go.Scatter(x=x, y=y, mode='markers'))
  rolling_mean = y.rolling(window=10, min_periods=1, center=True).mean()
  traces.append(go.Scatter(x=x, y=rolling_mean, mode='lines'))

  layout = go.Layout(
    title='Visitors to resaurant ' + air_id,
    yaxis=dict(title='# Visitors')
  )
  iplot(go.Figure(data=traces, layout=layout))


# In[ ]:


def plot_visitors_on_days_of_week_for_one_restaurant(data, air_visit_data, air_id):
  air_visit_and_date_info = pd.merge(data[DATE_INFO], air_visit_data, how='inner', on='visit_date', copy=True)

  # Rainbow-ranked by median visitor over all restaurants.
  COLORS = {
    'Saturday': 'red',
    'Sunday': 'orange',
    'Friday': 'green',
    'Thursday': 'blue',
    'Wednesday': 'purple',
    'Tuesday': 'brown',
    'Monday': 'black',
  }

  traces = []
  for day_of_week in COLORS:
    # Plots the number of visitors to this restaurant over time.
    x = air_visit_and_date_info['visit_date']
    y = air_visit_and_date_info.loc[air_visit_and_date_info['day_of_week'] == day_of_week]['visitors']
    # Adds points to the plot.
    traces.append(go.Scatter(
      x=x,
      y=y,
      mode='markers',
      name=day_of_week,
      line=dict(color=COLORS[day_of_week]),
    ))
    # Adds a rolling mean line.
    traces.append(go.Scatter(
      x=x,
      y=y.rolling(7, min_periods=1, center=True).mean(),
      mode='lines',
      name=day_of_week,
      line=dict(color=COLORS[day_of_week]),
    ))

  layout = go.Layout(
    title='Visitors to resaurant ' + air_id + ' by day of week, with rolling averages',
    yaxis=dict(title='# Visitors')
  )
  iplot(go.Figure(data=traces, layout=layout))


# In[ ]:


from scipy import stats

def plot_reservations_vs_visitors_for_one_restaurant(data, air_visit_data, air_id):
  air_reserve = data[AIR_RESERVE][data[AIR_RESERVE]['air_store_id'] == air_id]
  # Some test restaurants have no AIR reservation data.
  if air_reserve.shape[0] == 0:
    print('There\'s no reservation data for ' + air_id + '.')
  else:
    df = air_reserve.copy()
    # Converts datetimes to days. Also converts to np.datetime64 because the air_visit_data's date type np.datetime64[ns].
    df['visit_date'] = df['visit_datetime'].map(lambda dt: np.datetime64(dt.date()))
    # Groups reservations by visit day, and sums the # seats reserved
    reserved_seats = df.groupby('visit_date')['reserve_visitors'].sum()
    reserved_seats = reserved_seats.reset_index()  # Before the index consisted of dates. reset_index makes the index positions, and makes the dates a column
    reserved_seats_and_visitors = pd.merge(reserved_seats, air_visit_data, how='inner', on='visit_date')

    # Plots the number of visitors to this restaurant over time.
    x = reserved_seats_and_visitors['reserve_visitors']
    y = reserved_seats_and_visitors['visitors'] 

    layout = go.Layout(
      title='Visitors vs seats reserved for resaurant ' + air_id,
      xaxis=dict(title='# Reserve Visitors'),
      yaxis=dict(title='# Visitors')
    )

    traces = []
    traces.append(go.Scatter(x=x, y=y, text=reserved_seats_and_visitors['visit_date'], mode='markers'))
    # TODO: Size the point based on the average size of the requested reservation.
    # Color the point based on how early the reservations for that day were ploced.

    # Overlays the linear trend line of reserved seats vs visitors.
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept
    traces.append(go.Scatter(x=x, y=line, mode='lines'))
    
    fig = go.Figure(data=traces, layout=layout)
    iplot(fig)  
    
  if hpg_id:
    # TODO: plot HPG reservation data.
    pass


# In[ ]:


def plot_median_visitors_per_day_of_week_on_holiday_vs_non_holiday(data, air_visit_data, air_id):
  air_visit_and_dates = pd.merge(data[DATE_INFO], air_visit_data, how='inner', on='visit_date')
  days_of_week = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
  air_visits_on_holidays = air_visit_and_dates[air_visit_and_dates['holiday_flg'] == 1]
  air_visits_on_non_holidays = air_visit_and_dates[air_visit_and_dates['holiday_flg'] == 0]
  holiday_data = {}
  non_holiday_data = {}
  for day in days_of_week:
    holiday_data[day] = air_visits_on_holidays[air_visits_on_holidays['day_of_week'] == day]['visitors'].median()
    non_holiday_data[day] = air_visits_on_non_holidays[air_visits_on_non_holidays['day_of_week'] == day]['visitors'].median()

  traces = []
  traces.append(go.Bar(
    x=days_of_week,
    y=[holiday_data[day] for day in days_of_week],
    name='On a holiday',
  ))
  traces.append(go.Bar(
    x=days_of_week,
    y=[non_holiday_data[day] for day in days_of_week],
    name='On a non-holiday',
  ))
  layout = go.Layout(
    title='Median visitors to ' + air_id + ' on holidays and non-holidays',
    yaxis=dict(title='Median # Visitors')
  )
  iplot(go.Figure(data=traces, layout=layout))


# In[ ]:


# Plots data for several random restaurants.
NUM_RESTAURANTS = 7

# We're using Plotly in offline mode
# because it appears plotly the credentials on our GCE VM.
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import random

init_notebook_mode(connected=True)

# Converts date strings to datetime objects.
data[AIR_VISIT_DATA]['visit_date'] = pd.to_datetime(data[AIR_VISIT_DATA]['visit_date'])
data[DATE_INFO]['visit_date'] = pd.to_datetime(data[DATE_INFO]['visit_date'])
data[AIR_RESERVE]['visit_datetime'] = pd.to_datetime(data[AIR_RESERVE]['visit_datetime'])
data[AIR_RESERVE]['reserve_datetime'] = pd.to_datetime(data[AIR_RESERVE]['reserve_datetime'])

for i in range(NUM_RESTAURANTS):
  random_index = random.randint(0, data[SAMPLE_SUBMISSION].shape[0])
  air_id = data[SAMPLE_SUBMISSION]['id'][random_index][:len('air_00a91d42b08b08d9')]
  
  # Plots the number of visitors to this restaurant over time.
  air_visit_data = data[AIR_VISIT_DATA][data[AIR_VISIT_DATA]['air_store_id'] == air_id]
  
  # The given air store may or may not be represented in the hgp store data.
  air_ids = data[STORE_ID_RELATION]['air_store_id']
  idx = air_ids[air_ids == air_id].index
  hpg_id = ''
  if len(idx) > 0:
    hpg_id = data[STORE_ID_RELATION]['hpg_store_id'][idx[0]]

  plot_number_of_visitors_to_one_resaturant(data, air_visit_data, air_id)
  plot_visitors_on_days_of_week_for_one_restaurant(data, air_visit_data, air_id)
  plot_reservations_vs_visitors_for_one_restaurant(data, air_visit_data, air_id)
  plot_median_visitors_per_day_of_week_on_holiday_vs_non_holiday(data, air_visit_data, air_id)

