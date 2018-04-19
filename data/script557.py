
# coding: utf-8

# In[ ]:


import pandas as pd
from IPython.display import display
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# ## Loading all input files

# In[ ]:


air_reserve = pd.read_csv('../input/air_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])

hpg_reserve = pd.read_csv('../input/hpg_reserve.csv',parse_dates=['visit_datetime','reserve_datetime'])

air_store_info = pd.read_csv('../input/air_store_info.csv')

hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')

store_relation = pd.read_csv('../input/store_id_relation.csv')

date_info = pd.read_csv('../input/date_info.csv',parse_dates=['calendar_date'])

air_visit = pd.read_csv('../input/air_visit_data.csv',parse_dates=['visit_date'])

sample_submission = pd.read_csv('../input/sample_submission.csv')


# ## Snapshot

# In[ ]:


air_reserve.head(2) #all the air stores booking data


# In[ ]:


hpg_reserve.head(2) #all the hpg stores booking data


# In[ ]:


air_store_info.head(2) #description of air stores


# In[ ]:


hpg_store_info.head(2) #description of hpg stores


# In[ ]:


air_visit.head(2) #historical visits for air stores


# In[ ]:


date_info.head(2) 


# In[ ]:


store_relation.head(2) #hpg store to air store mapping


# ---

# ## Submission file treatment
# We are only predicting on air stores which are in this file

# In[ ]:


sample_submission.head(2) #air id and date is merged together


# **air id and date is merged together so we'll extract id and calendar date**

# In[ ]:


#https://www.kaggle.com/zeemeen/weighted-mean-running-10-sec-lb-0-509
sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])


# In[ ]:


sample_submission.head(2) #air id and date is merged together


# In[ ]:


sample_submission.apply(lambda c: c.nunique()) 


# **Unique number of stores to predict: 821**

# ---

# ## Data pre-processing

# In[ ]:


air_store_info.apply(lambda x: x.nunique())


# In[ ]:


air_reserve.apply(lambda x: x.nunique())


# In[ ]:


set(air_reserve.air_store_id) < set(air_store_info.air_store_id) 


# ***All the store ids in air reserve data is a subset of ids in air store info data***

# ### Combining different data sources

# In[ ]:


air_combine = pd.merge(air_reserve, air_store_info, on='air_store_id', how='outer') #joining reservation and store info data for air stores


# In[ ]:


air_combine.head(2)


# In[ ]:


hpg_combine = pd.merge(hpg_reserve, hpg_store_info, on='hpg_store_id', how='left') #joining reservation and store info data for hpg stores


# In[ ]:


hpg_combine.tail(2)


# **I'm assuming that air stores which are also listed in hpg have mutually exclusive bookings through air and hpg portal. As we are forecasting on air stores only, we'll pull information for those specific stores from the hpg dataset. For that we'll first jpin hpg dataset with store relation dataset to pull corresponding air store ids**

# In[ ]:


hpg_combine = pd.merge(hpg_combine, store_relation, on='hpg_store_id', how='right') #right join as we want data for only air stores


# In[ ]:


hpg_combine.head(2)


# In[ ]:


hpg_combine = hpg_combine.drop(['hpg_store_id'],axis =1) #don't require hpg_id now


# In[ ]:


hpg_combine.rename(columns={'hpg_genre_name': 'air_genre_name', 'hpg_area_name': 'air_area_name'}, inplace=True)#renaming column names to match up with the air_combine dataset


# In[ ]:


hpg_combine.tail(2)


# In[ ]:


air_combine.head(2)


# In[ ]:


air_combine.shape,hpg_combine.shape


# In[ ]:


air_combine = pd.concat([air_combine,hpg_combine],axis = 0) #combining data for air stores from both datasets


# In[ ]:


air_combine.tail(2)


# In[ ]:


air_combine.shape


# Now we need to extract the date from the datetime column to join it with date_info table

# In[ ]:


air_combine['visit_date'] = pd.to_datetime(air_combine['visit_datetime'].dt.date)


# In[ ]:


air_combine['reserve_date'] = pd.to_datetime(air_combine['reserve_datetime'].dt.date)


# In[ ]:


air_combine.head(2)


# In[ ]:


air_combine = pd.merge(air_combine, date_info, left_on='visit_date',right_on='calendar_date', how='left') #joining on visit_date column


# In[ ]:


air_combine.head(1)


# In[ ]:


air_combine = air_combine.drop(['visit_datetime','reserve_datetime'],axis = 1) #dropping unnecessary columns


# In[ ]:


air_combine.head(2)


# ### Missing values treatment

# In[ ]:


air_combine.isnull().sum() #null values


# Missing value:
# * 'reserve_visitors':0
# * 'visit_date':pd.to_datetime('01/01/2099')
# * 'reserve_date':pd.to_datetime('01/01/2099')
# * 'calendar_date':pd.to_datetime('01/01/2099')
# * 'day_of_week':'unknown'
# * 'holiday_flg':-99
# * 'latitude':-99
# * 'longitude':-99
# * 'air_genre_name':'unknown'
# * 'air_area_name':'unknown'

# In[ ]:


air_combine = air_combine.fillna({'reserve_visitors':0,'visit_date':pd.to_datetime('01/01/2099'),
                                  'reserve_date':pd.to_datetime('01/01/2099'),'calendar_date':pd.to_datetime('01/01/2099'),
                                  'day_of_week':'unknown','holiday_flg':-99,'latitude':-99,'longitude':-99,'air_genre_name':'unknown',
                                 'air_area_name':'unknown'})


# In[ ]:


air_combine.isnull().sum() #no null values


# ### Data type conversion

# In[ ]:


air_combine.dtypes


# In[ ]:


air_combine['holiday_flg'] = air_combine['holiday_flg'].astype('int8')

air_combine['day_of_week'] = air_combine['day_of_week'].astype('category')

air_combine['air_genre_name'] = air_combine['air_genre_name'].astype('category')

air_combine['air_area_name'] = air_combine['air_area_name'].astype('category')

air_combine['air_store_id'] = air_combine['air_store_id'].astype('category')

air_combine['reserve_visitors'] = air_combine['reserve_visitors'].astype('int8')


# In[ ]:


air_combine.dtypes


# ---

# ***So for now the combined data look like this. I haven't joined the air visit historical data to this combined dataset for now***

# In[ ]:


air_combine.head()


# In[ ]:


air_visit.head()


# In[ ]:


air_combine.to_csv('air_combine.csv',index = False)


# **Whenever you do some data processing on pandas especially when you're dealing with big datasets, you can store the intermediate pandas dataframe as feather format like this. It saves a lot of time while reading it back again. Much faster than pickle**

# In[ ]:


# air_combine.to_feather('air_combine_raw') 


# In[ ]:


# df_combine = pd.read_feather('air_combine_raw') #to read from feather format


# In[ ]:


# df_combine.head()


# ##### Let me know if you have any questions or suggestions
