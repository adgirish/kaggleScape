
# coding: utf-8

# ## Error/Failure Rate EDA and Feature Engineering Ideas
# 
# I wanted to know if certain lines or stations were correlated to higher error rates. While exploring, I found this code is also somewhat useful for feature engineering, such as taking the min/max values at each station.
# 
# ### Station 32
# 
# A total of 24,543 samples run through station 32, with a 4.7% error rate, compared the mean error rate 0.6%. It only has one feature, L3_S32_F3850, which has come up on other Kernels ranking feature importance.
# 
# Coincidentally (or maybe not), Station 31 is associated with the lowest failure rate at 0.27%

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set_color_codes("muted")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

numeric = "../input/train_numeric.csv"


# ## Sorting out Lines, Stations, and Features
# The function below creates a two dicts that isolate all the features that belong to specific Lines and Stations. This makes it easy to slice the data on a Line-by-Line or Station-by-Station basis.

# In[ ]:


features = pd.read_csv(numeric, nrows=1).drop(['Response', 'Id'], axis=1).columns.values

def orgainize(features):
    line_features = {}
    station_features = {}
    lines = set([f.split('_')[0] for f in features])
    stations = set([f.split('_')[1] for f in features])
    
    for l in lines:
        line_features[l] = [f for f in features if l+'_' in f]
        
    for s in stations:
        station_features[s] = [f for f in features if s+'_' in f]
        
            
    return line_features, station_features

line_features, station_features = orgainize(features)

print("Features in Station 32: {}".format( station_features['S32'] ))


# ## Exploring Stations
# 
# - Features - Total features in the Station
# - Samples - Total samples with measured values (non-NaN) >= 1 in the Station
# - Error rate - (Response==1) rate for samples in the Station.
# 
# Note* Samples run through multiple stations/lines, which was not taken into account, but may be important.
# 
# Note** A small percentage of samples have no data and were dropped.

# In[ ]:


station_error = []
for s in station_features:
    cols = ['Id', 'Response']
    cols.extend(station_features[s])
    df = pd.read_csv(numeric, usecols=cols).dropna(subset=station_features[s], how='all')
    error_rate = df[df.Response == 1].size / float(df[df.Response == 0].size)
    station_error.append([df.shape[1]-2, df.shape[0], error_rate]) 
    
station_data = pd.DataFrame(station_error, 
                         columns=['Features', 'Samples', 'Error_Rate'], 
                         index=station_features).sort_index()
station_data


# In[ ]:


plt.figure(figsize=(8, 20))
sns.barplot(x='Error_Rate', y=station_data.index.values, data=station_data, color="red")
plt.title('Error Rate between Production Stations')

plt.xlabel('Station Error Rate')
plt.show()


# ## Quick Feature Engineering Example
# Here's an example of how to use the station dict can be used to create new features or just reduce the size of the data.

# In[ ]:


data = pd.read_csv(numeric, nrows=100)

def make_features(df):
    new_features = pd.DataFrame({})
    for s in station_features.keys():
        station_data = df[station_features[s]]
        col = s+'_max'
        new_features[col] = station_data.max(axis=1).fillna(-1.)
        col = s+'_min'
        new_features[col] = station_data.min(axis=1).fillna(-1.)
    return new_features

data = make_features(data)
data.head()

