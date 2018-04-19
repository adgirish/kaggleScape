
# coding: utf-8

# Simple Analysis using pandas 
# 
# **Quality Parameter Encode
#  - Arsenic - 0
#  - Fluoride - 1
#  - Iron - 2
#  - Nitrate - 3
#  - Salinity - 4**
# 
# 
#  **Working on It .... more to come**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


meta_data = pd.read_csv("../input/IndiaAffectedWaterQualityAreas.csv",encoding='latin1')


# **Total Columns**

# In[ ]:


meta_data.columns


# **States**

# In[ ]:


meta_data['State Name'].unique()


# State wise Counts*

# In[ ]:


#meta_data['State Name'].value_counts()


# **Water Quality Parameters**

# In[ ]:


meta_data['Quality Parameter'].value_counts()


# ****State Wise Description With Count and High Frequency Count Quality Parameters ****

# In[ ]:


meta_data['Quality Parameter'].groupby(meta_data['State Name']).describe()


# State Wise Entry Count

# In[ ]:


#meta_data.groupby("State Name").size()


# **Data Month and Year Extraction**

# In[ ]:


import dateutil
meta_data['date'] = meta_data['Year'].apply(dateutil.parser.parse, dayfirst=True)
import datetime as dt
meta_data['date'] = pd.to_datetime(meta_data['date'])
meta_data['year'] = meta_data['date'].dt.year
meta_data['month'] = meta_data['date'].dt.month




# In[ ]:


State_Data = meta_data[['State Name', 'Quality Parameter', 'year','month']]
del State_Data ['year']
del State_Data ['month']


# **Used Label Encoder to Categorical Into Numerical**

# In[ ]:


import sklearn
from sklearn.preprocessing import LabelEncoder
numbers = LabelEncoder()


# **** Arsenic - 0
#  - Fluoride - 1
#  - Iron - 2
#  - Nitrate - 3
#  - Salinity - 4**

# In[ ]:


State_Data['Quality'] = numbers.fit_transform(State_Data['Quality Parameter'].astype('str'))


# In[ ]:


State_Data.head(5)


# In[ ]:


Group1 = State_Data.groupby(['State Name','Quality Parameter','Quality']).count()
Group1


# In[ ]:


#pd.DataFrame({'count' : State_Data.groupby( [ "State Name", "Quality"] ).size()}).reset_index()


# In[ ]:


State_Quality_Count = pd.DataFrame({'count' : State_Data.groupby( [ "State Name", "Quality","Quality Parameter"] ).size()}).reset_index()


# **Took Six Random States for Analysis**

# In[ ]:


TAMIL_NADU   =  State_Quality_Count[State_Quality_Count["State Name"] == "TAMIL NADU"]    
ANDHRA_PRADESH = State_Quality_Count[State_Quality_Count["State Name"] == "ANDHRA PRADESH"]
KERALA = State_Quality_Count[State_Quality_Count["State Name"] == "KERALA"]
KARNATAKA = State_Quality_Count[State_Quality_Count["State Name"] == "KARNATAKA"]
GUJARAT = State_Quality_Count[State_Quality_Count["State Name"] == "GUJARAT"]
MAHARASHTRA = State_Quality_Count[State_Quality_Count["State Name"] == "MAHARASHTRA"]




# In[ ]:


TAMIL_NADU


# In[ ]:


plt.figure(figsize=(6,4))
ax = sns.barplot(x="count", y ="Quality Parameter", data = TAMIL_NADU)
ax.set(xlabel='Count')
sns.despine(left=True, bottom=True)
plt.title("Water Quality Parameter In Tamil Nadu")


# In[ ]:


KARNATAKA


# In[ ]:


plt.figure(figsize=(6,4))
ax = sns.barplot(x="count", y ="Quality Parameter", data = KARNATAKA)
ax.set(xlabel='Count')
sns.despine(left=True, bottom=True)
plt.title("Water Quality Parameter In Karnataka")


# In[ ]:


MAHARASHTRA


# In[ ]:


plt.figure(figsize=(6,4))
ax = sns.barplot(x="count", y ="Quality Parameter", data = MAHARASHTRA)
ax.set(xlabel='Count')
sns.despine(left=True, bottom=True)
plt.title("Water Quality Parameter In Mahrashtra")


# In[ ]:


GUJARAT


# In[ ]:


plt.figure(figsize=(6,4))
ax = sns.barplot(x="count", y ="Quality Parameter", data = GUJARAT)
ax.set(xlabel='Count')
sns.despine(left=True, bottom=True)
plt.title("Water Quality Parameter In Gujarat")


# In[ ]:


ANDHRA_PRADESH


# In[ ]:


plt.figure(figsize=(6,4))
ax = sns.barplot(x="count", y ="Quality Parameter", data = ANDHRA_PRADESH)
ax.set(xlabel='Count')
sns.despine(left=True, bottom=True)
plt.title("Water Quality Parameter In Andhra Pradesh")


# In[ ]:


plt.figure(figsize=(6,4))
ax = sns.barplot(x="count", y ="Quality Parameter", data = KERALA)
ax.set(xlabel='Count')
sns.despine(left=True, bottom=True)
plt.title("Water Quality Parameter In Kerala")


# **Total Water Quality Parameter in INDIA**

# In[ ]:


x = State_Quality_Count.groupby('State Name')
plt.rcParams['figure.figsize'] = (9.5, 6.0)
genre_count = sns.barplot(y='Quality Parameter', x='count', data=State_Quality_Count, palette="Blues", ci=None)
plt.show()

