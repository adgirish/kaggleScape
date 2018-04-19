
# coding: utf-8

# ##High level insight on Wikipedia web traffic.
# 
# Note: If this helped you, some upvotes would be very much appreciated.
# 
# It is foolish to fear what we have yet to see and know ;) 

# ### Library and Settings
# Import required library and define constants

# In[ ]:


import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import calendar

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from bokeh.charts import TimeSeries, show


# ### File Size

# In[ ]:


for f in os.listdir('../input'):
    size_bytes = round(os.path.getsize('../input/' + f)/ 1000, 2)
    size_name = ["KB", "MB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    print(f.ljust(25) + str(s).ljust(7) + size_name[i])


# In[ ]:


train_df = pd.read_csv("../input/train_1.csv")
key_df = pd.read_csv("../input/key_1.csv")


# In[ ]:


print("Train".ljust(15), train_df.shape)
print("Key".ljust(15), key_df.shape)


# As we can see Train data frame is having less number of page details than number of mappings between keys and page. First and last rows of train and key data frame.

# In[ ]:


print(train_df[:4].append(train_df[-4:], ignore_index=True))


# In[ ]:


print(key_df[:4].append(key_df[-4:], ignore_index=True))


# Each article name has the following format: 'name_project_access_agent' . That would be a good idea to separate out all these 4 features to get better understanding of data

# In[ ]:


page_details = pd.DataFrame([i.split("_")[-3:] for i in train_df["Page"]])
page_details.columns = ["project", "access", "agent"]
page_details.describe()


# As we can see there are 9, 3 and 2 unique values for project, access and agent respectively. Why not have quick look on these values

# In[ ]:


project_columns = page_details['project'].unique()
access_columns = page_details['access'].unique()
agents_columns = page_details['agent'].unique()
print(list(page_details['project'].unique()))
print(list(page_details['access'].unique()))
print(list(page_details['agent'].unique()))


# So no NA values in here. Perfect. Lets merge

# In[ ]:


train_df = train_df.merge(page_details, how="inner", left_index=True, right_index=True)


# Boom. Let's plot the project wise monthly mean hits.

# In[ ]:


def graph_by(plot_hue, graph_columns):
    train_project_df = train_df.groupby(plot_hue).sum().T
    train_project_df.index = pd.to_datetime(train_project_df.index)
    train_project_df = train_project_df.groupby(pd.TimeGrouper('M')).mean().dropna()
    train_project_df['month'] = 100*train_project_df.index.year + train_project_df.index.month
    train_project_df = train_project_df.reset_index(drop=True)
    train_project_df = pd.melt(train_project_df, id_vars=['month'], value_vars=graph_columns)
    fig = plt.figure(1,figsize=[12,10])
    ax = sns.pointplot(x="month", y="value", hue=plot_hue, data=train_project_df)
    ax.set(xlabel='Year-Month', ylabel='Mean Hits')


# In[ ]:


graph_by("project", project_columns)


# Look, everyone is hitting English Wikipedia project more than any other project. Also, Russian Wikipedia is having same hike near to august 2016 as English Wikipedia. 
# 
# Now with English project in graph, it is hard to visualise other projects. Why not separate out English project and find some patterns if possible.

# In[ ]:


graph_by("project", [x for i,x in enumerate(project_columns) if i!=2])


# People rarely use mediawiki, commons or  zh. 
# 
# Quick check for access and agents as well

# In[ ]:


graph_by("access", access_columns)


# In[ ]:


graph_by("agent", agents_columns)


# We could not get data pattern from above graph. With just two values for agent why not get them plotted in two separate graphs and see how they behave.

# In[ ]:


graph_by("agent", agents_columns[0])


# In[ ]:


graph_by("agent", agents_columns[1])


# Lately observed, all-access and all-agents value for access and agents are summation of values for respective attributes. So each value other than all-access contribute in trend for all-access and all values other than all-agents contribute in trend of all-agents.

# To be continued...
