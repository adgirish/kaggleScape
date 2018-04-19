
# coding: utf-8

# This dataset is great for learning BigQuery. It's so small that in practice you'll never be able to hit any query limits so you can run as many tests as you want.
# 
# I'll show you how to run a simple query against BigQuery and export the results to Pandas. We'll start with a helper library that lets us do this in one line and then move on to exploring how that function works.
# 
# If you're looking to get started quickly you can just use bq_helper to execute a SQL query. When you're ready to dig deeper the [full BigQuery SQL reference is here](https://cloud.google.com/bigquery/docs/reference/standard-sql/).

# In[5]:


import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper


# In[6]:


# sample query from:
# https://cloud.google.com/bigquery/public-data/openaq#which_10_locations_have_had_the_worst_air_quality_this_month_as_measured_by_high_pm10
QUERY = """
        SELECT location, city, country, value, timestamp
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE pollutant = "pm10" AND timestamp > "2017-04-01"
        ORDER BY value DESC
        LIMIT 1000
        """


# The quick way to execute this query is to use [the bq_helper library](https://github.com/SohierDane/BigQuery_Helper):

# In[7]:


bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(QUERY)
df.head(3)


# But what is bq_helper actually doing under the hood? Let's replicate the same process through the core BigQuery API to find out.

# In[8]:


client = bigquery.Client()
query_job = client.query(QUERY)
rows = list(query_job.result(timeout=30))
for row in rows[:3]:
    print(row)


# The outputs look reasonable, but what's this storage format?

# In[9]:


type(rows[0])


# Per the [BigQuery Python API documentation](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html?highlight=google%20cloud%20bigquery%20table%20row#google.cloud.bigquery.table.Row), it turns out that  we can access the labels and data separately. This will allow us to make a clean export to pandas.

# In[10]:


list(rows[0].keys())


# In[11]:


list(rows[0].values())


# In[12]:


df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))


# In[16]:


df.head(3)


# In[14]:


df.info()


# In[15]:


df['value'].plot()


# At the time of writing, the single most polluted site was a small town in rural Chile. I thought that had to be a problem with the data but it turns out [to be plausible](http://www.coha.org/the-battle-to-breathe-chiles-toxic-threat/) due to wildfires generating a lot of smoke and a severe drought causing smog to linger. The most polluted sites are an order of magnitude worse than the other cities, so I hope for their sake that it's a transient problem!
