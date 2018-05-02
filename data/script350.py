
# coding: utf-8

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()


# In[ ]:



open_aq.head("global_air_quality")


# ## Which countries use a unit other than ppm to measure any type of pollution?

# Generating Query: 

# In[ ]:


query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """


# Casting to a DataFrame:

# In[ ]:


other_than_ppm = open_aq.query_to_pandas_safe(query1)


# Checking if everyting worked well 

# In[ ]:


other_than_ppm.head()


# It did indeed!
# 
# Alright! Let's have look at what we got.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (20, 6))
sns.countplot(other_than_ppm['country'])


# Let's get unique values.
# 
# **Final Solution:**

# In[ ]:


other_than_ppm['country'].unique()


# ## Which pollutants have a value of exactly 0?

# Same format again.
# 
# Generating Query:

# In[ ]:


query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """


# Casting to a DataFrame:

# In[ ]:


zero_value = open_aq.query_to_pandas_safe(query2)


# Checking if everything went well!

# In[ ]:


zero_value.head()


# Let's plot it on a count-plot!

# In[ ]:


plt.figure(figsize = (20, 6))
sns.countplot(zero_value['pollutant'])


# **Final Solution:**

# In[ ]:


zero_value['pollutant'].unique()


# **Final Output:**

# In[ ]:


other_than_ppm.to_csv("otp.csv")
zero_value.to_csv("zero.csv")


# **Bonus Analysis:** (Optional)

# In[ ]:


query3 = """SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality`
        """


# In[ ]:


df = open_aq.query_to_pandas_safe(query3)


# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr(), yticklabels=False)


# In[ ]:


sns.jointplot(df['longitude'], df['averaged_over_in_hours'], kind='reg')

