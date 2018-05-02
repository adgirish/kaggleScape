
# coding: utf-8

# This kernel shows how easy it can be to run a SQL query against a BigQuery table and get a pandas dataframe as the result. If you're interested in digging deeper, check out these references:
# - [A walkthrough of `bq_helper`](https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package)
# - [The BigQuery SQL documentation](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types)
# - [The BigQuery client API](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html)

# In[ ]:


import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper


# In[ ]:


QUERY = """
    SELECT
        extract(DAYOFYEAR from date_local) as day_of_year,
        aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
      city_name = "Los Angeles"
      AND state_name = "California"
      AND sample_duration = "24 HOUR"
      AND poc = 1
      AND EXTRACT(YEAR FROM date_local) = 2015
    ORDER BY day_of_year
        """


# In[ ]:


bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")


# In[ ]:


df = bq_assistant.query_to_pandas(QUERY)


# In[ ]:


df.plot(x='day_of_year', y='aqi', style='.');


# Based on 2015 it looks like the worst air quality days in Los Angeles take place in winter. However, there's a lot of day to day variation at all times of the year.
