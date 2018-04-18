
# coding: utf-8

# ## Which countries use a unit other than ppm to measure any type of pollution?
# ___

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")


# In[ ]:


# query to select all cities where
# unit is not equal to PPM
query = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit) != 'ppm'
            ORDER BY country
        """


# In[ ]:


# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
non_ppm_countries = open_aq.query_to_pandas_safe(query)

# display query results
non_ppm_countries


# # Which pollutants have a value of exactly 0?
# ___
# 
# This question is somewhat unclear...is it asking for all pollutants that have ever had a pollutant value of exactly zero (even if that pollutant has had other readings that were non-zero) or for pollutants whose total values are exactly zero across all readings? I'll try both below to see what happens.
# 
# The first query shows a list of all pollutants that have ever had a reading of zero across the entire reading history database:

# In[ ]:


# query to select all pollutants that have had a reading where pollutant value
# where pollutant value was equal to zero
query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant
        """

zero_pol_readings = open_aq.query_to_pandas_safe(query)

# display query results
zero_pol_readings


# The query below fails to run due to scanning more than 1 GB of data, unless the query is filtered using a WHERE clause (such as on the country). The query runs for a country value of 'LT' for example, but fails as too large for a country value of 'US'.

# In[ ]:


# query to select pollutants (if any) where total value across
# all readings is equal to zero
query = """SELECT pollutant, SUM(value) as tot_value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'LT'
            GROUP BY pollutant
            HAVING tot_value = 0
        """
zero_pol_readings2 = open_aq.query_to_pandas_safe(query)

# display query results
zero_pol_readings2

