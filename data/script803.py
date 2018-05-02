
# coding: utf-8

# <table>
#     <tr>
#         <td>
#         <center>
#         <font size="+1">If you haven't used BigQuery datasets on Kaggle previously, check out the <a href = "https://www.kaggle.com/rtatman/sql-scavenger-hunt-handbook/">Scavenger Hunt Handbook</a> kernel to get started.</font>
#         </center>
#         </td>
#     </tr>
# </table>

# # SELECT, FROM & WHERE
# 
# Today, we're going to learn how to use SELECT, FROM and WHERE to get data from a specific column based on the value of another column. For the purposes of this explanation, we'll be using this imaginary database, `pet_records` which has just one table in it, called `pets`, which looks like this:
# 
# ![](https://i.imgur.com/Ef4Puo3.png)
# 
# ### SELECT ... FROM
# ___
# 
# The most basic SQL query is to select a single column from a specific table. To do this, you need to tell SELECT which column to select and then specify what table that column is from using from. 
# 
# > **Do you need to capitalize SELECT and FROM?** No, SQL doesn't care about capitalization. However, it's customary to capitalize your SQL commands and it makes your queries a bit easier to read.
# 
# So, if we wanted to select the "Name" column from the pets table of the pet_records database (if that database were accessible as a BigQuery dataset on Kaggle , which it is not, because I made it up), we would do this:
# 
#     SELECT Name
#     FROM `bigquery-public-data.pet_records.pets`
# 
# Which would return the highlighted data from this figure.
# 
# ![](https://i.imgur.com/8FdVyFP.png)
# 
# ### WHERE ...
# ___
# 
# When you're working with BigQuery datasets, you're almost always going to want to return only certain rows, usually based on the value of a different column. You can do this using the WHERE clause, which will only return the rows where the WHERE clause evaluates to true.
# 
# Let's look at an example:
# 
#     SELECT Name
#     FROM `bigquery-public-data.pet_records.pets`
#     WHERE Animal = "Cat"
# 
# This query will only return the entries from the "Name" column that are in rows where the "Animal" column has the text "Cat" in it. Those are the cells highlighted in blue in this figure:
# 
# ![](https://i.imgur.com/Va52Qdl.png)
# 

# ## Example: What are all the U.S. cities in the OpenAQ dataset?
# ___
# 
# Now that you've got the basics down, let's work through an example with a real dataset. Today we're going to be working with the OpenAQ dataset, which has information on air quality around the world. (The data in it should be current: it's updated weekly.)
# 
# To help get you situated, I'm going to run through a complete query first. Then it will be your turn to get started running your queries!
# 
# First, I'm going to set up everything we need to run queries and take a quick peek at what tables are in our database.

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()


# I'm going to take a peek at the first couple of rows to help me see what sort of data is in this dataset.

# In[ ]:


# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")


# Great, everything looks good! Now that I'm set up, I'm going to put together a query. I want to select all the values from the "city" column for the rows there the "country" column is "us" (for "United States"). 
# 
# > **What's up with the triple quotation marks (""")?** These tell Python that everything inside them is a single string, even though we have line breaks in it. The line breaks aren't necessary, but they do make it much easier to read your query.

# In[ ]:


# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """


# > **Important:**  Note that the argument we pass to FROM is *not* in single or double quotation marks (' or "). It is in backticks (\`). If you use quotation marks instead of backticks, you'll get this error when you try to run the query: `Syntax error: Unexpected string literal` 
# 
# Now I can use this query to get information from our open_aq dataset. I'm using the `BigQueryHelper.query_to_pandas_safe()` method here because it won't run a query if it's larger than 1 gigabyte, which helps me avoid accidentally running a very large query. See the [Scavenger Hunt Handbook ](https://www.kaggle.com/rtatman/sql-scavenger-hunt-handbook/)for more details. 

# In[ ]:


# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)


# Now I've got a dataframe called us_cities, which I can use like I would any other dataframe:

# In[ ]:


# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()


# # Scavenger hunt
# ___
# 
# Now it's your turn! Here's the questions I would like you to get the data to answer:
# 
# * Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value *isn't* something, use "!=")
# * Which pollutants have a value of exactly 0?
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up).  "Forking" something is making a copy of it that you can edit on your own without changing the original.

# In[ ]:


# Your code goes here :)



# Please feel free to ask any questions you have in this notebook or in the [Q&A forums](https://www.kaggle.com/questions-and-answers)! 
# 
# Also, if you want to share or get comments on your kernel, remember you need to make it public first! You can change the visibility of your kernel under the "Settings" tab, on the right half of your screen.
