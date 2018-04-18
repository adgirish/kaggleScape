
# coding: utf-8

# # Introduction
# ____
# 
# SQL (pronounced as "see-quill" or "S-Q-L" ) is the primary programming language used to interact with databases. It is an important skill for any data scientist or aspiring data scientist.
# 
# [BigQuery](https://cloud.google.com/bigquery/) is a Google Cloud tool for working with **very** large databases. You can interact with data in BigQuery using SQL.
# 
# This series of tutorials and hands-on exercises will teach all the components you need to become effective with SQL and BigQuery.
# 
# ---
# 
# # Using BigQuery From Kaggle
# 
# This section describes basics for using BigQuery in Kaggle notebooks this includes:
# 
# * Getting your notebook set up
# * Checking the structure of the dataset (to help you when you want to write queries)
# * Checking the size of a query before you run it (to avoid accidentally asking for way more data than you wanted)
# * Running your first query
# 
# Intricacies of SQL will come in subsequent steps.
# 
# ---
# 
# ## Set-Up
# 
# The first step is to start a kernel using one of the BigQuery datasets as the data source. You can find these datasets by going to the [Datasets page](https://www.kaggle.com/datasets) and selecting "BigQuery" from the "File Types" drop down menu. (Or use [this link](https://www.kaggle.com/datasets?filetype=bigQuery).) 
# 
# Select a BigQuery dataset from that list, go to the dataset page for it and start a new kernel on it by hitting the "New Kernel" button. Right now, you can only use BigQuery databases with Python kernels.
# 
# In order to make your life easier, we'll use a Python package called `bq_helper`. It has helper functions for putting BigQuery results in Pandas DataFrames. 
# 
# You can use `bq_helper` in your kernel by importing it with the command

# In[ ]:


import bq_helper


# After adding a BigQuery package to our kernel and importing the helper package, create a BigQueryHelper object pointing to a specific dataset. 
# 
# Find what the dataset is called by checking out the dataset listing for your dataset and then navigating to the "Data" tab. For example, [here's a link to the Data tab of the Hacker News dataset](https://www.kaggle.com/hacker-news/hacker-news/data), which is what we'll use in this example.
# 
# If you go to the link I provided, you'll notice a blue rectangle with rounded edges near the top of the page that has the following text in it: "bigquery-public-data.hacker_news.comments". This tells you that you're looking of a summary of the "comments" table in the "hacker_news" dataset. The addresses of BigQuery datasets look like this:
# 
# ![](https://i.imgur.com/l11gdKx.png)
# 
# We will need to pass this information to BigQueryHelper in order to create our helper object. The active_project argument takes the BigQuery info, which is currently "bigquery-public-data" for all the BigQuery datasets on Kaggle. The dataset_name argument takes the name of the dataset we've added to our query. In this case it's "hacker_news". So we can create our BigQueryHelper object like so:

# In[ ]:


# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")


# Now that we've created our helper object, we can get started actually interacting with the dataset!

# ---
# ## Check Out The Structure of Your Dataset
# 
# We'll start by looking at the schema.
# 
# > **Schema**: A description of how data is organized within a dataset.
# 
# Viewing the schema will be very helpful later on as we start to put together queries. We can use the `BigQueryHelper.list_tables()` method to list all the files in the hacker_news dataset.

# BigQuery datasets can be very large, and there are some restrictions on how much data you can access. 
# 
# **Each Kaggle user can scan 5TB every 30 days for free.  If you go over your quota you're going to have to wait for it to reset.**
# 
# Don't worry, though: we'll teach you how to be careful when looking at BigQuery data to make sure you don't accidentally go over your quota.

# In[ ]:


# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()


# Now that we know what tables are in this dataset, we can get information on the columns in a specific table. In this example, we're looking at the information on the "full" table. 

# In[ ]:


# print information on all the columns in the "full" table
# in the hacker_news dataset
hacker_news.table_schema("full")


# Each SchemaField tells us about a specific column. In order, the information is:
# 
# * The name of the column
# * The datatype in the column
# * [The mode of the column](https://cloud.google.com/bigquery/docs/reference/rest/v2/tables#schema.fields.mode) (NULLABLE means that a column allows NULL values, and is the default)
# * A description of the data in that column
# 
# So, for the first column, we have the following schema field:
# 
# `SchemaField('by', 'string', 'NULLABLE', "The username of the item's author.",())`
# 
# This tells us that the column is called "by", that is has strings in it but that NULL values are allowed, and that it contains the username of the item's author.
# 
# We can use the `BigQueryHelper.head()` method to check just the first couple of lines of of the "full" table to make sure this is right. (Sometimes you'll run into databases out there where the schema isn't an accurate description of the data anymore, so it's good to check. This shouldn't be a problem with any of the BigQuery databases on Kaggle, though!)

# In[ ]:


# preview the first couple lines of the "full" table
hacker_news.head("full")


# The `BigQueryHelper.head()` method will also let us look at just the information in a specific column. If we want to see the first ten entries in the "by" column, for example, we can do that!

# In[ ]:


# preview the first ten entries in the by column of the full table
hacker_news.head("full", selected_columns="by", num_rows=10)


# Now that we know how to familiarize ourself with our datset, let's see how to check how big our queries are before we actually run them.
# 
# --- 
# 
# ## Check the size of your query before you run it
# 
# >  Because the datasets on BigQuery can be very large, there are some restrictions on how much data you can access.  Remember that you can scan 5TB every 30 days for free, and after that you'll need to wait until the end of that 30-day period.
# 
# The [biggest dataset currently on Kaggle](https://www.kaggle.com/github/github-repos) is 3 terabytes, so you can easily go past your 30-day quota by running just a couple of queries!
# 
# > **What's a query?** A query is small piece of SQL code that specifies what data would you like to scan from a databases, and how much of that data you would like returned. (Note that your quota is on data *scanned*, not the amount of data returned.)
# 
# One way to help avoid this is to estimate how big your query will be before you actually execute it. You can do this with the `BigQueryHelper.estimate_query_size()` method. For the rest of this notebook, I'll be using an example query that finding the scores for every Hacker News post of the type "job". Let's see how much data it will scan if we actually ran it.

# In[ ]:


# this query looks in the full table in the hacker_news
# dataset, then gets the score column from every row where 
# the type column has "job" in it.
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

# check how big this query will be
hacker_news.estimate_query_size(query)


# ---
# ## Run the Query
# 
# Now that we know how to check the size of the query (and make sure we're not scanning several terabytes of data!) we're ready to run our first query. You have two methods available to help you do this:
# 
# * *`BigQueryHelper.query_to_pandas(query)`*: This method takes a query and returns a Pandas dataframe.
# * *`BigQueryHelper.query_to_pandas_safe(query, max_gb_scanned=1)`*: This method takes a query and returns a Pandas dataframe only if the size of the query is less than the upperSizeLimit (1 gigabyte by default). 
# 
# Here's an example of a query that is larger than the specified upper limit.

# In[ ]:


# only run this query if it's less than 100 MB
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)


# And here's an example where the same query returns a dataframe. 

# In[ ]:


# check out the scores of job postings (if the 
# query is smaller than 1 gig)
job_post_scores = hacker_news.query_to_pandas_safe(query)


# Since this has returned a dataframe, we can work with it as we would any other dataframe. For example, we can get the mean of the column:

# In[ ]:


# average score for job posts
job_post_scores.score.mean()


# # Avoiding Common Mistakes when Querying Big Datasets
# ____
# 
# Big data is great! But working at a bigger scale makes your problems bigger too, like [this professor whose experiment racked up an unexpected $1000 bill](https://www.wired.com/2012/04/aws-bill-in-minutes/). Kaggle isn't charging for accessing BigQuery datasets, but these best practices can help you avoid trouble down the line.
# 
# * *Avoid using the asterisk *(**) in your queries.* The asterisk means “everything”. This may be okay with smaller datasets, but getting everything from a 4 terabyte dataset takes a long time and eats into your monthly usage limit.
# * *For initial exploration, look at just part of the table instead of the whole thing.* If you're just curious to see what data's in a table, preview it instead of scanning the whole table. The `BigQueryHelper.head()` method in our helper package does this. Like `head()` in Pandas or R, it returns just the first few rows for you to look at.
# * *Double-check the size of complex queries.* If you're planning on running what might be a large query, either estimate the size first or run it using the `BigQueryHelper.query_to_pandas_safe()` method.
# * *Be cautious about joining tables.* In particular, avoid joining a table with itself (i.e. a self-join) and try to avoid joins that return a table that's larger than the ones you're joining together. (You can double-check yourself by joining just the heads of the tables.)
# * *Don't rely on LIMIT*: One of the things that can be confusing when working with BigQuery datasets is the difference between the data you *scan* and the data you actually *get back* especially since it's the first one that actually counts against your quota. When you do something like select a column with LIMIT = 10, you'll only get 10 results back, but you'll scan the whole column (and that counts against your monthly usage limit).

# # Keep Going
# Get started with your first SQL commands **[here](https://www.kaggle.com/dansbecker/select-from-where).**
# 
# 
# ---
# 
# *This tutorial is part of the [SQL Series](https://www.kaggle.com/learn/sql) on Kaggle Learn.*
