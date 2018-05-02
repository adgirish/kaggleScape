
# coding: utf-8

# Welcome to the SQL Scavenger Hunt! Each day over the next five days we're going to be focusing on using different SQL commands to help us find the data we need to answer a specific question. I've put together this handbook to help you out by introducing you to SQL and BigQuery. In this handbook, I'll help you hit the ground running by covering the following topics: 
# 
# * [What are SQL & BigQuery and why you should learn them?](#What-are-SQL-&-BigQuery-and-why-you-should-learn-them?)
# * [How to use BigQuery datasets in kernels](#How-to-use-BigQuery-datasets-in-kernels)
# * [How to avoid common mistakes when querying big datasets](#How-to-avoid-common-mistakes-when-querying-big-datasets)
# 
# And I'll wrap up by collecting some helpful links, including the ones to the notebooks for the rest of the scavenger hunt. Let's get to it!

# # What are SQL & BigQuery and why you should learn them?
# ____
# 
# SQL (short for “Structured Query Language”, and said like either "see-quill" or "S-Q-L" ) is a programming language that allows you to interact with databases. For many databases out there, SQL is the *only* way to access the information in them and, as a result, it's an important skill for any data scientist or aspiring data scientist. (You don't need to take my word on this one: in our survey of data scientists we found that SQL was [the third most popular software tool for data science](https://www.kaggle.com/surveys/2017), right after Python and R.)
# 
# > **Why learn SQL?**: If you're currently looking for a data science job, being able to show that you're comfortable with SQL will open up more job opportunities for you. If you're currently *doing* data science, brushing up on your SQL skills will help you access more data sources and make it easier to get a subset of a database to work with locally. Plus, it's fun! :)
# 
# [BigQuery](https://cloud.google.com/bigquery/) is a Google Cloud product for storing and accessing very large databases very quickly. We've recently started making [some BigQuery datasets](https://www.kaggle.com/datasets?filetype=bigQuery) accessible via Kaggle. Since SQL is the easiest way to access these data in these datasets they make the perfect playground to help you get comfortable with this language.
# 
# >  Because the datasets on BigQuery can be very large, there are some restrictions on how much data you can access. The good news is that **each Kaggle user can scan 5TB every 30 days for free.** The bad news is that If you go over your quota you're going to have to wait for it to reset. 
# 
# Don't worry, though: in this handbook we'll teach you how to be careful when looking at BigQuery data to make sure you don't accidentally go over your quota.

# # How to use BigQuery datasets in kernels
# ____
# 
# In this section, we're going to go through how to get your first BigQuery notebook up and running and how to actually run a query. I'm going to cover:
# 
# * Getting your notebook set up
# * Checking the structure of the dataset (to help you when you want to write queries)
# * How to check the size of a query before you run it (to avoid accidentally asking for waaaay more data than you wanted)
# * How to run your first query! 
# * How to save the data from your query as a .csv to use later
# 
# I'm *not* going to cover all the intricacies of SQL. That's what the next five days of the Scavenger Hunt are for!
# 
# ## Getting set up
# ___
# The first step is to start a kernel using one of the BigQuery datasets as the data source. You can find these datasets by going to the [Datasets page](https://www.kaggle.com/datasets) and selecting "BigQuery" from the "File Types" drop down menu at the top of the list of datasets. (Alternatively, you can follow [this link](https://www.kaggle.com/datasets?filetype=bigQuery).) 
# 
# Once you've picked a BigQuery dataset, head to the dataset page for it and start a new kernel on it by hitting the "New Kernel" button, just as you would with any other dataset. Right now, you can only use BigQuery databases with Python kernels.
# 
# > **Can you work with BigQuery in R kernels?** Right now, you can only work with BigQuery in Python kernels, so make sure you don't accidentally start an R kernel instead. If you want to do your analysis in R (which is my personal preference) you can save the data you query as an output file and then add that to a new kernel. We'll go over how to do that later in this notebook. :)
# 
# In order to make your life easier, we've put together a Python package called `bq_helper` and pre-loaded it into kernels. It has some helper functions for getting data out of BigQuery that will simplify the process of getting data. If you're curious, you can check out [example kernels with more in-depth information at the end of this notebook](#Additional-information-and-resources).
# 
# You can use `bq_helper` in your kernel by importing it, like so:

# In[ ]:


# import our bq_helper package
import bq_helper 


# Now that we've added a BigQuery package to our kernel and imported the helper package, the next step is to create a BigQueryHelper object that points to a specific dataset. 
# 
# The first thing we need to do this is to know is what our specific dataset is called. You can find this by checking out the dataset listing for your dataset and then navigating to the "Data" tab. For example, [here's a link to the Data tab of the Hacker News dataset](https://www.kaggle.com/hacker-news/hacker-news/data), which is what we're going to be using in this kernel.
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

# ## Check out the structure of your dataset
# ____
# 
# The first thing you're going to want to do, like with any new dataset, is check out the way that data is structured. We're going to do that by looking at the schema.
# 
# > **Schema**: A description of how data is organized within a dataset.
# 
# Being able to access the information in the schema will be very helpful to us later on as we start to put together queries, so we're going to quickly go over how to do this and how to interpret the schemas once we get them. First, we can use the `BigQueryHelper.list_tables()` method to list all the files in the hacker_news dataset.

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

# ## Check the size of your query before you run it
# ____
# 
# BigQuery datasets are, true to their name, BIG. The [biggest dataset we've got on Kaggle so far](https://www.kaggle.com/github/github-repos) is 3 terabytes. Since the monthly quota for BigQuery queries is 5 terabytes, you can easily go past your 30-day quota by running just a couple of queries!
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


# Running this query will take around 150 MB. (The query size is returned in gigabytes.)
# 
# > **Important:** When you're writing your query, make sure that the name of the table (next to FROM) is in back ticks (\`), not single quotes ('). The reason for this is that the names of BigQuery tables contain periods in them, which in SQL are special characters. Putting the table name in back ticks protects the table name, so it's treated as a single string instead of being run as code.

# ## Actually run a query
# ___
# 
# Now that we know how to check the size of the query (and make sure we're not scanning several terabytes of data!) we're ready to actually run our first query. You have two methods available to help you do this:
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


# ## How to save the data from your query as a .csv
# ___
# 
# Now that we've got a dataframe, we might want to save it out as a .csv to use later or in a different kernel. To do this, we can use the following code, which will write our dataframe to a file called "job_post_scores.csv" in the output directory. 

# In[ ]:


# save our dataframe as a .csv 
job_post_scores.to_csv("job_post_scores.csv")


# In order to access this file later, you'll need to compile your notebook. You can do this using the "Publish" button (if you have the old editor) or the "New Snapshot" button (if you have the new editor). Both are in the upper right hand corner of the editor.
# 
# Once your notebook has compiled and your file is ready, it will show up under the "Output" tab of your notebook. (You should see an Output tab for this notebook with the file we just wrote to a .csv in it.) From there you can download it, or you can add the notebook with the output file as a data source to a new kernel and import the file to your notebook as you would any other .csv file. 
# 
# > *It can take a couple minutes for the Output tab to show up as your notebook compiles, especially if your file was very large. If you don't see it right away, give it a little bit and it should show up.*
# 
# And that's all you need to get started getting data from BigQuery datasets on Kaggle! During the scavenger hunt we'll learn more about how to use SQL to improve the power and flexibility of your queries. 

# # How to avoid common mistakes when querying big datasets
# ____
# 
# Big data is great! Until working at a bigger scale suddenly it makes your problems bigger too, like [this poor professor whose experiment racked up an unexpected $1000 bill](https://www.wired.com/2012/04/aws-bill-in-minutes/). Although Kaggle isn't charging for accessing BigQuery datasets, following these best practices can help you avoid trouble down the line. If you'd like to learn more, you can check out [all the BigQuery best practices here](https://cloud.google.com/bigquery/docs/best-practices).
# 
# * *Avoid using the asterisk *(**) in your queries.* As you might have seen before in regular expressions, the asterisk means “everything”. While this may be okay with smaller datasets, if you send a query to a 4 terabyte dataset and ask for “everything” you're going to scan waaaaay more than you bargained for (or that a kernel can handle).
# * *For initial exploration, look at just part of the table instead of the whole thing.* If you're just curious to see what data's in a table, preview it instead of scanning the whole table. We've included a method, `BigQueryHelper.head()`, in our helper package to help with this. Like `head()` in Pandas or R, it will just return the first few rows for you to look at.
# * *Double-check the size of complex queries.* If you're planning on running what might be a large query, either estimate the size first or run it using the `BigQueryHelper.query_to_pandas_safe()` method.
# * *Be cautious about joining tables.* In particular, avoid joining a table with itself (i.e. a self-join) and try to avoid joins that return a table that's larger than the ones you're joining together. (If you want to double-check yourself, you can try the join on just the heads of the tables involved.)
# * *Don't rely on LIMIT*: One of the things that can be confusing when working with BigQuery datasets is the difference between the data you *scan* and the data you actually *get back* especially since it's the first one that actually counts against your quota. When you do something like select a column with LIMIT = 10, you'll only get 10 results back... but you'll actually be scanning the whole column. It's not a big deal if your table has 1000 rows, but it's a much bigger deal if it has 10,000,000 rows!

# # Additional information and resources
# _____
# 
# ### Links to the workbook for each day
# 
# Now that you've read the handbook, you're ready to get started on the scavenger hunt! These notebooks were designed to be done at the pace of one per day, but feel free to work at your own pace: they'll be available indefinitely. 
# 
#  - [Day 1: SELECT FROM](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-1/)
#  - [Day 2: GROUP BY](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-2/) 
#  - [Day 3: ORDER BY &amp; Dates](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-3/)
#  - [Day 4: WITH &amp; AS](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-4/)
#  - [Day 5: JOIN](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-5/)
# 
# ### A video intro for working in Kernels
# 
# If you haven't worked in Kernels previously or just want to learn more about our new editor, I've recorded [a short intro video](https://www.youtube.com/watch?v=aMdk40hz30c) to help you get started. 
# 
# ### Develop a deeper understanding
# If you're interested in learning more about BigQuery specifically, Sohier has put together a number of useful kernels to help you learn about the ins and outs. A lot of the things discussed in these kernels have been integrated into our package, so these will let you see under the hood and gain a deeper understanding of what's going on.
# 
# * [A getting started guide that walks you through using the BigQuery package directly (this is a lot of what our package does for you)](https://www.kaggle.com/sohier/getting-started-with-big-query)
# * [A discussion of how to efficiently use resources in BigQuery](https://www.kaggle.com/sohier/efficient-resource-use-in-bigquery/)
# * [A more in-depth dicussion of the BigQuery API](https://www.kaggle.com/sohier/beyond-queries-exploring-the-bigquery-api)
# * [Converting BigQuery results to a Pandas dataframe](https://www.kaggle.com/sohier/how-to-integrate-bigquery-pandas)
# 
# ### Get inspired!
# 
# Here are a couple of interesting analysis to show you what sort of things are possible with BigQuery datasets.
# 
# * [What's the most common number of spaces to indent in Python code?](https://www.kaggle.com/hayatoy/most-common-indentation-space-count-in-python-code)
# * [In what contexts does Kaggle get mentioned in Hacker News?](https://www.kaggle.com/mrisdal/mentions-of-kaggle-on-hacker-news)
# 
# ### Learn more about BigQuery
# 
# The best place to learn more is in BigQuery documentation, which [you can find here](https://cloud.google.com/bigquery/docs/).
