
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
# 
# ___ 
# 
# ## Previous days:
# 
# * [**Day 1:** SELECT, FROM & WHERE](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-1/)
# 
# ____
# 

# # GROUP BY... HAVING and COUNT
# 
# Now that we know how to select the content of a column, we're ready to learn how to group your data and count things within those groups. This can help you answer questions like: 
# 
# * How many of each kind of fruit has our store sold?
# * How many species of animal has the vet office treated?
# 
# To do this, we're going to learn about three new techniques: GROUP BY, HAVING and COUNT. Once again, we're going to use this 100% made up table of information on various pets, which has three columns: one with the unique ID number for each pet, one with the name of the pet and one with the species of the animal (rabbit, cat or dog). 
# 
# ![](https://i.imgur.com/Ef4Puo3.png)
# 
# ### COUNT
# ___
# 
# COUNT(), as you may have guessed from the name, returns a count of things. If you pass it the name of a column, it will return the number of entries in that column. So if we SELECT the COUNT() of the ID column, it will return the number of ID's in that column.
# 
#     SELECT COUNT(ID)
#     FROM `bigquery-public-data.pet_records.pets`
#     
# This query, based on the table above, will return 4 because there are 4 ID's in this table.
#  
# ### GROUP BY
# ___
# 
# GROUP BY takes the name of one or more column and tells SQL that we want to treat rows that have the same value in that column as a single group when we apply aggregate functions like COUNT().
# 
# > An **aggregate function** takes in many values and returns one. Here, we're learning about COUNT() but there are other aggregate functions like SUM() and AVERAGE().
# 
# Note that because it tells SQL how to apply aggregate functions, it doesn't make sense to use GROUP BY without something like COUNT(). 
# 
# Let's look at an example. We want to know how many of each type of animal we have in our table. We can get this information by using GROUP BY to group together rows that have the same value in the “Animal” column, while using COUNT() to find out how many ID's we have in each group. You can see the general idea in this image:
# 
# ![](https://i.imgur.com/MFRhycu.png)
# 
# The query that will get us this information looks like this:
# 
#     SELECT Animal, COUNT(ID)
#     FROM `bigquery-public-data.pet_records.pets`
#     GROUP BY Animal
# 
# This query will return a table with two columns (Animal & COUNT(ID)) three rows (one for each distinct Animal). 
# 
# One thing to note is that if you SELECT a column that you don't pass to 1) GROUP BY or 2) use as input to an aggregate function, you'll get an error. So this query won't work, because the Name column isn't either passed to either an aggregate function or a GROUP BY clause:
# 
#     # NOT A VALID QUERY! "Name" isn't passed to GROUP BY
#     # or an aggregate function
#     SELECT Name, Animal, COUNT(ID)
#     FROM `bigquery-public-data.pet_records.pets`
#     GROUP BY Animal
#     
# If make this error, you'll get the error message `SELECT list expression references column (column's name) which is neither grouped nor aggregated at`.
# 
# ### GROUP BY ... HAVING
# ___
# 
# Another option you have when using GROUP BY is to specify that you want to ignore groups that don't meet certain criteria. So this query, for example, will only include groups that have more than one ID in them:
# 
#     SELECT Animal, COUNT(ID)
#     FROM `bigquery-public-data.pet_records.pets`
#     GROUP BY Animal
#     HAVING COUNT(ID) > 1
# 
# The only group that this query will return information on is the one in the cells highlighted in blue in this figure:
# 
# ![](https://i.imgur.com/8xutHzn.png)
# 
# As a result, this query will return a table with only one row, since this there only one group remaining. It will have two columns: one for "Animal", which will have "Cat" in it, and one for COUNT(ID), which will have 2 in it. 

# ## Example: Which Hacker News comments generated the most discussion?
# ___
# 
# Now we're ready to work through an example on a real dataset. Today, we're going to be using the Hacker News dataset, which contains information on stories & comments from the Hacker News social networking site. I want to know which comments on the site generated the most replies.
# 
# First, just like yesterday, we need to get our environment set up. I already know that I want the "comments" table, so I'm going to look at the first couple of rows of that to get started.

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")


# By looking at the documentation, I learned that the "parent" column has information on the comment that each comment was a reply to and the "id" column has the unique id used to identify each comment. So I can group by the "parent" column and count the "id" column in order to figure out the number of comments that were made as responses to a specific comment. 
# 
# Because I'm more interested in popular comments than unpopular comments, I'm also only going to return the groups that have more than ten id's in them. In other words, I'm only going to look at comments that had more than ten comment replies to them.

# In[ ]:


# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """


# Now that our query is ready, let's run it (safely!) and store the results in a dataframe: 

# In[ ]:


# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)


# And, just like yesterday, we have a dataframe we can treat like any other data frame:

# In[ ]:


popular_stories.head()


# Looks good! From here I could do whatever further analysis or visualization I'd like. 
# 
# > **Why is the column with the COUNT() data called f0_**? It's called this because COUNT() is the first (and in our case, only) aggregate function we used in this query. If we'd used a second one, it would be called "f1\_", the third would be called "f2\_", and so on. We'll learn how to name the output of aggregate functions later this week.
# 
# And that should be all you need to get started writing your own kernels with GROUP BY... WHERE and COUNT!

# # Scavenger hunt
# ___
# 
# Now it's your turn! Here's the questions I would like you to get the data to answer:
# 
# * How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
# * How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
# * **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up). "Forking" something is making a copy of it that you can edit on your own without changing the original.

# In[ ]:


# Your code goes here :)



# Please feel free to ask any questions you have in this notebook or in the [Q&A forums](https://www.kaggle.com/questions-and-answers)! 
# 
# Also, if you want to share or get comments on your kernel, remember you need to make it public first! You can change the visibility of your kernel under the "Settings" tab, on the right half of your screen.
