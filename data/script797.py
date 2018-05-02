
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
# * [**Day 2:** GROUP BY, HAVING & COUNT()](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-2/)
# 
# ____
# 

# # ORDER BY (and Dates!)
# 
# So far in our scavenger hunt, we've learned how to use the following clauses: 
#     
#     SELECT ... 
#     FROM ...
#     (WHERE) ...
#     GROUP BY ...
#     (HAVING) ...
# We also learned how to use the COUNT() aggregate function and, if you did the optional extra credit, possibly other aggregate functions as well. (If any of this is sounds unfamiliar to you, you can check out the earlier two days using the links above.)
# 
# Today we're going to learn how change the order that data is returned to us using the ORDER BY clause. We're also going to talk a little bit about how to work with dates in SQL, because they're sort of their own thing and can lead to headaches if you're unfamiliar with them.
# 
# 
# ### ORDER BY
# ___
# 
# First, let's learn how to use ORDER BY. ORDER BY is usually the last clause you'll put in your query, since you're going to want to use it to sort the results returned by the rest of your query.
# 
# We're going to be making queries against this version of the table we've been using an example over the past few days. 
# 
# > **Why would the order of a table change?** This can actually happen to active BigQuery datasets, since if your table is being added to regularly [it may be coalesced every so often and that will change the order of the data in your table](https://stackoverflow.com/questions/16854116/the-order-of-records-in-a-regularly-updated-bigquery-databaseg). 
# 
# You'll notice that, unlike in earlier days, our table is no longer sorted by the ID column. 
# 
# ![](https://i.imgur.com/QRgb4iL.png). 
# 
# ** Ordering by a numeric column**
# 
# When you ORDER BY a numeric column, by default the column will be sorted from the lowest to highest number. So this query will return the ID, Name and Animal columns, all sorted by the number in the ID column. The row with the lowest number in the ID column will be returned first.
# 
#     SELECT ID, Name, Animal
#     FROM `bigquery-public-data.pet_records.pets`
#     ORDER BY ID
# Visually, this looks something like this:
# 
# ![](https://i.imgur.com/zEXDTKS.png)
# 
#     
# ** Ordering by a text column**
# 
# You can also order by columns that have text in them. By default, the column you sort on will be sorted alphabetically from the beginning to the end of the alphabet.
# 
#     SELECT ID, Name, Animal
#     FROM `bigquery-public-data.pet_records.pets`
#     ORDER BY Animal
# ![](https://i.imgur.com/E7qjnf9.png)
# 
# ** Reversing the order**
# 
# You can reverse the sort order (reverse alphabetical order for text columns or high to low for numeric columns) using the DESC argument. 
# 
# > ** DESC** is short for "descending", or high-to-low.
# 
# So this query will sort the selected columns by the Animal column, but the values that are last in alphabetic order will be returned first.
# 
#     SELECT ID, Name, Animal
#     FROM `bigquery-public-data.pet_records.pets`
#     ORDER BY Animal DESC
# ![](https://i.imgur.com/DREYNFF.png)
#  
# ### Dates
# ____
# 
# Finally, let's talk about dates. I'm including these because they are something that I found particularly confusing when I first learned SQL, and I ended up having to use them all. the. time. 
# 
# There are two different ways that a date can be stored in BigQuery: as a DATE or as a DATETIME. Here's a quick summary:
# 
# **DATE format**
# 
# The DATE format has the year first, then the month, and then the day. It looks like this:
# 
#     YYYY-[M]M-[D]D
# * YYYY: Four-digit year
# * [M]M: One or two digit month
# * [D]D: One or two digit day
# 
# **DATETIME/TIMESTAMP format**
# 
# The DATETIME format is just like the date format... but with time added at the end. (The difference between DATETIME and TIMESTAMP is that the date and time information in a DATETIME is based on a specific timezone. On the other hand, a TIMESTAMP will be the same in all time zones, except for the time zone) . Both formats look like this:
# 
#     YYYY-[M]M-[D]D[( |T)[H]H:[M]M:[S]S[.DDDDDD]][time zone]
# * YYYY: Four-digit year
# * [M]M: One or two digit month
# * [D]D: One or two digit day
# * ( |T): A space or a T separator
# * [H]H: One or two digit hour (valid values from 00 to 23)
# * [M]M: One or two digit minutes (valid values from 00 to 59)
# * [S]S: One or two digit seconds (valid values from 00 to 59)
# * [.DDDDDD]: Up to six fractional digits (i.e. up to microsecond precision)
# * (TIMESTAMP only) [time zone]: String representing the time zone
# 
# ** Getting only part of a date **
# 
# Often, though, you'll only want to look at part of a date, like the year or the day. You can do this using the EXTRACT function and specifying what part of the date you'd like to extract. 
# 
# So this query will return one column with just the day of each date in the column_with_timestamp column: 
# 
#             SELECT EXTRACT(DAY FROM column_with_timestamp)
#             FROM `bigquery-public-data.imaginary_dataset.imaginary_table`
# One of the nice things about SQL is that it's very smart about dates and we can ask for information beyond just extracting part of the cell. For example, this query will return one column with just the week in the year (between 1 and 53) of each date in the column_with_timestamp column: 
# 
#             SELECT EXTRACT(WEEK FROM column_with_timestamp)
#             FROM `bigquery-public-data.imaginary_dataset.imaginary_table`
# SQL has a lot of power when it comes to dates, and that lets you ask very specific questions using this information. You can find all the functions you can use with dates in BigQuery [on this page](https://cloud.google.com/bigquery/docs/reference/legacy-sql), under "Date and time functions".  

# ## Example: Which day of the week do the most fatal motor accidents happen on?
# ___
# 
# Now we're ready to work through an example. Today, we're going to be using the US Traffic Fatality Records database, which contains information on traffic accidents in the US where at least one person died. (It's definitely a sad topic, but if we can understand this data and the trends in it we can use that information to help prevent additional accidents.)
# 
# First, just like yesterday, we need to get our environment set up. Since you already know how to look at schema information at this point, I'm going to let you do that on your own. 
# 
# > **Important note:** Make sure that you add the BigQuery dataset you're querying to your kernel. Otherwise you'll get 

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# We're going to look at which day of the week the most fatal traffic accidents happen on. I'm going to get the count of the unique id's (in this table they're called "consecutive_number") as well as the day of the week for each accident. Then I'm going sort my table so that the days with the most accidents are on returned first.

# In[ ]:


# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """


# Now that our query is ready, let's run it (safely!) and store the results in a dataframe: 

# In[ ]:


# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)


# And that gives us a dataframe! Let's quickly plot our data to make sure that it's actually been sorted:

# In[ ]:


# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")


# Yep, our query was, in fact, returned sorted! Now let's take a quick peek to figure out which days are the most dangerous:

# In[ ]:


print(accidents_by_day)


# To map from the numbers returned for the day of the week (the second column) to the actual day, I consulted [the BigQuery documentation on the DAYOFWEEK function](https://cloud.google.com/bigquery/docs/reference/legacy-sql#dayofweek), which says that it returns "an integer between 1 (Sunday) and 7 (Saturday), inclusively". So we can tell, based on our query, that in 2015 most fatal motor accidents occur on Sunday and Saturday, while the fewest happen on Tuesday.

# # Scavenger hunt
# ___
# 
# Now it's your turn! Here are the questions I would like you to get the data to answer:
# 
# * Which hours of the day do the most accidents occur during?
#     * Return a table that has information on how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each hour. Use either the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
#     * **Hint:** You will probably want to use the [EXTRACT() function](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#extract_1) for this.
# * Which state has the most hit and runs?
#     * Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up). "Forking" something is making a copy of it that you can edit on your own without changing the original.

# In[ ]:


# Your code goes here :)



# Please feel free to ask any questions you have in this notebook or in the [Q&A forums](https://www.kaggle.com/questions-and-answers)! 
# 
# Also, if you want to share or get comments on your kernel, remember you need to make it public first! You can change the visibility of your kernel under the "Settings" tab, on the right half of your screen.
