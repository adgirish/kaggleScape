
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
# * [**Day 3:** ORDER BY & Dates](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-3/)
# * [**Day 4:** WITH & AS](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-4/)
# 
# ____
# 

# ## JOIN
# ___
# 
# Whew, we've come a long way from Day 1! By now, you have the tools to get many different configurations of information from a single table. But what if your database has more than one table and you want to look at information from multiple tables?
# 
# That's where JOIN comes in! Today, we're going to learn about how to use JOIN to combine data from two tables. This will let us answer more types of questions. It's also one of the more complex parts of SQL. Don't worry, though, we're going to go through some examples together. 
# 
# ### JOIN
# ___
# 
# Let's keep working with our imaginary Pets dataset, but this time let's add a second table. The first table, "Pets", has three columns, with information on the ID number of each pet, the pet's name and the type of animal it is. The new table, "Owners", has three columns, with the ID number of each owner, the name of the owner and the ID number of their pet. 
# 
# ![](https://i.imgur.com/W4gYoNg.png)
# 
# Each row in each table is associated with a single pet and we refer to the same pets in both tables. We can tell this because there are two columns (ID in the "Pets" table and Pet_ID in the "Owners" table) that have the same information in them: the ID number of the pet. We can match rows that have the same value in these columns to get information that applies to a certain pet.
# 
# For example, we can see by looking at the Pets table that the pet that has the ID 1 is named Dr. Harris Bonkers. We can also tell by looking at the Owners table that the name of the owner who owns the pet with the ID 1 is named Aubrey Little. We can use this information to figure out that Dr. Harris Bonkers is owned by Aubrey Little. 
# 
# Fortunately, we don't have to do this by hand to figure out which owner's name goes with which pet name. We can use JOIN to do this for us! JOIN allows us to create a third, new, table that has information from both tables. For example, we might want to have a single table with just two columns: one with the name of the pet and one with the name of the owner. This would look something like this: 
# 
# ![](https://i.imgur.com/zqQdJTI.png)
# 
# The syntax to create that table looks like this:
# 
#     SELECT p.Name AS Pet_Name, o.Name as Owner_Name
#     FROM `bigquery-public-data.pet_records.pets` as p
#     INNER JOIN `bigquery-public-data.pet_records.owners` as o ON p.ID = o.Pet_ID
# Notice that since the ID column exists in both datasets, we have to clarify which one we want to use. When you're joining tables, it's a good habit to specificy which table all of your columns come from. That way you don't have to pull up the schema every time you go back to read the query.
# 
# The type of JOIN we're using today is called an INNER JOIN. That just means that a row will only be put in the final output table if the value in the column you're using to combine them shows up in both the tables you're joining. For example, if Tom's ID code of 4 didn't exist in the `Pets` table, we would only get 3 rows back from this query. There are other types of JOIN, but an INNER JOIN won't give you an output that's larger than your input tables, so it's a good one to start with.   
# 
# > **What does "ON" do?** It says which column in each table to look at to combine the tables. Here were using the "ID" column from the Pets table and the "Pet_ID" table from the Owners table.
# 
# Now that we've talked about the concept behind using JOIN, let's work through an example together.

# ## Example: How many files are covered by each license?
# ____
# 
# Today we're going to be using the GitHub Repos dataset. GitHub is an place for people to store & collaborate on different versions of their computer code. A "repo" is a collection of code associated with a specific project. 
# 
# Most public code on Github is shared under a specific license, which determines how it can be used and by who. For our example, we're going to look at how many different files have been released under each licenses. 
# 
# First, of course, we need to get our environment ready to go:

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")


# Now we're ready to get started on our query. This one is going to be a bit of a beast, so stick with me! The only new syntax we'll see is around the JOIN clause, everything is something we've already learned. :)
# 
# First, I'm going to specify which columns I'd like to be returned in the final table that's returned to me. Here, I'm selecting the COUNT of the "path" column from the sample_files table and then calling it "number_of_files". I'm *also* specifying that I was to include the "license" column, even though there's no "license" column in the "sample_files" table.
# 
#         SELECT L.license, COUNT(sf.path) AS number_of_files
#         FROM `bigquery-public-data.github_repos.sample_files` as sf
# Speaking of the JOIN clause, we still haven't actually told SQL we want to join anything! To do this, we need to specify what type of join we want (in this case an inner join) and how which columns we want to JOIN ON. Here, I'm using ON to specify that I want to use the "repo_name" column from the each table.
# 
#     INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
#             ON sf.repo_name = L.repo_name
# And, finally, we have a GROUP BY and ORDER BY clause that apply to the final table that's been returned to us. We've seen these a couple of times at this point. :)
# 
#         GROUP BY license
#         ORDER BY number_of_files DESC
#  Alright, that was a lot, but you should have an idea what each part of this query is doing. :) Without any further ado, let' put it into action.

# In[ ]:


# You can use two dashes (--) to add comments in SQL
query = ("""
        -- Select all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)


# Whew, that was a big query! But it gave us a nice tidy little table that nicely summarizes how many files have been committed under each license:  

# In[ ]:


# print out all the returned results
print(file_count_by_license)


# And that's how to get started using JOIN in BigQuery! There are many other kinds of joins (you can [read about some here](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#join-types)), so once you're very comfortable with INNER JOIN you can start exploring some of them. :)

# # Scavenger hunt
# ___
# 
# Now it's your turn! Here is the question I would like you to get the data to answer. Just one today, since you've been working hard this week. :)
# 
# *  How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language? (I'm looking for the number of commits per repo for all the repos written in Python.
#     * You'll want to JOIN the sample_files and sample_commits questions to answer this.
#     * **Hint:** You can figure out which files are written in Python by filtering results from the "sample_files" table using `WHERE path LIKE '%.py'`. This will return results where the "path" column ends in the text ".py", which is one way to identify which files have Python code.
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up). "Forking" something is making a copy of it that you can edit on your own without changing the original.

# In[ ]:


# Your code goes here :)



# Please feel free to ask any questions you have in this notebook or in the [Q&A forums](https://www.kaggle.com/questions-and-answers)! 
# 
# Also, if you want to share or get comments on your kernel, remember you need to make it public first! You can change the visibility of your kernel under the "Settings" tab, on the right half of your screen.
