
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('fivethirtyeight')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.


# ### What is [BigQuery](https://cloud.google.com/bigquery/what-is-bigquery)??
# 
# 
# Storing and querying massive datasets can be time consuming and expensive without the right hardware and infrastructure. Google BigQuery is an enterprise data warehouse that solves this problem by enabling super-fast SQL queries using the processing power of Google's infrastructure. Simply move your data into BigQuery and let us handle the hard work. You can control access to both the project and your data based on your business needs, such as giving others the ability to view or query your data.
# 
# You can access BigQuery by using a [web UI ](https://bigquery.cloud.google.com/project/nice-particle-195309)or a [command-line tool](https://cloud.google.com/bigquery/bq-command-line-tool), or by making calls to the BigQuery REST API using a variety of client libraries such as Java, .NET, or Python. There are also a variety of third-party tools that you can use to interact with BigQuery, such as visualizing the data or loading the data.
# Because the datasets on BigQuery can be very large, there are some restrictions on how much data you can access. 
# 
# *But You dont need to go to Google, Since Kaggle kernels allows you to access TeraBytes of data from Google cloud with all saftey measures like not letting your query go above memory limits and helper APIs. Thanks to [Sohier](https://www.kaggle.com/sohier)'s [BigQuery helper module](https://github.com/SohierDane/BigQuery_Helper/blob/master/bq_helper.py).
# Each Kaggle user can scan 5TB every 30 days for free.*
# 
# Let's first setup environment to run BigQueries in Kaggle Kernels.
# 
# ### Importing Kaggle's bq_helper package

# In[ ]:


import bq_helper 


# ### Creating a helper object for  bigquery dataset
# 
# The addresses of BigQuery datasets look like this![](https://i.imgur.com/l11gdKx.png)
# 
# for us dataset is **github_repos**
# 
# [Rachael](https://www.kaggle.com/rtatman) from Kaggle has ran a 5 days BigQuery Introductory challenge called SQL Scavenger Hunt. We will go through day 1 to 5 using Github Repos Dataset.
# 
# Image is taken from [SQL Scavenger Handbook](https://www.kaggle.com/rtatman/sql-scavenger-hunt-handbook)

# In[ ]:


github_repos = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "github_repos")


# ### Listing Tables

# In[ ]:


# print a list of all the tables in the github_repos dataset
github_repos.list_tables()


# ###  Printing Table Schema
# 

# In[ ]:


# print information on all the columns in the "commits" table
# in the github_repos dataset
github_repos.table_schema("commits")


# In[ ]:


# preview the first couple lines of the "commits" table
github_repos.head("commits")


# **[SQL Scavenger Hunt: Day 1](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-1)**
# ### SELECT ... FROM ... WHERE
# 
# Or first query is going to be simple select a single column from a specific table. To do this, you need to tell SELECT which column to select and then specify what table that column is from using FROM.
# 
# When you're working with BigQuery datasets, you're almost always going to want to return only certain rows, usually based on the value of a different column. You can do this using the WHERE clause, which will only return the rows where the WHERE clause evaluates to true.
# 
# ### What are sizes of Github Repositories??
# We will be using **contents** table.

# In[ ]:


github_repos.head("contents")


# ### Checking the size of our query before we run it
# Our Dataset is 3TBs so we can easily cross tha daily limit by running few queries. We should always estimate  how much data we need to scan for executing this query by **BigQueryHelper.estimate_query_size()** method.
# 

# In[ ]:


query1= """SELECT size
            FROM `bigquery-public-data.github_repos.contents`
            WHERE binary = True
            LIMIT 5000
        """
github_repos.estimate_query_size(query1)


# Our query actually scanned 2.34 GB of data. By default any query scanning more than 1GB of data will get cancelled by kaggle kernel environment.
# 
# ### Running a query
# 
# There are 2 ways to do this:
# 
# 1. BigQueryHelper.query_to_pandas(query): This method takes a query and returns a Pandas dataframe.
# 1. BigQueryHelper.query_to_pandas_safe(query, max_gb_scanned=1): This method takes a query and returns a Pandas dataframe only if the size of the query is less than the upperSizeLimit (1 gigabyte by default).
# 
# Here's an example of a query that is larger than the specified upper limit.

# In[ ]:


github_repo_sizes = github_repos.query_to_pandas_safe(query1, max_gb_scanned=2.34)
github_repo_sizes.head()


# Since this query has returned a pandasdataframe, we can work with it as we would any other dataframe. For example, we can get the min and max of the column size:
# 
# 

# In[ ]:


BYTES_PER_MB = 2**20
print("Minimum git repo size is " , github_repo_sizes.min()/BYTES_PER_MB, " bytes")
print("Maximum git repo size is " , github_repo_sizes.max()/BYTES_PER_MB, " bytes");


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(github_repo_sizes.divide(BYTES_PER_MB),color="black")
plt.savefig('github-sizes-on-head-branch.png')
plt.title("Sizes of Github Repos on Head Branch in MBs");


# ### How many github repositories are in form of binary files?
# A binary file is a file stored in binary format. A binary file is computer-readable but not human-readable. All executable programs are stored in binary files, as are most numeric data files.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'query2= """SELECT binary\n            FROM `bigquery-public-data.github_repos.contents`\n            LIMIT 50000\n        """\n\nbinary_files=github_repos.query_to_pandas_safe(query2)')


# In[ ]:


binary_files.head()
sns.countplot(binary_files.binary)
plt.savefig('github-binary-files.png')
plt.title("Binary Vs. Text Files");


# Looks like approximately 5% of the total files is dataset are binary files i.e executables rest all are normal text files.
# 
# ### Which are the Popular Languages in Github?

# In[ ]:


github_repos.head("languages")


# In[ ]:


#%%time
query3= """SELECT language
            FROM `bigquery-public-data.github_repos.languages`
            LIMIT 5000
        """
github_repos.estimate_query_size(query3)


# In[ ]:


github_languages = github_repos.query_to_pandas_safe(query3)
github_languages.head()


# In[ ]:


github_languages.language[0]
languagesList=[]
for lang in github_languages.language:
    languagesList.extend(lang)
languagesList[:5]


# In[ ]:


Languages_count={}
for lang in languagesList:
    if lang["name"] not in Languages_count:
        Languages_count[lang["name"]]=0
    Languages_count[lang["name"]]+=1
#Languages_count


# In[ ]:


import operator
sorted_Languages_counts = sorted(Languages_count.items(), key=operator.itemgetter(1),reverse=True)
sorted_Languages_counts[:15]


# In[ ]:


language = list(zip(*sorted_Languages_counts[:15]))[0]
count = list(zip(*sorted_Languages_counts[:15]))[1]
x_pos = np.arange(len(language))


# calculate slope and intercept for the linear trend line
slope, intercept = np.polyfit(x_pos, count, 1)
trendline = intercept + (slope * x_pos)
plt.figure(figsize=(12,8))
plt.plot(x_pos, trendline, color='black', linestyle='--')    
plt.bar(x_pos, count,align='center',color=sns.color_palette("gist_rainbow",len(x_pos)))
plt.xticks(x_pos, language,rotation=45) 
plt.title('Language Popularity Score')
plt.savefig('github-language-popularity.png');


# My favourite being Python, I must say it's in top 5.
# 
# ### Which are the trending repositories on Github ??
# 

# In[ ]:


github_repos.head("sample_repos")


# In[ ]:


query9 ="""
        SELECT repo_name, watch_count
        FROM `bigquery-public-data.github_repos.sample_repos`
        ORDER BY watch_count DESC 
        LIMIT 2000
        """
github_repos.estimate_query_size(query9)


# In[ ]:


github_repo_trending_repos = github_repos.query_to_pandas_safe(query9)
github_repo_trending_repos.head(15)


# In[ ]:


plt.figure(figsize=(12,8))
g = sns.barplot(y="repo_name", x="watch_count", data=github_repo_trending_repos[:20], palette="winter")
plt.title('Trending Github Repositories')
plt.ylabel("Repository Name")
plt.xlabel("Watch Count")
plt.savefig('github-trending-repo-by-watch-count.png');


# Wohoo my favourite **FreeCodeCamp** repo is at the top and much ahead of others.
# 
# 
# ### Who are the authors with Highest number of repositories?
# 
# The author on github is the person who originally wrote the code. 
# In other words, the author is the person who originally wrote the patch.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'query4= """SELECT author\n            FROM `bigquery-public-data.github_repos.commits`\n            LIMIT 500000\n        """\ngithub_repos.query_to_pandas_safe(query4)')


# In[ ]:


github_repo_authors = github_repos.query_to_pandas_safe(query4, max_gb_scanned=17.5)
github_repo_authors.head()


# In[ ]:


github_repo_authors.author[789]


# In[ ]:


authors_count={}
for author in github_repo_authors.author.values:
    if author["name"] not in authors_count:
        authors_count[author["name"]]=0
    authors_count[author["name"]]+=1
#authors_count


# In[ ]:


import operator
sorted_authors_counts = sorted(authors_count.items(), key=operator.itemgetter(1),reverse=True)
sorted_authors_counts[:15]


# In[ ]:


authors = list(zip(*sorted_authors_counts[:15]))[0]
count = list(zip(*sorted_authors_counts[:15]))[1]
y_pos = np.arange(len(authors))
plt.figure(figsize=(12,8))
plt.barh(y_pos,count,align='center',color=sns.color_palette("viridis",len(x_pos)))
plt.yticks(y_pos,authors,rotation=0) 
plt.title('Authors and their total Repositories')
plt.savefig('github-authors.png');


# **Auto Pilot **running first here with more than 1600 repositories.
# 
# ## Who are the committers with highest commits?
# 
# The committer on github, is assumed to be the person who committed the code on behalf of the original author. 
# Th commiter is who last applied the patch.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'query5= """SELECT committer\n            FROM `bigquery-public-data.github_repos.commits`\n            LIMIT 50000\n        """\ngithub_repos.query_to_pandas_safe(query5)')


# In[ ]:


github_repo_committers = github_repos.query_to_pandas_safe(query5, max_gb_scanned=17.5)
github_repo_committers.head()


# In[ ]:


committers_count={}
for committer in github_repo_committers.committer.values:
    if committer["name"] not in committers_count:
        committers_count[committer["name"]]=0
    committers_count[committer["name"]]+=1
#committers_count


# In[ ]:


import operator
sorted_committers_count = sorted(committers_count.items(), key=operator.itemgetter(1),reverse=True)
sorted_committers_count[:15]


# In[ ]:


committers = list(zip(*sorted_committers_count[:15]))[0]
count = list(zip(*sorted_committers_count[:15]))[1]
y_pos = np.arange(len(committers))

plt.figure(figsize=(12,8))
plt.barh(y_pos,count,align='center',color=sns.color_palette("inferno",len(y_pos)))
plt.yticks(y_pos,committers,rotation=0) 
plt.title('Committers and their total Repositories')
plt.savefig('github-committers.png');


# Looks like by default commiter is github followed by Duane F. King.
# 
# ### How commit messages look like?

# In[ ]:


get_ipython().run_cell_magic('time', '', 'query6 = """\nSELECT message\nFROM `bigquery-public-data.github_repos.commits`\nWHERE LENGTH(message) > 10 AND LENGTH(message) <= 50\nLIMIT 500\n"""\ngithub_repos.query_to_pandas_safe(query6)')


# In[ ]:


github_repo_messages = github_repos.query_to_pandas_safe(query6, max_gb_scanned=17.6)
github_repo_messages.head()


# In[ ]:


from wordcloud import WordCloud
import random

def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

commits = ' '.join(github_repo_messages.message).lower()
# wordcloud for display address
plt.figure(figsize=(12,6))
wc = WordCloud(background_color='gold', max_font_size=200,
                            width=1600,
                            height=800,
                            max_words=400,
                            relative_scaling=.5).generate(commits)
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))
#plt.imshow(wc)
plt.title("Github Commit Messages", fontsize=20)
plt.savefig('github-commit-messages-wordcloud.png')
plt.axis("off");


# ### Which Subject is most comman for Github Repositories??

# In[ ]:


get_ipython().run_cell_magic('time', '', 'query7 = """\nSELECT subject\nFROM `bigquery-public-data.github_repos.commits`\nWHERE LENGTH(subject) > 5 AND LENGTH(subject) <= 10\nLIMIT 500\n"""\ngithub_repos.query_to_pandas_safe(query7)')


# In[ ]:


github_repo_subject = github_repos.query_to_pandas_safe(query7, max_gb_scanned=8.8)
github_repo_subject.head()


# In[ ]:


from wordcloud import WordCloud
import random

commits = ' '.join(github_repo_subject.subject).lower()
# wordcloud for display address
plt.figure(figsize=(12,6))
wc = WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGBA"
, max_font_size=200,
                            width=1600,
                            height=800,
                            max_words=400,
                            relative_scaling=.5).generate(commits)
plt.imshow(wc)
#plt.imshow(wc)
plt.title("Github Subject", fontsize=20)
plt.savefig('github-subject-wordcloud.png')
plt.axis("off");


# **[SQL Scavenger Hunt: Day 2](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-2)**
# 
# ### GROUP BY... HAVING and COUNT
# Now we will be learning how to **GROUP** data **BY** particular column and **COUNT** common occurences.
# And also **ORDER** results **BY** count in ascending or DESCending order..
# 
# 
# ### Which are popular licenses?

# In[ ]:


get_ipython().run_cell_magic('time', '', 'query8 ="""\n        SELECT license, COUNT(*) AS count\n        FROM `bigquery-public-data.github_repos.licenses`\n        GROUP BY license\n        ORDER BY COUNT(*) DESC\n        """\ngithub_repos.query_to_pandas_safe(query8)')


# In[ ]:


github_repo_licenses = github_repos.query_to_pandas_safe(query8)
github_repo_licenses.head()


# In[ ]:


github_repo_licenses.shape


# In[ ]:


plt.figure(figsize=(12,9))
sns.barplot(y="license", x="count", data=github_repo_licenses, palette="viridis")
plt.title('Licenses in order of their popularity in Github Repositories')
plt.savefig('github-licenses-popularity.png');


# **[SQL Scavenger Hunt: Day 5](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-5)**
# 
# ### JOIN
# Till now we were using only single table. To get broader view of real life scenario where we have to use **JOIN** to join 2 tables together based on some column.
# 
# ### How many files are covered by each license?
# 

# In[ ]:


query10 = ("""
        -- Select all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

file_count_by_license = github_repos.query_to_pandas_safe(query10, max_gb_scanned=6)


# In[ ]:


file_count_by_license.head()


# In[ ]:


plt.figure(figsize=(12,9))
g = sns.barplot(y="license", x="number_of_files", data=file_count_by_license, palette="inferno")
plt.title(' Number of Files covered by each License')
plt.savefig('num-of-files-by-license.png')
plt.xlabel("");


# GNU General Public License v2.0	(gpl-2.0) Covered most of the files followed by MIT(mit) license.
# 
# ### How many commits have been made in repos written in the Python programming language?

# In[ ]:


query11 = """
WITH python_repos AS (
    SELECT DISTINCT repo_name -- Notice DISTINCT
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py')
SELECT commits.repo_name, COUNT(commit) AS num_commits
FROM `bigquery-public-data.github_repos.sample_commits` AS commits
JOIN python_repos
    ON  python_repos.repo_name = commits.repo_name
GROUP BY commits.repo_name
ORDER BY num_commits DESC
"""
github_repo_num_commits_distinct = github_repos.query_to_pandas_safe(query11, max_gb_scanned=10)
github_repo_num_commits_distinct


# In[ ]:


plt.figure(figsize=(12,9))
g = sns.barplot(y="repo_name", x="num_commits", data=github_repo_num_commits_distinct[:15], palette="inferno")
plt.title(' Top Python Github Repositories by their commits Count')
plt.savefig('python-by-commits.png')
plt.xlabel("");


# No doubt **LINUX Operating system**'s repo has been applied lot of patches .
# 
# ### Number of Python files in each repositories above
# 

# In[ ]:


query12 = """
SELECT repo_name, COUNT(path) AS num_python_files
FROM `bigquery-public-data.github_repos.sample_files`
WHERE repo_name IN ('torvalds/linux', 'apple/swift', 'Microsoft/vscode', 'facebook/react', 'tensorflow/tensorflow')
    AND path LIKE '%.py'
GROUP BY repo_name
ORDER BY num_python_files DESC
"""

github_repo_num_python_files = github_repos.query_to_pandas_safe(query12, max_gb_scanned=10)
github_repo_num_python_files


# In[ ]:


plt.figure(figsize=(12,9))
g = sns.barplot(y="repo_name", x="num_python_files", data=github_repo_num_python_files, palette="Spectral_r")
plt.title(' Python Github Repositories by their files Count')
plt.savefig('python-by-files-.png')
plt.xlabel("");


# Highest number of Python files are present in Tensorflow Deep Learning Library
# 
# On the similar lines we can find out
# 
# ### How many commits have been made in repos written in the Java programming language?

# In[ ]:


query13 = """
WITH java_repos AS (
    SELECT DISTINCT repo_name -- Notice DISTINCT
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.java')
SELECT commits.repo_name, COUNT(commit) AS num_commits
FROM `bigquery-public-data.github_repos.sample_commits` AS commits
JOIN java_repos
    ON  java_repos.repo_name = commits.repo_name
GROUP BY commits.repo_name
ORDER BY num_commits DESC
"""
github_repo_num_java_distinct = github_repos.query_to_pandas_safe(query13, max_gb_scanned=5.3)
github_repo_num_java_distinct


# In[ ]:


plt.figure(figsize=(12,9))
g = sns.barplot(y="repo_name", x="num_commits", data=github_repo_num_java_distinct, palette="PuBu")
plt.title(' Top Java Github Repositories by their commits Count')
plt.savefig('java-by_commits.png')
plt.xlabel("");


# ### How many times 'This should never happen' appears??
# 
# This query uses a smaller sample table to find how many times the comment "this should never happen" is present.

# In[ ]:


query14 ="""
SELECT
  SUM(copies)
FROM
  `bigquery-public-data.github_repos.sample_contents`
WHERE
  NOT binary
  AND content like '%This should never happen%'
LIMIT 500
"""
github_repos.estimate_query_size(query14)


# In[ ]:


this_should_never_happen_count=github_repos.query_to_pandas_safe(query14, max_gb_scanned=23.7)
this_should_never_happen_count


# HAHA . This has happened 68486 times.
# 
# ### How many GO files are there?

# In[ ]:


query15="""
SELECT COUNT(*)
FROM `bigquery-public-data.github_repos.sample_files`
WHERE path LIKE '%.go'
LIMIT 500
"""
github_repos.estimate_query_size(query15)


# In[ ]:


go_files_count=github_repos.query_to_pandas_safe(query15, max_gb_scanned=3.7)
go_files_count


# there are 629226 GO files.
# 
# ### How many Python files are there?

# In[ ]:


query16="""
SELECT COUNT(*)
FROM `bigquery-public-data.github_repos.sample_files`
WHERE path LIKE '%.py'
LIMIT 500
"""
github_repos.estimate_query_size(query16)


# In[ ]:


python_files_count=github_repos.query_to_pandas_safe(query16, max_gb_scanned=3.7)
python_files_count


# there are 1231972  python files.

# In[ ]:


query17="""
SELECT a.id id, size, content, binary, copies,
  sample_repo_name, sample_path
FROM (
  SELECT id
    , ANY_VALUE(path) sample_path
    , ANY_VALUE(repo_name) sample_repo_name
  FROM `bigquery-public-data.github_repos.sample_files` a
  WHERE PATH LIKE '%.sql'
  GROUP BY 1
) a
JOIN `bigquery-public-data.github_repos.sample_contents` b
ON a.id=b.id
"""
github_repos.estimate_query_size(query17)


# In[ ]:


q_tab_or_space = ('''
#standardSQL
WITH
  lines AS (
  SELECT
    SPLIT(content, '\\n') AS line,
    id
  FROM
    `bigquery-public-data.github_repos.sample_contents`
  WHERE
    sample_path LIKE "%.sql" )
SELECT
  Indentation,
  COUNT(Indentation) AS number_of_occurence
FROM (
  SELECT
    CASE
        WHEN MIN(CHAR_LENGTH(REGEXP_EXTRACT(flatten_line, r',\s*$')))>=1 THEN 'trailing'
        WHEN MIN(CHAR_LENGTH(REGEXP_EXTRACT(flatten_line, r"^ +")))>=1 THEN 'Space'
        ELSE 'Other'
    END AS Indentation
  FROM
    lines
  CROSS JOIN
    UNNEST(lines.line) AS flatten_line
  WHERE
    REGEXP_CONTAINS(flatten_line, r"^\s+")
  GROUP BY
    id )
GROUP BY
  Indentation
ORDER BY
  number_of_occurence DESC
''')

