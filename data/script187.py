
# coding: utf-8

# # SQL Scavenger Hunt: Day 5
# ## Example: How many files are covered by each license?

# In[ ]:


import bq_helper


# In[ ]:


github_repos = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='github_repos')


# In[ ]:


query = """
SELECT licenses.license, COUNT(files.id) AS num_files
FROM `bigquery-public-data.github_repos.sample_files` AS files
JOIN `bigquery-public-data.github_repos.licenses` AS licenses -- JOIN is the same as INNER JOIN
    ON licenses.repo_name = files.repo_name
GROUP BY license
ORDER BY num_files DESC
"""

license_counts = github_repos.query_to_pandas_safe(query, max_gb_scanned=10)


# In[ ]:


license_counts.shape


# In[ ]:


license_counts


# ## Scavenger hunt

# * How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language? (I'm looking for the number of commits per repo for all the repos written in Python.
# 
#     * You'll want to JOIN the sample_files and sample_commits questions to answer this.
# 
#     * **Hint:** You can figure out which files are written in Python by filtering results from the "sample_files" table using WHERE path LIKE '%.py'. This will return results where the "path" column ends in the text ".py", which is one way to identify which files have Python code.

# **Attempt 1: WITHOUT distinct Python repo names (Incorrect)**

# In[ ]:


query = """
WITH python_repos AS (
    SELECT repo_name
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py')
SELECT commits.repo_name, COUNT(commit) AS num_commits
FROM `bigquery-public-data.github_repos.sample_commits` AS commits
JOIN python_repos
    ON  python_repos.repo_name = commits.repo_name
GROUP BY commits.repo_name
ORDER BY num_commits DESC
"""

github_repos.query_to_pandas_safe(query, max_gb_scanned=10)


# **Attempt 2: WITH distinct Python repo names (Correct)**

# In[ ]:


query = """
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

github_repos.query_to_pandas_safe(query, max_gb_scanned=10)


# **Attempt 3: Alternative solution using distinct commits (Correct) - Credit to [David V.](https://www.kaggle.com/dvinuales)**

# In[ ]:


query = """
SELECT sf.repo_name as repo, COUNT(DISTINCT sc.commit) AS commits
FROM `bigquery-public-data.github_repos.sample_commits` as sc
INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
    ON sf.repo_name = sc.repo_name
WHERE sf.path LIKE '%.py'
GROUP BY repo
ORDER BY commits DESC
"""

github_repos.query_to_pandas_safe(query, max_gb_scanned=10)


# **Find the number of Python files in each repo above:**

# In[ ]:


query = """
SELECT repo_name, COUNT(path) AS num_python_files
FROM `bigquery-public-data.github_repos.sample_files`
WHERE repo_name IN ('torvalds/linux', 'apple/swift', 'Microsoft/vscode', 'facebook/react', 'tensorflow/tensorflow')
    AND path LIKE '%.py'
GROUP BY repo_name
ORDER BY num_python_files DESC
"""

github_repos.query_to_pandas_safe(query, max_gb_scanned=10)


# The results of Attempt 1 are equal to the results of Attempt 2 (or 3) times the number of Python files in the repo.
