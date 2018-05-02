
# coding: utf-8

# # Most Common Indentation Space Count in Python Code
# As you may know [PEP8](https://www.python.org/dev/peps/pep-0008/#indentation) defined Python indentation should follow 4 spaces, but is it really common nowadays? because I have often seen 2 spaces indentation in Python code.
# In this kernel, I extracted indentation style from GitHub repositories, inspired by [Aleksey's kernel](https://www.kaggle.com/residentmario/most-common-random-seeds).
# 
# Note: This query extracts number of spaces from head of the line, and choose minimum spaces as the indentation style of the code. This number may not always be the indentation.

# In[ ]:


import pandas as pd
from google.cloud import bigquery
client = bigquery.Client()


# In[ ]:


QUERY = ('''
#standardSQL
WITH
  lines AS (
  SELECT
    SPLIT(content, '\\n') AS line,
    id
  FROM
    `bigquery-public-data.github_repos.sample_contents`
  WHERE
    sample_path LIKE "%.py" )
SELECT
  space_count,
  COUNT(space_count) AS number_of_occurence
FROM (
  SELECT
    id,
    MIN(CHAR_LENGTH(REGEXP_EXTRACT(flatten_line, r"^ +"))) AS space_count
  FROM
    lines
  CROSS JOIN
    UNNEST(lines.line) AS flatten_line
  WHERE
    REGEXP_CONTAINS(flatten_line, r"^ +")
  GROUP BY
    id )
GROUP BY
  space_count
ORDER BY
  number_of_occurence DESC
''')

query_job = client.query(QUERY)

iterator = query_job.result(timeout=30)
rows = list(iterator)


# In[ ]:


rows = [dict(row) for row in rows]
df = pd.DataFrame(rows)


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df[:6].plot(kind='bar', x='space_count', y='number_of_occurence')

ax = plt.gca()
ax.set_ylabel('Number of Occurence')
ax.set_xlabel('Indentation Space Count')
pass


# 4 spaces indentation is quite popular, but 2 spaces indentation is 2nd place!
