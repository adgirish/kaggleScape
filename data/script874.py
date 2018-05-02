
# coding: utf-8

# In this kernel I'll show you some other useful BigQuery (BQ) API functions and provide resources for learning on your own.
# 
# Our first goal will be to list all of the tables available in the Hacker News dataset. To get started, we'll need to load the bigquery client. 
# 
# You may want to have the documentation open as you follow along:
# - [Python client guide](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/usage.html)
# - [BQ API reference](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html)

# In[ ]:


from google.cloud import bigquery


# In[ ]:


client = bigquery.Client()


# Next, we'll load the Hacker News dataset. There are a few gotchas to bear in mind here:
# - To load a [dataset](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html#google.cloud.bigquery.dataset.Dataset) you first need to generate a [dataset reference](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html#google.cloud.bigquery.dataset.DatasetReference) to point BQ to it. 
# - Any time you're working with BQ from Kaggle the project name is `bigquery-public-data`. 
# - Kaggle imposes some limitations on the BQ API to make it work from our platform, but most read-only commands will work. One key exception is `client.list_datasets` so you may want to look at one of the starter kernels to get the ID for your dataset.

# In[ ]:


hn_dataset_ref = client.dataset('hacker_news', project='bigquery-public-data')


# The method [client.dataset](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html#google.cloud.bigquery.dataset.DatasetReference) is named as if it returns a dataset, but it actually gives us  a dataset reference.

# In[ ]:


type(hn_dataset_ref)


# Once we have a reference, we can load the real dataset.

# In[ ]:


hn_dset = client.get_dataset(hn_dataset_ref)


# In[ ]:


type(hn_dset)


# Now we can use [client.list_tables](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html#google.cloud.bigquery.client.Client.list_tables) to get information about the tables within the dataset.

# In[ ]:


[x.table_id for x in client.list_tables(hn_dset)]


# Let's take a closer look at the schema for the `full` table. As with datasets, we'll need to pass a reference to the table to the `client.get_table` method.

# In[ ]:


hn_full = client.get_table(hn_dset.table('full'))


# In[ ]:


type(hn_full)


# Rather than looking at the documentation, I'll try being a little more independent while digging into the table commands.

# In[ ]:


[command for command in dir(hn_full) if not command.startswith('_')]


# The schema sounds helpful, so let's try that.

# In[ ]:


hn_full.schema


# It turns out that the schema is a necessary input for one of the more useful commands in BQ: [list_rows](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/reference.html#google.cloud.bigquery.client.Client.list_rows). `List_rows` returns a slice of a dataset without scanning any other section of the table. If you've ever written a BQ query that included a `limit` clause, which limits the data returned rather than the data scanned, you probably actually wanted `list_rows` instead.
# 
# I'd like to see a subset of the columns, but the `selected_fields` parameter requires a schema object as an input. We'll need to build a subset of the schema first to pass for that parameter.

# In[ ]:


schema_subset = [col for col in hn_full.schema if col.name in ('by', 'title', 'time')]
results = [x for x in client.list_rows(hn_full, start_index=100, selected_fields=schema_subset, max_results=10)]


# You can print the results directly, but I find that to be a little hard to read. It displays some boilerplate, then all of the values, then all of the keys.

# In[ ]:


print(results)


# Usually I'd probably pass this into pandas before doing further work, but for now we can just convert the `google.cloud.bigquery.table.Row` results to dicts to get a version that prints a bit more nicely.

# In[ ]:


for i in results:
    print(dict(i))


# Suppose we wanted to check what resources we would have consumed by doing a full table scan instead of using `list_rows`. Looks like the `num_bytes` method should help us there.

# In[ ]:


BYTES_PER_GB = 2**30
hn_full.num_bytes / BYTES_PER_GB


# This matches the value given on [the dataset documentation page](https://bigquery.cloud.google.com/table/bigquery-public-data:hacker_news.full?tab=details).

# In[ ]:


def estimate_gigabytes_scanned(query, bq_client):
    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun
    my_job_config = bigquery.job.QueryJobConfig()
    my_job_config.dry_run = True
    my_job = bq_client.query(query, job_config=my_job_config)
    BYTES_PER_GB = 2**30
    return my_job.total_bytes_processed / BYTES_PER_GB


# Now we can run a quick test checking the impact of  selecting one column versus an entire table. 

# In[ ]:


estimate_gigabytes_scanned("SELECT Id FROM `bigquery-public-data.hacker_news.stories`", client)


# In[ ]:


estimate_gigabytes_scanned("SELECT * FROM `bigquery-public-data.hacker_news.stories`", client)

