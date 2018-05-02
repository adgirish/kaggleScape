
# coding: utf-8

# This tutorial will walk you through the essentials of how to index & filter data with Pandas. Think of it as a greatly condensed, opinionated, version of [the official indexing documentation.](http://pandas.pydata.org/pandas-docs/stable/indexing.html#).
# 
# We'll start by loading Pandas and the data:

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('../input/parks.csv', index_col=['Park Code'])


# In[3]:


df.head(3)


# ### Indexing: Single Rows
# The simplest way to access a row is to pass the row number to the `.iloc` method. Note that first row is zero, just like list indexes.

# In[4]:


df.iloc[2]


# The other main approach is to pass a value from your dataframe's index to the `.loc` method:

# In[5]:


df.loc['BADL']


# ### Indexing: Multiple Rows
# If we need multiple rows, we can pass in multiple index values. Note that this changes the order of the results!

# In[6]:


df.loc[['BADL', 'ARCH', 'ACAD']]


# In[7]:


df.iloc[[2, 1, 0]]


# Slicing the dataframe just as if it were a list also works.

# In[8]:


df[:3]


# In[9]:


df[3:6]


# ### Indexing: Columns
# We can access a subset of the columns in a dataframe by placing the list of columns in brackets like so:

# In[10]:


df['State'].head(3)


# You can also access a single column as if it were an attribute of the dataframe, but only if the name has no spaces, uses only basic characters, and doesn't share a name with a dataframe method. So, `df.State` works:

# In[11]:


df.State.head(3)


# but `df.Park Code` will fail as there's a space in the name:

# In[12]:


df.Park Code


# We can only access the 'Park Code' column by passing its name as a string in brackets, like `df['Park Code']`. I recommend either always using that approach or always converting your column names into a valid format as soon as you read in the data so that you don't have to mix the two methods. It's just a bit tidier.
# 
# It's a good practice to clean your column names to prevent this sort of error. I'll use a very short cleaning function here since the names don't have any odd characters. By convention, the names should also be converted to lower case. Pandas is case sensitive, so future calls to all of the columns will need to be updated.

# In[13]:


df.columns = [col.replace(' ', '_').lower() for col in df.columns]
print(df.columns)


# ### Indexing: Columns and Rows
# If we need to subset by both columns and rows, you can stack the commands we've already learned.

# In[16]:


df[['state', 'acres']][:3]


# ### Indexing: Scalar Values
# As you may have noticed, everything we've tried so far returns a small dataframe or series. If you need a single value, simply pass in a single column and index value.

# In[17]:


df.state.iloc[2]


# Note that you will get a different return type if you pass a single value in a list.

# In[18]:


df.state.iloc[[2]]


# ### Selecting a Subset of the Data

# The main method for subsetting data in Pandas is called [boolean indexing](http://pandas.pydata.org/pandas-docs/stable/indexing.html#boolean-indexing). First, let's take a look at what pandas does when we ask it to evaluate a boolean:

# In[19]:


(df.state == 'UT').head(3)


# We get a series of the results of the boolean. Passing that series into a dataframe gives us the subset of the dataframe where the boolean evaluates to `True`.

# In[20]:


df[df.state == 'UT']


# Some of the logical operators are different:
# - `~` replaces `not`
# - `|` replaces `or`
# - `&` replaces `and`
# 
# If you have multiple arguments they'll need to be wrapped in parentheses. For example:

# In[21]:


df[(df.latitude > 60) | (df.acres > 10**6)].head(3)


# You can also use more complicated expressions, including lambdas.

# In[22]:


df[df['park_name'].str.split().apply(lambda x: len(x) == 3)].head(3)


# ### Key Companion Methods: `isin` and `isnull`
# These methods make it much easier and faster to perform some very common tasks. Suppose we wanted to find all parks on the West coast. `isin` makes that simple:

# In[23]:


df[df.state.isin(['WA', 'OR', 'CA'])].head()


# ### Less Common Methods
# Pandas offers many more indexing methods. You should probably stick to a few of them for the sake of keeping your code readable, but it's worth knowing they exist in case you need to read other people's code or have an unusual use case:
# 
# - There are other ways to slice data with brackets. For the sake of readability, please don't use of them.
# - `.at` and `.iat`: like `.loc` and `.iloc` but much faster in exchange for only working on a single column and only returning a single result.
# - `.eval`: fast evaluation of a limited set of simple operators. `.query` works by calling this.
# - `.ix`: deprecated method that tried to determine if an index should be evaluated with .loc or .iloc. This led to a lot of subtle bugs! If you see this, you're looking at old code that won't work any more.
# - `.get`: like `.loc`, but will return a default value if the key doesn't exist in the index. Only works on a single column/series.
# - `.lookup`: Not recommended. It's in the documentation, but it's unclear if this is actually still supported.
# - `.mask`: like boolean indexing, but returns a dataframe/series of the same size as the original and anywhere that the boolean evaluates to `True` is set to `nan`.
# - `.query`: similar to boolean indexing. Faster for large dataframes. Only supports a restricted set of operations; don't use if you need `isnull()` or other dataframe methods.
# - `.take`: equivalent to `.iloc`, but can operate on either rows or columns.
# - `.where`: like boolean indexing, but returns a dataframe/series of the same size as the original and anywhere that the boolean evaluates to `False` is set to `nan`.
# - [Multi-indexing](http://pandas.pydata.org/pandas-docs/stable/advanced.html): potentially useful for small to mid sized heirarchical datasets. Slow on larger datasets.
