
# coding: utf-8

# # Indexing, selecting, assigning reference
# 
# This is the reference component to the "Indexing, selecting, assigning" section of the Advanced Pandas track. For the workbook component, [click here](https://www.kaggle.com/residentmario/indexing-selecting-assigning-workbook).

# In[ ]:


import pandas as pd
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)


# Selecting specific values of a `pandas` `DataFrame` or `Series` to work on is an implicit step in almost any data operation you'll run. Hence a solid understanding of how to slice and dice a dataset is vital.

# ## Naive accessors
# 
# Native Python objects provide many good ways of indexing data. `pandas` carries all of these over, which helps make it easy to start with.
# 
# Consider this `DataFrame`:

# In[ ]:


reviews


# In Python we can access the property of an object by accessing it as an attribute. A `book` object, for example, might have a `title` property, which we can access by calling `book.title`. Columns in a `pandas` `DataFrame` work in much the same way. 
# 
# Hence to access the `country` property of our `reviews` we can use:

# In[ ]:


reviews.country


# If we have a `dict` object in Python, we can access its values using the indexing (`[]`) operator. Again, we can do the same with `pandas` `DataFrame` columns. It "just works":

# In[ ]:


reviews['country']


# These are the two ways of selecting a specific columnar `Series` out of a `pandas` `DataFrame`. Neither of them is more or less syntactically valid than the other, but the indexing operator `[]` does have the advantage that it can handle column names with reserved characters in them (e.g. if we had a `country providence` column, `reviews.country providence` wouldn't work).
# 
# Doesn't a `pandas` `Series` look kind of like a fancy `dict`? It pretty much is, so it's no surprise that, to drill down to a single specific value, we need only use the indexing operator `[]` once more:

# In[ ]:


reviews['country'][0]


# ## Index-based selection
# 
# The indexing operator and attribute selection are nice because they work just like they do in the rest of the Python ecosystem. As a novice, this makes them easy to pick up and use. However, `pandas` has its own accessor operators, `loc` and `iloc`. For more advanced operations, these are the ones you're supposed to be using.
# 
# `pandas` indexing works in one of two paradigms. The first is **index-based selection**: selecting data based on its numerical position in the data. `iloc` follows this paradigm.
# 
# To select the first row of data in this `DataFrame`, we may use the following:

# In[ ]:


reviews.iloc[0]


# Both `loc` and `iloc` are row-first, column-second. This is the opposite of what we do in native Python, which is column-first, row-second.
# 
# This means that it's marginally easier to retrieve rows, and marginally harder to get retrieve columns. To get a column with `iloc`, we can do the following:

# In[ ]:


reviews.iloc[:, 0]


# On its own the `:` operator, which also comes from native Python, means "everything". When combined with other selectors, however, it can be used to indicate a range of values. For example, to select the `country` column from just the first, second, and third row, we would do:

# In[ ]:


reviews.iloc[:3, 0]


# Or, to select just the second and third entries, we would do:

# In[ ]:


reviews.iloc[1:3, 0]


# It's also possible to pass a list:

# In[ ]:


reviews.iloc[[0, 1, 2], 0]


# Finally, it's worth knowing that negative numbers can be used in selection. This will start counting forwards from the _end_ of the values. So for example here are the last five elements of the dataset.

# In[ ]:


reviews.iloc[-5:]


# ## Label-based selection
# 
# The second paradigm for attribute selection is the one followed by the `loc` operator: **label-based selection**. In this paradigm it's the data index value, not its position, which matters.
# 
# For example, to get the first entry in `reviews`, we would now do the following:

# In[ ]:


reviews.loc[0, 'country']


# `iloc` is conceptually simpler than `loc` because it ignores the dataset's indices. When we use `iloc` we treat the dataset like a big matrix (a list of lists), one that we have to index into by position. `loc`, by contrast, uses the information in the indices to do its work. Since your dataset usually has meaningful indices, it's usually easier to do things using `loc` instead. For example, here's one operation that's much easier using `loc`:

# In[ ]:


reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]


# When choosing or transitioning between `loc` and `iloc`, there is one "gotcha" worth keeping in mind, which is that the two methods use slightly different indexing schemes.
# 
# `iloc` uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. So 0:10 will select entries 0,...,9. `loc`, meanwhile, indexes inclusively. So 0:10 will select entries 0,...,10.
# 
# Why the change? Remember that loc can index any stdlib type: strings, for example. If we have a DataFrame with index values `Apples, ..., Potatoes, ...`, and we want to select "all the alphabetical fruit choices between Apples and Potatoes", then it's a heck of a lot more convenient to index `df.loc['Apples':'Potatoes']` than it is to index something like `df.loc['Apples', 'Potatoet]` (`t` coming after `s` in the alphabet).
# 
# This is particularly confusing when the `DataFrame` index is a simple numerical list, e.g. `0,...,1000`. In this case `df.iloc[0:1000]` will return 999 entries, while `df.loc[0:1000]` return 1000 of them! Earlier versions of this tutorial did not point this out explicitly, leading to a lot of user confusion on some of the related answers, so we've included this note here explaining this issue.
# 
# Otherwise, the semantics of using `loc` are the same as those for `iloc`.

# ## Manipulating the index
# 
# Label-based selection derives its power from the labels in the index. Critically, the index we use is not immutable. We can manipulate the index in any way we see fit.
# 
# The `set_index` method can be used to do the job. Here is what happens when we `set_index` to the `title` field:

# In[ ]:


reviews.set_index("title")


# Performing a `set_index` is useful if you can come up with an index for the dataset which is better than the current one.

# ## Conditional selection
# 
# So far we've been indexing various strides of data, using structural properties of the `DataFrame` itself. To do *interesting* things with the data, however, we often need to ask questions based on conditions. 
# 
# For example, suppose that we're interested specifically in better-than-average wines produced in Italy.
# 
# We can start by asking each wine if it's Italian or not:

# In[ ]:


reviews.country == 'Italy'


# This operation produced a `Series` of `True`/`False` booleans based on the `country` of each record.  This result can then be used inside of `loc` to select the relevant data:

# In[ ]:


reviews.loc[reviews.country == 'Italy']


# This `DataFrame` has ~20,000 rows. The original had ~130,000. That means that around 15% of wines originate from Italy.
# 
# We also wanted to know which ones are better than average. Wines are reviewed on a 80-to-100 point scale, so this could mean wines that accrued at least 90 points.
# 
# We can use the ampersand (`&`) to bring the two questions together:

# In[ ]:


reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]


# Suppose we'll buy any wine that's made in Italy _or_ which is rated above average. For this we use a pipe (`|`):

# In[ ]:


reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]


# `pandas` comes with a few pre-built conditional selectors, two of which we will highlight here. The first is `isin`. `isin` is lets you select data whose value "is in" a list of values. For example, here's how we can use it to select wines only from Italy or France:

# In[ ]:


reviews.loc[reviews.country.isin(['Italy', 'France'])]


# The second is `isnull` (and its companion `notnull`). These methods let you highlight values which are or are not empty (`NaN`). For example, to filter out wines lacking a price tag in the dataset, here's what we would do:

# In[ ]:


reviews.loc[reviews.price.notnull()]


# ## Assigning data
# 
# Going the other way, assigning data to a `DataFrame` is easy. You can assign either a constant value:

# In[ ]:


reviews['critic'] = 'everyone'
reviews['critic']


# Or with an iterable of values:

# In[ ]:


reviews['index_backwards'] = range(len(reviews), 0, -1)
reviews['index_backwards']


# We will see much more `DataFrame` assignment going on in later sections of this tutorial.
