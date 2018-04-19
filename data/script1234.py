
# coding: utf-8

# # Renaming and combining reference
# 
# ## Introduction
# 
# This is the reference part of the "Renaming and combining" section of the Advanced Pandas tutorial. For the workbook section, click [here](https://www.kaggle.com/residentmario/renaming-and-combining-workbook).
# 
# Renaming is covered in its own section in the ["Essential Basic Functionality"](https://pandas.pydata.org/pandas-docs/stable/basics.html#renaming-mapping-labels) section of the extensive official documentation. Combining is covered by the ["Merge, join, concatenate"](https://pandas.pydata.org/pandas-docs/stable/merging.html) section there.

# In[ ]:


import pandas as pd
pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews


# ## Renaming
# 
# Oftentimes data will come to us with column names, index names, or other naming conventions that we are not satisfied with. In that case, we may use `pandas` renaming utility functions to change the names of the offending entries to something better.
# 
# The first function we'll introduce here is `rename`, which lets you rename index names and/or column names. For example, to change the `points` column in our dataset to `score`, we would do:

# In[ ]:


reviews.rename(columns={'points': 'score'})


# `rename` lets you rename index _or_ column values by specifying a `index` or `column` keyword parameter, respectively. It supports a variety of input formats, but I usually find a Python `dict` to be the most convenient one. Here is an example using it to rename some elements on the index.

# In[ ]:


reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})


# You'll probably rename columns very often, but rename index values very rarely.  For that, `set_index` is usually more convenient.
# 
# Both the row index and the column index can have their own `name` attribute. The complimentary `rename_axis` method may be used to change these names. For example:

# In[ ]:


reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')


# ## Combining

# When performing operations on a dataset we will sometimes need to combine different `DataFrame` and/or `Series` in non-trivial ways. `pandas` has three core methods for doing this. In order of increasing complexity, these are `concat`, `join`, and `merge`. Most of what `merge` can do can also be done more simply with `join`, so I will omit it and focus on the first two functions here.
# 
# The simplest combining method is `concat`. This function works just like the `list.concat` method in core Python: given a list of elements, it will smush those elements together along an axis.
# 
# This is useful when we have data in different `DataFrame` or `Series` objects but having the same fields (columns). One example: the [YouTube Videos dataset](https://www.kaggle.com/datasnaek/youtube-new), which splits the data up based on country of origin (e.g. Canada and the UK, in this example). If we want to study multiple countries simultaneously, we can use `concat` to smush them together:

# In[ ]:


canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])


# The middlemost combiner in terms of complexity is `pd.DataFrame.join`. `join` lets you combine different `DataFrame` objects which have an index in common. For example, to pull down videos that happened to be trending on the same day in _both_ Canada and the UK, we could do the following:

# In[ ]:


left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')


# The `lsuffix` and `rsuffix` parameters are necessary here because the data has the same column names in both British and Canadian datasets. If this wasn't true (because, say, we'd renamed them beforehand) we wouldn't need them.
