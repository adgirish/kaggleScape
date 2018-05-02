
# coding: utf-8

# # Summary functions and maps reference
# 
# This is the reference component to the "Summary functions and maps" section of the Advanced Pandas tutorial. For the workbook, [click here](https://www.kaggle.com/residentmario/summary-functions-and-maps-workbook).
# 
# This section overlaps with the comprehensive [Essential Basic Functionality](https://pandas.pydata.org/pandas-docs/stable/basics.html) section of the official `pandas` documentation.

# In[ ]:


import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
reviews.head()


# ## Summary functions
# 
# `pandas` provides many simple "summary functions" (not an official name) which restructure the data in some useful way. For example, consider the `describe` method:

# In[ ]:


reviews.points.describe()


# This method generates a high-level summary of the attributes of the given column. It is type-aware, meaning that its output changes based on the `dtype` of the input. The output above only makes sense for numerical data; for string data here's what we get:

# In[ ]:


reviews.taster_name.describe()


# If you want to get some particular simple summary statistic about a column in a `DataFrame` or a `Series`, there is usually a handful `pandas` function that makes it happen. For example, to see the mean of the points allotted (e.g. how well an averagely rated wine does), we can use the `mean` function:

# In[ ]:


reviews.points.mean()


# To see a list of unique values we can use the `unique` function:

# In[ ]:


reviews.taster_name.unique()


# To see a list of unique values _and_ how often they occur in the dataset, we can use the `value_counts` method:

# In[ ]:


reviews.taster_name.value_counts()


# ## Maps
# 
# A "map" is a term, borrowed from mathematics, for a function that takes one set of values and "maps" them to another set of values. In data science we often have a need for creating new representations from existing data, or for transforming data from the format it is in now to the format that we want it to be in later. Maps are what handle this work, making them extremely important for getting your work done!
# 
# There are two mapping functions that you will use often. The `Series` `map` is the first, and slightly simpler one. For example, suppose that we wanted to remean the scores the wines recieved to 0. We can do this as follows:

# In[ ]:


review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)


# `map` takes every value in the column it is being called on and converts it some new value using a function you provide it.
# 
# `map` takes a `Series` as input. The `DataFrame` `apply` function can be used to do the same thing _across_ columns, on the level of the entire dataset. Thus `apply` takes a `DataFrame` as input.

# In[ ]:


def remean_points(srs):
    srs.points = srs.points - review_points_mean
    return srs

reviews.apply(remean_points, axis='columns')


# `pandas` provides many common mapping operations as built-ins. For example, here's a faster way of remeaning our points column:

# In[ ]:


review_points_mean = reviews.points.mean()
reviews.points - review_points_mean


# In this code we are performing an operation between a lot of values on the left-hand side (everything in the `Series`) and a single value on the right-hand side (the mean value). `pandas` looks at this expression and figures out that we must mean to subtract that mean value from every value in the dataset.
# 
# `pandas` will also understand what to do if we perform these operations between `Series` of equal length. For example, an easy way of combining country and region information in the dataset would be to do the following:

# In[ ]:


reviews.country + " - " + reviews.region_1


# These operators are faster than the `map` or `apply` because they uses speed ups built into `pandas`. All of the standard Python operators (`>`, `<`, `==`, and so on) work in this manner.
# 
# However, they are not as flexible as `map` or `apply`, which can do more advanced things, like applying conditional logic, which cannot be done with addition and subtraction alone.
