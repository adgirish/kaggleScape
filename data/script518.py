
# coding: utf-8

# # Data types and missing data reference
# 
# This is the reference section of the "Data types and missing data" section of the tutorial. For the workbook, click [here](https://www.kaggle.com/residentmario/data-types-and-missing-data-workbook).
# 
# In this short section we will look at two inter-related concepts, data types and missing data. This section draws from the [Intro to data structures](https://pandas.pydata.org/pandas-docs/stable/dsintro.html) and [Working with missing data](https://pandas.pydata.org/pandas-docs/stable/missing_data.html) sections of the comprehensive official tutorial.

# In[ ]:


import pandas as pd
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)


# ## Data types
# 
# The data type for a column in a `DataFrame` or a `Series` is known as the `dtype`.
# 
# You can use the `dtype` property to grab the type of a specific column:

# In[ ]:


reviews.price.dtype


# Alternatively, the `dtypes` property returns the `dtype` of _every_ column in the dataset:

# In[ ]:


reviews.dtypes


# Data types tell us something about how `pandas` is storing the data internally. `float64` means that it's using a 64-bit floating point number; `int64` means a similarly sized integer instead, and so on.
# 
# One peculiarity to keep in mind (and on display very clearly here) is that columns consisting entirely of strings do not get their own type; they are instead given the `object` type.
# 
# It's possible to convert a column of one type into another wherever such a conversion makes sense by using the `astype` function. For example, we may transform the `points` column from its existing `int64` data type into a `float64` data type:

# In[ ]:


reviews.points.astype('float64')


# A `DataFrame` or `Series` index has its own `dtype`, too:

# In[ ]:


reviews.index.dtype


# `pandas` also supports more exotic data types: categorical data and timeseries data. Because these data types are more rarely used, we will omit them until a much later section of this tutorial.

# ## Missing data
# 
# Entries missing values are given the value `NaN`, short for "Not a Number". For technical reasons these `NaN` values are always of the `float64` dtype.
# 
# `pandas` provides some methods specific to missing data. To select `NaN` entreis you can use `pd.isnull` (or its companion `pd.notnull`). This is meant to be used thusly:

# In[ ]:


reviews[reviews.country.isnull()]


# Replacing missing values is a common operation.  `pandas` provides a really handy method for this problem: `fillna`. `fillna` provides a few different strategies for mitigating such data. For example, we can simply replace each `NaN` with an `"Unknown"`:

# In[ ]:


reviews.region_2.fillna("Unknown")


# Or we could fill each missing value with the first non-null value that appears sometime after the given record in the database. This is known as the backfill strategy:

# `fillna` supports a few strategies for imputing missing values. For more on that read [the official function documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html).
# 
# Alternatively, we may have a non-null value that we would like to replace. For example, suppose that since this dataset was published, reviewer Kerin O'Keefe has changed her Twitter handle from `@kerinokeefe` to `@kerino`. One way to reflect this in the dataset is using the `replace` method:

# In[ ]:


reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")


# The `replace` method is worth mentioning here because it's handy for replacing missing data which is given some kind of sentinel value in the dataset: things like `"Unknown"`, `"Undisclosed"`, `"Invalid"`, and so on.
