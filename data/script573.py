
# coding: utf-8

# # Method chaining reference
# 
# This is the referenc component of the "Method chaining" section of the Advanced Pandas tutorial. For the workbook component, click [here](https://www.kaggle.com/residentmario/method-chaining-workbook).

# In[ ]:


import pandas as pd
pd.set_option('max_rows', 5)
wine = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)


# ## Why method chaining?
# 
# Method chaining is the last topic we will cover in this first track of the Advanced Pandas tutorial. It is also the only section of this tutorial which is a technique or a pattern, not a function or variable.
# 
# Method chaining is a methodology for performing operations on a `DataFrame` or `Series` that emphasizes continuity. To demonstrate what I mean, here's a data cleaning and dropping operation (which you should be familiar with from the last section) done two different ways:

# In[ ]:


stars = ramen['Stars']
na_stars = stars.replace('Unrated', None).dropna()
float_stars = na_stars.astype('float64')
float_stars.head()


# In[ ]:


(ramen['Stars']
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .head())


# In the first statement we assign data to temporary variables, creating new ones as we go further and further along. In the second statement, written in the method chaining style, we instead "chain" our operations, one after the other, all on the same original `DataFrame`.
# 
# Most `pandas` operations can written in a method chaining style, and in the last couple years or so `pandas` has added more and more tools for making these sorts of statements easier to write. This paradigm comes to us from the R programming language&mdash;specifically, the `dpyler` module, part of the "Tidyverse".
# 
# Method chaining is advantageous for several reasons. One is that it lessens the need for creating and mentally tracking temporary variables. Another is that it emphasizes a correctly structured interative approach to working with data, where each operation is a "next step" after the last. Debugging is easy: just comment out operations that don't work until you get to one that does, and then start stepping forward again. And it looks kind of cool. =)
# 
# For a deeper exploration of why method chaining, read the [Method Chaining Section of the Modern Pandas Tutorial](https://tomaugspurger.github.io/method-chaining.html) (written by a `pandas` core dev).

# ## Assign and pipe
# 
# Now that we've learned all these ways of manipulating data with `pandas`, we're ready to take advantage of method chaining to write clear, clean data manipulation code. Now I'll introduce three additional methods useful for coding in this style.

# In[ ]:


wine.head()


# The first of these is `assign`. The `assign` method lets you create new columns or modify old ones inside of a `DataFrame` inline. For example, to fill the `region_1` field with the `province` field wherever the `region_1` is null (useful if we're mixing in our own categories), we would do:

# In[ ]:


wine.assign(
    region_1=wine.apply(lambda srs: srs.region_1 if pd.notnull(srs.region_1) else srs.province, 
                        axis='columns')
)


# Which is equivalent to:
# 
#     wine['region_1'] = wine['region_1'].apply(
#         lambda srs: srs.region_1 if pd.notnull(srs.region_1) else srs.province, 
#         axis='columns'
#     )
# 
# You can modify as many old columns and create as many new ones as you'd like with `assign`, but it does have the limitation that the column being modified must not have any reserved characters like periods (`.`) or spaces (` `) in the name.
# 
# The next method to know is `pipe`. `pipe` is a little mind-bending: it lets you perform an operation on the entire `DataFrame` at once, and replaces the current `DataFrame` which the output of your `pipe`.
# 
# For example, one way to change the give the `DataFrame` index a new name would be to do:

# In[ ]:


def name_index(df):
    df.index.name = 'review_id'
    return df

wine.pipe(name_index)


# `pipe` is a power tool: it comes in handy when you're performing _very_ intricate operations on your `DataFrame`. You won't need it often, but it'll be super useful when you do.
# 
# That concludes this tutorial! Bravo!
