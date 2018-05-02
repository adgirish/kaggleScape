
# coding: utf-8

# # Plotting with seaborn
# 
# <table>
# <tr>
# <td><img src="https://i.imgur.com/3cYy56H.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/V9jAreo.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/5a6dwtm.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/ZSsHzrA.png" width="350px"/></td>
# </tr>
# <tr>
# <td style="font-weight:bold; font-size:16px;">Count (Bar) Plot</td>
# <td style="font-weight:bold; font-size:16px;">KDE Plot</td>
# <td style="font-weight:bold; font-size:16px;">Joint (Hex) Plot</td>
# <td style="font-weight:bold; font-size:16px;">Violin Plot</td>
# </tr>
# <tr>
# <td>sns.countplot()</td>
# <td>sns.kdeplot()</td>
# <td>sns.jointplot()</td>
# <td>sns.violinplot()</td>
# </tr>
# <tr>
# <td>Good for nominal and small ordinal categorical data.</td>
# <td>Good for interval data.</td>
# <td>Good for interval and some nominal categorical data.</td>
# <td>Good for interval data and some nominal categorical data.</td>
# </tr>
# </table>
# 
# In the previous two sections we explored data visualization using the `pandas` built-in plotting tools. In this section, we'll do the same with `seaborn`.
# 
# `seaborn` is a standalone data visualization package that provides many extremely valuable data visualizations in a single package. It is generally a much more powerful tool than `pandas`; let's see why.

# In[2]:


import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
import seaborn as sns


# ## Countplot
# 
# The `pandas` bar chart becomes a `seaborn` `countplot`.

# In[3]:


sns.countplot(reviews['points'])


# Comparing this chart with the bar chart from two notebooks ago, we find that, unlike `pandas`, `seaborn` doesn't require us to shape the data for it via `value_counts`; the `countplot` (true to its name) aggregates the data for us!
# 
# `seaborn` doesn't have a direct analogue to the line or area chart. Instead, the package provides a `kdeplot`:

# ## KDE Plot

# In[4]:


sns.kdeplot(reviews.query('price < 200').price)


# KDE, short for "kernel density estimate", is a statistical technique for smoothing out data noise. It addresses an important fundamental weakness of a line chart: it will buff out outlier or "in-betweener" values which would cause a line chart to suddenly dip.
# 
# For example, suppose that there was just one wine priced 19.93\$, but several hundred prices 20.00\$. If we were to plot the value counts in a line chart, our line would dip very suddenly down to 1 and then back up to around 1000 again, creating a strangely "jagged" line. The line chart with the same data, shown below for the purposes of comparison, has exactly this problem!

# In[5]:


reviews[reviews['price'] < 200]['price'].value_counts().sort_index().plot.line()


# A KDE plot is better than a line chart for getting the "true shape" of interval data. In fact, I recommend always using it instead of a line chart for such data.
# 
# However, it's a worse choice for ordinal categorical data. A KDE plot expects that if there are 200 wine rated 85 and 400 rated 86, then the values in between, like 85.5, should smooth out to somewhere in between (say, 300). However, if the value in between can't occur (wine ratings of 85.5 are not allowed), then the KDE plot is fitting to something that doesn't exist. In these cases, use a line chart instead.
# 
# KDE plots can also be used in two dimensions.

# In[6]:


sns.kdeplot(reviews[reviews['price'] < 200].loc[:, ['price', 'points']].dropna().sample(5000))


# Bivariate KDE plots like this one are a great alternative to scatter plots and hex plots. They solve the same data overplotting issue that scatter plots suffer from and hex plots address, in a different but similarly visually appealing. However, note that bivariate KDE plots are very computationally intensive. We took a sample of 5000 points in this example to keep compute time reasonable.

# ## Distplot
# 
# The `seaborn` equivalent to a `pandas` histogram is the `distplot`. Here's an example:

# In[7]:


sns.distplot(reviews['points'], bins=10, kde=False)


# The `distplot` is a composite plot type. In the example above we've turned off the `kde` that's included by default, and manually set the number of bins to 10 (two possible ratings per bin), to get a clearer picture.

# ## Scatterplot and hexplot

# To plot two variables against one another in `seaborn`, we use `jointplot`.

# In[8]:


sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100])


# Notice that this plot comes with some bells and whistles: a correlation coefficient is provided, along with histograms on the sides. These kinds of composite plots are a recurring theme in `seaborn`. Other than that, the `jointplot` is just like the `pandas` scatter plot.
# 
# As in `pandas`, we can use a hex plot (by simply passing `kind='hex'`) to deal with overplotting:

# In[9]:


sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100], kind='hex', 
              gridsize=20)


# ## Boxplot and violin plot
# 
# `seaborn` provides a boxplot function. It creates a statistically useful plot that looks like this:

# In[10]:


df = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]

sns.boxplot(
    x='variety',
    y='points',
    data=df
)


# The center of the distributions shown above is the "box" in boxplot. The top of the box is the 75th percentile, while the bottom is the 25th percentile. In other words, half of the data is distributed within the box! The green line in the middle is the median.
# 
# The other part of the plot, the "whiskers", shows the extent of the points beyond the center of the distribution. Individual circles beyond *that* are outliers.
# 
# This boxplot shows us that although all five wines recieve broadly similar ratings, Bordeaux-style wines tend to be rated a little higher than a Chardonnay.
# 
# Boxplots are great for summarizing the shape of many datasets. They also don't have a limit in terms of numeracy: you can place as many boxes in the plot as you feel comfortable squeezing onto the page.
# 
# However, they only work for interval variables and nominal variables with a large number of possible values; they assume your data is roughly normally distributed (otherwise their design doesn't make much sense); and they don't carry any information about individual values, only treating the distribution as a whole.
# 
# I find the slightly more advanced `violinplot` to be more visually enticing, in most cases:

# In[11]:


sns.violinplot(
    x='variety',
    y='points',
    data=reviews[reviews.variety.isin(reviews.variety.value_counts()[:5].index)]
)


# A `violinplot` cleverly replaces the box in the boxplot with a kernel density estimate for the data. It shows basically the same data, but is harder to misinterpret and much prettier than the utilitarian boxplot.

# ## Why seaborn?
# 
# Having now seen both `pandas` plotting and the `seaborn` library in action, we are now in a position to compare the two and decide when to use which for what.
# 
# Recall the data we've been working with in this tutorial is in:

# In[12]:


reviews.head()


# This data is in a "record-oriented" format. Each individual row is a single record (a review); in aggregate, the list of all rows is the list of all records (all reviews). This is the format of choice for the most kinds of data: data corresponding with individual, unit-identifiable "things" ("records"). The majority of the simple data that gets generated is created in this format, and data that isn't can almost always be converted over. This is known as a "tidy data" format.
# 
# `seaborn` is designed to work with this kind of data out-of-the-box, for all of its plot types, with minimal fuss. This makes it an incredibly convenient workbench tool.
# 
# `pandas` is not designed this way. In `pandas`, every plot we generate is tied very directly to the input data. In essence, `pandas` expects your data being in exactly the right *output* shape, regardless of what the input is.
# 
# <!--
# In the previous section of this tutorial, we purposely evaded this issue by using supplemental datasets in a "just right" shape. Starting from the data that we already have, here's what it would take to generate a simple histogram:
# 
# ```python
# import numpy as np
# top_five_wines_scores = (
#     reviews
#         .loc[np.where(reviews.variety.isin(reviews.variety.value_counts().head(5).index))]
#         .loc[:, ['variety', 'points']]
#         .groupby('variety')
#         .apply(lambda df: pd.Series(df.points.values))
#         .unstack()
#         .T
# )
# top_five_wines_scores.plot.hist()
# ```
# 
# As we demonstrated above, to do the same thing in `seaborn`, all we need is:
# 
# ```python
# sns.distplot(reviews.points, bins=10, kde=False)
# ```
# 
# The difference is stark!
# -->
# 
# Hence, in practice, despite its simplicity, the `pandas` plotting tools are great for the initial stages of exploratory data analytics, but `seaborn` really becomes your tool of choice once you start doing more sophisticated explorations.
# 
# <!--
# My recommendations are:
# * Bar plot: 
#   * `pd.Series.plot.bar`
#   * `sns.countplot`
# * Scatter plot:
#   * `pd.Series.plot.scatter`
#   * `sns.jointplot`
# * Hex plot:
#   * `pd.Series.plot.hex`
#   * `sns.jointplot`
# * Line/KDE plot:
#   * `pd.Series.plot.line` for nominal categorical variables
#   * `sns.kdeplot` for interval variables
# * Box/Violin plot:
#   * `sns.boxplot`
#   * `sns.violinplot`
# * Histogram:
#    * `sns.distplot`
# -->

# # Examples
# 
# As in previous notebooks, let's now test ourselves by answering some questions about the plots we've used in this section. Once you have your answers, click on "Output" button below to show the correct answers.
# 
# 1. A `seaborn` `countplot` is equivalent to what in `pandas`?
# 2. A `seaborn` `jointplot` is equivalent to a what in `pandas`?
# 3. Why might a `kdeplot` not work very well for ordinal categorical data?
# 4. What does the "box" in a `boxplot` represent?

# In[13]:


from IPython.display import HTML
HTML("""
<ol>
<li>A seaborn countplot is like a pandas bar plot.</li>
<li>A seaborn jointplot is like a pandas hex plot.</li>
<li>KDEPlots work by aggregating data into a smooth curve. This is great for interval data but doesn't always work quite as well for ordinal categorical data.</li>
<li>The top of the box is the 75th percentile. The bottom of the box is the 25th percentile. The median, the 50th percentile, is the line in the center of the box. So 50% of the data in the distribution is located within the box!</li>
</ol>
""")


# Next, try forking this kernel, and see if you can replicate the following plots. To see the answers, click the "Input" button to unhide the code and see the answers. Here's the dataset we've been working with:

# In[14]:


pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon.head()


# And now, the plots:

# In[15]:


sns.countplot(pokemon['Generation'])


# In[16]:


sns.distplot(pokemon['HP'])


# In[17]:


sns.jointplot(x='Attack', y='Defense', data=pokemon)


# In[18]:


sns.jointplot(x='Attack', y='Defense', data=pokemon, kind='hex')


# Hint: for the next exercise, use the variables `HP` and `Attack`.

# In[22]:


sns.kdeplot(pokemon['HP'], pokemon['Attack'])


# In[20]:


sns.boxplot(x='Legendary', y='Attack', data=pokemon)


# In[21]:


sns.violinplot(x='Legendary', y='Attack', data=pokemon)


# ## Conclusion
# 
# `seaborn` is one of the most important, if not *the* most important, data visualization tool in the Python data viz ecosystem. In this notebook we looked at what features and capacities `seaborn` brings to the table. There's plenty more that you can do with the library that we won't cover here or elsewhere in the tutorial; I highly recommend browsing the terrific `seaborn` [Gallery page](https://seaborn.pydata.org/examples/index.html) to see more beautiful examples of the library in action.
# 
# [Click here to go to the next section, "Faceting with seaborn"](https://www.kaggle.com/residentmario/faceting-with-seaborn).
