
# coding: utf-8

# # Faceting with seaborn
# 
# <table>
# <tr>
# <td><img src="https://i.imgur.com/wU9M9gu.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/85d2nIj.png" width="350px"/></td>
# </tr>
# <tr>
# <td style="font-weight:bold; font-size:16px;">Facet Grid</td>
# <td style="font-weight:bold; font-size:16px;">Pair Plot</td>
# </tr>
# <tr>
# <td>sns.FacetGrid()</td>
# <td>sns.pairplot()</td>
# </tr>
# <tr>
# <td>Good for data with at least two categorical variables.</td>
# <td>Good for exploring most kinds of data.</td>
# </tr>
# </table>
# 
# So far in this tutorial we've been plotting data in one (univariate) or two (bivariate) dimensions, and we've learned how plotting in `seaborn` works. In this section we'll dive deeper into `seaborn` by exploring **faceting**.
# 
# Faceting is the act of breaking data variables up across multiple subplots, and combining those subplots into a single figure. So instead of one bar chart, we might have, say, four, arranged together in a grid.
# 
# In this notebook we'll put this technique in action, and see why it's so useful.

# In[ ]:


import pandas as pd
pd.set_option('max_columns', None)
df = pd.read_csv("../input/fifa-18-demo-player-dataset/CompleteDataset.csv", index_col=0)

import re
import numpy as np

footballers = df.copy()
footballers['Unit'] = df['Value'].str[-1]
footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0, 
                                    footballers['Value'].str[1:-1].replace(r'[a-zA-Z]',''))
footballers['Value (M)'] = footballers['Value (M)'].astype(float)
footballers['Value (M)'] = np.where(footballers['Unit'] == 'M', 
                                    footballers['Value (M)'], 
                                    footballers['Value (M)']/1000)
footballers = footballers.assign(Value=footballers['Value (M)'],
                                 Position=footballers['Preferred Positions'].str.split().str[0])


# (Note: the first code cell above contains some data pre-processing. This is extraneous, and so I've hidden it by default.)

# In[ ]:


footballers.head()


# In[ ]:


import seaborn as sns


# ## The FacetGrid
# 
# The core `seaborn` utility for faceting is the `FacetGrid`. A `FacetGrid` is an object which stores some information on how you want to break up your data visualization.
# 
# For example, suppose that we're interested in (as in the previous notebook) comparing strikers and goalkeepers in some way. To do this, we can create a `FacetGrid` with our data, telling it that we want to break the `Position` variable down by `col` (column).
# 
# Since we're zeroing in on just two positions in particular, this results in a pair of grids ready for us to "do" something with them:

# In[ ]:


df = footballers[footballers['Position'].isin(['ST', 'GK'])]
g = sns.FacetGrid(df, col="Position")


# From there, we use the `map` object method to plot the data into the laid-out grid.

# In[ ]:


df = footballers[footballers['Position'].isin(['ST', 'GK'])]
g = sns.FacetGrid(df, col="Position")
g.map(sns.kdeplot, "Overall")


# Passing a method into another method like this may take some getting used to, if this is your first time seeing this being done. But once you get used to it, `FacetGrid` is very easy to use.
# 
# By using an object to gather "design criteria", `seaborn` does an effective job seamlessly marrying the data *representation* to the data *values*, sparing us the need to lay the plot out ourselves.
# 
# We're probably interested in more than just goalkeepers and strikers, however. But if we squeezed all of the possible game positions into one row, the resulting plots would be tiny. `FacetGrid` comes equipped with a `col_wrap` parameter for dealing with this case exactly.

# In[ ]:


df = footballers

g = sns.FacetGrid(df, col="Position", col_wrap=6)
g.map(sns.kdeplot, "Overall")


# So far we've been dealing exclusively with one `col` (column) of data. The "grid" in `FacetGrid`, however, refers to the ability to lay data out by row *and* column.
# 
# For example, suppose we're interested in comparing the talent distribution for (goalkeepers and strikers specifically, to keep things succinct) across rival clubs Real Madrid, Atlético Madrid, and FC Barcelona.
# 
# As the plot below demonstrates, we can achieve this by passing `row=Position` and `col=Club` parameters into the plot.

# In[ ]:


df = footballers[footballers['Position'].isin(['ST', 'GK'])]
df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona', 'Atlético Madrid'])]

g = sns.FacetGrid(df, row="Position", col="Club")
g.map(sns.violinplot, "Overall")


# `FacetGrid` orders the subplots effectively arbitrarily by default. To specify your own ordering explicitly, pass the appropriate argument to the `row_order` and `col_order` parameters.

# In[ ]:


df = footballers[footballers['Position'].isin(['ST', 'GK'])]
df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona', 'Atlético Madrid'])]

g = sns.FacetGrid(df, row="Position", col="Club", 
                  row_order=['GK', 'ST'],
                  col_order=['Atlético Madrid', 'FC Barcelona', 'Real Madrid CF'])
g.map(sns.violinplot, "Overall")


# `FacetGrid` comes equipped with various lesser parameters as well, but these are the most important ones.

# ## Why facet?
# 
# In a nutshell, faceting is the easiest way to make your data visualization multivariate.
# 
# Faceting is multivariate because after laying out one (categorical) variable in the rows and another (categorical) variable in the columns, we are already at two variables accounted for before regular plotting has even begun.
# 
# And faceting is easy because transitioning from plotting a `kdeplot` to gridding them out, as here, is very simple. It doesn't require learning any new visualization techniques. The limitations are the same ones that held for the plots you use inside.
# 
# Faceting does have some important limitations however. It can only be used to break data out across singular or paired categorical variables with very low numeracy&mdash;any more than five or so dimensions in the grid, and the plots become too small (or involve a lot of scrolling). Additionally it involves choosing (or letting Python) an order to plot in, but with nominal categorical variables that choice is distractingly arbitrary.
# 
# Nevertheless, faceting is an extremely useful and applicable tool to have in your toolbox.

# ## Pairplot
# 
# Now that we understand faceting, it's worth taking a quick once-over of the `seaborn` `pairplot` function.
# 
# `pairplot` is a very useful and widely used `seaborn` method for faceting *variables* (as opposed to *variable values*). You pass it a `pandas` `DataFrame` in the right shape, and it returns you a gridded result of your variable values:

# In[ ]:


sns.pairplot(footballers[['Overall', 'Potential', 'Value']])


# By default `pairplot` will return scatter plots in the main entries and a histogram in the diagonal. `pairplot` is oftentimes the first thing that a data scientist will throw at their data, and it works fantastically well in that capacity, even if sometimes the scatter-and-histogram approach isn't quite appropriate, given the data types.

# # Examples
# 
# As in previous notebooks, let's now test ourselves by answering some questions about the plots we've used in this section. Once you have your answers, click on "Output" button below to show the correct answers.
# 
# 1. Suppose that we create an `n` by `n` `FacetGrid`. How big can `n` get?
# 2. What are the two things about faceting which make it appealing?
# 3. When is `pairplot` most useful?

# In[ ]:


from IPython.display import HTML
HTML("""
<ol>
<li>You should try to keep your grid variables down to five or so. Otherwise the plots get too small.</li>
<li>It's (1) a multivariate technique which (2) is very easy to use.</li>
<li>Pair plots are most useful when just starting out with a dataset, because they help contextualize relationships within it.</li>
</ol>
""")


# Next, try forking this kernel, and see if you can replicate the following plots. To see the answers, click the "Input" button to unhide the code and see the answers. Here's the dataset we've been working with:

# In[ ]:


import pandas as pd
import seaborn as sns

pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon.head(3)


# In[ ]:


g = sns.FacetGrid(pokemon, row="Legendary")
g.map(sns.kdeplot, "Attack")


# In[ ]:


g = sns.FacetGrid(pokemon, col="Legendary", row="Generation")
g.map(sns.kdeplot, "Attack")


# In[ ]:


sns.pairplot(pokemon[['HP', 'Attack', 'Defense']])


# ## Conclusion
# 
# In this notebook we explored `FacetGrid` and `pairplot`, two `seaborn` facilities for faceting your data, and discussed why faceting is so useful in a broad range of cases.
# 
# This technique is our first dip into multivariate plotting, an idea that we will explore in more depth with two other approaches in the next section.
# 
# [Click here to go to the next section, "Multivariate plotting"](https://www.kaggle.com/residentmario/multivariate-plotting).
