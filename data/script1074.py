
# coding: utf-8

# # Styling your plots
# 
# ## Introduction
# 
# Whenever exposing your work to an external audience (like, say, the Kaggle userbase), styling your work is a must. The defaults in `pandas` (and other tools) are rarely exactly right for the message you want to communicate. Tweaking your plot can greatly enhance the communicative power of your visualizations, helping to make your work more impactful.
# 
# In this section we'll learn how to style the visualizations we've been creating. Because there are *so many* things you can tweak in your plot, it's impossible to cover everything, so we won't try to be comprehensive here. Instead this section will cover some of the most useful basics: changing figure sizes, colors, and font sizes; adding titles; and removing axis borders.
# 
# An important skill in plot styling is knowing how to look things up. Comments like "I have been using Matplotlib for a decade now, and I still have to look most things up" are [all too common](https://youtu.be/aRxahWy-ul8?t=2m42s). If you're styling a `seaborn` plot, the library's [gallery](http://seaborn.pydata.org/examples/) and [API documentation](https://seaborn.pydata.org/api.html) are a great place to find styling options. And for both `seaborn` and `pandas` there is a wealth of information that you can find by looking up "how to do X with Y" on [StackOverflow](https://stackoverflow.com/) (replacing X with what you want to do, and Y with `pandas` or `seaborn`). If you want to change your plot in some way not covered in this brief tutorial, and don't already know what function you need to do it, searching like this is the most efficient way of finding it.

# In[ ]:


import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head(3)


# ## Points on style
# 
# Recall our bar plot from earlier:

# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar()


# Throughout this section we're going to work on making this plot look nicer.
# 
# This plot is kind of hard to see. So make it bigger! We can use the `figsize` parameter to do that.

# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar(figsize=(12, 6))


# `figsize` controls the size of the image, in inches. It expects a tuple of `(width, height)` values.
# 
# Next, we can change the color of the bars to be more thematic, using the `color` parameter.

# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12, 6),
    color='mediumvioletred'
)


# The text labels are very hard to read at this size. They fit the plot when our plot was very small, but now that the plot is much bigger we need much bigger labels. We can used `fontsize` to adjust this.

# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16
)


# We also need a `title`.

# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16,
    title='Rankings Given by Wine Magazine',
)


# However, this title is too small. Unfortunately, `pandas` doesn't give us an easy way of adjusting the title size.
# 
# Under the hood, `pandas` data visualization tools are built on top of another, lower-level graphics library called `matplotlib`. Anything that you build in `pandas` can be built using `matplotlib` directly. `pandas` merely make it easier to get that work done.
# 
# `matplotlib` *does* provide a way of adjusting the title size. Let's go ahead and do it that way, and see what's different:

# In[ ]:


import matplotlib.pyplot as plt

ax = reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16
)
ax.set_title("Rankings Given by Wine Magazine", fontsize=20)


# In the cell immediately above, all we've done is grabbed that object, assigned it to the variable `ax`, and then called `set_title` on `ax`. The `ax.set_title` method makes it easy to change the fontsize; the `title=` keyword parameter in the `pandas` library does not.
# 
# `seaborn`, covered in a separate section of the tutorial, *also* uses `matplotlib` under the hood. This means that the tricks above work there too. `seaborn` has its own tricks, too&mdash;for example, we can use the very convenient `sns.despine` method to turn off the ugly black border.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

ax = reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12, 6),
    color='mediumvioletred',
    fontsize=16
)
ax.set_title("Rankings Given by Wine Magazine", fontsize=20)
sns.despine(bottom=True, left=True)


# Prefect. This graph is more clearer than what we started with; it will do a much better job communicating the analysis to our readers.
# 
# There are many, many more things that you can do than just what we've shown here. Different plots provide different styling options: `color` is almost universal for example, while `s` (size) only makes sense in a scatterplot. For now, the operations we've shown here are enough to get you started.

# # Exercises
# 
# To put your design skills to the test, try forking this notebook and replicating the plots that follow. To see the answers, hit the "Input" button below to un-hide the code.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
pokemon.head(3)


# In[ ]:


pokemon.plot.scatter(x='Attack', y='Defense',
                     figsize=(12, 6),
                     title='Pokemon by Attack and Defense')


# In[ ]:


ax = pokemon['Total'].plot.hist(
    figsize=(12, 6),
    fontsize=14,
    bins=50,
    color='gray'
)
ax.set_title('Pokemon by Stat Total', fontsize=20)


# In[ ]:


ax = pokemon['Type 1'].value_counts().plot.bar(
    figsize=(12, 6),
    fontsize=14
)
ax.set_title("Pokemon by Primary Type", fontsize=20)
sns.despine(bottom=True, left=True)


# # Conclusion
# 
# In this section of the tutorial, we learned a few simple tricks for making our plots more visually appealing, and hence, more communicative. We also learned that there is another plotting library, `matplotlib`, which lies "underneath" the `pandas` data visualization tools, and which we can use to more finely manipulate our plots.
# 
# In the next section we will learn to compose plots together using a technique called subplotting.
# 
# [Click here to go to the next section, "Subplots"](https://www.kaggle.com/residentmario/subplots).
