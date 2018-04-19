
# coding: utf-8

# <table>
# <tr>
# <td><img src="https://i.imgur.com/BqJgyzB.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/ttYzMwD.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/WLmzj41.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/LjRTbCn.png" width="350px"/></td>
# </tr>
# <tr>
# <td style="font-weight:bold; font-size:16px;">Scatter Plot</td>
# <td style="font-weight:bold; font-size:16px;">Choropleth</td>
# <td style="font-weight:bold; font-size:16px;">Heatmap</td>
# <td style="font-weight:bold; font-size:16px;">Surface Plot</td>
# </tr>
# <tr>
# <td>go.Scatter()</td>
# <td>go.Choropleth()</td>
# <td>go.Heatmap()</td>
# <td>go.Surface()</td>
# </tr>
# <!--
# <tr>
# <td>Good for interval and some nominal categorical data.</td>
# <td>Good for interval and some nominal categorical data.</td>
# <td>Good for nominal and ordinal categorical data.</td>
# <td>Good for ordinal categorical and interval data.</td>
# </tr>
# -->
# </table>
# 
# # Introduction to plotly
# 
# So far in this tutorial we have been using `seaborn` and `pandas`, two mature libraries designed around `matplotlib`. These libraries all focus on building "static" visualizations: visualizations that have no moving parts. In other words, all of the plots we've built thus far could appear in a dead-tree journal article.
# 
# The web unlocks a lot of possibilities when it comes to interactivity and animations. There are a number of plotting libraries available which try to provide these features.
# 
# In this section we will examine `plotly`, an open-source plotting library that's one of the most popular of these libraries.

# In[ ]:


import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()


# `plotly` provides both online and offline modes. The latter injects the `plotly` source code directly into the notebook; the former does not. I recommend using `plotly` in offline mode the vast majority of the time, and it's the only mode that works on Kaggle (which disables network access in Python).

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


# We'll start by creating a basic scatter plot.

# In[ ]:


import plotly.graph_objs as go

iplot([go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])


# This chart is fully interactive. We can use the toolbar on the top-right to perform various operations on the data: zooming and panning, for example. When we hover over a data point, we get a tooltip. We can even save the plot as a PNG image.
# 
# This chart also demonstrates the *disadvantage* of this fancier plotting library. In order to keep performance reasonable, we had to limit ourselves to the first 1000 points in the dataset. While this was necessary anyway (to avoid too much overplotting) it's important to note that in general, interactive graphics are much, much more resource-intensive than static ones. It's easier to "max out" how many points of data you can show.
# 
# Notice the "shape" of the `plotly` API. `iplot` takes a list of plot objects and composes them for you, plotting the combined end result. This makes it easy to stack plots.
# 
# As another example, here's a KDE plot (what `plotly` refers to as a `Histogram2dContour`) *and* scatter plot of the same data.

# In[ ]:


iplot([go.Histogram2dContour(x=reviews.head(500)['points'], 
                             y=reviews.head(500)['price'], 
                             contours=go.Contours(coloring='heatmap')),
       go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])


# `plotly` exposes several different APIs, ranging in complexity from low-level to high-level. `iplot` is the highest-level API, and hence, the most convenient one for general-purpose use.
# 
# Personally I've always found the `plotly` `Surface` its most impressive feature (albeit one of the hardest to get right):

# In[ ]:


df = reviews.assign(n=0).groupby(['points', 'price'])['n'].count().reset_index()
df = df[df["price"] < 100]
v = df.pivot(index='price', columns='points', values='n').fillna(0).values.tolist()


# In[ ]:


iplot([go.Surface(z=v)])


# On Kaggle, `plotly` is often used to make **choropleths**. Choropleths are a type of map, where all of the entities (countries, US states, etc.) are colored according to some variable in the dataset. `plotly` is one of the most convenient plotting libraries available for making them.

# In[ ]:


df = reviews['country'].replace("US", "United States").value_counts()

iplot([go.Choropleth(
    locationmode='country names',
    locations=df.index.values,
    text=df.index,
    z=df.values
)])


# Overall, `plotly` is a powerful, richly interactive data visualization library. It allows us to generate plots with more "pizazz" than standard `pandas` or `seaborn` output.
# 
# The tradeoff is that while `pandas` and `seaborn` are well-established, `plotly` is still new. As a result, and in particular, `plotly` documentation is much harder to and find and interpret; the office documentation on the `plotly` website uses a mix of different features for plotting, which makes it harder to use than it has to be.
# 
# Additionally, it's important to recognize when interactivity is useful, and when it is not. The most effective plots do not need to use hovers or tooltips to get their work done. As a result `plotly`, though extremely attractive, is rarely more *useful* than an equivalent plot in `pandas` or `seaborn`.

# # Exercises
# 
# For the following exercise, try forking and running this notebook, and then reproducing the chart that follows. Hint: `Attack` on the x-axis, `Defense` on the y-axis.

# In[ ]:


import pandas as pd
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
pokemon.head(3)


# In[ ]:


import plotly.graph_objs as go

iplot([go.Scatter(x=pokemon['Attack'], y=pokemon['Defense'], mode='markers')])


# # Conclusion
# 
# In this section we looked at `plotly`, an interactive plotting library that produces very attractive-looking charts. It is one of a number of alternatives to `matplotlib`-based tools that provide first-class interactivity (`bokeh` is another one worth mentioning).
# 
# In the next section we'll look at another plotting library, `plotnine`, which is designed around a peculiar but powerful idea called the **grammar of graphics**.
# 
# [Click here to go the next section, "Grammar of graphics with plotnine"](https://www.kaggle.com/residentmario/grammer-of-graphics-with-plotnine-optional).
