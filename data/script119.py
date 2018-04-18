
# coding: utf-8

# # **Hey Kagglers, this is meant to be a fun little visualization tutorial using the [Seaborn](https://stanford.edu/~mwaskom/software/seaborn/index.html) library and [Alberto Barradas'](https://www.kaggle.com/abcsds) [Pokémon dataset](https://www.kaggle.com/abcsds/pokemon).**  
# # Whether you're following along or just skimming through, thanks for checking it out!

# ----------

# # Notebook Prep

# First, let's import the packages we'll be using in this kernel. 

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ----------

# # Data Import

# Now, let's [read](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) in the data with Pandas.  
# If you're working in something other than a Kaggle notebook, be sure to change the file location.

# In[ ]:


pkmn = pd.read_csv('../input/Pokemon.csv')


# Using the [head](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html) method, let's take a peak at the data.

# In[ ]:


pkmn.head()


# We've got a pretty simple format here! There's the Pokémon number, name, their type(s), their different stat values, and a convenient Total variable.

# ## Update Aug 30 2016:  
# I just realized that Generation and Legendary variables were added to the dataset.  
# I'm going to add a step here to drop the variables so that the rest of the code works as it did originally.  
# Apologies to anyone who forked the notebook and had trouble following along!

# In[ ]:


pkmn = pkmn.drop(['Generation', 'Legendary'],1)


# ----------

# # Plots with Seaborn

# To start things off, let's just make a [scatterplot](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.jointplot.html) based on two variables from the data set.  
# I'll use HP and Attack in this example, but feel free to do something different!

# In[ ]:


sns.jointplot(x="HP", y="Attack", data=pkmn);


# Nothing *too* informative here, but we can definitely see why the Seaborn library is so popular. With one short line of code, we get this really nice looking plot!

# Now let's see if we can make something a little bit prettier. How about a distribution of all six stats? We could even group it further using Pokémon type!  
# This might seem a little ambitious, but let's take it one step at a time.

# For starters, let's see if we can make a basic [box and whisker plot](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.boxplot.html) of a single variable.

# In[ ]:


sns.boxplot(y="HP", data=pkmn);


# Cool! Not too hard.  
# Now let's see if we can get all of the stats in there.

# As it turns out, if you don't specify an x or y argument, Seaborn will give you a plot for each numeric variable. Handy!

# In[ ]:


sns.boxplot(data=pkmn);


# Since the # variable doesn't make sense here, let's drop it from the table.  
# Total can be dropped as well, since we didn't originally want to include it and it's on a much larger scale.

# In[ ]:


pkmn = pkmn.drop(['Total', '#'],1)


# In[ ]:


sns.boxplot(data=pkmn);


# Alright, now all that's left is to include Pokémon type in this visualization.  
# One way to do this would be switch the graph to a [swarmplot](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.swarmplot.html) and color code the points by type.

# Trying to use the swarmplot function with the "hue" argument is going to give us some errors if we don't transform our data a bit though. The Seaborn website provides an example using Pandas' [melt](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.melt.html) function, so we'll give that a try!

# In[ ]:


pkmn = pd.melt(pkmn, id_vars=["Name", "Type 1", "Type 2"], var_name="Stat")


# So now our plot looks like this:

# In[ ]:


pkmn.head()


# The head method doesn't really do this transformation justice, but our dataset now has 4800 rows up from 800!  
# So let's go ahead and run this plot function!

# In[ ]:


sns.swarmplot(x="Stat", y="value", data=pkmn, hue="Type 1");


# Oh geez. That's uh... something.  
# I think we've got some cleaning up to do.

# Using a few Seaborn and Matplotlib functions, we can adjust how our plot looks.  
# On each line below, we will:   
# - [Make the plot larger](http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure)  
# - [Adjust the y-axis](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.ylim)  
# - Organize the point distribution by type  and make the individual points larger  
# - [Move the legend out of the way](http://matplotlib.org/users/legend_guide.html#legend-location)

# In[ ]:


plt.figure(figsize=(12,10))
plt.ylim(0, 275)
sns.swarmplot(x="Stat", y="value", data=pkmn, hue="Type 1", split=True, size=7)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);


# Alright! This is looking better!  
# For our final touch, we'll change the background to white and create a custom color palette that corresponds to each Pokémon type.  
# We'll use the Seaborn [color_palette](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.color_palette.html) function and a [with](https://www.python.org/dev/peps/pep-0343/) statement to accomplish this.

# In[ ]:


sns.set_style("whitegrid")
with sns.color_palette([
    "#8ED752", "#F95643", "#53AFFE", "#C3D221", "#BBBDAF",
    "#AD5CA2", "#F8E64E", "#F0CA42", "#F9AEFE", "#A35449",
    "#FB61B4", "#CDBD72", "#7673DA", "#66EBFF", "#8B76FF",
    "#8E6856", "#C3C1D7", "#75A4F9"], n_colors=18, desat=.9):
    plt.figure(figsize=(12,10))
    plt.ylim(0, 275)
    sns.swarmplot(x="Stat", y="value", data=pkmn, hue="Type 1", split=True, size=7)
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);


# Now things are looking pretty good!

# So that's the end of the tutorial for now, but feel free to keep going on your own.  
# You can try using a smaller sample of Pokémon types, find a way to incorporate the Type 2 variable somehow, or make a different kind of plot entirely!  
# If you find anything cool, let me know! I'd love to see what everyone else comes up with!  
# Thanks again for reading!
