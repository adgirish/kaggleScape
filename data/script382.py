
# coding: utf-8

# In data science, we are regularly challenged with the task of extracting meaning from complex and high dimensional datasets.  The rise of sophisticated machine learning and other analytic techniques has undoubtedly revolutionized how we view data, and the kinds and amount of information we can wring out of it. Modern computers have greatly surpassed our own abilities in number-crunching speed and in handling massive quantities of data. You might think that the story ends here, and we should just surrender to the machine.
# 
# However, there is still a place for us lowly humans: Our visual brains are phenomenal at detecting complex structure and discovering patterns in massive quantities of data.  In fact, that is just what our visual systems are designed to do.  Every second that your eyes are open, data (in the form of light on your retina) is pouring into visual areas of your brain. And yet remarkably, you have no problem at all recognizing a neat looking shell on a beach, or your mom's face in a large crowd.  Our brains are 'unsupervised-pattern-discovery-aficionados', and we can harness that power to make some fascinating discoveries about the nature of our world.
# 
# But hang on now, not so fast: there is one MAJOR drawback to our visual systems: we are essentially capped at perceiving in 3 dimensions (or 4 including changes over time), and most datasets today have much higher dimensionality.  
# 
# So, the question of the hour is: **how can we harness the incredible pattern-recognition powers of our brains to visualize complex and high dimensional datasets?**
# 
# In comes *dimensionality reduction*, stage right. Dimensionality reduction is just what it sounds like - reducing a high dimensional dataset to a lower dimensional space.  For example, say you have a dataset of mushrooms, where each row is comprised of a bunch of features of the mushroom, like cap size, cap shape, cap color, odor etc.  The simplest way to do dimensionality reduction is to just remove some of the features. However, this is problematic if the features you drop contain valuable diagnostic information about that mushroom. A slightly more sophisticated approach is to reduce the dimensionality of the dataset by only considering its **principal components**, or the combination of features that explains the most variance in the dataset. Using a technique called *principal components analysis* (or PCA), we can reduced the dimensionality of a dataset, while preserving as much of its precious variance as possible.
# 
# So, we are now equipped with the computational prowess to forage for mushrooms using HyperTools, a python toolbox for visualizing and manipulating high-dimensional data, which is built on top of familiar friends like matplotlib, scikit-learn and seaborn. For more about the open-source toolbox, you can visit the [code](https://github.com/ContextLab/hypertools) or read our [paper](https://arxiv.org/abs/1701.08290).

# ### Import libraries we need

# In[ ]:


import pandas as pd
import hypertools as hyp 
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read in the data with pandas

# In[ ]:


data = pd.read_csv('../input/mushrooms.csv')
data.head()


# ### pop off column indicating poisonous/non-poisonous

# In[ ]:


class_labels = data.pop('class')


# ### Now let's plot the high-dimensional data in a low dimensional space by converting it to a numpy array and passing it to hypertools. To handle text columns, hypertools will first convert each text column into a series of binary 'dummy' variables before performing the dimensionality reduction.  For example, if the 'cap size' column contained 'big' and 'small' labels, this single column would be turned into two binary columns: one for 'big' and one for 'small', where 1s represents the presence of that feature and 0s represents the absence. For more, see the documentation for pandas [get_dummies](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html) function.

# In[ ]:


fig, ax, data, _ = hyp.plot(data,'.') # if the number of features is greater than 3, the default is to plot in 3d


# ### From the plots above, it's clear that there are multiple clusters in this data. In addition to descriptive features, this dataset also contains whether or not the mushroom is poisonous.  Do these clusters have bearing on whether or not the mushroom is poisonous?  Let's try to understand this by coloring the points based on whether the mushroom is poisonous:

# In[ ]:


fig, ax, data, _ = hyp.plot(data,'.', group=class_labels, legend=list(set(class_labels)))


# ###  From the plot above it is clear that the clustering of the mushrooms carries information about whether or not they are poisonous (red refers to non-poisonous and blue is poisonous). It also looks like there are a number of distinct clusters that are poisonous/non-poisonous.
# 
# ### Let's use the 'cluster' feature of hypertools to discover these clusters using a k-means fitting procedure:

# In[ ]:


fig, ax, data, _ = hyp.plot(data, '.', n_clusters=23)

# you can also recover the cluster labels using the cluster tool
cluster_labels = hyp.tools.cluster(data, n_clusters=23, ndims=3) 
# hyp.plot(data, 'o', point_colors=cluster_labels, ndims=2)


# ### Sidenote: we can change the color palette using the palette argument.  Hypertools supports matplotlib and seaborn color palettes.

# In[ ]:


fig, ax, data, _ = hyp.plot(data,'.', group=cluster_labels, palette="deep")


# ### Hypertools uses PCA to reduce the dimensionality by default, but there are other ways to do dimensionality reduction.  Let's try reducing with various techniques, but keeping the cluster labels the same.

# ## Independent Components Analysis

# In[ ]:


fig, ax, data, _ = hyp.plot(data,'.', model='FastICA', group=class_labels, legend=list(set(class_labels)))


# ## t-SNE

# In[ ]:


fig, ax, data, _ = hyp.plot(data, '.', model='TSNE', group=class_labels, legend=list(set(class_labels)))


# This brief demo highlights how HyperTools can be used to explore high-dimensional data in only a few lines of code.  By combining and extending powerful open-source toolboxes such as matplotlib, scikit-learn and seaborn, this toolbox can help you to discover structure in complex, high-dimensional datasets.  This knowledge could then be used to guide future analysis decisions, such as choosing the right classification model for your dataset.  We think this approach will help data scientists to discover that you can learn a lot about a dataset by simply looking at it before you try a million different complicated machine learning techniques. To learn more about the package, visit our [readthedocs](http://readthedocs.org/projects/hypertools/), [code](https://github.com/ContextLab/hypertools), [binder of demos](http://mybinder.org/repo/contextlab/hypertools-paper-notebooks) and [paper](https://arxiv.org/abs/1701.08290).
