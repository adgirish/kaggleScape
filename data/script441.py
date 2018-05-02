
# coding: utf-8

# # Dimensionality Reduction and PCA for Fashion MNIST
# 
# Principal Components Analysis is the simplest example of dimensionality reduction. **Dimensionality reduction** is a the problem of taking a matrix with many observations, and "compressing it" to a matrix with fewer observations which preserves as much of the information in the full matrix as possible.
# 
# Principal components is the most straightforward of the methodologies for doing so. It relies on finding an orthonormal basis (a set of perpendicular vectors) within the dimensional space of the dataset which explain the largest possible amount of variance in the dataset. For example, here is PCA applied to a small two-dimensional problem:
# 
# ![](https://i.imgur.com/tlZIO7t.png)
# 
# PCA then remaps the values of the points in the dataset to their projection onto the newly minted bases, spitting out a matrix of observations with as few variables as you desire!
# 
# ## Basic application
# 
# The fashion MNIST dataset is a fun dataset to try this algorithm out on because it includes 28x28 variables in an easy-to-visualize form (a picture). What happens when we pick just two principal components, and try to map the result?

# In[1]:


import pandas as pd
df = pd.read_csv("../input/fashion-mnist_test.csv")
X = df.iloc[:, 1:]
y = df.iloc[:, :1]


# In[2]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)


# To visualize the result of applying PCA to our datasets, we'll plot how much weight each pixel in the clothing picture gets in the resulting vector, using a heatmap. This creates a nice, potentially interpretable picture of what each vector is "finding".

# In[3]:


pca.explained_variance_ratio_


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

fig, axarr = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(pca.components_[0, :].reshape(28, 28), ax=axarr[0], cmap='gray_r')
sns.heatmap(pca.components_[1, :].reshape(28, 28), ax=axarr[1], cmap='gray_r')
axarr[0].set_title(
    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[0]*100),
    fontsize=12
)
axarr[1].set_title(
    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[1]*100),
    fontsize=12
)
axarr[0].set_aspect('equal')
axarr[1].set_aspect('equal')

plt.suptitle('2-Component PCA')


# The first component looks like...some kind of large clothing object (e.g. not a shoe or accessor). The second component looks like negative space around a pair of pants.
# 
# What do the next two components look like?

# In[5]:


pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

fig, axarr = plt.subplots(2, 2, figsize=(12, 8))

sns.heatmap(pca.components_[0, :].reshape(28, 28), ax=axarr[0][0], cmap='gray_r')
sns.heatmap(pca.components_[1, :].reshape(28, 28), ax=axarr[0][1], cmap='gray_r')
sns.heatmap(pca.components_[2, :].reshape(28, 28), ax=axarr[1][0], cmap='gray_r')
sns.heatmap(pca.components_[3, :].reshape(28, 28), ax=axarr[1][1], cmap='gray_r')

axarr[0][0].set_title(
    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[0]*100),
    fontsize=12
)
axarr[0][1].set_title(
    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[1]*100),
    fontsize=12
)
axarr[1][0].set_title(
    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[2]*100),
    fontsize=12
)
axarr[1][1].set_title(
    "{0:.2f}% Explained Variance".format(pca.explained_variance_ratio_[3]*100),
    fontsize=12
)
axarr[0][0].set_aspect('equal')
axarr[0][1].set_aspect('equal')
axarr[1][0].set_aspect('equal')
axarr[1][1].set_aspect('equal')

plt.suptitle('4-Component PCA')
pass


# Both of these components look like they have some sort of shoe-related thing going on.
# 
# Each additional dimension we add to the PCA captures less and less of the variance in the model. The first component is the most important one, followed by the second, then the third, and so on.
# 
# When using PCA, it's a good idea to pick a decomposition with a reasonably small number of variables by looking for a "cutoff" in the effectiveness of the model. To do that, you can plot the explained variance ratio (out of all variance) for each of the components of the PCA. This goes thusly:

# In[6]:


import numpy as np

pca = PCA(n_components=10)
X_r = pca.fit(X).transform(X)

plt.plot(range(10), pca.explained_variance_ratio_)
plt.plot(range(10), np.cumsum(pca.explained_variance_ratio_))
plt.title("Component-wise and Cumulative Explained Variance")
pass


# Looking at this plot, it seems like the first and second components add a lot of information about the variables to the model, but every component after that is not particularly worthwhile. If we were using PCA as a processing step ourselves, we might want to cut off at $n=2$.

# ## Digging deeper
# 
# The understanding above (specifically, explained variance and understanding how to find and interpret which coefficients go into each dimension-reduced component) is sufficient for using PCA for plug-and-play modeling purposes. However, it's useful to have a "toolbox" for comprehending dimensionality reduction output.
# 
# Dimensionality reduction techniques (and more specifically PCA in this example notebook) can be used to understand clusters of variables which co-occur with one another. We can think of each new vector across the dataset values as being some new composite variable that the dimensionality reduction technique has come up with. For example, suppose we find that the third PCA vector is a good indicator of the "shoe-ness" of an item in the Fashion MNIST dataset. We could compute this vector, grab it, and push it back into our original matrix of features as a new column. If we were previously having trouble distinguishing shoes in the dataset, this might significantly increase the accuracy of our model!
# 
# Even if none of the variables in the dataset are worth pushing back to our original feature matrix, PCA will still help us probe for new features. If dimensionality reduction picks up shoe-ness in some sense, then that's a good indication that trying to define "shoe-ness" in the data is a good idea to try to engineer as a feature.
# 
# The next section applies a battery of techniques of this form. It is based on Selfish Gene's excellent notebooks elsewhere.
# 
# For the diagnoses that follow, we will use a standardly normalized 120-component PCA.

# In[16]:


from sklearn.preprocessing import normalize 
X_norm = normalize(X.values)
# X_norm = X.values / 255

from sklearn.decomposition import PCA

pca = PCA(n_components=120)
X_norm_r = pca.fit(X_norm).transform(X_norm)


# ### Mean and standard deviation summaries
# 
# To dig deeper into PCA, let's start by looking at the mean clothing image as as well as a heatmap of the clothing image standard deviations.

# In[8]:


sns.heatmap(pd.DataFrame(X_norm).mean().values.reshape(28, 28), cmap='gray_r')


# In[9]:


sns.heatmap(pd.DataFrame(X).std().values.reshape(28, 28), cmap='gray_r')


# Pixels on the edge of the image don't matter. The pixels with the least variance and highest mean value are shoe pixels from the center of the image, while pixels elsewhere in the "sweater outline" are the highest variance. This hints that shoes are relatively easily differenciable as a class of items, but the rest of the types of items in the dataset will be much harder.

# ### Assessing fit by reconstructing individual sample images
# 
# PCA compresses the 28x28 pixel values in the dataset to `n` vectors across those pixels. A way of diagnosing how well we did this is to compare original images with ones reconstructed from those vectors.

# In[33]:


def reconstruction(X, n, trans):
    """
    Creates a reconstruction of an input record, X, using the topmost (n) vectors from the
    given transformation (trans)
    
    Note 1: In this dataset each record is the set of pixels in the image (flattened to 
    one row).
    Note 2: X should be normalized before input.
    """
    vectors = [trans.components_[n] * X[n] for n in range(0, n)]
    
    # Invert the PCA transformation.
    ret = trans.inverse_transform(X)
    
    # This process results in non-normal noise on the margins of the data.
    # We clip the results to fit in the [0, 1] interval.
    ret[ret < 0] = 0
    ret[ret > 1] = 1
    return ret


# For example, here is how well a 120-variable reconstruction (15% as many variables as in the root dataset) does for the first image in the dataset, a T-shirt:

# In[34]:


fig, axarr = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(X_norm[0, :].reshape(28, 28), cmap='gray_r',
            ax=axarr[0])
sns.heatmap(reconstruction(X_norm_r[0, :], 120, pca).reshape(28, 28), cmap='gray_r',
            ax=axarr[1])
axarr[0].set_aspect('equal')
axarr[0].axis('off')
axarr[1].set_aspect('equal')
axarr[1].axis('off')


# In[41]:


def n_sample_reconstructions(X, n_samples=5, trans_n=120, trans=None):
    """
    Returns a tuple with `n_samples` reconstructions of records from the feature matrix X,
    as well as the indices sampled from X.
    """
    sample_indices = np.round(np.random.random(n_samples)*len(X))
    return (sample_indices, 
            np.vstack([reconstruction(X[int(ind)], trans_n, trans) for ind in sample_indices]))


def plot_reconstructions(X, n_samples=5, trans_n=120, trans=None):
    """
    Plots `n_samples` reconstructions.
    """
    fig, axarr = plt.subplots(n_samples, 3, figsize=(12, n_samples*4))
    ind, reconstructions = n_sample_reconstructions(X, n_samples, trans_n, trans)
    for (i, (ind, reconstruction)) in enumerate(zip(ind, reconstructions)):
        ax0, ax1, ax2 = axarr[i][0], axarr[i][1], axarr[i][2]
        sns.heatmap(X_norm[int(ind), :].reshape(28, 28), cmap='gray_r', ax=ax0)
        sns.heatmap(reconstruction.reshape(28, 28), cmap='gray_r', ax=ax1)
        sns.heatmap(np.abs(X_norm[int(ind), :] - reconstruction).reshape(28, 28), 
                    cmap='gray_r', ax=ax2)
        ax0.axis('off')
        ax0.set_aspect('equal')
        ax0.set_title("Original Image", fontsize=12)
        ax1.axis('off')
        ax1.set_aspect('equal')
        ax1.set_title("120-Vector Reconstruction", fontsize=12)
        ax2.axis('off')
        ax2.set_title("Original-Reconstruction Difference", fontsize=12)
        ax2.set_aspect('equal')


# In[42]:


plot_reconstructions(X_norm_r, n_samples=10, trans_n=120, trans=pca)


# A 120-vector reconstruction seems to do pretty well. It struggles the most when applied to images with holes in them. It has difficulty finding the edges of things. These problems are pretty typical of image datasets (many of which do not have the first problem, holes!).
# 
# By applying this sample reconstruction technique to more of the dataset, we can start to understand what features are and are not easily modeled. Similarly, by applying this technique whilst trying out different `n`, we can triangulate what aspects of the images are most-to-least easily modeled, by seeing how far up `n` has to go before we have "acceptably" captured a particular aspect of the dataset.
# 
# In this case, I suspect that finding the edges of clothing items that are not contiguous (e.g. the brace on a pair of heels, or the straps on a handbag) is the hardest part of this dataset to model. This presents the clue that we should worry about this aspect of the data quite a bit!

# ### Assessing how to interpret individual components using record similarity
# 
# When we want to understand the variables of a dataset at-a-glance, we can use the handy `pd.describe` function to do so. This function includes, among other things, the 0th, 25th, 50th, 75th, and 100th percentile of the dataset. Records (quantiles) very close to the edges of the usefulness of a variable are often very informative as to what that variable "does". This is true of ordinary variables, and all the more true of our computed ones, which we need all the help we can get interpreting!

# In[96]:


from sklearn.metrics import mean_squared_error

def quartile_record(X, vector, q=0.5):
    """
    Returns the data which is the q-quartile fit for the given vector.
    """
    errors = [mean_squared_error(X_norm[i, :], vector) for i in range(len(X_norm))]
    errors = pd.Series(errors)
    
    e_value = errors.quantile(q, interpolation='lower')
    return X[errors[errors == e_value].index[0], :]


# For example, here is the most-explained records in the dataset when it comes to the first principal component:

# In[106]:


sns.heatmap(quartile_record(X_norm, pca.components_[0], q=0.98).reshape(28, 28), 
            cmap='gray_r')


# Let's extend this idea to the first eight components in the dataset.

# In[128]:


def plot_quartiles(X, trans, n):

    fig, axarr = plt.subplots(n, 7, figsize=(12, n*2))
    for i in range(n):
        vector = trans.components_[i, :]
        sns.heatmap(quartile_record(X, vector, q=0.02).reshape(28, 28), 
            cmap='gray_r', ax=axarr[i][0], cbar=False)
        axarr[i][0].set_aspect('equal')
        axarr[i][0].axis('off')
        
        sns.heatmap(quartile_record(X, vector, q=0.1).reshape(28, 28), 
            cmap='gray_r', ax=axarr[i][1], cbar=False)
        axarr[i][1].set_aspect('equal')
        axarr[i][1].axis('off')
        
        sns.heatmap(quartile_record(X, vector, q=0.25).reshape(28, 28), 
            cmap='gray_r', ax=axarr[i][2], cbar=False)
        axarr[i][2].set_aspect('equal')
        axarr[i][2].axis('off')
        
        sns.heatmap(quartile_record(X, vector, q=0.5).reshape(28, 28), 
            cmap='gray_r', ax=axarr[i][3], cbar=False)
        axarr[i][3].set_aspect('equal')
        axarr[i][3].axis('off')

        sns.heatmap(quartile_record(X, vector, q=0.75).reshape(28, 28), 
            cmap='gray_r', ax=axarr[i][4], cbar=False)
        axarr[i][4].set_aspect('equal')
        axarr[i][4].axis('off')

        sns.heatmap(quartile_record(X, vector, q=0.9).reshape(28, 28), 
            cmap='gray_r', ax=axarr[i][5], cbar=False)
        axarr[i][5].set_aspect('equal')
        axarr[i][5].axis('off')        
        
        sns.heatmap(quartile_record(X, vector, q=0.98).reshape(28, 28), 
            cmap='gray_r', ax=axarr[i][6], cbar=False)        
        axarr[i][6].set_aspect('equal')
        axarr[i][6].axis('off')
        
    axarr[0][0].set_title('2nd Percentile', fontsize=12)
    axarr[0][1].set_title('10th Percentile', fontsize=12)
    axarr[0][2].set_title('25th Percentile', fontsize=12)
    axarr[0][3].set_title('50th Percentile', fontsize=12)
    axarr[0][4].set_title('75th Percentile', fontsize=12)
    axarr[0][5].set_title('90th Percentile', fontsize=12)
    axarr[0][6].set_title('98th Percentile', fontsize=12)


# In[130]:


plot_quartiles(X_norm, pca, 8)


# Each of these components does something interesting. The very first one seems to separate shoes from pants. The next one seperates uppers from pants. The third one, shoes from sandals. The fourth, items with huge sleeves from everything else. The eighth, bags from shoes; and so on!
# 
# Notice that *all of these vectors have meaning*, at least, more meaning than "pixel value 25, 25 is 27". Each component tells us the whatever-ness of something interesting. In this sense the PCA reconstruction is almost a more "natural" way of defining the same dataset; that is, it's a methodology that better fits the data itself, as opposed to structural constraints (like the need to describe individual pixel values) introduced by ease-of-use.
# 
# If you're doing supervised learning, you may want to look at this result and potentially pinch one similarity to or more of these componenets for use as a feature in your model. Otherwise, again, seeing where each "-ness" falls is useful for getting a deeper understanding of what you're working with.

# ### Assessing component fit using record similarity
# 
# Distinguishability between different dimensionality-determined attributes in the dataset is hard to establish emperically. One thing we can do to make it a little bit easier is plot out the errors (according to a metric of our choice; I'll keep using mean squared error) each component has relative to the data.

# In[133]:


def record_similarity(X, vector, metric=mean_squared_error):
    """
    Returns the record similarity to the vector, using MSE is the measurement metric.
    """
    return pd.Series([mean_squared_error(X_norm[i, :], vector) for i in range(len(X_norm))])


# In[149]:


fig, axarr = plt.subplots(2, 4, figsize=(12, 6))
axarr = np.array(axarr).flatten()

for i in range(0, 8):
    record_similarity(X_norm, pca.components_[i]).plot.hist(bins=50, ax=axarr[i])
    axarr[i].set_title("Component {0} Errors".format(i + 1), fontsize=12)
    axarr[i].set_xlabel("")
    axarr[i].set_ylabel("")


# In the histogram above, each element included in the bins is a measurement of the mean absolute error between a record in the dataset and a component (which are numerically labeled).
# 
# The worst-possible component would be one which has unfiromally distributed errors for all records in the dataset. This is bad because it is a "mean vector", e.g. the average response in the dataset.
# 
# A very good component would be one which has a bimodial distribution, as does the first component here. This indicates that there is a set of records which are *very distinguishable* from one another: a class A which is very close to this vector, and a class B which is very far away, with relatively few "middling" points. To understand *what* we are separating, construct and study a diagram like in the previous section. If we are using dimensionality reduction as part of our EDA before modeling, consider using similarity to this component as a feature!
# 
# Recall that very near to the beginning of the notebook we found that by far the highest gain in the dataset came from the first two components. This visualization illustrates why: those two components are the ones which are best separators of things (e.g. they are the most multimodal).
# 
# Components further into to dataset will tend to be normally distributed, with all of the various dataset records having more approximately equal distances from them. These components are picking up on features which are (1) more difficult to linearly separate and (2) occur in a smaller number of records.
# 
# Sometimes specific subfeatures demonstrated by this plot are worth exploring. For example, what is the contents of that very-near-the-metric clump at the beginning of Component 5? This demonstrates a set of things that are unusually homogeneous with regards to a particular component, and hence probably relatively easier to "tease out" (e.g. classify and separate) from the rest of the data.

# ## Conclusion
# 
# The advantage of PCA (and dimensionality reduction in general) is that it compresses your data down to something that is more effectively modeled. This means that it will, for example, compress away highly correlated and colinear variables, a useful thing to do when trying to run models that would otherwise be sensitive to these data problems.
# 
# As a pre-processing technique, dimensionality reduction makes the most sense when it is used to redesign a dataset with particularly many artificially defined features. Image datasets, for example, are a perfect target for PCA because they are entirely artificial: pixel values are simply not particularly meaningful to us as casual observers; whereas "does this look more like a shoe or a pair of pants" *is*.
# 
# This notebook demonstrated (hopefully) the power of PCA for lending insight to your analysis for just such an image dataset. It's worth knowing that the techniques here can be extended to non-image datasets, of course. The results will be harder to interpret. You'll need to spend more time staring at records to understand why the things are pointing the way they're pointing. But it's ultimately still a very powerful, very important technique.
# 
# Finally, PCA/dimensionality reduction can be used as a part of the pipeline search process to squeeze one last bit of performace out of a model. If the result is a model which is *significantly less* interpretable than the one you started with, I caution avoiding this if you can. Avoid putting into production models that you can't explain, or at least put a lot of effort into understanding them! See the comments for arguments on this particular point.
