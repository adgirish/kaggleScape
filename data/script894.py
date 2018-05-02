
# coding: utf-8

# #INTRODUCTION
# 
# In a nutshell, I will be using a Principal Component Analysis (PCA) based approach to analysing the IMDB dataset and then implementing some KMeans clustering to provide visualisations of any related clusters I find in the dataset. Some caveats : I will not be attempting to run any predictive models ( XGBoosting, SVM, Regression... that sort of thing). This notebook will purely be an exploratory and hopefully concise enough attempt to explain the idea of PCA as well as using a clustering method (KMeans) to extract meaningful relations out of it. 
# 
# A very high-level description of PCA is that it serves as a dimensionality reduction method on the features of our original dataset by projecting these features onto a lower dimension. Therefore if our original dataset contains 72 columns ( i.e features) and we manage to reduce these 72 columns down to 9 columns image the gains in time and processing speeds! However the question is how do we get the new data out of the PCA-reduced 9 columns? Via a clustering method!! In this case, I will use KMeans - more to come below. This notebook is organised as follows : 
# 
#  1. Filtering the Dataset to remove Null values and get only numerical columns. Standardising the features 
#  2. Using the measure of **Explained Variance** to motivate and inform our search on getting the right number of PCA projections. Refer to Sebastian Raschka's awesome piece on Explained variance ( and PCA in general) here : http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html. My notebook heavily borrows from his article so I need to give it a shout-out.
#  3. Implementing Principal Component Analysis 
#  4. Using KMeans clustering to investigate relationships in the PCA projections (if there are any at all) and creating visualisations using the said clusters.
#  5. Pseudo means of extracting KMeans clusters and use those as new features (meta features) in your predictions - STILL UNDER PROGRESS

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA # Principal Component Analysis module
from sklearn.cluster import KMeans # KMeans clustering 
import matplotlib.pyplot as plt # Python defacto plotting library
import seaborn as sns # More snazzy plotting library
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's import the movie dataset with imagination. The dataset will be called "movie" and let's inspect the first 5 rows of the dataframe with .head()

# In[ ]:


movie = pd.read_csv('../input/movie_metadata.csv') # reads the csv and creates the dataframe called movie
movie.head()


# # 1. DATA FILTERING AND CLEANSING
# **Filtering for Numerical values only**
# 
# As observed from the dataframe above, some columns contain numbers while others, words. Let's do some filtering to extract only the numbered columns and not the ones with words ( just for the purpose of this exercise). To do so I will create a Python list containing the numbered column names "num_list"

# In[ ]:


str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in movie.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = movie.columns.difference(str_list)         


# Using the magic of Pandas dataframe filtering, we can create a new dataframe (movie_num) containing just the numbers as such : 

# In[ ]:


movie_num = movie[num_list]
#del movie # Get rid of movie df as we won't need it now
movie_num.head()


# **Removal of Null values**

# Now since there still exists 'NaN' values in our dataframe, and these are Null values, we have to do something about them. In here, I will just do the naive thing of replacing these NaNs with zeros as such:

# In[ ]:


movie_num = movie_num.fillna(value=0, axis=1)


# **Standardisation** 
# 
# Finally we mentioned that we have to find some sort of way to standardise the data and for this, we use sklearn's StandardScaler.

# In[ ]:


X = movie_num.values
# Data Normalization
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


# Let's look at some hexbin visualisations first to get a feel for how the correlations between the different features compare to one another. In the hexbin plots, the lighter in color the hexagonal pixels, the more correlated one feature is to another.

# In[ ]:


movie.plot(y= 'imdb_score', x ='duration',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Duration',figsize=(12,8))
movie.plot(y= 'imdb_score', x ='gross',kind='hexbin',gridsize=45, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Gross',figsize=(12,8))


# From the Hexbin plots, one can tell see that the correlation between IMDB score and gross is one that is quite obvious to explain while an interesting result thrown up is that of the score and the duration. ( Interesting!)

# Anyway now - time for the customary heatmap per the tradition of most notebooks on Principal Component Analysis. The heatmap is generated to visually show how strongly correlated the values of the dataframe's columns are to one another. Therefore in this matrix the squares that are of a darker colour are more strongly correlated compared to the ones of lighter colour. 

# In[ ]:


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))
plt.title('Pearson Correlation of Movie Features')
# Draw the heatmap using seaborn
sns.heatmap(movie_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)


# As we can see from the heatmap, there are regions (features) where we can see quite positive linear correlations amongst each other, given the darker shade of the colours - top left-hand corner and bottom right quarter. This is a good sign as it means we may be able to find linearly correlated features for which we can perform PCA projections on.

# # 2. EXPLAINED VARIANCE MEASURE
# As alluded to in the Introduction, I will be using a particular measure called Explained Variance which will be useful in this context to help us determine the number of PCA projection components we should be looking at. Again this section heavily borrows from Sebastian Raschka's article on Principal Component Analysis so please follow his link for a much more detailed explanation on explained variance than I can do justice to : http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html

# In[ ]:


# Calculating Eigenvectors and eigenvalues of Cov matirx
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)


# Now having obtained the eigenvalues and eigenvectors, we will group them together by creating a list of eigenvalue, eigenvector tuples (immutable Python data objects). Following on from this we will sort the list  in order of Highest eigenvalue to lowest eigenvalue and then use the eigenvalues to calculate both the individual explained variance and the cumulative explained variance for visualisation.

# In[ ]:


# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance


# Now time to plot the explained variance graphs to see how our contributions look like. The cumulative explained variance is visualised in a blue step-plot while the individual explained variance is plotted via green bar charts as follows: 

# In[ ]:


# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 
plt.figure(figsize=(10, 5))
plt.bar(range(16), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')
plt.step(range(16), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()


# From the plot above, it can be seen that approximately 90% of the variance can be explained with the 9 principal components. Therefore for the purposes of this notebook, let's implement PCA with 9 components ( although to ensure that we are not excluding useful information, one should really go for 95% or greater variance level which corresponds to about 12 components).

# # 3. PRINCIPAL COMPONENT ANALYSIS 
# Having roughly identified how many components/dimensions we would like to project on, let's now implement sklearn's PCA module. 
# 
# The first line of the code contains the parameters "n_components" which states how many PCA components we want to project the dataset onto. Since we are going implement PCA with 9 components, therefore we set n_components = 9.  
# 
# The second line of the code calls the "fit_transform" method, which fits the PCA model with the standardised movie data X_std and applies the dimensionality reduction on this dataset. 

# In[ ]:


pca = PCA(n_components=9)
x_9d = pca.fit_transform(X_std)


# Awesome. Having now applied our specific PCA model with the movie dataset, let's visualise the first 2 projection components as a 2D scatter plot to see if we can get a quick feel for the underlying data. 

# In[ ]:


plt.figure(figsize = (9,7))
plt.scatter(x_9d[:,0],x_9d[:,1], c='goldenrod',alpha=0.5)
plt.ylim(-10,30)
plt.show()


# As a quick aside, my aim (or hope) in carrying out this quick and dirty plotting is to see if we can observe distinct clusters already present within the plots which would be able to tell us if our PCA-transformed data can indeed be linearly separable into different groups for later use as our new features. 
# 
# However from the 2D plot above of the first 2 PCA projections, the first visual impression is that there does not seem to be any discernible clusters. However keeping in mind that our PCA projections contain another 7 components, perhaps looking at plots with the other components may be fruitful. For now, let us assume that will be trying a 3-cluster (just as a naive guess) KMeans to see if we are able to visualise any distinct clusters.

# #4. VISUALISATIONS WITH KMEANS CLUSTERING
# A simple KMeans will now be applied to the PCA projection data. Each cluster will be visualised with a different colour so hopefully we will be able to pick out clusters by eye. 
# 
# To start off, we set up a KMeans clustering with sklearn's KMeans() and call the "fit_predict" method to compute cluster centers and predict cluster indices for the first and third PCA projections (to see if we can observe any appreciable clusters). We then define our own colour scheme and plot the scatter diagram as follows:

# In[ ]:


# Set a 3 KMeans clustering
kmeans = KMeans(n_clusters=3)
# Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(x_9d)

# Define our own color map
LABEL_COLOR_MAP = {0 : 'r',1 : 'g',2 : 'b'}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

# Plot the scatter digram
plt.figure(figsize = (7,7))
plt.scatter(x_9d[:,0],x_9d[:,2], c= label_color, alpha=0.5) 
plt.show()


# This KMeans plot looks more promising now as if our simple clustering model assumption turns out to be right, we can observe 3 distinguishable clusters via this color visualisation scheme.
# 
# Now, the plot above was only for 2 PCA projections out of the 9 projections that we currently have. However I would also like to generate a KMeans visualisation for other possible combinations of the projections against one another. I will use Seaborn's convenient **pairplot** function to do the job. Basically pairplot automatically plots all the features in the dataframe (in this case our PCA projected movie data) in pairwise manner. I will pairplot the first 3 projections against one another and the resultant plot is given below:

# In[ ]:


# Create a temp dataframe from our PCA projection data "x_9d"
df = pd.DataFrame(x_9d)
df = df[[0,1,2]] # only want to visualise relationships between first 3 projections
df['X_cluster'] = X_clustered


# In[ ]:


# Call Seaborn's pairplot to visualize our KMeans clustering on the PCA projected data
sns.pairplot(df, hue='X_cluster', palette= 'Dark2', diag_kind='kde',size=1.85)


# # Conclusion
