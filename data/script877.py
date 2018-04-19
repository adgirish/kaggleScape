
# coding: utf-8

# # A CLUSTER OF COLORS
# #### **PCA APPROACH**

# # 1. Introduction
# 
# This notebook is not meant to be an exhaustive EDA nor will it attempt to run fancy stuff like XGBoosting or Ensembling methods. The chief focus of this script will simply be to try out sklearn's PCA (Principal Decomposition Analysis) method on a small dataset, hence the choice to only look at mushroom colors. Therefore this notebook is organized as follows:
# 
#  - Label encoding the categorical values
#  - Pearson Correlation to investigate any linear dependence on the color features 
#  - PCA and KMeans clustering for visualization

# # 2. Extracting only the color features 
# The first step is to extract all the features in the dataset that point to the colors of the mushroom. Inspecting the data, we see that there are 6 columns (features) that allude to colors
# 
#  1. cap-color 
#  2. gill-color
#  3. stalk-color-above-ring
#  4. stalk-color-below-ring
#  5. veil-color
#  6. spore-print-color

# In[ ]:


# Importing the usual libraries
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

# Read in the mushroom data into a dataframe called "data" - what a creative name
data = pd.read_csv("../input/mushrooms.csv")
data.head()


# Therefore let's extract these 6 color columns into its own dataframe (data_color)

# In[ ]:


# I use a list "color_features" to store the color column names. 
# Not really sure if there is an easier way to do this. Do let me know if there is
color_features = []
for i in data.columns:
    if 'color' in i:
        color_features.append(i)
# create our color dataframe and inspect first 5 rows with head()
data_color = data[color_features]
data_color.head()


# ### Encoding categorical values
# We see that the colors are all categorical values. Therefore we need to encode. Since the color's categorical value correspond to one another across columns, I want to ensure that the encoding provides the same output across all columns.  Therefore my idea was to create a dictionary that contains the encoding for the unique values across the dataframe. 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
# List to store all unique categories
ListToEncode = pd.Series(data_color.values.ravel()).unique()
# Use sklearn Labelencoder for transformation
EncodedList = LabelEncoder().fit_transform(ListToEncode)

# Define a dictionary "encodedict" to store our encoding
encodedict = {}
for i in range(0, len(EncodedList)):
    encodedict.update({ListToEncode[i]:EncodedList[i]})

# Finally use dictionary to generate encoded dataframe
for i in range(len(data_color.columns)):
    for j in range(len(data_color['cap-color'].values)):
        data_color.values[j][i] =  encodedict[data_color.values[j][i]]
data_color.head()       
  


# Let's plot some hexplots for visualisation

# In[ ]:


data_color.plot(y= 'cap-color', x ='stalk-color-below-ring',kind='hexbin',gridsize=45, sharex=False, colormap='spectral', title='Hexbin of cap-color and stalk-color-below-ring')


# In[ ]:


data_color.plot(y= 'cap-color', x ='stalk-color-above-ring',kind='hexbin',gridsize=35, sharex=False, colormap='gnuplot', title='Hexbin of cap-color and stalk-color-above-ring')


# Just some pretty visuals, but let's delve deeper into the meat of the data by looking at Pearson Correlation.

# # 3. Correlation of color features
# Now let's look at the Pearson correlation of the color features as a sort of first attempt to identify how linearly related they are to one another.

# In[ ]:


# correlation matrix using the corr() method
data_corr = data_color.astype(float).corr()  # used the astype() or else I get empty results
data_corr


# And to visualize this with a more swanky heatmap that everyone is using these days.

# In[ ]:


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(7, 7))
plt.title('Pearson Correlation of Mushroom Features')
# Draw the heatmap using seaborn
sns.heatmap(data_color.astype(float).corr(),linewidths=0.5,vmax=1.0, square=True, annot=True)


# Seems that from this heatmap, we identify about 2 or 3 features that have some weakly to medium positive linear correlation with one another. Therefore as a rough heuristic, let's look at PCA-ing the features into 3 components. 

# # 4. Principal Component Analysis with KMeans Clustering

# Thankfully, the immense power of the sklearn module can be utilized to implement Principal Component Analysis conveniently. Check out the official sklearn link for a more detailed explanation : http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# 
# We also import the KMeans method so that we can use KMeans clustering to extract our PCA components.

# In[ ]:


# import the relevant modules
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# From the code below, since I am going to look at PCA with 3 components, therefore I assign the PCA parameter "n_components" to be equal to 3. The method of "fit_transform" fits the model with X ( mushroom color values ) and then reduces the dimensions of X to our stated 3 dimensions.

# In[ ]:


X = data_color.values
# calling sklearn PCA 
pca = PCA(n_components=3)
# fit X and apply the reduction to X 
x_3d = pca.fit_transform(X)

# Let's see how it looks like in 2D - could do a 3D plot as well
plt.figure(figsize = (7,7))
plt.scatter(x_3d[:,0],x_3d[:,1], alpha=0.1)
plt.show()


# With this 2D plot of the PCA projections, let's try to apply a simple KMeans and see if we can identify any clusters from the projections.

# In[ ]:


# Set a 3 KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
# Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(x_3d)


# Simple visualisation of the 3 clusters with a pre-defined color map

# In[ ]:


LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'g',
                   2 : 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
plt.figure(figsize = (7,7))
plt.scatter(x_3d[:,0],x_3d[:,1], c= label_color, alpha=0.1)
plt.show()


# # CLOSING REMARKS
# 
# Since I'm only starting out and am still very green behind the ears around data science, I will stop my notebook at this juncture. However, this PCA decomposition coupled with KMeans clustering (or other clustering methods) can be quite powerful, especially when you imagine that your dataset features contain 100s or 1000s of columns you are able to scale it down by an order of magnitude via this method. To take this further from a qualitative point of view, one would then extract the KMeans clusters and use those as new features in training the Machine Learning model should the effect of this dimensionality reduction + clustering prove helpful. 
# 
# Please feel free to leave comments and thoughts on how I could improve this notebook from a data science point of view or plotting point of view or organisational point of view or views from any other point.  :)
