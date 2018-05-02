
# coding: utf-8

# ## Resolving the Boat structure
# 
# Here I'm going to try to cluster the images based on boat, automatically. This can be useful for us in a variety of ways, for example it would allow us to create a separate model for each boat, or mask out which parts of the boat can't contain fish in order to help our classifier detect the fish.

# In[ ]:


import pandas as pd
import numpy as np
import glob
from sklearn import cluster
from scipy.misc import imread
import cv2
import skimage.measure as sm
# import progressbar
import multiprocessing
import random
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
new_style = {'grid': False}
plt.rc('axes', **new_style)

# Function to show 4 images
def show_four(imgs, title):
    #select_imgs = [np.random.choice(imgs) for _ in range(4)]
    select_imgs = [imgs[np.random.choice(len(imgs))] for _ in range(4)]
    _, ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(20, 3))
    plt.suptitle(title, size=20)
    for i, img in enumerate(select_imgs):
        ax[i].imshow(img)

# Function to show 8 images
def show_eight(imgs, title):
    select_imgs = [imgs[np.random.choice(len(imgs))] for _ in range(8)]
    _, ax = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(20, 6))
    plt.suptitle(title, size=20)
    for i, img in enumerate(select_imgs):
        ax[i // 4, i % 4].imshow(img)


# In[ ]:


select = 500 # Only load 500 images for speed
# Data loading
train_files = sorted(glob.glob('../input/train/*/*.jpg'), key=lambda x: random.random())[:select]
train = np.array([imread(img) for img in train_files])
print('Length of train {}'.format(len(train)))


# ### Image Size
# 
# The images in the training set are not all the same size, but rather there are a few distinct image sizes that we have to work with. Maybe there are some image sizes which only contain a single boat? Let's try treating the image sizes as BoatIDs and check!

# In[ ]:


print('Sizes in train:')
shapes = np.array([str(img.shape) for img in train])
pd.Series(shapes).value_counts()


# In[ ]:


for uniq in pd.Series(shapes).unique():
    show_four(train[shapes == uniq], 'Images with shape: {}'.format(uniq))
    plt.show()


# Here we can see that, with the exception of (854, 1518, 3), all the other image sizes all contain more than one boat. It looks like while this may be helpful, we need to take another approach to truly separate the boats.
# 
# ## Boat Clustering
# 
# Here I'm just going to work on just these 500 loaded images for speed, but there's nothing stopping you from doing exactly the same thing with all the images.
# 
# My approach has three steps:
# - Normalise all the images by subtracting the mean (of that image) and dividing by the stdev.
# - Treat the mean absolute pixel error between the images as the distance between the image.
# - Use this to create a precomputed distance matrix, and then pass the points to DBSCAN to create the boat clusters.

# In[ ]:


# Function for computing distance between images
def compare(args):
    img, img2 = args
    img = (img - img.mean()) / img.std()
    img2 = (img2 - img2.mean()) / img2.std()
    return np.mean(np.abs(img - img2))

# Resize the images to speed it up.
train = [cv2.resize(img, (224, 224), cv2.INTER_LINEAR) for img in train]

# Create the distance matrix in a multithreaded fashion
pool = multiprocessing.Pool(8)
#bar = progressbar.ProgressBar(max=len(train))
distances = np.zeros((len(train), len(train)))
for i, img in enumerate(train): #enumerate(bar(train)):
    all_imgs = [(img, f) for f in train]
    dists = pool.map(compare, all_imgs)
    distances[i, :] = dists


# Now I have a NxN matrix where N is the number of images, denoting the distances between the images.
# 
# Some clustering algorithms in SKLearn allow you to use a precomputed distance matrix instead of letting the algorithm compute it, which is very useful in cases like here where you can work out the distance between points but you can't give each image coordinates. Here I'll be using the DBSCAN algorithm.
# 
# Let's take a peek at our distance matrix:

# In[ ]:


print(distances)
plt.hist(distances.flatten(), bins=50)
plt.title('Histogram of distance matrix')
print('')


# We can see here that we have an area below 0.8. My hypothesis is that this is the area where two images are of the same boat. By default, DBSCAN considers up to a 0.5 distance to be in the same cluster. I tweaked the parameters and found that 0.6 is the best value, which makes reasonable sense looking at the histogram.

# In[ ]:


cls = cluster.DBSCAN(metric='precomputed', min_samples=5, eps=0.6)
y = cls.fit_predict(distances)
print(y)
print('Cluster sizes:')
print(pd.Series(y).value_counts())


# In[ ]:


for uniq in pd.Series(y).value_counts().index:
    if uniq != -1:
        size = len(np.array(train)[y == uniq])
        if size > 10:
            show_eight(np.array(train)[y == uniq], 'BoatID: {} - Image count {}'.format(uniq, size))
            plt.show()
        else:
            show_four(np.array(train)[y == uniq], 'BoatID: {} - Image count {}'.format(uniq, size))
            plt.show()            


# This is working surprisingly well!
# 
# However, we have a bunch of images that haven't been clustered into any BoatID. There are two possible reasons for these:
# 1. There are less than 5 images from the boat, which I have set as the threshold for a boatID.
# 2. The distance function is not good enough to cluster the images (eg. some images may be in the night vs the day - this would break my algorithm.)

# In[ ]:


size = len(np.array(train)[y == -1])
show_eight(np.array(train)[y == -1], 'BoatID: {} (Unclassified images) - Image count {}'.format(-1, size))


# Some of these I can cluster by eye, so it looks like the algorithm needs some tweaking to be able to cluster every image!
# 
# However, I think this can be considered a sucess as I've managed to cluster the boats very precisely, while clustering >75% of the boats. Since this is unsupervised, it can be run on the test set to in order to understand which boat is which!
# 
# Any feedback is very much appreciated, along with any upvotes for motivation ;)
# 
# Good luck!
