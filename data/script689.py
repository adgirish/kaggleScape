
# coding: utf-8

# # Validation split via VGG-based clustering

# In the NCFM competition it is challenging to set up a good validation strategy, since the training set contains many highly similar images. We use a clustering based on the level-4 VGG features in order separate similar images from non-similar ones.

# First, we import standard libraries and fix constants.

# In[ ]:


import h5py
import os

from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#path to training data
DATA_PATH = '../input/train'

#Number of clusters for K-Means
N_CLUSTS = 5#250

#Number of clusters used for validation
N_VAL_CLUSTS = 1#50

SEED = 42
np.random.seed(SEED)

##############################################
#######NORMALIZED IMAGE SIZE
##############################################
IMG_WIDTH = 640
IMG_HEIGHT = 360

##############################################
#######SUBSAMPLE DATA
##############################################

#how many images to take?
SAMP_SIZE = 8


# ## Subsample data 

# In order for the notebook to run on Kaggle scripts, we subsample the training data.

# In[ ]:


subsample = []
for fish in os.listdir(DATA_PATH):
    if(os.path.isfile(os.path.join(DATA_PATH, fish))): 
        continue
    subsample_class = [os.path.join(DATA_PATH, fish, fn) for 
                       fn in os.listdir(os.path.join(DATA_PATH, fish))]
    subsample += subsample_class
subsample = subsample[:SAMP_SIZE]


# ## Extract VGG features

# Next, we extract layer 4 VGG features from the images. The clustering will be based on these features. First, we load the VGG16 model pretrained on imagenet. Unfortunately, imagenet-weights are not available on Kaggle scripts.

# In[ ]:


base_model = VGG16(weights = None, include_top = False, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
#base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
model = Model(input = base_model.input, output = base_model.get_layer('block4_pool').output)


# After a preprocessing the images can be fed to the pretrained VGG16.

# In[ ]:


def preprocess_image(path):
    img = image.load_img(path, target_size = (IMG_HEIGHT, IMG_WIDTH))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis = 0)
    return preprocess_input(arr)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'preprocessed_images = np.vstack([preprocess_image(fn) for fn in subsample])\nvgg_features = model.predict(preprocessed_images)\nvgg_features = vgg_features.reshape(len(subsample), -1)')


# ## Cluster by K-Means 

# We cluster the images according to the K-Means algorithm.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'km = KMeans(n_clusters = N_CLUSTS, n_jobs = -1)\nclust_preds = km.fit_predict(StandardScaler().fit_transform(vgg_features))')


# Then, we select at random the clusters that will form the validation set.

# In[ ]:


val_clusters = np.random.choice(range(N_CLUSTS), N_VAL_CLUSTS, replace = False)
val_sample = np.array(subsample)[np.in1d(clust_preds, val_clusters)]

