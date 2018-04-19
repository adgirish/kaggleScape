
# coding: utf-8

# This notebook provides additional information about stage1_train dataset that should help to balance into train/valid for cross-validation. 
# 
# It starts from this interesting [thread](https://www.kaggle.com/c/data-science-bowl-2018/discussion/47640) about image types. Group A are histological slides, group B are fluorescent images, and group C are bright-field images. It can complete this other [kernel](https://www.kaggle.com/nhargan/defining-microscopy-type).
# 
# It looks we have the following breakdown:
# * fluorescent: 81.5%
# * histological: 16.1%
# * bright-field: 2.4%

# In[11]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io
import os
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from textwrap import wrap
np.random.seed(1234)
get_ipython().run_line_magic('matplotlib', 'inline')


# Initial variables.

# In[12]:


STAGE1_TRAIN = "../input/stage1_train"
STAGE1_TRAIN_IMAGE_PATTERN = "%s/{}/images/{}.png" % STAGE1_TRAIN
STAGE1_TRAIN_MASK_PATTERN = "%s/{}/masks/*.png" % STAGE1_TRAIN
IMAGE_ID = "image_id"
IMAGE_WIDTH = "width"
IMAGE_WEIGHT = "height"
HSV_CLUSTER = "hsv_cluster"
HSV_DOMINANT = "hsv_dominant"
TOTAL_MASK = "total_masks"


# Functions to load images. We keep only RGB channels as Alpha channel is always empty. 

# In[13]:


def image_ids_in(root_dir, ignore=[]):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids


# In[14]:


def read_image(image_id, space="rgb"):
    image_file = STAGE1_TRAIN_IMAGE_PATTERN.format(image_id, image_id)
    image = skimage.io.imread(image_file)
    # Drop alpha which is not used
    image = image[:, :, :3]
    if space == "hsv":
        image = skimage.color.rgb2hsv(image)
    return image


# In[15]:


# Get image width, height and count masks available.
def read_image_labels(image_id, space="rgb"):
    image = read_image(image_id, space = space)
    mask_file = STAGE1_TRAIN_MASK_PATTERN.format(image_id)
    masks = skimage.io.imread_collection(mask_file).concatenate()    
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = 255 #index + 1
    return image, labels, num_masks


# In[16]:


# Load stage 1 image identifiers.
train_image_ids = image_ids_in(STAGE1_TRAIN)


# Run KMeans on each image. Centroids provide dominant colors (based on provided colorspace). More details [here](https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/).

# In[17]:


def get_domimant_colors(img, top_colors=2):
    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters = top_colors)
    clt.fit(img_l)
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return clt.cluster_centers_, hist


# In[18]:


def get_images_details(image_ids):
    details = []
    for image_id in image_ids:
        image_hsv, labels, num_masks = read_image_labels(image_id, space="hsv")
        height, width, l = image_hsv.shape
        dominant_colors_hsv, dominant_rates_hsv = get_domimant_colors(image_hsv, top_colors=1)
        dominant_colors_hsv = dominant_colors_hsv.reshape(1, dominant_colors_hsv.shape[0] * dominant_colors_hsv.shape[1])
        info = (image_id, width, height, num_masks, dominant_colors_hsv.squeeze())
        details.append(info)
    return details


# In[19]:


META_COLS = [IMAGE_ID, IMAGE_WIDTH, IMAGE_WEIGHT, TOTAL_MASK]
COLS = META_COLS + [HSV_DOMINANT]


# In[20]:


details = get_images_details(train_image_ids)


# KMeans to split images by 3 types (based on dominant HSV colors distributions)

# In[21]:


trainPD = pd.DataFrame(details, columns=COLS)
X = (pd.DataFrame(trainPD[HSV_DOMINANT].values.tolist())).as_matrix()
kmeans = KMeans(n_clusters=3).fit(X)
clusters = kmeans.predict(X)
trainPD[HSV_CLUSTER] = clusters


# In[22]:


trainPD.head()


# Plot some examples for each cluster.

# In[23]:


def plot_images(images, images_rows, images_cols):
    f, axarr = plt.subplots(images_rows,images_cols,figsize=(16,images_rows*2))
    for row in range(images_rows):
        for col in range(images_cols):
            image_id = images[row*images_cols + col]
            image = read_image(image_id)
            height, width, l = image.shape
            ax = axarr[row,col]
            ax.axis('off')
            ax.set_title("%dx%d"%(width, height))
            ax.imshow(image)


# Some examples of cluster 0 (fluorescent images):

# In[24]:


plot_images(trainPD[trainPD[HSV_CLUSTER] == 0][IMAGE_ID].values, 6, 8)


# Some examples of cluster 1 (histological slides):

# In[25]:


plot_images(trainPD[trainPD[HSV_CLUSTER] == 1][IMAGE_ID].values, 6, 8)


# Some examples of cluster 2 (bright-field images):

# In[26]:


plot_images(trainPD[trainPD[HSV_CLUSTER] == 2][IMAGE_ID].values, 2, 8)


# In[27]:


P = trainPD.groupby(HSV_CLUSTER)[IMAGE_ID].count().reset_index()
P['Percentage'] = 100*P[IMAGE_ID]/P[IMAGE_ID].sum()
P


# In[28]:


f, ax = plt.subplots(1,1,figsize=(16,5))
r = trainPD.plot(kind="hist", bins=300, y = TOTAL_MASK, ax=ax, grid=True, title="Masks Histogram")


# Plot some images with masks:

# In[29]:


def plot_image_masks(image, labels, num_masks, image_id):
    f, ax = plt.subplots(1,3,figsize=(16,5))
    d = ax[0].axis('off')
    d = ax[0].imshow(image)
    d = ax[0].set_title("\n".join(wrap(image_id, 32)))
    d = ax[1].axis('off')
    d = ax[1].imshow(labels)
    d = ax[1].set_title("masks: %d"%num_masks)
    d = ax[2].axis('off')
    d = ax[2].imshow(image)
    d = ax[2].imshow(labels, alpha=0.5)
    d = ax[2].set_title("both")


# In[30]:


def display_image_masks(image_id):
    image, labels, num_masks = read_image_labels(image_id)
    plot_image_masks(image, labels, num_masks, image_id)


# In[31]:


display_image_masks("8f27ebc74164eddfe989a98a754dcf5a9c85ef599a1321de24bcf097df1814ca")


# In[32]:


display_image_masks(trainPD[trainPD[TOTAL_MASK] == trainPD[TOTAL_MASK].median()][IMAGE_ID].values[0])


# In[33]:


display_image_masks(trainPD[trainPD[TOTAL_MASK] == trainPD[TOTAL_MASK].min()][IMAGE_ID].values[0])


# In[34]:


display_image_masks(trainPD[trainPD[TOTAL_MASK] == trainPD[TOTAL_MASK].max()][IMAGE_ID].values[0])

