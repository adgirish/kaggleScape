
# coding: utf-8

# **Introduction**
# 
# This will be a short exploratory analysis with the goal of becoming more familiar with the 2018 Data Science Bowl dataset and identifying some possible hurdles that could have a negative effect on model performance.
# 
# **Contents:**
# - Importing and processing image data
# - Looking at the image metadata summary statistics
# - Plotting image width, height and area distributions
# - Plotting number of nuclei per image distribution
# - Plotting images with the most and fewest nuclei
# - Plotting the smallest and largest nuclei
# - Plotting a sample of images
# - Conclusion

# In[2]:


import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import cv2

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]


# In[3]:


# sys.stdout.flush()
# These lists will be used to store the images.
imgs = []
masks = []

# These lists will be used to store the image metadata that will then be used to create
# pandas dataframes.
img_data = []
mask_data = []
print('Processing images ... ')

# Loop over the training images. tqdm is used to display progress as reading
# all the images can take about 1 - 2 minutes.
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')
    
    # Get image.
    imgs.append(img)
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_area = img_width * img_height

    # Initialize counter. There is one mask for each annotated nucleus.
    nucleus_count = 1
    
    # Loop over the mask ids, read the images and gather metadata from them.
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask = imread(path + '/masks/' + mask_file)
        masks.append(mask)
        mask_height = mask.shape[0]
        mask_width = mask.shape[1]
        mask_area = mask_width * img_height
        
        # Sum and divide by 255 to get the number
        # of pixels for the nucleus. Masks are grayscale.
        nucleus_area = (np.sum(mask) / 255)
        
        mask_to_img_ratio = nucleus_area / mask_area
        
        # Append to masks data list that will be used to create a pandas dataframe.
        mask_data.append([n, mask_height, mask_width, mask_area, nucleus_area, mask_to_img_ratio])
        
        # Increment nucleus count.
        nucleus_count = nucleus_count + 1
    
    # Build image info list that will be used to create dataframe. This is done after the masks loop
    # because we want to store the number of nuclei per image in the img_data list.
    img_data.append([img_height, img_width, img_area, nucleus_count])


# **Create Images Metadata Dataframe**
# 
# Create a pandas data frame from the list of image metadata that was created in the loop above. This will make the data easier to manipulate and plot.
# 
# After creating the data frame we can take a look at its summary stats along with the first five and last five rows to make sure it looks ok and get familiar with it.

# In[4]:


# Create dataframe for images
df_img = pd.DataFrame(img_data, columns=['height', 'width', 'area', 'nuclei'])


# In[ ]:


df_img.describe()


# In[ ]:


df_img.head()


# In[ ]:


df_img.tail()


# **Create Image Masks Metadata Dataframe**
# 
# Let's create another data frame for the masks metadata.

# In[6]:


# Create dataframe for masks
df_mask = pd.DataFrame(mask_data, columns=['img_index', 'height', 'width', 'area', 'nucleus_area', 'mask_to_img_ratio'])


# In[ ]:


df_mask.describe()


# In[ ]:


df_mask.head()


# In[ ]:


df_mask.tail()


# **Plot some of the metadata distributions**

# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(10,5))
width_plt = sns.distplot(df_img['width'].values, ax=ax[0])
width_plt.set(xlabel='width (px)')
width_plt.set(ylim=(0, 0.01))
height_plt = sns.distplot(df_img['height'].values, ax=ax[1])
height_plt.set(xlabel='height (px)')
height_plt.set(ylim=(0, 0.015))
area_plt = sns.distplot(df_img['area'].values)
area_plt.set(xlabel="area (px)")
fig.show()
plt.tight_layout()


# The image dimensions do not appear to be equally distributed and have a somewhat bimodal distribution. This issue will need to be addressed before feeding the images to our model. The images could be resized to squares but this will cause images to be squashed either vertically or horizontally and would result in a loss of information. Most of the images look like they are 256x256 so the squashing/stretching might not be an issue. Scaling the smaller images up might be a better option than scaling the larger images down as this would also result in a loss of information. Different strategies should be tried here to see what works best. 

# In[ ]:


sns.distplot(df_img['nuclei'].values)
plt.xlabel("nuclei")
plt.show()


# The distribution of nuclei (masks) per image appears to be skewed to the right. There is quite a large range in the number of nuclei per image. 

# In[ ]:


plt.figure(figsize=(18, 18))
much_nuclei = df_img['nuclei'].argmax()
print(much_nuclei)
plt.grid(None)
plt.imshow(imgs[much_nuclei])


# There are 198 annotated nuclei in this image. It's an interesting structure, possibly some kind of creature!

# In[ ]:


plt.figure(figsize=(18, 18))
not_much_nuclei = df_img['nuclei'].argmin()
print(df_img['nuclei'].min())
plt.grid(None)
plt.imshow(imgs[not_much_nuclei])


# There appears to be a lot going on in this image but there are only two annotated nuclei.

# **Nuclei Sizes**
# 
# Let's take a look at some differently sized nuclei in the training set. 

# In[ ]:


smallest_mask_index = df_mask['mask_to_img_ratio'].argmin()

fig, ax = plt.subplots(1, 2, figsize=(16, 16))
ax[0].grid(None)
ax[0].imshow(masks[smallest_mask_index])
ax[1].grid(None)
ax[1].imshow(imgs[df_mask.iloc[[smallest_mask_index], [0]].values[0][0]])
plt.tight_layout()


# Wow! This nucleus is either very small or very far away! This makes me concerned about scaling down some of the larger images as some of these small nuclei could become undetectable or a least much more difficult to detect.

# In[ ]:


smallest_mask_resized_128 = resize(masks[smallest_mask_index], (128, 128))
smallest_mask_resized_256 = resize(masks[smallest_mask_index], (256, 256))
smallest_mask_resized_512 = resize(masks[smallest_mask_index], (512, 512))
print(np.sum(smallest_mask_resized_128))
print(np.sum(smallest_mask_resized_256))
print(np.sum(smallest_mask_resized_512))
fig, ax = plt.subplots(1, 3, figsize=(14, 14))
ax[0].grid(None)
ax[1].grid(None)
ax[2].grid(None)
ax[0].imshow(smallest_mask_resized_128)
ax[1].imshow(smallest_mask_resized_256)
ax[2].imshow(smallest_mask_resized_512)


# As you can see above, the nucleus mask completely disappears when the image is scaled down to 128x128 pixels.

# In[9]:


biggest_mask_index = df_mask['mask_to_img_ratio'].argmax()
biggest_mask_img_index = df_mask.iloc[[biggest_mask_index], [0]].values[0][0]

fig, ax = plt.subplots(1, 2, figsize=(12, 12))
ax[0].grid(None)
ax[1].grid(None)
ax[0].imshow(masks[biggest_mask_index])
ax[1].imshow(imgs[biggest_mask_img_index])
plt.tight_layout()


# In the image on the right, the nuclei appear to overlap each other. Let's see what the masks look like when stacked on top of each other.

# In[10]:


big_nuclei = df_mask.index[df_mask['img_index'] == biggest_mask_img_index]
plt.figure(figsize=(18, 18))
for i, mask_id in enumerate(big_nuclei):
    plt.grid(None)
    plt.imshow(masks[mask_id], interpolation='none', alpha=0.1)


# Whereas the nuclei in the image overlap the masks do not appear to do so. 

# In[14]:


sample_nuclei = df_img.sample(36).index
fig, ax = plt.subplots(9, 4, figsize=(16, 16))
row = 0
col = 0
for i, img_id in enumerate(sample_nuclei):
    ax[row, col].grid(False)
    ax[row, col].imshow(imgs[img_id])
    
    # Increment col index and reset each time
    # it gets to 4 to start a new row
    col = col + 1
    if(col == 4):
        col = 0
    
    # Increment row index every 4 items
    if((i + 1) % 4 == 0):
        row = row + 1
plt.tight_layout()


# There is a wide range of different nuclei sizes and shapes.

# **Conclusion**
# 
# There is a large range of image dimensions in the dataset and not all of the images are square. The smallest image was 256x256 and the largest was 1040x1388 pixels. The smallest nucleus was only a few pixels in size and was found in one of the larger images (1000x1000), resizing this image caused the tiny nucleus to disappear so resizing images should be approached with great caution. The size of nuclei vary a lot throughout the images in the training set and is likely to make detection more challenging. 
# 
# I have only scratched the surface of this dataset and there is much more to explore. A few more things I would like to look into are the distribution of color in the images, identifying different nuclei groups/clusters and taking a look at the test set.
# 
# *Any suggestions for improvements would be very helpful. Also, please don't hesitate to point out any mistakes I might have made (there are probably a lot of them!).*

# I adapted various parts of the following kernels while putting this together:
# - https://www.kaggle.com/jerrythomas/exploratory-analysis
# - https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
