
# coding: utf-8

# # Import required libraries
# ***
# First let's import required libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import sobel, rank
from skimage.feature import canny
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.util import img_as_ubyte, invert
from skimage.morphology import disk
from skimage import img_as_float
from skimage.morphology import reconstruction
from scipy.ndimage import gaussian_filter
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value


# # Notebook Parameters
# ***
# Here we define some parameters for this notebook.

# In[ ]:


INPUT_PATH = '../input/'


# # Read json files
# ***
# Pandas has a read_json method. Let's use it!

# In[ ]:


# Read the json files into a pandas dataframe
df_train = pd.read_json(INPUT_PATH + 'train.json')


# ## Plot Icebergs and boats (BAND 1)
# ***
# Lets plot Icebergs and boats!<br>
# To do this, we will convert the band dimensions to 75x75 which is the actual size of the images!

# In[ ]:


img.shape


# In[ ]:


img = np.array(df_train.iloc[0]['band_1']).reshape((75,75))
#img = np.array(img).view(np.uint8)
plt.subplot(1,3,1)
plt.imshow(img)
plt.subplot(1,3,2)
img2 = np.array(df_train.iloc[0]['band_2']).reshape((75,75))
plt.imshow(img2)
plt.subplot(1,3,3)
plt.imshow(img + img2)

#img.shape


# In[ ]:


def plot_image(image):
    
    
    #image = invert(image)
    #seed = np.copy(image)
    #seed[1:-1, 1:-1] = image.min()
    #mask = image
    #image = reconstruction(seed, mask, method='dilation')
    # Detect edges using canny
    #image = canny(image)
    #image = equalize_adapthist(image)
    # Show the image
    plt.imshow(image)

# FIrst get 9 random icebergs
df_plot = df_train.loc[df_train['is_iceberg'] == True].sample(9)

plt.figure(figsize=(10,10))
for i in range(3):
    # Set the current subplot
    plt.subplot(2,3,(i+1))
    # Reshape the array to 75x75
    image = np.array(df_plot.iloc[i]['band_1']).reshape((75,75)).astype(np.float32)
    plot_image(image)
    plt.subplot(2,3,(i+4))
    image = np.array(df_plot.iloc[i]['band_2']).reshape((75,75)).astype(np.float32)
    plot_image(image)
    
# Set the title
plt.suptitle('ICEBERGS!',fontsize=20)


# FIrst get 9 random icebergs
df_plot = df_train.loc[df_train['is_iceberg'] == False].sample(9)

plt.figure(figsize=(10,10))
for i in range(3):
     # Set the current subplot
    plt.subplot(2,3,(i+1))
    # Reshape the array to 75x75
    image = np.array(df_plot.iloc[i]['band_1']).reshape((75,75)).astype(np.float32)
    plot_image(image)
    plt.subplot(2,3,(i+4))
    image = np.array(df_plot.iloc[i]['band_2']).reshape((75,75)).astype(np.float32)
    plot_image(image)
    
plt.suptitle('BOATS!', fontsize=20)


# |## Plot Icebergs and boats (BAND 2)
# ***
# Do the same as previous step but on band 2

# In[ ]:


# FIrst get 9 random icebergs
df_plot = df_train.loc[df_train['is_iceberg'] == 1].sample(9)

plt.figure(figsize=(16,16))
for i in range(9):
    # Set the current subplot
    plt.subplot(3,3,i+1)
    # Reshape the array to 75x75
    image = np.array(df_plot.iloc[i]['band_2']).reshape((75,75))
    # Show the image
    plt.imshow(image)
    
# Set the title
plt.suptitle('ICEBERGS!',fontsize=20)


# FIrst get 9 random icebergs
df_plot = df_train.loc[df_train['is_iceberg'] == 0].sample(9)

plt.figure(figsize=(16,16))
for i in range(9):
     # Set the current subplot
    plt.subplot(3,3,i+1)
    # Reshape the array to 75x75
    image = np.array(df_plot.iloc[i]['band_2']).reshape((75,75))
    # Show the image
    plt.imshow(image)
    
plt.suptitle('BOATS!', fontsize=20)

