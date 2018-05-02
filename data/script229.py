
# coding: utf-8

# In this notebook, I examine the provided data for Kaggle's Humpback Whale ID challenge. I also look at data augmentations in an attempt to inflate the size of the training dataset.

# In[ ]:


import math
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


INPUT_DIR = '../input'


# In[ ]:


def plot_images_for_filenames(filenames, labels, rows=4):
    imgs = [plt.imread(f'{INPUT_DIR}/train/{filename}') for filename in filenames]
    
    return plot_images(imgs, labels, rows)
    
        
def plot_images(imgs, labels, rows=4):
    # Set figure to 13 inches x 8 inches
    figure = plt.figure(figsize=(13, 8))

    cols = len(imgs) // rows + 1

    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i], cmap='gray')


# In[ ]:


np.random.seed(42)


# ## Exploring the dataset

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# Let's plot a couple of images at random.

# In[ ]:


rand_rows = train_df.sample(frac=1.)[:20]
imgs = list(rand_rows['Image'])
labels = list(rand_rows['Id'])

plot_images_for_filenames(imgs, labels)


# The competition states that it's hard because: "there are only a few examples for each of 3,000+ whale ids", so let's take a look at the breakdown of number of image per category.

# In[ ]:


num_categories = len(train_df['Id'].unique())
     
print(f'Number of categories: {num_categories}')


# There appear to be too many categories to graph count by category, so let's instead graph the number of categories by the number of images in the category.

# In[ ]:


size_buckets = Counter(train_df['Id'].value_counts().values)


# In[ ]:


plt.figure(figsize=(10, 6))

plt.bar(range(len(size_buckets)), list(size_buckets.values())[::-1], align='center')
plt.xticks(range(len(size_buckets)), list(size_buckets.keys())[::-1])
plt.title("Num of categories by images in the training set")

plt.show()


# As we can see, the vast majority of classes only have a single image in them. This is going to make predictions very difficult for most conventional image classification models.

# In[ ]:


train_df['Id'].value_counts().head(3)


# In[ ]:


total = len(train_df['Id'])
print(f'Total images in training set {total}')


# New whale is the biggest category with 810, followed by `w_1287fbc`. New whale, I believe, is any whale that isn't in scientist's database. Since we can pick 5 potential labels per id, it's probably going to make sense to always include new_whale in our prediction set, since there's always an 8.2% change that's the right one. Let's take a look at one of the classes, to get a sense what flute looks like from the same whale.

# In[ ]:


w_1287fbc = train_df[train_df['Id'] == 'w_1287fbc']
plot_images_for_filenames(list(w_1287fbc['Image']), None, rows=9)


# In[ ]:


w_98baff9 = train_df[train_df['Id'] == 'w_98baff9']
plot_images_for_filenames(list(w_98baff9['Image']), None, rows=9)


# It's very difficult to build a validation set when most classes only have 1 image, so my thinking is to perform some aggressive data augmentation on the classes with < 10 images before creating a train/validation split. Let's take a look at a few examples of whales with only one example.

# In[ ]:


one_image_ids = train_df['Id'].value_counts().tail(8).keys()
one_image_filenames = []
labels = []
for i in one_image_ids:
    one_image_filenames.extend(list(train_df[train_df['Id'] == i]['Image']))
    labels.append(i)
    
plot_images_for_filenames(one_image_filenames, labels, rows=3)


# From these small sample sizes, it seems like > 50% of images are black and white, suggesting that a good initial augementation might be to just convert colour images to greyscale and add to the training set. Let's confirm that by looking at a sample of the images.

# In[ ]:


def is_grey_scale(img_path):
    """Thanks to https://stackoverflow.com/questions/23660929/how-to-check-whether-a-jpeg-image-is-color-or-gray-scale-using-only-python-stdli"""
    im = Image.open(img_path).convert('RGB')
    w,h = im.size
    for i in range(w):
        for j in range(h):
            r,g,b = im.getpixel((i,j))
            if r != g != b: return False
    return True


# In[ ]:


is_grey = [is_grey_scale(f'{INPUT_DIR}/train/{i}') for i in train_df['Image'].sample(frac=0.1)]
grey_perc = round(sum([i for i in is_grey]) / len([i for i in is_grey]) * 100, 2)
print(f"% of grey images: {grey_perc}")


# It might also be worth capturing the size of the images so we can get a sense of what we're dealing with.

# In[ ]:


img_sizes = Counter([Image.open(f'{INPUT_DIR}/train/{i}').size for i in train_df['Image']])

size, freq = zip(*Counter({i: v for i, v in img_sizes.items() if v > 1}).most_common(20))

plt.figure(figsize=(10, 6))

plt.bar(range(len(freq)), list(freq), align='center')
plt.xticks(range(len(size)), list(size), rotation=70)
plt.title("Image size frequencies (where freq > 1)")

plt.show()


# ## Data Augmentation

# In[ ]:


from keras.preprocessing.image import (
    random_rotation, random_shift, random_shear, random_zoom,
    random_channel_shift, transform_matrix_offset_center, img_to_array)


# In[ ]:


img = Image.open(f'{INPUT_DIR}/train/ff38054f.jpg')


# In[ ]:


img_arr = img_to_array(img)


# In[ ]:


plt.imshow(img)


# ### Random rotation

# In[ ]:


imgs = [
    random_rotation(img_arr, 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
    for _ in range(5)]
plot_images(imgs, None, rows=1)


# ### Random shift

# In[ ]:


imgs = [
    random_shift(img_arr, wrg=0.1, hrg=0.3, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
    for _ in range(5)]
plot_images(imgs, None, rows=1)


# ### Random shear

# In[ ]:


imgs = [
    random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
    for _ in range(5)]
plot_images(imgs, None, rows=1)


# ### Random zoom

# In[ ]:


imgs = [
    random_zoom(img_arr, zoom_range=(1.5, 0.7), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255
    for _ in range(5)]
plot_images(imgs, None, rows=1)


# ### Grey scale
# 
# We want to ensure that all colour images also have a grey scale version.

# In[ ]:


import random

def random_greyscale(img, p):
    if random.random() < p:
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    
    return img

imgs = [
    random_greyscale(img_arr, 0.5) * 255
    for _ in range(5)]

plot_images(imgs, None, rows=1)


# ### Flips
# 
# Usually for side-on image sets like this we'd include a veritical flip, however, in this case the veritical alignment is requirement to differentiate between flutes, so I'll leave it out.

# ### All together
# 
# Going to create an augmentation pipeline which will combine all the augs for a single predictions.

# In[ ]:


def augmentation_pipeline(img_arr):
    img_arr = random_rotation(img_arr, 18, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_zoom(img_arr, zoom_range=(0.9, 2.0), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_greyscale(img_arr, 0.4)

    return img_arr


# In[ ]:


imgs = [augmentation_pipeline(img_arr) * 255 for _ in range(5)]
plot_images(imgs, None, rows=1)

