
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import skimage.io
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# **Next section reads a random subset of 16 images from the training data set and shows them along with their nuclei masks. Odd rows in the display are the images and Even rows are the masks for the previous row of images.**

# In[ ]:


def read_image_labels(image_id):
    # most of the content in this function is taken from 'Example Metric Implementation' kernel 
    # by 'William Cukierski'
    image_file = "../input/stage1_train/{}/images/{}.png".format(image_id,image_id)
    mask_file = "../input/stage1_train/{}/masks/*.png".format(image_id)
    image = skimage.io.imread(image_file)
    masks = skimage.io.imread_collection(mask_file).concatenate()    
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + 1
    return image, labels

def plot_images_masks(image_ids):
    plt.close('all')
    fig, ax = plt.subplots(nrows=8,ncols=4, figsize=(50,50))
    for ax_index, image_id in enumerate(image_ids):
        image, labels = read_image_labels(image_id)
        img_row, img_col, mask_row, mask_col = int(ax_index/4)*2, ax_index%4, int(ax_index/4)*2 + 1, ax_index%4
        ax[img_row][img_col].imshow(image)
        ax[mask_row][mask_col].imshow(labels)

image_ids = check_output(["ls", "../input/stage1_train/"]).decode("utf8").split()
print("Total Images in Training set: {}".format(len(image_ids)))
random_image_ids = random.sample(image_ids, 16)
print("Randomly Selected Images: {}, their IDs: {}".format(len(random_image_ids), random_image_ids))
plot_images_masks(random_image_ids)
    

