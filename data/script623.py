
# coding: utf-8

# # Nucleus Masking - Data Science Bowl 2018

#     Shailesh Kumar Singh
#     sunnybeta322@gmail.com

# ## Problem

# ***MASK NUCLEUS OF CELLS IN IMAGES***

# ## Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
rcParams["figure.figsize"] = 16, 6
plt.style.use("seaborn-white")

import os
from subprocess import check_output

import tensorflow as tf


# ## Collect Data

# In[ ]:


PATH = "../input/"


# In[ ]:


print(check_output(["ls","../input/"]).decode("utf8"))


# In[ ]:


train1 = pd.read_csv(PATH + "stage1_train_labels.csv")
ss1 = pd.read_csv(PATH + "stage1_sample_submission.csv")


# ## Explore Data

# In[ ]:


train1.head()


# In[ ]:


print("There are {} rows of data.".format(train1.shape[0]))


# Our target variable is the "Encoded Pixel" column. For every image in the test set we have to obtain a mask for it.

# In[ ]:


TARGET = "EncodedPixels"


# In[ ]:


ss1.head()


# ### TRAINING IMAGES AND MASKS

# First, we will write functions for ease of viewing the images. I was confused about how the data was stored. So here is a tree diagram.

#     ../input/
#     
#             - stage1_sample_submissions.csv
#         
#             - stage1_train_labels.csv
#         
#             - stage1_train/
#                     - <image-id>/
#                             - images/
#                                     - <image-id>.png
#                             - masks/
#                                     - <mask_1>.png
#                                     - <mask_2>.png
#                                     .
#                                     .
#                                     .
#                                     - <mask_n>.png

# Next, create functions which returns the image or masks given the image id.

# In[ ]:


def dimg(idx):
    """
    Displays image corresponding to the id
    """
    img = mpimg.imread(PATH+"stage1_train/"+idx+"/"+"images/"+idx+".png")
    return img


# In[ ]:


def dmsk(idx):
    """
    Displays the masks corersponding to id
    """
    f = os.listdir(PATH+"stage1_train/"+idx+"/masks")[0]
    nim = mpimg.imread(PATH+"stage1_train/"+idx+"/masks/"+f)
    
    for m in os.listdir(PATH+"stage1_train/"+idx+"/masks")[1:]:
        nim += mpimg.imread(PATH+"stage1_train/"+idx+"/masks/"+m)
    
    return nim    


# In[ ]:


def dbth(idx):
    """
    Display both the mask and the image
    """
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(dimg(idx))
    ax[1].imshow(dmsk(idx), cmap="Purples")
    plt.show()


# Exmaple,

# In[ ]:


for ind in train1.sample(5)["ImageId"].index:
    print("Image ID:", ind)
    dbth(train1.iloc[ind,0])


# In[1]:


# Models Coming Soon


# ### ToDo
# 
# 1. Add "path" column to train1
# 2. Build Encoder in TF
# 3. First Submission by Jan 31
# 
# 
# 
# 
# 
