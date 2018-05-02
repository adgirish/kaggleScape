
# coding: utf-8

# Hello Kagglers!! I hope you are busy doing analysis and building models on Kaggle.  This competition is hosted by the 2018 CVPR workshop on autonomous driving (WAD). Here Kagglers are challenged to develop a robust segmentation algorithm that can be used for self-driving cars. I have been working in this field since last year, so I know how crucial this part is. Before we dive into the data analysis part, I would like to elaborate on some important aspects regarding segmentation:
# 
# * For a self-driving car, segmentation or to be more precise semantic segmentation gives your car a view of what lies ahead of it. Though little bit computationally expensive compared to the algorithms generating bounding boxes, semantic segmentation gives a car much more idea about scene understanding. Also, it overcomes all those critical situations where bounding box fails miserably.
# 
# * It is not possible to label each and every type of object/instance on the road. This poses another challenge i.e. how to interpret something that is labeled by the model as an `unknown` instance. How do you interpret the label map generated in this case?
# 
# * **Most important point** Self-driving cars run with chipsets integrated within the car system. Though most people use `Drive PX2` but not all of them. Self-driving car makes sense only when your model is able to run on an embedded device and you get efficient, if not almost real-time, FPS.  So, don't ensemble first. Try to develop something that is lightweight and is capable of running on an embedded device because that is the model for which this competition is hosted 
# 
# 
# Let's dive into the dataset!!
# 
# ![](https://media.giphy.com/media/mlBDoVLOGidEc/giphy.gif)

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import io
import glob
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from concurrent.futures import ProcessPoolExecutor
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from skimage.io import imread
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# Define some paths first
input_dir = Path('../input')
images_dir = input_dir / 'train_color'
labels_dir = input_dir / 'train_label'


# In[3]:


# Hoe many samples are there in the training dataset?
train_images = sorted(os.listdir(images_dir))
train_labels = sorted(os.listdir(labels_dir))

print("Number of images and labels in the training data: {}  and {} respectively".format(len(train_images), len(train_labels)))


# ### How the dataset is arranged?
# 
# It is actually very interesting how the information is encoded in label corresponding to an image. The data comes from a video stream which is a very common data collection technique when it comes to self-driving cars. Each image contains certain objects. Within an image, there can be a single object or multiple objects of same/different type. for example within a frame, there may be a single car or three cars or a car, a person and a bus.
# 
# Now each label map is an image of the same size as the original image. Here is the interesting part. All pixels except for the pixels belonging to a certain object we want to coonisder for our dataset, has a value of 255. All the pixels that belong to an object has a value greater tha 255 and is given a value in such a way that when you divide that value by 1000, you get the class of the object to which this pixel belongs to and doing a mod by 1000 will give you the instance number. 
# 
# For example, a pixel value of 33000 means it belongs to label 33 (a car), is instance #0, while the pixel value of 33001 means it also belongs to class 33 (a car) , and is instance #1. These represent two different cars in an image.

# In[4]:


# Define the label mappings 
labelmap = {0:'others', 
            1:'rover', 
            17:'sky', 
            33:'car', 
            34:'motorbicycle', 
            35:'bicycle', 
            36:'person', 
            37:'rider', 
            38:'truck', 
            39:'bus', 
            40:'tricycle', 
            49:'road', 
            50:'siderwalk', 
            65:'traffic_cone', 
            66:'road_pile', 
            67:'fence', 
            81:'traffic_light', 
            82:'pole', 
            83:'traffic_sign', 
            84:'wall', 
            85:'dustbin', 
            86:'billboard', 
            97:'building', 
            98:'bridge', 
            99:'tunnel', 
            100:'overpass', 
            113:'vegatation', 
            161:'car_groups', 
            162:'motorbicycle_group', 
            163:'bicycle_group', 
            164:'person_group', 
            165:'rider_group', 
            166:'truck_group', 
            167:'bus_group', 
            168:'tricycle_group'}


# In[6]:


# Create an empty dataframe
data_df = pd.DataFrame()
df_list = []

# Iterate over data. I have just shown it for 500 images just to save time 
for idx in range(500):
    # Get the image name and corresponding label
    img_name = train_images[idx]
    label_name = train_labels[idx]
    label = imread(labels_dir / train_labels[idx])
    pixel_classes = np.unique(label//1000)
    classes, instance_count = np.unique(pixel_classes, return_counts=True) # Courtesy:https://www.kaggle.com/jpmiller/cvpr-eda
    data_dict = dict(zip(classes, instance_count))
    df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
    df.rename(columns=labelmap, inplace=True)
    df['img'] = img_name
    df['label'] = label_name
    
    # Concate to the final dataframe
    #data_df = pd.concat([data_df, df], copy=False)
    # append to the list of intermediate df list
    df_list.append(df)
    
data_df = pd.concat(df_list, axis=0)
del df_list

# Fill the NaN with zero
data_df = data_df.fillna(0)

# Rearrange the columns
cols = data_df.columns.tolist()
cols = [x for x in cols if x not in ['img', 'label']]
cols = ['img', 'label'] + cols
data_df = data_df[cols]

# Display the results
data_df = data_df.reset_index(drop=True)
data_df.head(10)    


# In[7]:


# Let's have a look at some of the images 
sample_images = (data_df['img'][300:305]).reset_index(drop=True)
sample_labels = (data_df['label'][300:305]).reset_index(drop=True)

f, ax = plt.subplots(5,3, figsize=(20,20))
for i in range(5):
    img = imread(images_dir / sample_images[i])
    label = imread(labels_dir / sample_labels[i]) // 1000
    label[label!=0] = 255
    blended_image = Image.blend(Image.fromarray(img), Image.fromarray(label).convert('RGB'), alpha=0.8)
    
    ax[i, 0].imshow(img, aspect='auto')
    ax[i, 0].axis('off')
    ax[i, 1].imshow(label, aspect='auto')
    ax[i, 1].axis('off')
    ax[i, 2].imshow(blended_image, aspect='auto')
    ax[i, 2].axis('off')
plt.show()


# That's it folks!! I hope you enjoyed it. I will try to update this as soon as possible. Please upvote if you liked it!! 
