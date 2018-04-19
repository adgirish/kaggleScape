
# coding: utf-8

# ### Intro
# In this kernel I explore the dataset for the CVPR competition. We are given images from videos produced as a car drives around and records activity from the car's point of view. My primary purpose is to see what's in the images and get a general feel for the objects we are asked to segment. I'll also create a data frame of labels for the train set that can be used for more in-depth classification.
# 
# New: I've been trying to optimize the code for reading all train images and generating the labels. Maybe you have some ideas to speed it up?.

# In[22]:


import os
import random
import numpy as np
import pandas as pd 
from skimage import io, img_as_float
from numba import jit
from PIL import Image
import cv2
import tensorflow as tf
from IPython.display import display
import matplotlib.pyplot as plt
from tqdm import tqdm


# ### Files
# First let's quickly look at the default file structure. There are 3 directories and a sample submission.

# In[23]:


os.listdir('../input')


# Let's look first at the training images....

# In[24]:


def filecheck(dir):
    dir_size = 0
    filelist = os.listdir(dir)
    filelist.sort()
    print(dir)
    for i,name in enumerate(filelist):
        dir_size += os.path.getsize(os.path.join(dir, name))
    print("{:.1f} GB of {} files".format(dir_size/1024/1024/1024, i))
    print("showing sample files")
    print("\n".join(filelist[300:306]) + "\n")

dirs = ["../input/train_color","../input/train_label", "../input/test"]

for d in dirs[0:2]:
    filecheck(d)


# 92.3GB of image data for the train set images! The filenames are interesting. The prefixes of the files, "170908" might represent dates the pictures were taken, and the middle string might represent times and/or frame numbers. You can see that each "instance" - the middle set of numbers - has images from both Camera 5 and Camera 6. From what I've seen they always come in pairs like that, which probably means there are (at least) two cameras recording at the same time. As expected, each jpg image in the train set has a corresponding png image with the mask of objects to classify. 
# 
# Edit: This awesome kernel, [Recovering the Videos](https://www.kaggle.com/andrewrib/recovering-the-videos) string together sequential frames into videos.
# 
# Briefly looking at the test set files we see a more cyrptic naming convention. Our host refers to a "test video" in the Welcome post which suggests these files are also sequential somehow.

# In[25]:


j = os.listdir(dirs[2])
print("\n".join(j[0:6]))
print("{} files".format(len(j)))


# ### Images
# Jumping back to the training images - let's look at an image with labels.

# In[26]:


im = Image.open("../input/train_color/170908_061523257_Camera_5.jpg")
tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000
tlabel[tlabel != 0] = 255
plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))
display(plt.show())


# OK, so there are some vehicles and such on the road, just as we would expect. To see what those things are, we can look into the png file and extract the values emedded inside. From the Data page....
# 
# #### The training images labels are encoded in a format mixing spatial and label/instance information:
# 
# * All the images are the same size (width, height) of the original images
# * Pixel values indicate both the label and the instance.
# * Each label could contain multiple object instances.
# * int(PixelValue / 1000) is the label (class of object)
# * PixelValue % 1000 is the instance id
# * For example, a pixel value of 33000 means it belongs to label 33 (a car), is instance #0, while the pixel value of 33001 means it also belongs to class 33 (a car) , and is instance #1. These represent two different cars in an image.
# 
# Clever, eh? Let's look.

# In[27]:


classdict = {0:'others', 1:'rover', 17:'sky', 33:'car', 34:'motorbicycle', 35:'bicycle', 36:'person', 37:'rider', 38:'truck', 39:'bus', 40:'tricycle', 49:'road', 50:'siderwalk', 65:'traffic_cone', 66:'road_pile', 67:'fence', 81:'traffic_light', 82:'pole', 83:'traffic_sign', 84:'wall', 85:'dustbin', 86:'billboard', 97:'building', 98:'bridge', 99:'tunnel', 100:'overpass', 113:'vegatation', 161:'car_groups', 162:'motorbicycle_group', 163:'bicycle_group', 164:'person_group', 165:'rider_group', 166:'truck_group', 167:'bus_group', 168:'tricycle_group'}

tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png"))
cls = np.unique(tlabel)//1000
unique, counts = np.unique(cls, return_counts=True)
d = dict(zip(unique, counts))
df = pd.DataFrame.from_dict(d, orient='index').transpose()
df.rename(columns=classdict, inplace=True)
df


# According to the data we have 5 cars, a bus, a tricycle and a traffic cone (see note below for 'traffic cone'). OK, sure... Let's also look at Camera 6 for the same instance.
# 

# In[28]:


im = Image.open("../input/train_color/170908_073302618_Camera_6.jpg")
tlabel = np.asarray(Image.open("../input/train_label/170908_073302618_Camera_6_instanceIds.png"))//1000
tlabel[tlabel != 0] = 255
plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.6))

tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_6_instanceIds.png"))
cls = np.unique(tlabel)//1000
unique, counts = np.unique(cls, return_counts=True)
d = dict(zip(unique, counts))
df = pd.DataFrame.from_dict(d, orient='index').transpose()
df.rename(columns=classdict, inplace=True)

display(plt.show())
df


# Camera 6 shows a different view as we might expect. The [First Look](https://www.kaggle.com/aakashnain/firstlook) kernel has more instances of images and masks.

# ### Labels (masks)
# Let's now pull labels for the training images and look at some basic stats. (Note: The code below can probably be further optimized for better performance. I have it down to less than 1/2 of the original code. I'll still work with a sample here for now to save time.)

# In[29]:


allfilenames = os.listdir(dirs[1])
filenames = random.sample(allfilenames, 1000)
fullpaths = ["../input/train_label/" + f for f in filenames]
labarray = np.zeros((len(filenames), 66))

@jit
def getcounts():
    for i,f in enumerate(tqdm(fullpaths)):
        tlabel = io.imread(f, plugin='pil')
        cls = np.unique(tlabel)
        unique,counts = np.unique(cls//1000, return_counts=True)
        labarray[i, unique] = counts

getcounts()
labels_df = pd.DataFrame(labarray, index=filenames)
labels_df = labels_df.loc[:, (labels_df != 0).any(axis=0)]
labels_df.rename(columns=classdict, inplace=True)        
labels_df.head(6)


# We can look at the frequency of classes in the images by summing the occurrences across all images.

# In[31]:


labels_df.drop('others', axis=1, inplace=True)
classes_df = pd.melt(labels_df)
groups = classes_df.groupby('variable')
sums = groups.sum()


plt.figure();
sums.plot.bar()


# The most prevalent class is cars by far, as you might expect. I find it curious that most of the classes are not represented anywhere. Assuming the code is correct, it could be due to the limited sample, or the classes may be extremely rare. 
# 
# Edit: as pointed out in [this discussion](https://www.kaggle.com/c/cvpr-2018-autonomous-driving/discussion/53845), only 7 of the classes will be used for evaluation. These are car, motorbicycle, bicycle, person, truck, bus, and tricycle. Class 65, traffic cone, is actually a false label, It comes from pixel value 65535 which represents the "ignoring label".
# 
# Anyway, let's look at the differences among images. Here are histograms of Total Objects per image and Distinct Classes per image.

# In[32]:


labels_df['objects'] = labels_df.sum(axis=1)
labels_df['classes'] = labels_df[labels_df>0].count(axis=1)-1
labels_df.head()


# In[33]:


plt.figure();
plt.title("Total # of Objects")
labels_df['objects'].plot.hist()

plt.figure();
plt.title("# of Distinct Classes")
labels_df['classes'].plot.hist()


# It's interesting to see quite a difference among the images, expecially for total counts. 
# 
# There's a lot more that can be done with this data, of course, and I look forward to seeing some great kernels!
# 
# 
