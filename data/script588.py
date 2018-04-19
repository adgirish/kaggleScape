
# coding: utf-8

# # Plant Seedlings Segmentation with pure Computer Vision

# First of all, thanks for the popularity of this kernel. I hope it will help for you to create more accurate predictions

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from glob import glob
import seaborn as sns


# In[2]:


BASE_DATA_FOLDER = "../input"
TRAin_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "train")


# ### Read images
# First, I'll just read all the images. The images are in BGR (Blue/Green/Red) format because OpenCV uses this.
# 
# Btw... If you'd like to use RGB format, than you can use it, it won't effect the segmentation because we will use the HSV (Hue/Saturation/Value) color space for that.

# In[3]:


images_per_class = {}
for class_folder_name in os.listdir(TRAin_DATA_FOLDER):
    class_folder_path = os.path.join(TRAin_DATA_FOLDER, class_folder_name)
    class_label = class_folder_name
    images_per_class[class_label] = []
    for image_path in glob(os.path.join(class_folder_path, "*.png")):
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        images_per_class[class_label].append(image_bgr)


# ### Number of images per class

# In[4]:


for key,value in images_per_class.items():
    print("{0} -> {1}".format(key, len(value)))


# ### Plot images
# Plot images so we can see what the input looks like

# In[5]:


def plot_for_class(label):
    nb_rows = 3
    nb_cols = 3
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(6, 6))

    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            axs[i, j].xaxis.set_ticklabels([])
            axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].imshow(images_per_class[label][n])
            n += 1        


# In[6]:


plot_for_class("Small-flowered Cranesbill")


# In[7]:


plot_for_class("Maize")


# ### Preprocessing for the images:
# 
# Now comes the interesting and fun part!
# 
# I created separate functions so if you'd like to use these it is easier.
# 
# In the next block I'll explain what I am doing to make the segmentation happen.

# In[8]:


def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


# The `create_mask_for_plant` function: This function returns an image mask: Matrix with shape `(image_height, image_width)`. In this matrix there are only `0` and `1` values. The 1 values define the interesting part of the original image. But the question is...How do we create this mask?
# 
# This is a simple object detection problem, where we can use the color of the object.
# 
# The HSV color-space is suitable for color detection because with the Hue we can define the color and the saturation and value will define "different kinds" of the color. (For example it will detect the red, darker red, lighter red too). We cannot do this with the original BGR color space.
# 
# ![](https://www.mathworks.com/help/images/hsvcone.gif)
# 
# *image from https://www.mathworks.com/help/images/convert-from-hsv-to-rgb-color-space.html*
# 
# We have to set a range, which color should be detected:
# 
#     sensitivity = 35
#     lower_hsv = np.array([60 - sensitivity, 100, 50])
#     upper_hsv = np.array([60 + sensitivity, 255, 255])
#     
# After the mask is created with the `inRange` function, we can do a little *CV magic* (not close to magic, because this is almost the most basic thing in CV, but it is a cool buzzword, and this opertation is as awesome as simple it is) which is called *morphological operations* ([You can read more here](https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic4.htm)).
# 
# Basically with the *Close* operation we would like to keep the shape of the original objects (1 blobs on the mask image) but close the small holes. That way we can clarify our detection mask more.
# 
# ![](https://homepages.inf.ed.ac.uk/rbf/HIPR2/figs/closebin.gif)
# 
# *image from https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic4.htm*
# 
# After these steps we created the mask for the object.
# 

# In[9]:


# Test image to see the changes
image = images_per_class["Small-flowered Cranesbill"][97]

image_mask = create_mask_for_plant(image)
image_segmented = segment_plant(image)
image_sharpen = sharpen_image(image_segmented)

fig, axs = plt.subplots(1, 4, figsize=(20, 20))
axs[0].imshow(image)
axs[1].imshow(image_mask)
axs[2].imshow(image_segmented)
axs[3].imshow(image_sharpen)


# After this step we can see that the image on the right is more recognizable than the original image on the left.

# ----------------------------------------------

# From the mask image what we created (because we need that for the segmentation), we can extract some features. For example we can see how the area of the plant changes based on their classes.

# Of course from the contours we can extract much more information than the area of the
# contour and the number of components, but this is the one I would like to show you.
# 
# Additional read: https://en.wikipedia.org/wiki/Image_moment

# In[10]:


def find_contours(mask_image):
    return cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

def calculate_largest_contour_area(contours):
    if len(contours) == 0:
        return 0
    c = max(contours, key=cv2.contourArea)
    return cv2.contourArea(c)

def calculate_contours_area(contours, min_contour_area = 250):
    area = 0
    for c in contours:
        c_area = cv2.contourArea(c)
        if c_area >= min_contour_area:
            area += c_area
    return area


# In[11]:


areas = []
larges_contour_areas = []
labels = []
nb_of_contours = []
images_height = []
images_width = []

for class_label in images_per_class.keys():
    for image in images_per_class[class_label]:
        mask = create_mask_for_plant(image)
        contours = find_contours(mask)
        
        area = calculate_contours_area(contours)
        largest_area = calculate_largest_contour_area(contours)
        height, width, channels = image.shape
        
        images_height.append(height)
        images_width.append(width)
        areas.append(area)
        nb_of_contours.append(len(contours))
        larges_contour_areas.append(largest_area)
        labels.append(class_label)


# In[12]:


features_df = pd.DataFrame()
features_df["label"] = labels
features_df["area"] = areas
features_df["largest_area"] = larges_contour_areas
features_df["number_of_components"] = nb_of_contours
features_df["height"] = images_height
features_df["width"] = images_width


# In[13]:


features_df.groupby("label").describe()

