
# coding: utf-8

# # Getting a meaning of the score
# In this notebook I will take the train masks and by using erode and dilate try to get a sense of what the different dice scores mean.

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
import glob

get_ipython().run_line_magic('matplotlib', 'inline')


# Let's get the train images

# In[7]:


img_names = glob.glob(os.path.join('..','input', 'train_masks', '*.gif'))


# In[ ]:


print(len(img_names))


# On a first step create a visualization of what we are going to do.

# In[13]:


from PIL import Image 

def read_gif_image(img_name):
    img = Image.open(img_name).convert('RGB')
    img_arr = np.asarray(img.getdata(), dtype=np.uint8)
    img_arr = img_arr.reshape(img.size[1], img.size[0], 3)
    return img_arr[:, :, 0]


# In[14]:


def visualize_erode_dilate(img_name, row, col, width=100):
    img = read_gif_image(img_name)
    vis = img.copy()/255
    kernel = np.ones((3,3),np.uint8)
    for i in range(1, 5):
        vis += cv2.erode(img, kernel, iterations=i)/255
        vis += cv2.dilate(img, kernel, iterations=i)/255
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.grid()
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(vis[row:row+width, col:col+width], cmap='viridis')
    plt.grid()
    plt.subplot(122)
    plt.imshow(img[row:row+width, col:col+width], cmap='viridis')
    plt.grid()


# In[20]:


visualize_erode_dilate(img_names[1], 1000, 1100, 100)


# Above we can see a detail of the erodes and dilates that we are going to use for computing the dice score.

# ## Computing dice score

# In[21]:


def get_dice_sums(img, n_iterations):
    """
    Erodes or dilates the img based on the number of iterations 
    and computes the sum of the image and of the product
    """
    kernel = np.ones((3,3),np.uint8)
    if n_iterations < 0:
        pred = cv2.erode(img, kernel, iterations=(- n_iterations))/255
    else:
        pred = cv2.dilate(img, kernel, iterations=(n_iterations))/255
    return np.sum(pred), np.sum(img*pred/255)


# In[22]:


def compute_img_dice_sums(img_name, n_iterations):
    """
    Returns the sum of original image, prediction and product of them 
    for the number of required iterations
    """
    img = read_gif_image(img_name)
    img_sum = np.sum(img/255)
    sum_array = np.zeros((n_iterations*2+1, 3))
    sum_array[:, 0] = img_sum
    for i, n in enumerate(range(-n_iterations, n_iterations+1)):
        ret = get_dice_sums(img, n)
        sum_array[i, 1] = ret[0]
        sum_array[i, 2] = ret[1]
    
    return sum_array


# In[23]:


def compute_images_dice(img_list, n_iterations):
    """
    Computes dice for all the images given and the number 
    of erodes and dilates required
    """
    sum_array = np.zeros((n_iterations*2+1, 3))
    for img_name in img_list:
        sum_array += compute_img_dice_sums(img_name, n_iterations)
    dice = sum_array[:,2]*2/(sum_array[:,1]+sum_array[:,0])
    return dice


# In[24]:


n_iterations = 3
plt.figure(figsize=(12, 6))
for i in range(5):
    dice = compute_images_dice(np.random.choice(img_names, 10), n_iterations)
    x = np.arange(-n_iterations, n_iterations + 1)
    plt.plot(x, dice)
plt.xlabel('Erode/dilate iterations')
plt.ylabel('Dice score');


# So we can see that if the difference between masks is of 1 pixel the score drops to 0.996  
# There is people already with that scores. 
# 
# I think that means that the data is extremely clean, otherwise it would be difficult to reach those scores.
