
# coding: utf-8

# # Fast benchmark: Pillow vs OpenCV
# 
# *Background: when we deal with images in image-based problems and deploy a deep learning solution, it is better to have a fast image reading and transforming library. Let's compare Pillow and OpenCV python libraries on image loading and some basic transformations on source images from Carvana competition.*
# 
# [OpenCV](https://github.com/opencv/opencv): C++, python-wrapper
# 
# [Pillow](https://github.com/python-pillow/Pillow): Python, C
# 
# `
# `
# 
# Intuition says that Opencv should be a little faster, let's see this by examples
# 
# `
# `
# 
# *This question I asked myself after reading the PyTorch [documentation on image transformation](http://pytorch.org/docs/0.2.0/_modules/torchvision/transforms.html). Most of transformations take as input a PIL image.*
# 

# In[ ]:


import PIL
import cv2


# At first, let's get packages versions, specs and some info on the machine

# In[ ]:


print(cv2.__version__, cv2.__spec__)
print(cv2.getBuildInformation())


# In[ ]:


PIL.__version__, PIL.__spec__


# In[ ]:


get_ipython().system('cat /proc/cpuinfo | egrep "model name"')


# Data storage info: `ROTA 1` means rotational device

# In[ ]:


get_ipython().system('lsblk -o name,rota,type,mountpoint')


# Now let's setup the input data

# In[ ]:


import os
this_path = os.path.dirname('.')

INPUT_PATH = os.path.abspath(os.path.join(this_path, '..', 'input'))
TRAIN_DATA = os.path.join(INPUT_PATH, "train")
from glob import glob
filenames = glob(os.path.join(TRAIN_DATA, "*.jpg"))
len(filenames)


# In[ ]:


import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1 stage: 100 images, load image + blur + flip

# In[ ]:


import numpy as np
from PIL import Image, ImageOps

def stage_1_PIL(filename):
    img_pil = Image.open(filename)
    img_pil = ImageOps.box_blur(img_pil, radius=1)
    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    return np.asarray(img_pil)

def stage_1_cv2(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.blur(img, ksize=(3, 3))
    img = cv2.flip(img, flipCode=1)
    return img


# Let's compare briefly results of transformations on the first image. Results are not perfectly the same, but it is not important for the benchmark  

# In[ ]:


f = filenames[0]
r1 = stage_1_PIL(f) 
r2 = stage_1_cv2(f)

plt.figure(figsize=(16, 16))
plt.subplot(131)
plt.imshow(r1)
plt.subplot(132)
plt.imshow(r2)
plt.subplot(133)
plt.imshow(np.abs(r1 - r2))


# In[ ]:


get_ipython().run_line_magic('timeit', '-n5 -r3 [stage_1_PIL(f) for f in filenames[:100]]')


# In[ ]:


get_ipython().run_line_magic('timeit', '-n5 -r3 [stage_1_cv2(f) for f in filenames[:100]]')


# ## 1b stage: 100 images, blur + flip

# In[ ]:


def stage_1b_PIL(img_pil):
    img_pil = ImageOps.box_blur(img_pil, radius=1)
    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    return np.asarray(img_pil)

def stage_1b_cv2(img):    
    img = cv2.blur(img, ksize=(3, 3))
    img = cv2.flip(img, flipCode=1)
    return img


# In[ ]:


imgs_PIL = [Image.open(filename) for filename in filenames[:100]]


# In[ ]:


def cv2_open(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

imgs_cv2 = [cv2_open(filename) for filename in filenames[:100]]


# In[ ]:


get_ipython().run_line_magic('timeit', '-n5 -r3 [stage_1b_PIL(img_pil) for img_pil in imgs_PIL]')


# In[ ]:


get_ipython().run_line_magic('timeit', '-n5 -r3 [stage_1b_cv2(img) for img in imgs_cv2]')


# ## 2 stage: 500 images, load image + resize + 2 flips

# In[ ]:


import numpy as np
from PIL import Image, ImageOps


def stage_2_PIL(filename):
    img_pil = Image.open(filename)
    img_pil = img_pil.resize((512, 512), Image.CUBIC)
    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
    return np.asarray(img_pil)

def stage_2_cv2(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    img = cv2.flip(img, flipCode=1)
    img = cv2.flip(img, flipCode=0)
    return img


# Again let's compare briefly results of transformations on the first image:

# In[ ]:


f = filenames[0]
r1 = stage_2_PIL(f) 
r2 = stage_2_cv2(f)

plt.figure(figsize=(16, 16))
plt.subplot(131)
plt.imshow(r1)
plt.subplot(132)
plt.imshow(r2)
plt.subplot(133)
plt.imshow(np.abs(r1 - r2))


# In[ ]:


get_ipython().run_line_magic('timeit', '-n5 -r3 [stage_2_PIL(f) for f in filenames[:200]]')


# In[ ]:


get_ipython().run_line_magic('timeit', '-n5 -r3 [stage_2_cv2(f) for f in filenames[:200]]')

