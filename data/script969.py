
# coding: utf-8

# For everybody's convenient, I converted the annotated nuclei from https://nucleisegmentationbenchmark.weebly.com/ into a public dataset following the folder structure of the competition data. I hope that you find it usefull.. 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
from matplotlib import pyplot as plt
TRAIN_PATH = '../input/bowl2018-external/extra_data/extra_data/'
train_ids = [x for x in os.listdir(TRAIN_PATH) if os.path.isdir(TRAIN_PATH+x)]


# In[10]:


df = pd.DataFrame({'id':train_ids,'train_or_test':'train'})
df['path'] = df.apply(lambda x:TRAIN_PATH +'/{}/images/{}.tif'.format(x[0],x[0]), axis=1)
df['masks'] = df.apply(lambda x:TRAIN_PATH +'/{}/masks/'.format(x[0],x[0]), axis=1)
df.head()


# There are 29 Images are annotated by the following ids:

# In[11]:


df['id']


# Let us load one image and its masks:

# In[12]:


imid = 'TCGA-G9-6362-01Z-00-DX1'
image_path = df[df.id==imid].path.values[0]
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.show()


# Now we can read  the masks for the specific image. We have stored them as png files`. 

# In[13]:


mask_dir = df[df.id==imid].masks.values[0]
masks = os.listdir(mask_dir)
masks[:10]


# In[14]:


mimgs = []
i = 0
for mask in masks:
    mimg = cv2.imread(mask_dir + '/' + mask)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_BGR2GRAY)
    mimg[mimg==255] = 1
    mimgs.append(mimg)
    i = i + 1
print('Read ' + str(i) + ' masks for image ' + df[df.id==imid].id.values[0] )
mimgs = np.array(mimgs)
plt.figure(figsize=(8, 8))
total_mask = np.sum(mimgs, axis=0)
plt.imshow(total_mask)
plt.show()


# To make sure that masks are disconnected we can add them all into a single mask image using `np.sum` and whenever we find two or more pixels in common, assign them to background (0). In the example below we zoom into a 200x200 pixel window.

# In[15]:


plt.figure(figsize=(8, 8))
total_mask = np.sum(mimgs, axis=0)
plt.title('Mask with overlapps: ' + str(np.max(total_mask)) + ' masks overlap' )
plt.imshow(total_mask[400:600, 200:400])
plt.show()
plt.figure(figsize=(8, 8))
total_mask[total_mask>1] = 0
plt.title('Mask with NO overlapps:' + str(np.max(total_mask))  + ' masks overlap')
plt.imshow(total_mask[400:600, 200:400])
plt.show()


# For the same window we superimpose the masks above the image.

# In[16]:


img2 = cv2.bitwise_and(image[400:600, 200:400],image[400:600, 200:400],mask = total_mask[400:600, 200:400].astype(np.uint8))
plt.figure(figsize=(8, 8))
total_mask[total_mask>1] = 0
plt.title('Masks over image')
plt.imshow(img2)
plt.show()

