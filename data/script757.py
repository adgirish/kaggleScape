
# coding: utf-8

# In[1]:


import os
from glob import glob
from tqdm import tqdm
import pandas as pd
from skimage.io import imread
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot') 


# In[2]:


img_paths = glob(os.path.join('../input/train', '*.jpg'))
gt_dir = '../input/train_masks'

y = []
for i in tqdm(range(len(img_paths[:1000]))):  
    img_path = img_paths[i]    
    gt = imread(os.path.join(gt_dir, os.path.splitext(os.path.basename(img_path))[0]+'_mask.gif'))
    y.append(gt)
    
y = np.array(y)


# In[5]:


scales = np.array(range(1, 20)) * 0.05
mean_dices = []

for scale in scales:
    
    dices = []
    for i in tqdm(range(len(y))):
        
        mask = y[i]
        seg = cv2.resize(mask, dsize=None, fx=scale, fy=scale)
        seg = cv2.resize(seg, (1918, 1280))
        
        mask = mask > 127
        seg = seg > 127
        
        dice = 2.0 * np.sum(seg&mask) / (np.sum(seg) + np.sum(mask))
        dices.append(dice)
    
    dices = np.array(dices)
    mean_dices.append(np.mean(dices))

mean_dices = np.array(mean_dices)
        


# In[10]:


plt.figure(figsize=(15, 7))
plt.plot(scales, mean_dices)
plt.xlabel('scale')
plt.ylabel('dice')
for i in range(1, len(scales), 2):
    plt.text(scales[i], mean_dices[i], '%.5f'%mean_dices[i])
plt.show()

