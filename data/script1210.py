
# coding: utf-8

# # Random item access from BSON file
# 
# Here is one of some other methods to access item without iterating through the whole BSON file
# Idea is to store offsets and lenghts of all items and seek/read from binary file.
# 
# Following code creates a dictionary with key indexing item `_id` and values `(offset, length)`. It takes around 3 mins to execute.
# 

# In[ ]:


import os
import sys
import numpy as np # linear algebra
import pandas as pd
import bson
import cv2
import matplotlib.pyplot as plt


# In[ ]:


INPUT_PATH = os.path.join('..', 'input')

import struct
from tqdm import tqdm_notebook

num_dicts = 7069896 # according to data page

IDS_MAPPING = {}

length_size = 4 # number of bytes decoding item length

with open(os.path.join(INPUT_PATH, 'train.bson'), 'rb') as f, tqdm_notebook(total=num_dicts) as bar:
    item_data = []
    offset = 0
    while True:        
        bar.update()
        f.seek(offset)
        
        item_length_bytes = f.read(length_size)     
        if len(item_length_bytes) == 0:
            break                
        # Decode item length:
        length = struct.unpack("<i", item_length_bytes)[0]

        f.seek(offset)
        item_data = f.read(length)
        assert len(item_data) == length, "%i vs %i" % (len(item_data), length)
        
        # Check if we can decode
        item = bson.BSON.decode(item_data)
        
        IDS_MAPPING[item['_id']] = (offset, length)        
        offset += length            
            
def get_item(item_id):
    assert item_id in IDS_MAPPING
    with open(os.path.join(INPUT_PATH, 'train.bson'), 'rb') as f:
        offset, length = IDS_MAPPING[item_id]
        f.seek(offset)
        item_data = f.read(length)
        return bson.BSON.decode(item_data)


# In[ ]:


CATEGORY_NAMES_DF = pd.read_csv(os.path.join(INPUT_PATH, 'category_names.csv'))
level_tags = CATEGORY_NAMES_DF.columns[1:]


def decode(data):
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

# Method to compose a single image from 1 - 4 images
def decode_images(item_imgs):
    nx = 2 if len(item_imgs) > 1 else 1
    ny = 2 if len(item_imgs) > 2 else 1
    composed_img = np.zeros((ny * 180, nx * 180, 3), dtype=np.uint8)
    for i, img_dict in enumerate(item_imgs):
        img = decode(img_dict['picture'])
        h, w, _ = img.shape        
        xstart = (i % nx) * 180
        xend = xstart + w
        ystart = (i // nx) * 180
        yend = ystart + h
        composed_img[ystart:yend, xstart:xend] = img
    return composed_img

item = get_item(1234)

mask = CATEGORY_NAMES_DF['category_id'] == item['category_id']    
cat_levels = CATEGORY_NAMES_DF[mask][level_tags].values.tolist()[0]
cat_levels = [c[:25] for c in cat_levels]
title = str(item['category_id']) + '\n'
title += '\n'.join(cat_levels)
plt.title(title)
plt.imshow(decode_images(item['imgs']))
_ = plt.axis('off')

