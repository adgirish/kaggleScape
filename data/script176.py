
# coding: utf-8

# This script will create one folder per category and save all respective images on it.
# 
# The pattern to each file is  `../input/train/[category]/[_id]-[index].jpg`, where `index` is the position of image on each product.

# In[1]:


import bson
import numpy as np
import pandas as pd
import os
from tqdm import tqdm_notebook


# In[2]:


out_folder = '../output/train'

# Create output folder
if not os.path.exists(out_folder):
    os.makedirs(out_folder)


# In[3]:


# Create categories folders
categories = pd.read_csv('../input/category_names.csv', index_col='category_id')

for category in tqdm_notebook(categories.index):
    os.mkdir(os.path.join(out_folder, str(category)))


# In[4]:


num_products = 7069896  # 7069896 for train and 1768182 for test

bar = tqdm_notebook(total=num_products)
with open('../input/train.bson', 'rb') as fbson:

    data = bson.decode_file_iter(fbson)
    
    for c, d in enumerate(data):
        category = d['category_id']
        _id = d['_id']
        for e, pic in enumerate(d['imgs']):
            fname = os.path.join(out_folder, str(category), '{}-{}.jpg'.format(_id, e))
            with open(fname, 'wb') as f:
                f.write(pic['picture'])

        bar.update()

