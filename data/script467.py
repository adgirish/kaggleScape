
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_json("../input/train.json")


# In[ ]:


test = pd.read_json("../input/test.json")


# ### SHOW SINGLE IMAGES

# In[ ]:


train.head()


# In[ ]:


train.inc_angle = train.inc_angle.apply(lambda x: np.nan if x == 'na' else x)


# In[ ]:


img1 = train.loc[20, ['band_1', 'band_2']]


# In[ ]:


# 1st 2nd and 3rd 
img1 = np.stack([img1['band_1'], img1['band_2']], -1).reshape(75, 75, 2)


# In[ ]:


#### band 1
plt.imshow(img1[:, :, 0] )


# In[ ]:


# band 2
plt.imshow(img1[:, :, 1])


# In[ ]:


combined = img1[:, :, 0] / img1[:, :, 1]


# In[ ]:


r = img1[:, :, 0]
r = (r + abs(r.min())) / np.max((r + abs(r.min())))

g = img1[:, :, 1]
g = (g + abs(g.min())) / np.max(g + abs(g.min()))

b = img1[:, :, 0] / img1[:, :, 1]
b = (((b) / np.max(b)) + abs((b) / np.max(b))) / np.max((((b) / np.max(b)) + abs((b) / np.max(b))))


# In[ ]:


plt.imshow(np.dstack((r, g, b)))


# ### GET COLOR COMPOSITE

# In[ ]:


def color_composite(data):
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

        rgb = np.dstack((r, g, b))
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)


# In[ ]:


rgb_train = color_composite(train)


# In[ ]:


rgb_test = color_composite(test)


# In[ ]:


rgb_train.shape


# In[ ]:


rgb_test.shape


# ### LOOKING AT RANDOM SHIPS AND ICEBERGS AFTER COLOR COMPOSITE
# 
# - Color composition might allow us to identify different characteristics of images
# - Of course brightness of the images is effected by incidence angle but we will deal with it later
# - My general conclusion after color composition was that icebergs are showing more coherent and similar patterns within themselves where as ships are varying more

# In[ ]:


# look at random ships
print('Looking at random ships')
ships = np.random.choice(np.where(train.is_iceberg ==0)[0], 9)
fig = plt.figure(1,figsize=(15,15))
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    arr = rgb_train[ships[i], :, :]
    ax.imshow(arr)
    
plt.show()


# In[ ]:


# look at random icebergs
print('Looking at random icebergs')
icebergs = np.random.choice(np.where(train.is_iceberg ==1)[0], 9)
fig = plt.figure(1,figsize=(15,15))
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    arr = rgb_train[icebergs[i], :, :]
    ax.imshow(arr)
    
plt.show()


# ### TEST SAMPLES

# In[ ]:


idx = np.random.choice(range(0, len(test)), 9)
test_img = color_composite(test.iloc[idx])

# look at random icebergs
print('Looking at random test images')
fig = plt.figure(1,figsize=(15,15))
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    arr = test_img[i, :, :]
    ax.imshow(arr)
    
plt.show()


# ### SAVE NEW IMAGES 
# 
# This might be useful for dataloaders reading data from folders as train valid and test

# In[ ]:


os.makedirs('./data/composites', exist_ok= True)
os.makedirs('./data/composites/train', exist_ok=True)
os.makedirs('./data/composites/valid', exist_ok=True)
os.makedirs('./data/composites/test', exist_ok=True)

train_y, test_y = train_test_split(train.is_iceberg, test_size=0.3)
train_index, test_index = train_y.index, test_y.index

#save train images
for idx in train_index:
    img = rgb_train[idx]
    plt.imsave('./data/composites/train/' + str(idx) + '.jpg',  img)

#save valid images
for idx in test_index:
    img = rgb_train[idx]
    plt.imsave('./data/composites/valid/' + str(idx) + '.jpg',  img)

#save test images
for idx in range(len(test)):
    img = rgb_test[idx]
    plt.imsave('./data/composites/test/' + str(idx) + '.jpg',  img)


# 
# ### REFERENCES
# 
# https://spectraldifferences.wordpress.com/2015/05/06/create-colour-composites-for-alos-palsar-tiles/

# ### NEXT: CHECK THE INCIDENCE ANGLE TRANSFORMATION
