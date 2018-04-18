
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import skimage.io
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


input_dir = '../input'
train_dir = '../input/stage1_train'


# In[3]:


df_labels = pd.read_csv(os.path.join(input_dir, 'stage1_train_labels.csv'))


# # visualize random image

# In[4]:


def show_images(image_ids):
  plt.close('all')
  fig, ax = plt.subplots(nrows=len(image_ids),ncols=3, figsize=(50,50))

  for image_idx, image_id in enumerate(image_ids):
    image_path = os.path.join(train_dir, image_id, 'images', '{}.png'.format(image_id))
    mask_paths = os.path.join(train_dir, image_id, 'masks', '*.png')
  
    image = skimage.io.imread(image_path)
    masks = skimage.io.imread_collection(mask_paths).concatenate()
    mask = np.zeros(image.shape[:2], np.uint16)
    for mask_idx in range(masks.shape[0]):
      mask[masks[mask_idx] > 0] = mask_idx + 1
    other = mask == 0
    
    if len(image_ids) > 1:
      ax[image_idx, 0].imshow(image)
      ax[image_idx, 1].imshow(mask)
      ax[image_idx, 2].imshow(np.expand_dims(other, axis=2) * image)
    else:
      ax[0].imshow(image)
      ax[1].imshow(mask)
      ax[2].imshow(np.expand_dims(other, axis=2) * image)


# In[5]:


sample_image = df_labels.sample(n=1).iloc[0]
show_images([sample_image['ImageId']])


# # Statistics

# In[6]:


def get_nuclei_sizes():
  image_ids = list(df_labels.drop_duplicates(subset='ImageId')['ImageId'])
  def nuclei_size_stats(image_id):
    mask_paths = os.path.join(train_dir, image_id, 'masks', '*.png')
    masks = skimage.io.imread_collection(mask_paths).concatenate()
    masks = (masks > 0).astype(np.uint16)
    nuclei_sizes = np.sum(masks, axis=(1,2))
    return {'nuclei_size_min': np.min(nuclei_sizes),
            'nuclei_size_max': np.max(nuclei_sizes),
            'nuclei_size_mean': np.mean(nuclei_sizes),
            'nuclei_size_std': np.std(nuclei_sizes)}
  return pd.DataFrame.from_dict({image_id: nuclei_size_stats(image_id) for image_id in image_ids}, orient='index')


# In[7]:


df_nuclei_sizes = get_nuclei_sizes()


# In[8]:


def plot_stats(df):
  fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(64,64))
  def plot_with_set_font_size(key, ax):
    p = sns.distplot(df_stats[key], kde=False, rug=False, ax=ax)
    p.tick_params(labelsize=50)
    p.set_xlabel(key, fontsize=50)
  
  plot_with_set_font_size('mask_counts', axs[0,0])
  plot_with_set_font_size('nuclei_size_min', axs[1,0])
  plot_with_set_font_size('nuclei_size_max', axs[1,1])
  plot_with_set_font_size('nuclei_size_mean', axs[2,0])
  plot_with_set_font_size('nuclei_size_std', axs[2,1])


# In[9]:


df_mask_counts = df_labels.groupby(['ImageId']).count()
df_mask_counts.columns = ['mask_counts']
df_stats = df_mask_counts.join(df_nuclei_sizes)

display(df_stats.describe())
plot_stats(df_stats)


# ## lower percentile of mask counts

# In[10]:


lower_percentile_stats = df_stats.query('mask_counts < 15.25')
display(lower_percentile_stats.describe())
plot_stats(lower_percentile_stats)


# In[11]:


samples = lower_percentile_stats.sample(n=5)
display(samples)
show_images(list(samples.index))


# ## middle percentile of mask counts

# In[12]:


middle_percentile_stats = df_stats.query('mask_counts > 15.25 and mask_counts < 54.0')
display(middle_percentile_stats.describe())
plot_stats(middle_percentile_stats)


# In[13]:


samples = middle_percentile_stats.sample(n=5)
display(samples)
show_images(list(samples.index))


# ## upper percentile of mask counts

# In[14]:


upper_percentile_stats = df_stats.query('mask_counts > 54.0')
display(upper_percentile_stats.describe())
plot_stats(upper_percentile_stats)


# In[15]:


samples = upper_percentile_stats.sample(n=5)
display(samples)
show_images(list(samples.index))

