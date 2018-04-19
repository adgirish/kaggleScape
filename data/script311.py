
# coding: utf-8

# [<img src='https://lh3.googleusercontent.com/-tNe1vwwd_w4/VZ_m9E44C7I/AAAAAAAAABM/5yqhpSyYcCUzwHi-ti13MwovCb_AUD_zgCJkCGAYYCw/w256-h86-n-no/Submarineering.png'>](https://twitter.com/submarineering?lang=en)
# 

# **The main purpose of this Notebook is to apply image processing technics in order to provide some additional engineering features to help on the improvement of the classifier accuracy. ** 
# 
# I highly recommend to read and see some examples about image processing : 
# 
# http://scikit-image.org/
# 
# And my Notebooks  '**Submarineering.Size matters**', '**Submarineering.Objects solation**' :
# 
# https://www.kaggle.com/submarineering/submarineering-size-matters
# 
# https://www.kaggle.com/submarineering/submarineering-objects-isolation-0-75-lb
# 
# What can you learn? 
# 
# -An easy way to compare graphically the influency of differents attributes, and plot a 3d surface from the images using Plotly. 
# 
# -Undertanding  that the cleaning of data is fundamental for the classifier, as the learning process is automatic, unnecessary data will confuse to the algorithm. 
# 
# -Doesn't  matter which classifier or different algorithm you are going to use. This is always important.
# 
# -In this case I am focusing on the isolation of the object and calculate his volume . 
# 
# -The info provides by the water is irrelevant.
# 
# -**As a bonus, at the end, I explain how to generate useful features as result of the morphological analysis. ** 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.morphology import reconstruction
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train = pd.read_json('../input/train.json')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# load the training dataset.
train = pd.read_json('../input/train.json')


# In[ ]:


# Isolation function.
def iso(arr):
    image = img_as_float(np.reshape(np.array(arr), [75,75]))
    image = gaussian_filter(image,2.5)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image 
    dilated = reconstruction(seed, mask, method='dilation')
    return image-dilated


# In[ ]:


# Plotting to compare
arr = train.band_1[12]
dilated = iso(arr)
fig, (ax0, ax1) = plt.subplots(nrows=1,
                                    ncols=2,
                                    figsize=(16, 5),
                                    sharex=True,
                                    sharey=True)

ax0.imshow(np.reshape(np.array(arr), [75,75]))
ax0.set_title('original image')
ax0.axis('off')
ax0.set_adjustable('box-forced')

ax1.imshow(dilated, cmap='gray')
ax1.set_title('dilated')
ax1.axis('off')
ax1.set_adjustable('box-forced')


# In[ ]:


# Plotting to compare
arr = train.band_1[8]
dilated = iso(arr)
fig, (ax0, ax1) = plt.subplots(nrows=1,
                                    ncols=2,
                                    figsize=(16, 5),
                                    sharex=True,
                                    sharey=True)

ax0.imshow(np.reshape(np.array(arr), [75,75]))
ax0.set_title('original image')
ax0.axis('off')
ax0.set_adjustable('box-forced')

ax1.imshow(dilated, cmap='gray')
ax1.set_title('dilated')
ax1.axis('off')
ax1.set_adjustable('box-forced')


# In[ ]:


# Feature engineering iso1 and iso2.
train['iso1'] = train.iloc[:, 0].apply(iso)
train['iso2'] = train.iloc[:, 1].apply(iso)


# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
img = train.iso1[12]+train.iso2[12]
data = [
    go.Surface(
        z=img
    )
]
layout = go.Layout(
    title='Iceberg to 3D Surface',
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


img = train.iso1[8] + train.iso2[8]
data = [
    go.Surface(
        z= img
    )
]
layout = go.Layout(
    title='Ship to 3D Surface',
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


#Adding up every pixel value is equivalent to the volumen under the surface. 
def volume(arr):
    return np.sum(arr)


# In[ ]:


train['vol'] = (train.iloc[:, 0] + train.iloc[:, 1]).apply(volume)


# In[ ]:


# Additional features from the morphological analysis and how is working on discrimination.
train[train.is_iceberg==1]['iso1'].apply(np.max).plot(alpha=0.4)
train[train.is_iceberg==0]['iso1'].apply(np.max).plot(alpha=0.4)


# In[ ]:


# Additional features from the morphological analysis and how is working on discrimination.
train[train.is_iceberg==1]['iso2'].apply(np.max).plot(alpha=0.4)
train[train.is_iceberg==0]['iso2'].apply(np.max).plot(alpha=0.4)


# In[ ]:


# Volume. Additional features from the morphological analysis and how is working on discrimination.
train[train.is_iceberg==1]['vol'].plot(alpha=0.4)
train[train.is_iceberg==0]['vol'].plot(alpha=0.4)


# **NOTE :** It looks like images with incidence angles having less than or equal to 4 decimal are the naturally captured images, and those with greater precision are machine generated, as 'brassmonkey' describes very well. 
# In the data description of the competition is also refered as : 
# "Please note that we have included machine-generated images in the test set to prevent hand labeling. They are excluded in scoring."
# This is an important point to be in mind. 
# 

# **Conclusion.** As described in my Notebooks  **'Submarineering.Size matters** and **'Submarineering.Objects isolation**'', that I highly recommed :
# 
# https://www.kaggle.com/submarineering/submarineering-size-matters 
# 
# https://www.kaggle.com/submarineering/submarineering-objects-isolation-0-75-lb
# 
# the size can be used from multiple points of view.  Also the shape of the object to detect can give us a rich information about his class. Additional features can be obtain from these morphological properties, like the** Volume** ,that it clearly can help to the classification too.
# 
# These features could be improved :
# 
# -They can be categorized in order to help on the accuracy of the Classifier.
# 
# I hope these lines be useful for your. **Please vote up**.
# 
