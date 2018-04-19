
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df_train = pd.read_csv('../input/train_masks.csv')
print('Shape of training data: ' + str(df_train.shape) + '\n')
print(df_train.head())


# All our training data is in one column, titled `pixels`.
# 
# The Data page states: *train_masks.csv gives the training image masks in run-length encoded format.* So, we need to segment this data into something we can plot!

# In[ ]:


pixels = df_train['pixels'].values
print(pixels[0])
p1 = [] # Run-start pixel locations
p2 = [] # Run-lengths
p3 = [] # Number of data points per image

# Separate run-lengths and pixel locations into seperate lists
for p in pixels:
    x = str(p).split(' ')
    i = 0
    for m in x:
        if i % 2 == 0:
            p1.append(m)
        else:
            p2.append(m)
        i += 1
        
# Get number of data points in each image
i = 0
for p in pixels:
    x = str(p).split(' ')
    if len(x) == 1:
        p3.append(0)
    else:
        p3.append(len(x)/2)
    i += 1

# Get all absolute target values
targets = []
for start, length in zip(p1, p2):
    i = 0
    length = int(length)
    if start != 'nan':
        pix = int(start)
        while i <= length:
            targets.append(pix)
            pix += 1
            i += 1
        
print('\nTotal number of target pixels: ' + str(len(targets)))

# Remove NaNs
p4 = []
i = 0
for p in p1:
    if p == 'nan':
        i += 1
    else:
        p4.append(p)
p1 = p4
print('\nNumber of NaN in pixel locations: ' + str(i))


# We have seperated the run-length from pixel location, removed NaN data (where there is no data) and also got the number of data points in each image.
# 
# Now we can plot some graphs!

# In[ ]:


print('Number of pixel locations: ' + str(len(p1)))
print('    Number of run lengths: ' + str(len(p2)))
print('\nAverage number of pixel locations per image: ' + str(len(p1) / len(df_train.index)))


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.colors import LogNorm

p = np.array(p2).astype(int)
plt.hist(p, 25, normed=1, facecolor='red', alpha=0.75)
plt.title('Histogram of run-lengths')
plt.xlabel('Run length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

p = np.array(p1).astype(int)
plt.hist(p, 50, normed=1, facecolor='blue', alpha=0.75)
plt.title('Histogram of pixel location')
plt.xlabel('Pixel location')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

p = np.array(p3).astype(int)
plt.hist(p, 50, normed=1, facecolor='green', alpha=0.75)
plt.title('Histogram of data point count')
plt.xlabel('Number of data points in image')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

print("Now let's remove the images with zero target pixels and check again")

px = []
for x in p.tolist():
    if x != 0:
        px.append(x)
        
p = np.array(px).astype(int)
plt.hist(p, 50, normed=1, facecolor='green', alpha=0.75)
plt.title('Histogram of data point count')
plt.xlabel('Number of data points in image')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# ###Image analysis
# 
# First, we are going to convert these pixel values in the training data into X and Y positions on the photos so that we can see the most common positions. To do this, we need to get the dimensions of images.
# 
# I'm not going to print out the images in this script, if you want to see that [**click here**](https://www.kaggle.com/wcukierski/ultrasound-nerve-segmentation/plot-images)

# In[ ]:


import glob, os, cv2
ultrasounds = [img for img in glob.glob("../input/train/*.tif") if 'mask' not in img]

img = cv2.imread(ultrasounds[0])
height, width, channels = img.shape
print('Image dimensions: ' + str(height) + 'h x ' + str(width) + 'w - ' + str(channels) + ' channels')


# In[ ]:


xs = []
ys = []

for p in targets:
    p = int(p)
    xs.append(p % 580)
    ys.append(int(p / 580)) # int() helpfully rounds down
    
bins = 40
while bins <=320:
    plt.hist2d(xs, ys, bins=bins, norm=LogNorm())
    plt.colorbar()
    plt.title('Target pixel location histogram - ' + str(bins) + ' bins')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    bins = bins * 2

