
# coding: utf-8

# In this kernel I provide an Exploratory Data Analyis for the Iceberg classification challenge and some snippets showing how to generate new data to agument the initial dataset.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
train = pd.read_json('../input/train.json')
train.head()


# Let's check if the two sets, icebergs and ships, are balanced

# In[ ]:


f,ax = plt.subplots(1,1,figsize=(15,6))
sns.barplot(x=['not an iceberg','iceberg'],y=train.groupby(['is_iceberg'],as_index=False).count()['id'])
plt.show()


# Here you can see the distribution of the angles at which detections took place. There where null values I turned to 0

# In[ ]:


f,ax = plt.subplots(1,1,figsize=(15,6))
angles = [int(float(t)) if t!='na' else 0 for t in train['inc_angle']]
train['intangle'] = angles
sns.distplot(angles)
plt.show()


# In[ ]:


from PIL import Image

def normalizeArrays(data,i):
    arr1 = np.reshape(data['band_1'][i],(75,75))
    arr1 = arr1+ abs(np.asarray(arr1).min())
    arr1 = np.asarray(arr1/np.asarray(arr1).max())

    arr2 = np.reshape(data['band_2'][i],(75,75))
    arr2 = arr2+ abs(np.asarray(arr2).min())
    arr2 = np.asarray(arr2/np.asarray(arr2).max())
    
    return arr1,arr2

norm =[normalizeArrays(train,i) for i in range(len(train))]
train['norm1'] = [t[0] for t in norm]
train['norm2'] = [t[1] for t in norm]


# Let's have look at the normalized data

# In[ ]:


from PIL import ImageFilter
f,axarr = plt.subplots(2,3,figsize=(20,10))
iarray = []
for i in range(6):
    img = Image.fromarray(train['norm1'][i]*255)
    img = img.convert('L')
    img = img.filter(ImageFilter.SMOOTH_MORE)
    iarray.append(img)    
axarr[0][0].imshow(iarray[0])
axarr[0][1].imshow(iarray[1])
axarr[0][2].imshow(iarray[2])
axarr[1][0].imshow(iarray[3])
axarr[1][1].imshow(iarray[4])
axarr[1][2].imshow(iarray[5])
plt.show()


# In[ ]:


from PIL import ImageFilter
f,axarr = plt.subplots(2,3,figsize=(20,10))
iarray = []
for i in range(6):
    img = Image.fromarray(train['norm2'][i]*255)
    img = img.convert('L')
    img = img.filter(ImageFilter.SMOOTH_MORE)
    iarray.append(img)    
axarr[0][0].imshow(iarray[0])
axarr[0][1].imshow(iarray[1])
axarr[0][2].imshow(iarray[2])
axarr[1][0].imshow(iarray[3])
axarr[1][1].imshow(iarray[4])
axarr[1][2].imshow(iarray[5])
plt.show()


# from this kernel  https://www.kaggle.com/muonneutrino/exploration-transforming-images-in-python the idea of using X derivaate. What an amazing 3D effect :)

# In[ ]:


from scipy import signal
fig = plt.figure(1,figsize=(15,15))
xder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
for i in range(6):
    ax = fig.add_subplot(3,3,i+1)
    arr = signal.convolve2d(np.reshape(np.array(train['norm1'][i]),(75,75)),xder,mode='valid')
    ax.imshow(arr)
plt.show()


# Now we are going to extract some basic features

# In[ ]:


max1 = [pd.Series(train['band_1'][i]).max() for i in range(len(train))]
max2 = [pd.Series(train['band_2'][i]).max() for i in range(len(train))]
mean1 = [pd.Series(train['band_1'][i]).mean() for i in range(len(train))]
mean2 =[pd.Series(train['band_2'][i]).mean() for i in range(len(train))]
min1 = [pd.Series(train['band_1'][i]).min() for i in range(len(train))]
min2 = [pd.Series(train['band_2'][i]).min() for i in range(len(train))]
train['min1'] = min1
train['min2'] = min2
train['max1'] = max1
train['max2'] = max2
train['mean1'] = mean1
train['mean2'] = mean2


# In[ ]:


from scipy.stats import kendalltau

jp = sns.jointplot(x=train['max1'], y=train['intangle'], kind="hex")
jp.fig.set_size_inches(8,6)
jp.ax_joint.set_ylim(30,45)


# and see how the correlate with the target class (ships, iceberg)

# In[ ]:


import seaborn as sns
f,axarr = plt.subplots(1,1,figsize=(15,6))
sns.heatmap(train.corr())
plt.show()


# ## Dataset Augmentation

# From this Kernel https://www.kaggle.com/sinkie/keras-data-augmentation-with-multiple-inputs  the idea of using the Keras ImageDataGenerator to augment the input dataset

# In[ ]:


gin = np.asarray([np.asarray(p).reshape(75,75) for p in train['band_1']])
gin = gin.reshape(1604,75,75,1)


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False,rotation_range=90)
datagen.fit(gin)


# In[ ]:


fig = plt.figure(1,figsize=(15,15))
for X_batch, y_batch in datagen.flow(gin, train['is_iceberg'], batch_size=9):
    for i in range(0, 9):
        ax = fig.add_subplot(3,3,i+1)
        ax.imshow(X_batch[i].reshape(75, 75), cmap=plt.get_cmap('gray'))
    plt.show()
    break


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm

clf = GradientBoostingClassifier(n_estimators = 50,random_state=0)
features = ['min1','min2','max1','max2','mean1','mean2']
scores = cross_val_score(clf,train[features],train['is_iceberg'], cv=3)
clf.fit(train[features],train['is_iceberg'])
scores



# ## Prediction

# In[ ]:


test = pd.read_json('../input/test.json')


# In[ ]:


max1 = [pd.Series(test['band_1'][i]).max() for i in range(len(test))]
max2 = [pd.Series(test['band_2'][i]).max() for i in range(len(test))]
mean1 = [pd.Series(test['band_1'][i]).mean() for i in range(len(test))]
mean2 =[pd.Series(test['band_2'][i]).mean() for i in range(len(test))]
min1 = [pd.Series(test['band_1'][i]).min() for i in range(len(test))]
min2 = [pd.Series(test['band_2'][i]).min() for i in range(len(test))]
test['min1'] = min1
test['min2'] = min2
test['max1'] = max1
test['max2'] = max2
test['mean1'] = mean1
test['mean2'] = mean2


# In[ ]:


pred = clf.predict(test[features])
out = pd.DataFrame({'id':test['id'],'is_iceberg':pred})
out.to_csv('res.csv',header=True,index=False)


# In[ ]:


import numpy as np
im1 = np.reshape(train['band_1'][0],(75,75))


# In[ ]:


from PIL import Image
from matplotlib.pyplot import imshow
from scipy.misc import toimage
plt.imshow(np.reshape(train['band_1'][0],(75,75)))
plt.show()
plt.imshow(np.reshape(train['band_1'][1],(75,75)))
plt.show()
plt.imshow(np.reshape(train['band_1'][2],(75,75)))
plt.show()


# work in progress
