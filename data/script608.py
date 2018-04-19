
# coding: utf-8

# # Count the Sea Lions in the first image
# 
# - [Firs part:][1] 
#  - I am using the first picture to extract Sea Lion coordinates using blob detection
# 
# - [Second part:][2]
#  - I extract 64 by 64 images centered on the extracted coordinates
#  - In addition to the previous script, I add negative examples to the training data
#  - I train a simple keras model on the full data
#  - In the third part I will test on the training data, so overfitting is not a bug, is a feature
# 
# - Third part:
#  - Built a test set by tiling the first image
#  - Use the previously built model to decide if in each tile there is a Sea Lion
#  - **I am aware that I am training and testing on the same data**
# 
#   [1]: https://www.kaggle.com/radustoicescu/noaa-fisheries-steller-sea-lion-population-count/get-coordinates-using-blob-detection
#   [2]: https://www.kaggle.com/radustoicescu/noaa-fisheries-steller-sea-lion-population-count/use-keras-to-classify-sea-lions-0-91-accuracy

# # First Part

# In[ ]:


import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import skimage.feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, Cropping2D
from keras.utils import np_utils

from collections import Counter

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Initialize variables

# In[ ]:


class_names = ['adult_females', 'adult_males', 'juveniles', 'pups', 'subadult_males']

file_names = os.listdir("../input/Train/")
file_names = sorted(file_names, key=lambda 
                    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item)) 

# select a subset of files to run on
file_names = file_names[0:1]

# dataframe to store results in
coordinates_df = pd.DataFrame(index=file_names, columns=class_names)


# ### Extract coordinates

# In[ ]:


for filename in file_names:
    
    # read the Train and Train Dotted images
    image_1 = cv2.imread("../input/TrainDotted/" + filename)
    image_2 = cv2.imread("../input/Train/" + filename)
    
    cut = np.copy(image_2)
    
    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1,image_2)
    
    # mask out blackened regions from Train Dotted
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 20] = 0
    mask_1[mask_1 > 0] = 255
    
    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 20] = 0
    mask_2[mask_2 > 0] = 255
    
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    image_3 = cv2.bitwise_or(image_3, image_3, mask=mask_2) 
    
    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
    
    # detect blobs
    blobs = skimage.feature.blob_log(image_3, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
    
    adult_males = []
    subadult_males = []
    pups = []
    juveniles = []
    adult_females = [] 
    
    image_circles = image_1
    
    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        g,b,r = image_1[int(y)][int(x)][:]
        
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if r > 200 and g < 50 and b < 50: # RED
            adult_males.append((int(x),int(y)))
            cv2.circle(image_circles, (int(x),int(y)), 20, (0,0,255), 10) 
        elif r > 200 and g > 200 and b < 50: # MAGENTA
            subadult_males.append((int(x),int(y))) 
            cv2.circle(image_circles, (int(x),int(y)), 20, (250,10,250), 10)
        elif r < 100 and g < 100 and 150 < b < 200: # GREEN
            pups.append((int(x),int(y)))
            cv2.circle(image_circles, (int(x),int(y)), 20, (20,180,35), 10)
        elif r < 100 and  100 < g and b < 100: # BLUE
            juveniles.append((int(x),int(y))) 
            cv2.circle(image_circles, (int(x),int(y)), 20, (180,60,30), 10)
        elif r < 150 and g < 50 and b < 100:  # BROWN
            adult_females.append((int(x),int(y)))
            cv2.circle(image_circles, (int(x),int(y)), 20, (0,42,84), 10)  
            
        cv2.rectangle(cut, (int(x)-112,int(y)-112),(int(x)+112,int(y)+112), 0,-1)
            
    coordinates_df["adult_males"][filename] = adult_males
    coordinates_df["subadult_males"][filename] = subadult_males
    coordinates_df["adult_females"][filename] = adult_females
    coordinates_df["juveniles"][filename] = juveniles
    coordinates_df["pups"][filename] = pups


# In[ ]:


f, ax = plt.subplots(1,1,figsize=(10,16))
ax.imshow(cv2.cvtColor(image_circles, cv2.COLOR_BGR2RGB))
plt.show()


# # Second Part

# ### Extract 32 by 32 images of Sea Lions

# Use the previosly created image of the landscape with all the SeaLions cut out as template to extract negative examples

# In[ ]:


f, ax = plt.subplots(1,1,figsize=(10,16))
ax.imshow(cv2.cvtColor(cut, cv2.COLOR_BGR2RGB))
plt.show()


# In[ ]:


x = []
y = []

for filename in file_names:    
    image = cv2.imread("../input/Train/" + filename)
    for lion_class in class_names:
        for coordinates in coordinates_df[lion_class][filename]:
            thumb = image[coordinates[1]-32:coordinates[1]+32,coordinates[0]-32:coordinates[0]+32,:]
            if np.shape(thumb) == (64, 64, 3):
                x.append(thumb)
                y.append(lion_class)


# ### Add negative examples to the dataset

# In[ ]:


for i in range(0,np.shape(cut)[0],224):
    for j in range(0,np.shape(cut)[1],224):                
        thumb = cut[i:i+64,j:j+64,:]
        if np.amin(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)) != 0:
            if np.shape(thumb) == (64,64,3):
                x.append(thumb)
                y.append("negative")              
                


# In[ ]:


class_names.append("negative")


# In[ ]:


x = np.array(x)
y = np.array(y)


# ### Plot examples

# In[ ]:


for lion_class in class_names:
    f, ax = plt.subplots(1,10,figsize=(12,1.5))
    f.suptitle(lion_class)
    axes = ax.flatten()
    j = 0
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
        for i in range(j,len(x)):
            if y[i] == lion_class:
                j = i+1
                a.imshow(cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB))
                break


# ### One hot encoding

# In[ ]:


encoder = LabelBinarizer()
encoder.fit(y)
y = encoder.transform(y).astype(float)


# ### Build Keras model

# In[ ]:


model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))


model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# ### Training

# In[ ]:


history = model.fit(x, y, epochs=10, verbose=0)


# In[ ]:


plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # Third Part

# ### Build the test 

# In[ ]:


img = cv2.imread("../input/Train/" + filename)

x_test = []

for i in range(0,np.shape(img)[0],64):
    for j in range(0,np.shape(img)[1],64):                
        thumb = img[i:i+64,j:j+64,:]        
        if np.shape(thumb) == (64,64,3):
            x_test.append(thumb)

x_test = np.array(x_test)


# ### Predict

# In[ ]:


y_predicted = model.predict(x_test, verbose=0)


# In[ ]:


y_predicted = encoder.inverse_transform(y_predicted)


# In[ ]:


print(Counter(y_predicted).items())


# ### Correct numbers

# In[ ]:


reference = pd.read_csv('../input/Train/train.csv')
reference.ix[0:0]


# # Conclusions

# The numbers are way off, not what I expected. Especially since I tested on the same data I trained.
# 
# For some reason it detects a lot of pups.
# 
# The bad results could be in part a consequence of the huge number of negative examples in the test set.
