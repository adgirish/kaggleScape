
# coding: utf-8

# This Kernal implements a Keras + Tensorflow CNN for the StatOil Iceberg competition. It has yielded results of 0.1995 on the leaderboard. With some tuning and image filtering plus more of an inclusion of the incident angle, a better result could be yielded I'm sure.
# 
# The input is a 75x75x3 set of images. The output is a binary 0/1 where 1 is noteed as an iceberg. 
# 
# The set of images are band_1 (HH), band_2 (HV), and an combined band which would be (HH dot HV)/constant. However, since we are working with the images in dB, the 3rd band is modified to compenate for the log function yielding band_1 + band_2 -log(constant). The last term is neglected as when the images are scaled the 3rd term would be removed by the mathematics anyway.
# 
# This and other information can be found from: https://earth.esa.int/c/document_library/get_file?folderId=409229&name=DLFE-5566.pdf

# ## Imports

# In[ ]:


import pandas as pd 
import numpy as np 
import cv2 # Used to manipulated the images 
np.random.seed(1337) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though

# Import Keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


# ## Load Training Data

# In[ ]:


df_train = pd.read_json('../input/train.json') # this is a dataframe


# Need to reshape and feature scale the images:

# In[ ]:


def get_scaled_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)


# In[ ]:


Xtrain = get_scaled_imgs(df_train)


# Get the response variable "is_iceberg"

# In[ ]:


Ytrain = np.array(df_train['is_iceberg'])


# Some of the incident angle from the satellite are unknown and marked as "na". Replace these na with 0 and find the indices where the incident angle is >0 (this way you can use a truncated set or the full set of training data).

# In[ ]:


df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)


# You can now use the option of training with only known incident angles or the whole set. I found slightly better results training with only the known incident angles so:

# In[ ]:


Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]


# ## Adding images for training

# Now, the biggest improvement I had was by adding more data to train on. I did this by simply including horizontally and vertically flipped data. Using OpenCV this is easily done.

# In[ ]:


def get_more_images(imgs):
    
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
       
    more_images = np.concatenate((imgs,v,h))
    
    return more_images


# I rename the returned value so i have the option of using the original data set or the expanded data set

# In[ ]:


Xtr_more = get_more_images(Xtrain) 


# And then define the new response variable:

# In[ ]:


Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))


# ## CNN Keras Model

# Now the nitty gritty of the situation, the CNN model. This is a simplistic model that should give reasonable results. It is not tuned that well and there are plenty of options and changes you can try so as to improve it. At least you will get the idea:

# In[ ]:


def getModel():
    #Build keras model
    
    model=Sequential()
    
    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    #CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    #Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


# Now get the model and get ready to train

# In[ ]:


model = getModel()
model.summary()

batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')


# Now train the model! (Each epoch ran at about 10s on GPU)

# In[ ]:


model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)


# ## Results

# Load the best weights and check the score on the training data.

# In[ ]:


model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate(Xtrain, Ytrain, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])


# Now, to make a submission, load the test data and train the model and output a csv file.

# In[ ]:


df_test = pd.read_json('../input/test.json')
df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = (get_scaled_imgs(df_test))
pred_test = model.predict(Xtest)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print(submission.head(10))

submission.to_csv('submission.csv', index=False)


# The best submission with this I received was 0.1995 on the leaderboard. Have a go and see how well you can do!
