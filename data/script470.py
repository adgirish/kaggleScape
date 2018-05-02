
# coding: utf-8

# **UPDATE after finish**
# The final score of this model on the **private leaderboard was 0.1639**.  My best ensemble private score was 0.1477. However I choosed another model for the final solution and my final result is a bit dissapointment. But I learned a lot and this is why we all here :-) 
# 
# This was my first competition here at Kaggle. I learned a lot from the discussions and the kernels gladly published by more experienced Kagglers. 
# 
# Therefore I decided to give something back to the community. Here is my single best model scoring 0.1541 on the public leaderboard. 
# 
# Maybe it can help someone as a part of the final stacked solution. Thanks for upvoting if so ;-)
# 
# This work was built on those two great Kernels.
# 
# TheGruffalo
# Keras CNN - StatOil Iceberg LB 0.1995 (now 0.1516)
# [https://www.kaggle.com/cbryant/keras-cnn-statoil-iceberg-lb-0-1995-now-0-1516](http://https://www.kaggle.com/cbryant/keras-cnn-statoil-iceberg-lb-0-1995-now-0-1516)
# 
# QuantScientist
# Statoil CSV PyTorch SENet ensemble LB 0.1520
# [https://www.kaggle.com/solomonk/statoil-csv-pytorch-senet-ensemble-lb-0-1520](http://https://www.kaggle.com/solomonk/statoil-csv-pytorch-senet-ensemble-lb-0-1520)

# ## Imports

# In[ ]:


import pandas as pd 
import numpy as np 
import cv2 # Used to manipulated the images 
seed = 1234
np.random.seed(seed) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though

# Kfold
from sklearn.model_selection import StratifiedKFold

# Import Keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam


# In[ ]:


def get_scaled_imgs(df):
    """
    basic function for reshaping and rescaling data as images
    """
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


# ## Adding images for training

# Now, the biggest improvement I had was by adding more data to train on. I did this by simply including horizontally and vertically flipped data. Using OpenCV this is easily done.

# In[ ]:


def get_more_images(imgs):
    """
    augmentation for more data
    """    


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


# ## CNN Keras Model

# In[ ]:



def get_model():
    
    """
    Keras Sequential model

    """
    
    model=Sequential()
    
    # Conv block 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
   
    # Conv block 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    # Conv block 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    #Conv block 4
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    # Flatten before dense
    model.add(Flatten())

    #Dense 1
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))

    #Dense 2
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.0001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


# ## Load the data, and train the model usign K-fold CV
# 
# The traing here at Kaggle will be very slow. Better fork the notebook, download and run the code locally. Don't forget to set the epochs.

# In[ ]:



# Training Data
df_train = pd.read_json('../input/train.json') # this is a dataframe


Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])
df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)

Ytrain = Ytrain[idx_tr[0]]
Xtrain = Xtrain[idx_tr[0],...]

Xtr_more = get_more_images(Xtrain) 
Ytr_more = np.concatenate((Ytrain,Ytrain,Ytrain))

# K fold CV training
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
for fold_n, (train, test) in enumerate(kfold.split(Xtr_more, Ytr_more)):
    print("FOLD nr: ", fold_n)
    model = get_model()
    
    MODEL_FILE = 'mdl_simple_k{}_wght.hdf5'.format(fold_n)
    batch_size = 32
    mcp_save = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1, epsilon=1e-4, mode='min')

    # set the epochs to 30 before training on your GPU
    model.fit(Xtr_more[train], Ytr_more[train],
        batch_size=batch_size,
        epochs=1,
        verbose=1,
        validation_data=(Xtr_more[test], Ytr_more[test]),
        callbacks=[mcp_save, reduce_lr_loss])
    
    model.load_weights(filepath = MODEL_FILE)

    score = model.evaluate(Xtr_more[test], Ytr_more[test], verbose=1)
    print('\n Val score:', score[0])
    print('\n Val accuracy:', score[1])

    SUBMISSION = './result/simplenet/sub_simple_v1_{}.csv'.format(fold_n)

    df_test = pd.read_json('../input/test.json')
    df_test.inc_angle = df_test.inc_angle.replace('na',0)
    Xtest = (get_scaled_imgs(df_test))
    pred_test = model.predict(Xtest)

    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
    print(submission.head(10))

    submission.to_csv(SUBMISSION, index=False)
    print("submission saved")


# ## Combine the K-fold solutions together 

# In[ ]:


wdir = './result/simplenet/'
stacked_1 = pd.read_csv(wdir + 'sub_simple_v1_0.csv')
stacked_2 = pd.read_csv(wdir + 'sub_simple_v1_1.csv')
stacked_3 = pd.read_csv(wdir + 'sub_simple_v1_2.csv')
stacked_4 = pd.read_csv(wdir + 'sub_simple_v1_3.csv')
stacked_5 = pd.read_csv(wdir + 'sub_simple_v1_4.csv')
stacked_6 = pd.read_csv(wdir + 'sub_simple_v1_5.csv')
stacked_7 = pd.read_csv(wdir + 'sub_simple_v1_6.csv')
stacked_8 = pd.read_csv(wdir + 'sub_simple_v1_7.csv')
stacked_9 = pd.read_csv(wdir + 'sub_simple_v1_8.csv')
stacked_10 = pd.read_csv(wdir + 'sub_simple_v1_9.csv')
sub = pd.DataFrame()
sub['id'] = stacked_1['id']
sub['is_iceberg'] = np.exp(np.mean(
    [
        stacked_1['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_2['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_3['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_4['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_5['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_6['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_7['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_8['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_9['is_iceberg'].apply(lambda x: np.log(x)),
        stacked_10['is_iceberg'].apply(lambda x: np.log(x)),
        ], axis=0))

sub.to_csv(wdir + 'final_ensemble.csv', index=False, float_format='%.6f')    


