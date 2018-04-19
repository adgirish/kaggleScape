
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import keras as k
from keras.layers import Merge
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.callbacks import History
from keras.layers import Activation
from keras.models import model_from_json
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from scipy.ndimage import rotate as rot
np.random.seed(100)


# In[ ]:


file_path = '/Users/henok.s.mengistu/Documents/Henok\'s/Iceberg_challenge/data/processed/train.json'


# In[ ]:


train = pd.read_json(file_path)


# In[ ]:


print(train.head())
train.shape


# In[ ]:


train[train['inc_angle'] == 'na'].count()


# In[ ]:


train.inc_angle = train.inc_angle.map(lambda x: 0.0 if x == 'na' else x)


# In[ ]:


def transform (df):
    images = []
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75,75)
        band_2 = np.array(row['band_2']).reshape(75,75)
        band_3 = band_1 + band_2
        
        band_1_norm = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        band_2_norm = (band_2 - band_2. mean()) / (band_2.max() - band_2.min())
        band_3_norm = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        
        images.append(np.dstack((band_1_norm, band_2_norm, band_3_norm)))
    
    return np.array(images)


# In[ ]:


def augment(images):
    image_mirror_lr = []
    image_mirror_ud = []
    image_rotate = []
    for i in range(0,images.shape[0]):
        band_1 = images[i,:,:,0]
        band_2 = images[i,:,:,1]
        band_3 = images[i,:,:,2]
            
        # mirror left-right
        band_1_mirror_lr = np.flip(band_1, 0)
        band_2_mirror_lr = np.flip(band_2, 0)
        band_3_mirror_lr = np.flip(band_3, 0)
        image_mirror_lr.append(np.dstack((band_1_mirror_lr, band_2_mirror_lr, band_3_mirror_lr)))
        
        # mirror up-down
        band_1_mirror_ud = np.flip(band_1, 1)
        band_2_mirror_ud = np.flip(band_2, 1)
        band_3_mirror_ud = np.flip(band_3, 1)
        image_mirror_ud.append(np.dstack((band_1_mirror_ud, band_2_mirror_ud, band_3_mirror_ud)))
        
        #rotate 
        band_1_rotate = rot(band_1, 30, reshape=False)
        band_2_rotate = rot(band_2, 30, reshape=False)
        band_3_rotate = rot(band_3, 30, reshape=False)
        image_rotate.append(np.dstack((band_1_rotate, band_2_rotate, band_3_rotate)))
        
    mirrorlr = np.array(image_mirror_lr)
    mirrorud = np.array(image_mirror_ud)
    rotated = np.array(image_rotate)
    images = np.concatenate((images, mirrorlr, mirrorud, rotated))
    return images


# In[ ]:


train_X = transform(train)
train_y = np.array(train ['is_iceberg'])

indx_tr = np.where(train.inc_angle > 0)
print (indx_tr[0].shape)

train_y = train_y[indx_tr[0]]
train_X = train_X[indx_tr[0], ...]

train_X = augment(train_X)
train_y = np.concatenate((train_y,train_y, train_y, train_y))

print (train_X.shape)
print (train_y.shape)


# In[ ]:


model = k.models.Sequential()

model.add(k.layers.convolutional.Conv2D(64, kernel_size=(3,3), input_shape=(75,75,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(k.layers.convolutional.MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.convolutional.Conv2D(128, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(k.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.convolutional.Conv2D(128, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(k.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(k.layers.Dropout(0.3))

model.add(k.layers.convolutional.Conv2D(64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(k.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(k.layers.Dropout(0.3))

model.add(k.layers.Flatten())

model.add(k.layers.Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(k.layers.Dropout(0.2))

model.add(k.layers.Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(k.layers.Dropout(0.2))


model.add(k.layers.Dense(1))
model.add(Activation('sigmoid'))

mypotim=Adam(lr=0.01, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer = mypotim, metrics=['accuracy'])

model.summary()


# In[ ]:


batch_size = 64
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 0, mode= 'min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 7, verbose =1, 
                                   epsilon = 1e-4, mode='min', min_lr = 0.0001)
model_filepath='/Users/henok.s.mengistu/Documents/Henok\'s/Iceberg_challenge/weights.best.hdf5'
checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [early_stopping, checkpoint]


# In[ ]:


history = model.fit(train_X, train_y, batch_size = batch_size, epochs =20, verbose =1, validation_split = 0.1, 
          callbacks=callbacks_list)


# In[ ]:


print (history.history.keys())
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[ ]:


model_json = model.to_json()
with open("/Users/henok.s.mengistu/Documents/Henok\'s/Iceberg_challenge/model.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:


# load json and create model
json_file = open('/Users/henok.s.mengistu/Documents/Henok\'s/Iceberg_challenge/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('/Users/henok.s.mengistu/Documents/Henok\'s/Iceberg_challenge/weights.best.hdf5')
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer = mypotim, metrics=['accuracy'])


# In[ ]:


test_file = '/Users/henok.s.mengistu/Documents/Henok\'s/Iceberg_challenge/data-1/processed/test.json'
test = pd.read_json(test_file)
test.inc_angle = test.inc_angle.replace('na',0)
test_X = transform(test)
print (test_X.shape)


# In[ ]:


pred_test = loaded_model.predict(test_X, verbose=1)
submission = pd.DataFrame({'id': test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
submission.to_csv('/Users/henok.s.mengistu/Documents/Henok\'s/Iceberg_challenge/submission.csv', index=False)

