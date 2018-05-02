
# coding: utf-8

# # Xception model
# - for weight see: https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/41021

# In[ ]:


import os
import pickle
from keras.applications.xception import Xception
from keras.layers import Flatten, Dense, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import math

from keras.models import load_model
import keras.backend as K
from keras.metrics import top_k_categorical_accuracy
import tensorflow as tf

# MultiGPU model build on top of
# https://github.com/sallamander/multi-gpu-keras-tf/
from multiGPU import MultiGPUModel
import numpy as np


# In[ ]:


model_name = "xception_v2"
models_savename = "./models/" + model_name

train_data_dir = '/path/to/train/dir'
val_data_dir = '/path/to/val/dir'
classnames = pickle.load(open("/path/to/val/", "rb"))
batch_size = 86 * 3  # 258
img_width = 180
img_height = 180


# In[ ]:


model0 = Xception(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(img_width, img_height, 3))

for lay in model0.layers:
    lay.trainable = False
    
x = model0.output
x = GlobalAveragePooling2D(name='avg_pool')(x)

x = Dropout(0.2)(x)
x = Dense(len(classnames), activation='softmax', name='predictions')(x)
model0 = Model(model0.input, x)

# Train on 3GPUs
model = MultiGPUModel(model0, [0, 1, 2], int(batch_size/3))


# In[ ]:


# Data generator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = classnames,
        class_mode = 'categorical')

val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = classnames,
        class_mode = 'categorical')


# In[ ]:


os.makedirs("./models", exist_ok=True)
callbacks = [ModelCheckpoint(monitor='val_loss',
                             filepath= models_savename + '_{epoch:03d}-{val_loss:.7f}.hdf5',
                             save_best_only=False,
                             save_weights_only=False,
                             mode='max'),
             TensorBoard(log_dir='logs/{}'.format(model_name))]


# In[ ]:


# Train head
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9), metrics=[top_k_categorical_accuracy, 'accuracy'])
model.fit_generator(generator=train_generator,
                    steps_per_epoch=math.ceil(2000000 / batch_size),
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=validation_generator,
                    initial_epoch=0,
                    epochs=3,
                    use_multiprocessing=True,
                    max_queue_size=10,
                    workers = 8,
                    validation_steps=math.ceil(10000 / batch_size))

# train xception blocks 11+
for clayer in model.layers[4].layers:
    print("trainable:", clayer.name)
    if clayer.name.split("_")[0] in ["block{}".format(i) for i in range(10, 15)]:
        clayer.trainable = True

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00025), 
              metrics=[top_k_categorical_accuracy, 'accuracy'])
model.fit_generator(generator=train_generator,
                    steps_per_epoch=math.ceil(2000000 / batch_size),
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=validation_generator,
                    initial_epoch=3,
                    epochs=8,
                    use_multiprocessing=True,
                    max_queue_size=10,
                    workers = 8,
                    validation_steps=math.ceil(10000 / batch_size))

# Train the whole model
for clayer in model.layers[4].layers:
    clayer.trainable = True

# Note you need to recompile the whole thing. Otherwise you are not traing first layers    
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00025), 
              metrics=[top_k_categorical_accuracy, 'accuracy'])


init_epochs = 8  # We pretrained the model already

# Keep training for as long as you like.
for i in range(100):
    # gradually decrease the learning rate
    K.set_value(model.optimizer.lr, 0.95 * K.get_value(model.optimizer.lr))
    start_epoch = (i * 2)
    epochs = ((i + 1) * 2)    
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=math.ceil(2000000 / batch_size),
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        initial_epoch=start_epoch + init_epochs,
                        epochs=epochs + init_epochs,
                        use_multiprocessing=True,
                        max_queue_size=10,
                        workers = 8,
                        validation_steps=math.ceil(10000 / batch_size))

