
# coding: utf-8

# # Starter's Pack for Invasive Species Detection
# 
# #### Hello everyone! This is Chris, the main researcher behind the dataset featured in this competition!
# 
# First of all, we want to send a huge amount of thanks to Kaggle for making this nonprofit competition a real thing!
# 
# I am here to help everyone getting a head start in the competition! We are going to be building a pretty simple yet very powerful classificator based on the blog entry ["Building powerful image classification models using very little data"](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) by F. Chollet.
# 
# We are going to make use of Keras (Theano as backend) and the VGG16 pretrained weights. We will build our own fully connected layers to replace the top of VGG16 and later fine-tune the model for a few epochs (so we get some juicy extra AUC points in the leaderboard). The training is light, so it should run in an average Intel I5 laptop's CPU overnight!
# 
# To keep the code as similar as possible to the blog entry, this notebook is compiling together what in the blog are 2 different scripts. We are doing so sequentially, so it is easier to go to the source blog and get better understanding. First, we are going to predict features for our training and validation sets for the last convolutional layers and save those to disk. Then we are going to use those predicted features to train our little fully connected model that will tell us whether there is or not an invasive species.
# 

# In[ ]:


import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers

# path to the model weights file.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 200, 150


# For the code to work without changes on it, it is needed to download the training set and create a folder structure just like this one:
# 
# main.py <br>
# -----/data/train    (Here I am placing 1795 pictures from the training set)<br>
# -----/data/train/Invasive<br>
# -----/data/train/No_Invasive<br>
# -----/data/validation    (and here 500 pictures as our validation set)<br>
# -----/data/validation/Invasive<br>
# -----/data/validation/No_Invasive<br>
# 
# You can also make it work with everything in the same folder and making use of train_labels.csv by using the function flow(X,y) instead of flow_from_directory()
#     

# In[ ]:


train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 500
nb_epoch = 50


# We define the convolutional blocks of VGG16, load the weights and set everything to predict all of the bottleneck features.

# In[ ]:


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_height, img_width)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
    
    #We happen to have the data subdivided in folders (Presence vs Absence), so no label is needed.
    #However the dataset, as you download it, is a mix of everything together with a .csv containing 
    #the labels. So, the easiest option for you, would be to use the function flow(X,y) instead of 
    #flow_from_directory(), where 'X' is a numpy array containing the images and 'y' are the labels.
    #The other option is to use the labels to rearrange the images into the same folder structure and 
    #use this code as it is.
    print("Starting feature prediction for the training set")
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height,img_width),
        batch_size=5,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    print('Training features predicted! Starting feature prediction for the validation set')

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=50,
            class_mode=None,
            shuffle=False)
    
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
    print('Validation features predicted!')


# Now we define our fully connected top model that is going to be trained with the predicted features.

# In[ ]:


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy','rb'))
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


# In[ ]:


save_bottlebeck_features()


# In[ ]:


train_top_model()


# ## Great! Now we've gotten started and we have a nice accuracy already!
# Now we are going to start the fine tunning. This is the last script on the blog entry. So we pretty much are going to rebuild a decapitated VGG16 implating on it our own fully connected head. Then we are going to set as non-trainable the weights of the first four convolutional blocks. We will let it run for a few epochs an see what we can get from it!
# 
# To stick to Chollet's entry, we are going to define again VGG16 and our fully-connected model instead of reusing what we already wrote. We are going to load the weights for both of them, and then add them together sequentially into what is going to be our final model. We will set to non-trainable all of the Keras layers up to the 28th (1 to 4th conv blocks). We have to do so since, due to the low quantity of samples that we have, we would overfit the train set and start to wreck all of the learned features in the original VGG16 bottom layers.

# In[ ]:


nb_epoch = 10

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_height,img_width)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
#assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid')) 

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)

#model.load_weights('VGG16fineTuned.h5')


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:28]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.00002, momentum=0.9),
              metrics=['accuracy'])


# ## We run the fine tunning loop!
# Now it is our chance to use data augmentation which surely will help, given the low amount of samples.

# In[ ]:


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rotation_range=35,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255,
        rotation_range=15,
        horizontal_flip=True,
        shear_range=0.1)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height,img_width),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height,img_width),
        batch_size=20,
        class_mode='binary')

# fine-tune the model
## Callback for loss logging per epoch
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
history = LossHistory()
        
#We could use: 
#early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
#to stop the training once the model has converged.

#If we had run the training previously, we may want to load the weights now:
#model.load_weights('VGG16fineTuned.h5')


model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=[history]) #We could add "early_stopping" too if we want so.
model.save_weights('VGG16fineTuned.h5')


# ## Now it is your time to scrap everything I just said and build your own powerful net!
# 
# ## All the best!
