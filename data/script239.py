
# coding: utf-8

# This kernel implements a fully convolutional network (FCN) in Keras for the [Statoil/C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge). The input, `X`, will be a 75x75x3 normalized image, where `r=band_1, g=band_2 and b=band_1/band_2`. The output, `y`, will be a one-hot encoding of `is_iceberg`. The FCN consists of a series of convolutions and pooling, ending with a [GlobalAveragePooling2D](https://keras.io/layers/pooling/#globalaveragepooling2d) and a softmax activation.
# 
# This solution was inspired by the [fast.ai Deep Learning Part 1, Lesson 7](http://course.fast.ai/lessons/lesson7.html).

# ## Setup

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pdb
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread


# ## Load Training Data

# In[ ]:


train = pd.read_json('../input/train.json')


# Pre-process the images to a 75x75x3 rescale-normalized image, where r=band_1, g=band_2 and b=band_1/band_2.

# In[ ]:


def get_images(df):
    '''Create 3-channel 'images'. Return rescale-normalised images.'''
    images = []
    for i, row in df.iterrows():
        # Formulate the bands as 75x75 arrays
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        # Rescale
        r = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
        g = (band_2 - band_2.min()) / (band_2.max() - band_2.min())
        b = (band_3 - band_3.min()) / (band_3.max() - band_3.min())

        rgb = np.dstack((r, g, b))
        images.append(rgb)
    return np.array(images)


# In[ ]:


X = get_images(train)


# One-hot encode the output labels.

# In[ ]:


y = to_categorical(train.is_iceberg.values,num_classes=2)


# Split the data into 80% training and 20% validation.

# In[ ]:


Xtr, Xv, ytr, yv = train_test_split(X, y, shuffle=False, test_size=0.20)


# ## Fully Convolutional Network

# Build a network consisting of convolutions and pooling which will reduce the image space from 75x75 down to 4x4, while adding complexity in the channels. The final convolution produces one 4x4 channel per class (in this case we have 2 classes: boat and iceberg). The final stage uses global average pooling and a softmax activation.

# In[ ]:


def ConvBlock(model, layers, filters):
    '''Create [layers] layers consisting of zero padding, a convolution with [filters] 3x3 filters and batch normalization. Perform max pooling after the last layer.'''
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(BatchNormalization(axis=3))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


# In[ ]:


def create_model():
    '''Create the FCN and return a keras model.'''

    model = Sequential()

    # Input image: 75x75x3
    model.add(Lambda(lambda x: x, input_shape=(75, 75, 3)))
    ConvBlock(model, 1, 32)
    # 37x37x32
    ConvBlock(model, 1, 64)
    # 18x18x64
    ConvBlock(model, 1, 128)
    # 9x9x128
    ConvBlock(model, 1, 128)
    # 4x4x128
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(2, (3, 3), activation='relu'))
    model.add(GlobalAveragePooling2D())
    # 4x4x2
    model.add(Activation('softmax'))
    
    return model


# In[ ]:


# Create the model and compile
model = create_model()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])


# In[ ]:


model.summary()


# ## Train Network

# Train the network for 15 epochs.

# In[ ]:


init_epo = 0
num_epo = 30
end_epo = init_epo + num_epo


# In[ ]:


print ('lr = {}'.format(K.get_value(model.optimizer.lr)))
history = model.fit(Xtr, ytr, validation_data=(Xv, yv), batch_size=32, epochs=end_epo, initial_epoch=init_epo)
init_epo += num_epo
end_epo = init_epo + num_epo


# ## Heat maps

# The FCN will learn to distinguish between boats and icebergs using the final 2 4x4 channels. Each 4x4 channel represents one of the classes. Overlaying the 4x4 channel for each class on the image produces a heat map showing the "boatness" or the "bergness" of each section of the 4x4 grid.

# In[ ]:


l = model.layers
conv_fn = K.function([l[0].input, K.learning_phase()], [l[-4].output])


# In[ ]:


def get_cm(inp, label):
    '''Convert the 4x4 layer data to a 75x75 image.'''
    conv = np.rollaxis(conv_fn([inp,0])[0][0],2,0)[label]
    return scipy.misc.imresize(conv, (75,75), interp='nearest')


# In[ ]:


def info_img (im_idx):
    '''Generate heat maps for the boat (boatness) and iceberg (bergness) for image im_idx.'''
    if (yv[im_idx][1] == 1.0):
        img_type = 'iceberg'
    else:
        img_type = 'boat'
    inp = np.expand_dims(Xv[im_idx], 0)
    img_guess = np.round(model.predict(inp)[0],2)
    if (img_guess[1] > 0.5):
        guess_type = 'iceberg'
    else:
        guess_type = 'boat'
    cm0 = get_cm(inp, 0)
    cm1 = get_cm(inp, 1)
    print ('truth: {}'.format(img_type))
    print ('guess: {}, prob: {}'.format(guess_type, img_guess))
    plt.figure(1,figsize=(10,10))
    plt.subplot(121)
    plt.title('Boatness')
    plt.imshow(Xv[im_idx])
    plt.imshow(cm0, cmap="cool", alpha=0.5)
    plt.subplot(122)
    plt.title('Bergness')
    plt.imshow(Xv[im_idx])
    plt.imshow(cm1, cmap="cool", alpha=0.5)


# Some example heat maps plotted with the mean-normalized image.

# In[ ]:


info_img(13)


# With proper training, the validation loss along with the heatmaps get better. Using additional training strategies, this network achieved 0.193 on the Leaderboard.

# ## Creating a Submission

# In[ ]:


test = pd.read_json('../input/test.json')
Xtest = get_images(test)
test_predictions = model.predict_proba(Xtest)
submission = pd.DataFrame({'id': test['id'], 'is_iceberg': test_predictions[:, 1]})
submission.to_csv('sub_fcn.csv', index=False)


# In[ ]:


submission.head(5)

