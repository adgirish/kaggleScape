
# coding: utf-8

# In[ ]:


#
# Finetune the Inception V3 network on the CDiscount dataset.
#
# Taken from https://keras.io/applications/#usage-examples-for-image-classification-models


# In[ ]:


import os
import pickle
import itertools
import io
import time
import bson
import threading

import pandas as pd
from scipy.misc import imread
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import keras


# In[ ]:


def create_model(num_classes=None):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(4096, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


# In[ ]:


def grouper(n, iterable):
    '''
    Given an iterable, it'll return size n chunks per iteration.
    Handles the last chunk too.
    '''
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

class threadsafe_iter:
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def get_features_label(documents, batch_size=32, return_labels=True):
    '''
    Given a document return X, y
    
    X is scaled to [0, 1] and consists of all images contained in document.
    y is given an integer encoding.
    '''
    
    
    for batch in grouper(batch_size, documents): 
        images = []
        labels = []

        for document in batch:
            category = document.get('category_id', '')
            img = document.get('imgs')[0]
            data = io.BytesIO(img.get('picture', None))
            im = imread(data)

            if category:    
                label = labelencoder.transform([category])
            else:
                label = None

            im = im.astype('float32') / 255.0

            images.append(im)
            labels.append(label)

        if return_labels:
            yield np.array(images), np.array(labels)
        else:
            yield np.array(images)


# In[ ]:


if os.path.isfile('labelencoder.pkl'):
    with open('labelencoder.pkl', 'rb') as f:
        labelencoder = pickle.load(f)
    categories = pd.read_csv('categories.csv')
    
else:
    # Get the category ID for each document in the training set.
    documents = bson.decode_file_iter(open('../input/train.bson', 'rb'))
    categories = [(d['_id'], d['category_id']) for d in documents]
    categories = pd.DataFrame(categories, columns=['id', 'cat'])

    # Create a label encoder for all the labels found
    labelencoder = LabelEncoder()
    labelencoder.fit(categories.cat.unique().ravel())
    
    with open('labelencoder.pkl', 'wb') as f:
        pickle.dump(labelencoder, f)
        
    categories.to_csv('categories.csv')


# In[ ]:


# load the previous model

try:
    inception = keras.models.load_model('inceptionv3-finetune.h5')
except:
    inception = create_model(num_classes=len(labelencoder.classes_))

# So we can look at the progress on Tensorboard
callback = keras.callbacks.TensorBoard(
    log_dir='./logs/inception/2/{}'.format(time.time())
)

generator = get_features_label(bson.decode_file_iter(open('../input/train.bson', 'rb')))

# docs says train for a few epocs (LOL!)
# Each step is 32 images.

# 200 epochs x  500 steps x 32 images -> 3 200 000 images / ~7M
inception.fit_generator(
    generator=generator,
    epochs=320,
    steps_per_epoch=500,
    callbacks=[callback],
    validation_data=generator,
    validation_steps=50
)

inception.save('inceptionv3-finetune.h5')


# In[ ]:


# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in inception.layers[:249]:
    layer.trainable = False
for layer in inception.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
inception.compile(optimizer=SGD(lr=0.00001, momentum=0.9),
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# So we can look at the progress on Tensorboard
callback = keras.callbacks.TensorBoard(
    log_dir='./logs/inception/{}'.format(time.time())
)

generator = get_features_label(bson.decode_file_iter(open('data/train.bson', 'rb')))

# docs says train for a few epocs (LOL!)
# Each step is 32 images.

# 200 epochs x  steps x 32 images -> 320 000 images / ~7M
inception.fit_generator(
    generator=generator,
    epochs=320,
    steps_per_epoch=500,
    callbacks=[callback],
    validation_data=generator,
    validation_steps=50
)

inception.save('inceptionv3-finetune-2.h5')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# create a submission\n\ngenerator = get_features_label(bson.decode_file_iter(open(\'data/test.bson\', \'rb\')), return_labels=False)\n\npredictions = []\n\nfor i, batch in enumerate(generator):\n    output = inception.predict(batch)\n    labels = labelencoder.inverse_transform(output.argmax(axis=1))\n    predictions.extend(labels.tolist())\n    \n    if i and (i % 200 == 0):\n        print("{} images predicted.".format(len(predictions)))')


# In[ ]:


with open('predictions.pkl', 'wb') as pf:
    pickle.dump(predictions, pf)

submission = pd.read_csv('data/sample_submission.csv')
submission.category_id = predictions
submission.to_csv('submission.csv', index=False)

