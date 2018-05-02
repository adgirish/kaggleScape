
# coding: utf-8

# #Intro
# First of all thanks to Kjetil Åmdal-Sævik for providing excellent code for data preparation.
# Being a novice python programmer, my code may not be that much efficient but it may serve as a starting point for using TensorFlow.

# In[1]:


import os
import sys
import numpy as np
import tensorflow as tf
import random
import math
import warnings
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# In[2]:


# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


# In[3]:


# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]


# In[4]:


# Get and resize train images and masks
images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
labels = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    images[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    labels[n] = mask

X_train = images
Y_train = labels

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')


# In[5]:


def shuffle():
    global images, labels
    p = np.random.permutation(len(X_train))
    images = X_train[p]
    labels = Y_train[p]


# In[6]:


def next_batch(batch_s, iters):
    if(iters == 0):
        shuffle()
    count = batch_s * iters
    return images[count:(count + batch_s)], labels[count:(count + batch_s)]


# In[7]:


def deconv2d(input_tensor, filter_size, output_size, out_channels, in_channels, name, strides = [1, 1, 1, 1]):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]
    out_shape = tf.stack([batch_size, output_size, output_size, out_channels])
    filter_shape = [filter_size, filter_size, out_channels, in_channels]
    w = tf.get_variable(name=name, shape=filter_shape)
    h1 = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='SAME')
    return h1


# In[8]:


def conv2d(input_tensor, depth, kernel, name, strides=(1, 1), padding="SAME"):
    return tf.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel, strides=strides, padding=padding, activation=tf.nn.relu, name=name)


# In[9]:


X = tf.placeholder(tf.float32, [None, 128, 128, 3])
Y_ = tf.placeholder(tf.float32, [None, 128, 128, 1])
lr = tf.placeholder(tf.float32)


# In[10]:


net = conv2d(X, 32, 1, "Y0") #128

net = conv2d(net, 64, 3, "Y2", strides=(2, 2)) #64

net = conv2d(net, 128, 3, "Y3", strides=(2, 2)) #32


net = deconv2d(net, 1, 32, 128, 128, "Y2_deconv") # 32
net = tf.nn.relu(net)

net = deconv2d(net, 2, 64, 64, 128, "Y1_deconv", strides=[1, 2, 2, 1]) # 64
net = tf.nn.relu(net)

net = deconv2d(net, 2, 128, 32, 64, "Y0_deconv", strides=[1, 2, 2, 1]) # 128
net = tf.nn.relu(net)

logits = deconv2d(net, 1, 128, 1, 32, "logits_deconv") # 128

loss = tf.losses.sigmoid_cross_entropy(Y_, logits)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)


# In[ ]:


# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_count = 0
display_count = 1
for i in range(10000):
    # training on batches of 10 images with 10 mask images
    if(batch_count > 67):
        batch_count = 0    

    batch_X, batch_Y = next_batch(10, batch_count)

    batch_count += 1

    feed_dict = {X: batch_X, Y_: batch_Y, lr: 0.0005}
    loss_value, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

    if(i % 500 == 0):
        print(str(display_count) + " training loss:", str(loss_value))
        display_count +=1
        
print("Done!")


# **Test on the data that is not seen by the network during training:**

# In[ ]:


ix = 3 #random.randint(0, 64) #len(X_test) - 1 = 64
test_image = X_test[ix].astype(float)
imshow(test_image)
plt.show()


# In[ ]:


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# In[ ]:


#print(ix)
test_image = np.reshape(test_image, [-1, 128 , 128, 3])
test_data = {X:test_image}

test_mask = sess.run([logits],feed_dict=test_data)
test_mask = np.reshape(np.squeeze(test_mask), [IMG_WIDTH , IMG_WIDTH, 1])
for i in range(IMG_WIDTH):
    for j in range(IMG_HEIGHT):
            test_mask[i][j] = int(sigmoid(test_mask[i][j])*255)
imshow(test_mask.squeeze().astype(np.uint8))
plt.show()

