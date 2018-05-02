
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import skimage.io
import matplotlib.pyplot as plt
from skimage import transform
import os
import shutil
from tqdm import tqdm
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


def read_image_labels(image_id):
    # most of the content in this function is taken from 'Example Metric Implementation' kernel 
    # by 'William Cukierski'
    image_file = "../input/stage1_train/{}/images/{}.png".format(image_id,image_id)
    mask_file = "../input/stage1_train/{}/masks/*.png".format(image_id)
    image = skimage.io.imread(image_file)
    masks = skimage.io.imread_collection(mask_file).concatenate()    
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + 1
    return image, labels

def data_aug(image,label,angel=30,resize_rate=0.9):
    flip = random.randint(0, 1)
    size = image.shape[0]
    rsize = random.randint(np.floor(resize_rate*size),size)
    w_s = random.randint(0,size - rsize)
    h_s = random.randint(0,size - rsize)
    sh = random.random()/2-0.25
    rotate_angel = random.random()/180*np.pi*angel
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angel)
    # Apply transform to image data
    image = transform.warp(image, inverse_map=afine_tf,mode='edge')
    label = transform.warp(label, inverse_map=afine_tf,mode='edge')
    # Randomly corpping image frame
    image = image[w_s:w_s+size,h_s:h_s+size,:]
    label = label[w_s:w_s+size,h_s:h_s+size]
    # Ramdomly flip frame
    if flip:
        image = image[:,::-1,:]
        label = label[:,::-1]
    return image, label


# Here comes an example of randomly rotate and resize an image.
# 
# The augmented sample could be not the same shape as the original  image, they should all be resize before feed in deep learning model.

# In[ ]:


image_ids = check_output(["ls", "../input/stage1_train/"]).decode("utf8").split()
image_id = image_ids[random.randint(0,len(image_ids))]
image, labels = read_image_labels(image_id)
plt.subplot(221)
plt.imshow(image)
plt.subplot(222)
plt.imshow(labels)

new_image, new_labels = data_aug(image,labels,angel=5,resize_rate=0.9)
plt.subplot(223)
plt.imshow(new_image)
plt.subplot(224)
plt.imshow(new_labels)


# The following code is to save the augmented data locally.
#  It is only useful for Linux operating system

# In[ ]:


def make_data_augmentation(image_ids,split_num):
    for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
        image,labels = read_image_labels(image_id)
        if not os.path.exists("../input/stage1_train/{}/augs/".format(image_id)):
            os.makedirs("../input/stage1_train/{}/augs/".format(image_id))
        if not os.path.exists("../input/stage1_train/{}/augs_masks/".format(image_id)):
            os.makedirs("../input/stage1_train/{}/augs_masks/".format(image_id))
            
        # also save the original image in augmented file 
        plt.imsave(fname="../input/stage1_train/{}/augs/{}.png".format(image_id,image_id), arr = image)
        plt.imsave(fname="../input/stage1_train/{}/augs_masks/{}.png".format(image_id,image_id),arr = labels)

        for i in range(split_num):
            new_image, new_labels = data_aug(image,labels,angel=5,resize_rate=0.9)
            aug_img_dir = "../input/stage1_train/{}/augs/{}_{}.png".format(image_id,image_id,i)
            aug_mask_dir = "../input/stage1_train/{}/augs_masks/{}_{}.png".format(image_id,image_id,i)
            plt.imsave(fname=aug_img_dir, arr = new_image)
            plt.imsave(fname=aug_mask_dir,arr = new_labels)

def clean_data_augmentation(image_ids):
    for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
        if os.path.exists("../input/stage1_train/{}/augs/".format(image_id)):
            shutil.rmtree("../input/stage1_train/{}/augs/".format(image_id))
        if os.path.exists("../input/stage1_train/{}/augs_masks/".format(image_id)):
            shutil.rmtree("../input/stage1_train/{}/augs_masks/".format(image_id))


image_ids = check_output(["ls", "../input/stage1_train/"]).decode("utf8").split()
split_num = 10
#make_data_augmentation(image_ids,split_num)
#clean_data_augmentation(image_ids)


# Here we also provide our tensorflow version UNet, the visualization on tensorboard is shown as below:
# ![graph](http://i67.tinypic.com/105v91z.png)

# In[ ]:


def get_variable(name,shape):
    return tf.get_variable(name, shape, initializer = tf.contrib.layers.xavier_initializer())

def UNet(X):
    ### Unit 1 ###
    with tf.name_scope('Unit1'):
        W1_1 =   get_variable("W1_1", [3,3,3,16] )
        Z1 = tf.nn.conv2d(X,W1_1, strides = [1,1,1,1], padding = 'SAME')
        A1 = tf.nn.relu(Z1)
        W1_2 =   get_variable("W1_2", [3,3,16,16] )
        Z2 = tf.nn.conv2d(A1,W1_2, strides = [1,1,1,1], padding = 'SAME')
        A2 = tf.nn.relu(Z2) 
        P1 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 2 ###
    with tf.name_scope('Unit2'):
        W2_1 =   get_variable("W2_1", [3,3,16,32] )
        Z3 = tf.nn.conv2d(P1,W2_1, strides = [1,1,1,1], padding = 'SAME')
        A3 = tf.nn.relu(Z3)
        W2_2 =   get_variable("W2_2", [3,3,32,32] )
        Z4 = tf.nn.conv2d(A3,W2_2, strides = [1,1,1,1], padding = 'SAME')
        A4 = tf.nn.relu(Z4) 
        P2 = tf.nn.max_pool(A4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 3 ###
    with tf.name_scope('Unit3'):
        W3_1 =   get_variable("W3_1", [3,3,32,64] )
        Z5 = tf.nn.conv2d(P2,W3_1, strides = [1,1,1,1], padding = 'SAME')
        A5 = tf.nn.relu(Z5)
        W3_2 =   get_variable("W3_2", [3,3,64,64] )
        Z6 = tf.nn.conv2d(A5,W3_2, strides = [1,1,1,1], padding = 'SAME')
        A6 = tf.nn.relu(Z6) 
        P3 = tf.nn.max_pool(A6, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 4 ###
    with tf.name_scope('Unit4'):
        W4_1 =   get_variable("W4_1", [3,3,64,128] )
        Z7 = tf.nn.conv2d(P3,W4_1, strides = [1,1,1,1], padding = 'SAME')
        A7 = tf.nn.relu(Z7)
        W4_2 =   get_variable("W4_2", [3,3,128,128] )
        Z8 = tf.nn.conv2d(A7,W4_2, strides = [1,1,1,1], padding = 'SAME')
        A8 = tf.nn.relu(Z8) 
        P4 = tf.nn.max_pool(A8, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 5 ###
    with tf.name_scope('Unit5'):
        W5_1 =   get_variable("W5_1", [3,3,128,256] )
        Z9 = tf.nn.conv2d(P4,W5_1, strides = [1,1,1,1], padding = 'SAME')
        A9 = tf.nn.relu(Z9)
        W5_2 =   get_variable("W5_2", [3,3,256,256] )
        Z10 = tf.nn.conv2d(A9,W5_2, strides = [1,1,1,1], padding = 'SAME')
        A10 = tf.nn.relu(Z10) 
    ### Unit 6 ###
    with tf.name_scope('Unit6'):
        W6_1 =   get_variable("W6_1", [3,3,256,128] )
        U1 = tf.layers.conv2d_transpose(A10, filters = 128, kernel_size = 2, strides = 2, padding = 'SAME')
        U1 = tf.concat([U1, A8],3)
        W6_2 =   get_variable("W6_2", [3,3,128,128] )
        Z11 = tf.nn.conv2d(U1,W6_1, strides = [1,1,1,1], padding = 'SAME')
        A11 = tf.nn.relu(Z11)
        Z12 = tf.nn.conv2d(A11,W6_2, strides = [1,1,1,1], padding = 'SAME')
        A12 = tf.nn.relu(Z12)
    ### Unit 7 ###
    with tf.name_scope('Unit7'):
        W7_1 =   get_variable("W7_1", [3,3,128,64] )
        U2 = tf.layers.conv2d_transpose(A12, filters = 64, kernel_size = 2, strides = 2, padding = 'SAME')
        U2 = tf.concat([U2, A6],3)
        Z13 = tf.nn.conv2d(U2,W7_1, strides = [1,1,1,1], padding = 'SAME')
        A13 = tf.nn.relu(Z13)
        W7_2 =   get_variable("W7_2", [3,3,64,64] )
        Z14 = tf.nn.conv2d(A13,W7_2, strides = [1,1,1,1], padding = 'SAME')
        A14 = tf.nn.relu(Z14)
    ### Unit 8 ###
    with tf.name_scope('Unit8'):
        W8_1 =   get_variable("W8_1", [3,3,64,32] )
        U3 = tf.layers.conv2d_transpose(A14, filters = 32, kernel_size = 2, strides = 2, padding = 'SAME')
        U3 = tf.concat([U3, A4],3)
        Z15 = tf.nn.conv2d(U3,W8_1, strides = [1,1,1,1], padding = 'SAME')
        A15 = tf.nn.relu(Z15)
        W8_2 =   get_variable("W8_2", [3,3,32,32] )
        Z16 = tf.nn.conv2d(A15,W8_2, strides = [1,1,1,1], padding = 'SAME')
        A16 = tf.nn.relu(Z16)
    ### Unit 9 ###
    with tf.name_scope('Unit9'):
        W9_1 =   get_variable("W9_1", [3,3,32,16] )
        U4 = tf.layers.conv2d_transpose(A16, filters = 16, kernel_size = 2, strides = 2, padding = 'SAME')
        U4 = tf.concat([U4, A2],3)
        Z17 = tf.nn.conv2d(U4,W9_1, strides = [1,1,1,1], padding = 'SAME')
        A17 = tf.nn.relu(Z17)
        W9_2 =   get_variable("W9_2", [3,3,16,16] )
        Z18 = tf.nn.conv2d(A17,W9_2, strides = [1,1,1,1], padding = 'SAME')
        A18 = tf.nn.relu(Z18)
    ### Unit 10 ###
    with tf.name_scope('out_put'):
        W10 =    get_variable("W10", [1,1,16,1] )
        Z19 = tf.nn.conv2d(A18,W10, strides = [1,1,1,1], padding = 'SAME')
        A19 = tf.nn.sigmoid(Z19)
        Y_pred = A19
    return Y_pred

def loss_function(y_pred, y_true):
    cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true,y_pred))
    return cost

def mean_iou(y_pred,y_true):
    y_pred_ = tf.to_int64(y_pred > 0.5)
    y_true_ = tf.to_int64(y_true > 0.5)
    score, up_opt = tf.metrics.mean_iou(y_true_, y_pred_, 2)
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


# To train the model we can use a graph dictionary like this:

# In[ ]:


# build the graph as a dictionary
def build_graph():
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):
            with tf.name_scope('input'):
                x_ = tf.placeholder(tf.float32, shape=(None,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
                y_ = tf.placeholder(tf.float32, shape=(None,IMG_HEIGHT, IMG_WIDTH, 1))
            y_pred = UNet(x_)
            with tf.name_scope('loss'):
                loss = loss_function(y_pred,y_)
        with tf.device("/cpu:0"):
            with tf.name_scope("metrics"):
                iou = mean_iou(y_pred,y_)
        model_dict = {'graph': g, 'inputs': [x_, y_],'Iou':iou,'Loss':loss, 'y_pred':y_pred}
    return model_dict


# **prediction result:**
# 
# The model use one channel label to classify if a pixel is belong to mask or background.
# 
# After passing forward the network, the result is shown as below:
# ![result](http://i64.tinypic.com/ruy7sz.png)
# Fig[1],[2] is the ground truth and prediction of a vaildation image.
# Fig[3],[4] is the input and the prediction on test image
# 
# The overall OUI on the validation set is near 90%

# **Three quesion remain for this network:**
# 
# 1. The threshold of the prediction pixcel is hard to determined. I think the OSTU adative threshold could be useful for the prediction.
# 
# 2. The prediction would make some very sparse pixel which should not be consider as a Nuclei. Thus a Low pass filter or Markov Random Field could be apply to expel those pixels
# 
# 3. How to regenerate a reliable seperated mask from the concatenated mask. the result generated by morphology from skimage is not reliable for cell that has interception.
#  
# I would be happy if someone could share their points of view on the question above.
