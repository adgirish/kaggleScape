
# coding: utf-8

# ## **Import the required libraries**

# In[ ]:


# import time
import time
t1 = time.time()


# In[ ]:


import math
import random
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Making the results reproducible (knowing the random seed of the two libraries)**

# In[ ]:


# np_rand_seed = random.randint(0,100)
# tf_rand_seed = random.randint(0,100)
np_rand_seed = 97
tf_rand_seed = 82
np.random.seed(np_rand_seed)


# # **1. Load and Inspect the data**

# In[ ]:


data = pd.read_json('../input/train.json')
test_data = pd.read_json('../input/test.json')


# In[ ]:


data.head(5)


# In[ ]:


test_data.head(5)


# In[ ]:


print("Shape of train set:", data.shape)
print("Shape of test set:", test_data.shape)


# In[ ]:


print("Shape of band 1:",  np.shape(data.band_1.iloc[0]))
print("Shape of band 2:",  np.shape(data.band_2.iloc[0]))


# In[ ]:


print("Type of band 1:",  type(data.band_1.iloc[0]))
print("Type of band 2:",  type(data.band_2.iloc[0]))


# # **2. Feature Engineering**

# ## **2.1 Feature engineering on train set**

# ### **2.1.1 Replacing the na in inc_anlge with mean**********

# In[ ]:


data[data['inc_angle']=='na'] = data[data['inc_angle']!='na']['inc_angle'].mean()


# ### **2.1.2 Converting the angle from degrees to radian******

# In[ ]:


data['inc_angle'] = data['inc_angle'].apply(lambda x: math.radians(x))


# In[ ]:


data.inc_angle.head()


# ### ** 2.1.3 Finding and droping points with mismatch band1 and band2 data**

# **Function which return the count and the index of mismatched data**

# In[ ]:


def find_missing_data(series, shape):
    
    '''function which return the count and the index of mismatched data'''    
    count = 0
    missing_list = []
    for i,x in enumerate(series):   
        if np.shape(series.iloc[i]) != shape:
            missing_list.append(i)
            count += 1
            
    return missing_list, count


# **Count and list of mismatched points in band1**

# In[ ]:


missing_list1, count1 = find_missing_data(data.band_1, (5625,))
print("count: ", count1)
print("missing data: ", missing_list1)


# **Count and list of mismatched points in band2**

# In[ ]:


missing_list2, count2 = find_missing_data(data.band_2, (5625,))
print("count: ", count1)
print("missing data: ", missing_list2)


# **Check if the missing points are same**

# In[ ]:


missing_list1 == missing_list2


# **Function to drop data by index**

# In[ ]:


def drop_data(df, index):
    
    '''function to drop data by index'''
    return df.drop(df.index[index])


# **Drop the points with mismatched images**

# In[ ]:


data = drop_data(data, missing_list1)


# In[ ]:


data.shape


# In[ ]:


print("Number of positive classes: ", len(data[data['is_iceberg'] == 1.0]))
print("Number of negative classes: ", len(data[data['is_iceberg'] == 0.0]))


# ### 2.1.4 Scale the image data

# **3 standardization to technique we can try on**

# In[ ]:


def standardise_vector(vector):
    '''standardise vector'''
    standardised_vector = (np.array(vector) - np.mean(vector)) / np.std(vector)
    return standardised_vector.tolist()


# In[ ]:


def mean_normalise_vector(vector):
    '''mean normalize vector'''
    normalised_vector = (np.array(vector) - np.mean(vector)) / (np.max(vector) - np.min(vector))
    return normalised_vector.tolist()


# In[ ]:


def min_max_scaler(vector, minimum = 0, maximum = 1):
    '''minmaxscaler'''
    X_std  = (np.array(vector) - np.min(vector)) / (np.max(vector) - np.min(vector))
    scaled_vector = X_std * (maximum - minimum) + minimum
    return scaled_vector.tolist()


# **We will use standardisation as the  normalization technique since this works well with images**

# In[ ]:


data['band_1'] = data['band_1'].apply(standardise_vector)
data['band_2'] = data['band_2'].apply(standardise_vector)


# In[ ]:


data.head(5)


# ### **2.1.5 Reshaping the band1 and band2 data into 2D image**

# In[ ]:


band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2"]])


# In[ ]:


print("Shape of band 1 image:",band_1.shape)
print("Shape of band 2 image:",band_2.shape)


# ## **2.2 Feature engieering on test Set**

# **We carry out the same feature engineering as carried out on train set**

# In[ ]:


test_data['inc_angle'] = test_data['inc_angle'].apply(lambda x: math.radians(x))


# In[ ]:


test_data.inc_angle.head()


# In[ ]:


missing_list3, count3 = find_missing_data(test_data.band_1, (5625,))
print("count: ", count3)
print("missing data: ", missing_list3)


# In[ ]:


missing_list4, count4 = find_missing_data(test_data.band_2, (5625,))
print("count: ", count4)
print("missing data: ", missing_list4)


# In[ ]:


test_data['band_1'] = test_data['band_1'].apply(standardise_vector)
test_data['band_2'] = test_data['band_2'].apply(standardise_vector)


# In[ ]:


band_1_test = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_data["band_1"]])
band_2_test = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_data["band_2"]])


# In[ ]:


print("Shape of test set band 1 image:",band_1_test.shape)
print("Shape of test set band 2 image:",band_2_test.shape)


# # **3. Train/test/validation split**

# **Extract the labels and angles of train set**

# In[ ]:


labels = data.is_iceberg.as_matrix()
angles = data.inc_angle.as_matrix()


# **Carry out splits**

# In[ ]:


# randomly choosing the train and validation indices
train_indices = np.random.choice(len(labels), round(len(labels)*0.75), replace=False)
validation_indices = np.array(list(set(range(len(labels))) - set(train_indices)))

# extract train set
band_1_train = band_1[train_indices]
band_2_train = band_2[train_indices]
angles_train = angles[train_indices]
labels_train = labels[train_indices]

# extract validation set
band_1_validation = band_1[validation_indices]
band_2_validation = band_2[validation_indices]
angles_validation = angles[validation_indices]
labels_validation = labels[validation_indices]

# extract test set
band_1_test = band_1_test
band_2_test = band_2_test
angles_test = test_data.inc_angle.as_matrix()
iD = test_data.id.as_matrix()


# **Covert the types of all data to float**

# In[ ]:


band_1_train = band_1_train.astype(np.float32)
band_1_validation = band_1_validation.astype(np.float32)
band_1_test = band_1_test.astype(np.float32)
band_2_train = band_2_train.astype(np.float32)
band_2_validation = band_2_validation.astype(np.float32)
band_2_test = band_2_test.astype(np.float32)
angles_train = angles_train.astype(np.float32)
angles_validation = angles_validation.astype(np.float32)
angles_test = angles_test.astype(np.float32)
labels_train = labels_train.astype(np.float32)
labels_validation = labels_validation.astype(np.float32)
iD = iD.astype(np.str)


# In[ ]:


# delete the unnecessary variables out of memory
del(data, test_data, band_1, band_2)


# **Examine the shape of the data**

# In[ ]:


print("Shape of band_1_train:",band_1_train.shape)
print("Shape of band_2_train:",band_1_train.shape)
print("Shape of angles_train:",angles_train.shape)
print("Shape of labels_train:",labels_train.shape)
print("Shape of band_1_validation:",band_1_validation.shape)
print("Shape of band_2_validation:",band_2_validation.shape)
print("Shape of angles_validation:",angles_validation.shape)
print("Shape of labels_validation:",labels_validation.shape)
print("Shape of band_1_test:",band_1_test.shape)
print("Shape of band_2_test:",band_2_test.shape)
print("Shape of angles_test:",angles_test.shape)
print("Shape of iD:",iD.shape)


# # **4. Augmenting train set**

# ## **4.1 Functions to carry out different augmentation technique**

# **4.1.1 Image Rotation**

# In[ ]:


def rotate_image(img, angle = 20):
    
    '''a function to rotate image by a given degree'''
    
    # rotate image
    original = img.copy()

    M_rotate = cv2.getRotationMatrix2D((37,37),angle,1)
    img_new = cv2.warpAffine(img,M_rotate,(75,75))
    
    length_row = 0
    length_column = 0
    boundary_step = 5
    
    for i in range(len(img_new)):
        if img_new[0,i]!=float(0.0):
            length_row = i
            break
    for i in range(len(img_new)):
        if img_new[i,0]!=float(0.0):
            length_column = i
            break
    
    # subsitute the padding from original image
    img_new[:length_column+boundary_step,:length_row+boundary_step] =     original[:length_column+boundary_step,:length_row+boundary_step] 
    img_new[-(length_row+boundary_step):,:length_column+boundary_step] =     original[-(length_row+boundary_step):,:length_column+boundary_step]
    img_new[:length_row+boundary_step,-(length_column+boundary_step):] =     original[:length_row+boundary_step,-(length_column+boundary_step):]
    img_new[-(length_column+boundary_step):,-(length_row+boundary_step):] =     original[-(length_column+boundary_step):,-(length_row+boundary_step):]
    
    return img_new


# **4.1.2 Horizontal translation**

# In[ ]:


def translate_horizontal(image, shift_horizontal = 5):
    
    '''a function to translate image horizontally by a shift'''
    
    # horizontally shift image
    img = image.copy()
    
    shift_vertical = 0; 
    if shift_horizontal<0:
        image_slice = img[:,shift_horizontal:].copy()
    if shift_horizontal>0:
        image_slice = img[:,:shift_horizontal].copy()
    M_translate = np.float32([[1,0,shift_horizontal],[0,1,shift_vertical]])
    img_new = cv2.warpAffine(img,M_translate,(75,75))
    
    # subsitute the padding from original image
    if shift_horizontal<0:
        img_new[:,shift_horizontal:] = image_slice
    if shift_horizontal>0:
        img_new[:,:shift_horizontal] = image_slice
        
    return img_new.reshape(75,75).astype(np.float32)


# **4.1.3 Vertical translation**

# In[ ]:


def translate_vertical(image, shift_vertical = 5):
    
    '''a function to translate image vertically by a shift'''
    
    # vertically shift image
    img = image.copy()
    
    shift_horizontal = 0;
    if shift_vertical<0:
        image_slice = img[shift_vertical:,:].copy()
    if shift_vertical>0:
        image_slice = img[:shift_vertical,:].copy()
    M_translate = np.float32([[1,0,shift_horizontal],[0,1,shift_vertical]])
    img_new = cv2.warpAffine(img,M_translate,(75,75))
    
    # subsitute the padding from original image
    if shift_vertical<0:
        img_new[shift_vertical:,:] = image_slice
    if shift_vertical>0:
        img_new[:shift_vertical,:] = image_slice
        
    return img_new.reshape(75,75).astype(np.float32)


# **4.1.4 Translation along positive diagonal**

# In[ ]:


def translate_positive_diagonal(image, shift_diagonal = 5):
    
    '''a function to translate image along positive diagonal'''
    
    # translate image along positive diagonal
    img = image.copy()
    
    if shift_diagonal<0:
        hor_slice = img[shift_diagonal:,:].copy()
        ver_slice = img[:,shift_diagonal:].copy()
    else:
        hor_slice = img[:shift_diagonal,:].copy()
        ver_slice = img[:,:shift_diagonal].copy()
    M_translate = np.float32([[1,0,shift_diagonal],[0,1,shift_diagonal]])
    img_new = cv2.warpAffine(img,M_translate,(75,75))
    
    # subsitute the padding from original image
    if shift_diagonal<0:
        img_new[shift_diagonal:,:] = hor_slice
        img_new[:,shift_diagonal:] = ver_slice
    else:
        img_new[:shift_diagonal,:] = hor_slice
        img_new[:,:shift_diagonal] = ver_slice
    
    return img_new.reshape(75,75).astype(np.float32)


# **4.1.5 Translation along negative diagonal**

# In[ ]:


def translate_negative_diagonal(image, shift_diagonal = 5):
    
    '''a function to translate image along negative diagonal'''
    
    # translate image along negative diagonal
    img = image.copy()
    
    if shift_diagonal<0:
        hor_slice = img[:-shift_diagonal,:].copy()
        ver_slice = img[:,shift_diagonal:].copy()
    if shift_diagonal>0:
        hor_slice = img[-shift_diagonal:,:].copy()
        ver_slice = img[:,:shift_diagonal].copy()
    M_translate = np.float32([[1,0,shift_diagonal],[0,1,-shift_diagonal]])
    img_new = cv2.warpAffine(img,M_translate,(75,75))
    
    # subsitute the padding from original image
    if shift_diagonal<0:
        img_new[:-shift_diagonal,:] = hor_slice
        img_new[:,shift_diagonal:] = ver_slice
    if shift_diagonal>0:
        img_new[-shift_diagonal:,:] = hor_slice
        img_new[:,:shift_diagonal] = ver_slice
        
    return img_new.reshape(75,75).astype(np.float32)


# **4.1.6 Flip Image**

# In[ ]:


def flip(image, direction = 0):
    
    '''a function to flip image'''
    img = image.copy()
    return cv2.flip(img,direction)


# **4.1.7 Zoom image**

# In[ ]:


def zoom(image, zoom_shift = 5):
    
    '''a function to zoom image'''
    
    # zoom image
    img = image.copy()
    
    # zoom in 
    if zoom_shift>0:
        # scale
        img_new = cv2.resize(img, (75+zoom_shift*2,75+zoom_shift*2)) 
        # crop
        img_new = img_new[zoom_shift:-zoom_shift,zoom_shift:-zoom_shift] 
    # zoom out
    else:
        zoom_shift *=-1
        
        hor_top = img[:zoom_shift,:]
        hor_bottom =img[-zoom_shift:,:]
        ver_left = img[:,:zoom_shift]
        ver_right = img[:,-zoom_shift:]
        
        # scale
        img_new = cv2.resize(img, (75-zoom_shift*2,75-zoom_shift*2)) 
        # zero padding
        img_new = cv2.copyMakeBorder(img_new,zoom_shift,zoom_shift,zoom_shift,zoom_shift,
                                     cv2.BORDER_CONSTANT,value=0.0)
        # subsitute the padding from original image
        img_new[:zoom_shift,:] = hor_top
        img_new[-zoom_shift:,:] = hor_bottom
        img_new[:,:zoom_shift] = ver_left
        img_new[:,-zoom_shift:] = ver_right     
        
    return img_new.reshape(75,75).astype(np.float32)


# ## **4.2 Displaying augmented samples**

# In[ ]:


matplotlib.rcParams['figure.figsize'] = (20.0, 14.0)
image = band_1_test[3].copy()
plt.subplot(3, 5, 1)
plt.title("Original Image")
plt.imshow(image)
plt.subplot(3, 5, 2)
generated_image = rotate_image(image,40)
plt.title("Rotation by +ve degree")
plt.imshow(generated_image)
plt.subplot(3, 5, 3)
generated_image = rotate_image(image,-40)
plt.title("Rotation by -ve degree")
plt.imshow(generated_image)
plt.subplot(3, 5, 4)
generated_image = translate_horizontal(image,10)
plt.title("Horizonation translation to right")
plt.imshow(generated_image)
plt.subplot(3, 5, 5)
generated_image = translate_horizontal(image,-10)
plt.title("Horizonation translation to left")
plt.imshow(generated_image)
plt.subplot(3, 5, 6)
generated_image = translate_vertical(image,10)
plt.title("Vertical translation downward")
plt.imshow(generated_image)
plt.subplot(3, 5, 7)
generated_image = translate_vertical(image,-10)
plt.title("Vertical translation upward")
plt.imshow(generated_image)
plt.subplot(3, 5, 8)
generated_image = translate_positive_diagonal(image,10)
plt.title("SE translation")
plt.imshow(generated_image)
plt.subplot(3, 5, 9)
generated_image = translate_positive_diagonal(image,-10)
plt.title("NW translation")
plt.imshow(generated_image)
plt.subplot(3, 5, 10)
generated_image = translate_negative_diagonal(image,10)
plt.title("NE translation")
plt.imshow(generated_image)
plt.subplot(3, 5, 11)
generated_image = translate_negative_diagonal(image,-10)
plt.title("SW translation")
plt.imshow(generated_image)
plt.subplot(3, 5, 12)
generated_image = flip(image,0)
plt.title("Vertical flip")
plt.imshow(generated_image)
plt.subplot(3, 5, 13)
generated_image = flip(image,1)
plt.title("Horizontal flip")
plt.imshow(generated_image)
plt.subplot(3, 5, 14)
generated_image = zoom(image,10)
plt.title("Zoom in")
plt.imshow(generated_image)
plt.subplot(3, 5, 15)
generated_image = zoom(image,-10)
plt.title("Zoom out")
plt.imshow(generated_image)
plt.show()


# ## **4.3 Augmentation of train set**

# In[ ]:


def augment_data(band1, band2, angles, labels):
    
    '''a function to augment band1 and band2 image'''
    
    # list to store the generated data
    band1_generated = []
    band2_generated = []
    angles_generated = []
    labels_generated = []
    
    # iterate through each point in train set
    for i in range(labels.shape[0]):
        
        # rotate by positive degree
        angle = np.random.randint(5,20)
        band1_generated.append(rotate_image(band1[i],angle)) 
        band2_generated.append(rotate_image(band2[i],angle))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # rotate by negative degree
        angle = np.random.randint(5,20)
        band1_generated.append(rotate_image(band1[i],-angle)) 
        band2_generated.append(rotate_image(band2[i],-angle))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # positive horizontal shift
        shift = np.random.randint(3,7)
        band1_generated.append(translate_horizontal(band1[i],+shift)) 
        band2_generated.append(translate_horizontal(band2[i],+shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # negative horizontal shift
        shift = np.random.randint(3,7) 
        band1_generated.append(translate_horizontal(band1[i],-shift)) 
        band2_generated.append(translate_horizontal(band2[i],-shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # positive vertical shift
        shift = np.random.randint(0,7)  
        band1_generated.append(translate_vertical(band1[i],+shift)) 
        band2_generated.append(translate_vertical(band2[i],+shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # negative vertical shift
        shift = np.random.randint(3,7) 
        band1_generated.append(translate_vertical(band1[i],-shift)) 
        band2_generated.append(translate_vertical(band2[i],-shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # translate along positive diagonal in positive direction
        shift = np.random.randint(3,7)  
        band1_generated.append(translate_positive_diagonal(band1[i],+shift)) 
        band2_generated.append(translate_positive_diagonal(band2[i],+shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # translate along positive diagonal in negative direction
        shift = np.random.randint(3,7)  
        band1_generated.append(translate_positive_diagonal(band1[i],-shift)) 
        band2_generated.append(translate_positive_diagonal(band2[i],-shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # translate along negative diagonal in positive direction
        shift = np.random.randint(3,7)   
        band1_generated.append(translate_negative_diagonal(band1[i],+shift)) 
        band2_generated.append(translate_negative_diagonal(band2[i],+shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # translate along negative diagonal in negative direction
        shift = np.random.randint(3,7)   
        band1_generated.append(translate_negative_diagonal(band1[i],-shift)) 
        band2_generated.append(translate_negative_diagonal(band2[i],-shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # vertical flip
        band1_generated.append(flip(band1[i],0)) 
        band2_generated.append(flip(band2[i],0))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # horizontal flip
        band1_generated.append(flip(band1[i],1)) 
        band2_generated.append(flip(band2[i],1))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # zoom in image
        zoom_shift = np.random.randint(2,5)
        band1_generated.append(zoom(band1[i],zoom_shift)) 
        band2_generated.append(zoom(band2[i],zoom_shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])
        
        # zoom out image
        zoom_shift = np.random.randint(2,5) 
        band1_generated.append(zoom(band1[i],-zoom_shift)) 
        band2_generated.append(zoom(band2[i],-zoom_shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])        
        
    # convert the generated data into numpy array
    band1_generated = np.array(band1_generated)
    band2_generated = np.array(band2_generated)
    angles_generated = np.array(angles_generated)
    labels_generated = np.array(labels_generated)
    
    # concatenate the generated data to original train set
    band1_augmented = np.concatenate((band1, band1_generated),axis=0)
    band2_augmented = np.concatenate((band2, band2_generated),axis=0)
    angles_augmented = np.concatenate((angles, angles_generated),axis=0)
    labels_augmented = np.concatenate((labels, labels_generated),axis=0)
    
    return band1_augmented, band2_augmented, angles_augmented, labels_augmented


# In[ ]:


# augment train set
band_1_train, band_2_train, angles_train, labels_train =     augment_data(band_1_train, band_2_train, angles_train, labels_train)


# **Examine the shape of augmented data**

# In[ ]:


print("Shape of band_1_train:",band_1_train.shape)
print("Shape of band_2_train:",band_2_train.shape)
print("Shape of angles_train:",angles_train.shape)
print("Shape of labels_train:",labels_train.shape)


# # **5. Concatenate the band1 and band2 data into 3D image**

# **Here we stack band_1, band_2, and average of the two to create a 3D image**

# In[ ]:


image_train = np.concatenate([band_1_train[:, :, :, np.newaxis],
                             band_2_train[:, :, :, np.newaxis],
                             ((band_1_train+band_2_train)/2)[:, :, :, np.newaxis]],
                             axis=-1)


# In[ ]:


image_validation = np.concatenate([band_1_validation[:, :, :, np.newaxis],
                             band_2_validation[:, :, :, np.newaxis],
                             ((band_1_validation+band_2_validation)/2)[:, :, :, np.newaxis]],
                             axis=-1)


# In[ ]:


image_test = np.concatenate([band_1_test[:, :, :, np.newaxis],
                             band_2_test[:, :, :, np.newaxis],
                             ((band_1_test+band_2_test)/2)[:, :, :, np.newaxis]],
                             axis=-1)


# In[ ]:


# delete the unnecessary variables out of memory
del(band_1_train, band_1_validation, band_1_test, band_2_train, band_2_validation, band_2_test)


# **Examine the shape of 3D images**

# In[ ]:


print("Shape of image_train:",image_train.shape)
print("Shape of image_validation:",image_validation.shape)
print("Shape of image_test:",image_test.shape)


# # **6. Creating Convolutional Neural Network**

# **Import tensorflow and reset default graph**

# In[ ]:


import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
tf.set_random_seed(tf_rand_seed)
# sess = tf.InteractiveSession()


# ## **6.1 One hot encoding labels**

# In[ ]:


labels_train = pd.get_dummies(labels_train).as_matrix()
labels_validation = pd.get_dummies(labels_validation).as_matrix()


# In[ ]:


print("Shape of labels_train:", labels_train.shape)
print("Shape of labels_validation:", labels_validation.shape)


# ## **6.2 Create placeholders**

# In[ ]:


# image dimensions
width = 75
height = 75
num_channels = 3
flat = width * height
num_classes = 2


# **Create placeholder for image, labels,  dropout keep probability, and optionally angle**

# In[ ]:


image = tf.placeholder(tf.float32, shape=[None, height, width, num_channels])
# angle = tf.placeholder(tf.float32, shape= [None, 1])
y_true = tf.placeholder(tf.int32, shape=[None, num_classes])
keep_prob = tf.placeholder(tf.float32)


# ## **6.3 Create functions for creating deep learning layers**

# In[ ]:


def create_weights(shape):
    '''a function to create weight tensor'''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
 
def create_biases(size):
    '''a function to create bias tensor'''
    return tf.Variable(tf.constant(0.05, shape=[size]))


# In[ ]:


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               max_pool_filter_size,
                               num_filters):  
    
    '''a function to create convoutional layer'''
    
    # create filter for the convolutional layer
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    
    # create biases
    biases = create_biases(num_filters)
    
    # create covolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
    
    # add the bias to the convolutional layer
    layer += biases
    
    # relu activation layer fed into layer
    layer = tf.nn.relu(layer)
    
    # max pooling to half the size of the image
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, max_pool_filter_size, max_pool_filter_size, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
        
    # return the output layer of the convolution
    return layer


# In[ ]:


def create_flatten_layer(layer):
    
    '''a function for creating flattened layer from convolutional output'''
    
    # extract the shape of the layer
    layer_shape = layer.get_shape()
    # calculate the number features of the flattened layer
    num_features = layer_shape[1:4].num_elements()
    # create the flattened layer
    layer = tf.reshape(layer, [-1, num_features])
    # return the layer
    return layer


# In[ ]:


def create_fc_layer(input,          
                    num_inputs,    
                    num_outputs,
                    use_relu=True,
                    dropout = False, 
                    keep_prob = 0.2):
    
    '''a function for creating fully connected layer'''
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    
    # matrix multiplication between input and weight matrix
    layer = tf.matmul(input, weights) + biases
    
    # add relu activation if wanted
    if use_relu:
        layer = tf.nn.relu(layer)
        
    # if dropout is wanted add dropout
    if dropout:        
        layer = tf.nn.dropout(layer, keep_prob)
    
    # return layer
    return layer


# ## **6.4 Create Layers of Covnet**

# In[ ]:


# paramters for 1st convolutional layer
conv1_features = 64
conv1_filter_size = 3
max_pool_size1 = 2

# paramters for 2nd convolutional layer
conv2_features = 128
conv2_filter_size = 3
max_pool_size2 = 2

# paramters for 3rd convolutional layer
conv3_features = 128
conv3_filter_size = 3
max_pool_size3 = 2

# paramters for 4th convolutional layer
conv4_features = 64
conv4_filter_size = 3
max_pool_size4 = 2

# number of featuers of 1st fully connected layer
fc_layer_size1 = 512

# number of featuers of 2nd fully connected layer
fc_layer_size2 = 256


# **Create convolutional layer 1**

# In[ ]:


layer_conv1 = create_convolutional_layer(input=image,
                                         num_input_channels= num_channels,
                                         conv_filter_size = conv1_filter_size,
                                         max_pool_filter_size = max_pool_size1,
                                         num_filters = conv1_features)
layer_conv1


# **Create convolutional layer 2**

# In[ ]:


layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels= conv1_features,
                                         conv_filter_size = conv2_filter_size,
                                         max_pool_filter_size = max_pool_size2,
                                         num_filters = conv2_features)
layer_conv2


# **Create convolutional layer 3**

# In[ ]:


layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels= conv2_features,
                                         conv_filter_size = conv3_filter_size,
                                         max_pool_filter_size = max_pool_size3,
                                         num_filters = conv3_features)
layer_conv3


# **Create convolutional layer 4**

# In[ ]:


layer_conv4 = create_convolutional_layer(input=layer_conv3,
                                         num_input_channels= conv3_features,
                                         conv_filter_size = conv4_filter_size,
                                         max_pool_filter_size = max_pool_size4,
                                         num_filters = conv4_features)
layer_conv4


# **Flatten the output of last convolutional layer**

# In[ ]:


layer_flat = create_flatten_layer(layer_conv4)
layer_flat


# **Create a connected layer for angle and concat this with the fully connected layer (OPTIONAL)**

# In[ ]:


# layer_angle = create_fc_layer(input = angle,
#                               num_inputs=1,
#                               num_outputs=1,
#                               use_relu= True)


# In[ ]:


# combined_layer = tf.concat((layer_flat, layer_angle), axis=1)


# In[ ]:


# layer_fc1 = create_fc_layer(input=combined_layer,
#                             num_inputs=combined_layer.get_shape()[1:4].num_elements(),
#                             num_outputs=fc_layer_size1,
#                             use_relu=True,
#                             dropout =True,
#                             keep_prob = keep_prob)


# **Create the first fully connected layer**

# In[ ]:


layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size1,
                            use_relu=True,
                            dropout =True,
                            keep_prob = keep_prob)
layer_fc1


# **Create the second  fully connected layer**

# In[ ]:


layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size1,
                            num_outputs=fc_layer_size2,
                            use_relu=True,
                            dropout =True,
                            keep_prob = keep_prob)
layer_fc2


# **Create the output layer**

# In[ ]:


output_layer = create_fc_layer(input=layer_fc2,
                     num_inputs = fc_layer_size2,
                     num_outputs = num_classes,
                     use_relu=False)
output_layer


# ## **6.5 Create prediction & accuracy metric**

# In[ ]:


# softmax operation on the output layer
y_pred = tf.nn.softmax(output_layer)
# extract the vector of predicted class
y_pred_cls = tf.argmax(y_pred, axis=1, output_type=tf.int32)
# extract the vector of labels
y_true_cls = tf.argmax(y_true, axis=1, output_type=tf.int32)


# In[ ]:


# extract the vector of correct prediction
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# operation to calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ## **6.6 Create Optimizer**

# In[ ]:


# operation to calculate cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer,
                                                    labels=y_true)
# mean of cross entropy to act as the loss
loss = tf.reduce_mean(cross_entropy)


# In[ ]:


# sess.run(tf.global_variables_initializer())
# loss.eval(feed_dict={image: image_validation,
#                          angle: np.transpose([angles_validation]),
#                          y_true: labels_validation, keep_prob: 1.0})


# In[ ]:


# learning rate of optimizer
learning_rate = (1e-3)*0.30
# train step
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# # **7. Train Model**

# In[ ]:


# lists to store the train loss, validation loss, validation accuracy at each iteration
train_loss = []
valid_loss = []
valid_acc = []

# batch size
batch_size = 255
# max iteration
max_iter = 700


# **Here we train and save the model with the highest accuracy or lowest loss. But here I think it is wise to save the model with lowest loss**

# In[ ]:


# create a saver object
saver = tf.train.Saver(max_to_keep=1)

# variables to store the accuracy, loss, iteration of our best model
best_accuracy = 0
best_loss = 1000000
best_iteration = None

iteration = 0

# create a graph session and optimize under it
with tf.Session() as sess:
    
    # initialize variables
    sess.run(tf.global_variables_initializer())

    # while 57 minutes have not elapsed (to finish before the kernel is killed)
    while (time.time()-t1) < 3420:
        
        # break if max iteration is reached
        if iteration >= max_iter:
            break

        # randomly choosing the indices of the batch 
        rand_index = np.random.choice(labels_train.shape[0], size=batch_size)

        # extract the batch image and labels
        image_rand = image_train[rand_index]
#         angles_rand = angles_train[rand_index]
        labels_rand = labels_train[rand_index]

        # feed dictionary for batch
        feed_dict_batch =  {image: image_rand,
#                             angle: np.transpose([angles_rand]),
                            y_true: labels_rand,
                            keep_prob: 0.7}
        # feed dictionary for train
        feed_dict_train =  {image: image_rand,
#                             angle: np.transpose([angles_rand]),
                            y_true: labels_rand,
                            keep_prob: 1.0}
        # feed dictionary for validation
        feed_dict_validation =  {image: image_validation,
#                                  angle: np.transpose([angles_validation]),
                                 y_true: labels_validation,
                                 keep_prob: 1.0}
        
        # execute optimization step
        sess.run(train_step, feed_dict=feed_dict_batch)

        # calculate temporary train loss and append it to the designated list
        temp_train_loss = loss.eval(session=sess, feed_dict=feed_dict_train)
        train_loss.append(temp_train_loss)
        # calculate temporary validation loss and append it to the designated list
        temp_validation_loss = loss.eval(session=sess, feed_dict=feed_dict_validation)
        valid_loss.append(temp_validation_loss)
        # calculate temporary validation accuracy and append it to the designated list
        temp_validation_accuracy = accuracy.eval(session=sess, feed_dict=feed_dict_validation)
        valid_acc.append(temp_validation_accuracy)

        # if the valid loss is tied with best recorded so far but valid acc is better then
        # update the parameters of the best model and save the model
        if (temp_validation_loss == best_loss) and (temp_validation_accuracy > best_accuracy):
            best_accuracy = temp_validation_accuracy
            best_loss = temp_validation_loss
            best_iteration = iteration           
            saver.save(sess, './my-model', global_step = best_iteration)
        
        # if valid accuracy is better than best recorded so far then update the best valid accuracy
        if temp_validation_accuracy > best_accuracy:
            best_accuracy = temp_validation_accuracy
        
        # if valid loss is better than best recorded so far then
        # update the parameters of the best model and save the model
        if temp_validation_loss < best_loss:
            best_loss = temp_validation_loss
            best_iteration = iteration          
            saver.save(sess, './my-model', global_step = best_iteration)

        # print metric info
        print("iterations:",iteration,
              "| train_loss:", temp_train_loss,
              "| validation_loss:", temp_validation_loss,
              "| valid_accuracy:", temp_validation_accuracy)
        
        # increment iteration
        iteration = iteration+1


# In[ ]:


# delete unnecessary variables out of memory
del(image_train, image_validation, angles_train, angles_validation, labels_train, labels_validation)


# # **8. Save the submission and performance metrics of our best model**

# In[ ]:


# t5 = time.time()

with tf.Session() as sess:    
    
    # restore the best model
    model_path = "./"+"my-model-"+str(best_iteration)
    saver.restore(sess, model_path)
    
    # break the test set into k folds other wise kernel will be out of memory
    n = len(iD)
    k = 12
    step = n//k
    
    # array to store the prediction
    preds = np.array([])

    # iterate through each fold
    for i in range(k):

        # start and end indices of the fold
        start = (step*i)
        end = (step*(i+1)) 
    
        # feed dictionary for the fold
        feed_dict_test =  {image: image_test[start:end],
#                            angle: np.transpose([angles_test[start:end]]),
                           keep_prob: 1.0}

        # evaluate predictions of the fold
        fold_preds = y_pred.eval(session=sess, feed_dict = feed_dict_test)[:,1]
        # append the predictions of the fold to the designated array
        preds = np.append(preds, fold_preds)
    
    # save the submission csv file
    submission_path = "./submission.csv"
    submission = pd.DataFrame({"id": iD, "is_iceberg": preds})
    submission.to_csv(submission_path, header = True, index=False)
    
    # save the csv file containing performance metrics of the best model 
    results = pd.DataFrame([int(best_iteration),train_loss[best_iteration],
                            valid_loss[best_iteration], valid_acc[best_iteration]],
                           index=["iteration", "train loss", "valid loss", "accuracy"],
                           columns = ["results"])    
    results_path = "./results.csv"    
    results.to_csv(results_path, header = True, index=True)
    
# t6 = time.time()
# print("time take for prediction: ", t6-t5)


# # **9. Visualization of the performance**

# ## **9.1 Plot of loss over iteration**

# In[ ]:


plt.figure(figsize=(16, 8), dpi= 80, facecolor='w', edgecolor='k')
iterations = list(range(1,iteration+1))
plt.plot(iterations, train_loss, label = "train loss")
plt.plot(iterations, valid_loss, label = "valid loss")
plt.title("Loss")
plt.xlabel("iter")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()


# ## **9.2 Plot of training accuracy over iteration**

# In[ ]:


plt.figure(figsize=(16, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(iterations, valid_acc, label = "train loss")
plt.title("Accuracy")
plt.xlabel("iter")
plt.ylabel("accuracy")
plt.grid()
plt.show()


# # **10. Advice**

# Looking at the plot it is safe to say that the loss would have gone lower if we increase the number of iterations. I would advise to train the model using higher computational power to decrease iteration time and increase kernel time, an easy way to do that is by running the script on google cloud (such as dataflow) or amazon web service instances having high memory. Another advice I would give if you take the first advise is to increase the batch size to 500 to stabalize the optimization.
