
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten

import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# First we will read in the csv's so we can see some more information on the filenames and breeds

# In[ ]:


df_train = pd.read_csv('../input/labels.csv')
df_test = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


df_train.head(10)


# We can see that the breed needs to be one-hot encoded for the final submission, so we will now do this.

# In[ ]:


targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse = True)


# In[ ]:


one_hot_labels = np.asarray(one_hot)


# Next we will read in all of the images for test and train, using a for loop through the values of the csv files. I have also set an im_size variable which sets the size for the image to be re-sized to,  90x90 px, you should play with this number to see how it affects accuracy.

# In[ ]:


im_size = 90


# In[ ]:


x_train = []
y_train = []
x_test = []


# In[ ]:


i = 0 
for f, breed in tqdm(df_train.values):
    img = cv2.imread('../input/train/{}.jpg'.format(f))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img, (im_size, im_size)))
    y_train.append(label)
    i += 1


# In[ ]:


for f in tqdm(df_test['id'].values):
    img = cv2.imread('../input/test/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (im_size, im_size)))


# In[ ]:


y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.
x_test  = np.array(x_test, np.float32) / 255.


# We check the shape of the outputs to make sure everyting went as expected.

# In[ ]:


print(x_train_raw.shape)
print(y_train_raw.shape)
print(x_test.shape)


# 
# We can see above that there are 120 different breeds. We can put this in a num_class variable below that can then be used when creating the CNN model.

# In[ ]:


num_class = y_train_raw.shape[1]


# It is important to create a validation set so that you can gauge the performance of your model on independent data, unseen to the model in training. We do this by splitting the current training set (x_train_raw) and the corresponding labels (y_train_raw) so that we set aside 30 % of the data at random and put these in validation sets (X_valid and Y_valid).
# 
# * This split needs to be improved so that it contains images from every class, with 120 separate classes some can not be represented and so the validation score is not informative. 

# In[ ]:


X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)


# Now we build the CNN architecture. Here we are using a pre-trained model VGG19 which has already been trained to identify many different dog breeds (as well as a lot of other objects from the imagenet dataset see here for more information: http://image-net.org/about-overview). Unfortunately it doesn't seem possible to downlod the weights from within this kernel so make sure you set the weights argument to 'imagenet' and not None, as it currently is below.
# 
# We then remove the final layer and instead replace it with a single dense layer with the number of nodes corresponding to the number of breed classes we have (120).

# In[ ]:


# Create the base pre-trained model
# Can't download weights in the kernel
base_model = VGG19(#weights='imagenet',
    weights = None, include_top=False, input_shape=(im_size, im_size, 3))

# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()


# In[ ]:


model.fit(X_train, Y_train, epochs=1, validation_data=(X_valid, Y_valid), verbose=1)


# Remember, accuracy is low here because we are not taking advantage of the pre-trained weights as they cannot be downloaded in the kernel. This means we are training the wights from scratch and I we have only run 1 epoch due to the hardware constraints in the kernel.
# 
# Next we will make our predictions.

# In[ ]:


preds = model.predict(x_test, verbose=1)


# In[ ]:


sub = pd.DataFrame(preds)
# Set column names to those generated by the one-hot encoding earlier
col_names = one_hot.columns.values
sub.columns = col_names
# Insert the column id from the sample_submission at the start of the data frame
sub.insert(0, 'id', df_test['id'])
sub.head(5)

