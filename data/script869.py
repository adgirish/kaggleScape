
# coding: utf-8

# # Intro
# **This is Lesson 7 in the [Deep Learning](https://www.kaggle.com/learn/machine-learning) track**  
# 
# The models you've built so far have relied on pre-trained models.  But they aren't the ideal solution for many use cases.  In this lesson, you will learn how to build totally new models.
# 
# # Lesson
# 

# In[1]:


from IPython.display import YouTubeVideo
YouTubeVideo('YbNE3zhtsoo', width=800, height=450)


# # Sample Code

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

a=pd.read_csv("../input/digit-recognizer/train.csv")
a.drop('label', axis=1)

img_rows, img_cols = 28, 28
num_classes = 10

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

train_size = 30000
train_file = "../input/digit-recognizer/train.csv"
raw_data = pd.read_csv(train_file)

x, y = data_prep(raw_data)

model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)


# # Your Turn
# You are ready to [build your own model](https://www.kaggle.com/dansbecker/exercise-modeling-from-scratch).
# 
# # Keep Going
# [Move on](https://www.kaggle.com/dansbecker/dropout-and-strides-for-larger-models) to learn two techniques that will make your models run faster and more robust to overfitting.
