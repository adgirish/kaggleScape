
# coding: utf-8

# # Exercise Introduction
# 
# The cameraman who shot our deep learning videos mentioned a frustrating problem that we could solve with deep learning.  
# 
# He offers a service that scans photographs and slides to store them digitally.  He uses a machine that quickly scans many photos. But depending on the orientation of the original photo, many images are digitized sideways.  He currently spends a lot of time looking find which photos need to be rotated sideways, so he can fix them.
# 
# It would save him a lot of time if this process could be automated.  In this exercise, you will build a model that distinguishes which photos are sideways and which are upright.
# 
# If you were going to sell this service commercially, you might use a large dataset to train the model. But we'll have great success with even a small dataset.  We'll work with a small dataset of dog pictures, half of which are rotated sideways.
# 
# Specifying and compiling the model look the same as in the example you've seen. But you'll need to make some changes to fit the model.
# 
# # 1) Specify the Model
# 
# Since this is your first time, you won't yet be able to create this from scratch. 
# 
# We've filled in most of the code you'll need, but left some critical pieces blank.  
# 
# Fill in the blanks (marked with `____`) in the  code below. Then uncomment those lines and run the cell.
# 

# In[ ]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

# num_classes is the number of categories your model chooses between for each prediction
# num_classes = ____
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# my_new_model = Sequential()
# my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
# my_new_model.add(Dense(num_classes, activation='softmax'))

# The value below is either True or False.  If you choose the wrong answer, your modeling results
# won't be very good.  Recall whether the first layer should be trained/changed or not.
# my_new_model.layers[0].trainable = ____


# # 2) Compile the Model
# 
# We have again provided most of the code, and left one extremely important piece blank. Fill in the blank (marked with `____`).  Then uncomment that line of code and run the cell.

# In[ ]:


# We are calling the compile command for some python object. 
# Which python object is being compiled? Fill in the answer so the compile command works.
# ____.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# # 3) Fit Model
# 
# **Your training data is in the directory `../input/dogs-gone-sideways/images/train`. The validation data is in `../input/dogs-gone-sideways/images/val`**. Use that information when setting up `train_generator` and `validation_generator`.
# 
# You have 220 images of training data and 217 of validation data.  For the training generator, choose a batch size of 10. Figure out the appropriate value of `steps_per_epoch` in your `fit_generator` call?   It isn't the same as in the example.
# 
# Fill in all the blanks (again marked as `____`).  Then uncomment each line and run the cell of code.  Watch as your model trains the weights and the accuracy improves.

# In[ ]:


from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

#train_generator = data_generator.flow_from_directory(
#        directory = ____,
#        target_size=(image_size, image_size),
#        batch_size=____,
#        class_mode='categorical')

#validation_generator = data_generator.flow_from_directory(
#        directory = ____,
#        target_size=(image_size, image_size),
#        class_mode='categorical')

#my_new_model.fit_generator(
#        train_generator,
#        steps_per_epoch=____,
#        validation_data=____,
#        validation_steps=1)


# 
# Can you tell from the results what fraction of the time your model was correct in the validation data? 
# 
# In the next step, we'll see if we can improve on that.
# 
# # Keep Going
# Move on to learn about [data augmentation](https://www.kaggle.com/dansbecker/data-augmentation/).  It is a clever and easy way to improve your models. Then you'll apply data augmentation to this automatic image rotation problem.
# 
