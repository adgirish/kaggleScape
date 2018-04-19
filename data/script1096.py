
# coding: utf-8

# # Intro
# 
# **This is Lesson 4 in the [Deep Learning](https://www.kaggle.com/learn/deep-learning) track**  
# 
# At the end of this lesson, you will be able to use transfer learning to build highly accurate computer vision models for your custom purposes, even when you have relatively little data.
# 
# # Lesson
# 

# In[1]:


from IPython.display import YouTubeVideo
YouTubeVideo('mPFq5KMxKVw', width=800, height=450)


# # Sample Code
# 
# ### Specify Model

# In[2]:


from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False


# ### Compile Model

# In[3]:


my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# ### Fit Model

# In[12]:


from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
# The ImageDataGenerator was previously generated with
# data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
# recent changes in keras require that we use the following instead:
data_generator = ImageDataGenerator() 

train_generator = data_generator.flow_from_directory(
        '../input/urban-and-rural-photos/rural_and_urban_photos/train',
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '../input/urban-and-rural-photos/rural_and_urban_photos/val',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        validation_data=validation_generator,
        validation_steps=1)


# ### Note on Results:
# The printed validation accuracy can be meaningfully better than the training accuracy at this stage. This can be puzzling at first.
# 
# It occurs because the training accuracy was calculated at multiple points as the network was improving (the numbers in the convolutions were being updated to make the model more accurate).  The network was still quite when the model saw the first training images, since the weights hadn't been trained/improved much yet.  Those first training results were averaged into the measure above.
# 
# The validation loss and accuracy measures were calculated **after** the model had gone through all the data.  So the network had been fully trained when these scores were calculated.
# 
# This isn't a serious issue in practice, and we tend not to worry about it.

# # Your Turn
# [Write your own kernel to do transfer learning](https://www.kaggle.com/dansbecker/exercise-using-transfer-learning/).
# 
# # Keep Going
# After the exercise, you move on to [data augmentation](https://www.kaggle.com/dansbecker/data-augmentation/).  It's a clever (and easy) trick to improve your models.
