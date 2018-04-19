
# coding: utf-8

# # Intro
# 
# **This is Lesson 5 in the [Deep Learning](https://www.kaggle.com/learn/machine-learning) track**  
# 
# At the end of this lesson, you will be able to use data augmentation. This trick that makes it seem like you have far more data than you actually have, resulting in even better models..
# 
# # Lesson
# 

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo('ypt_BAotCLo', width=800, height=450)


# # Sample Code
# 
# We have some model set-up code which you've seen before.  It's not our focus for the moment, so it is hidden (but optionally visible by clicking the "code" button below.)

# In[ ]:


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

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# ### Fitting a Model With Data Augmentation

# In[ ]:


from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
# The ImageDataGenerator previously specified preprocessing_function=preprocess_input
# recent changes in keras require that we remove that argument

data_generator_with_aug = ImageDataGenerator( # preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

train_generator = data_generator_with_aug.flow_from_directory(
        '../input/urban-and-rural-photos/rural_and_urban_photos/train',
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

data_generator_no_aug = ImageDataGenerator() # preprocessing_function=preprocess_input)
validation_generator = data_generator_no_aug.flow_from_directory(
        '../input/urban-and-rural-photos/rural_and_urban_photos/val',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=1)


# # Exercise
# Move on to [apply data augmentation](https://www.kaggle.com/dansbecker/exercise-data-augmentation) yourself.
# 
# # Continue
# After the exercise, move on for [a deeper understanding of deep learning](https://www.kaggle.com/dansbecker/a-deeper-understanding-of-deep-learning/).
