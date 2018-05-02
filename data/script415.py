
# coding: utf-8

# # Intro
# 
# **This is Lesson 3 in the [Deep Learning](https://www.kaggle.com/learn/de-learning) track**  
# 
# At the end of this lesson, you will be able to write TensorFlow and Keras code to use one of the best models in computer vision.
# 
# # Lesson
# 

# In[1]:


from IPython.display import YouTubeVideo
YouTubeVideo('Epn3ryqr-F8', width=800, height=450)


# # Sample Code
# 
# ### Choose Images to Work With

# In[ ]:


from os.path import join

image_dir = '../input/dog-breed-identification/train/'
img_paths = [join(image_dir, filename) for filename in 
                           ['0246f44bb123ce3f91c939861eb97fb7.jpg',
                            '84728e78632c0910a69d33f82e62638c.jpg',
                            '8825e914555803f4c67b26593c9d5aff.jpg',
                            '91a5e8db15bccfb6cfa2df5e8b95ec03.jpg']]


# ### Function to Read and Prep Images for Modeling

# In[ ]:


import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)


# ### Create Model with Pre-Trained Weights File. Make Predictions

# In[ ]:


from tensorflow.python.keras.applications import ResNet50

my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)


# ### Visualize Predictions

# In[ ]:


import sys
# Add directory holding utility functions to path to allow importing
sys.path.append('/kaggle/input/python-utility-code-for-deep-learning-exercises/utils')
from decode_predictions import decode_predictions

from IPython.display import Image, display

most_likely_labels = decode_predictions(preds, top=3, class_list_path='../input/resnet50/imagenet_class_index.json')

for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])


# # Exercise
# Now you are ready to [use a powerful TensorFlow model](https://www.kaggle.com/dansbecker/my-first-exercise-with-tensorflow-and-keras/) yourself.
# 
# # Continue
# After the exercise, continue to learn about [Transfer Learning](https://www.kaggle.com/dansbecker/transfer-learning/).  Transfer learning will let you leverage pre-trained models for purposes far beyond what they were originally built for. Most people are amazed when they first experience the power of transfer learning.
# 
# ---
# **[Deep Learning Track Home](https://www.kaggle.com/learn/deep-learning)**
