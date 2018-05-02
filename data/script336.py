
# coding: utf-8

# ![flowers](https://media.giphy.com/media/6igrowTjYLNvi/giphy.gif)

# Hello Kagglers. Working on Kaggle kernels is fun. The purpose of this kernel is totally different. Here I am not going to do a typical EDA or typical data modelling but I would love to share some cool things. We are going to dive into following topics:
# * How to add pre-trained Keras models to your kernel and answer the question **Why do I need to do that at all?**
# * What generator should I use for my model- inbuilt or a custom one?
# * How to effectively use Keras ImageDataGenerator in kernels?

# ## Adding Keras pre-trained models to your kernel
# 
# Transfer learning (Here I am assuming that you know about it) **almost always** works. Before doing some serious modelling, people like me always starts with transfer learning to get a baseline. For this, we need pre-trained models. Keras provides a lot of SOTA pre-trained models. When you want to use a pre-trained architecture for the first time, Keras download the weights for the corresponding model *but* Kernels can't use network connection to download pretrained keras model weights. So, the big question is `If Kernels can't use network connection to download pre-trained weights, how can I use them at all?` 
# 
# This is a great question and for people who are beginners or just getting started on Kaggle kernels, this can be very confusing. In order to use, pre-trained Keras model weights, people have uploaded the weights to a kernel and published it. Now here is the catch. **You can add the output of any other kernel as input data source for your kernel **. Follow these simple steps:
# * On the top-left of your notebook, there is a `Input Files` cell. Expand it by clicking the `+` button.
# * You will see a list of input data files on the left along with the description of the data on the right.
# * Click the add `Add Data Source` button. A window will appear.
# * In the search bar, search like this `VGG16 pretrained` or `Keras-pretrained`.
# * Choose the kernel you want to add. That's it!!
# 
# Now if you expand your `Input Files` cell again, you will the pre-trained model as input files along with your dataset.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import glob
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# You can see that my kernel has two kind of input files:
# * flowers-recognition dataset
# * vgg16 pre-trained model kernel that I added to my kernel
# 
# Keras requires the pre-trained weights to be present in the `.keras/models` cache directory. This is how you do it

# In[ ]:


# Check for the directory and if it doesn't exist, make one.
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
    
# make the models sub-directory
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)


# In[ ]:


# Copy the weights from your input files to the cache directory
get_ipython().system('cp ../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 ~/.keras/models/')


# That's it!! Now, you can use pre-trained models for transfer learning or fine-tuning. 

# In[ ]:


# Define some paths
input_path = Path('../input/flowers-recognition/flowers/')
flowers_path = input_path / 'flowers'


# In[ ]:


# Each species of flower is contained in a separate folder . Get all the sub directories
flower_types = os.listdir(flowers_path)
print("Types of flowers found: ", len(flower_types))
print("Categories of flowers: ", flower_types)


# In[ ]:


# In order to keep track of my data details or in order to do some EDA, I always try to 
# get the information in a dataframe. After all, pandas to the rescue!!

# A list that is going to contain tuples: (species of the flower, corresponding image path)
flowers = []

for species in flower_types:
    # Get all the file names
    all_flowers = os.listdir(flowers_path / species)
    # Add them to the list
    for flower in all_flowers:
        flowers.append((species, str(flowers_path /species) + '/' + flower))

# Build a dataframe        
flowers = pd.DataFrame(data=flowers, columns=['category', 'image'], index=None)
flowers.head()


# In[ ]:


# Let's check how many samples for each category are present
print("Total number of flowers in the dataset: ", len(flowers))
fl_count = flowers['category'].value_counts()
print("Flowers in each category: ")
print(fl_count)


# In[ ]:


# Let's do some visualization too
plt.figure(figsize=(12,8))
sns.barplot(x=fl_count.index, y=fl_count.values)
plt.title("Flowers count for each category", fontsize=16)
plt.xlabel("Category", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.show()


# In[ ]:


# Let's visualize flowers from each category

# A list for storing names of some random samples from each category
random_samples = []

# Get samples fom each category 
for category in fl_count.index:
    samples = flowers['image'][flowers['category'] == category].sample(4).values
    for sample in samples:
        random_samples.append(sample)



# Plot the samples
f, ax = plt.subplots(5,4, figsize=(15,10))
for i,sample in enumerate(random_samples):
    ax[i//4, i%4].imshow(mimg.imread(random_samples[i]))
    ax[i//4, i%4].axis('off')
plt.show()    


# **What generator should I use for my model-  a custom one or the default Keras ImageDataGenerator?**
# 
# This is a very interesting question. I would say that it actually depends on how your dataset is arranged or how are you going to set up your data. These are the following scenarios I can think of along with the corresponding solutions. If you think of any more, do let me know in the comments section.
# 
# * **Data is arranged class-wise in separate directories with corresponding names**: This is the best way to arrange your data, if possible. Although it takes some time to arrange the data in such a way but it is the way to go if you want to use the Keras ImageDataGenerator efficiently as it requires data to be separated class wise in different folders. Once you have this, you need to arrange your data like this:
# ```
# data/
#     train/
#         category1/(contains all images related to category1)  
#         category2/(contains all images related to category2)
#         ...
#         ...
#             
#     validation/
#          category1/(contains all images related to category1)  
#         category2/(contains all images related to category2)
#         ...
#         ...
# ```
# For this kernel, later in the notebook, I will show how to make this structure within the kernel for using ImageDataGenerator
# 
# * **All data is within one folder and you have meta info about the images** This is a very usual case. When we quickly crawl data, we generally store the met info about the images in a csv and allthe images are stored in a single folder. There are two ways to deal with this situatio, provided you don't want all the segregation of images as in the first step.
#   * Define your own simple python generator which yields batches of images and labels while reading the csv
#   * Use another high-level api such as `Dataset` api and let it do the work for you. 

# Let's look at how to get the structure defined in the  first step above. If you are not aware, jupyter is pretty powerful and you can use bash directly within the notebook.

# In[ ]:


# Make a parent directory `data` and two sub directories `train` and `valid`
get_ipython().run_line_magic('mkdir', '-p data/train')
get_ipython().run_line_magic('mkdir', '-p data/valid')

# Inside the train and validation sub=directories, make sub-directories for each catgeory
get_ipython().run_line_magic('cd', 'data')
get_ipython().run_line_magic('mkdir', '-p train/daisy')
get_ipython().run_line_magic('mkdir', '-p train/tulip')
get_ipython().run_line_magic('mkdir', '-p train/sunflower')
get_ipython().run_line_magic('mkdir', '-p train/rose')
get_ipython().run_line_magic('mkdir', '-p train/dandelion')

get_ipython().run_line_magic('mkdir', '-p valid/daisy')
get_ipython().run_line_magic('mkdir', '-p valid/tulip')
get_ipython().run_line_magic('mkdir', '-p valid/sunflower')
get_ipython().run_line_magic('mkdir', '-p valid/rose')
get_ipython().run_line_magic('mkdir', '-p valid/dandelion')

get_ipython().run_line_magic('cd', '..')

# You can verify that everything went correctly using ls command


# For each category, copy samples to the train and validation directory which we defined in the above step. The number of samples you want in your training and validation set is upto you. 

# In[ ]:


for category in fl_count.index:
    samples = flowers['image'][flowers['category'] == category].values
    perm = np.random.permutation(samples)
    # Copy first 30 samples to the validation directory and rest to the train directory
    for i in range(30):
        name = perm[i].split('/')[-1]
        shutil.copyfile(perm[i],'./data/valid/' + str(category) + '/'+ name)
    for i in range(31,len(perm)):
        name = perm[i].split('/')[-1]
        shutil.copyfile(perm[i],'./data/train/' + str(category) + '/' + name)


# In[ ]:


# Define the generators

batch_size = 8
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # more than two classes

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/valid',
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='categorical')


# In[ ]:


def get_model():
    # Get base model 
    base_model = VGG16(include_top=False, input_shape=(150,150,3))
    # Freeze the layers in base model
    for layer in base_model.layers:
        layer.trainable = False
    # Get base model output 
    base_model_ouput = base_model.output
    
    # Add new layers
    x = Flatten()(base_model.output)
    x = Dense(500, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(5, activation='softmax', name='fc2')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model


# In[ ]:


# Get the model
model = get_model()
# Compile it
opt = Adam(lr=1e-3, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#Summary
model.summary()


# In[ ]:


# Fit the genertor 
model.fit_generator(
        train_generator,
        steps_per_epoch=4168 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=150 // batch_size)


# That's all folks. I hope you enjoyed this. One last thing: Kaggle kernels doesn't provide you GPU, so the training time will depend on your architecture and size of your dataset. Also, if you find this kernel helpful, please upvote!!
