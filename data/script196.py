
# coding: utf-8

# # 0. First of all
# 
# This kernel is the tutorial to explore and visualize datasets and train Convolutional Neural Network (CNN) on keras.
# 
# I'm not good at English. So, **please post a comment if there are any unknown points:)**  
# 
# Additionally, this kernel is unfinished, still writing. I will do my best!  
# This kernel is getting better little by little. **Many thanks to all of the comments.**

# # 1. Environment construction
# 
# In this kernel, you will mainly compute with python on Colfax Cluster.  
# 
# The datasets (test, train, additional) had extracted and placed in /data/kaggle/ directory on Colfax Cluster.  
# So, you don't have to download datasets on your local machine.  
# Of course, you can download them while reading this kernel for killing time.
# 
# Datasets are here: 
# [https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data)
# 
# Note:  
# You can use extracted datasets on kaggle's kernel too.  
# But I recommend using Colfax Cluster from the point of computing speed and making your original submission.

# ## 1-1. Setting ssh connection to Colfax Cluster
# 
# Sign up to Colfax Cluster:
# [https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening#Intel-Tutorial](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening#Intel-Tutorial)  
# 
# Reference to ssh connection:[https://access.colfaxresearch.com/?p=connect](https://access.colfaxresearch.com/?p=connect)  
# 
# Remember:
# 
#     chmod 600 ~/Downloads/colfax-access-key-****
# 
# 

# ## 1-2. Build enviroment after connect to Colfax Cluster by ssh colfax
# 
# Make own environment, install opencv, etc.
# 
#     ssh colfax
#     conda create --name test_env jupyter
#     source activate test_env
#     conda install numpy pandas opencv scikit-learn matplotlib tensorflow keras jupyter
# 
# 

# ## 1-3. Configure Jupyter Notebook, port and password (Thanks to everyone commented)
# 
# For avoid port collision and access by other user's access.
# 
# ### Select port (recommended)
# 
# Select port number, not likely to make collision. Default is 8888.  
# If port collides, another port will be used (like 8888 -> 8889).    
# It's a hassle, so use unique port.
# 
# If you are not familiar with network port configurations, I think ephemeral ports (49152 - 65535) are useful.  
# Here is the script to choose random ephemeral ports.
# Of course, you can choose your favorite number in 49152 - 65535.
# 
#     python -c "import random; ports = range(49152, 65535 + 1); random.shuffle(ports); print ports[0]"
# 
# ### Make Password hash (recommended)
# 
# Run the bellow command, input password twice, then you get hashed-password.
# 
#     python -c "from notebook.auth import passwd; print passwd()"
#     # e.g.) => sha1:237ca8abda58:9aef98cbcbae988caab4b9f86084ff22a1b2b373
# 
# ### Generate and edit config file
# 
# Generate config file (~/.jupyter/jupyter_notebook_config.py)
# 
#     jupyter notebook --generate-config
# 
# Edit via vi like this
# 
#     vi ~/.jupyter/jupyter_notebook_config.py
#     ....
#     # c.NotebookApp.password = u''
#     c.NotebookApp.password = u'sha1:237ca8abda58:9aef98cbcbae988caab4b9f86084ff22a1b2b373'
#     ....
#     # c.NotebookApp.port = 8888
#     c.NotebookApp.port = 1234
# 
# If you are not familiar with vi, use the bellow scripts (need to edit)
# 
#     echo "c.NotebookApp.password = u'sha1:237ca8abda58:9aef98cbcbae988caab4b9f86084ff22a1b2b373'\n" >> ~/.jupyter/jupyter_notebook_config.py
#     echo "c.NotebookApp.port = 1234\n" >> ~/.jupyter/jupyter_notebook_config.py
# 
# If you missed something, you can regenerate (over-write) config file.
# 
#     jupyter notebook --generate-config
# 
# Note: If your setting port makes port collision unfortunately, another port will be used.
# 
# 

# ## 1-4. Connect to Colfax Cluster by ssh tunneling, and run Jupyter Notebook
# 
# Before runnig Jupyter Notebook, once logout.
# 
#     logout
# 
# And, runnig Jupyter Notebook via ssh tunneling
# 
#     ssh -L 1234:localhost:1234 colfax -Y
#     source activate test_env
#     jupyter notebook --no-browser
# 
# **Note: The window (ran command) should be kept opened!**
# 
# Note: You can change port this step (Special thanks to Sriracha's comment)
# 
#     ssh -L 4321:localhost:4321 colfax -Y
#     source activate test_env
#     jupyter notebook --no-browser --port=4321
# 

# ## 1-5. Access to Jupyter Notebook on Colfax Cluste by your local machine's browser
# 
#     Access by your web browser (e.g. Google Chrome) on your local machine (e.g. Windows, Mac...)
# 
# [http://localhost:1234/](http://localhost:1234/)
#  or [http://127.0.0.1:1234/](http://127.0.0.1:1234/) (if you can't)

# # 2. Listing dataset image files

# ## 2-0. Setting of dataset's directories

# In[ ]:


import platform
import os

if 'c001' in platform.node(): 
    # platform.node() => 'c001' or like 'c001-n030' on Colfax
    abspath_dataset_dir_train_1 = '/data/kaggle/train/Type_1'
    abspath_dataset_dir_train_2 = '/data/kaggle/train/Type_2'
    abspath_dataset_dir_train_3 = '/data/kaggle/train/Type_3'
    abspath_dataset_dir_test    = '/data/kaggle/test/'
    abspath_dataset_dir_add_1   = '/data/kaggle/additional/Type_1'
    abspath_dataset_dir_add_2   = '/data/kaggle/additional/Type_2'
    abspath_dataset_dir_add_3   = '/data/kaggle/additional/Type_3'
elif '.local' in platform.node():
    # platform.node() => '*.local' on my local MacBook Air
    abspath_dataset_dir_train_1 = '/abspath/to/train/Type_1'
    abspath_dataset_dir_train_2 = '/abspath/to/train/Type_2'
    abspath_dataset_dir_train_3 = '/abspath/to/train/Type_3'
    abspath_dataset_dir_test    = '/abspath/to/test/'
    abspath_dataset_dir_add_1   = '/abspath/to/additional/Type_1'
    abspath_dataset_dir_add_2   = '/abspath/to/additional/Type_2'
    abspath_dataset_dir_add_3   = '/abspath/to/additional/Type_3'
else:
    # For kaggle's kernels environment (docker container?)
    abspath_dataset_dir_train_1 = '/kaggle/input/train/Type_1'
    abspath_dataset_dir_train_2 = '/kaggle/input/train/Type_2'
    abspath_dataset_dir_train_3 = '/kaggle/input/train/Type_3'
    abspath_dataset_dir_test    = '/kaggle/input/test/'
    abspath_dataset_dir_add_1   = '/kaggle/input/additional/Type_1'
    abspath_dataset_dir_add_2   = '/kaggle/input/additional/Type_2'
    abspath_dataset_dir_add_3   = '/kaggle/input/additional/Type_3'

    
def get_list_abspath_img(abspath_dataset_dir):
    list_abspath_img = []
    for str_name_file_or_dir in os.listdir(abspath_dataset_dir):
        if ('.jpg' in str_name_file_or_dir) == True:
            list_abspath_img.append(os.path.join(abspath_dataset_dir, str_name_file_or_dir))
    list_abspath_img.sort()
    return list_abspath_img


list_abspath_img_train_1 = get_list_abspath_img(abspath_dataset_dir_train_1)
list_abspath_img_train_2 = get_list_abspath_img(abspath_dataset_dir_train_2)
list_abspath_img_train_3 = get_list_abspath_img(abspath_dataset_dir_train_3)
list_abspath_img_train   = list_abspath_img_train_1 + list_abspath_img_train_2 + list_abspath_img_train_3

list_abspath_img_test    = get_list_abspath_img(abspath_dataset_dir_test)

list_abspath_img_add_1   = get_list_abspath_img(abspath_dataset_dir_add_1)
list_abspath_img_add_2   = get_list_abspath_img(abspath_dataset_dir_add_2)
list_abspath_img_add_3   = get_list_abspath_img(abspath_dataset_dir_add_3)
list_abspath_img_add     = list_abspath_img_add_1   + list_abspath_img_add_2   + list_abspath_img_add_3

# 0: Type_1, 1: Type_2, 2: Type_3
list_answer_train        = [0] * len(list_abspath_img_train_1) + [1] * len(list_abspath_img_train_2) + [2] * len(list_abspath_img_train_3)
list_answer_add          = [0] * len(list_abspath_img_add_1) + [1] * len(list_abspath_img_add_2) + [2] * len(list_abspath_img_add_3)


# ## 2-1. Check the (small part of) absolute paths

# In[ ]:


print(list_abspath_img_train_1[0:2])
print(list_abspath_img_train_2[0:2])
print(list_abspath_img_train_3[0:2])
print(list_abspath_img_train[0:4])
print(list_abspath_img_test[0:3])
print(list_abspath_img_add_1[0:2])
print(list_abspath_img_add_2[0:2])
print(list_abspath_img_add_3[0:2])
print(list_abspath_img_add[0:4])


# ## 2-2. Counting number of image files
# 
# Pandas is powerful data analysis toolkit. It is very useful to input, output and analyze csv files.
# 
# Check [10 Minutes to pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html) if you have time.

# In[ ]:


import pandas


pandas_columns = ['Number of image files']
pandas_index   = ['train_1', 'train_2', 'train_3', 'train', 'test', 'add_1', 'add_2', 'add_3', 'add', 'train + add', 'total']
pandas_data    = [len(list_abspath_img_train_1), len(list_abspath_img_train_2), len(list_abspath_img_train_3), len(list_abspath_img_train), len(list_abspath_img_test), len(list_abspath_img_add_1), len(list_abspath_img_add_2), len(list_abspath_img_add_3), len(list_abspath_img_add), len(list_abspath_img_train) + len(list_abspath_img_add), len(list_abspath_img_train) + len(list_abspath_img_test) + len(list_abspath_img_add)]

pandas.DataFrame(pandas_data, index = pandas_index, columns = pandas_columns)


# ## 2-3. Showing the ratio (Type 1, Type 2, Type 3)
# 
# It’s usually a good idea to check the deviation of dataset.  
# In my experience of another competition, my model's training accuracy was more than 80%, but all prediction were same  
# (my bullshit model was the master of selecting the majority 😭 ).

# In[ ]:


pandas_columns = ['Type_1', 'Type_2', 'Type_3']
pandas_index   = ['train', 'test', 'add']

ratio_train    = [x / len(list_abspath_img_train) for x in [len(list_abspath_img_train_1), len(list_abspath_img_train_2), len(list_abspath_img_train_3)]]
ratio_test     = ['?', '?', '?']
ratio_add      = [x / len(list_abspath_img_add) for x in [len(list_abspath_img_add_1), len(list_abspath_img_add_2), len(list_abspath_img_add_3)]]

pandas_data    = [ratio_train, ratio_test, ratio_add]

pandas.DataFrame(pandas_data, index = pandas_index, columns = pandas_columns)


# # Check dataset image pixel sizes

# In[ ]:


'''
import cv2


abspath_output_csv = './check_img_shape.csv'

file_output_csv = open(abspath_output_csv, 'w')
file_output_csv.write('abspath,shape_1,shape_2,shape_3\n')
file_output_csv.close()

for abspath_img in (list_abspath_img_train + list_abspath_img_test):
    str_shape = str(cv2.imread(abspath_img).shape)
    str_shape = str_shape.replace('(', '').replace(')', '').replace(' ', '')
    file_output_csv = open(abspath_output_csv, 'a')
    file_output_csv.write('%s,%s\n' % (abspath_img, str_shape))
    file_output_csv.close()
'''

'''
It will spend a lot of time to run. So I comment-out in the kernel notebook.
I uploaded './check_img_shape.csv' on Google Drive.
'''

"https://drive.google.com/open?id=0B2kJp7wSl9SIZTgtOWlTSmtDT2s"


# # 3. Show images by matplotlib

# In[ ]:


import cv2
import matplotlib.pyplot


def sub_func_load_img(abspath_img):
    img_rgb = cv2.cvtColor(cv2.imread(abspath_img), cv2.COLOR_BGR2RGB)
    return img_rgb

def show_img(abspath_img):
    matplotlib.pyplot.imshow(sub_func_load_img(abspath_img))
    matplotlib.pyplot.show()


# In[ ]:


# Show the first image

show_img(list_abspath_img_train[0])


# In[ ]:


# Another Usage, using string of image file's path

if 'c001' in platform.node():
    abspath_img = '/data/kaggle/test/81.jpg' # on Colfax Cluster
else:
    abspath_img = '../input/test/81.jpg' # on Kaggle's Kernel

show_img(abspath_img)


# ## Resampling images
# 
# For input CNN, unify all images into 640 * 480 RGB images
#  (fixed aspect-ratio and filled blank with black color).
# 
# This is just only one example out of many.
# 

# ## Step 0:  Unify sidelong images into vertically long images 
# 
# This step can be skipped.

# In[ ]:


import numpy


def sub_func_rotate_img_if_need(img_rgb):
    if img_rgb.shape[0] >= img_rgb.shape[1]:
        return img_rgb
    else:
        return numpy.rot90(img_rgb)



if 'c001' in platform.node():
    abspath_img = '/data/kaggle/test/81.jpg' # on Colfax Cluster
else:
    abspath_img = '../input/test/81.jpg' # on Kaggle Kernel

    
img_rgb = sub_func_load_img(abspath_img)

matplotlib.pyplot.imshow(img_rgb)
matplotlib.pyplot.show()

matplotlib.pyplot.imshow(sub_func_rotate_img_if_need(img_rgb))
matplotlib.pyplot.show()


# ## Step 1: Resize image with same aspect-ratio
# 
# sidelong images -> (640, *, 3)
# 
# vertically long images -> (*, 480, 3)

# In[ ]:


def sub_func_resize_img_same_ratio(img_rgb):
    if img_rgb.shape[0] / 640.0 >= img_rgb.shape[1] / 480.0:
        img_resized_rgb = cv2.resize(img_rgb, (int(640.0 * img_rgb.shape[1] / img_rgb.shape[0]), 640)) # (640, *, 3)
    else:
        img_resized_rgb = cv2.resize(img_rgb, (480, int(480.0 * img_rgb.shape[0] / img_rgb.shape[1]))) # (*, 480, 3)
    return img_resized_rgb


if 'c001' in platform.node():
    abspath_img = '/data/kaggle/test/81.jpg' # on Colfax Cluster
else:
    abspath_img = '../input/test/81.jpg' # on Kaggle Kernel

    
img_rgb = sub_func_load_img(abspath_img)

matplotlib.pyplot.imshow(img_rgb)
matplotlib.pyplot.show()
print(img_rgb.shape)

matplotlib.pyplot.imshow(sub_func_resize_img_same_ratio(img_rgb))
matplotlib.pyplot.show()
print(sub_func_resize_img_same_ratio(img_rgb).shape)

# Step 0 + Step 1 -> (*, 480, 3), Accidentally this example -> (640 ,480, 3)
matplotlib.pyplot.imshow(sub_func_resize_img_same_ratio(sub_func_rotate_img_if_need(img_rgb)))
matplotlib.pyplot.show()
print(sub_func_resize_img_same_ratio(sub_func_rotate_img_if_need(img_rgb)).shape)


# Step 2: Fill blank with black-color

# In[ ]:


def sub_func_fill_img(img_rgb):
    if img_rgb.shape[0] == 640:
        int_resize_1    = img_rgb.shape[1]
        int_fill_1      = (480 - int_resize_1 ) // 2
        int_fill_2      =  480 - int_resize_1 - int_fill_1
        numpy_fill_1    =  numpy.zeros((640, int_fill_1, 3), dtype=numpy.uint8)
        numpy_fill_2    =  numpy.zeros((640, int_fill_2, 3), dtype=numpy.uint8)
        img_filled_rgb = numpy.concatenate((numpy_fill_1, img_rgb, numpy_fill_1), axis=1)
    elif img_rgb.shape[1] == 480:
        int_resize_0    = img_rgb.shape[0]
        int_fill_1      = (640 - int_resize_0 ) // 2
        int_fill_2      =  640 - int_resize_0 - int_fill_1
        numpy_fill_1 =  numpy.zeros((int_fill_1, 480, 3), dtype=numpy.uint8)
        numpy_fill_2 =  numpy.zeros((int_fill_2, 480, 3), dtype=numpy.uint8)
        img_filled_rgb = numpy.concatenate((numpy_fill_1, img_rgb, numpy_fill_1), axis=0)
    else:
        raise ValueError
    return img_filled_rgb


matplotlib.pyplot.imshow(img_rgb)
matplotlib.pyplot.show()
print(img_rgb.shape)

# Step 1 + Step 2
matplotlib.pyplot.imshow(sub_func_fill_img(sub_func_resize_img_same_ratio(img_rgb)))
matplotlib.pyplot.show()
print(sub_func_fill_img(sub_func_resize_img_same_ratio(img_rgb)).shape)


# Step 0 + Step 1 + Step 2
matplotlib.pyplot.imshow(sub_func_fill_img(sub_func_resize_img_same_ratio(sub_func_rotate_img_if_need(img_rgb))))
matplotlib.pyplot.show()
print(sub_func_fill_img(sub_func_resize_img_same_ratio(sub_func_rotate_img_if_need(img_rgb))).shape)


# ## Finally: Step 0 + Step 1 + Step 2

# In[ ]:


def sub_func_resample_img(abspath_img):
    img = sub_func_load_img(abspath_img)
    img = sub_func_rotate_img_if_need(img)
    img = sub_func_resize_img_same_ratio(img)
    img = sub_func_fill_img(img)
    return img

def show_resample_img(abspath_img):
    matplotlib.pyplot.imshow(sub_func_resample_img(abspath_img))
    matplotlib.pyplot.show()


# In[ ]:


show_img(list_abspath_img_train[0])
print(sub_func_load_img(list_abspath_img_train[0]).shape)

show_resample_img(list_abspath_img_train[0])
print(sub_func_resample_img(list_abspath_img_train[0]).shape)

if 'c001' in platform.node():
    abspath_img = '/data/kaggle/test/81.jpg' # on Colfax Cluster
else:
    abspath_img = '../input/test/81.jpg' # on Kaggle Kernel

show_img(abspath_img)
print(sub_func_load_img(abspath_img).shape)

show_resample_img(abspath_img)
print(sub_func_resample_img(abspath_img).shape)


# In[ ]:


matplotlib.pyplot.imshow(cv2.resize(sub_func_resample_img(abspath_img), (224, 224)))
matplotlib.pyplot.show()


# ## Parallel computation
# 
#     multiprocessing.cpu_count() -> 8   # Jupyter Notebook on Colfax Cluster
#     multiprocessing.cpu_count() -> 256 # on Colfax Cluster, using `qsub` to run script

# In[ ]:


import multiprocessing


def multi_func_resample_img(list_abspath_img):
    multiprocessing_pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    return multiprocessing_pool.map(sub_func_resample_img, list_abspath_img)


list_img_train = multi_func_resample_img(list_abspath_img_train[0:4])

for resample_img in list_img_train:
    matplotlib.pyplot.imshow(resample_img)
    matplotlib.pyplot.show()


# ## To Be Continued...

# ## Run script on Colfax Cluster (MEMO, Re-write at training section)
# 
# We can run `python ./check_img_shape.py` on terminal directory.  
# But this method can use only 8 cpu-core, and the process will be terminated if spent long-time.  
# 
# If you run `qsub ./check_img_shape.sh`, you can use 256 cpu-core.  
# 
# ### Make check_img_shape.sh (via `vi`, `echo -e`, etc.).  
# 
# The contents of check_img_shape.sh is bellow.    
# u???? is your user name like u2000.    
# 
#     source activate test_env
#     python /home/u????/check_img_shape.py
# 
# ### Compute by qsub
# 
#     qsub ./check_img_shape.sh
# 
# ### Check runnning status:
# 
#     qstat
# 
# ### After runnning:
# 
#     STDOUT -> ./check_img_shape.sh.o0000
#     STDERR -> ./check_img_shape.sh.e0000
# 
# If you need, check it:
# 
#     cat ./check_img_shape.sh.o*
#     cat ./check_img_shape.sh.e*y 
