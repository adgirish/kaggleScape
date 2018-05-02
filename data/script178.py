
# coding: utf-8

# # Visual Data Analysis

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For exaample, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# How many images we have in train and test datasets

# In[4]:


get_ipython().system('ls ../input/train/ | wc -l')
get_ipython().system('ls ../input/train_masks/ | wc -l')
get_ipython().system('ls ../input/test/ | wc -l')


# For example, train filenames looks like

# In[5]:


get_ipython().system('ls ../input/train/ | grep c_01.jpg')


# In[7]:


import os 
from glob import glob

INPUT_PATH = '../input'
DATA_PATH = INPUT_PATH
TRAIN_DATA = os.path.join(DATA_PATH, "train")
TRAIN_MASKS_DATA = os.path.join(DATA_PATH, "train_masks")
TEST_DATA = os.path.join(DATA_PATH, "test")
TRAIN_MASKS_CSV_FILEPATH = os.path.join(DATA_PATH, "train_masks.csv")
METADATA_CSV_FILEPATH = os.path.join(DATA_PATH, "metadata.csv")

TRAIN_MASKS_CSV = pd.read_csv(TRAIN_MASKS_CSV_FILEPATH)
METADATA_CSV = pd.read_csv(METADATA_CSV_FILEPATH)


# In[8]:


train_files = glob(os.path.join(TRAIN_DATA, "*.jpg"))
train_ids = [s[len(TRAIN_DATA)+1:-4] for s in train_files]

test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = [s[len(TEST_DATA)+1:-4] for s in test_files]


# In[12]:


def get_filename(image_id, image_type):
    check_dir = False
    if "Train" == image_type:
        ext = 'jpg'
        data_path = TRAIN_DATA
        suffix = ''
    elif "Train_mask" in image_type:
        ext = 'gif'
        data_path = TRAIN_MASKS_DATA
        suffix = '_mask'
    elif "Test" in image_type:
        ext = 'jpg'
        data_path = TEST_DATA
        suffix = ''
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}{}.{}".format(image_id, suffix, ext))


# In[17]:


import cv2
from PIL import Image


def get_image_data(image_id, image_type, **kwargs):
    if 'mask' in image_type:
        img = _get_image_data_pil(image_id, image_type, **kwargs)
    else:
        img = _get_image_data_opencv(image_id, image_type, **kwargs)
    return img

def _get_image_data_opencv(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _get_image_data_pil(image_id, image_type, return_exif_md=False, return_shape_only=False):
    fname = get_filename(image_id, image_type)
    try:
        img_pil = Image.open(fname)
    except Exception as e:
        assert False, "Failed to read image : %s, %s. Error message: %s" % (image_id, image_type, e)

    if return_shape_only:
        return img_pil.size[::-1] + (len(img_pil.getbands()),)

    img = np.asarray(img_pil)
    assert isinstance(img, np.ndarray), "Open image is not an ndarray. Image id/type : %s, %s" % (image_id, image_type)
    if not return_exif_md:
        return img
    else:
        return img, img_pil._getexif()


# ## Display a single car with its mask

# In[19]:


image_id = train_ids[0]

plt.figure(figsize=(20, 20))
img = get_image_data(image_id, "Train")
mask = get_image_data(image_id, "Train_mask")
img_masked = cv2.bitwise_and(img, img, mask=mask)

print("Image shape: {} | image type: {} | mask shape: {} | mask type: {}".format(img.shape, img.dtype, mask.shape, mask.dtype) )

plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(mask)
plt.subplot(133)
plt.imshow(img_masked)


# ## Display 500 random cars from train dataset

# In[ ]:


_train_ids = list(train_ids)
np.random.shuffle(_train_ids)
_train_ids = _train_ids[:500]
tile_size = (256, 256)
n = 8

m = int(np.ceil(len(_train_ids) * 1.0 / n))
complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)

counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    for j in range(n):
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        if counter == len(_train_ids):
            break
        image_id = _train_ids[counter]; counter+=1
        img = get_image_data(image_id, 'Train')
        img = cv2.resize(img, dsize=tile_size)
        img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
        complete_image[ys:ye, xs:xe, :] = img[:,:,:]
    if counter == len(_train_ids):
        break    


# In[ ]:


m = complete_image.shape[0] / (tile_size[0] + 2)
k = 8
n = int(np.ceil(m / k))
for i in range(n):
    plt.figure(figsize=(20, 20))
    ys = i*(tile_size[0] + 2)*k
    ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
    plt.imshow(complete_image[ys:ye,:,:])
    plt.title("Training dataset, part %i" % i)


# ## How many different car in all datasets:

# In[9]:


len(METADATA_CSV['id'].unique()), len(METADATA_CSV['id'])


# ## How many different cars in train dataset:

# In[11]:


TRAIN_MASKS_CSV['id'] = TRAIN_MASKS_CSV['img'].apply(lambda x: x[:-7])
len(TRAIN_MASKS_CSV['id'].unique()), len(TRAIN_MASKS_CSV['id'].unique()) * 16


# In[12]:


all_318_car_ids = TRAIN_MASKS_CSV['id'].unique()


# ## Display all 318 cars at '03' angle from train dataset

# In[13]:


all_318_cars_image_ids = [_id + '_03' for _id in all_318_car_ids]


# In[ ]:


_train_ids = list(all_318_cars_image_ids)
tile_size = (256, 256)
n = 8

m = int(np.ceil(len(_train_ids) * 1.0 / n))
complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)

counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    for j in range(n):
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        if counter == len(_train_ids):
            break
        image_id = _train_ids[counter]; counter+=1
        img = get_image_data(image_id, 'Train')
        img = cv2.resize(img, dsize=tile_size)
        img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)
        complete_image[ys:ye, xs:xe, :] = img[:,:,:]
    if counter == len(_train_ids):
        break   


# In[ ]:


m = complete_image.shape[0] / (tile_size[0] + 2)
k = 8
n = int(np.ceil(m / k))
for i in range(n):
    plt.figure(figsize=(20, 20))
    ys = i*(tile_size[0] + 2)*k
    ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])
    plt.imshow(complete_image[ys:ye,:,:])
    plt.title("All 318 cars from train dataset, part %i" % i)


# ## Which cars are present in the train dataset:

# In[ ]:


METADATA_CSV.index = METADATA_CSV['id']
train_metadata_csv = METADATA_CSV.loc[TRAIN_MASKS_CSV['id'].unique(),:]


# In[ ]:


import seaborn as sns
sns.countplot(y="make", data=train_metadata_csv, palette="Greens_d")


# ### Search for similar cars that have same year, make, model and trim1 

# In[ ]:


train_gb_year_make_model_trim1 = train_metadata_csv.groupby(['year', 'make', 'model', 'trim1'])
len(train_gb_year_make_model_trim1.groups)


# In[ ]:


similar_cars = [k for k in train_gb_year_make_model_trim1.groups if len(train_gb_year_make_model_trim1.groups[k]) > 1]


# #### Display similar cars

# In[ ]:


for gname in similar_cars:
    _ids = train_gb_year_make_model_trim1.get_group(gname)['id']
    _trim2 = train_gb_year_make_model_trim1.get_group(gname)['trim2']
    plt.figure(figsize=(14, 6))
    plt.suptitle("{}".format(gname))    
    n = len(_ids)
    for i, _id in enumerate(_ids):
        plt.subplot(1, n, i + 1)
        plt.title('{}'.format(_trim2[i]))
        img = get_image_data(_id + '_03', 'Train')
        plt.imshow(img)            


# ## Which cars are present in the test dataset:

# In[ ]:


test_dataset_ids = list(set(METADATA_CSV['id']) - set(TRAIN_MASKS_CSV['id']))
len(test_dataset_ids), len(METADATA_CSV['id'])


# In[ ]:


test_metadata_csv = METADATA_CSV.loc[test_dataset_ids,:]
sns.countplot(y="make", data=test_metadata_csv, palette="Greens_d")


# ### Search for similar cars that have same year, make, model and trim1 

# In[ ]:


test_metadata_csv.loc[test_metadata_csv['trim1'].isnull(), 'trim1'] = '-'
test_gb_year_make_model_trim1 = test_metadata_csv.groupby(['year', 'make', 'model', 'trim1'])
len(test_gb_year_make_model_trim1.groups)


# In[ ]:


similar_cars = [k for k in test_gb_year_make_model_trim1.groups if len(test_gb_year_make_model_trim1.groups[k]) > 1]
len(similar_cars)


# #### Display some of similar cars

# In[ ]:


k = 5 
for gname in similar_cars[:20]:
    _ids = test_gb_year_make_model_trim1.get_group(gname)['id']      
    _trim2 = test_gb_year_make_model_trim1.get_group(gname)['trim2']    
    plt.figure(figsize=(14, 6))
    plt.suptitle("{}".format(gname))    
    n = min(len(_ids), k)
    m = int(np.ceil(len(_ids) * 1.0 / k))
    for i, _id in enumerate(_ids):
        plt.subplot(m, n, i + 1)    
        plt.title("{}".format(_trim2[i]))
        img = get_image_data(_id + '_03', 'Test')
        plt.imshow(img)        
    


# ## Are there 'same' cars in train and test ?

# In[ ]:


METADATA_CSV['in_train'] = False
METADATA_CSV['in_test'] = False

METADATA_CSV.loc[test_dataset_ids, 'in_test'] = True
METADATA_CSV.loc[TRAIN_MASKS_CSV['id'].unique(), 'in_train'] = True


# No cars with the same ids

# In[ ]:


METADATA_CSV[METADATA_CSV['in_train'] & METADATA_CSV['in_test']]


# In[ ]:


METADATA_CSV.loc[METADATA_CSV['trim1'].isnull(), 'trim1'] = '-'
gb_year_make_model_trim1 = METADATA_CSV.groupby(['year', 'make', 'model', 'trim1'])
len(gb_year_make_model_trim1.groups)


# In[ ]:


similar_cars = [k for k in gb_year_make_model_trim1.groups if len(gb_year_make_model_trim1.groups[k]) > 1]
len(similar_cars)


# In[ ]:


gb_year_make_model_trim1.get_group(similar_cars[0])


# We see that model BMW Z4 Z4 sDrive35i 2014 is in test and train dataset

# ### Display some of similar cars

# In[ ]:


k = 5 
for gname in similar_cars[:10]:
    _ids = gb_year_make_model_trim1.get_group(gname)['id']      
    _trim2 = gb_year_make_model_trim1.get_group(gname)['trim2']
    _in_train = gb_year_make_model_trim1.get_group(gname)['in_train']
    _in_test = gb_year_make_model_trim1.get_group(gname)['in_test']    
    
    plt.figure(figsize=(14, 6))
    plt.suptitle("{}".format(gname))    
    n = min(len(_ids), k)
    m = int(np.ceil(len(_ids) * 1.0 / k))
    for i, _id in enumerate(_ids):
        plt.subplot(m, n, i + 1)    
        plt.title("{}\ntrain={}, test={}\n{}".format(_trim2[i], _in_train[i], _in_test[i], _id))
        image_type = "Train" if  _in_train[i] else "Test"
        img = get_image_data(_id + '_03', image_type)
        plt.imshow(img)        
    


# In[ ]:


cond = lambda k: (len(gb_year_make_model_trim1.groups[k]) > 1) and gb_year_make_model_trim1.get_group(k)[['in_train', 'in_test']].any().all()
models_in_train_and_test = [k for k in gb_year_make_model_trim1.groups if cond(k)]
len(models_in_train_and_test)


# ### Display only models that present in train and test
# Train image is display with its mask and test images are blended with the train mask

# In[ ]:


sns.set_style("whitegrid", {'axes.grid' : False})


# In[ ]:


k = 5 
for gname in models_in_train_and_test[:10]:
    _ids = gb_year_make_model_trim1.get_group(gname)['id']      
    _trim2 = gb_year_make_model_trim1.get_group(gname)['trim2']
    _in_train = gb_year_make_model_trim1.get_group(gname)['in_train']
    _in_test = gb_year_make_model_trim1.get_group(gname)['in_test']    
    
    train_index = np.where(_in_train == True)[0][0]    
    first_train_mask = get_image_data(_ids[train_index] + '_03', "Train_mask")    
    
    plt.figure(figsize=(14, 6))
    plt.suptitle("{}".format(gname))    
    n = min(len(_ids), k)
    m = int(np.ceil(len(_ids) * 1.0 / k))
    for i, _id in enumerate(_ids):
        plt.subplot(m, n, i + 1)    
        plt.title("{}\ntrain={}, test={}\n{}".format(_trim2[i], _in_train[i], _in_test[i], _id))
        image_type = "Train" if  _in_train[i] else "Test"
        img = get_image_data(_id + '_03', image_type)
        if _in_train[i]:
            img = cv2.bitwise_and(img, img, mask=first_train_mask)
            plt.imshow(img)
        else:
            plt.imshow(img)
            plt.imshow(first_train_mask, alpha=0.50)

    

