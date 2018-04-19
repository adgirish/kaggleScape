
# coding: utf-8

# # Type 1 clustering
# 
# I want to understand what kind of images we have acording to the standard procedure, for example described [here](http://www.gfmer.ch/ccdc/pdf/module5.pdf). Namely,
# do we have images :
# 
# - native cervix
# - acetic acid
# - lugol iodine
# 
# of the same patient ?
#  
# **Edit**: Clustering method updated
# 

# In[ ]:


import os
from glob import glob

import numpy as np

TRAIN_DATA = "../input/train"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files])

print("Train data")
print(len(type_1_files), len(type_2_files), len(type_3_files))
print("Type 1", type_1_ids[:10])
print("Type 2", type_2_ids[:10])
print("Type 3", type_3_ids[:10])

ADDITIONAL_DATA = "../input/additional"
additional_type_1_files = glob(os.path.join(ADDITIONAL_DATA, "Type_1", "*.jpg"))
additional_type_1_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_1"))+1:-4] for s in additional_type_1_files])
additional_type_2_files = glob(os.path.join(ADDITIONAL_DATA, "Type_2", "*.jpg"))
additional_type_2_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_2"))+1:-4] for s in additional_type_2_files])
additional_type_3_files = glob(os.path.join(ADDITIONAL_DATA, "Type_3", "*.jpg"))
additional_type_3_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_3"))+1:-4] for s in additional_type_3_files])

print("Additional data")
print(len(additional_type_1_files), len(additional_type_2_files), len(additional_type_2_files))
print("Type 1", additional_type_1_ids[:10])
print("Type 2", additional_type_2_ids[:10])
print("Type 3", additional_type_3_ids[:10])



def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or         image_type == "Type_2" or         image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or           image_type == "AType_2" or           image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type[1:])
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))


def get_image_data(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


import matplotlib.pylab as plt

def plt_st(l1,l2):
    plt.figure(figsize=(l1,l2))


# Idea is to use clustering on images of one type to group data

# In[ ]:


def compute_histogram(img, hist_size=100):
    hist = cv2.calcHist([img], [0], mask=None, histSize=[hist_size], ranges=(0, 255))
    hist = cv2.normalize(hist, dst=hist)
    return hist


# In[ ]:


type_ids=(type_1_ids, additional_type_1_ids)
image_types = ["Type_1", "AType_1"]
ll = [int(len(ids)) for ids in type_ids]

id_type_list = []
for ids, image_type in zip(type_ids, image_types):
    for image_id in ids:
        id_type_list.append((image_id, image_type))


# In[ ]:


print("Total number of images: ", len(id_type_list))
# Find empty images:
empty_images = []
for image_id, image_type in id_type_list:
    size = os.path.getsize(get_filename(image_id, image_type))
    if size == 0:
        empty_images.append((image_id, image_type))
print("Number of empty images: ", len(empty_images))


# In[ ]:


# Remove empty images from id_type_list
for image_id, image_type in empty_images:
    id_type_list.remove((image_id, image_type))


# In[ ]:


import cv2


# In[ ]:


RESIZED_IMAGES = {}


# In[ ]:


image_size = (256, 256)
center = (image_size[0]//2, image_size[1]//2)
hist_size = 30

crop_size = 30

n_features = 3 * hist_size
X = np.zeros((len(id_type_list), n_features), dtype=np.float32)
for i, (image_id, image_type) in enumerate(id_type_list):
    
    key = (image_id, image_type)
    if key in RESIZED_IMAGES:
        img = RESIZED_IMAGES[key]
    else:
        img = get_image_data(image_id, image_type)
        img = cv2.resize(img, dsize=image_size[::-1])    
        RESIZED_IMAGES[key] = img
    
    # crop 
    proc = img[center[1]-crop_size:center[1]+crop_size,center[0]-crop_size:center[0]+crop_size,:]
    # Blur 
    proc = cv2.GaussianBlur(proc, (7, 7), 0)
    hsv = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)
    hue = hsv[:,:,0]
    sat = hsv[:,:,1]
    val = hsv[:,:,2]
    hist_hue = compute_histogram(hue, hist_size)
    hist_sat = compute_histogram(sat, hist_size)    
    hist_val = compute_histogram(val, hist_size)    
    X[i, 0:hist_size] = hist_hue[:,0]
    X[i, hist_size:2*hist_size] = hist_sat[:,0]
    X[i, 2*hist_size:] = hist_val[:,0]


# In[ ]:


n_classes = 10

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n_classes)
kmeans.fit(X)
y_classes = kmeans.predict(X)
_ = plt.hist(y_classes)


# In[ ]:


all_classes_images = []

for class_index in range(n_classes):    
    class_indices = np.where(y_classes == class_index)[0]
    n = 10    
    m = int(np.ceil(len(class_indices) / n)) 
    one_class_image = np.zeros((m*(image_size[0]+2), n*(image_size[1]+2), 3), dtype=np.uint8)    
    
    counter = 0
    for i in range(m):
        ys = i*(image_size[1] + 2)
        ye = ys + image_size[1]
        for j in range(n):
            xs = j*(image_size[0] + 2)
            xe = xs + image_size[0]
            if counter == len(class_indices):
                break
            image_id, image_type = id_type_list[class_indices[counter]]; counter+=1
            key = (image_id, image_type)
            assert key in RESIZED_IMAGES, "WTF"
            img = RESIZED_IMAGES[key]                
            img = cv2.putText(img, image_id + ' | ' + str(image_type) + ' | ' + str(class_index), (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            one_class_image[ys:ye, xs:xe, :] = img[:,:,:]

        if counter == len(class_indices):
            break
    
    all_classes_images.append(one_class_image)


# In[ ]:


class_index = 0

m = all_classes_images[class_index].shape[0] / (image_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(image_size[0] + 2)*15
    ye = min((i+1)*(image_size[0] + 2)*15, all_classes_images[class_index].shape[0])
    plt.imshow(all_classes_images[class_index][ys:ye,:,:])
    plt.title("Class %i, part %i" % (class_index, i))


# In[ ]:


class_index = 1

m = all_classes_images[class_index].shape[0] / (image_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(image_size[0] + 2)*15
    ye = min((i+1)*(image_size[0] + 2)*15, all_classes_images[class_index].shape[0])
    plt.imshow(all_classes_images[class_index][ys:ye,:,:])
    plt.title("Class %i, part %i" % (class_index, i))


# In[ ]:


class_index = 2

m = all_classes_images[class_index].shape[0] / (image_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(image_size[0] + 2)*15
    ye = min((i+1)*(image_size[0] + 2)*15, all_classes_images[class_index].shape[0])
    plt.imshow(all_classes_images[class_index][ys:ye,:,:])
    plt.title("Class %i, part %i" % (class_index, i))


# In[ ]:


class_index = 3

m = all_classes_images[class_index].shape[0] / (image_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(image_size[0] + 2)*15
    ye = min((i+1)*(image_size[0] + 2)*15, all_classes_images[class_index].shape[0])
    plt.imshow(all_classes_images[class_index][ys:ye,:,:])
    plt.title("Class %i, part %i" % (class_index, i))


# In[ ]:


class_index = 4

m = all_classes_images[class_index].shape[0] / (image_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(image_size[0] + 2)*15
    ye = min((i+1)*(image_size[0] + 2)*15, all_classes_images[class_index].shape[0])
    plt.imshow(all_classes_images[class_index][ys:ye,:,:])
    plt.title("Class %i, part %i" % (class_index, i))


# In[ ]:


class_index = 5

m = all_classes_images[class_index].shape[0] / (image_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(image_size[0] + 2)*15
    ye = min((i+1)*(image_size[0] + 2)*15, all_classes_images[class_index].shape[0])
    plt.imshow(all_classes_images[class_index][ys:ye,:,:])
    plt.title("Class %i, part %i" % (class_index, i))


# In[ ]:


class_index = 6

m = all_classes_images[class_index].shape[0] / (image_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(image_size[0] + 2)*15
    ye = min((i+1)*(image_size[0] + 2)*15, all_classes_images[class_index].shape[0])
    plt.imshow(all_classes_images[class_index][ys:ye,:,:])
    plt.title("Class %i, part %i" % (class_index, i))


# In[ ]:


class_index = 7

m = all_classes_images[class_index].shape[0] / (image_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(image_size[0] + 2)*15
    ye = min((i+1)*(image_size[0] + 2)*15, all_classes_images[class_index].shape[0])
    plt.imshow(all_classes_images[class_index][ys:ye,:,:])
    plt.title("Class %i, part %i" % (class_index, i))


# In[ ]:


class_index = 8

m = all_classes_images[class_index].shape[0] / (image_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(image_size[0] + 2)*15
    ye = min((i+1)*(image_size[0] + 2)*15, all_classes_images[class_index].shape[0])
    plt.imshow(all_classes_images[class_index][ys:ye,:,:])
    plt.title("Class %i, part %i" % (class_index, i))


# In[ ]:


class_index = 9

m = all_classes_images[class_index].shape[0] / (image_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(image_size[0] + 2)*15
    ye = min((i+1)*(image_size[0] + 2)*15, all_classes_images[class_index].shape[0])
    plt.imshow(all_classes_images[class_index][ys:ye,:,:])
    plt.title("Class %i, part %i" % (class_index, i))

