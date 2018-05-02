
# coding: utf-8

# # Data exploration
# 
# - Visualization of all training data, all testing data
# - Visualize some additional training data
# - Clustering of training and test data
# - Basic skin detection
# - Some stats using jpg exif

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
from glob import glob
TRAIN_DATA = "../input/train"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files])

print(len(type_1_files), len(type_2_files), len(type_3_files))
print("Type 1", type_1_ids[:10])
print("Type 2", type_2_ids[:10])
print("Type 3", type_3_ids[:10])


# In[ ]:


TEST_DATA = "../input/test"
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = np.array([s[len(TEST_DATA)+1:-4] for s in test_files])
print(len(test_ids))
print(test_ids[:10])


# In[ ]:


ADDITIONAL_DATA = "../input/additional"
additional_type_1_files = glob(os.path.join(ADDITIONAL_DATA, "Type_1", "*.jpg"))
additional_type_1_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_1"))+1:-4] for s in additional_type_1_files])
additional_type_2_files = glob(os.path.join(ADDITIONAL_DATA, "Type_2", "*.jpg"))
additional_type_2_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_2"))+1:-4] for s in additional_type_2_files])
additional_type_3_files = glob(os.path.join(ADDITIONAL_DATA, "Type_3", "*.jpg"))
additional_type_3_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_3"))+1:-4] for s in additional_type_3_files])

print(len(additional_type_1_files), len(additional_type_2_files), len(additional_type_2_files))
print("Type 1", additional_type_1_ids[:10])
print("Type 2", additional_type_2_ids[:10])
print("Type 3", additional_type_3_ids[:10])


# In[ ]:


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


# In[ ]:


import matplotlib.pylab as plt

def plt_st(l1,l2):
    plt.figure(figsize=(l1,l2))


# ## Display all train images of Type_1, Type_2, Type_3

# In[ ]:


tile_size = (256, 256)
n = 15

complete_images = []
for k, type_ids in enumerate([type_1_ids, type_2_ids, type_3_ids]):
    m = int(np.ceil(len(type_ids) * 1.0 / n))
    complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
    train_ids = sorted(type_ids)
    counter = 0
    for i in range(m):
        ys = i*(tile_size[1] + 2)
        ye = ys + tile_size[1]
        for j in range(n):
            xs = j*(tile_size[0] + 2)
            xe = xs + tile_size[0]
            if counter == len(train_ids):
                break
            image_id = train_ids[counter]; counter+=1
            img = get_image_data(image_id, 'Type_%i' % (k+1))
            img = cv2.resize(img, dsize=tile_size)
            img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)
            complete_image[ys:ye, xs:xe, :] = img[:,:,:]
        if counter == len(train_ids):
            break
    complete_images.append(complete_image)           


# In[ ]:


plt_st(20, 20)
plt.imshow(complete_images[0])
plt.title("Training dataset of type %i" % (1))


# In[ ]:


index = 1
m = complete_images[index].shape[0] / (tile_size[0] + 2)
n = int(np.ceil(m / 20.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(tile_size[0] + 2)*20
    ye = min((i+1)*(tile_size[0] + 2)*20, complete_images[index].shape[0])
    plt.imshow(complete_images[index][ys:ye,:,:])
    plt.title("Training dataset of type %i, part %i" % (index + 1, i))


# In[ ]:


index = 2
m = complete_images[index].shape[0] / (tile_size[0] + 2)
n = int(np.ceil(m / 20.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(tile_size[0] + 2)*20
    ye = min((i+1)*(tile_size[0] + 2)*20, complete_images[index].shape[0])
    plt.imshow(complete_images[index][ys:ye,:,:])
    plt.title("Training dataset of type %i, part %i" % (index + 1, i))


# ### Display all test images

# In[ ]:


tile_size = (256, 256)
n = 15
m = int(np.ceil(len(test_ids) * 1.0 / n))
complete_test_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
counter = 0
for i in range(m):
    ys = i*(tile_size[1] + 2)
    ye = ys + tile_size[1]
    for j in range(n):
        xs = j*(tile_size[0] + 2)
        xe = xs + tile_size[0]
        if counter == len(test_ids):
            break
        image_id = test_ids[counter]; counter+=1
        img = get_image_data(image_id, 'Test')
        img = cv2.resize(img, dsize=tile_size)
        img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)
        complete_test_image[ys:ye, xs:xe, :] = img[:,:,:]
    if counter == len(test_ids):
        break


# In[ ]:


m = complete_test_image.shape[0] / (tile_size[0] + 2)
n = int(np.ceil(m / 20.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(tile_size[0] + 2)*20
    ye = min((i+1)*(tile_size[0] + 2)*20, complete_test_image.shape[0])
    plt.imshow(complete_test_image[ys:ye,:,:])
    plt.title("Test dataset, part %i" % (i))


# ## Display 500 addtional train images of Type_1, Type_2, Type_3

# In[ ]:


tile_size = (256, 256)
n = 15
ll = 500
complete_images = []
for k, type_ids in enumerate([additional_type_1_ids[:ll], additional_type_2_ids[:ll], additional_type_3_ids[:ll]]):
    m = int(np.ceil(len(type_ids) * 1.0 / n))
    complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
    train_ids = sorted(type_ids)
    counter = 0
    for i in range(m):
        ys = i*(tile_size[1] + 2)
        ye = ys + tile_size[1]
        for j in range(n):
            xs = j*(tile_size[0] + 2)
            xe = xs + tile_size[0]
            if counter == len(train_ids):
                break
            image_id = train_ids[counter]; counter+=1
            img = get_image_data(image_id, 'AType_%i' % (k+1))
            img = cv2.resize(img, dsize=tile_size)
            img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)
            complete_image[ys:ye, xs:xe, :] = img[:,:,:]
        if counter == len(train_ids):
            break
    complete_images.append(complete_image)       


# In[ ]:


index = 0
m = complete_images[index].shape[0] / (tile_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(tile_size[0] + 2)*15
    ye = min((i+1)*(tile_size[0] + 2)*15, complete_images[index].shape[0])
    plt.imshow(complete_images[index][ys:ye,:,:])
    plt.title("Additional Training dataset (500 images) of type %i, part %i" % (index + 1, i))


# In[ ]:


index = 1
m = complete_images[index].shape[0] / (tile_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(tile_size[0] + 2)*15
    ye = min((i+1)*(tile_size[0] + 2)*15, complete_images[index].shape[0])
    plt.imshow(complete_images[index][ys:ye,:,:])
    plt.title("Additional Training dataset (500 images) of type %i, part %i" % (index + 1, i))


# In[ ]:


index = 2
m = complete_images[index].shape[0] / (tile_size[0] + 2)
n = int(np.ceil(m / 15.0))
for i in range(n):
    plt_st(20, 20)
    ys = i*(tile_size[0] + 2)*15
    ye = min((i+1)*(tile_size[0] + 2)*15, complete_images[index].shape[0])
    plt.imshow(complete_images[index][ys:ye,:,:])
    plt.title("Additional Training dataset (500 images) of type %i, part %i" % (index + 1, i))


# ## Basic skin detection

# In[ ]:


img_1 = get_image_data('1023', 'Type_1')
img_2 = get_image_data('531', 'Type_1')
img_3 = get_image_data('596', 'Type_1')
img_4 = get_image_data('1061', 'Type_1')
img_5 = get_image_data('1365', 'Type_2')


# In[ ]:


def sieve(image, size):
    """
    Filter removes small objects of 'size' from binary image
    Input image should be a single band image of type np.uint8
    Idea : use Opencv findContours
    """
    sqLimit = size**2
    linLimit = size*4
    outImage = image.copy()
    image, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(hierarchy) > 0:
        hierarchy = hierarchy[0]
        index = 0
        while index >= 0:
            contour = contours[index]
            p = cv2.arcLength(contour, True)
            s = cv2.contourArea(contour)
            r = cv2.boundingRect(contour)
            if s <= sqLimit and p <= linLimit:
                outImage[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = 0
            index = hierarchy[index][0]
    else:
        pass
        # print("No contours found")
    return outImage


# in HSV :
skin_range_1_min = np.array([120, 0, 0], dtype=np.uint8)
skin_range_1_max = np.array([255, 255, 255], dtype=np.uint8)

skin_range_2_min = np.array([0, 0, 0], dtype=np.uint8)
skin_range_2_max = np.array([45, 255, 255], dtype=np.uint8)

skin_kernel_size = 7
skin_sieve_min_size = 5

def detect_skin(image):
    proc = cv2.medianBlur(image, 7)
    ### Detect skin
    image_hsv = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)
    skin_like_mask = cv2.inRange(image_hsv, skin_range_1_min, skin_range_1_max)
    skin_like_mask_2 = cv2.inRange(image_hsv, skin_range_2_min, skin_range_2_max)
    skin_like_mask = cv2.bitwise_or(skin_like_mask, skin_like_mask_2)    
    # Filter the skin mask :
    skin_mask = sieve(skin_like_mask, skin_sieve_min_size)
    kernel = np.ones((skin_kernel_size, skin_kernel_size), dtype=np.int8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)    
    # Apply skin mask
    skin_segm_rgb = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin_segm_rgb

for image in [img_1, img_2, img_3, img_4, img_5]:       
    image = cv2.resize(image, dsize=(512, 512))
    skin_segm_rgb = detect_skin(image)
    plt_st(12, 4)
    plt.subplot(121)
    plt.title("Original image")    
    plt.imshow(image)
    plt.subplot(122)
    plt.title("Skin segmentation")
    plt.imshow(skin_segm_rgb)


# In[ ]:


tile_size = (256, 256)
n = 15

complete_images = []
for k, type_ids in enumerate([type_1_ids, ]):
    m = int(np.floor(len(type_ids) / n))
    complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 3), dtype=np.uint8)
    train_ids = sorted(type_ids)
    counter = 0
    for i in range(m):
        ys = i*(tile_size[1] + 2)
        ye = ys + tile_size[1]
        for j in range(n):
            xs = j*(tile_size[0] + 2)
            xe = xs + tile_size[0]
            image_id = train_ids[counter]; counter+=1
            img = get_image_data(image_id, 'Type_%i' % (k+1))
            img = cv2.resize(img, dsize=tile_size)
            img = detect_skin(img)
            img = cv2.putText(img, image_id, (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=3)
            complete_image[ys:ye, xs:xe, :] = img[:,:,:]
    complete_images.append(complete_image)    


# ### Apply skin segmentation on all training data and visualize the result

# In[ ]:


plt_st(20, 20)
plt.imshow(complete_images[0])
plt.title("Training dataset of type %i" % (0))


# ## Clustering
# 
# - Take a number of images from all classified images and test images
# - Compute histogram on hue channel
# - Perform 5 classes clustering on the data 

# In[ ]:


def compute_histogram(img, hist_size=100):
    hist = cv2.calcHist([img], [0], mask=None, histSize=[hist_size], ranges=(0, 255))
    hist = cv2.normalize(hist, dst=hist)
    return hist

#for image in [img_1, img_2, img_3, img_4, img_5]:       
#    image = cv2.resize(image, dsize=(512, 512))    
#    hue = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:,:,0]
#    hist = compute_histogram(hue)
#    plt_st(12, 4)
#    plt.subplot(131)
#    plt.title("Original image")    
#    plt.imshow(image)
#    plt.subplot(132)
#    plt.title("Hue")    
#    plt.imshow(hue, cmap='gray')
#    plt.subplot(133)
#    plt.title("Histogram")
#    plt.plot(hist)


# In[ ]:


train_nb_samples = 100
type_ids=(type_1_ids, type_2_ids, type_3_ids, test_ids)
image_types = ["Type_1", "Type_2", "Type_3", "Test"]
ll = [int(len(ids)) for ids in type_ids]

count = 0
train_id_type_list = []
while count < train_nb_samples:
    for l, ids, image_type in zip(ll, type_ids, image_types):
        image_id = ids[count % l]
        train_id_type_list.append((image_id, image_type))
    count += 1


# In[ ]:


image_size = (256, 256)
hist_size = 100
X = np.zeros((len(train_id_type_list), hist_size), dtype=np.float32)
Y = np.zeros((len(train_id_type_list), 2), dtype=np.float32)
for i, (image_id, image_type) in enumerate(train_id_type_list):
    img = get_image_data(image_id, image_type)
    img = cv2.resize(img, dsize=image_size[::-1])
    hue = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,0]
    hist = compute_histogram(hue, hist_size)    
    X[i, :] = hist[:, 0]
    Y[i, :] = (hist.mean(), hist.std())


# In[ ]:


plt.title("Image Hue Histogram std vs mean")
plt.scatter(Y[:, 0], Y[:, 1], s=50, cmap='viridis');


# In[ ]:


np_classes = 5


# In[ ]:


#from sklearn.cluster import KMeans
#kmeans = KMeans(n_clusters=np_classes)
#kmeans.fit(X)
#y_kmeans = kmeans.predict(X)
#_ = plt.hist(y_kmeans)


# In[ ]:


from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=np_classes, affinity='nearest_neighbors', assign_labels='kmeans')
y_spectral = model.fit_predict(X)
_ = plt.hist(y_spectral)


# In[ ]:


image_size = (256, 256)
all_classes_images = []
for class_index in range(np_classes):
    
    class_indices = np.where(y_spectral == class_index)[0]
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
            image_id, image_type = train_id_type_list[class_indices[counter]]; counter+=1
            img = get_image_data(image_id, image_type)
            img = cv2.resize(img, dsize=image_size)
            img = cv2.putText(img, image_id + ' | ' + str(image_type) + ' | ' + str(class_index), (5,img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            one_class_image[ys:ye, xs:xe, :] = img[:,:,:]

        if counter == len(class_indices):
            break
    
    all_classes_images.append(one_class_image)


# In[ ]:


for class_index in range(np_classes):
    plt_st(20, 20)
    plt.imshow(all_classes_images[class_index])
    plt.title("Class %i" % (class_index)) 


# ## Some stats using jpg exif
# 
# We can explore metadata of all these images. If exif metadata is present in images, we can found out camera name, camera type, acquisition date and time etc.

# In[ ]:


from PIL import Image
import seaborn as sns

def _get_image_data_pil(image_id, image_type, return_exif_md=False):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    try:
        img_pil = Image.open(fname)
    except Exception as e:
        assert False, "Failed to read image : %s, %s. Error message: %s" % (image_id, image_type, e)

    img = np.asarray(img_pil)
    assert isinstance(img, np.ndarray), "Open image is not an ndarray. Image id/type : %s, %s" % (image_id, image_type)
    if not return_exif_md:
        return img
    else:
        return img, img_pil._getexif()


# In[ ]:


exif_stats = pd.DataFrame(columns=['Image_id', 'Image_type', 'Camera_name', 'Camera_type', 'Datetime', 'ISO'])

type_ids=(type_1_ids, type_2_ids, type_3_ids, test_ids)
image_types = ["Type_1", "Type_2", "Type_3", "Test"]

counter = 0
for ids, image_type in zip(type_ids, image_types):
    print('--', image_type)
    for image_id in ids:
        img, exif_data = _get_image_data_pil(image_id, image_type, return_exif_md=True)
        if isinstance(exif_data, dict):
            exif_stats.loc[counter, :] = [image_id, image_type, 
                                          exif_data[271], 
                                          exif_data[272], 
                                          exif_data[306],
                                          exif_data[34855]] 
        else:            
            exif_stats.loc[counter, :] = [image_id, image_type, 
                                          'NA', 
                                          'NA', 
                                          'NA', 
                                          'NA'] 
        counter+=1


# In[ ]:


exif_stats['Camera_name'] = exif_stats['Camera_name'].str.lower()
exif_stats['YMD'] = exif_stats['Datetime'].apply(lambda x: x[:10])
data_mask = exif_stats['Camera_name'] != 'na'


# In[ ]:


exif_stats.head(10)


# In[ ]:


sns.countplot(data=exif_stats, x='Camera_name', hue='Image_type')


# In[ ]:


plt_st(12, 12)
sns.countplot(data=exif_stats[data_mask].sort_values(['YMD']), y='YMD')

