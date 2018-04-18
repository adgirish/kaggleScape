
# coding: utf-8

# # Detecting night photos
# 
# In the dataset, there are both day and night photos. Theoretically, it should be a good idea to train separate CNNs for them, as the color distribution is somewhat different, and, because of this, joining both types of photos in one dataset could be inefficient when it comes to training a CNN. The following notebook explores the idea of detecting day and night photos and separating them.

# In[ ]:


import numpy as np
import glob
import random
import matplotlib.pyplot as plt
import os
from sklearn import cluster
from sklearn import neighbors
from scipy.misc import imread, imsave


# It seems obvious that "night" photos are a little bit more greenish. Let's explore this a little more.
# 
# Let's plot some images alongside with mean of their components (R/G/B) below.

# In[ ]:


# load images
imgs_to_load = 20

preview_files = sorted(glob.glob('../input/train/*/*.jpg'), key=lambda x: random.random())[:imgs_to_load]
preview = np.array([imread(img) for img in preview_files])

def show_loaded_with_means(imgs):
    rows_total = int(len(preview) / 4)
    for i in range(rows_total):
        _, img_ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(8, 2))
        _, imgmean_ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(8, 2))
        for j in range(4):
            img = preview[i*4+j]
            img_mean = np.mean(img, axis=(0,1))
            # calculate squared means to amplify green dominance effect
            img_mean = np.power(img_mean, 2)
            
            # show plots
            img_ax[j].axis('off')
            img_ax[j].imshow(img)
            imgmean_ax[j].bar(range(3), img_mean, width=0.3, color='blue')
            imgmean_ax[j].set_xticks(np.arange(3) + 0.3 / 2)
            imgmean_ax[j].set_xticklabels(['R', 'G', 'B'])

show_loaded_with_means(preview)
plt.show()


# It can be seen that in night photos, green component shows clear dominance over R/B components, when it comes to its mean. This is something that should be easily picked up by k-means algorithm.

# Before applying it, let's first explore one more idea for representing each training image. If G component dominance is something that determines whether photo is day or night, we could, for mean of each component, store sum of differences between it and different component means. That way, dominance would mean higher number, and non-dominance would be punished.

# In[ ]:


imgs_to_load = 20

preview_files = sorted(glob.glob('../input/train/*/*.jpg'), key=lambda x: random.random())[:imgs_to_load]
preview = np.array([imread(img) for img in preview_files])

def show_loaded_with_mean_differences(imgs):
    rows_total = int(len(preview) / 4)
    for i in range(rows_total):
        _, img_ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(8, 2))
        _, imgmean_ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(8, 2))
        for j in range(4):
            # calculate features of an image
            img = preview[i*4+j]
            img_mean = np.mean(img, axis=(0,1))
            img_features = np.zeros(3)
            img_features[0] = (img_mean[0] - img_mean[1]) + (img_mean[0] - img_mean[2])
            img_features[1] = (img_mean[1] - img_mean[0]) + (img_mean[1] - img_mean[2])
            img_features[2] = (img_mean[2] - img_mean[0]) + (img_mean[2] - img_mean[1])
            
            # display plots
            img_ax[j].axis('off')
            img_ax[j].imshow(img)
            imgmean_ax[j].bar(range(3), img_features, width=0.3, color='blue')
            imgmean_ax[j].set_xticks(np.arange(3) + 0.3 / 2)
            imgmean_ax[j].set_xticklabels(['R', 'G', 'B'])

show_loaded_with_mean_differences(preview)
plt.show()


# Looks promising - it seems high G component values are only achieved by night photos! This is the approach we'll use with k-means.

# In[ ]:


# one cluster will be day photos, the other one night photos
knn_cls = 2
# increase this number while training locally for better results
training_imgs = 50

training_files = sorted(glob.glob('../input/train/*/*.jpg'), key=lambda x: random.random())[:training_imgs]
training = np.array([imread(img) for img in training_files])
training_means = np.array([np.mean(img, axis=(0, 1)) for img in training])
training_features = np.zeros((training_imgs, 3))
for i in range(training_imgs):
    training_features[i][0] = (training_means[i][0] - training_means[i][1])
    training_features[i][0] += (training_means[i][0] - training_means[i][2])
    training_features[i][1] = (training_means[i][1] - training_means[i][0])
    training_features[i][1] += (training_means[i][1] - training_means[i][2])
    training_features[i][2] = (training_means[i][2] - training_means[i][0])
    training_features[i][2] += (training_means[i][2] - training_means[i][1])

kmeans = cluster.KMeans(n_clusters=knn_cls).fit(training_features)
print(np.bincount(kmeans.labels_))


# In[ ]:


def show_four(imgs, title):
    _, ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(8, 2))
    plt.suptitle(title, size=8)
    for i, img in enumerate(imgs[:4]):
        ax[i].axis('off')
        ax[i].imshow(img)

for i in range(knn_cls):
    cluster_i = training[np.where(kmeans.labels_ == i)]
    show_four(cluster_i[:4], 'cluster' + str(i))


# It seems like the clustering was successful. When training locally on more images, entire training data splits successfully between night and day photos, with ~1-2 photos being misclassified, which seems like a good result, as entire training dataset consists of ~3777 images.
# 
# Running the script below should generate 'clustered' directory inside training folder, which should contain split data.

#     batch = 100
#     
#     # now load all training examples and cluster them
#     CLUSTER_FOLDER = os.path.abspath('./data/train/clustered')
#     training_filenames = sorted(glob.glob('./data/train/*/*.jpg'))
#     
#     # make directories if they doesn't exist
#     if not os.path.isdir(CLUSTER_FOLDER):
#         os.makedirs(CLUSTER_FOLDER)
#     
#     for cluster_num in xrange(knn_cls):
#         single_cluster_folder = os.path.join(CLUSTER_FOLDER, str(cluster_num))
#         if not os.path.isdir(single_cluster_folder):
#             os.mkdir(single_cluster_folder)
#     
#     saved_files = 0
#     while saved_files < len(training_filenames):
#         training_files = training_filenames[saved_files:saved_files+batch]
#         training = np.array([imread(img) for img in training_files])
#         training_means = np.array([np.mean(img, axis=(0, 1)) for img in training])
#         training_features = np.zeros((training_imgs, 3))
#         for i in xrange(len(training)):
#             training_features[i][0] = (training_means[i][0] - training_means[i][1])
#             training_features[i][0] += (training_means[i][0] - training_means[i][2])
#             training_features[i][1] = (training_means[i][1] - training_means[i][0])
#             training_features[i][1] += (training_means[i][1] - training_means[i][2])
#             training_features[i][2] = (training_means[i][2] - training_means[i][0])
#             training_features[i][2] += (training_means[i][2] - training_means[i][1])
#             
#         img_cls = kmeans.predict(training_features)
#         
#         for i, img in enumerate(training):
#             cluster = img_cls[i]
#             save_path = path.join(CLUSTER_FOLDER, str(cluster))
#             class_name = path.basename(path.dirname(training_files[i]))
#             save_path = path.join(save_path, class_name)
#             if not path.isdir(save_path):
#                 os.makedirs(save_path)
#             save_path = path.join(save_path, path.basename(training_files[i]))
#             print save_path
#             imsave(save_path, img)
#             saved_files += 1
#         
#         print str(saved_files) + "/" + str(len(training_filenames))

# One idea that makes sense, would be to develop and apply some kind of "night" filter to make all images look like they come from same distribution (which should be easier than a "day" filter, as intuitively night photos contain less color information than day photos, and it is probably easier to "lose" some information than to "gain" it), but I have no experience in image processing and apart from simple heuristics, I can't come up with approach that would look natural.
# 
# This is my first take on a serious kaggle competition, and first kernel ever, so any feedback is welcome ;)
# (I suspect some of my heuristics are a little more complicated than they should be,  and maybe some of them don't even make much sense...)
