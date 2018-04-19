
# coding: utf-8

# # Crop, Save and View Nodules in 3D
# 
# Final results:
# 
# ![enter image description here][1]
# 
# This kernel shows you how to crop the nodule **with given coordinates** in patient's CT scan, how to save it in .npy and .mhd/.raw file, and how to view it in **3D**. About half of the code is learned from other kernels. (Tutorial: U-Net Segmentation Approach to Cancer Diagnosis by [Jonathan Mulholland and Aaron Sander, Booz Allen Hamilton][2], Full Preprocessing Tutorial by [Guido Zuidhof][3], Candidate Generation and LUNA16 preprocessing by [ArnavJain][4]) The most fun part of this code is also learned from the Internet. I wanted a better way to visualize the region of interest my code generated. Then I searched online and found a method to save .mhd/.raw file posted by [Price Jackson][5], with [source code][6]. The part I found useful was built on the MIT licensed work by [Bing Jian and Baba C. Vemuri][7]. I happened to know [Fiji][8] was a good tool to view volume stacks in 3D, with original intensity values. After integrating them together, I find visualize and check the results is not that much of pain. Actually, it comes with some fun and may give you some insights on the journey. This incites me to open a kernel here, though most of the code are from others.
# 
# Running the code below, with given CT scan and given coordinates, will give you a cropped nodule in [19, 19, 19] dimensional numpy array, with spacing [1, 1, 1]mm. I will use LUNA16 as input data in the code. Because it comes with some annotated nodules. So I cannot run it here but I put the psedo-output in Markdown cells. The code has been tested in Python 3.5.
# 
# 
#   [1]: https://i.gyazo.com/24c585876be7e54a2fc20a40fbf3b2e9.gif
#   [2]: https://www.kaggle.com/c/data-science-bowl-2017#tutorial
#   [3]: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
#   [4]: https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
#   [5]: https://sites.google.com/site/pjmedphys/tutorials/medical-images-in-python
#   [6]: https://sites.google.com/site/pjmedphys/scripts
#   [7]: https://code.google.com/archive/p/diffusion-mri/
#   [8]: https://imagej.net/Fiji/Downloads

# ### Read annotation data and define preprocessing method
# 
# Please change the input path to fit your environment

# In[ ]:


import SimpleITK as sitk
import numpy as np

from glob import glob
import pandas as pd
import scipy.ndimage

import mhd_utils_3d


## Read annotation data and filter those without images
# Learned from Jonathan Mulholland and Aaron Sander, Booz Allen Hamilton
# https://www.kaggle.com/c/data-science-bowl-2017#tutorial

# Set input path
# Change to fit your environment
luna_path = './LUNA16/'
luna_subset_path = luna_path + 'subset0_samples/'
file_list = glob(luna_subset_path + "*.mhd")

df_node = pd.read_csv(luna_path+'annotations.csv')

def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)

# map file full path to each record 
df_node['file'] = df_node['seriesuid'].map(lambda file_name: get_filename(file_list, file_name))
df_node = df_node.dropna()

## Define resample method to make images isomorphic, default spacing is [1, 1, 1]mm
# Learned from Guido Zuidhof
# https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode = 'nearest')
    
    return image, new_spacing


# ### Define methods to save data
# 
# I defined a method called "save_nodule" in this part. And it is called by:
# 
#     save_nodule(nodule_crop, name_index)
#     # nodule_crop is a 3 dimensional numpy array, name_index is the name of the file. I use the index in annotation.csv.
# 
# If your time is limited, just try to call the method with your data. If it works, then you can focus on building the model. It will save three files: **.npy**, **.mhd**, **.raw**.

# In[ ]:


#!/usr/bin/env python
#coding=utf-8

#======================================================================
#Program:   Diffusion Weighted MRI Reconstruction
#Link:      https://code.google.com/archive/p/diffusion-mri
#Module:    $RCSfile: mhd_utils.py,v $
#Language:  Python
#Author:    $Author: bjian $
#Date:      $Date: 2008/10/27 05:55:55 $
#Version:   
#           $Revision: 1.1 by PJackson 2013/06/06 $
#               Modification: Adapted to 3D
#               Link: https://sites.google.com/site/pjmedphys/tutorials/medical-images-in-python
# 
#           $Revision: 2   by RodenLuo 2017/03/12 $
#               Modication: Adapted to LUNA2016 data set for DSB2017
#               Link: 
#======================================================================

import os
import numpy
import array

def write_meta_header(filename, meta_dict):
    header = ''
    # do not use tags = meta_dict.keys() because the order of tags matters
    tags = ['ObjectType','NDims','BinaryData',
       'BinaryDataByteOrderMSB','CompressedData','CompressedDataSize',
       'TransformMatrix','Offset','CenterOfRotation',
       'AnatomicalOrientation',
       'ElementSpacing',
       'DimSize',
       'ElementType',
       'ElementDataFile',
       'Comment','SeriesDescription','AcquisitionDate','AcquisitionTime','StudyDate','StudyTime']
    for tag in tags:
        if tag in meta_dict.keys():
            header += '%s = %s\n'%(tag,meta_dict[tag])
    f = open(filename,'w')
    f.write(header)
    f.close()
    
def dump_raw_data(filename, data):
    """ Write the data into a raw format file. Big endian is always used. """
    #Begin 3D fix
    data=data.reshape([data.shape[0],data.shape[1]*data.shape[2]])
    #End 3D fix
    rawfile = open(filename,'wb')
    a = array.array('f')
    for o in data:
        a.fromlist(list(o))
    #if is_little_endian():
    #    a.byteswap()
    a.tofile(rawfile)
    rawfile.close()
    
def write_mhd_file(mhdfile, data, dsize):
    assert(mhdfile[-4:]=='.mhd')
    meta_dict = {}
    meta_dict['ObjectType'] = 'Image'
    meta_dict['BinaryData'] = 'True'
    meta_dict['BinaryDataByteOrderMSB'] = 'False'
    meta_dict['ElementType'] = 'MET_FLOAT'
    meta_dict['NDims'] = str(len(dsize))
    meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])
    meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd','.raw')
    write_meta_header(mhdfile, meta_dict)

    pwd = os.path.split(mhdfile)[0]
    if pwd:
        data_file = pwd +'/' + meta_dict['ElementDataFile']
    else:
        data_file = meta_dict['ElementDataFile']

    dump_raw_data(data_file, data)
    
def save_nodule(nodule_crop, name_index):
    np.save(str(name_index) + '.npy', nodule_crop)
    write_mhd_file(str(name_index) + '.mhd', nodule_crop, nodule_crop.shape[::-1])


# ### Read CT scan data, process and save 

# In[ ]:


## Collect patients with nodule and crop the nodule
# In this code snippet, the cropped nodule is a [19, 19, 19] volume with [1, 1, 1]mm spacing.
# Learned from Jonathan Mulholland and Aaron Sander, Booz Allen Hamilton
# https://www.kaggle.com/c/data-science-bowl-2017#tutorial

# Change the number in the next line to process more
for patient in file_list[:1]:
    print(patient)
    
    # Check whether this patient has nodule or not
    if patient not in df_node.file.values:
        print('Patient ' + patient + 'Not exist!')
        continue
    patient_nodules = df_node[df_node.file == patient]
    
    full_image_info = sitk.ReadImage(patient)
    full_scan = sitk.GetArrayFromImage(full_image_info)
    
    origin = np.array(full_image_info.GetOrigin())[::-1] # get [z, y, x] origin
    old_spacing = np.array(full_image_info.GetSpacing())[::-1] # get [z, y, x] spacing
    
    image, new_spacing = resample(full_scan, old_spacing)
    
    print('Resample Done')
    

    for index, nodule in patient_nodules.iterrows():
        nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX]) 
        # Attention: Z, Y, X

        v_center = np.rint( (nodule_center - origin) / new_spacing )
        v_center = np.array(v_center, dtype=int)

#         print(v_center)
        window_size = 9 # This will give you the volume length = 9 + 1 + 9 = 19
        # Why the magic number 19, I found that in "LUNA16/annotations.csv", 
        # the 95th percentile of the nodules' diameter is about 19.
        # This is kind of a hyperparameter, will affect your final score.
        # Change it if you want.
        zyx_1 = v_center - window_size # Attention: Z, Y, X
        zyx_2 = v_center + window_size + 1

#         print('Crop range: ')
#         print(zyx_1)
#         print(zyx_2)

        # This will give you a [19, 19, 19] volume
        img_crop = image[ zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2] ]
        
        # save the nodule 
        save_nodule(img_crop, index)
    
    print('Done for this patient!\n\n')
print('Done for all!')


# Sample output:
# 
#     ./LUNA16/subset0_samples/1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd
#     Resample Done
#     Done for this patient!
# 
# 
#     Done for all!

# ### Plot in 2D

# In[ ]:


## Plot volume in 2D

import numpy as np
from matplotlib import pyplot as plt

def plot_nodule(nodule_crop):
    
    # Learned from ArnavJain
    # https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    f, plots = plt.subplots(int(nodule_crop.shape[0]/4)+1, 4, figsize=(10, 10))
    
    for z_ in range(nodule_crop.shape[0]): 
        plots[int(z_/4), z_ % 4].imshow(nodule_crop[z_,:,:])
    
    # The last subplot has no image because there are only 19 images.
    plt.show()
    
# Plot one example
img_crop = np.load('25.npy')
plot_nodule(img_crop)


# Sample output:
# 
# [![https://gyazo.com/31240b93448349629652fe56cfd3f48f](https://i.gyazo.com/31240b93448349629652fe56cfd3f48f.png)](https://gyazo.com/31240b93448349629652fe56cfd3f48f)

# ### Plot in 3D by Fiji

# In your output path (default is the same as your working directory), there should be some .mhd/.raw files. Follow the steps below to open it in 3D viewer. (I also made a video (18min) and posted in [this discussion thread][1]. You may want to watch it if you like video tutorials. Excuse me that I'm not fluent in English. So the video is recorded kind of slow, you may want to watch it with 1.5 or 2 times speed.)
# 
# ### Steps to open nodules in 3D Viewer 
# 1. Download [Fiji][2]
# 2. Drag one .mhd file to Fiji **status bar**
# 
#     ![enter image description here][3]
# 
# 3. Click on the new image window. Press "control +" to zoom in, "control -" to zoom out.
# 4. In the **Menubar**, Click "Image > Stacks > Orthogonal Views" to see it. Scroll to go through all slices. **Notice that when your cursor moves around in the image window, the Fiji Status Bar shows you the XY coordinates and the value, this value is the same as that in Python numpy array**
# 5. In the **Menubar**, Click "Plugins > 3D viewer", in the "Add ..." window, change "Resampling factor" to 1, click "OK", click "OK" to convert to 8-bit.
# 6. Click on the "ImageJ 3D Viewer", In the **Menubar**, click "Edit > Adjust threshold". Drag threshold bar to around 150.
# 7.  **Drag** the object to rotate it. Hold "shift" and **drag** to move it. Scroll to zoom. 
# 8. In the **Menubar**, click "View > Start animation" to activate it.
# 9. Click on the image stack, in the **Menubar**, click "Image > Adjust > threshold", check "Dark background", move the first threshold bar to around -400, click "Apply > OK > Yes". 
# 10. Either use step 5 to open another 3D viewer, or click on the current "ImageJ 3D Viewer", in the **Menubar** click "add > from image" to add another object. Click on the object and "shift" drag to move it. Thresholding on the raw image then creating 3D object gives you different rendering. 
# 
# **Sample visualizations:**
# 
# 25 in LUNA16/annotation.csv
# 
# ![enter image description here][4]
# 
# 26 in LUNA16/annotation.csv
# 
# ![enter image description here][5]
# 
# 
# 
# 
# **Please feel free to comment if you have questions or suggestions. Please upvote the above mentioned kernels if you find they are helpful. Please upvote this kernel if it helps you on the journey.**
# 
# 
# 
# ## Updates:
# 
# Changed "nodule_crop.shape" to "nodule_crop.shape[::-1]" in method "save_nodule(nodule_crop, name_index)"
# 
#   [1]: https://www.kaggle.com/c/data-science-bowl-2017/discussion/28502
#   [2]: https://imagej.net/Fiji/Downloads
#   [3]: https://imagej.net/_images/6/67/Fiji-main-window.jpg
#   [4]: https://i.gyazo.com/d9eea0182a10af8b8af8f6cf88c3a0e7.gif
#   [5]: https://i.gyazo.com/3c69952f966704b4055e25bc07495748.gif
