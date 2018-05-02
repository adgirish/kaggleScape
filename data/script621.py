
# coding: utf-8

# **Overview**
# 
# This Kernel uses only the test data. However, some parameters have been tuned on the training data.
# It reads the images, convert them into binary after some preprocessing steps, Then, it simply labels connected components and considers each obtained component as a nuclei.

# **Reading data**

# In[ ]:


import os
import cv2
import numpy as np

test_dirs = os.listdir("../input/stage1_test")
test_filenames=["../input/stage1_test/"+file_id+"/images/"+file_id+".png" for file_id in test_dirs]
test_images=[cv2.imread(imagefile) for imagefile in test_filenames]

# Any results you write to the current directory are saved as output.


# **List of operations to be performed on each image**

# In[ ]:


def process(img_rgb):
    #green channel happends to produce slightly better results
    #than the grayscale image and other channels
    img_gray=img_rgb[:,:,1]#cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #morphological opening (size tuned on training data)
    circle7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    img_open=cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, circle7)
    #Otsu thresholding
    img_th=cv2.threshold(img_open,0,255,cv2.THRESH_OTSU)[1]
    #Invert the image in case the objects of interest are in the dark side
    if(np.sum(img_th==255)>np.sum(img_th==0)):
        img_th=cv2.bitwise_not(img_th)
    #second morphological opening (on binary image this time)
    bin_open=cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7) 
    #connected components
    cc=cv2.connectedComponents(bin_open)[1]
    #cc=segment_on_dt(bin_open,20)
    return cc


# **Computing output for each image**

# In[ ]:


test_connected_components=[process(img)  for img in test_images]


# **RLE encoding**
# 
# Taken from this [kernel](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277) but slghtly adapted to our case.

# In[ ]:


def rle_encoding(cc):
    values=list(np.unique(cc))
    values.remove(0)
    RLEs=[]
    for v in values:
        dots = np.where(cc.T.flatten() == v)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1):
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        RLEs.append(run_lengths)
    return RLEs

test_RLEs=[rle_encoding(cc) for cc in test_connected_components]


# **Writing submission file**

# In[ ]:


with open("submission_image_processing.csv", "a") as myfile:
    myfile.write("ImageId,EncodedPixels\n")
    for i,RLEs in enumerate(test_RLEs):
        for RLE in RLEs:
            myfile.write(test_dirs[i]+","+" ".join([str(i) for i in RLE])+"\n")

