
# coding: utf-8

# # Introduction
# This example shows that how you could use OpenCV to detect and count white areas in an image. 

# # 1. Pick a random image

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')

ID = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
FILE = "../input/stage1_train/{}/images/{}.png".format(ID,ID)


# # 2. Read the image and find appropriate filter settings

# In[ ]:


img = cv2.imread(FILE,0)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# # 3. Plot the filtering process

# In[ ]:


# Plot Here
plt.figure(figsize=(15,5))
images = [blur, 0, th3]
titles = ['Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
plt.subplot(1,3,1),plt.imshow(images[0],'gray')
plt.title(titles[0]), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.hist(images[0].ravel(),256)
plt.title(titles[1]), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(images[2],'gray')
plt.title(titles[2]), plt.xticks([]), plt.yticks([])


# # 4. Count the number of contours in the filtered image

# In[ ]:


im = cv2.imread(FILE)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(img, contours, -1, (255,255,255), 3)
plt.figure(figsize=(10,10))
plt.subplot(1,2,1),plt.title('Original Image'),plt.imshow(im)#,'red')
plt.subplot(1,2,2),plt.title('OpenCV.findContours'),plt.imshow(img,'gray')#,'red')

print('number of detected contours: ',len(contours))


# *Interested in OpenCV? Checkout [OpenCV-Python-Toturials](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)*
