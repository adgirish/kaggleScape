
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import random

# Definitions
IMAGES_TO_SHOW = 20

# Plot image
def plot_image(img, title=None):
    plt.figure(figsize=(15,20))
    plt.title(title)
    plt.imshow(img)
    plt.show()
    
# Draw elipsis on image
def draw_ellipse(mask):
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    im, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    has_ellipse = len(contours) > 0
    if has_ellipse:
        cnt = contours[0]
        ellipse = cv2.fitEllipse(cnt)
        cx, cy = np.array(ellipse[0], dtype=np.int)
        m3[cy-2:cy+2,cx-2:cx+2] = (255, 0, 0)
        cv2.ellipse(m3, ellipse, (0, 255, 0), 1)
        
    return has_ellipse, m3


# In[ ]:


# Read some files
mfiles = glob.glob("../input/train/*_mask.tif")
random.shuffle(mfiles) # Shuffle for random results

files_with_ellipse = 0
for mfile in mfiles:
    mask = cv2.imread(mfile, -1)  # imread(..., -1) returns grayscale images
    has_ellipse, mask_with_ellipse = draw_ellipse(mask)
    if has_ellipse:
        files_with_ellipse = files_with_ellipse+1
        plot_image(mask_with_ellipse, mfile)
        if files_with_ellipse > IMAGES_TO_SHOW:
            break

