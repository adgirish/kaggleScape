
# coding: utf-8

# ### Get dot coordinates using blob_log from skimage library

# In[ ]:


import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import skimage.feature
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


classes = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups", "error"]

file_names = os.listdir("../input/Train/")
file_names = sorted(file_names, key=lambda 
                    item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item)) 

# select a subset of files to run on
file_names = file_names[0:2]

# dataframe to store results in
count_df = pd.DataFrame(index=file_names, columns=classes).fillna(0)


# In[ ]:


for filename in file_names:
    
    # read the Train and Train Dotted images
    image_1 = cv2.imread("../input/TrainDotted/" + filename)
    image_2 = cv2.imread("../input/Train/" + filename)
    
    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1,image_2)
    
    # mask out blackened regions from Train Dotted
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 20] = 0
    mask_1[mask_1 > 0] = 255
    
    mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    mask_2[mask_2 < 20] = 0
    mask_2[mask_2 > 0] = 255
    
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    image_5 = cv2.bitwise_or(image_4, image_4, mask=mask_2) 
    
    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = cv2.cvtColor(image_5, cv2.COLOR_BGR2GRAY)
    
    # detect blobs
    blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02)
    
    # prepare the image to plot the results on
    image_7 = cv2.cvtColor(image_6, cv2.COLOR_GRAY2BGR)
    
    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b,g,r = image_1[int(y)][int(x)][:]
        
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if r > 200 and b < 50 and g < 50: # RED
            count_df["adult_males"][filename] += 1
            cv2.circle(image_7, (int(x),int(y)), 8, (0,0,255), 2)            
        elif r > 200 and b > 200 and g < 50: # MAGENTA
            count_df["subadult_males"][filename] += 1
            cv2.circle(image_7, (int(x),int(y)), 8, (250,10,250), 2)            
        elif r < 100 and b < 100 and 150 < g < 200: # GREEN
            count_df["pups"][filename] += 1
            cv2.circle(image_7, (int(x),int(y)), 8, (20,180,35), 2) 
        elif r < 100 and  100 < b and g < 100: # BLUE
            count_df["juveniles"][filename] += 1 
            cv2.circle(image_7, (int(x),int(y)), 8, (180,60,30), 2)
        elif r < 150 and b < 50 and g < 100:  # BROWN
            count_df["adult_females"][filename] += 1
            cv2.circle(image_7, (int(x),int(y)), 8, (0,42,84), 2)            
        else:
            count_df["error"][filename] += 1
            cv2.circle(image_7, (int(x),int(y)), 8, (255,255,155), 2)
    
    # output the results
          
    f, ax = plt.subplots(3,2,figsize=(10,16))
    (ax1, ax2, ax3, ax4, ax5, ax6) = ax.flatten()
    plt.title('%s'%filename)
    
    ax1.imshow(cv2.cvtColor(image_2[700:1200,2130:2639,:], cv2.COLOR_BGR2RGB))
    ax1.set_title('Train')
    ax2.imshow(cv2.cvtColor(image_1[700:1200,2130:2639,:], cv2.COLOR_BGR2RGB))
    ax2.set_title('Train Dotted')
    ax3.imshow(cv2.cvtColor(image_3[700:1200,2130:2639,:], cv2.COLOR_BGR2RGB))
    ax3.set_title('Train Dotted - Train')
    ax4.imshow(cv2.cvtColor(image_5[700:1200,2130:2639,:], cv2.COLOR_BGR2RGB))
    ax4.set_title('Mask blackened areas of Train Dotted')
    ax5.imshow(image_6[700:1200,2130:2639], cmap='gray')
    ax5.set_title('Grayscale for input to blob_log')
    ax6.imshow(cv2.cvtColor(image_7[700:1200,2130:2639,:], cv2.COLOR_BGR2RGB))
    ax6.set_title('Result')

    plt.show()


# ### Check count results

# In[ ]:


count_df


# ### Reference counts

# In[ ]:


reference = pd.read_csv('../input/Train/train.csv')
reference.ix[0:1]

