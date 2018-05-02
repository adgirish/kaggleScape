
# coding: utf-8

# This kernels aims at segmenting the cervix using the technique presented in this paper: https://www.researchgate.net/publication/24041301_Automatic_Detection_of_Anatomical_Landmarks_in_Uterine_Cervix_Images

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

TRAIN_DATA = "../input/train"

types = ['Type_1']#,'Type_2','Type_3']
type_ids = []

for type in enumerate(types):
    type_i_files = glob(os.path.join(TRAIN_DATA, type[1], "*.jpg"))
    type_i_ids = np.array([s[len(TRAIN_DATA)+8:-4] for s in type_i_files])
    type_ids.append(type_i_ids[:5])

def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or         image_type == "Type_2" or         image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or           image_type == "AType_2" or           image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type)
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


# First, we crop the image in order to remove the circular frames that might be present. This is done by finding the largest inscribed rectangle to the thresholded image. The image is then cropped to this rectangle. (see these videos for an explanation of the algorithm: https://www.youtube.com/watch?v=g8bSdXCG-lA, https://www.youtube.com/watch?v=VNbkzsnllsU)

# In[ ]:


def maxHist(hist):
    maxArea = (0, 0, 0)
    height = []
    position = []
    for i in range(len(hist)):
        if (len(height) == 0):
            if (hist[i] > 0):
                height.append(hist[i])
                position.append(i)
        else: 
            if (hist[i] > height[-1]):
                height.append(hist[i])
                position.append(i)
            elif (hist[i] < height[-1]):
                while (height[-1] > hist[i]):
                    maxHeight = height.pop()
                    area = maxHeight * (i-position[-1])
                    if (area > maxArea[0]):
                        maxArea = (area, position[-1], i)
                    last_position = position.pop()
                    if (len(height) == 0):
                        break
                position.append(last_position)
                if (len(height) == 0):
                    height.append(hist[i])
                elif(height[-1] < hist[i]):
                    height.append(hist[i])
                else:
                    position.pop()    
    while (len(height) > 0):
        maxHeight = height.pop()
        last_position = position.pop()
        area =  maxHeight * (len(hist) - last_position)
        if (area > maxArea[0]):
            maxArea = (area, len(hist), last_position)
    return maxArea
            

def maxRect(img):
    maxArea = (0, 0, 0)
    addMat = np.zeros(img.shape)
    for r in range(img.shape[0]):
        if r == 0:
            addMat[r] = img[r]
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
        else:
            addMat[r] = img[r] + addMat[r-1]
            addMat[r][img[r] == 0] *= 0
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
    return (int(maxArea[3]+1-maxArea[0]/abs(maxArea[1]-maxArea[2])), maxArea[2], maxArea[3], maxArea[1], maxArea[0])

def cropCircle(img):
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    img = cv2.resize(img, dsize=tile_size)
            
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
            
    ff = np.zeros((gray.shape[0],gray.shape[1]), 'uint8') 
    cv2.drawContours(ff, main_contour, -1, 1, 15)
    ff_mask = np.zeros((gray.shape[0]+2,gray.shape[1]+2), 'uint8')
    cv2.floodFill(ff, ff_mask, (int(gray.shape[1]/2), int(gray.shape[0]/2)), 1)
    
    rect = maxRect(ff)
    rectangle = [min(rect[0],rect[2]), max(rect[0],rect[2]), min(rect[1],rect[3]), max(rect[1],rect[3])]
    img_crop = img[rectangle[0]:rectangle[1], rectangle[2]:rectangle[3]]
    cv2.rectangle(ff,(min(rect[1],rect[3]),min(rect[0],rect[2])),(max(rect[1],rect[3]),max(rect[0],rect[2])),3,2)
    
    return [img_crop, rectangle, tile_size]


# “For an initial delineation of the cervix, we use two features: 
# 
#  - the *a* color channel of the source image in Lab color space (the higher the value of *a* , the “redder” the pixel color)
#  - *R*, the distance of a pixel from the image center. The *R* feature provides spatial information and supports the extraction of continuous regions within the image plane."

# In[ ]:


def Ra_space(img, Ra_ratio, a_threshold):
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w*h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w/2-i)*(w/2-i) + (h/2-j)*(h/2-j))
            Ra[i*h+j, 0] = R
            Ra[i*h+j, 1] = min(imgLab[i][j][1], a_threshold)
            
    Ra[:,0] /= max(Ra[:,0])
    Ra[:,0] *= Ra_ratio
    Ra[:,1] /= max(Ra[:,1])

    return Ra


# "The image is separated next into two clusters in the 2-D (*a-R*) feature space; we use Gaussian mixture modeling, initialized by a K-means procedure."

# In[ ]:


def get_and_crop_image(image_id, image_type):
    img = get_image_data(image_id, image_type)
    initial_shape = img.shape
    [img, rectangle_cropCircle, tile_size] = cropCircle(img)
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = Ra_space(imgLab, 1.0, 150)
    a_channel = np.reshape(Ra[:,1], (w,h))
    
    g = mixture.GaussianMixture(n_components = 2, covariance_type = 'diag', random_state = 0, init_params = 'kmeans')
    image_array_sample = shuffle(Ra, random_state=0)[:1000]
    g.fit(image_array_sample)
    labels = g.predict(Ra)
    labels += 1 # Add 1 to avoid labeling as 0 since regionprops ignores the 0-label.
    
    # The cluster that has the highest a-mean is selected.
    labels_2D = np.reshape(labels, (w,h))
    gg_labels_regions = measure.regionprops(labels_2D, intensity_image = a_channel)
    gg_intensity = [prop.mean_intensity for prop in gg_labels_regions]
    cervix_cluster = gg_intensity.index(max(gg_intensity)) + 1

    mask = np.zeros((w * h,1),'uint8')
    mask[labels==cervix_cluster] = 255
    mask_2D = np.reshape(mask, (w,h))

    cc_labels = measure.label(mask_2D, background=0)
    regions = measure.regionprops(cc_labels)
    areas = [prop.area for prop in regions]

    regions_label = [prop.label for prop in regions]
    largestCC_label = regions_label[areas.index(max(areas))]
    mask_largestCC = np.zeros((w,h),'uint8')
    mask_largestCC[cc_labels==largestCC_label] = 255

    img_masked = img.copy()
    img_masked[mask_largestCC==0] = (0,0,0)
    img_masked_gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY);
            
    _,thresh_mask = cv2.threshold(img_masked_gray,0,255,0)
            
    kernel = np.ones((11,11), np.uint8)
    thresh_mask = cv2.dilate(thresh_mask, kernel, iterations = 1)
    thresh_mask = cv2.erode(thresh_mask, kernel, iterations = 2)
    _, contours_mask, _ = cv2.findContours(thresh_mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours_mask, key = cv2.contourArea, reverse = True)[0]
    cv2.drawContours(img, main_contour, -1, 255, 3)
    
    x,y,w,h = cv2.boundingRect(main_contour)
    
    rectangle = [x+rectangle_cropCircle[2],
                 y+rectangle_cropCircle[0],
                 w,
                 h,
                 initial_shape[0],
                 initial_shape[1],
                 tile_size[0],
                 tile_size[1]]

    return [image_id, img, rectangle]


# In[ ]:


def parallelize_image_cropping(image_ids):
    out = open('rectangles.csv', "w")
    out.write("image_id,type,x,y,w,h,img_shp_0_init,img_shape1_init,img_shp_0,img_shp_1\n")
    imf_d = {}
    p = Pool(cpu_count())
    for type in enumerate(types):
        partial_get_and_crop = partial(get_and_crop_image, image_type = type[1])    
        ret = p.map(partial_get_and_crop, image_ids[type[0]])
        for i in range(len(ret)):
            out.write(image_ids[type[0]][i])
            out.write(',' + str(type[1]))
            out.write(',' + str(ret[i][2][0]))
            out.write(',' + str(ret[i][2][1]))
            out.write(',' + str(ret[i][2][2]))
            out.write(',' + str(ret[i][2][3]))
            out.write(',' + str(ret[i][2][4]))
            out.write(',' + str(ret[i][2][5]))
            out.write(',' + str(ret[i][2][6]))
            out.write(',' + str(ret[i][2][7]))
            out.write('\n')
            img = get_image_data(image_ids[type[0]][i], type[1])
            if(img.shape[0] > img.shape[1]):
                tile_size = (int(img.shape[1]*256/img.shape[0]), 256)
            else:
                tile_size = (256, int(img.shape[0]*256/img.shape[1]))
            img = cv2.resize(img, dsize=tile_size)
            cv2.rectangle(img,
                          (ret[i][2][0], ret[i][2][1]), 
                          (ret[i][2][0]+ret[i][2][2], ret[i][2][1]+ret[i][2][3]),
                          255,
                          2)
            plt.imshow(img)
            plt.show()
        ret = []
    out.close()

    return


# In[ ]:


parallelize_image_cropping(type_ids)

