
# coding: utf-8

# ## Fast run length encoding, tested on the provided training mask data.

# In[1]:


import time

import numpy as np
import pandas as pd
from scipy import ndimage

from matplotlib import pyplot as plt

PROJECT_PATH = '..'
INPUT_PATH = PROJECT_PATH + '/input'

TRAIN_MASKS_CSV_PATH = INPUT_PATH + '/train_masks.csv'
TRAIN_MASKS_PATH = INPUT_PATH + '/train_masks'


# In[2]:


def read_train_masks():
    global train_masks
    train_masks = pd.read_csv(TRAIN_MASKS_CSV_PATH)
    print(train_masks.head())


read_train_masks()


# In[3]:


def read_mask_image(car_code, angle_code):
    mask_img_path = TRAIN_MASKS_PATH + '/' + car_code + '_' + angle_code + '_mask.gif';
    mask_img = ndimage.imread(mask_img_path, mode = 'L')
    mask_img[mask_img <= 127] = 0
    mask_img[mask_img > 127] = 1
    return mask_img


def show_mask_image(car_code, angle_code):
    mask_img = read_mask_image(car_code, angle_code)
    plt.imshow(mask_img, cmap = 'Greys_r')
    plt.show()


show_mask_image('00087a6bd4dc', '04')


# In[4]:


def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def test_rle_encode():
    test_mask = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
    assert rle_to_string(rle_encode(test_mask)) == '7 2 11 2'
    num_masks = len(train_masks['img'])
    print('Verfiying RLE encoding on', num_masks, 'masks ...')
    time_read = 0.0 # seconds
    time_rle = 0.0 # seconds
    time_stringify = 0.0 # seconds
    for mask_idx in range(num_masks):
        img_file_name = train_masks.loc[mask_idx, 'img']
        car_code, angle_code = img_file_name.split('.')[0].split('_')
        t0 = time.clock()
        mask_image = read_mask_image(car_code, angle_code)
        time_read += time.clock() - t0
        t0 = time.clock()
        rle_truth_str = train_masks.loc[mask_idx, 'rle_mask']
        rle = rle_encode(mask_image)
        time_rle += time.clock() - t0
        t0 = time.clock()
        rle_str = rle_to_string(rle)
        time_stringify += time.clock() - t0
        assert rle_str == rle_truth_str
        if mask_idx and (mask_idx % 500) == 0:
            print('  ..', mask_idx, 'tested ..')
    print('Time spent reading mask images:', time_read, 's, =>',             1000*(time_read/num_masks), 'ms per mask.')
    print('Time spent RLE encoding masks:', time_rle, 's, =>',             1000*(time_rle/num_masks), 'ms per mask.')
    print('Time spent stringifying RLEs:', time_stringify, 's, =>',             1000*(time_stringify/num_masks), 'ms per mask.')


test_rle_encode()

