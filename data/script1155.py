
# coding: utf-8

# ## Some code to save you time
# 
# A slight improvement on 'rle_encode' from my kernel https://www.kaggle.com/stainsby/fast-tested-rle used in the Carvana Image Masking Challenge. This kernel also reads all stage 1 test and training data into memory.
# 
# The RLE encoding routine is tested by:
# 
# 1. RLE encoding all masks and then decoding them and checking that the output matches the input mask.
# 2. Checking the generated RLEs against the supplied data in `stage1_train_labels.csv`.
# 
# Note that there was a issue with an earlier version encoding in the wrong direction that has now been fixed. Thanks to [Lam Dang](https://www.kaggle.com/lamdang) for pointing this out. **This may depend on the library that you use to read image files, so be aware that you will need to preprocess your mask with a transpose operation  if your library reads images as width × height instead of height × width** (thanks to [firolino](https://www.kaggle.com/firolino)).

# In[ ]:


import os, time
from pathlib import Path

import numpy as np

import imageio
import matplotlib.pyplot as plt


# In[ ]:


WORKING_DIR = Path(os.getcwd())
PROJECT_DIR = WORKING_DIR.parent
INPUT_DIR = PROJECT_DIR / 'input'
TRAIN_DIR = INPUT_DIR / 'stage1_train'
TEST_DIR = INPUT_DIR / 'stage1_test'

# Images with errors that should be skipped. Adjust as required.
TRAIN_ERROR_IDS = [
    '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80'
]


# ## Data input routines

# In[ ]:


def image_ids_in(root_dir, is_train_data=False):
    ids = []
    for id in os.listdir(root_dir):
        if id in TRAIN_ERROR_IDS:
            print('Skipping ID due to bad training data:', id)
        else:
            ids.append(id)
    return ids

TRAIN_IMAGE_IDS = image_ids_in(TRAIN_DIR, is_train_data=True)
TEST_IMAGE_IDS = image_ids_in(TEST_DIR)

print('Examples:', TRAIN_IMAGE_IDS[22], TEST_IMAGE_IDS[22])


# In[ ]:


def load_images(root_dir, ids, get_masks=False):
    images = []
    masks = []
    image_sizes = []
    for id in ids:
        item_dir = root_dir / id
        image_path = item_dir / 'images' / (id + '.png')
        image = imageio.imread(str(image_path))
        image = image[:, :, :3] # remove the alpha channel as it is not used
        images.append(image)
        image_sizes.append(image.shape[:2])
        if get_masks:
            mask_sequence = []
            masks_dir = item_dir / 'masks'
            mask_paths = masks_dir.glob('*.png')
            for mask_path in mask_paths:
                mask = imageio.imread(str(mask_path)) # 0 and 255 values
                mask = (mask > 0).astype(np.uint8) # 0 and 1 values
                mask_sequence.append(mask)
            masks.append(mask_sequence)
    if get_masks:
        return images, masks, image_sizes
    else:
        return images, image_sizes

TRAIN_IMAGES, TRAIN_MASKS, TRAIN_IMAGE_SIZES = load_images(TRAIN_DIR, TRAIN_IMAGE_IDS, True)
TEST_IMAGES, TEST_IMAGE_SIZES  = load_images(TEST_DIR, TEST_IMAGE_IDS, False)


# In[ ]:


def show_image_shape_stats(image_sizes, image_ids):
    print('  no. of images:', len(image_sizes))
    print('  first five shapes:', image_sizes[:5])
    image_sizes = np.asarray(image_sizes)
    image_ids = np.asarray(image_ids)
    print('  min. width:', image_sizes[:, 1].min(), '; max width:', image_sizes[:, 1].max())
    print('  min. height:', image_sizes[:, 0].min(), '; max height:', image_sizes[:, 0].max())
    pixel_counts = image_sizes.prod(axis=1)
    sorted_pixel_count_indices = np.argsort(pixel_counts)[::-1] # biggest to smallest
    print('  biggest images (by pixel count):\n   ',           '\n    '.join(image_ids[sorted_pixel_count_indices[:3]]))
    print('  smallest images (by pixel count):\n   ',           '\n    '.join(image_ids[sorted_pixel_count_indices[-3:]]))

print('Train image stats:')
show_image_shape_stats(TRAIN_IMAGE_SIZES, TRAIN_IMAGE_IDS)
print('\nTest image stats:')
show_image_shape_stats(TEST_IMAGE_SIZES, TEST_IMAGE_IDS)


# ## RLE routines

# In[ ]:


def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


# Used only for testing.
# This is copied from https://www.kaggle.com/paulorzp/run-length-encode-and-decode.
# Thanks to Paulo Pinto.
def rle_decode(rle_str, mask_shape, mask_dtype):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T


# ### Tests

# In[ ]:


# Used for testing only.
def read_sample_training_rles():
    is_header = True
    id_rle_map = {}
    for line in open(INPUT_DIR / 'stage1_train_labels.csv').readlines():
        if is_header:
            is_header = False
            continue
        id, rle = line.split(',')
        id = id.strip()
        rle = rle.strip()
        assert len(id) > 0
        assert len(rle) > 0
        if id in TRAIN_ERROR_IDS:
            continue
        rles = id_rle_map.get(id)
        if not rles:
            rles = set([])
            id_rle_map[id] = rles
        rles.add(rle)
    ids = id_rle_map.keys()
    print('Read sample RLEs for', len(ids), 'images')
    return id_rle_map
    

SAMPLE_TRAINING_RLES = read_sample_training_rles()


def test_rle_encode():
    test_mask = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]]).T
    assert rle_to_string(rle_encode(test_mask)) == '7 2 11 3'
    test_mask = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]]).T
    assert rle_to_string(rle_encode(test_mask)) == '7 2 11 3 16 1'
    test_mask = np.asarray([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]]).T
    assert rle_to_string(rle_encode(test_mask)) == '1 1 7 2 11 3'
    test_mask = np.asarray([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]]).T
    assert rle_to_string(rle_encode(test_mask)) == '1 1 7 2 11 3 16 1'
    num_images = len(TRAIN_IMAGES)
    print('Verfiying RLE encoding on', num_images, 'mask sequences ...')
    time_rle = 0.0 # seconds
    time_stringify = 0.0 # seconds
    mask_count = 0
    for image_idx in range(num_images):
        image_id = TRAIN_IMAGE_IDS[image_idx]
        mask_sequence = TRAIN_MASKS[image_idx]
        num_masks = len(mask_sequence)
        sample_rles = SAMPLE_TRAINING_RLES[image_id]
        assert num_masks == len(sample_rles),                 'number of masks should match that of RLEs in supplied sample'
        for mask_idx in range(num_masks):
            mask = mask_sequence[mask_idx]
            t0 = time.clock()
            rle = rle_encode(mask)
            assert len(rle) % 2 == 0 , 'RLE array length should be even'
            time_rle += time.clock() - t0
            t0 = time.clock()
            rle_str = rle_to_string(rle)
            time_stringify += time.clock() - t0
            assert rle_str in sample_rles, 'RLE not found in supplied sample'
            regenerated_mask = rle_decode(rle_str, mask.shape, mask.dtype)
            assert mask.dtype == regenerated_mask.dtype
            assert np.array_equal(mask.shape, regenerated_mask.shape),                     repr(mask.shape) + ' v. ' + repr(regenerated_mask.shape)
            assert np.array_equal(mask, regenerated_mask),                     'mask does not match regenerated mask'
            if mask_count and (mask_count % 5000) == 0:
                print('  ..', mask_count, 'masks tested ..')
            mask_count += 1
    print('Total number of masks encoded:', mask_count)
    print('Time spent RLE encoding masks:', time_rle, 's =>',             1000*(time_rle/mask_count), 'ms per mask.')
    print('Time spent stringifying RLEs:', time_stringify, 's =>',             1000*(time_stringify/mask_count), 'ms per mask.')


test_rle_encode()

