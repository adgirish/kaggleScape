
# coding: utf-8

# This notebook is a reworked version of William Cukierski's kernel [here](https://www.kaggle.com/wcukierski/example-metric-implementation). It gives the same results.
# 
# I wanted to use matrix multiplication for intersection and matrix maximum for union as this is easier to get my head around that the histogram approach. It will also help if I later code in C++ with binary maps. I also managed to simplify some of the formulae.
# 
# In the process, I noted that the evaluation method stipulates that the IOU score is based on the prediction masks, which is relevant where there is a mismatch between prediction and ground truth. Specifically if two true nuclei are fused together as one in the prediction or vice versa. So I think this code should work ok in the general case.
# 
# Hopefully this will help some of you. All comments welcome

# This is an example notebook to demonstrate how the IoU metric works for a single image. Please note: this is not the official scoring implementation, but should work in the same manner.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import skimage.segmentation

# Load a single image and its associated masks
id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
file = "../input/stage1_train/{}/images/{}.png".format(id,id)
mfile = "../input/stage1_train/{}/masks/*.png".format(id)
image = skimage.io.imread(file)
masks = skimage.io.imread_collection(mfile).concatenate()
height, width, _ = image.shape
num_masks = masks.shape[0]

# Make a ground truth array and summary label image
y_true = np.zeros((num_masks, height, width), np.uint16)
y_true[:,:,:] = masks[:,:,:] // 255  # Change ground truth mask to zeros and ones

labels = np.zeros((height, width), np.uint16)
labels[:,:] = np.sum(y_true, axis=0)  # Add up to plot all masks
    
# Show label images
fig = plt.figure()
plt.imshow(image)
plt.title("Original image")
fig = plt.figure()
plt.imshow(y_true[3])
plt.title("One example ground truth mask")
fig = plt.figure()
plt.imshow(labels)
plt.title("All ground truth masks")

# Simulate an imperfect submission
offset = 2 # offset pixels
y_pr1 = y_true[:19, offset:, offset:]  # To remove 'item 20' as per other kernel
y_pr2 = y_true[20:, offset:, offset:]
y_pred = np.concatenate((y_pr1, y_pr2), axis=0)
y_pred = np.pad(y_pred, ((0,0), (0, offset), (0, offset)), mode="constant")
#y_pred[y_pred == 20] = 0 # Remove one object
#y_pred, _, _ = skimage.segmentation.relabel_sequential(y_pred) # Relabel objects
yptot = np.sum(y_pred, axis=0)  # Sum individual predictions for plotting

# Show simulated predictions
fig = plt.figure()
plt.imshow(y_pred[3])
plt.title("An example simulated imperfect submission")
fig = plt.figure()
plt.imshow(yptot)
plt.title("All simulated imperfect submissions")
plt.show()

# Compute number of objects
num_true = len(y_true)
num_pred = len(y_pred)
print("Number of true objects:", num_true)
print("Number of predicted objects:", num_pred)

# Compute iou score for each prediction
iou = []
for pr in range(num_pred):
    bol = 0  # best overlap
    bun = 1e-9  # corresponding best union
    for tr in range(num_true):
        olap = y_pred[pr] * y_true[tr]  # Intersection points
        osz = np.sum(olap)  # Add the intersection points to see size of overlap
        if osz > bol:  # Choose the match with the biggest overlap
            bol = osz
            bun = np.sum(np.maximum(y_pred[pr], y_true[tr]))  # Union formed with sum of maxima
    iou.append(bol / bun)

# Loop over IoU thresholds
p = 0
print("Thresh\tTP\tFP\tFN\tPrec.")
for t in np.arange(0.5, 1.0, 0.05):
    matches = iou > t
    tp = np.count_nonzero(matches)  # True positives
    fp = num_pred - tp  # False positives
    fn = num_true - tp  # False negatives
    p += tp / (tp + fp + fn)
    print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, tp / (tp + fp + fn)))

print("AP\t-\t-\t-\t{:1.3f}".format(p / 10))


# This matches the table in Willam Cukierski's kernel.
