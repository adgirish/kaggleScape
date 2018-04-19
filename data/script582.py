
# coding: utf-8

# # Intro
# Step-by-step guide for extracting features from shapes by turning them into time-series. The functions are optimized for the Swedish Leaf Dataset as it is published on Kaggle.
# 
# Data Scientists spend vast majority of their time by data preparation, not model optimization. So let's dive into that a bit.
# 
# ## [Here is the second part][1], it is more advanced and [follows PEP8][2]
# 
# ![Feature Extraction from leaf contours][3]
# 
# 
# 
#   [1]: https://www.kaggle.com/lorinc/leaf-classification/feature-extraction-v4/
#   [2]: https://www.python.org/dev/peps/pep-0008/
#   [3]: https://raw.githubusercontent.com/lorinc/kaggle-notebooks/master/extracted_leaf_shape.png

# In[ ]:


# imports
import numpy as np                     # numeric python lib

import matplotlib.image as mpimg       # reading images to numpy arrays
import matplotlib.pyplot as plt        # to plot any graph
import matplotlib.patches as mpatches  # to draw a circle at the mean contour

from skimage import measure            # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality


# matplotlib setup
get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import rcParams
rcParams['figure.figsize'] = (6, 6)      # setting default size of plots


# In[ ]:


# reading an image file using matplotlib into a numpy array
# good ones: 11, 19, 23, 27, 48, 53, 78, 218
img = mpimg.imread('../input/images/53.jpg')

# using image processing module of scipy to find the center of the leaf
cy, cx = ndi.center_of_mass(img)

plt.imshow(img, cmap='Set3')  # show me the leaf
plt.scatter(cx, cy)           # show me its center
plt.show()


# Later we might want to switch to another measure of centrality, based on how efficient this center is, when we generate a time-series from the shape, using the distance between the edge and the center.
# 
# One way to do that is just measure the (Euclidean) distance between the center and the edge... but there is a better way - we project the Cartesian coordinates into Polar coordinates.
# 
# But before that, we need to [find the edges][1] of the leaf.
# 
#   [1]: http://scikit-image.org/docs/dev/auto_examples/

# In[ ]:


# scikit-learn imaging contour finding, returns a list of found edges
contours = measure.find_contours(img, .8)

# from which we choose the longest one
contour = max(contours, key=len)

# let us see the contour that we hopefully found
plt.plot(contour[::,1], contour[::,0], linewidth=0.5)  # (I will explain this [::,x] later)
plt.imshow(img, cmap='Set3')
plt.show()


# Now we can project this contour - that is, thousands of pairs of (x,y) coordinates - into the Polar coordinate system.
# ![Cartesian to Polar][1]
# [1]: http://jwilson.coe.uga.edu/emat6680fa11/lee/asnmt11hylee/fig2.jpg

# In[ ]:


# cartesian to polar coordinates, just as the image shows above
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]

# just calling the transformation on all pairs in the set
polar_contour = np.array([cart2pol(x, y) for x, y in contour])

# and plotting the result
plt.plot(polar_contour[::,1], polar_contour[::,0], linewidth=0.5)
plt.show()


# What on earth is this. We expected a time-series, then we got a monster-leaf.
# 
# Let's try the projection again, but move the leaf to (0,0) first this time.
# 
# This gives a chance to look into something we already used, but not explained: [numpy.ndarray indexing][1]. "nd" stands for "n-dimensional" and there is a powerful syntax helps us to select whatever slice we need from our array. The syntax is `array[start:stop:step]`. Also, this syntax allows us to select into subdimensions.
# 
# 
#   [1]: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html

# In[ ]:


# numpy BASIC indexing example, see link above for more

x = np.array([[[1,11,111], [2,22,222], [3,33,333]], 
              [[4,44,444], [5,55,555], [6,66,666]], 
              [[7,77,777], [8,88,888], [9,99,999]]])

# reverse the first dimension
# take the 0th element
# and take its last element
x[::-1, 0, -1]


# So, using this, let us demean the contour data.
# 
# Demean (making its mean 0 by shifting points) is needed: the polar coordnate projection failed to yield what we want, because the shape is in the +,+ part of the Cartesian system, not around the center.
# 
# Why the contour, why not the image? First, image is big, contour is small. Second, image is just hundreds n hundreds of bits on a grid, not (x,y) pairs. It is not the right data format for us.

# In[ ]:


# numpy is smart and assumes the same about us
# if we substract a number from an array of numbers,
# it assumes that we wanted to substract from all members
contour[::,1] -= cx  # demean X
contour[::,0] -= cy  # demean Y


# In[ ]:


# checking if we succeeded to move the center to (0,0)
plt.plot(-contour[::,1], -contour[::,0], linewidth=0.5)
plt.grid()
plt.scatter(0, 0)
plt.show()


# Now we can try to project it again into polar space.

# In[ ]:


# just calling the transformation on all pairs in the set
polar_contour = np.array([cart2pol(x, y) for x, y in contour])

# and plotting the result
rcParams['figure.figsize'] = (12, 6)
plt.subplot(121)
plt.scatter(polar_contour[::,1], polar_contour[::,0], linewidth=0, s=.5, c=polar_contour[::,1])
plt.title('in Polar Coordinates')
plt.grid()
plt.subplot(122)
plt.scatter(contour[::,1],             # x axis is radians
            contour[::,0],             # y axis is distance from center
            linewidth=0, s=2,          # small points, w/o borders
            c=range(len(contour)))     # continuous coloring (so that plots match)
plt.scatter(0, 0)
plt.title('in Cartesian Coordinates')
plt.grid()
plt.show()


# Ok, I consider this beginning part done.
# 
# Challenges:
# 
# - This is definitely not a time-series yet, as one x can have multiple y values. (I do not know yet if it is a good thing for the classification and feature extraction or not)
# - I have to solve that the same leaf if I rotate it, is still equivalent from feature extraction and classification point of view.
# - And - of course - I'm yet to find and quantify meaningful and generic features.
# 
# And a funny thought - what if I cut each sample into half along the symmetry axis and treat them as two samples? How shameless would that be? I get double sample size, I generalize better, I simplify and reduce the features I work with... 
# 
# ![Evil genius laughter][1]
# 
#   [1]: http://www.jimbowley.com/wp-content/uploads/2011/06/DrEvilPinky.jpg

# In[ ]:


# check a few scikitlearn image feature extractions, if they can help us

from skimage.feature import corner_harris, corner_subpix, corner_peaks, CENSURE

detector = CENSURE()
detector.detect(img)

coords = corner_peaks(corner_harris(img), min_distance=5)
coords_subpix = corner_subpix(img, coords, window_size=13)

plt.subplot(121)
plt.title('CENSURE feature detection')
plt.imshow(img, cmap='Set3')
plt.scatter(detector.keypoints[:, 1], detector.keypoints[:, 0],
              2 ** detector.scales, facecolors='none', edgecolors='r')

plt.subplot(122)
plt.title('Harris Corner Detection')
plt.imshow(img, cmap='Set3')  # show me the leaf
plt.plot(coords[:, 1], coords[:, 0], '.b', markersize=5)
plt.show()


# I really want to find features that generalize well, but this is just random noise.
# 
# How about finding local maxima and minima on the time series instead?

# In[ ]:


from scipy.signal import argrelextrema

# for local maxima
c_max_index = argrelextrema(polar_contour[::,0], np.greater, order=50)
c_min_index = argrelextrema(polar_contour[::,0], np.less, order=50)

plt.subplot(121)
plt.scatter(polar_contour[::,1], polar_contour[::,0], 
            linewidth=0, s=2, c='k')
plt.scatter(polar_contour[::,1][c_max_index], 
            polar_contour[::,0][c_max_index], 
            linewidth=0, s=30, c='b')
plt.scatter(polar_contour[::,1][c_min_index], 
            polar_contour[::,0][c_min_index], 
            linewidth=0, s=30, c='r')

plt.subplot(122)
plt.scatter(contour[::,1], contour[::,0], 
            linewidth=0, s=2, c='k')
plt.scatter(contour[::,1][c_max_index], 
            contour[::,0][c_max_index], 
            linewidth=0, s=30, c='b')
plt.scatter(contour[::,1][c_min_index], 
            contour[::,0][c_min_index], 
            linewidth=0, s=30, c='r')

plt.show()


# Ok, I surprised myself. This worked out pretty well. I think, I can build an extremely efficient feature from this.
# 
# But this method is NOT robust yet.
# 
# - It is not finding the tips, but the points with the greatest distance from center. (look at leaf#19)
# - It will miserably fail on a more complex, or unfortunately rotated leaf. (look at leaf#78)
# 
# I really have to smooth the contour to find the dominant symmetry. Also, it would give me a better generalization about the shape. Then I could measure the distance between the smooth contour and the actual one to quantify the variability of the contour.
# 
# There is a feature in `scipy.ndimage` called 'Mathematical Morphology' that can do stuff like this: 
# 
# ![Scipy's Mathematical Morphology functions][1]
# 
# 
#   [1]: http://www.scipy-lectures.org/_images/morpho_mat.png

# In[ ]:


def cont(img):
    return max(measure.find_contours(img, .8), key=len)

# let us set the 'brush' to a 6x6 circle
struct = [[ 0., 0., 1., 1., 0., 0.],
          [ 0., 1., 1., 1., 1., 0.],  
          [ 1., 1., 1., 1., 1., 1.], 
          [ 1., 1., 1., 1., 1., 1.], 
          [ 1., 1., 1., 1., 1., 1.], 
          [ 0., 1., 1., 1., 1., 0.],
          [ 0., 0., 1., 1., 0., 0.]]

erosion = cont(ndi.morphology.binary_erosion(img, structure=struct).astype(img.dtype))
closing = cont(ndi.morphology.binary_closing(img, structure=struct).astype(img.dtype))
opening = cont(ndi.morphology.binary_opening(img, structure=struct).astype(img.dtype))
dilation = cont(ndi.morphology.binary_dilation(img, structure=struct).astype(img.dtype))

plt.imshow(img.T, cmap='Greys', alpha=.2)
plt.plot(erosion[::,0], erosion[::,1], c='b')
plt.plot(opening[::,0], opening[::,1], c='g')
plt.plot(closing[::,0], closing[::,1], c='r')
plt.plot(dilation[::,0], dilation[::,1], c='k')
#plt.xlim([220, 420])
#plt.ylim([250, 420])
plt.xlim([0, 400])
plt.ylim([400, 800])
plt.show()


# This does not make ANY sense. It looks like there is noise around the edges of the original image, and I was not aware of that.

# In[ ]:


plt.imshow(img.astype(bool).astype(float), cmap='hot')
plt.show()


# Here we go. We have noise. This challenge just got way more realistic. (Sh*t.) It seems I must learn how to denoise.
# 
# Ok. The pixels are 8bit greyscale in the original image. Check if I can separate the leaf from the noise by their value.
# 
# If not, I have a nasty plan. I invert the whole image, do a binary_closing on it to fill up the small holes, then invert it back - now without the noise.

# In[ ]:


erosion = cont(ndi.morphology.binary_erosion(img > 254, structure=struct).astype(img.dtype))
closing = cont(ndi.morphology.binary_closing(img > 254, structure=struct).astype(img.dtype))
opening = cont(ndi.morphology.binary_opening(img > 254, structure=struct).astype(img.dtype))
dilation = cont(ndi.morphology.binary_dilation(img > 254, structure=struct).astype(img.dtype))

plt.imshow(img.T, cmap='Greys', alpha=.2)
plt.plot(erosion[::,0], erosion[::,1], c='b')
plt.plot(opening[::,0], opening[::,1], c='g')
plt.plot(closing[::,0], closing[::,1], c='r')
plt.plot(dilation[::,0], dilation[::,1], c='k')
plt.xlim([0, 400])
plt.ylim([400, 800])
plt.show()


# From these 2 binary morphology tests, it is clear:
# 
# - the leaf has debris around it's edge
# - there are not 100% white pixels at the edge
# 
# But for me this second red contour is just perfect. I will use that as baseline. (yes, I loose some minimal contour information, but it makes my measurements more accurate in exchange)

# In[ ]:


# While I was looking up how to threshold an image, I have found this.
# The further away from the edges a pixel is, the higher value it gets.
# This is important, because it describes the morphology of the leaf better, 
# than a simple euclidean distance from the center, because it considers
# concave parts differently, and that's an important feature I wish to keep.

# This is very promising. Using this, I will probably be able to find symmetry.

# I also believe, that using this distance map, I will be able to separate
# the core shape of the leaf and the edge texture, which are two distinct,
# pretty good features.

dist_2d = ndi.distance_transform_edt(img)
plt.imshow(img, cmap='Greys', alpha=.2)
plt.imshow(dist_2d, cmap='plasma', alpha=.2)
plt.contour(dist_2d, cmap='plasma')
plt.show()

