
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import glob, os


# # Read in files
# 
# This is pretty routine stuff.
# 
# * We get a list of jpeg files, reading them in as needed with `matplotlib.pyplot.imread`.

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
smjpegs = [f for f in glob.glob("../input/train_sm/*.jpeg")]
print(smjpegs[:9])


# In[ ]:


set175 = [smj for smj in smjpegs if "set175" in smj]
print(set175)


# # Basic exploration
# 
# Just look at image dimensions, confirm it's 3 band (RGB), byte scaled (0-255).

# In[ ]:


first = plt.imread('../input/train_sm/set175_1.jpeg')
dims = np.shape(first)
print(dims)


# In[ ]:


np.min(first), np.max(first)


# For any image specific classification, clustering, etc. transforms we'll want to 
# collapse spatial dimensions so that we have a matrix of pixels by color channels.

# In[ ]:


pixel_matrix = np.reshape(first, (dims[0] * dims[1], dims[2]))
print(np.shape(pixel_matrix))


# Scatter plots are a go to to look for clusters and separatbility in the data, but these are busy and don't reveal density well, so we
# switch to using 2d histograms instead. The data between bands is really correlated, typical with
# visible imagery and why most satellite image analysts prefer to at least have near infrared values.

# In[ ]:


#plt.scatter(pixel_matrix[:,0], pixel_matrix[:,1])
_ = plt.hist2d(pixel_matrix[:,1], pixel_matrix[:,2], bins=(50,50))


# In[ ]:


fifth = plt.imread('../input/train_sm/set175_5.jpeg')
dims = np.shape(fifth)
pixel_matrix5 = np.reshape(fifth, (dims[0] * dims[1], dims[2]))


# In[ ]:


_ = plt.hist2d(pixel_matrix5[:,1], pixel_matrix5[:,2], bins=(50,50))


# We can look at variations between the scenes now and see that there's a significant
# amount of difference, probably due to sensor angle and illumination variation. Raw band
# differences will need to be scaled or thresholded for any traditional approach.

# In[ ]:


_ = plt.hist2d(pixel_matrix[:,2], pixel_matrix5[:,2], bins=(50,50))


# In[ ]:


plt.imshow(first)


# In[ ]:


plt.imshow(fifth)


# Without coregistering portions of the image, the naive red band subtraction for change indication
# basically just shows the location shift between images.

# In[ ]:


plt.imshow(first[:,:,2] - fifth[:,:,1])


# In[ ]:


second = plt.imread('../input/train_sm/set175_2.jpeg')
plt.imshow(first[:,:,2] - second[:,:,2])


# In[ ]:


plt.imshow(second)


# # Initial impressions
# 
# Images aren't registered, so an image registration process between images with common overlap would probably be the first step in a traditional approach.
# Using a localizer in a deep learning context would probably be the newfangled way to tackle this.
# 
# Image content and differences will be dominated by topographic and built variations
# due to sensor orientation, resolution differences between scenes, and some registration accuracy will be impossible to factor out as
# the image hasn't been orthorectified and some anciliary data would be required for it
# to be done, e.g. georeferenceing against a previously orthorectified image.
# 
# So this is basically a basic computer vision task that deep learning will be a good fit for. The usual preprocessing steps
# and data expectations you'd see in remote sensing aren't fulfilled by this dataset.

# In[ ]:


# simple k means clustering
from sklearn import cluster

kmeans = cluster.KMeans(5)
clustered = kmeans.fit_predict(pixel_matrix)

dims = np.shape(first)
clustered_img = np.reshape(clustered, (dims[0], dims[1]))
plt.imshow(clustered_img)


# In[ ]:


plt.imshow(first)


# In[ ]:


ind0, ind1, ind2, ind3 = [np.where(clustered == x)[0] for x in [0, 1, 2, 3]]


# This code doesn't run on the server.
# 
# ```python
# from mpl_toolkits.mplot3d import Axes3D
# 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# 
# plot_vals = [('r', 'o', ind0),
#              ('b', '^', ind1),
#              ('g', '8', ind2),
#              ('m', '*', ind3)]
# 
# for c, m, ind in plot_vals:
#     xs = pixel_matrix[ind, 0]
#     ys = pixel_matrix[ind, 1]
#     zs = pixel_matrix[ind, 2]
#     ax.scatter(xs, ys, zs, c=c, marker=m)
# 
# ax.set_xlabel('Blue channel')
# ax.set_ylabel('green channel')
# ax.set_zlabel('Red channel')
# ```

# In[ ]:


# quick look at color value histograms for pixel matrix from first image
import seaborn as sns
sns.distplot(pixel_matrix[:,0], bins=12)
sns.distplot(pixel_matrix[:,1], bins=12)
sns.distplot(pixel_matrix[:,2], bins=12)


# In[ ]:


# even subsampling is throwing memory error for me, :p
#length = np.shape(pixel_matrix)[0]
#rand_ind = np.random.choice(length, size=50000)
#sns.pairplot(pixel_matrix[rand_ind,:])


# # Day 2
# 
# We'll start by considering the entire sequence of a different image set this time and look at strategies
# for matching features across scenes.

# In[ ]:


set79 = [smj for smj in smjpegs if "set79" in smj]
print(set79)


# In[ ]:


img79_1, img79_2, img79_3, img79_4, img79_5 =   [plt.imread("../input/train_sm/set79_" + str(n) + ".jpeg") for n in range(1, 6)]


# In[ ]:


img_list = (img79_1, img79_2, img79_3, img79_4, img79_5)

plt.figure(figsize=(8,10))
plt.imshow(img_list[0])
plt.show()


# Tracking dimensions across image transforms is annoying, so we'll make a class to do that.
# Also I'm going to use this brightness normalization transform and visualize the image that
# way, good test scenario for class.

# In[ ]:


class MSImage():
    """Lightweight wrapper for handling image to matrix transforms. No setters,
    main point of class is to remember image dimensions despite transforms."""
    
    def __init__(self, img):
        """Assume color channel interleave that holds true for this set."""
        self.img = img
        self.dims = np.shape(img)
        self.mat = np.reshape(img, (self.dims[0] * self.dims[1], self.dims[2]))

    @property
    def matrix(self):
        return self.mat
        
    @property
    def image(self):
        return self.img
    
    def to_flat_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form when
        derived image would only have one band."""
        return np.reshape(derived, (self.dims[0], self.dims[1]))
    
    def to_matched_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form."""
        return np.reshape(derived, (self.dims[0], self.dims[1], self.dims[2]))


# In[ ]:


msi79_1 = MSImage(img79_1)
print(np.shape(msi79_1.matrix))
print(np.shape(msi79_1.img))


# # Brightness Normalization
# 
# Brightness Normalization is preprocessing strategy you can apply prior to using strategies
# to identify materials in a scene, if you want your matching algorithm
# to be robust across variations in illumination. See [Wu's paper](https://pantherfile.uwm.edu/cswu/www/my%20publications/2004_RSE.pdf).

# In[ ]:


def bnormalize(mat):
    """much faster brightness normalization, since it's all vectorized"""
    bnorm = np.zeros_like(mat, dtype=np.float32)
    maxes = np.max(mat, axis=1)
    bnorm = mat / np.vstack((maxes, maxes, maxes)).T
    return bnorm


# In[ ]:


bnorm = bnormalize(msi79_1.matrix)
bnorm_img = msi79_1.to_matched_img(bnorm)
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()


# In[ ]:


msi79_2 = MSImage(img79_2)
bnorm79_2 = bnormalize(msi79_2.matrix)
bnorm79_2_img = msi79_2.to_matched_img(bnorm79_2)
plt.figure(figsize=(8,10))
plt.imshow(bnorm79_2_img)
plt.show()


# In[ ]:


msinorm79_1 = MSImage(bnorm_img)
msinorm79_2 = MSImage(bnorm79_2_img)

_ = plt.hist2d(msinorm79_1.matrix[:,2], msinorm79_2.matrix[:,2], bins=(50,50))


# In[ ]:


_ = plt.hist2d(msinorm79_1.matrix[:,1], msinorm79_2.matrix[:,1], bins=(50,50))


# In[ ]:


_ = plt.hist2d(msinorm79_1.matrix[:,0], msinorm79_2.matrix[:,0], bins=(50,50))


# In[ ]:


import seaborn as sns
sns.distplot(msinorm79_1.matrix[:,0], bins=12)
sns.distplot(msinorm79_1.matrix[:,1], bins=12)
sns.distplot(msinorm79_1.matrix[:,2], bins=12)


# In[ ]:


plt.figure(figsize=(8,10))
plt.imshow(img79_1)
plt.show()


# In[ ]:


np.max(img79_1[:,:,0])


# # Using thresholds with brightness normalization
# 
# Ok, so what am I even doing here? Well, my goal is to try and figure out simple threshold selection
# methods for getting high albedo targets out of a scene so I could then theoretically track them
# between scenes. For example, a simple blob/aggregation to centroid (in coordinates or in subsampled
# image bins) would give me a means to look at plausible structural similarities in distributions
# between scenes, then use that to anchor a comparison of things that change.
# 
# The brightness normalization step is helpful because thresholds that aren't anchored by a
# preprocessing step end up being arbitrary and can't generalize between scenes even in the same
# image set, whereas thresholds following brightness normalization tend to pull out materils that stand
# out from the background more reliably. See the following demonstration:

# In[ ]:


plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(img79_1[:,:,0] > 230)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()


# In[ ]:


plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(img79_2[:,:,0] > 230)
plt.subplot(122)
plt.imshow(img79_2)
plt.show()


# In[ ]:


print(np.min(bnorm79_2_img[:,:,0]))
print(np.max(bnorm79_2_img[:,:,0]))
print(np.mean(bnorm79_2_img[:,:,0]))
print(np.std(bnorm79_2_img[:,:,0]))


# In[ ]:


plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm79_2_img[:,:,0] > 0.98)
plt.subplot(122)
plt.imshow(img79_2)
plt.show()


# In[ ]:


plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm_img[:,:,0] > 0.98)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()


# In[ ]:


plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow((bnorm79_2_img[:,:,0] > 0.9999) &            (bnorm79_2_img[:,:,1] < 0.9999) &            (bnorm79_2_img[:,:,2] < 0.9999))
plt.subplot(122)
plt.imshow(img79_2)
plt.show()


# In[ ]:


plt.figure(figsize=(10,15))
plt.subplot(121)
plt.imshow(bnorm_img[:,:,0] > 0.995)
plt.subplot(122)
plt.imshow(img79_1)
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
plt.subplot(121)
plt.plot(bnorm_img[2000, 1000, :])
plt.subplot(122)
plt.plot(img79_1[2000, 1000, :])


# In[ ]:


from scipy import spatial

pixel = msi79_1.matrix[2000 * 1000, :]
np.shape(pixel)


# # Something's borked here
# 
# Think I'm gonna have to verify cosine similarity behavior for scipy here.
# 
# ```python
# def spectral_angle_mapper(pixel):
#     return lambda p2: spatial.distance.cosine(pixel, p2)
# 
# match_pixel = np.apply_along_axis(spectral_angle_mapper(pixel), 1, msi79_1.matrix)
# 
# plt.figure(figsize=(10,6))
# plt.imshow(msi79_1.to_flat_img(match_pixel < 0.0000001))
# 
# def summary(mat):
#     print("Max: ", np.max(mat),
#           "Min: ", np.min(mat),
#           "Std: ", np.std(mat),
#           "Mean: ", np.mean(mat))
#     
# summary(match_pixel)
# ```

# # Rudimentary Transforms, Edge Detection, Texture

# In[ ]:


set144 = [MSImage(plt.imread(smj)) for smj in smjpegs if "set144" in smj]


# In[ ]:


plt.imshow(set144[0].image)


# In[ ]:


import skimage
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import sobel


# # Sobel Edge Detection
# 
# A Sobel filter is one means of getting a basic edge magnitude/gradient image. Can be useful to
# threshold and find prominent linear features, etc. Several other similar filters in skimage.filters
# are also good edge detectors: `roberts`, `scharr`, etc. and you can control direction, i.e. use
# an anisotropic version.

# In[ ]:


# a sobel filter is a basic way to get an edge magnitude/gradient image
fig = plt.figure(figsize=(8, 8))
plt.imshow(sobel(set144[0].image[:750,:750,2]))


# In[ ]:


from skimage.filters import sobel_h

# can also apply sobel only across one direction.
fig = plt.figure(figsize=(8, 8))
plt.imshow(sobel_h(set144[0].image[:750,:750,2]), cmap='BuGn')


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(3)
pca.fit(set144[0].matrix)
set144_0_pca = pca.transform(set144[0].matrix)
set144_0_pca_img = set144[0].to_matched_img(set144_0_pca)


# In[ ]:


fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,0], cmap='BuGn')


# In[ ]:


fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,1], cmap='BuGn')


# In[ ]:


fig = plt.figure(figsize=(8, 8))
plt.imshow(set144_0_pca_img[:,:,2], cmap='BuGn')


# # GLCM Textures
# 
# Processing time can be pretty brutal so we subset the image. We'll create texture images so
# we can characterize each pixel by the texture of its neighborhood.
# 
# GLCM is inherently anisotropic but can be averaged so as to be rotation invariant. For more on GLCM, see [the tutorial](http://www.fp.ucalgary.ca/mhallbey/tutorial.htm).
# 
# A good article on use in remote sensing is [here](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=4660321&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D4660321):
# 
# Pesaresi, M., Gerhardinger, A., & Kayitakire, F. (2008). A robust built-up area presence index by anisotropic rotation-invariant textural measure. Selected Topics in Applied Earth Observations and Remote Sensing, IEEE Journal of, 1(3), 180-192.

# In[ ]:


sub = set144[0].image[:150,:150,2]


# In[ ]:


def glcm_image(img, measure="dissimilarity"):
    """TODO: allow different window sizes by parameterizing 3, 4. Also should
    parameterize direction vector [1] [0]"""
    texture = np.zeros_like(sub)

    # quadratic looping in python w/o vectorized routine, yuck!
    for i in range(img.shape[0] ):  
        for j in range(sub.shape[1] ):  
          
            # don't calculate at edges
            if (i < 3) or                (i > (img.shape[0])) or                (j < 3) or                (j > (img.shape[0] - 4)):          
                continue  
        
            # calculate glcm matrix for 7 x 7 window, use dissimilarity (can swap in
            # contrast, etc.)
            glcm_window = img[i-3: i+4, j-3 : j+4]  
            glcm = greycomatrix(glcm_window, [1], [0],  symmetric = True, normed = True )   
            texture[i,j] = greycoprops(glcm, measure)  
    return texture


# In[ ]:


dissimilarity = glcm_image(sub, "dissimilarity")


# In[ ]:


fig = plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.imshow(dissimilarity, cmap="bone")
plt.subplot(1,2,2)
plt.imshow(sub, cmap="bone")


# # HSV Transform
# 
# Since this contest is about time series ordering, I think it's possible there may be useful
# information in a transform to HSV color space. HSV is useful for identifying shadows and illumination, as well
# as giving us a means to identify similar objects that are distinct by color between scenes (hue), 
# though there's no guarantee the hue will be stable.

# In[ ]:


from skimage import color

hsv = color.rgb2hsv(set144[0].image)


# In[ ]:


fig = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(set144[0].image, cmap="bone")
plt.subplot(2,2,2)
plt.imshow(hsv[:,:,0], cmap="bone")
plt.subplot(2,2,3)
plt.imshow(hsv[:,:,1], cmap='bone')
plt.subplot(2,2,4)
plt.imshow(hsv[:,:,2], cmap='bone')


# In[ ]:


fig = plt.figure(figsize=(8, 8))
plt.subplot(2,2,1)
plt.imshow(set144[0].image[:200,:200,:])
plt.subplot(2,2,2)
plt.imshow(hsv[:200,:200,0], cmap="PuBuGn")
plt.subplot(2,2,3)
plt.imshow(hsv[:200,:200,1], cmap='bone')
plt.subplot(2,2,4)
plt.imshow(hsv[:200,:200,2], cmap='bone')


# In[ ]:


fig = plt.figure(figsize=(8, 6))
plt.imshow(hsv[200:500,200:500,0], cmap='bone')


# In[ ]:


hsvmsi = MSImage(hsv)


# # Shadow Detection
# 
# We can apply a threshold to the V band now to find dark areas that are probably thresholds. Let's
# look at the distribution of all values then work interactively to find a good filter value.

# In[ ]:


import seaborn as sns
sns.distplot(hsvmsi.matrix[:,0], bins=12)
sns.distplot(hsvmsi.matrix[:,1], bins=12)
sns.distplot(hsvmsi.matrix[:,2], bins=12)


# In[ ]:


plt.imshow(hsvmsi.image[:,:,2] < 0.4, cmap="plasma")


# In[ ]:


fig = plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.imshow(set144[0].image[:250,:250,:])
plt.subplot(1,2,2)
plt.imshow(hsvmsi.image[:250,:250,2] < 0.4, cmap="plasma")


# In[ ]:


fig = plt.figure(figsize=(8, 8))
img2 = plt.imshow(set144[0].image[:250,:250,:], interpolation='nearest')
img3 = plt.imshow(hsvmsi.image[:250,:250,2] < 0.4, cmap='binary_r', alpha=0.4)
plt.show()


# Could we glean something useful about sun position from shadow orientation if we could accurately
# reference the image?

# # Image Registration
# 
# This is an earlier form the library found [here](https://github.com/matejak/imreg_dft).
# 
# BSD family license, reproduced below with copyright so I can utilize similar functions here where
# import isn't available.
# 
# This version can be found [here](http://www.lfd.uci.edu/~gohlke/code/imreg.py.html).

# In[ ]:


# -*- coding: utf-8 -*-
# imreg.py

# Copyright (c) 2011-2014, Christoph Gohlke
# Copyright (c) 2011-2014, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""FFT based image registration.

Implements an FFT-based technique for translation, rotation and scale-invariant
image registration [1].

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2013.01.18

Requirements
------------
* `CPython 2.7 or 3.3 <http://www.python.org>`_
* `Numpy 1.7 <http://www.numpy.org>`_
* `Scipy 0.12 <http://www.scipy.org>`_
* `Matplotlib 1.2 <http://www.matplotlib.org>`_  (optional for plotting)

Notes
-----
The API and algorithms are not stable yet and are expected to change between
revisions.

References
----------
(1) An FFT-based technique for translation, rotation and scale-invariant
    image registration. BS Reddy, BN Chatterji.
    IEEE Transactions on Image Processing, 5, 1266-1271, 1996
(2) An IDL/ENVI implementation of the FFT-based algorithm for automatic
    image registration. H Xiea, N Hicksa, GR Kellera, H Huangb, V Kreinovich.
    Computers & Geosciences, 29, 1045-1055, 2003.
(3) Image Registration Using Adaptive Polar Transform. R Matungka, YF Zheng,
    RL Ewing. IEEE Transactions on Image Processing, 18(10), 2009.

Examples
--------
>>> im0 = imread('t400')
>>> im1 = imread('Tr19s1.3')
>>> im2, scale, angle, (t0, t1) = similarity(im0, im1)
>>> imshow(im0, im1, im2)

>>> im0 = imread('t350380ori')
>>> im1 = imread('t350380shf')
>>> t0, t1 = translation(im0, im1)

"""

from __future__ import division, print_function

import math

import numpy
from numpy.fft import fft2, ifft2, fftshift

try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii

__version__ = '2013.01.18'
__docformat__ = 'restructuredtext en'
__all__ = ['translation', 'similarity']


def translation(im0, im1):
    """Return translation vector to register images."""
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = numpy.unravel_index(numpy.argmax(ir), shape)
    if t0 > shape[0] // 2:
        t0 -= shape[0]
    if t1 > shape[1] // 2:
        t1 -= shape[1]
    return [t0, t1]


def similarity(im0, im1):
    """Return similarity transformed image im1 and transformation parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.

    A similarity transformation is an affine transformation with isotropic
    scale and without shear.

    Limitations:
    Image shapes must be equal and square.
    All image areas must have same scale, rotation, and shift.
    Scale change must be less than 1.8.
    No subpixel precision.

    """
    if im0.shape != im1.shape:
        raise ValueError("Images must have same shapes.")
    elif len(im0.shape) != 2:
        raise ValueError("Images must be 2 dimensional.")

    f0 = fftshift(abs(fft2(im0)))
    f1 = fftshift(abs(fft2(im1)))

    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    del h

    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)

    f0 = fft2(f0)
    f1 = fft2(f1)
    r0 = abs(f0) * abs(f1)
    ir = abs(ifft2((f0 * f1.conjugate()) / r0))
    i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
    angle = 180.0 * i0 / ir.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        ir = abs(ifft2((f1 * f0.conjugate()) / r0))
        i0, i1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)
        angle = -180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")

    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    im2 = ndii.zoom(im1, 1.0/scale)
    im2 = ndii.rotate(im2, angle)

    if im2.shape < im0.shape:
        t = numpy.zeros_like(im0)
        t[:im2.shape[0], :im2.shape[1]] = im2
        im2 = t
    elif im2.shape > im0.shape:
        im2 = im2[:im0.shape[0], :im0.shape[1]]

    f0 = fft2(im0)
    f1 = fft2(im2)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = numpy.unravel_index(numpy.argmax(ir), ir.shape)

    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]

    im2 = ndii.shift(im2, [t0, t1])

    # correct parameters for ndimage's internal processing
    if angle > 0.0:
        d = int((int(im1.shape[1] / scale) * math.sin(math.radians(angle))))
        t0, t1 = t1, d+t0
    elif angle < 0.0:
        d = int((int(im1.shape[0] / scale) * math.sin(math.radians(angle))))
        t0, t1 = d+t1, d+t0
    scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

    return im2, scale, angle, [-t0, -t1]


def similarity_matrix(scale, angle, vector):
    """Return homogeneous transformation matrix from similarity parameters.

    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector (of size 2).

    The order of transformations is: scale, rotate, translate.

    """
    S = numpy.diag([scale, scale, 1.0])
    R = numpy.identity(3)
    angle = math.radians(angle)
    R[0, 0] = math.cos(angle)
    R[1, 1] = math.cos(angle)
    R[0, 1] = -math.sin(angle)
    R[1, 0] = math.sin(angle)
    T = numpy.identity(3)
    T[:2, 2] = vector
    return numpy.dot(T, numpy.dot(R, S))


def logpolar(image, angles=None, radii=None):
    """Return log-polar transformed image and log base."""
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = numpy.empty((angles, radii), dtype=numpy.float64)
    theta.T[:] = -numpy.linspace(0, numpy.pi, angles, endpoint=False)
    #d = radii
    d = numpy.hypot(shape[0]-center[0], shape[1]-center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    radius = numpy.empty_like(theta)
    radius[:] = numpy.power(log_base, numpy.arange(radii,
                                                   dtype=numpy.float64)) - 1.0
    x = radius * numpy.sin(theta) + center[0]
    y = radius * numpy.cos(theta) + center[1]
    output = numpy.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output, log_base


def highpass(shape):
    """Return highpass filter to be multiplied with fourier transform."""
    x = numpy.outer(
        numpy.cos(numpy.linspace(-math.pi/2., math.pi/2., shape[0])),
        numpy.cos(numpy.linspace(-math.pi/2., math.pi/2., shape[1])))
    return (1.0 - x) * (2.0 - x)


def imread(fname, norm=True):
    """Return image data from img&hdr uint8 files."""
    with open(fname+'.hdr', 'r') as fh:
        hdr = fh.readlines()
    img = numpy.fromfile(fname+'.img', numpy.uint8, -1)
    img.shape = int(hdr[4].split()[-1]), int(hdr[3].split()[-1])
    if norm:
        img = img.astype(numpy.float64)
        img /= 255.0
    return img


def imshow(im0, im1, im2, im3=None, cmap=None, **kwargs):
    """Plot images using matplotlib."""
    from matplotlib import pyplot
    if cmap is None:
        cmap = 'coolwarm'
    if im3 is None:
        im3 = abs(im2 - im0)
    pyplot.subplot(221)
    pyplot.imshow(im0, cmap, **kwargs)
    pyplot.subplot(222)
    pyplot.imshow(im1, cmap, **kwargs)
    pyplot.subplot(223)
    pyplot.imshow(im3, cmap, **kwargs)
    pyplot.subplot(224)
    pyplot.imshow(im2, cmap, **kwargs)
    pyplot.show()


# We read in two files from the original set and compare them.

# In[ ]:


img1 = plt.imread(set175[1])
img2 = plt.imread(set175[3])
plt.figure(figsize=(8,10))
plt.subplot(121)
plt.imshow(img1)
plt.subplot(122)
plt.imshow(img2)


# Now transform `img2` red band to align with `img1` red band.

# In[ ]:


img3, scale, angle, (t0, t1) = similarity(img1[:,:,2], img2[:,:,2])


# # Viewing registration
# 
# We should be looking at:
#     
# * UL: template for transformation
# * UR: image that was transformed
# * LL: diff after transformation
# * LR: transformed image

# In[ ]:


imshow(img1[:,:,2], img2[:,:,2], img3)


# In[ ]:


# just diff
plt.imshow(img3 - img1[:,:,2])


# In[ ]:


# well, was working in a standalone private notebook, need to figure out what's off.
# it's fairly obviously offset/translation/shift is wrong somehow.
plt.imshow(img3)


# # NN with Downscaled Red Band Image
# 
# I'm exploring the best way to feed images into a siamese net like the one described [here](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zagoruyko_Learning_to_Compare_2015_CVPR_paper.pdf) and [here](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf).
# 
# Ideally you'd want to build the DNN to function this way:
# 
# * Feed in two images as in diagram in paper [1]:
# "A central-surround two-stream network that uses a
# siamese-type architecture to process each stream"
# * Train the DNN to learn a simple comparison function: 1 if image is before, 0 if false (no simultaneously captured images so no equality).
# * Once you have this comparator, then you can apply a sorting alrogithm to sort each dataset.
# 
# There are lots of details to be worked out here, still -- mainly trying to select overlapping image patches.

# In[ ]:


from skimage.transform import downscale_local_mean

img1 = plt.imread(set175[1])[:,:,2]
img2 = plt.imread(set175[3])[:,:,2]


# In[ ]:


np.shape(img1), np.shape(img2)


# In[ ]:


img1ds = downscale_local_mean(img1, (10, 10))
img2ds = downscale_local_mean(img2, (10, 10))
plt.figure(figsize=(8,10))
plt.subplot(121)
plt.imshow(img1ds[:225,:300])
plt.subplot(122)
plt.imshow(img2ds[:225,:300])


# For prototyping my goal is to get a bunch of downscaled images so that I can feed them in to test
# plausibility of basic NN architecture decisions.
# 
# Dims are subsampled and hard-coded for now so we don't have wrong dimensions showing up here and there.

# In[ ]:


def read_and_downscale(f):
    img = plt.imread(f)
    ds = downscale_local_mean(img[:,:,2], (10, 10))
    return ds[:225, :300]


# In[ ]:


set79 = [read_and_downscale("../input/train_sm/set79_" + str(n) + ".jpeg") for n in range(1, 6)]


# In[ ]:


def read_ds_set(setn):
    match_pre = "../input/train_sm/set" + str(setn) + "_"
    return [read_and_downscale(match_pre + str(n) + ".jpeg") for n in range(1, 6)]


# In[ ]:


set79 = read_ds_set(79)
set285 = read_ds_set(285)
set35 = read_ds_set(35)
set175 = read_ds_set(175)


# In[ ]:


plt.imshow(set79[0])


# In[ ]:


plt.imshow(set285[2])


# In[ ]:


plt.imshow(set35[3])


# # pairing function
# 
# We want to traverse multiple lists of images and construct all naive before/after pairs.
# 
# * Note there are actually more before/after pairs in this space, but this blows up quickly, at
#   least for little Kaggle notebooks.

# In[ ]:


def get_pairs(imgls):
    pairl = []
    for imgl in imgls:
        pairl += [(a,b) for a,b in zip(imgl[:-1], imgl[1:])]
    return pairl
            
paired = get_pairs([set35, set79, set285, set175])


# In[ ]:


# Let's just start with a dorky example.
rev_pairs = [(imgb, imga) for imga, imgb in paired]


# In[ ]:


img_a, img_b = rev_pairs[0]
concat_img = np.vstack((img_a, img_b))
print(np.shape(concat_img))
plt.imshow(concat_img)


# In[ ]:


import random

def concatter(imgpairs):
    for a, b in imgpairs:
        yield np.vstack((a, b))

concats = [cimg for cimg in concatter(paired + rev_pairs)]
random.shuffle(concats)


# # Oops!
# 
# Shuffled before supplying labels elsewhere about whether it's forward or backward comparison.
# 
# Going to have to leave this for tonight anyways.

# In[ ]:


plt.figure(figsize=(10,15))
plt.subplot(321)
plt.imshow(concats[0])
plt.subplot(322)
plt.imshow(concats[1])
plt.subplot(323)
plt.imshow(concats[2])
plt.subplot(324)
plt.imshow(concats[3])
plt.subplot(325)
plt.imshow(concats[4])
plt.subplot(326)
plt.imshow(concats[5])
plt.show()


# In[ ]:


# Need to simplify or build up next block from convnet template.


# ```python
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# 
# # Need to add dropouts
# model = Sequential()
# 
# # conv layer 1
# model.add(Convolution2D(50,1,2,2))
# model.add(Activation('relu'))
# 
# # conv layer 2
# model.add(Convolution2D(50, 32, 2, 2))
# model.add(Activation('relu')) 
# model.add(MaxPooling2D(poolsize=(2,2)))
# 
# # conv layer 3
# model.add(Convolution2D(50, 50, 2, 2))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(poolsize=(2,2)))
# 
# # feed to fully connected 
# model.add(Flatten())
# 
# # first fully connected
# model.add(Dense(1000, 128, init='glorot_uniform'))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# 
# # next fully connected
# model.add(Dense(128, 64, init='glorot_uniform'))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# 
# # last fully connected which outputs comparison result
# model.add(Dense(64, 1, init='glorot_uniform'))
# model.add(Activation('sigmoid'))
# 
# # compile model
# # model.compile(loss='binary_crossentropy', optimizer="rmsprop")
# ```
