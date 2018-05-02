
# coding: utf-8

# # Countless Sea Lions
# 
# In this challenge we are given a set of images and are asked to calculate the number of sea lions in each image, split by a few categories that are indicated in extra overlay images:
# 
# * red: adult males
# * magenta: subadult males
# * brown: adult females
# * blue: juveniles
# * green: pups
# 
# 
# ----------
# 
# 
# In this notebook we will be looking in detail at:
# 
# * numeric (counts of sea lions) feature exploration, both with correlation analysis and clustermaps
# * marker images and how we can use them to extract location and image patches for the marked sea lions
# * template matching to build a sea lion detector with OpenCV.
# 
# Let's get started with an initial peek at the stats of the data and later plot some sample images.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as plt
print(check_output(["ls", "../input"]).decode("utf8"))
from glob import glob
import seaborn as sns
from scipy import stats

df = pd.read_csv('../input/Train/train.csv')
print("{} training samples total".format(df.shape[0]))
df.head()


# Total counts of all sea lion types over all images.

# In[ ]:


df[['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']].sum(axis=0).plot.barh()


# # Correlations of sea lion counts
# 
# Now we will check how the counts of sea lions in the training images correlate which each other and how they are distributed.

# In[ ]:


def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.3f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

g = sns.PairGrid(df[['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']], palette=["red"])
#g.map_upper(plt.scatter, s=10)
g.map_lower(plt.scatter, s=10)
g.map_diag(sns.distplot, kde=False)
#g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_lower(corrfunc)
#sns.pairplot(df)


# It intuitively makes sense that the correlation of number of adult females and pups is rather strong, slightly more than **0.81** in that case. Also there is quite a strong correlation between adult male and female counts.

# # Distribution of sea lion counts
# 
# For that we want to get a first idea of what frequent patterns of sea lion fractions are present in the images.  We'll use seaborn's clustermap to visualize the potential clusters, but first, let's normalize the counts.
# 
# We will do that by summing over all columns for each row, that will give us the total count of sea lions for each image.

# In[ ]:


all_types = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']
all_normalized_types = ['normalized_'+t for t in all_types]
row_counts = df[['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']].sum(axis=1)

for t in all_types:
    df['normalized_'+t] = df[t].divide(row_counts)

df.head()


# In[ ]:


sns.clustermap(
    df[all_normalized_types].fillna(0.0),
    col_cluster=False,
    cmap=plt.get_cmap('viridis'),
    figsize=(12,10)
)


# From the above clustermap we can see that there are a few image clusters by sea lion fractions. For example, in the bottom left corner, we have a bunch of images that mostly contain adult males. Young males don't seem to hang around adult males.
# 
# Pups we normaly see when there is a moderate fraction of adult females present. It seems that when there are alot of adult females around, there aren't as many pups. Still mating maybe? :)
# 
# Also, if there are a lot of juveniles, there don't seem to be a lot of adult females.
# 
# All of this information can be used in a later stage of the competition to regularize the predicted counts towards one of the clustered proportions that we've found here.

# # Sea lion clustering
# 
# We'll now compute pairwise similarities of all of the images, where each image is represented as the fraction of sea lion types present in it. This is a slightly different perspective on the clustermap from above. We can now reason about pairwise similarities of images, e.g. the thick yellow line spectrum at the bottom likely corresponds to the mostly adult male pictures.

# In[ ]:


from scipy.spatial.distance import pdist, squareform

sq_dists = squareform(pdist(df[all_normalized_types].values))
sq_dists[np.isnan(sq_dists)] = 0.0
sns.clustermap(
    sq_dists,
    cmap=plt.get_cmap('viridis'),
    figsize=(12,10)
)


# # Sample Images
# 
# Each image in the training set as an unique training id and gives the counts of each sea lion category. 
# 
# Let's load a few images and have a look at them.

# In[ ]:


training_images = glob('../input/Train/*.jpg')
training_dotted = glob('../input/TrainDotted/*.jpg')
len(training_images), len(training_dotted)


# To the kernels on kaggle there only seem to be eleven images available.

# In[ ]:


fig = plt.figure(figsize=(16,10))
for i in range(4):
    ax = fig.add_subplot(2,2,i+1)
    plt.imshow(plt.imread(training_images[i]))


# The images seem to be really large and diverse with very different counts of sea lions on them. Let's have a look at the correlations of the sea lion counts.

# # Dotted training images
# 
# We'll now load a sample file with colored dot annotations, crop the image and show the dottet and non-dotted version.

# In[ ]:


from skimage.io import imread, imshow
from skimage.util import crop
import cv2

cropped_dotted = cv2.cvtColor(cv2.imread('../input/TrainDotted/8.jpg'), cv2.COLOR_BGR2RGB)[500:1500,2000:2800,:]
cropped_raw = cv2.cvtColor(cv2.imread('../input/Train/8.jpg'), cv2.COLOR_BGR2RGB)[500:1500,2000:2800,:]

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(1,2,1)
plt.imshow(cropped_dotted)
ax = fig.add_subplot(1,2,2)
plt.imshow(cropped_raw)


# # Plot the markers only
# 
# There are also brown markers which are removed by our thresholding and are also not very present in the difference image itself.

# In[ ]:


diff = cv2.subtract(cropped_dotted, cropped_raw)
diff = diff/diff.max()
plt.figure(figsize=(12,8))
plt.imshow((diff > 0.20).astype(float))
plt.grid(False)


# Thanks to [Asymptote's][1] kernel, we can have the locations of the thresholded markers and do 2D density estimation on them. This will give us an idea how sea lions like to hang around each other.
# 
# 
#   [1]: https://www.kaggle.com/asymptote

# In[ ]:


diff = cv2.absdiff(cropped_dotted, cropped_raw)
gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
ret,th1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cnts = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
print("Sea Lions Found: {}".format(len(cnts)))

x, y = [], []

lion_patches = []

for loc in cnts:
    ((xx, yy), _) = cv2.minEnclosingCircle(loc)

    # store patches of some sea lions
    if xx > 10 and xx < gray.shape[1] - 10:
        lion_patches.append(cropped_raw[yy-10:yy+10, xx-10:xx+10])

    x.append(xx)
    y.append(yy)

x = np.array(x)
y = np.array(y)


# The code below will take each of the markers as individual coordinates and feed it to a 2D kernel density estimation using a very common kernel, the gaussian one. We then show the estimated density by showing it's contour plot.
# 
# See [stackoverflow][1] for a very detailed explanation of the code below.
# 
# 
#   [1]: http://stackoverflow.com/questions/36957149/density-map-heatmaps-in-matplotlib

# In[ ]:


from scipy.stats.kde import gaussian_kde

k = gaussian_kde(np.vstack([x, y]), bw_method=0.5)
xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))


# In[ ]:


fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# alpha=0.5 will make the plots semitransparent
ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
ax2.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)

ax1.set_xlim(x.min(), x.max())
ax1.set_ylim(y.min(), y.max())
ax2.set_xlim(x.min(), x.max())
ax2.set_ylim(y.min(), y.max())

ax1.imshow(cropped_raw, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
ax2.imshow(cropped_raw, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')


# # Sea lion patches
# 
# Above we've also extracted a few quadratic shaped patches around each of the markers. Sea lions are not quadradic but pups come a little closer to quadratic shapes when viewed from top.
# 
# We can see that the extracted patches approximate the pups pretty good.

# In[ ]:


n_images_total = 16
n_images_per_row = 4

fig = plt.figure(figsize=(16,12))
for i in range(n_images_total):
    ax = fig.add_subplot(4,n_images_per_row,i+1)
    plt.grid(False)
    imshow(lion_patches[i])


# # Template matching
# 
# Given how large each of the images is, an ideal setting would be to have each sealion detected and properly extracted and then fed to an image classification pipeline, that can distinguish between the five classes of sealions. We would then classify each extracted sealion and simply sum them up for each category.
# 
# A first step to take into that direction is to build a sealion detector. For that we can utilize template matching, readily implemented in OpenCV. This approach is rather simple and will most likely be outperformed by a properly trained CNN, but let's see.
# 
# I will now select a few templates and run them all against our cropped image to see how well the templates generalize to unseen sea lions.
# 

# In[ ]:


# read images again for template matching code
cropped_dotted = cv2.cvtColor(cv2.imread('../input/TrainDotted/8.jpg'), cv2.COLOR_BGR2RGB)[1000:2000,2000:2800,:]
cropped_raw = cv2.cvtColor(cv2.imread('../input/Train/8.jpg'), cv2.COLOR_BGR2RGB)[1000:2000,2000:2800,:]
plt.imshow(cropped_raw)


# In[ ]:


plt.clf()
sealions = [
    cropped_raw[35:90, 505:520],
    cropped_raw[40:60, 510:515],
    cropped_raw[930:945, 610:665],
    cropped_raw[935:940, 630:645],
    cropped_raw[658:678, 395:448],
    cropped_raw[668:673, 415:420]
]
fig = plt.figure(figsize=(12,8))
for i in range(len(sealions)):
    ax = fig.add_subplot(1,len(sealions),i+1)
    imshow(sealions[i])


# The next section runs six different template matching methods over the cropped image from above. It shows the activation of each pixel, given the template. We can see, that different methods, activate differently over the cropped image.
# 
# To detect a sealion with this method, we will search for the maximum or the maxima in the activation map and indicate it with a rectangular box.

# In[ ]:


# All the 6 methods for comparison in a list
# not using all, to let kernel finish quickly
methods = [
    'cv2.TM_CCOEFF',
    'cv2.TM_CCOEFF_NORMED',
    'cv2.TM_CCORR',
    #'cv2.TM_CCORR_NORMED',
    #'cv2.TM_SQDIFF',
    #'cv2.TM_SQDIFF_NORMED'
]

def templateMatchFor(image, sealion):
    w, h = sealion.shape[1], sealion.shape[0]
    for meth in methods:
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(image,sealion,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(image,top_left, bottom_right, 255, 2)

        plt.figure(figsize=(12,8))
        plt.subplot(121)
        plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(image,cmap = 'gray')
        plt.title('Detected Point')
        plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()

[templateMatchFor(cropped_raw, sealion) for sealion in sealions]


# Ok, so by hand selecting a few of the sea lions as templates and matching them against our image with three different methods does yield very good generalization, in fact it did very poor, we only detected nine sea lions. :)
# By selecting only the color as a matching criterion, we've now detected a few more sea lions. Most probably though, this method is very sensitive to noise.
# The template matching doesn't seem right for our task, the shape and color of each of the sea lions is very different after all.

# ### to be continued.. :)
# 
# 
