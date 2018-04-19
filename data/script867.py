
# coding: utf-8

# In[ ]:


#imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.image as mpimg
import random
from PIL import Image
import collections as co
import cv2
import scipy as sp
import copy

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

figWidth = figHeight = 10
whoAmI = 24601
random.seed(whoAmI)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
sns.set_style("dark")
# Any results you write to the current directory are saved as output.


# # Whales
# 
# Whales are cool! Mostly because I think animals are generally awesome, but Whales are truly badasses of the great blue sea. Let's try to identify them!
# 
# ## Summary Statistics

# In[ ]:


print(len(os.listdir("../input/train")))
print(len(os.listdir("../input/test")))


# In this case, we have more test data than we have training data. This means we have to especially optimize our models for out-of-sample prediction.

# In[ ]:


trainFrame = pd.read_csv("../input/train.csv")


# In[ ]:


trainFrame.shape


# Just to confirm, we do have the same number of training IDs as images in `../input/train`.

# In[ ]:


len(trainFrame["Id"].unique())


# We have a ton of unique labels! This is already shaping up to be a hard problem.
# 
# ## Distribution of Labels

# In[ ]:


idCountFrame = trainFrame.groupby("Id",as_index = False)["Image"].count()


# In[ ]:


idCountFrame = idCountFrame.rename(columns = {"Image":"numImages"})


# In[ ]:


idCountFrame["density"] = idCountFrame["numImages"] / np.sum(idCountFrame["numImages"])


# In[ ]:


idCountFrame = idCountFrame.sort_values("density",ascending = False)


# In[ ]:


#rank them
idCountFrame["rank"] = range(idCountFrame.shape[0])
idCountFrame["logRank"] = np.log(idCountFrame["rank"] + 1)


# In[ ]:


plt.plot(idCountFrame["logRank"],idCountFrame["density"])
plt.xlabel("$\log(Rank)$")
plt.ylabel("Density")
plt.title("$\log(Rank)$-Density Plot for our Labels")


# _Figure 1: $\log(Rank)$-Density Plot for our labels._
# 
# We see an extreme distribution forming, where we have many IDs with very few observations. This might suggest a severe imbalanced classes problem; We may in this case want to do some form of SMOTE for effectively accounting for imbalanced classes. Other than that, we will likely need to do many image transformations in order to bump up the observation numbers for some of our classes.
# 

# In[ ]:


topLev = 10
idCountFrame.iloc[0:topLev,:]


# _Table 1: Top 10 Most common labels._
# 
# We see that the largest class by a longshot is the `new_whale` label.

# ## Let's watch some whales
# 
# Let's open up some of the images themselves.
# 

# In[ ]:


trainDir = "../input/train"
testDir = "../input/test"


# In[ ]:


sampleImageFilename = "../input/train/00022e1a.jpg"
sampleImage = mpimg.imread(sampleImageFilename)
plt.imshow(sampleImage)


# _Figure 1: One of the images present in the training set._
# 
# A few things stand out about this image:
# 
# 1. It's already been altered some degree, seemingly emphasizes the R of the RGB spectrum present in the data. This might suggest that we will need to consider a classifier that is somewhat RGB-agnostic.
# 
# 2. It has a few parts of the image that aren't relevant to classification (see IDing at the bottom). I'm not sure whether this is supposed to correlate to a particular label in our training set, so it might be worthwhile exploring.

# In[ ]:


iiwcInfo = idCountFrame[idCountFrame["Id"].str.contains("iiwc")]
numberInfo = idCountFrame[idCountFrame["Id"].str.contains("1034")]
print(iiwcInfo)
print(numberInfo)


# In[ ]:


trainFrame[trainFrame["Id"] == "w_103488f"]


# Doesn't look to be correlated with ID. This is probably filler information from the camera capture.

# In[ ]:


#sample a couple of pictures
numSampled = 4
sampledPicNames = random.sample(os.listdir(trainDir),numSampled)
#then read the images
readImages = [mpimg.imread(trainDir + os.sep + sampledPicNames[i])
             for i in range(len(sampledPicNames))]
#then plot
fig, subplots = plt.subplots(2,2)
fig.set_size_inches(figWidth,figHeight)
for i in range(len(readImages)):
    subplots[int(i / 2),i % 2].imshow(readImages[i])


# _Figure 2: A couple of sample images from the training set._
# 
# A couple of interesting things are going on here:
# 
# 1. They are not all the same shape. This means we either need a shape-agnostic model. Or we will have to collapse dimensions of the pictures into the smallest recommended size.
# 
# 2. They are not all the same color spectrum with respect to RGB. We see the bottom two are in black-and-wihte, and the top two are in full color. This suggests again that we should be looking for a model or a feature extraction pipeline that is flexible to different coloring schemes in the pictures.

# In[ ]:


#then sample the test set
numSampled = 4
sampledPicNames = random.sample(os.listdir(testDir),numSampled)
#then read the images
readImages = [mpimg.imread(testDir + os.sep + sampledPicNames[i])
             for i in range(len(sampledPicNames))]
#then plot
fig, subplots = plt.subplots(2,2)
fig.set_size_inches(figWidth,figHeight)
for i in range(len(readImages)):
    subplots[int(i / 2),i % 2].imshow(readImages[i])


# _Figure 3: Some whales from the test set._
# 
# Similar situation where we have some images that are greyscaled (bottom-left), redscaled (top-left and bottom-right), and some that are in full color (top-right). At the very least, it does suggest a similar image quality issues in the test set as there is in the training set.
# 
# ### What is the scale of our image scaling issue?
# 
# I'm interesting in figuring out how much that scaling is an issue in our dataset. Let's check the distribution of image sizes.

# In[ ]:


imageSizes = co.Counter([Image.open(f'../input/train/{filename}').size
                        for filename in os.listdir("../input/train")])


# In[ ]:


imageSizeFrame = pd.DataFrame(list(imageSizes.most_common()),columns = ["imageDim","count"])


# In[ ]:


#get density
imageSizeFrame["density"] = imageSizeFrame["count"] / np.sum(imageSizeFrame["count"])
#get rank
imageSizeFrame["rank"] = range(imageSizeFrame.shape[0])
imageSizeFrame["logRank"] = np.log(imageSizeFrame["rank"] + 1)


# In[ ]:


#then plot
plt.plot(imageSizeFrame["logRank"],imageSizeFrame["density"])
plt.xlabel("$\log(Rank)$")
plt.ylabel("Density")
plt.title("$\log(Rank)$-Density Plot for image sizes in the training set")


# _Figure 4: $\log(Rank)$-Density PLot for image sizes in the training set._
# 
# We see that we have over $e^7 \approx 1100$ different image sizes. This means we will need to do some substantial resizing to do.

# In[ ]:


topLev = 10
imageSizeFrame.iloc[0:topLev,:]


# _Table 10: Top 10 Most Common Image Sizes in the training set._
# 
# We see that around $11\%$ of our training observations are of size $(1050,600)$, with around $9.65\%$ being of size $(1050,700)$. These would be good options for starting to standardize image sizes, but let's confirm that these are also the highest rank image sizes in the test set as well.

# In[ ]:


testImageSizesCounter = co.Counter([Image.open(f'../input/test/{filename}').size
                                    for filename in os.listdir("../input/test")])


# In[ ]:


testImageSizeFrame = pd.DataFrame(list(testImageSizesCounter.most_common()),
                                  columns = ["imageDim","count"])


# In[ ]:


#get density
testImageSizeFrame["density"] = testImageSizeFrame["count"] / np.sum(testImageSizeFrame["count"])
#get rank
testImageSizeFrame["rank"] = range(testImageSizeFrame.shape[0])
testImageSizeFrame["logRank"] = np.log(testImageSizeFrame["rank"] + 1)


# In[ ]:


topLev = 10
testImageSizeFrame.iloc[0:topLev,:]


# _Table 3: Top 10 Image Sizes for the test set._
# 
# The top two in the case are also $(1050,600)$ and $(1050,700)$. This suggests that if we pick one of the top image sizes to standardize the training set, it'll be a reasonabe choice in the test set as well.

# ### How many images are on different color scales?
# 
# We saw from our early EDA that some of the images are either on a greyscale or redscale format, which is different from typical RGB pictures. One of the questions we have is, how many images are on these different color sclaes?
# 
# #### Grayscale
# 
# We will use the following function for testing whether an image is grayscaled.

# In[ ]:


def is_grey_scale(givenImage):
    """Adopted from 
    https://www.kaggle.com/lextoumbourou/humpback-whale-id-data-and-aug-exploration"""
    w,h = givenImage.size
    for i in range(w):
        for j in range(h):
            r,g,b = givenImage.getpixel((i,j))
            if r != g != b: return False
    return True


# In[ ]:


sampleFrac = 0.1
#get our sampled images
imageList = [Image.open(f'../input/train/{imageName}').convert('RGB')
            for imageName in trainFrame['Image'].sample(frac=sampleFrac)]


# In[ ]:


isGreyList = [is_grey_scale(givenImage) for givenImage in imageList]


# In[ ]:


#then get proportion greyscale
np.sum(isGreyList) / len(isGreyList)


# We see that around half of the images in the training set are greyscale. This suggests to me that we need to create image transformations that are very agnostic to the RGB spectrum (i.e. bump up the number of greyscaled images in the smaller classes).
# 
# #### Redscale

# Unfortunately, I haven't found a function yet to identify whether there is redscaling in a picture. As indiciative of the bottom-right of Figure 3, there are in fact instances of redscaling in the dataset. If someone has a recommendation on how to identify redscaling in an image, that would be useful knowledge for this EDA.

# ### How different is our training set from our test set?
# 
# One of the questions that will be essential for out of-sample prediction optimization is to check how different our training set is in aggregate to our test set. Here's my approach. Take $D_{test}$ to be the test set and $D_{train}$ to be the training set.
# 
# 1. Sample $Y_1,...,Y_{1000} \sim D_{test}$ and $X_1,...,X_{1000} \sim D_{train}.$ Create pairs $\{(X_1,Y_1),...,(X_{1000},Y_{1000})\}.$
# 
# 2. Convert all $X_1,...,X_{1000},Y_1,...,Y_{1000}$ to black and white.
# 
# 3. Get the pixel value distribution for $X_1,...,X_{1000},Y_1,...,Y_{1000}$. Compute [Wasserstein Distance](https://en.wikipedia.org/wiki/Wasserstein_metric) for the pixel distributions for pairs $\{(X_1,Y_1),...,(X_{1000},Y_{1000})\}.$
# 
# 4. Compute the mean Wasserstein Distance $\hat{WS}$ from this sample.
# 
# 5. Bootstrap mean null Wasserstein Distance by performing $1000$ simulations of $1000$ samples each from $D = D_{test} \cup D_{train}.$
# 
# 6. Compute $p$-value for $\hat{WS}.$

# In[ ]:


#first get filenames
trainImageFilenames = os.listdir("../input/train")
testImageFilenames = os.listdir("../input/test")


# In[ ]:


#sample 1000 from each
sampleSize = 1000
trainImageFilenamesSample = random.sample(trainImageFilenames,sampleSize)
testImageFilenamesSample = random.sample(testImageFilenames,sampleSize)


# In[ ]:


#then get images
trainImageSample = [cv2.imread(f'../input/train/{trainImageFilename}',0)
                    for trainImageFilename in trainImageFilenamesSample]
testImageSample = [cv2.imread(f'../input/test/{testImageFilename}',0)
                    for testImageFilename in testImageFilenamesSample]


# In[ ]:


#then get histograms for each
colorMax = 256
trainImageHists =  [cv2.calcHist([trainImage],[0],None,[colorMax],[0,colorMax]).squeeze()
                    for trainImage in trainImageSample]
testImageHists =  [cv2.calcHist([testImage],[0],None,[colorMax],[0,colorMax]).squeeze()
                    for testImage in testImageSample]


# In[ ]:


#normalize each
trainImageHists = [trainImageHist / np.sum(trainImageHist) for trainImageHist in trainImageHists]
testImageHists = [testImageHist / np.sum(testImageHist) for testImageHist in testImageHists]


# In[ ]:


#then get wasserstein distances
wassersteinDistances = [sp.stats.energy_distance(trainImageHists[i],testImageHists[i])
                        for i in range(len(trainImageHists))]


# In[ ]:


testStatistic = np.mean(wassersteinDistances)


# In[ ]:


testStatistic


# This is relatively small for Wasserstein Distances. Let's see if this is anything significant based on our train-test split.

# In[ ]:


def bootstrapMeanWassersteinDistance(imageList,numSamples):
    """Helper for bootstrapping the mean wasserstein distance from a given filename list"""
    #first get full sample
    fullSampleImages = random.sample(imageList,numSamples * 2)
    #then get train-test split by indices
    fullSampleImageIndices = [i for i in range(len(fullSampleImages))]
    trainImageSampleIndices = random.sample(fullSampleImageIndices,numSamples)
    testImageSampleIndices = list(set(fullSampleImageIndices) - set(trainImageSampleIndices))
    #then actually get said images
    trainImageSample = [fullSampleImages[i] for i in trainImageSampleIndices]
    testImageSample = [fullSampleImages[i] for i in testImageSampleIndices]
    #then get histograms
    colorMax = 256
    trainImageHists =  [cv2.calcHist([trainImage],[0],None,[colorMax],[0,colorMax]).squeeze()
                        for trainImage in trainImageSample]
    testImageHists =  [cv2.calcHist([testImage],[0],None,[colorMax],[0,colorMax]).squeeze()
                        for testImage in testImageSample]
    #normalize each
    trainImageHists = [trainImageHist / np.sum(trainImageHist) 
                       for trainImageHist in trainImageHists]
    testImageHists = [testImageHist / np.sum(testImageHist) 
                      for testImageHist in testImageHists]
    #then get wasserstein distances
    wassersteinDistances = [sp.stats.energy_distance(trainImageHists[i],testImageHists[i])
                        for i in range(len(trainImageHists))]
    #then get test statistic
    return np.mean(wassersteinDistances)


# In[ ]:


def runSimulations(imageList,numSims,numSamples):
    """Helper that bootstraps our full distribution of mean wasserstein distances"""
    wdDist = [bootstrapMeanWassersteinDistance(imageList,numSamples) for i in range(numSims)]
    return wdDist


# In[ ]:


#then form filename list
trainImageFilenames = [f'../input/train/{trainImageFilename}'
                       for trainImageFilename in trainImageFilenames]
testImageFilenames = [f'../input/test/{testImageFilename}'
                       for testImageFilename in testImageFilenames]


# In[ ]:


filenameList = copy.deepcopy(trainImageFilenames)
filenameList.extend(testImageFilenames)


# In[ ]:


#and because we would crash Kaggle if we loaded in all the images, let's just load in 8000
metaSampleSize = 8000
filenameSample = random.sample(filenameList,metaSampleSize)


# In[ ]:


imageList = [cv2.imread(filename,0) for filename in filenameSample]


# In[ ]:


numSims = 100
numSamples = 1000
wdDist = runSimulations(imageList,numSims,numSamples)


# In[ ]:


wdDistVec = np.array(wdDist)
np.mean(wdDistVec > testStatistic)


# Not significant! This suggests that our training samples are generally similar to our test samples, so there isn't systematic bias in this context.
# 
# # Conclusion
# 
# We have a couple takeaways from the EDA:
# 
# * There are many classes that are underrepresented in the dataset. This suggests that we will need to make a large number of transformations to our underrepresented classes in order to give them a re-balance this classification problem.
# 
# * We see that there are thousands of different image sizes in this dataset. We may need to find a way to crop and pad thes images until they fit the largest image size class of $(1050,600)$ or the second largest, which is $(1050,700).$
# 
# * We see that we have a mixture of different color schemes to the pictures. Many of them are greyscale, some of them are redscale, and a portion of them are genuine color pictures. We need to ensure that our classifier is color scheme agnostic by greyscaling and redscaling the color pictures we can find.
# 
# * We see that by our Wasserstein Distance exercise, the pixel makeup of the training set is insignificantly different from the pixel makeup of the test set. Thus, we would argue that the test set is not that systematically different than the training set, which puts us in a good place for out-of-sample prediction.
# 
# Next step: transformations! Coming soon...
