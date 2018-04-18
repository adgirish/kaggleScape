
# coding: utf-8

# # Introduction <a name="introduction"></a>
# 
# This Kernel explore the **train** and **test** datasets from [Google Landmark Recognition Challenge](https://www.kaggle.com/c/landmark-recognition-challenge). References [1-2] were used as a starting point for this Kernel. As the images in the datasets will have to be downloaded in order to conduct an analysis on the images itselfs, the Kernel is not covering the image analysis part. We include code (from Reference [3]) that will allow one competitor to retrieve tags informations from the url images.   
# 
# Please feel free to **fork and further develop** this Kernel.   
# 
# 
# ![Petronas Twin Towers, Kuala Lumpur, Malaysia](http://lh4.ggpht.com/-Szw4nwa8izg/StLpb6miB4I/AAAAAAAAAJk/cDTWbVgI4Lg/s1600/)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from IPython.core.display import HTML 
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from urllib import request
from io import BytesIO
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read data  <a name="readdata"></a>

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')


# # Inspect the data <a name="inspectdata"></a>

# ## Data shape

# In[ ]:


print("Train data shape -  rows:",train_df.shape[0]," columns:", train_df.shape[1])
print("Test data size -  rows:",test_df.shape[0]," columns:", test_df.shape[1])


# ## Glimpse the data
# 
# Let's inspect the train and test sets

# In[ ]:


train_df.head()


# Train set has three columns, first being an id for the image, the second being an url for the image and the third the id of the landmark associated with the image.

# In[ ]:


test_df.head()


# Test set has two columns, first being an id for the image, the second being an url for the image.

# Let's see now the expected format for the submission file

# In[ ]:


submission.head()


# Submission has two columns, first being an id for the image, the second being the landmark. This has two elements: an landmark id that is associated with the image and its corresponding confidence score. Some query images may contain no landmarks. For these, one can submit no landmark id (and no confidence score).

# ## Data quality
# 
# Let's look into more details to the data quality
# 
# 
# ### Train data quality
# 
# Let's see if we do have missing values in the training set

# In[ ]:


# missing data in training data set
missing = train_df.isnull().sum()
all_val = train_df.count()

missing_train_df = pd.concat([missing, all_val], axis=1, keys=['Missing', 'All'])
missing_train_df


# We see that we do not have any missing values (null values) in the training data
# 
# ### Test data quality
# 
# Let's see if we do have missing values in the test set

# In[ ]:


# missing data in training data set
missing = test_df.isnull().sum()
all_val = test_df.count()

missing_test_df = pd.concat([missing, all_val], axis=1, keys=['Missing', 'All'])
missing_test_df


# We can see that we do not have any missing values (null values) in the test data
# 
# 
# ## Unique values
# 
# Let's inspect the train and test data to check now many unique values are
# 

# In[ ]:


train_df.nunique()


# In the train dataset, there are only 14951 unique landmark_id data. All id's and url's are unique. 
# 
# Let's see now the test data to check now many unique values are

# In[ ]:


test_df.nunique()


# All id's and url's are unique in the test data as well. Let's now check if we do have any id's or url's that are in both train and test set. 

# In[ ]:


# concatenate train and test datasets
concatenated = pd.concat([train_df, test_df])
# print the shape of the resulted data.frame
concatenated.shape


# In[ ]:


concatenated.nunique()


# All id's and url's are unique for the concatenated data. That means we do not have any id's or url's from train dataset leaked in the test data set as well.

# ## Landmarks
# 
# We already know how many distincts landmarks there are in the train set. Let's inspect now how many occurences are for these landscapes in the train set.

# In[ ]:


plt.figure(figsize = (8, 8))
plt.title('Landmark id density plot')
sns.kdeplot(train_df['landmark_id'], color="tomato", shade=True)
plt.show()


# Let's represent the same data as a density plot

# In[ ]:


plt.figure(figsize = (8, 8))
plt.title('Landmark id distribuition and density plot')
sns.distplot(train_df['landmark_id'],color='green', kde=True,bins=100)
plt.show()


# Let's look now to the most frequent landmarks in the train set and also to the least frequent landmarks.

# In[ ]:


th10 = pd.DataFrame(train_df.landmark_id.value_counts().head(10))
th10.reset_index(level=0, inplace=True)
th10.columns = ['landmark_id','count']
th10


# Most frequent landmark has 50337 apparitions in train dataset.

# In[ ]:


# Plot the most frequent landmark occurences
plt.figure(figsize = (6, 6))
plt.title('Most frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=th10,
            label="Count", color="darkgreen")
plt.show()


# In[ ]:


tb10 = pd.DataFrame(train_df.landmark_id.value_counts().tail(10))
tb10.reset_index(level=0, inplace=True)
tb10.columns = ['landmark_id','count']
tb10


# In[ ]:


# Plot the least frequent landmark occurences
plt.figure(figsize = (6,6))
plt.title('Least frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=tb10,
            label="Count", color="orange")
plt.show()


# Least frequent landmarks have only one occurence in the train dataset.

# # Image paths <a name="imagepaths"></a>
# 
# Let's check the image paths. When we first analyzed the images, we noticed that there are just few main repositories used. Let's try now to find the names of these repositories.

# In[ ]:


# Extract repositories names for train data
ll = list()
for path in train_df['url']:
    ll.append((path.split('//', 1)[1]).split('/', 1)[0])
train_df['site'] = ll
# Extract repositories names for test data
ll = list()
for path in test_df['url']:
    ll.append((path.split('//', 1)[1]).split('/', 1)[0])
test_df['site'] = ll


# Let's check the shape again for train and test datasets.

# In[ ]:


print("Train data shape -  rows:",train_df.shape[0]," columns:", train_df.shape[1])
print("Test data size -  rows:",test_df.shape[0]," columns:", test_df.shape[1])


# We added to train and test data sets one more column, `site`, storing the name of the image repository. Let's also glimpse the train and test again, to check on the new column values.

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# Let's group now on `site` name. We process both the train and test data.

# In[ ]:


train_site = pd.DataFrame(train_df.site.value_counts())
test_site = pd.DataFrame(test_df.site.value_counts())


# The sites in train data are:

# In[ ]:


train_site


# In[ ]:


# Plot the site occurences in the train dataset
trsite = pd.DataFrame(list(train_site.index),train_site['site'])
trsite.reset_index(level=0, inplace=True)
trsite.columns = ['Count','Site']
plt.figure(figsize = (6,6))
plt.title('Sites storing images - train dataset')
sns.set_color_codes("pastel")
sns.barplot(x = 'Site', y="Count", data=trsite, color="blue")
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()


# We can observe that most of the images in the train dataset are stored on 4 sites, *lh3.googleusercontent.com*, *lh4.googleusercontent.com*, *lh5.googleusercontent.com* and *lh6.googleusercontent.com*.
# 
# The sites in test dataset are:

# In[ ]:


test_site


# In[ ]:


# Plot the site occurences in the test dataset
tesite = pd.DataFrame(list(test_site.index),test_site['site'])
tesite.reset_index(level=0, inplace=True)
tesite.columns = ['Count','Site']
plt.figure(figsize = (6,6))
plt.title('Sites storing images - test dataset')
sns.set_color_codes("pastel")
sns.barplot(x = 'Site', y="Count", data=tesite, color="magenta")
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()


# We can observe that most of the images in the test dataset are stored on one site, *lh3.googleusercontent.com*, which is also the one with most content stored for train dataset.
# Let's look now to the images.

# # Image thumbnails <a name="imagethumbnails"></a>
# 
# Let's inspect also the images. We create a function to display a certain number of images, giving a list of images urls. We show here a number of `50` images of the `Petronas Twin Towers` in Kuala Lumpur, which is the 5th ranged landmark in the selection of landmarks, based on number of occurences.
# 
# We will define two functions to display landmarks.
# 

# In[ ]:


def displayLandmarkImages(urls):
    
    imageStyle = "height: 60px; margin: 2px; float: left; border: 1px solid blue;"
    imagesList = ''.join([f"<img style='{imageStyle}' src='{u}' />" for _, u in urls.iteritems()])

    display(HTML(imagesList))
    
    
def displayLandmarkImagesLarge(urls):
    
    imageStyle = "height: 100px; margin: 2px; float: left; border: 1px solid blue;"
    imagesList = ''.join([f"<img style='{imageStyle}' src='{u}' />" for _, u in urls.iteritems()])

    display(HTML(imagesList))


# In[ ]:


IMAGES_NUMBER = 50
landmarkId = train_df['landmark_id'].value_counts().keys()[5]
urls = train_df[train_df['landmark_id'] == landmarkId]['url'].head(IMAGES_NUMBER)
displayLandmarkImages(urls)


# Let's visualize now 5 images for each of the first 5 landmarks, ordered by the number of occurences.

# In[ ]:


LANDMARK_NUMBER = 5
IMAGES_NUMBER = 5
landMarkIDs = pd.Series(train_df['landmark_id'].value_counts().keys())[1:LANDMARK_NUMBER+1]
for landMarkID in landMarkIDs:
    url = train_df[train_df['landmark_id'] == landMarkID]['url'].head(IMAGES_NUMBER)
    displayLandmarkImagesLarge(url)


# # Extracting Exif data and GPS data <a name="extractexif"></a>
# 
# We will not be able to use the following code with this Kernel (feel free to download it) because there are some missing libraries support. It is not actually allowed to stream image data (we are allowed to display images, thought) on Kaggle so two libraries that will help us to do this are missing. The original code is from Reference [3] with a small correction for the way the image data taken from [Anokas](https://www.kaggle.com/anokas)'s Kernel (Reference [4]).

# In[ ]:



class ImageMetaData(object):
    '''
    Extract the exif data from any image. Data includes GPS coordinates, 
    Focal Length, Manufacture, and more.
    '''
    exif_data = None
    image = None

    def __init__(self, img_path):
        
        response = request.urlopen(url)
        image_data = response.read()
        self.image = Image.open(BytesIO(image_data))
        self.get_exif_data()
        super(ImageMetaData, self).__init__()

    def get_exif_data(self):
        """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
        exif_data = {}
        info = self.image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                if decoded == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_decoded = GPSTAGS.get(t, t)
                        gps_data[sub_decoded] = value[t]

                    exif_data[decoded] = gps_data
                else:
                    exif_data[decoded] = value
        self.exif_data = exif_data
        return exif_data

    def get_if_exist(self, data, key):
        if key in data:
            return data[key]
        return None

    def convert_to_degress(self, value):

        """Helper function to convert the GPS coordinates 
        stored in the EXIF to degress in float format"""
        d0 = value[0][0]
        d1 = value[0][1]
        d = float(d0) / float(d1)

        m0 = value[1][0]
        m1 = value[1][1]
        m = float(m0) / float(m1)

        s0 = value[2][0]
        s1 = value[2][1]
        s = float(s0) / float(s1)

        return d + (m / 60.0) + (s / 3600.0)

    def get_lat_lng(self):
        """Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)"""
        lat = None
        lng = None
        exif_data = self.get_exif_data()
        #print(exif_data)
        if "GPSInfo" in exif_data:      
            gps_info = exif_data["GPSInfo"]
            gps_latitude = self.get_if_exist(gps_info, "GPSLatitude")
            gps_latitude_ref = self.get_if_exist(gps_info, 'GPSLatitudeRef')
            gps_longitude = self.get_if_exist(gps_info, 'GPSLongitude')
            gps_longitude_ref = self.get_if_exist(gps_info, 'GPSLongitudeRef')
            if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
                lat = self.convert_to_degress(gps_latitude)
                if gps_latitude_ref != "N":                     
                    lat = 0 - lat
                lng = self.convert_to_degress(gps_longitude)
                if gps_longitude_ref != "E":
                    lng = 0 - lng
        return lat, lng
    
    


# ## Retrieve metadata example
# 
# Here is an example of usage of the ImageMetaData function.
# 
# > meta_data =  ImageMetaData(urls.head(1))  
# > latlng =meta_data.get_lat_lng()  
# > print(latlng)  
# > exif_data = meta_data.get_exif_data()  
# > print(exif_data)  
# 

# # Baseline submission
# 
# We are using a random guess, normalized by the frequency in the training set to prepare a submission file. The solution is picked up from Kevin Mader's Kernel, [Baseline Landmark Model](ttps://www.kaggle.com/kmader/baseline-landmark-model).
# 

# In[ ]:


# take the most frequent label
freq_label = train_df['landmark_id'].value_counts()/train_df['landmark_id'].value_counts().sum()

# submit the most freq label
submission['landmarks'] = '%d %2.2f' % (freq_label.index[0], freq_label.values[0])
submission.to_csv('submission.csv', index=False)

np.random.seed(2018)
r_idx = lambda : np.random.choice(freq_label.index, p = freq_label.values)

r_score = lambda idx: '%d %2.4f' % (freq_label.index[idx], freq_label.values[idx])
submission['landmarks'] = submission.id.map(lambda _: r_score(r_idx()))
submission.to_csv('rand_submission.csv', index=False)


# # Feedback requested <a name="fr"></a>
# Your suggestions and comments for improvement of this Kernel are much appreciated. And, of course, if you like it, **upvote**!

# # References <a name="ref"></a><a class="anchor" id="ref"></a>
# 
# 
# [1] Max Diebold, Simple exploration of Google Recognition,  https://www.kaggle.com/mxdbld/simple-exploration-of-google-recognition  
# [2] Ashok LathwalI, Introduction and overview,   https://www.kaggle.com/codename007/introduction-and-overview   
# [3] Extract GPS & Exif Data from Images using Python, https://www.codingforentrepreneurs.com/blog/extract-gps-exif-images-python/  
# [4] Python3 Dataset Downloader with progress bar, https://www.kaggle.com/anokas/python3-dataset-downloader-with-progress-bar  
# [5] Kevin Mader, Baseline Landmark Model, https://www.kaggle.com/kmader/baseline-landmark-model  
# 
# 
