
# coding: utf-8

# # Exploratory Analysis
# 
# During my undergraduate studies I had biology as one of my subjects and I enjoyed preparing the slides and looking at them using a microscope. So these images are interesting, and I get to learn a bit more about how Deep Learning impact the medical field.
# 
# I liked the video overview given by [Dr. Anne Carpenter](https://www.kaggle.com/drannecarpenter) and I am sure I will be able to learn quite a few things by participating in this.
# 
# The images consist of different modalities and magnification as mentioned in this [thread](https://www.kaggle.com/c/data-science-bowl-2018/discussion/47640)
# 
# > We want a single model that just works across all kinds of image modalities, no matter the size of the nuclei or the color scheme. Such a model could be built into software that biologists use with all kinds of microscopes and eliminate the need for them to train on their individual data or provide metadata about their cell type, microscope, resolution, etc.
# 
# > <cite>[Dr. Anne Carpenter](https://www.kaggle.com/drannecarpenter)</cite>
# 
# ### Disclaimer
# 
# My knowledge of Computer Vision, Microbiology & Deep Learning is very limited. I have been trying to follow courses and learn these interesting techniques, but I have realized that this is a vast field and there are too many things to learn.
# 
# * My approach may not be the right one, but I am still going to try
# * Most of the code here is borrowed from what others have been doing. I have just tweaked a few things here and there and expanded on the steps, so that I can see and understand what is happening behind the scenes.
# * Any suggestions to improve the code here is welcome.

# ## References
# 
# * [Adrian Rosebrock's PyImageSearch](https://www.pyimagesearch.com) which covers great tutorials on OpenCV

# ## Approach
# 
# I am a newbie to Deep Learning and Computer Vision and this competition is my playground for learning. I will approach this competition with the limited knowledge that I have, and I will also attempt to incorporate methods that others are experimenting with, if I can grasp them and can implement on my own. 
# 
# There are a lot of people who have built interesting kernels which take different approaches to reach the end goal.

# ## Questions
# 
# Here are the questions I set out to answer in this notebook.
# 
# * How many types of images are there?
# * Are the types of images in train and test similar or vastly different?
# * How does the image segmentation and object detection compare to the masks provided for the training set?

# In[ ]:


import numpy as np
import pandas as pd
import os
import glob
import cv2
import math
import seaborn as sns
import json

sns.set()
sns.set_palette("husl")


# In[ ]:


TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

RANDOM_SEED=75

OUTPUT_PATH = './'
CONTOUR_EXTRACT_MODE = cv2.RETR_TREE


# Ensuring that only the folders are picked up and any junk files in the same path are ignored

# In[ ]:


train_ids = [x for x in os.listdir(TRAIN_PATH) if os.path.isdir(TRAIN_PATH+x)]
test_ids = [x for x in os.listdir(TEST_PATH) if os.path.isdir(TEST_PATH+x)]


# Create a pandas dataframe combining all images and marking them as train or test. This way we can do a comparison across all images.

# In[ ]:


df = pd.DataFrame({'id':train_ids,'train_or_test':'train'})
df = df.append(pd.DataFrame({'id':test_ids,'train_or_test':'test'}))

df.groupby(['train_or_test']).count()


# There are 670 training images and 65 test images making the test set less than 10% of the training set.
# 
# Build the paths for the individual images

# In[ ]:


df['path'] = df.apply(lambda x:'../input/stage1_{}/{}/images/{}.png'.format(x[1],x[0],x[0]), axis=1)


# ### Observations
# 
# I had initially glanced through some of the sample images and there were different types of images. This was before I read the thread on modalities and zooms. 
# 
# I saw that some were grayscale with black background and nuclei in grayscale intensities. Some of them were in color and some of them seemed to be black on white. Now, I have read that most of the CV tasks work best on grayscale images and it makes sense if all images follow a similar pattern of white on black. The code below helps us cluster the image into two groups. My assumption is that the group that forms the larger cluster is probably the background coolor and the other one is the foreground color. 

# Create a histogram and identify the centroid of the group as the color intensity whose variations are most similar to it. 
# > <cite>[k-means-color-clustering](https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/)</cite>

# In[ ]:


from sklearn.cluster import KMeans

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


# Given a path for an image, load the image and extract the shape attributes. Perform KMeans clustering on the RGB format. Using this extract the background and foreground colors. Since we have the color values for different channels it would be interesting to see how these values are spread across the images. I have calculated the grayscale value as an average of all three colors and this is marked as teh background and foreground color. 
# 
# If the foreground color is darker than the background color mark the image to be inverted before processing. 
# 
# There are different methods for converting a RGB to grayscale and I can revisit this later if the average value is not good enough.

# In[ ]:


def get_image_info(path, clusters=2):
    image = cv2.imread(path)
    height,width,_ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = clusters)
    clt.fit(image)
    hist = centroid_histogram(clt)
    
    bg_idx, fg_idx = 0, clusters-1
    if hist[bg_idx] < hist[fg_idx]:
        bg_idx, fg_idx = clusters-1, 0
    
    bg_red, bg_green, bg_blue = clt.cluster_centers_[bg_idx]
    fg_red, fg_green, fg_blue = clt.cluster_centers_[fg_idx]
    
    bg_color = sum(clt.cluster_centers_[bg_idx])/3
    fg_color = sum(clt.cluster_centers_[fg_idx])/3
    max_color_pct = hist[bg_idx]
    min_color_pct = hist[fg_idx]
    
    return (pd.Series([height,width,
                       bg_red, bg_green, bg_blue, bg_color,
                       fg_red, fg_green, fg_blue, fg_color,
                       hist[bg_idx],hist[fg_idx],
                       fg_color < bg_color]))


# Collect information about the images and save it.

# In[ ]:


image_info = os.path.join(OUTPUT_PATH,'images.json')

if os.path.isfile(image_info):
    with open(image_info, 'r') as datafile:
        data = json.load(datafile)
        df = pd.read_json(path_or_buf=data, orient='records')
        data = None
else:
    names = ['height','width',
             'bg_red', 'bg_green', 'bg_blue','bg_color',
             'fg_red', 'fg_green', 'fg_blue','fg_color',
             'bg_color_pct','fg_color_pct','invert']

    df[names] = df['path'].apply(lambda x: get_image_info(x))
    df['shape'] = df[['height','width']].apply(lambda x: '{:04d}x{:04d}'.format(x[0], x[1]), axis=1)

    with open(image_info, 'w') as outfile:
        json.dump(df.to_json(orient='records'), outfile)


# In[ ]:


len(df['shape'].unique()),len(df['width'].unique()), len(df['height'].unique())


# ## Distribition of images
# 
# Time to play around with some graphs and see how the data is spread across all images.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[ ]:


agg = df[['shape','train_or_test','id']].groupby(['shape','train_or_test']).count().unstack()
agg.columns = agg.columns.droplevel()


# In[ ]:


agg.plot.barh(stacked=True,figsize=(16,4))
plt.show()


# In[ ]:


agg[agg['train'].isnull()]


# The distribution is certainly skewed, which I guess is expected. As Anne mentioned there are some image sizes in test set that are not in the training set. These could be of different modality or magnification. There are 7 out of 16 different shapes that are in Test data set only and not in the train data set.
# 
# We can pick a random sample from each image size and have a look.

# In[ ]:


def show_image(ax,title,image):
    ax.grid(None)
    ax.set_title(title)
    ax.imshow(image)


# In[ ]:


def n_of_each(df, n = 4):
    shapes = df['shape'].unique()
    sample = pd.DataFrame()
    
    for shape in shapes:
        sample = sample.append(df[df['shape']==shape].sample(n, replace=True))
    
    return sample.sort_values(by=['shape']).reset_index()


# In[ ]:


def show_row_col(sample,cols,path_col='path',image_col=None,label_col='title',mode='file'):
    rows = math.ceil(len(sample)/cols)
    
    fig, ax = plt.subplots(rows,cols,figsize=(5*cols,5*rows))
    
    for index, data in sample.iterrows():
    
        title = data[label_col]
        if mode=='image':
            image = np.array(data[image_col],dtype=np.uint8)
            #image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.imread(data[path_col])
            image = cv2.cvtColor(image,cv2.COLOR_BGRA2RGB)

        row = index // cols
        col = index % cols
        show_image(ax[row,col],title,image)

    plt.show()    


# In[ ]:


sample = n_of_each(df)


# In[ ]:


sample['label'] = sample[['shape','train_or_test']].apply(lambda x: '{},{}'.format(x[0],x[1]), axis=1)
show_row_col(sample,4,path_col='path',label_col='label',mode='file')


# ## Masks
# 
# Extract the masks metadata into a separate file. This should include 
# 
# * mask_id
# * a separate index number for all the mask objects
# * minimum bounding rectangle
# * angle of rotation (longer edge horizontal)
# * masked area
# * length & breadth
# 
# By using contour identification process individual mask objects can be extracted and rotated to a common rectangular shape and we can then get answers to more questions .

# Bounded rotation ensures that resulting rotated image is not clipped.

# In[ ]:


def rotate_bound(image, cX, cY, angle, box = None):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    #(cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)) 


# In[ ]:


def rotate_points(points,center,origin,angle):
    rbox = np.array(points.copy())
    
    rbox[:,0] -= center[0]
    rbox[:,1] -= center[1]
    
    cos = math.cos(np.radians(angle))
    sin = math.sin(np.radians(angle))
    rmat = [[cos, -sin],
            [sin, cos]]
    
    rbox = np.dot(np.matrix(rbox),rmat)
    
    rbox[:,0] += origin[0]  
    rbox[:,1] += origin[1]
    
    return np.array(rbox)


# Given a mask and an image perform the following actions
# 
# * Identify the contours in the mask
# * Apply mask on the image to leave only the identified objects
# * Identify contours in the mask
# * For each contour extract metadata about the mask
# * Extract the nuclei and rotate it to get a clipped horizontal rectangle

# In[ ]:


def extract(mask, image, mask_id, image_id, frame=3):
    _, contours, _ = cv2.findContours(mask, CONTOUR_EXTRACT_MODE, cv2.CHAIN_APPROX_NONE)

    data = []
    
    if len(image.shape) > 2:
        all_nuclei = image.copy()
        for i in range(image.shape[2]):
            all_nuclei[:,:,i] = np.bitwise_and(all_nuclei[:,:,i],mask)
    else:
        all_nuclei = np.bitwise_and(image,mask)

    for contour in contours:
        ((cx, cy), r) =  cv2.minEnclosingCircle(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        angle = rect[2]
    
        x1 = max(0,math.floor(cx-r-frame))
        y1 = max(0,math.floor(cy-r-frame))

        x2 = min(mask.shape[1],math.ceil(cx+r+frame+1))
        y2 = min(mask.shape[0],math.ceil(cy+r+frame+1))

        sq_mask =  mask[y1:y2,x1:x2]
        square =  all_nuclei[y1:y2,x1:x2,:]
        
        offset_cx = int(cx-x1)
        offset_cy = int(cy-y1)
        
        # offset the points
        box[:,0] -= x1
        box[:,1] -= y1
        # print(box)
        
        rotated = rotate_bound(sq_mask,offset_cx,offset_cy,-angle)
        nuclei = rotate_bound(square,offset_cx,offset_cy,-angle)
        rbox = rotate_points(box,(offset_cx,offset_cy),(nuclei.shape[1]/2,nuclei.shape[0]/2),angle)

        xmin, xmax = max(0,math.floor(min(rbox[:,0]))), math.ceil(max(rbox[:,0]))
        ymin, ymax = max(0,math.floor(min(rbox[:,1]))), math.ceil(max(rbox[:,1]))

        #print(xmin,xmax,ymin,ymax)
        nuclei = nuclei[ymin:ymax+1,xmin:xmax+1]
        h = ymax - ymin + 1
        w = xmax - xmin + 1
        if h > w:
            nuclei = rotate_bound(nuclei,int(w/2),int(h/2),90)
            angle = angle -90
            #h,w = nuclei.shape
            
        data.append({
            'image_id':image_id,
            'mask_id':mask_id,
            'nuclei':nuclei.tolist(),
            'width':nuclei.shape[1],
            'height':nuclei.shape[0],
            'cx':cx,
            'cy':cy,
            'square':square.tolist(),
            'offset_cx':offset_cx,
            'offset_cy':offset_cy,
            'radius':r,
            'box':box.tolist(),
            'angle':angle
        })
        
    return(data)


# Masks are in the masks directory and each mask is in a different file. 
# 

# In[ ]:


# lambda function to flatten a list

flatten = lambda l: [item for sublist in l for item in sublist]


# In[ ]:


def get_masks(image_id, prefix, invert=False, enhance=True):
    
    image_file = '{}/{}/images/{}.png'.format(prefix,image_id,image_id)
    image = cv2.imread(image_file) #, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image,cv2.COLOR_BGRA2RGB)
    
    if enhance:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
        for channel in range(image.shape[2]):
            image[:,:,channel] = clahe.apply(image[:,:,channel])

    if invert:
        image = np.invert(image)

    data = []
    mask_files = glob.glob(os.path.join('{}/{}/masks/'.format(prefix,image_id),'*.png'))
    
    index = 0
    for mask_file in mask_files:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask_id = os.path.basename(mask_file).split('.')[0]
        info = extract(mask, image, mask_id, image_id)
        
        data.append(info)
        index +=1 
    
    return(flatten(data))


# In[ ]:


train = df[df['train_or_test'] == 'train'][['id','invert']]


# In[ ]:


prefix = '../input/stage1_train'
data = train.apply(lambda x: get_masks(x[0],prefix,x[1]), axis = 1)
data = flatten(data)
masks = pd.DataFrame(data)
mdf = masks.merge(df[['id','shape','invert']],how='inner', on=None, left_on = 'image_id', right_on='id')


# In[ ]:


print(len(masks),list(masks.columns))


# In[ ]:


mdf = masks.merge(df[['id','shape','invert']],how='inner', on=None, left_on = 'image_id', right_on='id')


# # Processing Steps
# 
# The images below show the original, one of the masks, masked image and extracted square of the mask.
# 

# In[ ]:


def show_process(df):
    
    rows = len(df)
    fix, ax = plt.subplots(rows,6,figsize=(5*6,5*rows))
    for row,data in df.iterrows():

        image_file = '{}/{}/images/{}.png'.format(prefix,data['image_id'],data['image_id'])
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image,cv2.COLOR_BGRA2RGB)
        show_image(ax[row,0],'original {}'.format(data['shape']),image)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
        enhanced = image.copy()
        for channel in range(enhanced.shape[2]):
            enhanced[:,:,channel] = clahe.apply(enhanced[:,:,channel])

        circled = cv2.circle(enhanced, (int(data['cx']),int(data['cy'])), 
                             int(data['radius']+3), 
                             color=(255,255,0), thickness=3) 
        show_image(ax[row,1],'enhanced',circled)

        mask_file = '{}/{}/masks/{}.png'.format(prefix,data['image_id'],data['mask_id'])
        mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
        
        circled = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
        circled = cv2.circle(circled, (int(data['cx']),int(data['cy'])), 
                             int(data['radius']+3), 
                             color=(255,255,0), thickness=3) 

        show_image(ax[row,2],'mask',circled)

        masked_image = enhanced.copy()
        if data['invert']:
            masked_image = np.invert(masked_image)

        for channel in range(masked_image.shape[2]):
            masked_image[:,:,channel] = np.bitwise_and(masked_image[:,:,channel],mask)

        circled = cv2.circle(masked_image, (int(data['cx']),int(data['cy'])), 
                             int(data['radius']+3), 
                             color=(0,255,255), thickness=3) 
        show_image(ax[row,3],'inverted & masked',cv2.cvtColor(circled,cv2.COLOR_BGRA2RGB))

        square = np.array(data['square'], dtype=np.uint8)
        circled = cv2.circle(square, (int(data['offset_cx']),int(data['offset_cy'])), 
                             int(data['radius']+3), 
                             color=(255,255,0), thickness=1) 
        
        show_image(ax[row,4],'Cropped',circled)

        nuclei = np.array(data['nuclei'], dtype=np.uint8)
        show_image(ax[row,5],'rotated and clipped',nuclei)


# The images below show the processing steps followed to extract the final nuclei. There is one example of each sample from the training set. I added a circle on the images to show the original location and to show that the nuclei on edges can be clipped based on which edge they are on.
# 
# * Original image
# * Image after applying contrast stretch
# * A mask for the image
# * After converting dark on bright to bright on dark and applying the mask
# * Extracting a square. the circle shows the original center.
# * Rotating the image so that long edge is horizontal and clipping the boundaries.

# In[ ]:


detail = mdf[mdf['image_id'].isin(sample['id'])]
sample_one = n_of_each(detail,n=1)

show_process(sample_one)


# I had assumed that each mask is one nucleus. I also assumed that the masked sections are completely solid and would not have holes. I wanted to confirm this assumption, but missed out. While looking at the image extraction progression above, I noticed that in some cases the masked section identified was a part of the mask and not the entire object. In these cases, the process I have used has made me detect more than one contour.
# 
# I wonder if creating combining the contours together into a single set will give me a single composite image.

# ### Masks with holes

# In[ ]:


nuclei_by_mask = mdf[['mask_id','image_id']].groupby(['mask_id']).count().reset_index()
nuclei_by_mask = nuclei_by_mask.rename(columns={"image_id": "count"})
nuclei_by_mask = nuclei_by_mask[nuclei_by_mask['count']>1]
nuclei_by_mask = nuclei_by_mask.sort_values(by=['count'], ascending=False).reset_index()

nuclei_by_mask.head()


# Let's have a look at the mask with 14 elements

# In[ ]:


data = mdf[mdf['mask_id'] == nuclei_by_mask.loc[0]['mask_id']].copy().reset_index()

show_process(data)


# It looks like the mask has holes, and this may be intentional. CV2 find contours is identifying the parent contour and the holes as child contours. I think there is a way to identify whether the contour is a child or not. If yes I can exclude the child contours.

# Turns out that cv2 has a mode RETR_EXTERNAL which will exclude all child elements and this solves the problem above. 

# In[ ]:


holes_in_masks = mdf[mdf['mask_id'].isin(nuclei_by_mask['mask_id'].unique())].copy().reset_index()


# In[ ]:


CONTOUR_EXTRACT_MODE = cv2.RETR_EXTERNAL


# In[ ]:


masks_info = os.path.join(OUTPUT_PATH,'masks.json')
data = train.apply(lambda x: get_masks(x[0],prefix,x[1]), axis = 1)
data = flatten(data)
masks = pd.DataFrame(data) 
with open(masks_info, 'w') as outfile:
    json.dump(data, outfile)


# In[ ]:


mdf = masks.merge(df[['id','shape','invert']],how='inner', on=None, left_on = 'image_id', right_on='id')
len(mdf)


# In[ ]:


count_by_mask = mdf[['mask_id','image_id']].groupby(['mask_id']).count().reset_index()
count_by_mask = count_by_mask.rename(columns={"image_id": "count"})

count_by_mask = count_by_mask[count_by_mask['count']>1].copy().reset_index()
count_by_mask


# But there are still 11 images where the number of contours identified is 2. This means that these masks have a break between two sections and this is causing the

# ## Masks with gaps

# In[ ]:


split_masks = mdf[mdf['mask_id'].isin(count_by_mask['mask_id'].unique())].copy().reset_index()

show_process(split_masks)


# In[ ]:


def tag_anomaly(label,df):
    if type(df) == 'pandas.core.series.Series':
        items = df[['image_id','mask_id']].copy()
    else:
        items = df[['image_id','mask_id']].copy().drop_duplicates()
    items['issue'] = label
    
    return(items)


# In[ ]:


anomalies = pd.DataFrame()
anomalies = anomalies.append(tag_anomaly('Holes in mask',holes_in_masks))
anomalies = anomalies.append(tag_anomaly('Split mask',split_masks))


# In[ ]:


agg = mdf[['shape','mask_id']].groupby(['shape']).count()
#agg.columns = agg.columns.droplevel()
agg.plot.barh(stacked=True,figsize=(16,4))
plt.show()


# In[ ]:


len(mdf[mdf['shape']=='1040x1388'])


# The distribution of the number of nuclei identified by different shapes is also quite wide. There are four images sizes where the total number of nuclei identified is greater than 2000. 

# In[ ]:


agg = mdf[['shape','mask_id','image_id']].groupby(['shape','image_id']).count().reset_index()
agg.groupby(['shape'])['mask_id'].agg(['mean','min','max','std']).reset_index()


# In[ ]:


nuclei_sample = n_of_each(mdf,8)


# In[ ]:


len(nuclei_sample)/8


# In[ ]:


show_row_col(nuclei_sample,8,image_col='nuclei',label_col='shape',mode='image')


# My first attempt at extracting the nuclei ended up with quite a lot of faded objects, so I added the contrast stretching to make the nuclei more visible. Since the sample itself is random, every run may not give the same results. I did see some very small objects which to my eye looked like black patches. 

# ## Very Small Nuclei
# 
# Filtering the data set by the resulting height and width less than 10 pixels, I want to look at the smaller objects to verify that the code I wrote is not incorrect and these are actually as identified by Dr Anne & team.

# In[ ]:


len(mdf[mdf['height'] < 3])


# In[ ]:


def show_mask(mdf, mask_id, prefix):
    image_id = mdf[mdf['mask_id']==mask_id]['image_id'].squeeze()
    image_file = '{}/{}/images/{}.png'.format(prefix,image_id,image_id)
    mask_file = '{}/{}/masks/{}.png'.format(prefix,image_id,mask_id) 
    
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGRA2RGB)
    
    mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    
    fig,ax = plt.subplots(1,2,figsize = (20,20))
    ax[0].grid(None)
    ax[0].imshow(image)
    ax[1].set_title('Image')
    ax[1].grid(None)
    ax[1].imshow(mask)
    ax[1].set_title('Mask')


# In[ ]:


mdf.plot.scatter('width','height')
plt.show()


# In[ ]:


fig, axes = plt.subplots(2,1, figsize=(20,20))
sns.violinplot('shape','height', data=mdf, ax = axes[0])
axes[0].set_title('Height Distribution by Shape')

axes[0].yaxis.grid(True)
axes[0].set_xlabel('Shape')
axes[0].set_ylabel('Height')

sns.violinplot('shape','width', data=mdf, ax = axes[1])
axes[1].set_title('Width Distribution by Shape')

axes[1].yaxis.grid(True)
axes[1].set_xlabel('Shape')
axes[1].set_ylabel('Width')

plt.show()


# Overall size distribution of the nuclei
# 
# * The largest resolution 1040x1388 has larger nuclei width averaging around 110.
# * Most of the remaining shapes have average width between 10 & 30
# * One outlier is the 360x360 where there are groups which are generally bigger than 40, but there a re quite a few which are smaller than 20.

# In[ ]:


small = mdf[mdf['width'] < 10]

fig, axes = plt.subplots(2,1, figsize=(20,20))
sns.violinplot('shape','height', data=small, ax = axes[0])
axes[0].set_title('Height Distribution by Shape (Width < 10)')

axes[0].yaxis.grid(True)
axes[0].set_xlabel('Shape')
axes[0].set_ylabel('Height')

sns.violinplot('shape','width', data=small, ax = axes[1])
axes[1].set_title('Width Distribution by Shape')

axes[1].yaxis.grid(True)
axes[1].set_xlabel('Shape')
axes[1].set_ylabel('Width')

plt.show()


# Interesting distribution of the nuclei image sizes. When width is less that 10 pixels, the height also is less than 10, however there are images where height is less than 10 but the width is greater than 10
# 
# * Smaller sized (<5px) images are mostly in the image sizes 256x256, 256x360, and 360x360 
# * Rest of the images have larger overall images with few outliers in the smaller sizes

# In[ ]:


small = mdf[mdf['height'] < 10]
small = small[small['width']> 10]

fig, axes = plt.subplots(2,1, figsize=(20,20))
sns.violinplot('shape','height', data=small, ax = axes[0])
axes[0].set_title('Height Distribution by Shape (Height < 10 & Width > 10)')

axes[0].yaxis.grid(True)
axes[0].set_xlabel('Shape')
axes[0].set_ylabel('Height')

sns.violinplot('shape','width', data=small, ax = axes[1])
axes[1].set_title('Width Distribution by Shape (Height < 10 & Width > 10)')

axes[1].yaxis.grid(True)
axes[1].set_xlabel('Shape')
axes[1].set_ylabel('Width')

plt.show()


# There seem to be some extreme cases where height is less than 10pixels but the width is greater than 30

# In[ ]:


small = mdf[(mdf['height'] < 10) & (mdf['width']> 30)]

fig, axes = plt.subplots(2,1, figsize=(20,20))
sns.violinplot('shape','height', data=small, ax = axes[0])
axes[0].set_title('Height Distribution by Shape (Height < 10 & Width > 30)')

axes[0].yaxis.grid(True)
axes[0].set_xlabel('Shape')
axes[0].set_ylabel('Height')

sns.violinplot('shape','width', data=small, ax = axes[1])
axes[1].set_title('Width Distribution by Shape (Height < 10 & Width > 30)')

axes[1].yaxis.grid(True)
axes[1].set_xlabel('Shape')
axes[1].set_ylabel('Width')

plt.show()


# In[ ]:


small = mdf[mdf['height'] < 10]

g = sns.FacetGrid(small,col="shape", col_wrap=4, size = 4)
g = g.map(plt.scatter, "height", "width")


# In[ ]:


small = mdf[(mdf['height'] < 5) & (mdf['width'] > 20)]

g = sns.FacetGrid(small,col="shape", col_wrap=4, size = 4)
g = g.map(plt.scatter, "height", "width")


# In[ ]:


len(small)


# Ditribution of the nuclei counts identified in images by shape of original image

# In[ ]:


agg = mdf[['image_id','shape','mask_id']].groupby(['shape','image_id']).count().reset_index()
agg = agg.rename(columns={'mask_id':'mask_count'})

g = sns.FacetGrid(agg ,col="shape", size = 4,  sharex="none", col_wrap=3)
g = g.map(plt.hist,'mask_count')


# Overall distribution of the number of nuclei annotated by image.

# In[ ]:


agg = mdf[['image_id','mask_id']].groupby(['image_id']).count().reset_index()
agg = agg.rename(columns={'mask_id':'mask_count'})

fig, ax = plt.subplots(figsize=(20,5))
sns.distplot(agg['mask_count'],ax=ax,rug=True)
plt.show()


# In[ ]:


agg.loc[agg['mask_count']<10,'mask_count'].plot.hist()
plt.show()


# There are very few images with nuclei count less than 5

# In[ ]:


agg[agg['mask_count']<4]


# Processing details where only a single nuclei is annotated.

# In[ ]:


low_count = mdf[mdf['image_id'].isin(agg.loc[agg['mask_count']<2,'image_id'].unique())].copy().reset_index()

show_process(low_count)


# In[ ]:


anomalies = anomalies.append(tag_anomaly('Invalid Masks',low_count.loc[2]))


# The third one in the above set has already been identified an incorrect train file. This is missing annotations. 
# 
# Processing details where only two nuclei are annotated.

# In[ ]:


low_count = mdf[mdf['image_id'].isin(agg.loc[agg['mask_count']==2,'image_id'].unique())].copy().reset_index()

show_process(low_count)


# * Image 1: Is this supposed to be two nuclei?
# * Image 3: Is a middle one missing from the masks?

# In[ ]:


low_count = mdf[mdf['image_id'].isin(agg.loc[agg['mask_count']==3,'image_id'].unique())].copy().reset_index()

show_process(low_count)


# ### Anomalies

# In[ ]:


abnormal = mdf[(mdf['height'] < 4) & 
               (mdf['width'] > 30)].copy().reset_index()
show_process(abnormal)


# All of these look like incorrect annotation.

# In[ ]:


anomalies = anomalies.append(tag_anomaly('Look like lines',abnormal))


# In[ ]:


abnormal = mdf[(mdf['height'] < 6) & 
               (mdf['height'] > 4) & 
               (mdf['width'] > 30)].copy().reset_index()
len(abnormal)
show_process(abnormal)


# In[ ]:


anomalies = anomalies.append(tag_anomaly('Look like lines',abnormal.loc[0]))


# In[ ]:


with open(os.path.join(OUTPUT_PATH,'anomalies.json'), 'w') as outfile:
    json.dump(anomalies.to_json(orient='records'), outfile)


# Maybe it is possible to detect if the extracted nuclei image dimensions are an outlier or not.

# In[ ]:


agg = anomalies[['issue','mask_id']].groupby('issue').count()
agg = agg.rename(columns={'mask_id':'counts'})
agg


# ## Conclusion
# 
# Sometimes mistakes help. Using RETR_TREE was a mistake, but it helped me identify masks that have holes. 
# 
# There seem to be issues in the data. Some have already been identified, some are yet to be identified. 
# 
# There are some missing annotations also as mentioned in this [Thread to post data quality issues](https://www.kaggle.com/c/data-science-bowl-2018/discussion/47770)
# 
# > This might be a case where annotators disagree. For me, if I could see that something was a nucleus, even behind a piece of debris like this, I would annotate it. Thankfully, debris like this is not common.
# 
# > I agree b1eb0123fe2d8c825694b193efb7b923d95effac9558ee4eaf3116374c2c94fe is missing two nuclei off to the right. I'm also disturbed that there are tiny black dots, indicating holes in the nuclei which should not be there. (I cannot swear to it, but in general nuclei as we have annotated them should NOT have any holes) 
# 
# > For 19f0653c33982a416feed56e5d1ce6849fd83314fd19dfa1c5b23c6b66e9868a I guess it depends how you generated the image you are showing on the right. You are right that in general they ought to be multi-colored whereas your result indicates the masks are improperly merged. Just not sure whether that is happening in the annotations or in your code that generates the multicolor visualization (I assume the former of course, in which case it's definitely an error). 
# 
# > For 9bb6e39d5f4415bc7554842ee5d1280403a602f2ba56122b87f453a62d37c06e indeed, that object is an error. I caught many of these before the training set went out but apparently not all! 
# 
# > For 1f0008060150b5b93084ae2e4dabd160ab80a95ce8071a321b80ec4e33b58aca: Those pairs of nuclei should indeed be split. Those are errors.
# > stage1_train/58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921 I agree that nucleus ought to have been identified
# 
# > I agree that image 12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40 was badly annotated.
# > 0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9. Single mask seems to cover two objects
