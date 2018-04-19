
# coding: utf-8

# # What this kernel is about:
# There are visualisations of diferent image types present in train & test datasets, code is mainly from https://www.kaggle.com/mpware/stage1-eda-microscope-image-types-clustering
# Main impact  of this kernel is creating a mosaic from train and test data. 
# Skip to part 5 to see complited mosaics. 
# UPD. In the comment section you can find csv with: original img id, cluster, big picture id to use on your own

# ## 1. Imports

# In[1]:


# Import necessary modules and set global constants and variables. 
      
import pandas as pd                 
import numpy as np                                       
from sklearn.cluster import KMeans
from scipy.ndimage.morphology import binary_fill_holes
import cv2                         # To read and manipulate images
import os                          # For filepath, directory handling
import sys                         # System-specific parameters and functions
import tqdm                        # Use smart progress meter
import seaborn as sns              # For pairplots
import matplotlib.pyplot as plt    # Python 2D plotting library
import matplotlib.cm as cm         # Color map
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Global constants.
TRAIN_DIR = '../input/data-science-bowl-2018/stage1_train'
TEST_DIR = '../input/data-science-bowl-2018/stage1_test'
IMG_DIR_NAME = 'images'   # Folder name including the image
MASK_DIR_NAME = 'masks'   # Folder name including the masks
    

# Display working/train/test directories.
print('TRAIN_DIR = {}'.format(TRAIN_DIR))
print('TEST_DIR = {}'.format(TEST_DIR))


# ## 2. Functions

# In[3]:


# Collection of methods for data operations. Implemented are functions to read  
# images/masks from files and to read basic properties of the train/test data sets.

def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None,space='bgr'):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size: 
        img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)
    if space == 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img

def read_train_data_properties(train_dir, img_dir_name, mask_dir_name):
    """Read basic properties of training images and masks"""
    tmp = []
    for i,dir_name in enumerate(next(os.walk(train_dir))[1]):

        img_dir = os.path.join(train_dir, dir_name, img_dir_name)
        mask_dir = os.path.join(train_dir, dir_name, mask_dir_name) 
        num_masks = len(next(os.walk(mask_dir))[2])
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(train_dir,dir_name,FULL_MASK_DIR_NAME,img_name_id+'_mask.png')
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0]/img_shape[1], img_shape[2], num_masks,
                    img_path, mask_dir,mask_path])

    train_df = pd.DataFrame(tmp, columns = ['img_id', 'img_height', 'img_width',
                                            'img_ratio', 'num_channels', 
                                            'num_masks', 'image_path', 'mask_dir','mask_path'])
    return train_df


def read_test_data_properties(test_dir, img_dir_name):
    """Read basic properties of test images."""
    tmp = []
    for i,dir_name in enumerate(next(os.walk(test_dir))[1]):

        img_dir = os.path.join(test_dir, dir_name, img_dir_name)
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0]/img_shape[1], img_shape[2], img_path])

    test_df = pd.DataFrame(tmp, columns = ['img_id', 'img_height', 'img_width',
                                           'img_ratio', 'num_channels', 'image_path'])
    return test_df

def load_raw_data(image_size=(256, 256), space = 'bgr',load_mask=True):
    """Load raw data."""
    # Python lists to store the training images/masks and test images.
    x_train, y_train, x_test = [],[],[]

    # Read and resize train images/masks. 
    print('Loading and resizing train images and masks ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(train_df['image_path']), total=len(train_df)):
        img = read_image(train_df['image_path'].loc[i], target_size=image_size,space = space)
        if load_mask:
            mask = read_image(train_df['mask_path'].loc[i],
                              color_mode=cv2.IMREAD_GRAYSCALE,
                              target_size=image_size)
            #mask = read_mask(train_df['mask_dir'].loc[i], target_size=image_size)
            y_train.append(mask)
        x_train.append(img)
        
    # Read and resize test images. 
    print('Loading and resizing test images ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(test_df['image_path']), total=len(test_df)):
        img = read_image(test_df['image_path'].loc[i], target_size=image_size,space=space)
        x_test.append(img)

    # Transform lists into 4-dim numpy arrays.
    x_train = np.array(x_train)
    #if load_mask:
    y_train = np.array(y_train)
    #y_train = np.expand_dims(np.array(y_train), axis=4)
    x_test = np.array(x_test)
    print('Data loaded')
    if load_mask:
        return x_train, y_train, x_test
    else:
        return x_train, x_test

def get_domimant_colors(img, top_colors=1):
    """Return dominant image color"""
    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters = top_colors)
    clt.fit(img_l)
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return clt.cluster_centers_, hist

def cluster_images_by_hsv():
    """Clusterization based on hsv colors. Adds 'hsv_cluster' column to tables"""
    print('Loading data')
    x_train_hsv,x_test_hsv = load_raw_data(image_size=None,space='hsv',load_mask=False)
    x_hsv = np.concatenate([x_train_hsv,x_test_hsv])
    print('Calculating dominant hsv for each image')
    dominant_hsv = []
    for img in tqdm.tqdm(x_hsv):
        res1, res2 = get_domimant_colors(img,top_colors=1)
        dominant_hsv.append(res1.squeeze())
    print('Calculating clusters')
    kmeans = KMeans(n_clusters=3).fit(dominant_hsv)
    train_df['HSV_CLUSTER'] = kmeans.predict(dominant_hsv[:len(x_train_hsv)])
    test_df['HSV_CLUSTER'] = kmeans.predict(dominant_hsv[len(x_train_hsv):])
    print('Images clustered')
    return None

def plot_images(selected_images_df,images_rows=4,images_cols=8,plot_figsize=4):
    """Plot image_rows*image_cols of selected images. Used to visualy check clusterization"""
    f, axarr = plt.subplots(images_rows,images_cols,figsize=(plot_figsize*images_cols,images_rows*plot_figsize))
    for row in range(images_rows):
        for col in range(images_cols):
            if (row*images_cols + col) < selected_images_df.shape[0]:
                image_path = selected_images_df['image_path'].iloc[row*images_cols + col]
            else:
                continue
            img = read_image(image_path)
            height, width, l = img.shape
            ax = axarr[row,col]
            ax.axis('off')
            ax.set_title("%dx%d"%(width, height))
            ax.imshow(img)


# In[4]:


# Basic properties of images/masks. 
# train_df = read_train_data_properties(TRAIN_DIR, IMG_DIR_NAME, MASK_DIR_NAME)
# test_df = read_test_data_properties(TEST_DIR, IMG_DIR_NAME)
# cluster_images_by_hsv()
# train_df.to_csv('./train_df.csv',index=False)
# test_df.to_csv('./test_df.csv',index=False)

# We don't need to compute everything (especially clusters) every time. simly load them
train_df = pd.read_csv('../input/test-train-df/train_df.csv')
test_df = pd.read_csv('../input/test-train-df/test_df.csv')

# we need to change filepath from my filesystem to kaggle filesystem
train_change_filepath = lambda x: '../input/data-science-bowl-2018/stage1_train/{0}/images/{0}.png'.format(x.split('/')[-1][:-4])
test_change_filepath = lambda x: '../input/data-science-bowl-2018/stage1_test/{0}/images/{0}.png'.format(x.split('/')[-1][:-4])
train_df.image_path = train_df.image_path.map(train_change_filepath)
train_df.drop(['mask_dir','mask_path'],inplace=True,axis = 1)
test_df.image_path = test_df.image_path.map(test_change_filepath)


# In[5]:


train_df.head()


# 
# ## 3. Train & Test clusters visualization

# #### Train data

# In[6]:


for idx in range(3):
    print("Images in cluster {}: {}".format(idx,train_df[train_df['HSV_CLUSTER'] == idx].shape[0]))


# In[7]:


plot_images(train_df[train_df['HSV_CLUSTER'] == 0],2,4)


# In[8]:


plot_images(train_df[train_df['HSV_CLUSTER'] == 1],2,4)


# In[9]:


plot_images(train_df[train_df['HSV_CLUSTER'] == 2],2,4)


# #### Test data

# In[10]:


for idx in range(3):
    print("Images in cluster {}: {}".format(idx,test_df[test_df['HSV_CLUSTER'] == idx].shape[0]))


# In[11]:


plot_images(test_df[test_df['HSV_CLUSTER'] == 0],2,4)


# In[12]:


plot_images(test_df[test_df['HSV_CLUSTER'] == 2],2,4)


# In[13]:


plot_images(test_df[test_df['HSV_CLUSTER'] == 1],2,4)


# -----
# ## 4. Load & Preprocess data

# In[14]:


# Read images/masks from files and resize them. Each image and mask 
# is stored as a 3-dim array where the number of channels is 3 and 1, respectively.
x_train, x_test = load_raw_data(load_mask=False,image_size=None)


# In[15]:


x_train.shape


# ## 5. Mosaic hypotesis
# Lets try to make big images from 4 small images

# In[16]:


from sklearn.neighbors import NearestNeighbors
# nn == Nearest Neighbors in the comments


# In[17]:


def combine_images(data,indexes):
    """ Combines img from data using indexes as follows:
        0 1
        2 3 
    """
    up = np.hstack([data[indexes[0]],data[indexes[1]]])
    down = np.hstack([data[indexes[2]],data[indexes[3]]])
    full = np.vstack([up,down])
    return full

def make_mosaic(data,return_connectivity = False, plot_images = False,external_df = None):
    """Find images with simular borders and combine them to one big image"""
    if external_df is not None:
        external_df['mosaic_idx'] = np.nan
        external_df['mosaic_position'] = np.nan
        # print(external_df.head())
    
    # extract borders from images
    borders = []
    for x in data:
        borders.extend([x[0,:,:].flatten(),x[-1,:,:].flatten(),
                        x[:,0,:].flatten(),x[:,-1,:].flatten()])
    borders = np.array(borders)

    # prepare df with all data
    lens = np.array([len(border) for border in borders])
    img_idx = list(range(len(data)))*4
    img_idx.sort()
    position = ['up','down','left','right']*len(data)
    nn = [None]*len(position)
    df = pd.DataFrame(data=np.vstack([img_idx,position,borders,lens,nn]).T,
                      columns=['img_idx','position','border','len','nn'])
    uniq_lens = df['len'].unique()
    
    for idx,l in enumerate(uniq_lens):
        # fit NN on borders of certain size with 1 neighbor
        nn = NearestNeighbors(n_neighbors=1).fit(np.stack(df[df.len == l]['border'].values))
        distances, neighbors = nn.kneighbors()
        real_neighbor = np.array([None]*len(neighbors))
        distances, neighbors = distances.flatten(),neighbors.flatten()

        # if many borders are close to one, we want to take only the closest
        uniq_neighbors = np.unique(neighbors)

        # difficult to understand but works :c
        for un_n in uniq_neighbors:
            # min distance for borders with same nn
            min_index = list(distances).index(distances[neighbors == un_n].min())
            # check that min is double-sided
            double_sided = distances[neighbors[min_index]] == distances[neighbors == un_n].min()
            if double_sided and distances[neighbors[min_index]] < 1000:
                real_neighbor[min_index] = neighbors[min_index]
                real_neighbor[neighbors[min_index]] = min_index
        indexes = df[df.len == l].index
        for idx2,r_n in enumerate(real_neighbor):
            if r_n is not None:
                df['nn'].iloc[indexes[idx2]] = indexes[r_n]
    
    # img connectivity graph. 
    img_connectivity = {}
    for img in df.img_idx.unique():
        slc = df[df['img_idx'] == img]
        img_nn = {}

        # get near images_id & position
        for nn_border,position in zip(slc[slc['nn'].notnull()]['nn'],
                                      slc[slc['nn'].notnull()]['position']):

            # filter obvious errors when we try to connect bottom of one image to bottom of another
            # my hypotesis is that images were simply cut, without rotation
            if position == df.iloc[nn_border]['position']:
                continue
            img_nn[position] = df.iloc[nn_border]['img_idx']
        img_connectivity[img] = img_nn

    imgs = []
    indexes = set()
    mosaic_idx = 0
    
    # errors in connectivity are filtered 
    good_img_connectivity = {}
    for k,v in img_connectivity.items():
        if v.get('down') is not None:
            if v.get('right') is not None:
                # need down right image
                # check if both right and down image are connected to the same image in the down right corner
                if (img_connectivity[v['right']].get('down') is not None) and img_connectivity[v['down']].get('right') is not None:
                    if img_connectivity[v['right']]['down'] == img_connectivity[v['down']]['right']:
                        v['down_right'] = img_connectivity[v['right']]['down']
                        temp_indexes = [k,v['right'],v['down'],v['down_right']]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        # надо тут фильтровать что они не одинаковые
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data,temp_indexes))
                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left','up_right','down_left','down_right']
                            mosaic_idx += 1
                        continue
            if v.get('left') is not None:
                # need down left image
                if img_connectivity[v['left']].get('down') is not None and img_connectivity[v['down']].get('left') is not None:
                    if img_connectivity[v['left']]['down'] == img_connectivity[v['down']]['left']:
                        v['down_left'] = img_connectivity[v['left']]['down']
                        temp_indexes = [v['left'],k,v['down_left'],v['down']]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data,temp_indexes))
                        
                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left','up_right','down_left','down_right']
                            
                            mosaic_idx += 1 
                        continue
        if v.get('up') is not None:
            if v.get('right') is not None:
                # need up right image
                if img_connectivity[v['right']].get('up') is not None and img_connectivity[v['up']].get('right') is not None:
                    if img_connectivity[v['right']]['up'] == img_connectivity[v['up']]['right']:
                        v['up_right'] = img_connectivity[v['right']]['up']
                        temp_indexes = [v['up'],v['up_right'],k,v['right']]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data,temp_indexes))
                        
                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left','up_right','down_left','down_right']
                            
                            mosaic_idx += 1 
                        continue
            if v.get('left') is not None:
                # need up left image
                if img_connectivity[v['left']].get('up') is not None and img_connectivity[v['up']].get('left') is not None:
                    if img_connectivity[v['left']]['up'] == img_connectivity[v['up']]['left']:
                        v['up_left'] = img_connectivity[v['left']]['up']
                        temp_indexes = [v['up_left'],v['up'],v['left'],k]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data,temp_indexes))
                        
                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left','up_right','down_left','down_right']
                            
                            mosaic_idx += 1 
                        continue

    # same images are present 4 times (one for every piece) so we need to filter them
    print('Images before filtering: {}'.format(np.shape(imgs)))
    
    # can use np. unique only on images of one size, flatten first, then select
    flattened = np.array([i.flatten() for i in imgs])
    uniq_lens = np.unique([i.shape for i in flattened])
    filtered_imgs = []
    for un_l in uniq_lens:
        filtered_imgs.extend(np.unique(np.array([i for i in imgs if i.flatten().shape == un_l]),axis=0))
        
    filtered_imgs = np.array(filtered_imgs)
    print('Images after filtering: {}'.format(np.shape(filtered_imgs)))
    
    if return_connectivity:
        print(good_img_connectivity)
    
    if plot_images:
        for i in filtered_imgs:
            plt.imshow(i)
            plt.show()
            
    # list of not combined images. return if you need
    not_combined = list(set(range(len(data))) - indexes)
    
    if external_df is not None:
        #un_mos_id = external_df[external_df.mosaic_idx.notnull()].mosaic_idx.unique()
        #mos_dict = {k:v for k,v in zip(un_mos_id,range(len(un_mos_id)))}
        #external_df.mosaic_idx = external_df.mosaic_idx.map(mos_dict)
        ## print(temp.mosaic_idx.shape[0])
        ## print(len(temp.mosaic_idx[temp.mosaic_idx.isnull()] ))
        ## print(len(list(range(temp.mosaic_idx.shape[0]-len(temp.mosaic_idx[temp.mosaic_idx.isnull()]),
        ##                     temp.mosaic_idx.shape[0]))))
        external_df.loc[external_df[external_df['mosaic_idx'].isnull()].index,'mosaic_idx'] = range(
            int(np.nanmax(external_df.mosaic_idx.unique())) + 1,
            int(np.nanmax(external_df.mosaic_idx.unique())) + 1 + len(external_df.mosaic_idx[external_df.mosaic_idx.isnull()]))
        external_df['mosaic_idx'] = external_df['mosaic_idx'].astype(np.int32)
        if return_connectivity:
            return filtered_imgs, external_df, good_img_connectivity
        else:
            return filtered_imgs, external_df
    if return_connectivity:
        return filtered_imgs,good_img_connectivity
    else:
        return filtered_imgs


# In[18]:


make_mosaic(x_test,return_connectivity=True,plot_images=True);


# In[19]:


make_mosaic(x_train,return_connectivity=False,plot_images=True);


# In[20]:


## This is how connectivity graph look like
make_mosaic(x_test,return_connectivity=True,plot_images=False);


# In[24]:


# code which makes csv with clusters and mosaic ids for test data
imgs, data_frame = make_mosaic(x_test,return_connectivity=False,plot_images=False,external_df=test_df);
data_frame[['img_id','HSV_CLUSTER','mosaic_idx','mosaic_position']].head(20)

