
# coding: utf-8

# ### Tested: 
# ######## the resulting file is 85GB , run time on 8 cores: 53 minutes
# <img src="https://sites.google.com/site/sgdysregulation/img/imag.png" >

# ### Multiprocessing imporvement inspired by 
# [StackOverflow:Understanding Multiprocessing: Shared Memory Management, Locks and Queues in Python](https://stackoverflow.com/questions/20742637/understanding-multiprocessing-shared-memory-management-locks-and-queues-in-pyt)

# In[ ]:


import pandas as pd
import numpy as np
import bson
import h5py

import os


import multiprocessing as mp


import cv2 #opencv helpful for storing image as array
import itertools #helps in parallel processing

import scipy as scp

import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import base64


# In[ ]:


REC_SIZE = 7069896
path = '../input/'
kaggle =True
# path='./'
# kaggle = False
get_ipython().system('ls "$path"')


# #### seprerate lock and queue used for reading/writing

# In[ ]:


def process_batch(args):
    """
    INPUT: args where args[0] is a list of tuples of size 4
    (itr,CHUNK_SIZE ,in_file,out_file)
    reads a batch of data from the BSON file "in_file" 
    puts the part of the input itr into a read queue.
    converts the images to base64
    write the images to dataset "imgs" in h5 file out_file
    writes the columns [category_id, _id, img_num] 
    corresponding to the imgs into dataset "meta" in h5 file out_file
    finally puts the part of the input itr into a write queue.
    """
    t0 = time.time()
    
    itr,CHUNK_SIZE ,in_file,out_file= args[0]
    if not os.path.exists(in_file):
        return
    print('Processing Batch {} , Batch size: {}'.format(itr,CHUNK_SIZE))
    
    lock_r = args[1]
    queue_r = args[2]
    lock_w = args[3] 
    queue_w = args[4] 
    
    lock_r.acquire()
    with open(in_file,'rb') as b:
        iterator = bson.decode_file_iter(b)
        df = pd.DataFrame(list(itertools.islice(iterator,
                                itr*CHUNK_SIZE,
                                (itr+1)*CHUNK_SIZE))).set_index(['category_id','_id'])
    lock_r.release()
    queue_r.put(itr)
    
    df = df['imgs'].apply(pd.Series).stack().apply(
        lambda x: base64.b64encode(x['picture']))
    data = df.index.to_frame().reset_index(drop=True),np.vstack(df.values)
    df = None
    lock_w.acquire()
    try:
        with h5py.File(out_file) as hdf:
            if 'imgs' not in hdf.keys():
                dt = h5py.special_dtype(vlen=bytes)
                dset = hdf.create_dataset('imgs', shape= (data[1].shape),
                                          maxshape=(None, data[1].shape[1]), chunks=True,
                                          compression="lzf",dtype=dt)
                iset = hdf.create_dataset(name = 'meta', shape= data[0].shape,
                                          maxshape=(None,data[0].shape[1]),
                                          chunks=True,
                                          dtype=np.int64,compression="lzf")
            else:
                dset = hdf['imgs']
                dset.resize((dset.shape[0]+data[1].shape[0],1)) 
                iset = hdf['meta']
                iset.resize((iset.shape[0]+data[0].shape[0],iset.shape[1])) 
            dset[-data[1].shape[0]:,...] = data [1]
            iset[-data[0].shape[0]:,...] = data [0]
            hdf.close()
    except Exception as e:
        print('write failed',e)
    data = None
    lock_w.release()
    t1= time.time()-t0
    print('Batch {} processing Done! Time: {:} mins, {:.2f}secs'.format(itr, t1//60,t1%60))
    queue_w.put(itr)


# In[ ]:



def read_queue(queue):
    """Turns a qeue into a normal python list."""
    results = []
    while not queue.empty():
        result = queue.get()
        results.append(result)
    return results


# In[ ]:


def make_iterator(args, lock_r, queue_r,lock_w, queue_w):
    """Makes an iterator over args and passes the lock an queue to each element."""
    return ((arg, lock_r, queue_r,lock_w, queue_w) for arg in args)


# In[ ]:


def start_processing(in_file,
                     out_file, 
                     CHUNK_SIZE,
                     EP,
                     SP,
                     ncores=4):
    """Starts the manager
    
    :param in_file the BSON file with byte images
    :param out_file the HDF file with base64 images 
    in "image" dataset and meta data in "meta" dataset
    :param CHUNK_SIZE the batch size per process
    :param  EP the end postion (last batch)
    :param  SP the start postion (first batch)
    :param  ncores the number of CPU cores to use
    """
    
    args = list(zip(range(EP-1,SP-1,-1),
                    [CHUNK_SIZE]*(EP-SP),
                    [in_file]*(EP-SP),
                    [out_file]*(EP-SP)))

    result =  manager(process_batch, args, ncores)
    return result


# In[ ]:


def manager(jobfunc, args, ncores):
    """Runs a pool of processes WITH a Manager for the lock and queue.

    """
    mypool = mp.Pool(ncores)
    lock_r = mp.Manager().Lock()
    queue_r = mp.Manager().Queue()
    lock_w = mp.Manager().Lock()
    queue_w = mp.Manager().Queue()
    iterator = make_iterator(args, lock_r, queue_r,lock_w, queue_w)
    mypool.map(jobfunc, iterator)
    mypool.close()
    mypool.join()

    return read_queue(queue_r),read_queue(queue_w)


# In[ ]:


"""Run """

in_file= '{}train.bson'.format(path)
out_file= 'train.h5'
CHUNK_SIZE = 2**17
#default values
EP = 1+(REC_SIZE//CHUNK_SIZE)
SP = 0
if kaggle:
    get_ipython().system('rm "$out_file"')
    CHUNK_SIZE = 2**8
    EP = 1+(REC_SIZE//CHUNK_SIZE)
    SP = EP-16
print(SP,':',EP)

t0 = time.time()
try:
    res = start_processing(in_file = in_file,
                     out_file = out_file,
                     CHUNK_SIZE = CHUNK_SIZE,
                     EP = EP,
                    SP = SP,ncores = min(mp.cpu_count(),8))
except Exception as e:
    print('Failed:',e)
t1= time.time()-t0
print('-'*40)
print('Done! Total Processing Time: {:} mins, {:.2f}secs'.format(t1//60,t1%60))


# ### Retrieve files

# In[ ]:


def read_h5(file,num_of_rec,frRec=0):
    """
    retrieves certain number of records `num_od_rec` from the file `file`
    records start at `frRec` 
    """
    cols =['category_id','_id','img_num']
    with h5py.File(file) as hdf:

        data = hdf['imgs'][frRec:num_of_rec]
        index = hdf['meta'][frRec:num_of_rec]
        hdf.close()
    df = pd.DataFrame(index,columns= cols)
    df['imgs'] = data
    df['imgs'] = df['imgs'].apply(
        lambda bStr: cv2.imdecode(
                            np.fromstring(
                                base64.b64decode(bStr),
                                dtype='uint8'),
                            cv2.IMREAD_COLOR))
    return df.set_index(cols)
df = read_h5(out_file,10)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


n=np.random.randint(0,len(df))
img= df.iloc[n,-1]
imgVSH = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)[...,::-1] #change back to RGB

fig,axs = plt.subplots(2,2,figsize=(16,8))
axs = axs.flatten()

titles = ['Intensity(Value)','Saturation','Hue']
cmaps = ['gray',mpl.cm.GnBu,mpl.cm.GnBu]
for i,ax in enumerate(axs[1:]):
    ax.imshow(imgVSH[...,i],cmap=cmaps[i])
    ax.set_title(titles[i])
    ax.axis('off')
axs[0].imshow(img[...,::-1])
axs[0].set_title('Original')
axs[0].axis('off');

plt.tight_layout();
plt.show();


# In[ ]:


# TODO show how to process data on disk

