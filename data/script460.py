
# coding: utf-8

# This notebook showcases a fast way to read data from BSON into a generator for Keras.  
# The idea is strongly inspired from https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson  
# 
# Since I don't have a SSD, the original Generator is ~3s per batch , I just added the feature to read chunks of file and shuffle into batch to improve speed.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import bson


# In[3]:


import cv2


# In[4]:


from tqdm import *
import struct


# # Load meta data

# From https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson

# In[5]:


def read_bson(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df


# In[6]:


TRAIN_BSON_FILE = '../input/train.bson'


# In[7]:


meta_data = read_bson(TRAIN_BSON_FILE, 7069896, with_categories=True)


# # Test read performance

# The read performance on HDD is orders pr magnitude higher when random read vs contiguous read

# In[8]:


def get_obs(file, offset, length):
    file.seek(offset)
    return bson.BSON.decode(file.read(length))


# In[9]:


file = open(TRAIN_BSON_FILE, 'rb')


# ## Contiguous read

# In[10]:


get_ipython().run_cell_magic('timeit', '', "i = np.random.choice(meta_data.shape[0], size=1)[0]\nsample = meta_data.iloc[i:i+256]\nres = []\nfor _id, row in sample.iterrows():\n    obs = get_obs(file, row['offset'], row['length'])\n    assert _id == obs['_id'] ")


# ## Random read

# In[11]:


get_ipython().run_cell_magic('timeit', '', "sample = meta_data.sample(256)\nfor _id, row in sample.iterrows():\n    obs = get_obs(file, row['offset'], row['length'])\n    assert _id == obs['_id']")


# It takes ~600 ms here but on my hdd it is 3.15 s per loop. With more preprocessing of the batch, my GPU would be starving all the time

# ## Semi contiguous read

# Here we can simulate getting a chunk and read only a sample from it. Since the offset are nearby in the file and sorted, it is still quite fast

# In[12]:


get_ipython().run_cell_magic('timeit', '', "i = np.random.choice(meta_data.shape[0], size=1)[0]\nsample = meta_data.iloc[i:i+10000].sample(256).sort_index()\nres = []\nfor _id, row in sample.iterrows():\n    obs = get_obs(file, row['offset'], row['length'])\n    assert _id == obs['_id'] ")


# ## Semi contiguous read wo  sort

# In[13]:


get_ipython().run_cell_magic('timeit', '', "i = np.random.choice(meta_data.shape[0], size=1)[0]\nsample = meta_data.iloc[i:i+10000].sample(256)\nres = []\nfor _id, row in sample.iterrows():\n    obs = get_obs(file, row['offset'], row['length'])\n    assert _id == obs['_id'] ")


# # Generator

# In[14]:


from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.applications import inception_v3


# In[15]:


from threading import Lock


# ** Let's define 2 mode of read: contiguous read for test (faster) and block random read for train (slower but not too much) **

# In[16]:


def contiguous_read(bson_file):
    while True:
        iter_file = bson.decode_file_iter(open(bson_file, 'rb'))
        for obs in iter_file:
            yield obs


# The idea is to split file into chunks of rather small size and read chunk in shuffled order

# In[17]:


def block_reader(bson_file, meta_data, chunk_size=1000, shuffle=False):
    assert isinstance(meta_data, pd.DataFrame)
    assert len(meta_data.columns.intersection(['offset', 'length'])) == 2
    # prepare metadata
    n_obs = meta_data.shape[0]
    meta_data_sorted = meta_data.sort_values('offset', ascending=True)
    meta_chunks = np.array_split(meta_data_sorted, 
                                 np.ceil(n_obs/chunk_size))
    open_file = open(bson_file, 'rb')
    while True:
        # Generate chunks order
        chunk_indexes = np.arange(len(meta_chunks))
        if shuffle:
            chunk_indexes = np.random.permutation(chunk_indexes)
        # Iterate over chunks
        for ind in chunk_indexes:
            chunk = meta_chunks[ind]
            for _id, _meta in chunk.iterrows():
                open_file.seek(_meta['offset'])
                obs = bson.BSON.decode(open_file.read(_meta['length']))
                yield obs


# Chunk can still note shuffled inside, so we need a batching mecanism with some cache to shuffle more (inspired by tensorflow Iterator)

# In[18]:


def batch(generator, batch_size, shuffle=False, cache_size=10000):
    gen_stopped = False
    cached_data = []
    while True:        
        # Fill up cache
        while (len(cached_data) < cache_size) & (gen_stopped == False):
            try:
                cached_data.append(next(generator))
            except StopIteration:
                gen_stopped = True
                break
        # Stop if there is nothing left
        if len(cached_data) == 0:
            return
        # Generate batch       
        batch_data = []
        bsize = min(batch_size, len(cached_data))
        if shuffle:
            inds = np.random.choice(len(cached_data), size=(bsize,), replace=False)
        else:
            inds = np.arange(bsize)
        
        for i in sorted(inds, reverse=True):
            try:
                batch_data.append(cached_data.pop(i))            
            except IndexError:
                print(i, bsize, len(cached_data))
        yield batch_data


# Some functions to process data

# In[19]:


def get_img(obs, img_size=180, keep=None):
    if keep is None:
        keep = np.random.choice(len(obs['imgs']))
    else:
        keep = 0
    byte_str = obs['imgs'][keep]['picture']
    img = cv2.imdecode(np.fromstring(byte_str, dtype=np.uint8), 
                       cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size,img_size))
    return img


# In[20]:


def preprocess_batch(batch_data, labels=None, weights=None, img_size=180):
    batch_size = len(batch_data)
    X = np.zeros(shape=(batch_size, img_size, img_size, 3), dtype=np.float32)
    y = np.zeros(shape=(batch_size,), dtype=np.float32)
    w = np.ones(shape=(batch_size,), dtype=np.float32)
    for ind, obs in enumerate(batch_data):
        _id = obs['_id']
        X[ind] = get_img(obs, img_size=img_size)
        if labels is not None:
            y[ind] = labels[_id]
        if weights is not None:
            w[ind] = weights[_id]
    X = inception_v3.preprocess_input(X)
    return X, y, w


# ** Let's wrap it up in a Iterator with lock so we can do multithreading later**

# In[21]:


class BSONIterator(Iterator):
    def __init__(self, bson_file, batch_size=32, preprocess_batch_func=None, metadata=None,
                 shuffle=False, chunk_size=1000, shuffle_cache=100000):
        if shuffle:
            self.obs_generator = block_reader(bson_file, metadata, chunk_size=chunk_size, shuffle=True)
        else:
            self.obs_generator = contiguous_read(bson_file)
        self.batch_generator = batch(self.obs_generator, batch_size=batch_size, 
                                     shuffle=shuffle, cache_size=shuffle_cache)
        self.preprocess_batch_func = preprocess_batch_func
        self.lock = Lock()
        
    def next(self):
        with self.lock:
            batch_data = next(self.batch_generator)
        if self.preprocess_batch_func is None:
            return batch_data
        else:
            return self.preprocess_batch_func(batch_data)
        
    def __next__(self):
        return self.next()


# ## Test

# In[22]:


from tqdm import *


# ## contiguous read
# It is recommended to create a separate BSON file with validation data so you can read the whole file

# In[23]:


gen = contiguous_read(TRAIN_BSON_FILE)
for _ in tqdm(range(10000)):
    obs = next(gen)


# ## Block read without shuffle

# In[24]:


gen = block_reader(TRAIN_BSON_FILE, meta_data, chunk_size=1000, shuffle=False)
next(gen) #warm up
for _ in tqdm(range(10000)):
    obs = next(gen)


# ## Block read with shuffle
# The performance hit is hard but it is still much better pure random read

# In[25]:


gen = block_reader(TRAIN_BSON_FILE, meta_data.sample(100000), chunk_size=1000, shuffle=True)
next(gen) #warm up
for _ in tqdm(range(10000)):
    obs = next(gen)


# ## Let's try some more realistic settings

# ** Testing settings: reading the whole file **

# In[26]:


gen = BSONIterator(TRAIN_BSON_FILE, batch_size=256)
next(gen) #warm up
for _ in tqdm(range(1000)):
    batch_data = next(gen)


# ** Training settings: random read**

# In[28]:


# Let's simulate case where train is 90% of total obs
train_meta = meta_data.sample(frac=0.9, replace=False)


# In[29]:


gen = BSONIterator(TRAIN_BSON_FILE, batch_size=256, shuffle=True, metadata=train_meta)
next(gen) #warm up
for _ in tqdm(range(1000)):
    batch_data = next(gen)


# In[30]:


import functools


# ** Add preprocessing ** 

# In[31]:


gen = BSONIterator(TRAIN_BSON_FILE, batch_size=256, shuffle=True, metadata=train_meta, 
                   preprocess_batch_func=functools.partial(preprocess_batch, 
                                                           img_size=128, 
                                                           labels=meta_data.category_id))
next(gen) #warm up
for _ in tqdm(range(100)):
    batch_data = next(gen)


# ** Simulate (poorly) multithreading**

# In[32]:


from sklearn.externals.joblib import Parallel, delayed


# In[33]:


gen = BSONIterator(TRAIN_BSON_FILE, batch_size=256, shuffle=True, metadata=train_meta, 
                   preprocess_batch_func=functools.partial(preprocess_batch, 
                                                           img_size=128, 
                                                           labels=meta_data.category_id))
next(gen) #warm up
_ = Parallel(n_jobs=4, backend='threading', verbose=1)(delayed(next)(gen) 
                                                   for _ in tqdm(range(100)))


# 5 batches/s should be enough especially most of the time my GPU can only hold batchsize of 128. 

# ** DISCLAIMER ** : I did not actually test the shuffling capability but the BSON file seem already shuffled so it should do the job.  
# One can tune the chunk_size smaller and cache_size higher for more shuffling vs cost of memory and/or speed
# 

# ** Hope some might find this helpful. Enjoy GPU feeding! **
