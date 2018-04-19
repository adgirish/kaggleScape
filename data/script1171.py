
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from pandas import DataFrame
import gc

from IPython.display import Image
from IPython.core.display import HTML

from scipy.sparse import csr_matrix

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[2]:


with open("../input/train.json") as datafile1: #first check if it's a valid json file or not
    train_data = json.load(datafile1)
with open("../input/test.json") as datafile2: #first check if it's a valid json file or not
    test_data = json.load(datafile2)
with open("../input/validation.json") as datafile3: #first check if it's a valid json file or not
    valid_data = json.load(datafile3)


# In[3]:


print("####" * 10)
print("## Training Data.")
print(train_data.keys())
print(train_data["info"])
print(train_data["license"])
print(len(train_data["images"]))
print(len(train_data["annotations"]))
print(train_data["images"][:10])
print(train_data["annotations"][:10])

print("\n\n")
print("####" * 10)
print("## Validation Data.")
print(valid_data.keys())
print(len(valid_data["images"]))
print(len(valid_data["annotations"]))
print(valid_data["images"][:10])
print(valid_data["annotations"][:10])

print("\n\n")
print("####" * 10)
print("## Test Data.")
print(test_data.keys())
print(len(test_data["images"]))
print(test_data["images"][:10])

#print(train_data["images"])


# In[4]:


train_imgs_df = pd.DataFrame.from_records(train_data["images"])
train_imgs_df["url"] = train_imgs_df["url"]
train_labels_df = pd.DataFrame.from_records(train_data["annotations"])
#train_labels_df = train_labels_df["labelId"].apply(lambda x: [int(i) for i in x])
train_df = pd.merge(train_imgs_df,train_labels_df,on="imageId",how="outer")
train_df["imageId"] = train_df["imageId"].astype(np.int)
print(train_df.head())
print(train_df.dtypes)

valid_imgs_df = pd.DataFrame.from_records(valid_data["images"])
valid_imgs_df["url"] = valid_imgs_df["url"]
valid_labels_df = pd.DataFrame.from_records(valid_data["annotations"])
#valid_labels_df = valid_labels_df["labelId"].apply(lambda x: [int(i) for i in x])
valid_df = pd.merge(valid_imgs_df,valid_labels_df,on="imageId",how="outer")
valid_df["imageId"] = valid_df["imageId"].astype(np.int)
print(valid_df.head())
print(valid_df.dtypes)

test_df = pd.DataFrame.from_records(test_data["images"])
test_df["url"] = test_df["url"]
test_df["imageId"] = test_df["imageId"].astype(np.int)
print(test_df.head())
print(test_df.dtypes)


# In[5]:


del train_data
del valid_data
del test_data
gc.collect()


# In[6]:


print("####" * 10)
print("## Training Data.")
print(train_df.isna().any())

print("\n\n")
print("####" * 10)
print("## Validation Data.")
print(valid_df.isna().any())

print("\n\n")
print("####" * 10)
print("## Testing Data.")
print(test_df.isna().any())


# In[7]:


train_image_arr = train_df[["imageId","labelId"]].apply(lambda x: [(x["imageId"],int(i)) for i in x["labelId"]], axis=1).tolist()
train_image_arr = [item for sublist in train_image_arr for item in sublist]
train_image_row = np.array([d[0] for d in train_image_arr]).astype(np.int)
train_image_col = np.array([d[1] for d in train_image_arr]).astype(np.int)
train_image_vals = np.ones(len(train_image_col))
train_image_mat = csr_matrix((train_image_vals, (train_image_row, train_image_col)))
print(train_image_mat.shape)


# In[8]:


labels = train_image_mat.sum(0).astype(np.int)
print(labels)


# In[9]:


## Class distribution.
plt.figure(figsize=(30,18))
labels_inds = np.arange(len(labels.tolist()[0]))
sns.barplot(labels_inds,  labels.tolist()[0])
plt.xlabel('label id', fontsize=6)
plt.ylabel('Count', fontsize=16)
plt.title("Distribution of labels", fontsize=18)


# In[14]:


def display_label(label_id, label_mat, df, num_disp=8):
    data_col = train_image_mat.getcol(label_id)
    tar_col = np.random.choice(np.where(data_col.toarray() == 1.0)[0],size=num_disp).tolist()
    urls = df[df["imageId"].isin(tar_col)]["url"].tolist()
    img_style = "width: 180px; margin: 0px; float: left; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for u in urls])
    header_str = "<h2>Label {:d}</h2>".format(label_id)
    display(HTML(header_str))
    display(HTML(images_list))


# In[ ]:


for label in range(1,train_image_mat.shape[1]):
    display_label(label, train_image_mat, train_df,2)

