
# coding: utf-8

# # Starting kit for PyTorch Deep Learning
# 
# Welcome to this tutorial to get started on PyTorch for this competition.
# PyTorch is a promising port of Facebook's Torch to Python.
# 
# It's only 3 months old but has an already promising feature set.
# Unfortunately it's very very raw, and I had a lot of troubles to get started with very basic things:
# - data loading
# - building a basic CNN
# - training
# 
# Hopefully this will help you getting started using PyTorch on this dataset.

# ## Importing libraries
# Please note that we do not import numpy but PyTorch wrapper for Numpy

# In[ ]:


import pandas as pd
from torch import np # Torch wrapper for Numpy

import os
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.preprocessing import MultiLabelBinarizer


# ## Setting up global variables

# In[ ]:


IMG_PATH = '../input/train-jpg/'
IMG_EXT = '.jpg'
TRAIN_DATA = '../input/train_v2.csv'


# ## Loading the data - first part - DataSet
# 
# This is probably the most obscure part of PyTorch. Most examples use well known datasets (MNIST ...) and have a custom loader or forces you to have a specific folder structure similar to this:
# 
# * data
#     * train
#           * dogs
#           * cats
#      * validation
#           * dogs
#           * cats
#      * test
#           * test
# 
# Data loading in PyTorch is in 2 parts
# 
# First the data must be wrapped in a __Dataset__ class with a getitem method that from an index return X_train[index] and y_train[index] and a length method. A Dataset is basically a data storage.
# 
# The following solution loads the image name from a CSV and file path + extension and can be adapted easily for most Kaggle challenges. You won't have to write your own ;).
# 
# The code will:
# 
# - Check that all images in CSV exist in the folder
# - Use ScikitLearn MultiLabelBinarizer to OneHotEncode the labels, mlb.inverse_transform(predictions) can be used to get back the textual labels from the predictions
# - Apply PIL transformations to the images. See [here](http://pytorch.org/docs/torchvision/transforms.html) for the supported list.
# - Use ToTensor() to convert from an image with color scale 0-255 to a Tensor with color scale 0-1.
# 
# Note: We use PIL instead of OpenCV because it's Torch default image loader and is compatible with `ToTensor()` method. An fast loader called accimage is currently in development and was published 3 days ago [here](https://github.com/pytorch/accimage).
# 
# Note 2: This only provides a mapping to the data, **the data is not loaded in memory at this point**. The next part will show you how to load only what is needed for the batch in memory. This is a huge advantage compared to kernels that must load all images at once.

# In[ ]:


class KaggleAmazonDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
    
        tmp_df = pd.read_csv(csv_path)
        assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), "Some images referenced in the CSV file were not found"
        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)


# In[ ]:


transformations = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])

dset_train = KaggleAmazonDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)


# ## Loading the data - second part - DataLoader
# 
# As was said, loading the data is in 2 parts, we provided PyTorch with a data storage, and we have to tell it how to load it. This is done with __DataLoader__
# 
# The DataLoader defines how you retrieve the images + labels from the dataset. You can tell it to:
# 
# * Set the batch size.
# * Shuffle and sample the data randomly, hence implementing __train_test_split__ (check SubsetRandomSampler [here](http://pytorch.org/docs/data.html?highlight=sampler))
# * Improve performance by loading data via  separate thread `num_worker` and using `pin_memory` for CUDA. Documentation [here](http://pytorch.org/docs/notes/cuda.html?highlight=dataloader).

# In[ ]:


train_loader = DataLoader(dset_train,
                          batch_size=256,
                          shuffle=True,
                          num_workers=4 # 1 for CUDA
                         # pin_memory=True # CUDA only
                         )


# ## Creating your Neural Network
# 
# This is tricky, you need  to compute yourself the in_channels and out_channels of your filters hence the 2304 input for the Dense layer. The first input 3 corresponds to the number of channels of your image, the 17 output corresponds to the number of target labels.

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)

model = Net() # On CPU
# model = Net().cuda() # On GPU


# ## Defining your training function

# In[ ]:


optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# In[ ]:


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


# ## Training your model

# In[ ]:


for epoch in range(1, 2):
    train(epoch)


# # Thank you for your attention
# 
# Hopefully that will help you get started. I still have a lot to figure out in PyTorch like:
# 
# * Implementing the train / validation split
# * Figure out data augmentation (and not just random transformations or images)
# * Implementing early stopping
# * Automating computation of intermediate layers
# * Improving the display of each epochs
# 
# If you liked the kernel don't forget to vote and don't hesitate to comment.

# ## Full code
# 
# I have published my full code of the competition in my [GitHub](https://github.com/mratsim/Amazon_Forest_Computer_Vision). (Note: I only worked on it in the first 2 weeks, so it probably lacks the latest findings)
# 
# You will find:
#   - [A script that output the mean and stddev of your image if you want to train from scratch](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/compute-mean-std.py#L28)
# 
#   - [Using weighted loss function](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/main_pytorch.py#L61)
# 
#   - [Logging your experiment](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/main_pytorch.py#L89)
# 
#   - [Composing data augmentations](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/main_pytorch.py#L103), also [here](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p_data_augmentation.py#L181).
# Note use [Pillow-SIMD](https://python-pillow.org/pillow-perf/) instead of PIL/Pillow. It is even faster than OpenCV
# 
#   - [Loading from a CSV that contains image path - 61 lines yeah](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p2_dataload.py#L23)
# 
#   - [Equivalent in Keras - 216 lines ugh](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/k_dataloader.py). Note: so much lines were needed because by default in Keras you either have the data augmentation with ImageDataGenerator or lazy loading of images with "flow_from_directory" and there is no flow_from_csv
# 
#   - [Model finetuning with custom PyCaffe weights](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p_neuro.py#L139)
# 
#   - Train_test_split, [PyTorch version](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p_model_selection.py#L4) and [Keras version](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/k_model_selection.py#L4)
# 
# - [Weighted sampling training so that the model view rare cases more often](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/main_pytorch.py#L131-L140)
# 
#  - [Custom Sampler creation, example for the balanced sampler](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p_sampler.py)
# 
#  - [Saving snapshots each epoch](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/main_pytorch.py#L171)
# 
#  - [Loading the best snapshot for prediction](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/pytorch_predict_only.py#L83)
# 
#  - [Failed word embeddings experiments](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/Embedding-RNN-Autoencoder.ipynb) to [combine image and text data](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/Dual_Feed_Image_Label.ipynb)
# 
#  - [Combined weighted loss function (softmax for unique weather tags, BCE for multilabel tags)](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p2_loss.py#L36)
# 
#  - [Selecting the best F2-threshold](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p2_metrics.py#L38) via stochastic search at the end of each epoch to [maximize validation score](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/526128239a6abcbb32fbf5b34ed8cc7a3cd87c4e/src/p2_validation.py#L49). This is then saved along model parameter.
# 
#   - [CNN-RNN combination (work in progress)](https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p3_neuroRNN.py#L10)
