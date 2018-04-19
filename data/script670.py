
# coding: utf-8

# 
# 
# ## PyTorch Speech Recognition Challenge
# 
# https://www.kaggle.com/c/tensorflow-speech-recognition-challenge
# 
# 
# Notebooks: <a href="https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/Kaggle-PyTorch/tf/PyTorch%20Speech%20Recognition%20Challenge%20Starter.ipynb"> On GitHub</a>
# 
# 
# #### References:
# 
# - http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# 
# - https://www.bountysource.com/issues/44576966-a-tutorial-on-writing-custom-datasets-samplers-and-using-transforms
# 
# - https://medium.com/towards-data-science/my-first-kaggle-competition-9d56d4773607
# 
# - https://github.com/sohyongsheng/kaggle-planet-forest
# 
# - https://github.com/rwightman/pytorch-planet-amazon/blob/master/dataset.py
# 
# - https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/43624
# 
# ## PyTorch dada sets
# 
# - Convert the audio files into images (spectogram) 
# - Create a CSV file consisting of the good labels 
# - Then write a custom PyTorch data loader
# - Simple CNN
# 
# ## Issues:
# - Problem with the loss function for the multi-class case during training, loss is negative
# 
# #### Shlomo Kashani

# # PyTorch Imports
# 

# In[ ]:


get_ipython().run_line_magic('reset', '-f')
import torch
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
from subprocess import call
# call(["nvcc", "--version"]) does not work
get_ipython().system(' nvcc --version')
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())

print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

import numpy
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

import pandas
import pandas as pd

import logging
handler=logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)
get_ipython().run_line_magic('matplotlib', 'inline')

# !pip install psutil
import psutil
import os
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)

cpuStats()


# In[ ]:


use_cuda = torch.cuda.is_available()
# use_cuda = False

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


# # Setting up global variables
# 
# - Root folder
# - Audio folder
# - Audio Label folder

# In[ ]:


DATA_ROOT ='d:/db/data/tf/'
IMG_PATH = DATA_ROOT + '/picts/train/'
IMG_EXT = '.png'
IMG_DATA_LABELS = DATA_ROOT + '/train_v2.csv'


# # Turn WAV into Images
# - See https://www.kaggle.com/timolee/audio-data-conversion-to-images-eda
# 

# In[ ]:


audio_path = 'd:/db/data/tf/train/audio/'
pict_Path = 'd:/db/data/tf//picts/train/'
test_pict_Path = 'd:/db/data/tf//picts/test/'
test_audio_path = 'd:/db/data/tf//test/audio/'
samples = []


if not os.path.exists(pict_Path):
    os.makedirs(pict_Path)

if not os.path.exists(test_pict_Path):
    os.makedirs(test_pict_Path)
    
subFolderList = []

for x in os.listdir(audio_path):
    if os.path.isdir(audio_path + '/' + x):
        subFolderList.append(x)
        if not os.path.exists(pict_Path + '/' + x):
            os.makedirs(pict_Path +'/'+ x)


# In[ ]:


# #### Function: convert audio to spectogram images

# def wav2img(wav_path, targetdir='', figsize=(4,4)):
#     """
#     takes in wave file path
#     and the fig size. Default 4,4 will make images 288 x 288
#     """
#     fs = 44100 # sampling frequency
    
#     # use soundfile library to read in the wave files
#     test_sound, samplerate = sf.read(wav_path)
    
#     # make the plot
#     fig = plt.figure(figsize=figsize)
#     S, freqs, bins, im = plt.specgram(test_sound, NFFT=1024, Fs=samplerate, noverlap=512)
#     plt.show
#     plt.axis('off')
    
#     ## create output path
#     output_file = wav_path.split('/')[-1].split('.wav')[0]
#     output_file = targetdir +'/'+ output_file
#     plt.savefig('%s.png' % output_file)
#     plt.close()


# def wav2img_waveform(wav_path, targetdir='', figsize=(4,4)):
#     test_sound, samplerate = sf.read(sample_audio[0])
#     fig = plt.figure(figsize=figsize)
#     plt.plot(test_sound)
#     plt.axis('off')
#     output_file = wav_path.split('/')[-1].split('.wav')[0]
#     output_file = targetdir +'/'+ output_file
#     plt.savefig('%s.png' % output_file)
#     plt.close()

# ### Convert Training Audio
# #### Loop through source audio and save as pictures 
# # (may take a while) may also consider running at commandline. 
# # Code is limited to 3 folders and 10 files each, get rid of array limits to process the entire directory

# # c:\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:7221: RuntimeWarning: divide by zero encountered in log10
# #   Z = 10. * np.log10(spec)

# for i, x in enumerate(subFolderList):
#     print(i, ':', x)
#     # get all the wave files
#     all_files = [y for y in os.listdir(audio_path + x) if '.wav' in y]
#     for file in all_files:
#         try:
#             wav2img(audio_path + x + '/' + file, pict_Path + x)                
#         except Exception:
#             pass
            


# # Generate lables into a CSV, which is easier for PyTorch Dataset class

# In[ ]:


# from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# from collections import defaultdict
# d = defaultdict(LabelEncoder)

# # Build the pictures path
# subFolderList = []
# for x in os.listdir(pict_Path):
#     if os.path.isdir(pict_Path + '/' + x):
#         subFolderList.append(x)        
            
# good_labels=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
# POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()

# # print (type(POSSIBLE_LABELS))
# # print (type(good_labels))
# columns = ['img', 'label-str','fullpath']
# df_pred=pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)
# # df_pred.id.astype(int)

# for i, x in enumerate(subFolderList):
#     if (x in POSSIBLE_LABELS):
#     #     print(i, ':', x)
#         # get all the wave files
#         all_files = [y for y in os.listdir(pict_Path + x) if '.png' in y]
#         for file in all_files:
#     #         print (audio_path + x + '/' + file, pict_Path + x)
#             fullPath=pict_Path + x + '/' + file
#     #         print (fullPath)
#             df_pred = df_pred.append({'img':file, 'label-str':x,'fullpath':fullPath},ignore_index=True)
#     #         print (pict_Path + x)    
    

# # Encode the categorical labels as numeric data
# df_pred['label'] = LabelEncoder().fit_transform(df_pred['label-str'])
# # Make sure we dont save the header
# df_pred.to_csv(IMG_DATA_LABELS, columns=('img','label-str','fullpath', 'label'), index=None, header=False)
# df_pred.to_csv(IMG_DATA_LABELS +'_header', columns=('img','label-str','fullpath', 'label'), index=None, header=True)

# # img,label,fullpath
# # 00176480_nohash_0.wav,down,d:/db/data/tf/train/audio/down/00176480_nohash_0.wav
    
# df_pred.head(3)


# # The Torch Dataset Class

# In[ ]:


import time
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from collections import defaultdict
d = defaultdict(LabelEncoder)

# Encoding the variable
# X_df_train_SINGLE = X_df_train_SINGLE.apply(lambda x: d[x.name].fit_transform(x))
# X_df_train_SINGLE=X_df_train_SINGLE.apply(LabelEncoder().fit_transform)
# Inverse the encoded
# fit.apply(lambda x: X_df_train_SINGLE[x.name].inverse_transform(x))
# Using the dictionary to label future data
# df.apply(lambda x: X_df_train_SINGLE[x.name].transform(x))
# answers_1_SINGLE = list (X_df_train_SINGLE[singleResponseVariable].values)
# answers_1_SINGLE= map(int, answers_1_SINGLE)

def encode_onehot(df, cols):  
    vec = DictVectorizer()    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

try:
    from PIL import Image
except ImportError:
    import Image
    
class GenericImageDataset(Dataset):    

    def __init__(self, csv_path, img_path, img_ext, transform=None):
        
        t = time.time()        
        lgr.info('CSV path {}'.format(csv_path))
        lgr.info('IMG path {}'.format(img_path))        
        
        assert img_ext in ['.png']
        
        tmp_df = pd.read_csv(csv_path, header=None) # img,label,fullpath
                        
        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        # Encoding the variables                
        lgr.info("DF CSV:\n" + str (tmp_df.head(3)))
                        
        self.X_train = tmp_df[2]        
        
        self.y_train = self.mlb.fit_transform(tmp_df[1].str.split()).astype(np.float32)           
        self.y_train=self.y_train.reshape((self.y_train.shape[0]*10,1)) # Must be reshaped for PyTorch!                
        
#         y_df = encode_onehot(tmp_df, cols=[tmp_df[1]])
#         self.y_train = y_df 
        
        lgr.info('y_train {}'.format(self.y_train))
                
#         self.y_train = tmp_df[3].astype(np.float32)                          
#         self.y_train = self.mlb.fit_transform(tmp_df[1].str.split()).astype(np.float32)
#         self.y_train = tmp_df[3].astype(np.float32)       
#         d = defaultdict(LabelEncoder)
#         self.y_train =tmp_df[1].apply(lambda x: d[x].fit_transform(x))
    
#         tmp_df=one_hot(tmp_df,tmp_df[1])
#         self.y_train = tmp_df[1].astype(np.float32)       
#         encoder = LabelEncoder()
#         encoder.fit(tmp_df[1])
#         self.y_train = encoder.transform(tmp_df[1]).astype(np.float32)
#         self.y_train=self.y_train.reshape((self.y_train.shape[0],1)) # Must be reshaped for PyTorch!
                
        lgr.info('[*]Dataset loading time {}'.format(time.time() - t))
        lgr.info('[*] Data size is {}'.format(len(self)))
        
        lgr.info("DF CSV:\n" + str (tmp_df.head(5)))
        
        print ()

    def __getitem__(self, index):
#         lgr.info ("__getitem__:" + str(index))
        path=self.img_path + self.X_train[index]
        path=self.X_train[index]
#         lgr.info (" --- get item path:" + path)
        img = Image.open(path)
        img = img.convert('RGB')
        if self.transform is not None: # TypeError: batch must contain tensors, numbers, or lists; 
                                     #found <class 'PIL.Image.Image'>
            img = self.transform(img)
#             print (str (type(img))) # <class 'torch.FloatTensor'>                
#         label = torch.from_numpy(self.y_train[index])
        label = (self.y_train[index])
        return img, label

    def __len__(self):
        l=len(self.X_train.index)
#         lgr.info ("Lenght:" +str(l))
        return (l)       

    @staticmethod        
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    @staticmethod    
    def flaotTensorToImage(img, mean=0, std=1):
        """convert a tensor to an image"""
        img = np.transpose(img.numpy(), (1, 2, 0))
        img = (img*std+ mean)*255
        img = img.astype(np.uint8)    
        return img    
    
    @staticmethod
    def toTensor(img):
        """convert a numpy array of shape HWC to CHW tensor"""
        img = img.transpose((2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(img).float()
        return tensor/255.0    


# # The Torch transforms.ToTensor() methood
# 
# - Converts: a PIL.Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

# In[ ]:


transformations = transforms.Compose([transforms.ToTensor()])


# # The Torch DataLoader Class
# 
# - Will load our GenericImageDataset
# - Can be regarded as a list (or iterator, technically). 
# - Each time it is invoked will provide a minibatch of (img, label) pairs.

# In[ ]:


dset_train = GenericImageDataset(IMG_DATA_LABELS,IMG_PATH,IMG_EXT,transformations)


# # Train Validation Split
# 
# - Since there is no train_test_split method in PyTorch, we have to split a Training dataset into training and validation sets.

# In[ ]:


batch_size = 16 # on GTX 1080
global_epoches = 10
LR = 0.0005
MOMENTUM = 0.95
validationRatio=0.11    

class FullTrainningDataset(torch.utils.data.Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds)>=offset+length, Exception("Parent Dataset not long enough")
        super(FullTrainningDataset, self).__init__()
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        return self.full_ds[i+self.offset]
    


def trainTestSplit(dataset, val_share=validationRatio):
    val_offset = int(len(dataset)*(1-val_share))
    print("Offest:" + str(val_offset))
    return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset, val_offset, len(dataset)-val_offset)

 
train_ds, val_ds = trainTestSplit(dset_train)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

print(train_loader)
print(val_loader)


# # Test the DataLoader Class

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

imagesToShow=4

for i, data in enumerate(train_loader, 0):
    lgr.info('i=%d: '%(i))            
    images, labels = data            
    num = len(images)
    
    ax = plt.subplot(1, imagesToShow, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    
    for n in range(num):
        image=images[n]
        label=labels[n]
        plt.imshow (GenericImageDataset.flaotTensorToImage(image))
        
    if i==imagesToShow-1:
        break    


# # The NN model
# 
# - We will use a several CNNs with conv(3x3) -> bn -> relu -> pool(4x4) -> fc.
# 
# - In PyTorch, a model is defined by a subclass of nn.Module. It has two methods:
# 
# - `__init__: constructor. Create layers here. Note that we don't define the connections between layers in this function.`
# 
# 
# - `forward(x): forward function. Receives an input variable x. Returns a output variable. Note that we actually connect the layers here dynamically.` 

# In[ ]:


dropout = torch.nn.Dropout(p=0.30)
class ConvRes(nn.Module):
    def __init__(self, insize, outsize):
        super(ConvRes, self).__init__()
        drate = .3
        self.math = nn.Sequential(
            nn.BatchNorm2d(insize),
            # nn.Dropout(drate),
            torch.nn.Conv2d(insize, outsize, kernel_size=2, padding=2),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.math(x)


class ConvCNN(nn.Module):
    def __init__(self, insize, outsize, kernel_size=7, padding=2, pool=2, avg=True):
        super(ConvCNN, self).__init__()
        self.avg = avg
        self.math = torch.nn.Sequential(
            torch.nn.Conv2d(insize, outsize, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(outsize),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(pool, pool),
        )
        self.avgpool = torch.nn.AvgPool2d(pool, pool)

    def forward(self, x):
        x = self.math(x)
        if self.avg is True:
            x = self.avgpool(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.cnn1 = ConvCNN(3, 32, kernel_size=7, pool=4, avg=False)
        self.cnn2 = ConvCNN(32, 32, kernel_size=5, pool=2, avg=True)
        self.cnn3 = ConvCNN(32, 32, kernel_size=5, pool=2, avg=True)

        self.res1 = ConvRes(32, 64)

        self.features = nn.Sequential(
            self.cnn1, dropout,
            self.cnn2,
            self.cnn3,
            self.res1,
        )

        self.classifier = torch.nn.Sequential(
            nn.Linear(3136, 1),
        )
        self.sig = nn.Sigmoid()
  
    def forward(self, x):
        x = self.features(x)
#         print (x.data.shape)
        x = x.view(x.size(0), -1)
#         print (x.data.shape)
        x = self.classifier(x)
#         print (x.data.shape)
        x = self.sig(x)
        return x

    
if use_cuda:
    lgr.info ("Using the GPU")
    model = Net().cuda() # On GPU
else:
    lgr.info ("Using the CPU")
    model = Net() # On CPU

lgr.info('Model {}'.format(model))



# #  Loss and Optimizer
# 
# - Select a loss function and the optimization algorithm.

# In[ ]:


loss_func=torch.nn.BCELoss()
loss_func = nn.MultiLabelSoftMarginLoss()
# loss_func = torch.nn.CrossEntropyLoss()
# NN params
LR = 0.005
MOMENTUM= 0.9
optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=5e-5) #  L2 regularization
if use_cuda:
    lgr.info ("Using the GPU")    
    model.cuda()
    loss_func.cuda()

lgr.info (optimizer)
lgr.info (loss_func)


# # Start training in Batches
# 
# See example here:
# http://codegists.com/snippet/python/pytorch_mnistpy_kernelmode_python
# 
# https://github.com/pytorch/examples/blob/53f25e0d0e2710878449900e1e61d31d34b63a9d/mnist/main.py

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
 

    
clf=model 
opt= optimizer
loss_history = []
acc_history = []
 
def train(epoch):
    clf.train() # set model in training mode (need this because of dropout)
     
    # dataset API gives us pythonic batching 
    for batch_idx, (data, target) in enumerate(train_loader):
        
        if use_cuda:
            data, target = Variable(data.cuda(async=True)), Variable(target.cuda(async=True)) # On GPU                
        else:            
            data, target = Variable(data), Variable(target) # RuntimeError: expected CPU tensor (got CUDA tensor)                           
                 
        # forward pass, calculate loss and backprop!
        opt.zero_grad()
        preds = clf(data)
        if use_cuda:
            loss = loss_func(preds, target).cuda()
#             loss = F.log_softmax(preds).cuda() # TypeError: log_softmax() takes exactly 1 argument (2 given)
#             loss = F.nll_loss(preds, target).cuda() # https://github.com/torch/cutorch/issues/227
            
        else:
            loss = loss_func(preds, target)
#             loss = F.log_softmax(preds)
#             loss = F.nll_loss(preds, target.long()) # RuntimeError: multi-target not supported at /pytorch/torch/lib/THNN/generic/ClassNLLCriterion.c:22
        loss.backward()
        
        opt.step()
        
        
        if batch_idx % 100 == 0:
            loss_history.append(loss.data[0])
            lgr.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))              

            
start_time = time.time()    

for epoch in range(1, 5):
    print("Epoch %d" % epoch)
    train(epoch)    
end_time = time.time()
print ('{} {:6.3f} seconds'.format('GPU:', end_time-start_time))
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.show()


# In[ ]:


criterion = loss_func
all_losses = []
val_losses = []


if __name__ == '__main__':

    for epoch in range(global_epoches):
        print('Epoch {}'.format(epoch + 1))
        print('*' * 5 + ':')
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
    
            img, label = data
            if use_cuda:
                img, label = Variable(img.cuda(async=True)), Variable(label.cuda(async=True))  # On GPU
            else:
                img, label = Variable(img), Variable(
                    label)  # RuntimeError: expected CPU tensor (got CUDA tensor)
    
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.data[0] * label.size(0)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if i % 100 == 0:
                all_losses.append(running_loss / (batch_size * i))
                print('[{}/{}] Loss: {:.6f}'.format(
                    epoch + 1, global_epoches, running_loss / (batch_size * i),
                    running_acc / (batch_size * i)))
                
    
#                 loss_cost = loss.data[0]                                
#                 # RuntimeError: can't convert CUDA tensor to numpy (it doesn't support GPU arrays). 
#                 # Use .cpu() to move the tensor to host memory first.        
#                 prediction = (model(img).data).float() # probabilities         
#         #         prediction = (net(X_tensor).data > 0.5).float() # zero or one
#         #         print ("Pred:" + str (prediction)) # Pred:Variable containing: 0 or 1
#         #         pred_y = prediction.data.numpy().squeeze()            
#                 pred_y = prediction.cpu().numpy().squeeze()
#                 target_y = label.cpu().data.numpy()

#                 tu = (log_loss(target_y, pred_y),roc_auc_score(target_y,pred_y ))
#                 print ('LOG_LOSS={}, ROC_AUC={} '.format(*tu))  
        
    
        print('Finish {} epoch, Loss: {:.6f}'.format(epoch + 1, running_loss / (len(train_ds))))
    
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for data in val_loader:
            img, label = data
    
            if use_cuda:
                img, label = Variable(img.cuda(async=True), volatile=True),Variable(label.cuda(async=True), volatile=True)  # On GPU
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
    
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)
    
        print('VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_ds))))
        val_losses.append(eval_loss / (len(val_ds)))
        print()
    


# In[ ]:


get_ipython().run_cell_magic('bash', '', "jupyter nbconvert \\\n    --to=slides \\\n    --reveal-prefix=https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.2.0/ \\\n    --output=py09.html \\\n    './09 PyTorch Kaggle Image Data-set loading with CNN'")

