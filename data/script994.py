
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm_notebook
import seaborn as sns


# In[ ]:


data = pd.read_json('../input/train.json')
test = pd.read_json('../input/train.json')


# In[ ]:


data['band_1'] = data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
data['band_2'] = data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
test['band_1'] = test['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
test['band_2'] = test['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors='coerce')
test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')

train = data.sample(frac=0.8)
val = data[~data.isin(train)].dropna()


# In[ ]:


train.head()


# ## check NaNs, ignore inc_angle for now

# In[ ]:


train.info()


# ## Class Balance

# In[ ]:


sns.countplot(train['is_iceberg']);


# In[ ]:


def plot_sample(df, idx):
    c = ( 'Not Hotdog', 'Hotdog')
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(df['band_1'].iloc[idx])
    ax2.imshow(df['band_2'].iloc[idx])
    ax3.hist(df['band_1'].iloc[idx].ravel(), bins=256, fc='k', ec='k');
    ax4.hist(df['band_2'].iloc[idx].ravel(), bins=256, fc='k', ec='k');
    f.set_figheight(10)
    f.set_figwidth(10)
    plt.suptitle(c[df['is_iceberg'].iloc[idx]])
    plt.show()


# In[ ]:


plot_sample(train, 764)


# In[ ]:


plot_sample(train, 2)


# ## Concat Bands into (N, 2, 75, 75) images

# In[ ]:


band_1_tr = np.concatenate([im for im in train['band_1']]).reshape(-1, 75, 75)
band_2_tr = np.concatenate([im for im in train['band_2']]).reshape(-1, 75, 75)
full_img_tr = np.stack([band_1_tr, band_2_tr], axis=1)

band_1_val = np.concatenate([im for im in val['band_1']]).reshape(-1, 75, 75)
band_2_val = np.concatenate([im for im in val['band_2']]).reshape(-1, 75, 75)
full_img_val = np.stack([band_1_val, band_2_val], axis=1)

band_1_test = np.concatenate([im for im in test['band_1']]).reshape(-1, 75, 75)
band_2_test = np.concatenate([im for im in test['band_2']]).reshape(-1, 75, 75)
full_img_test = np.stack([band_1_test, band_2_test], axis=1)


# ## Dataset and DataLoader

# In[ ]:


train_imgs = torch.from_numpy(full_img_tr).float()
train_targets = torch.from_numpy(train['is_iceberg'].values).long()
train_dataset = TensorDataset(train_imgs, train_targets)

val_imgs = torch.from_numpy(full_img_val).float()
val_targets = torch.from_numpy(val['is_iceberg'].values).long()
val_dataset = TensorDataset(val_imgs, val_targets)


test_imgs  = torch.from_numpy(full_img_test).float()



# In[ ]:


train_dataset[0]


# ## Define simple model

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.batch = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 18 * 18, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.batch(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[ ]:


net = Net()


# ## Train

# In[ ]:


epochs = 2
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters())


# In[ ]:


# utils 
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=None):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window_size = window_size

    def reset(self):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(y_true, y_pred):
    y_true = y_true.float()
    _, y_pred = torch.max(y_pred, dim=-1)
    return (y_pred.float() == y_true).float().mean()
    
def fit(train, val, epochs, batch_size):
    print('train on {} images validate on {} images'.format(len(train), len(val)))
    net.train()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    for epoch in tqdm_notebook(range(epochs), total=epochs):
        running_loss = AverageMeter()
        running_accuracy = AverageMeter()
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        pbar = tqdm_notebook(train_loader, total=len(train_loader))
        for data, target in pbar:
            data, target = Variable(data), Variable(target)
            output = net(data)
            loss = criterion(output, target)
            acc = accuracy(target.data, output.data)
            running_loss.update(loss.data[0])
            running_accuracy.update(acc)
            pbar.set_description("[ loss: {:.4f} | acc: {:.4f} ] ".format(
                running_loss.avg, running_accuracy.avg))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("[ loss: {:.4f} | acc: {:.4f} ] ".format(running_loss.avg, running_accuracy.avg))
        for val_data, val_target in val_loader:
            val_data, val_target = Variable(val_data), Variable(val_target)
            output = net(val_data)
            val_loss = criterion(output, val_target)
            val_acc = accuracy(val_target.data, output.data)
            val_loss_meter.update(val_loss.data[0])
            val_acc_meter.update(val_acc)
        pbar.set_description("[ loss: {:.4f} | acc: {:.4f} | vloss: {:.4f} | vacc: {:.4f} ] ".format(
        running_loss.avg, running_accuracy.avg, val_loss_meter.avg, val_acc_meter.avg))
        print("[ loss: {:.4f} | acc: {:.4f} | vloss: {:.4f} | vacc: {:.4f} ] ".format(
        running_loss.avg, running_accuracy.avg, val_loss_meter.avg, val_acc_meter.avg))


# In[ ]:


fit(train_dataset, val_dataset, 3, 32)


# In[ ]:


fit(train_dataset, val_dataset, 3, 32)


# ## Overfitting at 4 epochs
