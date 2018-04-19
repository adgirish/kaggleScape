
# coding: utf-8

# In[1]:


import os
from pathlib import Path
import IPython.display as ipd

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[2]:


print(check_output(["ls", "../input/train"]).decode("utf8"))

folders = os.listdir("../input/train/audio")
print(folders)


# In[3]:


train_audio_path = '../input/train/audio'

train_labels = os.listdir(train_audio_path)
train_labels.remove('_background_noise_')
print(f'Number of labels: {len(train_labels)}')

labels_to_keep = ['yes', 'no', 'up', 'down', 'left',
                  'right', 'on', 'off', 'stop', 'go', 'silence']

train_file_labels = dict()
for label in train_labels:
    files = os.listdir(train_audio_path + '/' + label)
    for f in files:
        train_file_labels[label + '/' + f] = label

train = pd.DataFrame.from_dict(train_file_labels, orient='index')
train = train.reset_index(drop=False)
train = train.rename(columns={'index': 'file', 0: 'folder'})
train = train[['folder', 'file']]
train = train.sort_values('file')
train = train.reset_index(drop=True)
print(train.shape)

def remove_label_from_file(label, fname):
    return fname[len(label)+1:]

train['file'] = train.apply(lambda x: remove_label_from_file(*x), axis=1)
train['label'] = train['folder'].apply(lambda x: x if x in labels_to_keep else 'unknown')


# In[4]:


sample_rate, samples = wavfile.read(str(train_audio_path) + '/house/61e50f62_nohash_1.wav')
frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)


# In[5]:


sample_rate


# In[7]:


frequencies


# In[8]:


times


# In[9]:


fig = plt.figure(figsize = (10,10))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Spectogram - House')
ax1.imshow(spectogram)


# In[10]:


sample_rate, samples = wavfile.read(str(train_audio_path) + '/eight/25132942_nohash_2.wav')
frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)
fig = plt.figure(figsize = (10,10))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Spectogram - Eight')
ax1.imshow(spectogram)


# In[11]:


sample_rate, samples = wavfile.read(str(train_audio_path) + '/happy/43f57297_nohash_0.wav')
frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)
fig = plt.figure(figsize = (10,10))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Spectogram - Happy')
ax1.imshow(spectogram)


# In[12]:


sample_rate, samples = wavfile.read(str(train_audio_path) + '/three/19e246ad_nohash_0.wav')
frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)
fig = plt.figure(figsize = (10,10))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.set_title('Spectogram - Three')
ax1.imshow(spectogram)

