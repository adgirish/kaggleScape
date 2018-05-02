
# coding: utf-8

# In[ ]:


import os
from pathlib import Path
from subprocess import check_output

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Check data folders:

# In[ ]:


print(check_output(["ls", "../input"]))
print(check_output(["ls", "../input/train"]).decode("utf8"))
folders = os.listdir("../input/train/audio")
print(folders)


# Load labels and wav file paths into dataframe:

# In[ ]:


train_audio_path = '../input/train/audio'
train_labels = os.listdir(train_audio_path)
print(f'Number of labels: {len(train_labels)}')

wavs = []
labels = []
for label in train_labels:
    if label == '_background_noise_':
        continue
    files = os.listdir(train_audio_path + '/' + label)
    for f in files:
        if not f.endswith('wav'):
            continue
        wavs.append(f)
        labels.append(label)

train = pd.DataFrame({'file':wavs,'label':labels})
train.info()


# ## Labels
# Explore label frequencies

# In[ ]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.countplot(ax=ax, x="label", data=train)
print(train.label.unique())


# ## Spectrograms

# In[ ]:


def spectrogram(file, label):
    eps=1e-10
    sample_rate, samples = wavfile.read(str(train_audio_path) + '/' + label + '/' + file)
    frequencies, times, spectrogram = signal.stft(samples, sample_rate, nperseg = sample_rate/50, noverlap = sample_rate/75)
    return np.log(np.abs(spectrogram).T+eps)


# Explore the spectrograms for each label:

# In[ ]:


num_samples = 5
labels = train.label.unique()
fig, axes = plt.subplots(len(labels),num_samples, figsize = (16, len(labels)*4))
for i,label in enumerate(labels):
    files = train[train.label==label].file.sample(num_samples)
    axes[i][0].set_title(label)
    for j, file in enumerate(files):
        specgram = spectrogram(file, label)
        axes[i][j].axis('off')
        axes[i][j].matshow(specgram)

