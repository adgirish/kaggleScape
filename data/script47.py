
# coding: utf-8

# Based on tensorflow starter code from https://www.kaggle.com/alexozerin/end-to-end-baseline-tf-estimator-lb-0-72

# In[ ]:


import os
import re
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}
len(id2name)


# In[ ]:


def load_data(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label, label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    
    columns_list = ['label', 'label_id', 'user_id', 'wav_file']
    
    train_df = pd.DataFrame(train, columns = columns_list)
    valid_df = pd.DataFrame(val, columns = columns_list)
    
    return train_df, valid_df


# In[ ]:


train_df, valid_df = load_data('./data/')


# In[ ]:


train_df.head()


# In[ ]:


train_df.label.value_counts()


# In[ ]:


silence_files = train_df[train_df.label == 'silence']
train_df      = train_df[train_df.label != 'silence']


# In[ ]:


from scipy.io import wavfile


# In[ ]:


def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav


# In[ ]:


silence_data = np.concatenate([read_wav_file(x) for x in silence_files.wav_file.values])


# In[ ]:


from scipy.signal import stft


# In[ ]:


def process_wav_file(fname):
    wav = read_wav_file(fname)
    
    L = 16000  # 1 sec
    
    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L)
        wav = wav[i:(i+L)]
    elif len(wav) < L:
        rem_len = L - len(wav)
        i = np.random.randint(0, len(silence_data) - rem_len)
        silence_part = silence_data[i:(i+L)]
        j = np.random.randint(0, rem_len)
        silence_part_left  = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        wav = np.concatenate([silence_part_left, wav, silence_part_right])
    
    specgram = stft(wav, 16000, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
    phase = np.angle(specgram[2]) / np.pi
    amp = np.log1p(np.abs(specgram[2]))
    
    return np.stack([phase, amp], axis = 2)


# In[ ]:


import random
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate, Dense, Dropout
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.utils import to_categorical


# In[ ]:


def train_generator(train_batch_size):
    while True:
        this_train = train_df.groupby('label_id').apply(lambda x: x.sample(n = 2000))
        shuffled_ids = random.sample(range(this_train.shape[0]), this_train.shape[0])
        for start in range(0, len(shuffled_ids), train_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + train_batch_size, len(shuffled_ids))
            i_train_batch = shuffled_ids[start:end]
            for i in i_train_batch:
                x_batch.append(process_wav_file(this_train.wav_file.values[i]))
                y_batch.append(this_train.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield x_batch, y_batch


# In[ ]:


def valid_generator(val_batch_size):
    while True:
        ids = list(range(valid_df.shape[0]))
        for start in range(0, len(ids), val_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + val_batch_size, len(ids))
            i_val_batch = ids[start:end]
            for i in i_val_batch:
                x_batch.append(process_wav_file(valid_df.wav_file.values[i]))
                y_batch.append(valid_df.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield x_batch, y_batch


# In[ ]:


x_in = Input(shape = (257,98,2))
x = BatchNormalization()(x_in)
for i in range(4):
    x = Conv2D(16*(2 ** i), (3,3))(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (1,1))(x)
x_branch_1 = GlobalAveragePooling2D()(x)
x_branch_2 = GlobalMaxPool2D()(x)
x = concatenate([x_branch_1, x_branch_2])
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(POSSIBLE_LABELS), activation = 'soft')(x)
model = Model(inputs = x_in, outputs = x)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


from keras_tqdm import TQDMNotebookCallback
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# In[ ]:


callbacks = [EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=1,
                           min_delta=0.01,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=3,
                               verbose=1,
                               epsilon=0.01,
                               mode='min'),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/starter.hdf5',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min'),
             TQDMNotebookCallback()]


# In[ ]:


history = model.fit_generator(generator=train_generator(64),
                              steps_per_epoch=344,
                              epochs=20,
                              verbose=2,
                              callbacks=callbacks,
                              validation_data=valid_generator(64),
                              validation_steps=int(np.ceil(valid_df.shape[0]/64)))


# In[ ]:


model.load_weights('./weights/starter.hdf5')


# In[ ]:


test_paths = glob(os.path.join('./data/', 'test/audio/*wav'))


# In[ ]:


def test_generator(test_batch_size):
    while True:
        for start in range(0, len(test_paths), test_batch_size):
            x_batch = []
            end = min(start + test_batch_size, len(test_paths))
            this_paths = test_paths[start:end]
            for x in this_paths:
                x_batch.append(process_wav_file(x))
            x_batch = np.array(x_batch)
            yield x_batch


# In[ ]:


predictions = model.predict_generator(test_generator(64), int(np.ceil(len(test_paths)/64)))


# In[ ]:


classes = np.argmax(predictions, axis=1)


# In[ ]:


# last batch will contain padding, so remove duplicates
submission = dict()
for i in range(len(test_paths)):
    fname, label = os.path.basename(test_paths[i]), id2name[classes[i]]
    submission[fname] = label


# In[ ]:


with open('starter_submission.csv', 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))

