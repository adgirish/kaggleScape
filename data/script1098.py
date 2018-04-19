
# coding: utf-8

# # WavCeption V1: just a 1-D Inception approach 

# I just wanted to share a little toy I have been playing with and gave me **amazing results**. As I currently don't have time, I would like to share it to see how people plays with it :-D. The **WavCeption V1** network seems to produce impressive results compared to a regular convolutional neural network, but in this competition it seems that there is a hard-work on the pre-processing and unknown tracks management. It is based on the Google's inception network, the same idea. 
# 
#  I wrote some weeks ago a module implementing it so that it is easy to build an 1D-inception network by connecting lots of these modules in cascade (as you will see below).
#  
#  Unfortunately and due to several Kaggle constraints, it won't run in the kernel machine, so I encourage you to download it and run it in your own machine.
#  
#  By running the model for 12h without struggling too much I achieved 0.76 in the leaderboard (with 0.84 in local test). Some other trials in the same line gave me 0.89 in local, so there is a huge improvement in how you deal with the unknown clips :-D

# ## Load modules and libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd 
import os
import shutil
import glob
import random
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import IPython
from numpy.fft import rfft, irfft
import numpy as np
import random
import itertools

from scipy.io import wavfile
import IPython.display as ipd
import matplotlib.pyplot as plt
import scipy as sp
import tensorflow as tf


# ## Noise generation functions 

# The code in this section has been borrowed and adapted from: 
# https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/generator.py

# In[ ]:


def ms(x):
    """Mean value of signal `x` squared.
    :param x: Dynamic quantity.
    :returns: Mean squared of `x`.
    """
    return (np.abs(x)**2.0).mean()

def normalize(y, x=None):
    """normalize power in y to a (standard normal) white noise signal.
    Optionally normalize to power in signal `x`.
    #The mean power of a Gaussian with :math:`\\mu=0` and :math:`\\sigma=1` is 1.
    """
    #return y * np.sqrt( (np.abs(x)**2.0).mean() / (np.abs(y)**2.0).mean() )
    if x is not None:
        x = ms(x)
    else:
        x = 1.0
    return y * np.sqrt( x / ms(y) )
    #return y * np.sqrt( 1.0 / (np.abs(y)**2.0).mean() )

def white_noise(N, state=None):
    state = np.random.RandomState() if state is None else state
    return state.randn(N)

def pink_noise(N, state=None):

    state = np.random.RandomState() if state is None else state
    uneven = N%2
    X = state.randn(N//2+1+uneven) + 1j * state.randn(N//2+1+uneven)
    S = np.sqrt(np.arange(len(X))+1.) # +1 to avoid divide by zero
    y = (irfft(X/S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)

def blue_noise(N, state=None):
    """
    Blue noise. 
    
    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    
    Power increases with 6 dB per octave.
    Power density increases with 3 dB per octave. 
    
    """
    state = np.random.RandomState() if state is None else state
    uneven = N%2
    X = state.randn(N//2+1+uneven) + 1j * state.randn(N//2+1+uneven)
    S = np.sqrt(np.arange(len(X)))# Filter
    y = (irfft(X*S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)

def brown_noise(N, state=None):
    """
    Violet noise.
    
    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    
    Power decreases with -3 dB per octave.
    Power density decreases with 6 dB per octave. 
    """
    state = np.random.RandomState() if state is None else state
    uneven = N%2
    X = state.randn(N//2+1+uneven) + 1j * state.randn(N//2+1+uneven)
    S = (np.arange(len(X))+1)# Filter
    y = (irfft(X/S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)

def violet_noise(N, state=None):
    """
    Violet noise. Power increases with 6 dB per octave. 
    
    :param N: Amount of samples.
    :param state: State of PRNG.
    :type state: :class:`np.random.RandomState`
    
    Power increases with +9 dB per octave.
    Power density increases with +6 dB per octave. 
    
    """
    state = np.random.RandomState() if state is None else state
    uneven = N%2
    X = state.randn(N//2+1+uneven) + 1j * state.randn(N//2+1+uneven)
    S = (np.arange(len(X)))# Filter
    y = (irfft(X*S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


# ## Tensorflow utilities

# Utilities to modularize tensorflow common actions

# In[ ]:


# Tf Utils
def get_tensorflow_configuration(device="0", memory_fraction=1):
    """
    Function for selecting the GPU to use and the amount of memory the process is allowed to use
    :param device: which device should be used (str)
    :param memory_fraction: which proportion of memory must be allocated (float)
    :return: config to be passed to the session (tf object)
    """
    device = str(device)
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
    config.gpu_options.visible_device_list = device
    return(config)


def start_tensorflow_session(device="0", memory_fraction=1):
    """
    Starts a tensorflow session taking care of what GPU device is going to be used and
    which is the fraction of memory that is going to be pre-allocated.
    :device: string with the device number (str)
    :memory_fraction: fraction of memory that is going to be pre-allocated in the specified
    device (float [0, 1])
    :return: configured tf.Session
    """
    return(tf.Session(config=get_tensorflow_configuration(device=device, memory_fraction=memory_fraction)))


def get_summary_writer(session, logs_path, project_id, version_id):
    """
    For Tensorboard reporting
    :param session: opened tensorflow session (tf.Session)
    :param logs_path: path where tensorboard is looking for logs (str)
    :param project_id: name of the project for reporting purposes (str)
    :param version_id: name of the version for reporting purposes (str)
    :return summary_writer: the tensorboard writer
    """
    path = os.path.join(logs_path,"{}_{}".format(project_id, version_id)) 
    if os.path.exists(path):
        shutil.rmtree(path)
    summary_writer = tf.summary.FileWriter(path, graph_def=session.graph_def)
    return(summary_writer)


# ## Paths management module

# Modules to deal with the paths

# In[ ]:


# Common paths
def _norm_path(path):
    """
    Decorator function intended for using it to normalize a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """
    def normalize_path(*args, **kwargs):
        return os.path.normpath(path(*args, **kwargs))
    return normalize_path


def _assure_path_exists(path):
    """
    Decorator function intended for checking the existence of a the output of a path retrieval function. Useful for
    fixing the slash/backslash windows cases.
    """
    def assure_exists(*args, **kwargs):
        p=path(*args, **kwargs)
        assert os.path.exists(p), "the following path does not exist: '{}'".format(p)
        return p
    return assure_exists


def _is_output_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an output path retrieval
    function
    """
    @_norm_path
    @_assure_path_exists
    def check_existence_or_create_it(*args, **kwargs):
        if not os.path.exists(path(*args, **kwargs)):
            "Path does not exist... creating it: {}".format(path(*args, **kwargs))
            os.makedirs(path(*args, **kwargs))
        return path(*args, **kwargs)
    return check_existence_or_create_it


def _is_input_path(path):
    """
    Decorator function intended for grouping the functions which are applied over the output of an input path retrieval
    function
    """
    @_norm_path
    @_assure_path_exists
    def check_existence(*args, **kwargs):
        return path(*args, **kwargs)
    return check_existence

@_is_input_path
def get_train_path():
    path = "../input/train"
    return path

@_is_input_path
def get_test_path():
    path = "../input/test"
    return path

@_is_input_path
def get_train_audio_path():
    path = os.path.join(get_train_path(), "audio")
    return path

@_is_input_path
def get_scoring_audio_path():
    path = os.path.join(get_test_path(), "audio")
    return path

@_is_output_path
def get_submissions_path():
    path = "../working/output"
    return path

@_is_output_path
def get_silence_path():
    path = "../working/silence"
    return path


# ## Utilities

# Common general-purpose utilities

# In[ ]:


# Utilities
flatten = lambda l: [item for sublist in l for item in sublist]

def batching(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


# ## Data Tools

# Data handling tools

# In[ ]:


# Data tools
def read_wav(filepath, pad=True):
    """
    Given the filepath of a wav file, this function reads it, normalizes it and pads
    it to assure it has 16k samples.
    :param filepath: existing filepath of a wav file (str)
    :param pad: is padding required? (bool)
    :returns: the sample and the target variable (tuple of (np.array, str))
    """
    sample_rate, x = wavfile.read(filepath)
    target = os.path.split(os.path.split(filepath)[0])[1]
    assert sample_rate==16000
    if pad:
        return np.pad(x, (0, 16000-len(x)), mode="constant")/32768, target
    else:
        return x/32768, target

def get_batcher(list_of_paths, batch_size, label_encoder=None, scoring=False):
    """
    Builds a batch generator given a list of batches
    :param list_of_paths: list of tuples with elements of format (filepath, target) (list)
    :param batch_size: size of the batch (int)
    :param label_encoder: fitted LabelEncoder (sklearn.LabelEncoder|optional)
    :param scoring: should the target be considered? (bool)
    :returns: batch generator
    """
    for filepaths in batching(list_of_paths, batch_size):
        wavs, targets = zip(*list(map(read_wav, filepaths)))
        if scoring:
            yield np.expand_dims(np.row_stack(wavs), 2), filepaths
        else:
            if label_encoder is None:
                yield np.expand_dims(np.row_stack(wavs), 2), np.row_stack(targets)
            else:
                yield np.expand_dims(np.row_stack(wavs), 2), np.expand_dims(label_encoder.transform(np.squeeze(targets)),1)


# ## Architecture building blocks

# Inception-1D (a.k.a wavception) is a module I designed some weeks ago for this problem. It substantially enhances the performance of a regular convolutional neural net.

# In[ ]:


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.999, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)
    
    

def inception_1d(x, is_train, depth, norm_function, activ_function, name):
    """
    Inception 1D module implementation. 
    :param x: input to the current module (4D tensor with channels-last)
    :param is_train: it is intented to be a boolean placeholder for controling the BatchNormalization behavior (0D tensor)
    :param depth: linearly controls the depth of the network (int)
    :param norm_function: normalization class (same format as the BatchNorm class above)
    :param activ_function: tensorflow activation function (e.g. tf.nn.relu) 
    :param name: name of the variable scope (str)
    """
    with tf.variable_scope(name):
        x_norm = norm_function(name="norm_input")(x, train=is_train)

        # Branch 1: 64 x conv 1x1 
        branch_conv_1_1 = tf.layers.conv1d(inputs=x_norm, filters=16*depth, kernel_size=1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_1_1")
        branch_conv_1_1 = norm_function(name="norm_conv_1_1")(branch_conv_1_1, train=is_train)
        branch_conv_1_1 = activ_function(branch_conv_1_1, "activation_1_1")

        # Branch 2: 128 x conv 3x3 
        branch_conv_3_3 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_3_3_1")
        branch_conv_3_3 = norm_function(name="norm_conv_3_3_1")(branch_conv_3_3, train=is_train)
        branch_conv_3_3 = activ_function(branch_conv_3_3, "activation_3_3_1")

        branch_conv_3_3 = tf.layers.conv1d(inputs=branch_conv_3_3, filters=32*depth, kernel_size=3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_3_3_2")
        branch_conv_3_3 = norm_function(name="norm_conv_3_3_2")(branch_conv_3_3, train=is_train)
        branch_conv_3_3 = activ_function(branch_conv_3_3, "activation_3_3_2")

        # Branch 3: 128 x conv 5x5 
        branch_conv_5_5 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_5_5_1")
        branch_conv_5_5 = norm_function(name="norm_conv_5_5_1")(branch_conv_5_5, train=is_train)
        branch_conv_5_5 = activ_function(branch_conv_5_5, "activation_5_5_1")

        branch_conv_5_5 = tf.layers.conv1d(inputs=branch_conv_5_5, filters=32*depth, kernel_size=5, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_5_5_2")
        branch_conv_5_5 = norm_function(name="norm_conv_5_5_2")(branch_conv_5_5, train=is_train)
        branch_conv_5_5 = activ_function(branch_conv_5_5, "activation_5_5_2")

        # Branch 4: 128 x conv 7x7
        branch_conv_7_7 = tf.layers.conv1d(inputs=x_norm, filters=16, kernel_size=1, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_7_7_1")
        branch_conv_7_7 = norm_function(name="norm_conv_7_7_1")(branch_conv_7_7, train=is_train)
        branch_conv_7_7 = activ_function(branch_conv_7_7, "activation_7_7_1")

        branch_conv_7_7 = tf.layers.conv1d(inputs=branch_conv_7_7, filters=32*depth, kernel_size=5, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           padding="same", name="conv_7_7_2")
        branch_conv_7_7 = norm_function(name="norm_conv_7_7_2")(branch_conv_7_7, train=is_train)
        branch_conv_7_7 = activ_function(branch_conv_7_7, "activation_7_7_2")

        # Branch 5: 16 x (max_pool 3x3 + conv 1x1)
        branch_maxpool_3_3 = tf.layers.max_pooling1d(inputs=x_norm, pool_size=3, strides=1, padding="same", name="maxpool_3")
        branch_maxpool_3_3 = norm_function(name="norm_maxpool_3_3")(branch_maxpool_3_3, train=is_train)
        branch_maxpool_3_3 = tf.layers.conv1d(inputs=branch_maxpool_3_3, filters=16, kernel_size=1, 
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              padding="same", name="conv_maxpool_3")

        # Branch 6: 16 x (max_pool 5x5 + conv 1x1)
        branch_maxpool_5_5 = tf.layers.max_pooling1d(inputs=x_norm, pool_size=5, strides=1, padding="same", name="maxpool_5")
        branch_maxpool_5_5 = norm_function(name="norm_maxpool_5_5")(branch_maxpool_5_5, train=is_train)
        branch_maxpool_5_5 = tf.layers.conv1d(inputs=branch_maxpool_5_5, filters=16, kernel_size=1, 
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              padding="same", name="conv_maxpool_5")

        # Branch 7: 16 x (avg_pool 3x3 + conv 1x1)
        branch_avgpool_3_3 = tf.layers.average_pooling1d(inputs=x_norm, pool_size=3, strides=1, padding="same", name="avgpool_3")
        branch_avgpool_3_3 = norm_function(name="norm_avgpool_3_3")(branch_avgpool_3_3, train=is_train)
        branch_avgpool_3_3 = tf.layers.conv1d(inputs=branch_avgpool_3_3, filters=16, kernel_size=1,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              padding="same", name="conv_avgpool_3")

        # Branch 8: 16 x (avg_pool 5x5 + conv 1x1)
        branch_avgpool_5_5 = tf.layers.average_pooling1d(inputs=x_norm, pool_size=5, strides=1, padding="same", name="avgpool_5")
        branch_avgpool_5_5 = norm_function(name="norm_avgpool_5_5")(branch_avgpool_5_5, train=is_train)
        branch_avgpool_5_5 = tf.layers.conv1d(inputs=branch_avgpool_5_5, filters=16, kernel_size=1, 
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              padding="same", name="conv_avgpool_5")

        # Concatenate
        output = tf.concat([branch_conv_1_1, branch_conv_3_3, branch_conv_5_5, branch_conv_7_7, branch_maxpool_3_3, 
                           branch_maxpool_5_5, branch_avgpool_3_3, branch_avgpool_5_5], axis=-1)
        return output


# ## Load and prepare Data

# In[ ]:


# Synthetic and provided noise addition
filepaths_noise = glob.glob(os.path.join(get_train_audio_path(), "_background_noise_", "*.wav"))

noise = np.concatenate(list(map(lambda x: read_wav(x, False)[0], filepaths_noise)))
noise = np.concatenate([noise, noise[::-1]])
synthetic_noise = np.concatenate([white_noise(N=16000*30, state=np.random.RandomState(655321)), 
                                  blue_noise(N=16000*30, state=np.random.RandomState(655321)),
                                  pink_noise(N=16000*30, state=np.random.RandomState(655321)),
                                  brown_noise(N=16000*30, state=np.random.RandomState(655321)),
                                  violet_noise(N=16000*30, state=np.random.RandomState(655321)),
                                  np.zeros(16000*60)])
synthetic_noise /= np.max(np.abs(synthetic_noise))
synthetic_noise = np.concatenate([synthetic_noise, (synthetic_noise+synthetic_noise[::-1])/2])
all_noise = np.concatenate([noise, synthetic_noise])


# In[ ]:


np.random.seed(655321)
random.seed(655321)

path = get_silence_path()

if not os.path.exists(path):
    os.makedirs(path) # It fails in kaggle kernel due to the read-only filesystem

for noise_clip_no in tqdm(range(8000)):
    if noise_clip_no<=4000:
        idx = np.random.randint(0, len(noise)-16000)
        clip = noise[idx:(idx+16000)]
    else:
        idx = np.random.randint(0, len(synthetic_noise)-16000)
        clip = synthetic_noise[idx:(idx+16000)]
    wavfile.write(os.path.join(path, "{0:04d}.wav".format(noise_clip_no)), 16000, 
                               ((32767*clip/np.max(np.abs(clip))).astype(np.int16)))
    


# In[ ]:


filepaths = glob.glob(os.path.join(get_train_audio_path(), "**/*.wav"), recursive=True)
filepaths += glob.glob(os.path.join(get_silence_path(), "**/*.wav"), recursive=True)
filepaths = list(filter(lambda fp: "_background_noise_" not in fp, filepaths))
validation_list = open(os.path.join(get_train_path(), "validation_list.txt")).readlines()
test_list = open(os.path.join(get_train_path(), "testing_list.txt")).readlines()
validation_list = list(map(lambda fn: os.path.join(get_train_audio_path(), fn.strip()), validation_list))
testing_list = list(map(lambda fn: os.path.join(get_train_audio_path(), fn.strip()), test_list))
training_list = np.setdiff1d(filepaths, validation_list+testing_list).tolist()


# In[ ]:


random.seed(655321)
random.shuffle(filepaths)
random.shuffle(validation_list)
random.shuffle(testing_list)
random.shuffle(training_list)


# In[ ]:


# Quick Unit-Tests
# Test number of files and their consistencies
assert all(map(lambda fp: os.path.splitext(fp)[1]==".wav", filepaths))
assert len(filepaths)==64727 - 6 + 8000
assert len(training_list) == len(filepaths) - 6798 - 6835 
assert len(validation_list) == 6798
assert len(testing_list) == 6835

# Test file existence
assert all(map(lambda fn: os.path.exists(os.path.join(fn)), validation_list))
assert all(map(lambda fn: os.path.exists(os.path.join(fn)), testing_list))
assert all(map(lambda fn: os.path.exists(os.path.join(fn)), training_list))
assert set(validation_list + testing_list + training_list) == set(filepaths)

# Test non-overlap among sets
assert len(np.intersect1d(validation_list, testing_list))==0
assert len(np.intersect1d(training_list, testing_list))==0
assert len(np.intersect1d(training_list, validation_list))==0


# In[ ]:


# Classes processing
cardinal_classes = list(set(map(lambda fp:os.path.split(os.path.split(fp)[0])[1], filepaths)))
le_classes = LabelEncoder().fit(cardinal_classes)
Counter(map(lambda fp:os.path.split(os.path.split(fp)[0])[1], filepaths))


# In[ ]:


# Quick Unit-Tests
# Test data preparation
_gen_test = get_batcher(filepaths, 1000)
batch_a_wav, batch_a_target = next(_gen_test)
batch_b_wav, batch_b_target = next(_gen_test)
_gen_test_le = get_batcher(filepaths, 1000, label_encoder=le_classes)
batch_le_wav, batch_le_target = next(_gen_test_le)

# Test batch matrix shape coherences
assert batch_a_wav.shape == (1000, 16000, 1)
assert batch_le_wav.shape == (1000, 16000, 1)
assert batch_a_wav.shape == batch_b_wav.shape == batch_le_wav.shape

# Test batch reproducibility
assert np.sum(np.abs(batch_a_wav-batch_b_wav)) != 0
assert len(batch_a_target) == len(batch_b_target) == len(batch_le_target)
assert any(batch_a_target != batch_b_target)

# Test classes label encoder
assert all(batch_le_target == np.expand_dims(le_classes.transform(np.squeeze(batch_a_target)),1))


# ## Architecture design

# Here it comes, WavCeption design

# In[ ]:


class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Architecture:
    def __init__(self, class_cardinality, seq_len=16000, name="architecture"):
        self.seq_len = seq_len
        self.class_cardinality = class_cardinality
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.name=name
        self.define_computation_graph()
        
        #Aliases
        self.ph = self.placeholders
        self.op = self.optimizers
        self.summ = self.summaries

    def define_computation_graph(self):
        # Reset graph
        tf.reset_default_graph()
        self.placeholders = NameSpacer(**self.define_placeholders())
        self.core_model = NameSpacer(**self.define_core_model())
        self.losses = NameSpacer(**self.define_losses())
        self.optimizers = NameSpacer(**self.define_optimizers())
        self.summaries = NameSpacer(**self.define_summaries())

    def define_placeholders(self):
        with tf.variable_scope("Placeholders"):
            wav_in = tf.placeholder(dtype=tf.float32, shape=(None, self.seq_len, 1), name="wav_in")
            is_train = tf.placeholder(dtype=tf.bool, shape=None, name="is_train")
            target = tf.placeholder(dtype=tf.int32, shape=(None, 1), name="target")
            acc_dev = tf.placeholder(dtype=tf.float32, shape=None, name="acc_dev")
            loss_dev = tf.placeholder(dtype=tf.float32, shape=None, name="loss_dev")
            return({"wav_in": wav_in, "target": target, "is_train": is_train, "acc_dev": 
                    acc_dev, "loss_dev": loss_dev})
        
    def define_core_model(self):
        with tf.variable_scope("Core_Model"):
            x = inception_1d(x=self.placeholders.wav_in, is_train=self.placeholders.is_train, 
                             norm_function=BatchNorm, activ_function=tf.nn.relu, depth=1,
                             name="Inception_1_1")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=1, name="Inception_1_2")
            x = tf.layers.max_pooling1d(x, 2, 2, name="maxpool_1")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=1, name="Inception_2_1")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=1, name="Inception_2_3")
            x = tf.layers.max_pooling1d(x, 2, 2, name="maxpool_2")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=2, name="Inception_3_1")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=2, name="Inception_3_2")
            x = tf.layers.max_pooling1d(x, 2, 2, name="maxpool_3")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=2, name="Inception_4_1")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=2, name="Inception_4_2")
            x = tf.layers.max_pooling1d(x, 2, 2, name="maxpool_4")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=3, name="Inception_5_1")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=3, name="Inception_5_2")
            x = tf.layers.max_pooling1d(x, 2, 2, name="maxpool_5")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=3, name="Inception_6_1")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=3, name="Inception_6_2")
            x = tf.layers.max_pooling1d(x, 2, 2, name="maxpool_6")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=4, name="Inception_7_1")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=4, name="Inception_7_2")
            x = tf.layers.max_pooling1d(x, 2, 2, name="maxpool_7")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=4, name="Inception_8_1")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=4, name="Inception_8_2")
            x = tf.layers.max_pooling1d(x, 2, 2, name="maxpool_8")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=4, name="Inception_9_1")
            x = inception_1d(x=x, is_train=self.placeholders.is_train, norm_function=BatchNorm, 
                             activ_function=tf.nn.relu, depth=4, name="Inception_9_2")
            x = tf.layers.max_pooling1d(x, 2, 2, name="maxpool_9")
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(BatchNorm(name="bn_dense_1")(x,train=self.placeholders.is_train),
                                128, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="dense_1")
            output = tf.layers.dense(BatchNorm(name="bn_dense_2")(x,train=self.placeholders.is_train),
                                self.class_cardinality, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="output")
            return({"output": output})
        
    def define_losses(self):
        with tf.variable_scope("Losses"):
            softmax_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.placeholders.target), 
                                                                        logits=self.core_model.output,
                                                                        name="softmax")
            return({"softmax": softmax_ce})

    def define_optimizers(self):
        with tf.variable_scope("Optimization"):
            op = self.optimizer.minimize(self.losses.softmax)
            return({"op": op})

    def define_summaries(self):
        with tf.variable_scope("Summaries"):
            ind_max = tf.squeeze(tf.cast(tf.argmax(self.core_model.output, axis=1), tf.int32))
            target = tf.squeeze(self.placeholders.target)
            acc= tf.reduce_mean(tf.cast(tf.equal(ind_max, target), tf.float32))
            loss = tf.reduce_mean(self.losses.softmax)
            train_scalar_probes = {"accuracy": acc, 
                                   "loss": loss}
            train_performance_scalar = [tf.summary.scalar(k, tf.reduce_mean(v), family=self.name) 
                                        for k, v in train_scalar_probes.items()]
            train_performance_scalar = tf.summary.merge(train_performance_scalar)

            dev_scalar_probes = {"acc_dev": self.placeholders.acc_dev, 
                                 "loss_dev": self.placeholders.loss_dev}
            dev_performance_scalar = [tf.summary.scalar(k, v, family=self.name) for k, v in dev_scalar_probes.items()]
            dev_performance_scalar = tf.summary.merge(dev_performance_scalar)
            return({"accuracy": acc, "loss": loss, "s_tr": train_performance_scalar, "s_de": dev_performance_scalar})


# ## Run model 

# You should use a GPU to run the model if you don't want it to take forever... In addition, you should decide when to stop the network to make the prediction. It took me 12h in a Titan X Pascal.

# In[ ]:


net = Architecture(class_cardinality=len(cardinal_classes), name="wavception")


# In[ ]:


sess = start_tensorflow_session(device="1")
sw = get_summary_writer(sess, "~/.logs_tensorboard/", "wavception", "V1") # Adjust your tensorboard logs path here
c=0


# In[ ]:


sess.run(tf.global_variables_initializer())


# In[ ]:


np.random.seed(655321)
random.seed(655321)


# In[ ]:


for epoch in range(50000):
    random.shuffle(training_list)
    batcher = get_batcher(training_list, 16, le_classes)
    for i, (batch_x, batch_y) in enumerate(batcher):
        _, loss, acc, s = sess.run([net.op.op, net.losses.softmax, net.summ.accuracy, net.summ.s_tr],
                                 feed_dict={net.ph.wav_in: batch_x, net.ph.target: batch_y, 
                                            net.ph.is_train: True})
        print("[{0:04d}|{1:04d}] Accuracy train: {2:.2f}%".format(epoch, i, acc*100))
        sw.add_summary(s, c)
        
        if c%1000==0: # Validation
            accuracies_dev=[]
            losses_dev=[]
            batcher = get_batcher(validation_list, 16, le_classes)
            for i, (batch_x, batch_y) in enumerate(batcher):
                acc, loss= sess.run([net.summ.accuracy, net.summ.loss], 
                               feed_dict={net.ph.wav_in: batch_x, net.ph.target: batch_y, 
                                          net.ph.is_train: False})
                accuracies_dev.append(acc)
                losses_dev.append(loss)
            s = sess.run(net.summ.s_de, feed_dict={net.ph.acc_dev: np.mean(accuracies_dev),
                                                        net.ph.loss_dev: np.mean(losses_dev)})
            sw.add_summary(s, c)
        c += 1


# Test accuracy

# In[ ]:


accuracies=[]
batcher = get_batcher(testing_list, 64, le_classes)
for i, (batch_x, batch_y) in tqdm(enumerate(batcher)):
    acc= sess.run(net.summ.accuracy, feed_dict={net.ph.wav_in: batch_x, net.ph.target: batch_y, 
                                                     net.ph.is_train: False})
    accuracies.append(acc)
        


# ## Prediction and submission building

# In[ ]:


scoring_list = glob.glob(os.path.join(get_scoring_audio_path(), "*.wav"), recursive=True)


# In[ ]:


batcher = get_batcher(scoring_list, 80, le_classes, scoring=True)


# In[ ]:


fns = []
prds = []
for i, (batch_x, filepaths) in tqdm(enumerate(batcher)):
    pred = sess.run(net.core_model.output, feed_dict={net.ph.wav_in: batch_x, net.ph.is_train: False})
    fns.extend(map(lambda f:os.path.split(f)[1], filepaths))
    prds.extend(map(lambda f:np.argmax(pred, axis=1).tolist(), pred))


# Note: I implemented here a quick & dirty way of solving the unknown clips problem. It still performs well (~76 LB) but there are much more smart ways to do it ;-).

# In[ ]:


# Submission storage
df=pd.DataFrame({"fname":fns, "label": prds})
df.label = le_classes.inverse_transform(df.label)
df.loc[~df.label.isin(["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence"]), "label"]="unknown"
df.to_csv(os.path.join(get_submissions_path(), "submission.csv"), index=False)

