
# coding: utf-8

# In this notebook, we'll see if we can use some machine learning to classify whether a subject is relaxing, or doing math problems, using data from our dataset.
# 
# Along the way, we'll learn how to get the subject data we want, and how to turn the raw EEG data into useable feature vectors.
# 
# Ready?
# 
# OK, first, we'll perform a few cleaning steps.
# 
# - We'll convert all the timestamps from strings into Python datetimes.
# - We'll convert the lists from strings into np arrays of floats.

# In[ ]:


import json
import pandas as pd

df = pd.read_csv("../input/eeg-data.csv")

# convert to arrays from strings
df.raw_values = df.raw_values.map(json.loads)
df.eeg_power = df.eeg_power.map(json.loads)


# Next, we'll grab some subject data. We're interested in the "relax" and "math" tasks, so we'll need to get the readings with those labels.

# In[ ]:



relax = df[df.label == 'relax']
math = df[(df.label == 'math1') |
          (df.label == 'math2') |
          (df.label == 'math3') |
          (df.label == 'math4') |
          (df.label == 'math5') |
          (df.label == 'math6') |
          (df.label == 'math7') |
          (df.label == 'math8') |
          (df.label == 'math9') |
          (df.label == 'math10') |
          (df.label == 'math11') |
          (df.label == 'math12') ]

len(relax)
len(math)


# Now that we have our feature vectors, let's try to build a binary classifier!
# 
# An SVM should do the trick for now. We'll make a `cross_val_svm` convenience method for doing n-fold cross-validation on the data.

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn import svm
def cross_val_svm (X,y,n):
    clf = svm.SVC()
    scores = cross_val_score(clf, X, y, cv=n)
    return scores                                              


# We'll also make a `vectors_labels` convenience function to produce an `X` list of vectors and a `y` list of labels, given two lists of vectors as input.

# In[ ]:


def vectors_labels (list1, list2):
    def label (l):
        return lambda x: l
    X = list1 + list2
    y = list(map(label(0), list1)) + list(map(label(1), list2))
    return X, y


# OK! Now, let's try the simplest feature vectors that could possibly work: the EEG power arrays produced by the Neurosky device.
# 
# To keep things from getting **too** crazy, let's just use data from just one (random) subject for these examples.

# In[ ]:


one_math = math[math['id']==12]
one_relax = relax[relax['id']==12]
X, y = vectors_labels(one_math.eeg_power.tolist(), one_relax.eeg_power.tolist())
cross_val_svm(X,y,7)


# Not so impressive, is it?
# 
# Let's build some better feature vectors. Roughly, we take each group of 512 raw values produced by the device, and FFT them to produce a power spectrum. Then, we take groups of 3 power spectra, average them, and logarithmically bin the result to produce feature vectors of 100 values.
# 
# I wrote a [blog post](http://blog.cosmopol.is/eeg/2015/06/26/pre-processing-EEG-consumer-devices.html) about this technique, if you're interested in more depth about how these feature vectors work. There'a also a [paper about this](http://people.ischool.berkeley.edu/~chuang/pubs/MMJC15.pdf) if you're into that sort of thing.
# 
# Ok, feature vector time!

# In[ ]:


from scipy import stats
from scipy.interpolate import interp1d
import itertools
import numpy as np

def spectrum (vector):
    '''get the power spectrum of a vector of raw EEG data'''
    A = np.fft.fft(vector)
    ps = np.abs(A)**2
    ps = ps[:len(ps)//2]
    return ps

def binned (pspectra, n):
    '''compress an array of power spectra into vectors of length n'''
    l = len(pspectra)
    array = np.zeros([l,n])
    for i,ps in enumerate(pspectra):
        x = np.arange(1,len(ps)+1)
        f = interp1d(x,ps)#/np.sum(ps))
        array[i] = f(np.arange(1, n+1))
    index = np.argwhere(array[:,0]==-1)
    array = np.delete(array,index,0)
    return array

def feature_vector (readings, bins=100): # A function we apply to each group of power spectra
  '''
  Create 100, log10-spaced bins for each power spectrum.
  For more on how this particular implementation works, see:
  http://coolworld.me/pre-processing-EEG-consumer-devices/
  '''
  bins = binned(list(map(spectrum, readings)), bins)
  return np.log10(np.mean(bins, 0))

ex_readings = one_relax.raw_values[:3]
feature_vector(ex_readings)

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def vectors (df):
    return [feature_vector(group) for group in list(grouper(3, df.raw_values.tolist()))[:-1]]


# In[ ]:


X,y = vectors_labels(
    vectors(one_math),
    vectors(one_relax))

cross_val_svm(X,y,7).mean()


# Now that's more like it! I bet we can do even better if we scale the data:

# In[ ]:


from sklearn import preprocessing
X = preprocessing.scale(X)
cross_val_svm(X,y,7).mean()


# Woohoo!
# 
# Let's see what kind of accuracy we get classifying relax and math readings for each subject
#  in the dataset.

# In[ ]:


def estimated_accuracy (subject):
    m = math[math['id']==subject]
    r = relax[relax['id']==subject]
    X,y = vectors_labels(vectors(m),vectors(r))
    X=preprocessing.scale(X)
    return cross_val_svm(X,y,7).mean()

[('subject '+str(subj), estimated_accuracy(subj)) for subj in range(1,31)]


# Not bad! 
# 
# I wonder why some people are easier to classify than others? Maybe not everyone was paying attention to the math (or not everyone was relaxing during the relax task).
# 
# Well, I trust this will be enough to make your own feature vectors, and explore the data yourself. Enjoy!
