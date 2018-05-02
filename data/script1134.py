
# coding: utf-8

# In the forum **efg2** commented about test images present in the training data set. Here's the confirmation.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np 
import pandas as pd 
import glob
import hashlib
from scipy.misc import imresize
import matplotlib.pyplot as plt
import seaborn as sns

hashes = {}
labels = {} #map fname -> label

print('computing md5 of training data:')
for base_dir in ["additional", "train"]:
    print(base_dir)
    for fname in glob.glob("../input/{}/**/*.jpg".format(base_dir)):
        labels[fname] = fname.split('/')[-2]
        h = hashlib.md5(open(fname, 'rb').read()).digest()        
        if h in hashes:
            hashes[h].append(fname)            
        else:
            hashes[h] = [fname]
print(len(hashes))   


# In[ ]:


repeated = sum(1 for k,v in hashes.items() if len(v) > 1)
print("files appearing more than once in train + additional:")
print(repeated)
        


# In[ ]:


print("identical files with different labels:")
for k,v in hashes.items():
    if len(v) > 1:
        c = set([labels[x] for x in v])
        if len(c) > 1:
            print(v, c)
            


# In[ ]:


# find test files also present in training data
leaks = []
for fname in glob.glob("../input/test/*jpg"):
    h = hashlib.md5(open(fname, 'rb').read()).digest()
    if h in hashes:
        leaks.append((fname, hashes[h]))


# In[ ]:


leaks


# In[ ]:


for t1, t2 in leaks:
    plt.figure()
    plt.title("{}, {} - {}".format(t2[0].split('/')[3], t1, t2))
    plt.imshow(np.hstack([plt.imread(t1), plt.imread(t2[0])]))
    


# This corresponds to 4% of the test set (22/512 samples). Interestingly, it appears that these images are incorrectly labeled (can be confirmed easily via LB probing).
