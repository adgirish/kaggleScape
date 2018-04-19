
# coding: utf-8

# I have been struggling for a while on how to spell check questions while only using allowed data/software.  Here is the solution I am using now.
# 
# It is an adaptation of Peter Norvig's spell checker.  It uses word2vec ordering of words to approximate word probabilities.  Indeed, Google word2vec apparently orders words in decreasing order of frequency in the training corpus.
# 
# This kernel requires to download Google's word2vec: https://github.com/mmihaltz/word2vec-GoogleNews-vectors .  I have put it in my ../data directory.
# 
# I don't think that  the notebooks can run on Kaggle kernels but I share it anyway as many have already downloaded the word2vec embedding.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin.gz', 
                                                        binary=True)

words = model.index2word

w_rank = {}
for i,word in enumerate(words):
    w_rank[word] = i

WORDS = w_rank


# The rest of the code is a simple modification of Peter Norvig's code. Instead of computing the frequency of each word we use the above rank.

# In[ ]:


import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

def P(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# That's it. If you have downloaded word2vec then you can start using this code.  Here are few examples of what it does.
# 
# correction('quikly') returns quickly
# 
# correction('israil') returns israel
# 
# correction('neighbour') returns neighbor

# If you like this notebook then please upvote (button at the top right).
