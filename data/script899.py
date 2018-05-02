
# coding: utf-8

# This is a simple Bayesian spell checker - credit to Peter Norvig ([Go here for a full description][1])
# 
# 
#   [1]: http://www.norvig.com/spell-correct.html

# In[ ]:


import numpy as np 
import pandas as pd 
import re
from collections import Counter


# In[ ]:


class PeterNovigSpellingChecker:
    WORDS = Counter()
    N = 0
    def __init__(self, alltext):
        self.WORDS = Counter(re.findall(r'\w+', alltext.lower()))
        self.N=sum(self.WORDS.values())
    def P(self, word): 
        return self.WORDS[word] / self.N
    def candidates(self, word): 
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])
    def edits1(self, word):
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)       
    def edits2(self, word): 
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
    def known(self, words): 
        return set(w for w in words if w in self.WORDS)
    def correction(self, word): 
        return max(self.candidates(word), key=self.P)


# Load training data and do some cleanup

# In[ ]:


train = pd.read_json('../input/train.json')
train['features'] = train["features"].apply(lambda x: " ".join(x))
train['features'].replace(to_replace=r'\d+', value='', inplace=True, regex=True)
train['features'].replace(to_replace=r'[^A-Za-z ]+', value='', inplace=True, regex=True)


# In[ ]:


train['features'][:5]


# Now put all the feature text into the spell checker

# In[ ]:


mycorpus = ' '.join(train.features.values)
sp = PeterNovigSpellingChecker(mycorpus)


# Now let us see how it performs on some misspellings

# In[ ]:


x = 'dogss catss Dorman'
print(' '.join(sp.correction(a) for a in x.split(' ')))


# So it works!
