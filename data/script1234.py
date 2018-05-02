
# coding: utf-8

# **I took an MBTI test a few years ago, curious to see results using classifiers today**

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import *
import sklearn

train = pd.read_csv('../input/mbti-type/mbti_1.csv')
us = pd.read_csv('../input/meta-kaggle/Users.csv')
ps = pd.read_csv('../input/meta-kaggle/ForumMessages.csv')
mbti = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition', 
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 
        'J':'Judging', 'P': 'Perceiving'}


# In[ ]:


ps = pd.merge(ps, us, how='left', left_on='AuthorUserId', right_on='Id')
d = {}
for i in range(len(ps)):
    if ps['DisplayName'][i] in d:
        d[str(ps['DisplayName'][i])] += ' ' + str(ps['Message'][i])
    else:
        d[str(ps['DisplayName'][i])] = str(ps['Message'][i])


# **Substitute with your UserId**

# In[ ]:


etc = ensemble.ExtraTreesClassifier(n_estimators = 20, max_depth=4, n_jobs = -1)
tfidf = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = decomposition.TruncatedSVD(n_components=10)
model = pipeline.Pipeline([('tfidf1', tfidf), ('tsvd1', tsvd), ('etc', etc)])
model.fit(train['posts'], train['type'])
pred = model.predict([d['the1owl']])[0]
print(pred, ' '.join([mbti[l] for l in list(pred)]))

