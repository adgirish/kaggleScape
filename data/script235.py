
# coding: utf-8

# This is a basic LogisticRegression model trained using the data from https://www.kaggle.com/eoveson/convai-datasets-baseline-models
# 
# The baseline model in that kernal is tuned a little to get the data for this kernal This kernal scored 0.044 in the LB

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scipy import sparse
# set stopwords

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/dataset/train_with_convai.csv')
test = pd.read_csv('../input/dataset/test_with_convai.csv')


# In[ ]:


feats_to_concat = ['comment_text', 'toxic_level', 'attack', 'aggression']
# combining test and train
alldata = pd.concat([train[feats_to_concat], test[feats_to_concat]], axis=0)
alldata.comment_text.fillna('unknown', inplace=True)


# In[ ]:


vect_words = TfidfVectorizer(max_features=50000, analyzer='word', ngram_range=(1, 1))
vect_chars = TfidfVectorizer(max_features=20000, analyzer='char', ngram_range=(1, 3))


# In[ ]:


all_words = vect_words.fit_transform(alldata.comment_text)
all_chars = vect_chars.fit_transform(alldata.comment_text)


# In[ ]:


train_new = train
test_new = test


# In[ ]:


train_words = all_words[:len(train_new)]
test_words = all_words[len(train_new):]

train_chars = all_chars[:len(train_new)]
test_chars = all_chars[len(train_new):]


# In[ ]:


feats = ['toxic_level', 'attack']
# make sparse matrix with needed data for train and test
train_feats = sparse.hstack([train_words, train_chars, alldata[feats][:len(train_new)]])
test_feats = sparse.hstack([test_words, test_chars, alldata[feats][len(train_new):]])


# In[ ]:


col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

only_col = ['toxic']

preds = np.zeros((test_new.shape[0], len(col)))

for i, j in enumerate(col):
    print('===Fit '+j)
    
    model = LogisticRegression(C=4.0, solver='sag')
    print('Fitting model')
    model.fit(train_feats, train_new[j])
      
    print('Predicting on test')
    preds[:,i] = model.predict_proba(test_feats)[:,1]


# In[ ]:


subm = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('feat_lr_2cols.csv', index=False)

