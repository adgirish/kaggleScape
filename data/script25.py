
# coding: utf-8

# In[ ]:


# I used the kernel and changed  LogisticRegression to Ridge.
#https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams


# In[ ]:


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])


# In[ ]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


# In[ ]:


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)


# In[ ]:


train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})


# In[ ]:


for class_name in class_names:
    train_target = train[class_name]
    classifier = Ridge(alpha=20, copy_X=True, fit_intercept=True, solver='auto',
                        max_iter=100,   normalize=False, random_state=0,  tol=0.0025)
    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict(test_features)


# In[ ]:


submission.to_csv('submission.csv', index=False)

