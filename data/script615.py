
# coding: utf-8

# # **This notebook's best result: val_acc is 0.8779, val_loss is 0.3129**

# # **1. Few Preprocessings**
# # **2. Model: FastText by Keras**
# ## **2.1** Change Preprocessings:
# - Do lower case 

# In[ ]:


import numpy as np

import pandas as pd

from collections import defaultdict

import keras
import keras.backend as K
from keras.layers import Dense, GlobalAveragePooling1D, Embedding
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

np.random.seed(7)


# In[ ]:


df = pd.read_csv('./../input/train.csv')
a2c = {'EAP': 0, 'HPL' : 1, 'MWS' : 2}
y = np.array([a2c[a] for a in df.author])
y = to_categorical(y)


# # 1. **Few Preprocessings**
# 
# In traditional NLP tasks, preprocessings play an important role, but...
# 
# ## **Low-frequency words**
# In my experience, fastText is very fast, but I need to delete rare words to avoid overfitting.
# 
# **NOTE**:
# Some keywords are rare words, such like *Cthulhu* in *Cthulhu Mythos* of *Howard Phillips Lovecraft*.
# But these are useful for this task.
# 
# ## **Removing Stopwords**
# 
# Nothing.
# To identify author from a sentence, some stopwords play an important role because one has specific usages of them.
# 
# ## **Stemming and Lowercase**
# 
# Nothing.
# This reason is the same for stopwords removing.
# And I guess some stemming rules provided by libraries is bad for this task because all author is the older author.
# 
# ## **Cutting long sentence**
# 
# Too long documents are cut.
# 
# ## **Punctuation**
# 
# Because I guess each author has unique punctuations's usage in the novel, I separate them from words.
# 
# e.g. `Don't worry` -> `Don ' t worry`
# 
# ## **Is it slow?**
# 
# Don't worry! FastText is a very fast algorithm if it runs on CPU. 

# # **Let's check character distribution per author**

# In[ ]:


counter = {name : defaultdict(int) for name in set(df.author)}
for (text, author) in zip(df.text, df.author):
    text = text.replace(' ', '')
    for c in text:
        counter[author][c] += 1

chars = set()
for v in counter.values():
    chars |= v.keys()
    
names = [author for author in counter.keys()]

print('c ', end='')
for n in names:
    print(n, end='   ')
print()
for c in chars:    
    print(c, end=' ')
    for n in names:
        print(counter[n][c], end=' ')
    print()


# # **Summary of character distribution**
# 
# - HPL and EAP used non ascii characters like a `ä`.
# - The number of punctuations seems to be good feature
# 

# # **Preprocessing**
# 
# My preproceeings are 
# 
# - Separate punctuation from words
# - Remove lower frequency words ( <= 2)
# - Cut a longer document which contains `256` words

# In[ ]:


def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    return text


# In[ ]:


def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
        
    docs = []
    for doc in df.text:
        doc = preprocess(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    
    return docs


# In[ ]:


min_count = 2

docs = create_docs(df)
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

maxlen = 256

docs = pad_sequences(sequences=docs, maxlen=maxlen)


# # **2. Model: FastText by Keras**
# 
# FastText is very fast and strong baseline algorithm for text classification based on Continuous Bag-of-Words model a.k.a Word2vec.
# 
# FastText contains only three layers:
# 
# 1. Embeddings layer: Input words (and word n-grams) are all words in a sentence/document
# 2. Mean/AveragePooling Layer: Taking average vector of Embedding vectors
# 3. Softmax layer
# 
# There are some implementations of FastText:
# 
# - Original library provided by Facebook AI research: https://github.com/facebookresearch/fastText
# - Keras: https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py
# - Gensim: https://radimrehurek.com/gensim/models/wrappers/fasttext.html
# 
# Original Paper: https://arxiv.org/abs/1607.01759 : More detail information about fastText classification model

# # My FastText parameters are:
# 
# - The dimension of word vector is 20
# - Optimizer is `Adam`
# - Inputs are words and word bi-grams
#   - you can change this parameter by passing the max n-gram size to argument of `create_docs` function.
# 

# In[ ]:


input_dim = np.max(docs) + 1
embedding_dims = 20


# In[ ]:


def create_model(embedding_dims=20, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


# In[ ]:


epochs = 25
x_train, x_test, y_train, y_test = train_test_split(docs, y, test_size=0.2)

model = create_model()
hist = model.fit(x_train, y_train,
                 batch_size=16,
                 validation_data=(x_test, y_test),
                 epochs=epochs,
                 callbacks=[EarlyStopping(patience=2, monitor='val_loss')])


# ### **Result**
# 
# - Best val_loss is 0.3409
# - Best val_acc is 0.8700
# 
# 

# # **2.1 Change Preprocessings**
# 
# Next, I change some parameters and preprocessings to improve fastText model.
# ## **2.1.1 Do lower case**

# In[ ]:


docs = create_docs(df)
tokenizer = Tokenizer(lower=True, filters='')
tokenizer.fit_on_texts(docs)
num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

tokenizer = Tokenizer(num_words=num_words, lower=True, filters='')
tokenizer.fit_on_texts(docs)
docs = tokenizer.texts_to_sequences(docs)

maxlen = 256

docs = pad_sequences(sequences=docs, maxlen=maxlen)

input_dim = np.max(docs) + 1


# In[ ]:


epochs = 16
x_train, x_test, y_train, y_test = train_test_split(docs, y, test_size=0.2)

model = create_model()
hist = model.fit(x_train, y_train,
                 batch_size=16,
                 validation_data=(x_test, y_test),
                 epochs=epochs,
                 callbacks=[EarlyStopping(patience=2, monitor='val_loss')])


# **Result**
# 
# - Best val_loss is 0.3129
# - Best val_acc is 0.8787

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
docs = create_docs(test_df)
docs = tokenizer.texts_to_sequences(docs)
docs = pad_sequences(sequences=docs, maxlen=maxlen)
y = model.predict_proba(docs)

result = pd.read_csv('../input/sample_submission.csv')
for a, i in a2c.items():
    result[a] = y[:, i]


# In[ ]:


result.to_csv('fastText_result.csv', index=False)

