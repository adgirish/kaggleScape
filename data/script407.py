
# coding: utf-8

# # LSTM + GloVe + Cross-validation + LearningRate changes + ...
# 
# # Notes
# 1) GRU is very similar to LSTM and not better
# 
# 2) GloVe dimension is very important. I recommend to use GloVe 840b 300d if you can (it's very hard to use it in kaggle kernels)
# 
# 3) Cross Validation is interesting for hiperparameters tuning, but for higher score you shoudn't use validation_split
# 
# 4) First Epoch is very unstable. So I use small LR on first step
# 
# 5) Dataset size is small. So you may use some additional datasets and then finetune model
# 
# 6) It's hard not to overfit the model and I have n't found yet a good way to solve this problem. BatchNormalization/Dropout don't really help. 
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46494#263247
# Maybe Reccurent Batch Normalization can help, but it is'not implemented in keras.
# 
# 7) Use Attention layer from here (AttLayer):
# https://github.com/dem-esgal/textClassifier/blob/master/textClassifierHATT.py
# 
# Thanks to (https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-051)

# In[1]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
print('hi')


# Glove dimension 100

# In[2]:


path = '../input/'
comp = 'jigsaw-toxic-comment-classification-challenge/'
#EMBEDDING_FILE=f'{path}glove-vectors/glove.6B.100d.txt'
EMBEDDING_FILE=f'{path}glove6b50d/glove.6B.50d.txt'

TRAIN_DATA_FILE=f'{path}{comp}train.csv'
TEST_DATA_FILE=f'{path}{comp}test.csv'


# Set some basic config parameters:

# In[3]:


embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use


# Read in our data and replace missing values:

# In[4]:


train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values


# Standard keras preprocessing, to turn each comment into a list of word indexes of equal length (with truncation or padding as needed).

# In[5]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# Read the glove word vectors (space delimited strings) into a dictionary from word->vector.

# In[6]:


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))


# Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when generating the random init.

# In[7]:


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std


# In[8]:


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# ROC AUC for CV in Keras see for details: https://gist.github.com/smly/d29d079100f8d81b905e

# In[15]:


import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch, score))


# Bidirectional LSTM with half-size embedding with two fully connected layers

# In[10]:


inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
x = Bidirectional(LSTM(50, return_sequences=True,dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = BatchNormalization()(x)
x = Dense(50, activation="relu")(x)
#x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)

import keras.backend as K
def loss(y_true, y_pred):
     return K.binary_crossentropy(y_true, y_pred)
    
model.compile(loss=loss, optimizer='nadam', metrics=['accuracy'])


# Now we're ready to fit out model! Use `validation_split` when for hyperparams tuning

# In[ ]:



def schedule(ind):
    a = [0.002,0.003, 0.000]
    return a[ind]
lr = callbacks.LearningRateScheduler(schedule)
[X_train, X_val, y_train, y_val] = train_test_split(X_t, y, train_size=0.95)

ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

model.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_val, y_val), callbacks=[lr, ra_val])
#model.fit(X_t, y, batch_size=64, epochs=3, callbacks=[lr])


# And finally, get predictions for the test set and prepare a submission CSV:

# In[14]:


y_test = model.predict([X_te], batch_size=1024, verbose=1)
sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
sample_submission[list_classes] = y_test
sample_submission.to_csv('LSTM-submission.csv', index=False)

