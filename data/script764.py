
# coding: utf-8

# # **0.29 LB Score + Beginner NLP Tutorial**

# > ## Introduction
# 
# This is my first Kaggle Competition on NLP. This notebook is heavily inspired from the kernels of 
# 1. abhishek : https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
# 2. nzw0301 : https://www.kaggle.com/nzw0301/simple-keras-fasttext-val-loss-0-31
# 3. sudalairajkumar : https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author
# Thank you for your brilliant kernels and tutorials.
# 
# Approach followed :
# 1. Simple Feature Engineering - Punctuation,Stop Words,Glove Sentence vectors
# 2. Creating stack features from
#  - Simple Features such as  tfidf and count vectors for words and chars and applying multinomial naive bayes(mnb)- tfidf+words+mnb,tfidf+chars+mnb,count+words+mnb,count+chars+mnb
#  - Conv Nets on keras texttosequence, NNs on glove sentence vectors and Fast Text
# 3. XGBoost, which is the final model which will use the simple features and stack features as input
# 
# Five Fold Crossvalidation is used in all the models along with early stopping for those involving epochs.
# 

# > ## Imports
# 
# In particular, the important ones are scikit and keras which will deal with the data modelling and feature extraction

# In[ ]:


# Imports
import pandas as pd
import sys
import glob
import errno
import csv
import numpy as np
from nltk.corpus import stopwords
import re
import nltk.data
import nltk
import os
from collections import OrderedDict
from subprocess import check_call
from shutil import copyfile
from sklearn.metrics import log_loss
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import mpld3
mpld3.enable_notebook()
import seaborn as sns
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from tqdm import tqdm
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalAveragePooling1D,Merge,Lambda,Input,GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D,TimeDistributed
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from keras.layers.merge import concatenate
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras import initializers
from keras import backend as K
from sklearn.linear_model import SGDClassifier as sgd
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping


# > ## Read data

# In[ ]:


# Read data
train = "../input/spooky-author-identification/train.csv"
test = "../input/spooky-author-identification/test.csv"
wv = "../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt"
X_train = pd.read_csv( train, header=0,delimiter="," )
X_test = pd.read_csv( test, header=0,delimiter="," )

authors = ['EAP','MWS','HPL']
Y_train = LabelEncoder().fit_transform(X_train['author'])


# >## Clean Data
# 
# This is to extract the pure words from the texts

# In[ ]:


# Clean data
def clean(X_train,X_test):
    X_train['words'] = [re.sub("[^a-zA-Z]"," ", data).lower().split() for data in X_train['text']]
    X_test['words'] = [re.sub("[^a-zA-Z]"," ", data).lower().split() for data in X_test['text']]
    return X_train,X_test
X_train,X_test = clean(X_train,X_test)


# > # Simple Feature Engineering
# 
# I am using viz. three types of features
# 1. Punctuations - Each author favours a particular type(s) of punctuation. Hence I've taken sets of punctuations which are similar in behaviour. They are ";:" , ",." , "?" , "'" , """ and the combined set of all of these.
# 2. Stop word percentage- Stop word percentage of each text 
# 
# 

# In[ ]:


# Feature Engineering
# Punctuation
punctuations = [{"id":1,"p":"[;:]"},{"id":2,"p":"[,.]"},{"id":3,"p":"[?]"},{"id":4,"p":"[\']"},{"id":5,"p":"[\"]"},{"id":6,"p":"[;:,.?\'\"]"}]
for p in punctuations:
    punctuation = p["p"]
    _train =  [ sentence.split() for sentence in X_train['text'] ]
    X_train['punc_'+str(p["id"])] = [len([word for word in sentence if bool(re.search(punctuation, word))])*100.0/len(sentence) for sentence in _train]    

    _test =  [ sentence.split() for sentence in X_test['text'] ]
    X_test['punc_'+str(p["id"])] = [len([word for word in sentence if bool(re.search(punctuation, word))])*100.0/len(sentence) for sentence in _test]    



# In[ ]:


# Feature Engineering
# Stop Words
_dist_train = [x for x in X_train['words']]
X_train['stop_word'] = [len([word for word in sentence if word in stopwords.words('english')])*100.0/len(sentence) for sentence in _dist_train]

_dist_test = [x for x in X_test['words']]
X_test['stop_word'] = [len([word for word in sentence if word in stopwords.words('english')])*100.0/len(sentence) for sentence in _dist_test]    


# > ## Stack Features
# 
# Thanks sudalairajkumar, for explaining this technique.
# 
# Stack features are features which are in itself predictions from existing features. Existing features which have many dimensions are good usecases to try out feature stacking on.
# 
# The motivation for using stack features is:
# 1. To create features with respect to documents such as tfidf and counts, both word based and character based.
# 2. To reduce the dimensionality of the huge feature vectors of the above so as to fit the data better.
# 3. Multinomial Bayes works well due to the multinomial distribution and strong independence assumptions in the model.
# 
# 

# In[ ]:


# Feature Engineering
# tfidf - words - nb
def tfidfWords(X_train,X_test):
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
    full_tfidf = tfidf_vec.fit_transform(X_train['text'].values.tolist() + X_test['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(X_train['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(X_test['text'].values.tolist())
    return train_tfidf,test_tfidf,full_tfidf
    
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def do_tfidf_MNB(X_train,X_test,Y_train):
    train_tfidf,test_tfidf,full_tfidf = tfidfWords(X_train,X_test)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([X_train.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = Y_train[dev_index], Y_train[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    return pred_train,pred_full_test

pred_train,pred_test = do_tfidf_MNB(X_train,X_test,Y_train)
X_train["tfidf_words_nb_eap"] = pred_train[:,0]
X_train["tfidf_words_nb_hpl"] = pred_train[:,1]
X_train["tfidf_words_nb_mws"] = pred_train[:,2]
X_test["tfidf_words_nb_eap"] = pred_test[:,0]
X_test["tfidf_words_nb_hpl"] = pred_test[:,1]
X_test["tfidf_words_nb_mws"] = pred_test[:,2]


# In[ ]:


# Feature Engineering
# tfidf - chars - nb
def tfidfWords(X_train,X_test):
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,5),analyzer='char')
    full_tfidf = tfidf_vec.fit_transform(X_train['text'].values.tolist() + X_test['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(X_train['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(X_test['text'].values.tolist())
    return train_tfidf,test_tfidf
    
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def do(X_train,X_test,Y_train):
    train_tfidf,test_tfidf = tfidfWords(X_train,X_test)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([X_train.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = Y_train[dev_index], Y_train[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    return pred_train,pred_full_test
pred_train,pred_test = do(X_train,X_test,Y_train)
X_train["tfidf_chars_nb_eap"] = pred_train[:,0]
X_train["tfidf_chars_nb_hpl"] = pred_train[:,1]
X_train["tfidf_chars_nb_mws"] = pred_train[:,2]
X_test["tfidf_chars_nb_eap"] = pred_test[:,0]
X_test["tfidf_chars_nb_hpl"] = pred_test[:,1]
X_test["tfidf_chars_nb_mws"] = pred_test[:,2]


# In[ ]:


# Feature Engineering
# count - words - nb
def countWords(X_train,X_test):
    count_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
    count_vec.fit(X_train['text'].values.tolist() + X_test['text'].values.tolist())
    train_count = count_vec.transform(X_train['text'].values.tolist())
    test_count = count_vec.transform(X_test['text'].values.tolist())
    return train_count,test_count
    
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def do_count_MNB(X_train,X_test,Y_train):
    train_count,test_count=countWords(X_train,X_test)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([X_train.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = train_count[dev_index], train_count[val_index]
        dev_y, val_y = Y_train[dev_index], Y_train[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_count)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    return pred_train,pred_full_test

pred_train,pred_test = do_count_MNB(X_train,X_test,Y_train)
X_train["count_words_nb_eap"] = pred_train[:,0]
X_train["count_words_nb_hpl"] = pred_train[:,1]
X_train["count_words_nb_mws"] = pred_train[:,2]
X_test["count_words_nb_eap"] = pred_test[:,0]
X_test["count_words_nb_hpl"] = pred_test[:,1]
X_test["count_words_nb_mws"] = pred_test[:,2]


# In[ ]:


# Feature Engineering
# count - chars - nb
def countChars(X_train,X_test):
    count_vec = CountVectorizer(ngram_range=(1,7),analyzer='char')
    count_vec.fit(X_train['text'].values.tolist() + X_test['text'].values.tolist())
    train_count = count_vec.transform(X_train['text'].values.tolist())
    test_count = count_vec.transform(X_test['text'].values.tolist())
    return train_count,test_count
    
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def do_count_chars_MNB(X_train,X_test,Y_train):
    train_count,test_count=countChars(X_train,X_test)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([X_train.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = train_count[dev_index], train_count[val_index]
        dev_y, val_y = Y_train[dev_index], Y_train[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_count)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.
    return pred_train,pred_full_test

pred_train,pred_test = do_count_chars_MNB(X_train,X_test,Y_train)
X_train["count_chars_nb_eap"] = pred_train[:,0]
X_train["count_chars_nb_hpl"] = pred_train[:,1]
X_train["count_chars_nb_mws"] = pred_train[:,2]
X_test["count_chars_nb_eap"] = pred_test[:,0]
X_test["count_chars_nb_hpl"] = pred_test[:,1]
X_test["count_chars_nb_mws"] = pred_test[:,2]


# > ## Creating sentence vectors from individual word vectors
# 
# Thanks abhishek, for explaining this technique.
# sent2vec creates a normalized vector for the whole sentence from Glove embeddings. This works better than simply averaging or summing up the individual word vectors to obtain sentence vectors.
# 
# These are simple features.

# In[ ]:


# load the GloVe vectors in a dictionary:

def loadWordVecs():
    embeddings_index = {}
    f = open(wv)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def sent2vec(embeddings_index,s): # this function creates a normalized vector for the whole sentence
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stopwords.words('english')]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(100)
    return v / np.sqrt((v ** 2).sum())

def doGlove(x_train,x_test):
    embeddings_index = loadWordVecs()
    # create sentence vectors using the above function for training and validation set
    xtrain_glove = [sent2vec(embeddings_index,x) for x in tqdm(x_train)]
    xtest_glove = [sent2vec(embeddings_index,x) for x in tqdm(x_test)]
    xtrain_glove = np.array(xtrain_glove)
    xtest_glove = np.array(xtest_glove)
    return xtrain_glove,xtest_glove,embeddings_index

glove_vecs_train,glove_vecs_test,embeddings_index = doGlove(X_train['text'],X_test['text'])
X_train[['sent_vec_'+str(i) for i in range(100)]] = pd.DataFrame(glove_vecs_train.tolist())
X_test[['sent_vec_'+str(i) for i in range(100)]] = pd.DataFrame(glove_vecs_test.tolist())



# > ## Using Neural Networks and Facebook's Fasttext
# 
# 1. NN => Conv Net which uses keras text to sequences to obtain input features
# 2. NN_Glove =>  Simple Neural Net which uses sentence glove features
# 3. Fasttext => Uses fasttext implementation via keras. Thanks nzw0301, for explaining this technique
# 
# All of the above are used to obtain stack features.

# In[ ]:


# Using Neural Networks and Facebook's Fasttext
earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

# NN
def doAddNN(X_train,X_test,pred_train,pred_test):
    X_train["nn_eap"] = pred_train[:,0]
    X_train["nn_hpl"] = pred_train[:,1]
    X_train["nn_mws"] = pred_train[:,2]
    X_test["nn_eap"] = pred_test[:,0]
    X_test["nn_hpl"] = pred_test[:,1]
    X_test["nn_mws"] = pred_test[:,2]
    return X_train,X_test

def initNN(nb_words_cnt,max_len):
    model = Sequential()
    model.add(Embedding(nb_words_cnt,32,input_length=max_len))
    model.add(Dropout(0.3))
    model.add(Conv1D(64,
                     5,
                     padding='valid',
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    return model

def doNN(X_train,X_test,Y_train):
    max_len = 70
    nb_words = 10000
    
    print('Processing text dataset')
    texts_1 = []
    for text in X_train['text']:
        texts_1.append(text)

    print('Found %s texts.' % len(texts_1))
    test_texts_1 = []
    for text in X_test['text']:
        test_texts_1.append(text)
    print('Found %s texts.' % len(test_texts_1))
    
    tokenizer = Tokenizer(num_words=nb_words)
    tokenizer.fit_on_texts(texts_1 + test_texts_1)
    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)

    xtrain_pad = pad_sequences(sequences_1, maxlen=max_len)
    xtest_pad = pad_sequences(test_sequences_1, maxlen=max_len)
    del test_sequences_1
    del sequences_1
    nb_words_cnt = min(nb_words, len(word_index)) + 1

    # we need to binarize the labels for the neural net
    ytrain_enc = np_utils.to_categorical(Y_train)
    
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([xtrain_pad.shape[0], 3])
    for dev_index, val_index in kf.split(xtrain_pad):
        dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initNN(nb_words_cnt,max_len)
        model.fit(dev_X, y=dev_y, batch_size=32, epochs=4, verbose=1,validation_data=(val_X, val_y),callbacks=[earlyStopping])
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(xtest_pad)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
    return doAddNN(X_train,X_test,pred_train,pred_full_test/5)

## NN Glove

def doAddNN_glove(X_train,X_test,pred_train,pred_test):
    X_train["nn_glove_eap"] = pred_train[:,0]
    X_train["nn_glove_hpl"] = pred_train[:,1]
    X_train["nn_glove_mws"] = pred_train[:,2]
    X_test["nn_glove_eap"] = pred_test[:,0]
    X_test["nn_glove_hpl"] = pred_test[:,1]
    X_test["nn_glove_mws"] = pred_test[:,2]
    return X_train,X_test

def initNN_glove():
    # create a simple 3 layer sequential neural net
    model = Sequential()

    model.add(Dense(128, input_dim=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(3))
    model.add(Activation('softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def doNN_glove(X_train,X_test,Y_train,xtrain_glove,xtest_glove):
    # scale the data before any neural net:
    scl = preprocessing.StandardScaler()
    ytrain_enc = np_utils.to_categorical(Y_train)
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    xtrain_glove = scl.fit_transform(xtrain_glove)
    xtest_glove = scl.fit_transform(xtest_glove)
    pred_train = np.zeros([xtrain_glove.shape[0], 3])
    
    for dev_index, val_index in kf.split(xtrain_glove):
        dev_X, val_X = xtrain_glove[dev_index], xtrain_glove[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initNN_glove()
        model.fit(dev_X, y=dev_y, batch_size=32, epochs=10, verbose=1,validation_data=(val_X, val_y),callbacks=[earlyStopping])
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(xtest_glove)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
    return doAddNN_glove(X_train,X_test,pred_train,pred_full_test/5)

# Fast Text

def doAddFastText(X_train,X_test,pred_train,pred_test):
    X_train["ff_eap"] = pred_train[:,0]
    X_train["ff_hpl"] = pred_train[:,1]
    X_train["ff_mws"] = pred_train[:,2]
    X_test["ff_eap"] = pred_test[:,0]
    X_test["ff_hpl"] = pred_test[:,1]
    X_test["ff_mws"] = pred_test[:,2]
    return X_train,X_test


def initFastText(embedding_dims,input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def preprocessFastText(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    return text

def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
        
    docs = []
    for doc in df.text:
        doc = preprocessFastText(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    
    return docs

def doFastText(X_train,X_test,Y_train):
    min_count = 2

    docs = create_docs(X_train)
    tokenizer = Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts(docs)
    num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

    tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
    tokenizer.fit_on_texts(docs)
    docs = tokenizer.texts_to_sequences(docs)

    maxlen = 300

    docs = pad_sequences(sequences=docs, maxlen=maxlen)
    input_dim = np.max(docs) + 1
    embedding_dims = 20

    # we need to binarize the labels for the neural net
    ytrain_enc = np_utils.to_categorical(Y_train)

    docs_test = create_docs(X_test)
    docs_test = tokenizer.texts_to_sequences(docs_test)
    docs_test = pad_sequences(sequences=docs_test, maxlen=maxlen)
    xtrain_pad = docs
    xtest_pad = docs_test
    
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([xtrain_pad.shape[0], 3])
    for dev_index, val_index in kf.split(xtrain_pad):
        dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initFastText(embedding_dims,input_dim)
        model.fit(dev_X, y=dev_y, batch_size=32, epochs=25, verbose=1,validation_data=(val_X, val_y),callbacks=[earlyStopping])
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(docs_test)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
    return doAddFastText(X_train,X_test,pred_train,pred_full_test/5)

X_train,X_test = doFastText(X_train,X_test,Y_train)
X_train,X_test = doNN(X_train,X_test,Y_train)
X_train,X_test = doNN_glove(X_train,X_test,Y_train,glove_vecs_train,glove_vecs_test)


# > ## Final model 
# 
# Uses XGBoost which contains all the simple and stack features as input

# In[ ]:


# Final Model
# XGBoost
def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = child
    param['subsample'] = 0.8
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 2000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model

def do(X_train,X_test,Y_train):
    drop_columns=["id","text","words"]
    x_train = X_train.drop(drop_columns+['author'],axis=1)
    x_test = X_test.drop(drop_columns,axis=1)
    y_train = Y_train
    
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([x_train.shape[0], 3])
    for dev_index, val_index in kf.split(x_train):
        dev_X, val_X = x_train.loc[dev_index], x_train.loc[val_index]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, x_test, seed_val=0, colsample=0.7)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("cv scores : ", cv_scores)
    return pred_full_test/5
result = do(X_train,X_test,Y_train)

