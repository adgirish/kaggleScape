
# coding: utf-8

# This script intends to be a starter script for Keras using pre-trained word embeddings.
# 
# **Word embedding:**
# 
# [Word embedding][1] is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers. They are also called as word vectors.
# 
# Two commonly used word embeddings are:
# 
# 1.  [Google word2vec][2]
# 2. [Stanford Glove][3]
# 
# In this notebook, we will use the GloVe word vector which is downloaded from [this link][4] 
# 
# Let us first import the necessary packages.
# 
# 
#   [1]: https://en.wikipedia.org/wiki/Word_embedding
#   [2]: https://code.google.com/archive/p/word2vec/
#   [3]: https://nlp.stanford.edu/projects/glove/
#   [4]: http://nlp.stanford.edu/data/glove.6B.zip

# In[ ]:


import os
import csv
import codecs
import numpy as np
import pandas as pd
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, merge, LSTM, Lambda, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import sys


# Let us specify the constants that are needed for the model.
# 
#  1. MAX_SEQUENCE_LENGTH : number of words from the question to be used
#  2. MAX_NB_WORDS : maximum size of the vocabulary
#  3. EMBEDDING_DIM : dimension of the word embeddings

# In[ ]:


BASE_DIR = '../input/'
GLOVE_DIR = BASE_DIR + '/WordEmbeddings/Glove/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.01


# As the first step, let us read the word vectors text file into a dictionary where the word is the key and the 300 dimensional vector is its corresponding value.
# 
# Note : This will throw an error here since the word vectors are not here in Kaggle environment.

# In[ ]:


print('Indexing word vectors.')
embeddings_index = {}
f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf-8')
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


# Now read the train and test questions into list of questions.

# In[ ]:


print('Processing text dataset')
texts_1 = [] 
texts_2 = []
labels = []  # list of label ids
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(values[3])
        texts_2.append(values[4])
        labels.append(int(values[5]))
print('Found %s texts.' % len(texts_1))

test_texts_1 = []
test_texts_2 = []
test_labels = []  # list of label ids
with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        test_texts_1.append(values[1])
        test_texts_2.append(values[2])
        test_labels.append(values[0])
print('Found %s texts.' % len(test_texts_1))


# Using keras tokenizer to tokenize the text and then do padding the sentences to 30 words

# In[ ]:


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_labels = np.array(test_labels)
del test_sequences_1
del test_sequences_2
del sequences_1
del sequences_2
import gc
gc.collect()


# Now let us create the embedding matrix where each row corresponds to a word.

# In[ ]:


print('Preparing embedding matrix.')
# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


# Now its time to build the model. Let us specify the model architecture. First layer is the embedding layer.

# In[ ]:


embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# In embedding layer, 'trainable' is set to False so as to not train the word embeddings during the back propogation.
# 
# The neural net architecture is as follows:
# 
# 1. Word embeddings of each question is passed to a 1-dimensional convolution layer followed by max pooling.
# 2. It is followed by one dense layer for each of the two questions
# 3. The outputs from both the dense layers are merged together
# 4. It is followed by a dense layer
# 5. Final layer is a sigmoid layer

# In[ ]:


# Model Architecture #
sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = Conv1D(128, 3, activation='relu')(embedded_sequences_1)
x1 = MaxPooling1D(10)(x1)
x1 = Flatten()(x1)
x1 = Dense(64, activation='relu')(x1)
x1 = Dropout(0.2)(x1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = Conv1D(128, 3, activation='relu')(embedded_sequences_2)
y1 = MaxPooling1D(10)(y1)
y1 = Flatten()(y1)
y1 = Dense(64, activation='relu')(y1)
y1 = Dropout(0.2)(y1)

merged = merge([x1,y1], mode='concat')
merged = BatchNormalization()(merged)
merged = Dense(64, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = BatchNormalization()(merged)
preds = Dense(1, activation='sigmoid')(merged)
model = Model(input=[sequence_1_input,sequence_2_input], output=preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])


# **Model training and predictions :**
# 
# Uncomment the below cell and run it in local as it is exceeding the time limits here.

# In[ ]:


pass
#model.fit([data_1,data_2], labels, validation_split=VALIDATION_SPLIT, nb_epoch=1, batch_size=1024, shuffle=True)
#preds = model.predict([test_data_1, test_data_2])
#print(preds.shape)

#out_df = pd.DataFrame({"test_id":test_labels, "is_duplicate":preds.ravel()})
#out_df.to_csv("test_predictions.csv", index=False)


# This scores about 0.55 when run locally using the word embedding. Got better scores using LSTM and Time Distributed layer.
# 
# Try different architectures and have a happy learning.

# Hope this helps to get started with keras and word embeddings in this competition.

# **References :**
# 
#  1. [On word embeddings - part 1][1] by Sebastian Ruder
#  2. [Blog post][2] by fchollet
#  3. [Code][3] by Abhishek Thakur
#  4. [Code][4] by Bradley Pallen
# 
# 
#   [1]: http://sebastianruder.com/word-embeddings-1/
#   [2]: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
#   [3]: https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question
#   [4]: https://github.com/bradleypallen/keras-quora-question-pairs
