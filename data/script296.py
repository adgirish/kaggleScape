
# coding: utf-8

# **In this notebook, we are going to tackle the same toxic classification problem just like my previous notebooks but this time round, we are going deeper with the use of Character-level features and Convolutional Neural Network (CNNs). **
# 
# ***Updated with saved model and submission below***
# 
# 
# ![](https://i.imgur.com/okCCLAU.jpg)
# 
# 
# **Why do we consider the idea of using char-gram features?**
# 
# 
# You might noticed that there are a lot of sparse misspellings due to the nature of the dataset. When we train our model using the word vectors from our training set, we might be missing out some genuine words and mispellings that are not present in the training set but yet present in our prediction set. Sometimes that wouldn't affect the model's capability to make good judgement, but most of the time, it's unable to correctly classify because the misspelt words are not in the model's "dictionary". 
# 
# Hence, if we could "go deeper" by splitting the sentence into a list of characters instead of words, the chances that the same characters that are present in both training and prediction set are much higher. You could imagine that this approach introduce another problem: an explosion of dimensions. One of the ways to tackle this problem is to use CNN as it's designed to solve high-dimensional dataset like images. Traditionally, CNN is used to solve computer vision problems but there's an increased trend of using CNN not just in Kaggle competitions but also in papers written by researchers too. Therefore, I believe it deserve a writeup and without much ado, let's see how we can apply CNN to our competition at hand.
# 
# I have skipped some elaboration of some concepts like embeddings which I have went through in my previous notebooks, so take a look at these if you are interested in learning more:
# 
# * [Do Pretrained Embeddings Give You The Extra Edge?](https://www.kaggle.com/sbongo/do-pretrained-embeddings-give-you-the-extra-edge)
# * [[For Beginners] Tackling Toxic Using Keras](https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras)

# **A brief glance at Convolutional Neural Network (CNNs)**
# 
# CNN is basically a feed-forward neural network that consists of several layers such as the convolution, pooling and some densely connected layers that we are familiar with.
# 
# ![](https://i.imgur.com/aa46tRe.png)
# 
# Firstly, as seen in the above picture, we feed the data(image in this case) into the convolution layer. The convolution layer works by sliding a window across the input data and as it slides, the window(filter) applies some matrix operations with the underlying data that falls in the window. And when you eventually collect all the result of the matrix operations, you will have a condensed output in another matrix(we call it a feature map).
# 
# ![](https://i.imgur.com/wSbiLCi.gif)
# 
# With the resulting matrix at hand, you do a max pooling that basically down-samples or in another words decrease the number of dimensions without losing the essence. 
# 
# ![](https://i.imgur.com/Cphci9k.png)
# 
# Consider this simplified image of max pooling operation above. In the above example, we slide a 2 X 2 filter window across our dataset in strides of 2. As it's sliding, it grabs the maximum value and put it into a smaller-sized matrix.
# 
# There are different ways to down-sample the data such as min-pooling, average-pooling and in max-pooling, you simply take the maximum value of the matrix. Imagine that you have a list: [1,4,0,8,5]. When you do max-pooling on this list, you will only retain the value "8". Indirectly, we are only concerned about the existence of 8, and not the location of it. Despite it's simplicity, it's works quite well and it's a pretty niffy way to reduce the data size.
# 
# Again, with the down-sized "after-pooled" matrix, you could feed it to a densely connected layer which eventually leads to prediction.
# 
# **How does this apply to NLP in our case?**
# 
# Now, forget about real pixels about a minute and imagine using each tokenized character as a form of pixel in our input matrix. Just like word vectors, we could also have character vectors that gives a lower-dimension representation. So for a list of 10 sentences that consists of 50 characters each, using a 30-dimensional embedding will allow us to feed in a 10x50x30 matrix into our convolution layer.
# ![](https://i.imgur.com/g59nKYc.jpg)
# Looking at the above picture, let's just focus(for now) on 1 sentence instead of a list. Each character is represented in a row (8 characters), and each embedding dimension is represented in a column (5 dimensions) in this starting matrix.
# 
# You would begin the convolution process by using filters of different dimensions to "slide" across your initial matrix to get a lower-dimension feature map. There's something I deliberately missed out earlier: filters. 
# 
# ![](https://i.imgur.com/Lwa7wBG.gif)
# The sliding window that I mention earlier are actually filters that are designed to capture different distinctive features in the input data. By defining the dimension of the filter, you can control the window of infomation you want to "summarize". To translate back in the picture, each of the feature maps could contain 1 high level representation of the embeddings for each character.
# 
# 
# Next, we would apply a max pooling to get the maximum value in each feature map. In our context, some characters in each filter would be selected through this max pooling process based on their values. As usual, we would then feed into a normal densely connected layer that outputs to a softmax function which gives the probabilities of each class.
# 
# Note that my explanation hides some technical details to facilitate understanding. There's a whole load of things that you could tweak with CNN. For instance, the stride size which determine how often the filter will be applied, narrow VS wide CNN, etc.
# 
# **Okay! Let's see how we could implement CNN in our competition.**

# As always, we start off with the importing of relevant libraries and dataset:

# In[ ]:


import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU,Conv1D,MaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
from sklearn.model_selection import train_test_split
from keras.models import load_model


# In[ ]:


train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
submit = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
submit_template = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv', header = 0)


# Split into training and test set:

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]], test_size = 0.10, random_state = 42)


# Store the comments as seperate variables for further processing.

# In[ ]:


list_sentences_train = X_train["comment_text"]
list_sentences_test = X_test["comment_text"]
list_sentences_submit = submit["comment_text"]


# In our previous notebook, we have began using Kera's helpful Tokenizer class to help us do the gritty text processing work. We are going to use it again to help us split the text into characters by setting the "char_level" parameter to true.

# In[ ]:


max_features = 20000
tokenizer = Tokenizer(num_words=max_features,char_level=True)


# This function allows Tokenizer to create an index of the tokenized unique characters. Eg. a=1, b=2, etc

# In[ ]:


tokenizer.fit_on_texts(list(list_sentences_train))


# Then we get back a list of sentences with the sequence of indexes which represent each character.

# In[ ]:


list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_sentences_test = tokenizer.texts_to_sequences(list_sentences_test)
list_tokenized_submit = tokenizer.texts_to_sequences(list_sentences_submit)


# Since there are sentences with varying length of characters, we have to get them on a constant size. Let's put them to a length of 500 characters for each sentence:

# In[ ]:


maxlen = 500
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_sentences_test, maxlen=maxlen)
X_sub = pad_sequences(list_tokenized_submit, maxlen=maxlen)


# Just in case you are wondering, the reason why I used 500 is because most of the number of characters in a sentence falls within 0 to 500:

# In[ ]:


totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
plt.hist(totalNumWords)
plt.show()


# Finally, we can start buliding our model.

# First, we set up our input layer. As mentioned in the Keras documentation, we have to include the shape for the very first layer and Keras will automatically derive the shape for the rest of the layers.

# In[ ]:


inp = Input(shape=(maxlen, ))


# We use an embedding size of 240. That also means that we are projecting characters on a 240-dimension vector space. It will output a (num of sentences X 500 X 240) matrix. We have talked about embedding layer in my previous notebooks, so feel free to check them out.

# In[ ]:


embed_size = 240
x = Embedding(len(tokenizer.word_index)+1, embed_size)(inp)


# Here's the meat of our notebook. With the output of embedding layer, we feed it into a convolution layer. We use a window size of 4 (remember it's 5 in the earlier picture above) and 100 filters (it's 6 in the earlier picture above) to extract the features in our data. That also means we slides a window across the 240 dimensions of embeddings for each of the 500 characters and it will result in a (num of sentences X 500 X 100) matrix. Notice that we have set padding to "same". What does this padding means?
# ![](https://i.imgur.com/hITQent.png)
# For simplicity sake, let's imagine we have a 32 x 32 x 3 input matrix and a 5 x 5 x 3 filter, if you apply the filter on the matrix with 1 stride, you will end up with a 28 x 28 x 3 matrix. In the early stages, you would want to preserve as much information as possible, so you will want to have a 32 x 32 x 3 matrix back. If we add(padding) some zeros around the original input matrix, we will be sure that the result output matrix dimension will be the same. But if you really want to have the resulting matrix to be reduced, you can set the padding parameter to "valid".

# In[ ]:


x = Conv1D(filters=100,kernel_size=4,padding='same', activation='relu')(x)


# Then we pass it to the max pooling layer that applies the max pool operation on a window of every 4 characters. And that is why we get an output of (num of sentences X 125 X 100) matrix.

# In[ ]:


x=MaxPooling1D(pool_size=4)(x)


# Next, we pass it to the Bidriectional LSTM that we are famliar with, since the previous notebook. 

# In[ ]:


x = Bidirectional(GRU(60, return_sequences=True,name='lstm_layer',dropout=0.2,recurrent_dropout=0.2))(x)


# Afterwhich, we apply a max pooling again but this time round, it's a global max pooling. What's the difference between this and the previous max pooling attempt?
# 
# In the previous max pooling attempt, we merely down-sampled a single 2nd dimension, which contains the number of characters. From a matrix of:
# (num of sentences X 500 X 100)
# it becomes:
# (num of sentences X 125 X 100)
# which is still a 3d matrix.
# 
# But in global max pooling, we perform pooling operation across several dimensions(2nd and 3rd dimension) into a single dimension. So it outputs a:
# (num of sentences X 120) 2D matrix.

# In[ ]:


x = GlobalMaxPool1D()(x)


# Now that we have a 2D matrix, it's convenient to plug it into the densely connected layer, followed by a relu activation function.

# In[ ]:


x = Dense(50, activation="relu")(x)


# We'll pass it through a dropout layer and a densely connected layer that eventually passes to a sigmoid function.

# In[ ]:


x = Dropout(0.2)(x)
x = Dense(6, activation="sigmoid")(x)


# You could experiment with the dropout rate and size of the dense connected layer to see it could decrease overfitting.
# 
# Finally, we move on to train the model with 6 epochs and the results seems pretty decent. The training loss decreases steadily along with validation loss until at the 5th or 6th epoch where traces of overfitting starts to emerge.

# In[ ]:


model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                 metrics=['accuracy'])


# In[ ]:


model.summary()


# Due to Kaggle kernel time limit, I have pasted the training output of these 6 epochs.

# In[ ]:


batch_size = 32
epochs = 6
#uncomment below to train in your local machine
#hist = model.fit(X_t,y_train, batch_size=batch_size, epochs=epochs,validation_data=(X_te,y_test),callbacks=callbacks_list)


# Train on 143613 samples, validate on 15958 samples
# 
# Epoch 1/6
# 143613/143613 [==============================] - 2580s 18ms/step - loss: 0.0786 - acc: 0.9763 - val_loss: 0.0585 - val_acc: 0.9806
# 
# Epoch 2/6
# 143613/143613 [==============================] - 2426s 17ms/step - loss: 0.0582 - acc: 0.9804 - val_loss: 0.0519 - val_acc: 0.9816
# 
# Epoch 3/6
# 143613/143613 [==============================] - 2471s 17ms/step - loss: 0.0531 - acc: 0.9816 - val_loss: 0.0489 - val_acc: 0.9823
# 
# Epoch 4/6
# 143613/143613 [==============================] - 2991s 21ms/step - loss: 0.0505 - acc: 0.9821 - val_loss: 0.0484 - val_acc: 0.9829
# 
# Epoch 5/6
# 143613/143613 [==============================] - 3023s 21ms/step - loss: 0.0487 - acc: 0.9826 - val_loss: 0.0463 - val_acc: 0.9829
# 
# Epoch 6/6
# 143613/143613 [==============================] - 2961s 21ms/step - loss: 0.0474 - acc: 0.9830 - val_loss: 0.0463 - val_acc: 0.9831

# **UPDATE**
# 
# I have uploaded the saved model in this notebook so that you could even continue the training process. To load the model and do a prediction, you could do this:

# In[ ]:


model = load_model('../input/epoch-6-model/model-e6.hdf5')

batch_size = 32
y_submit = model.predict(X_sub,batch_size=batch_size,verbose=1)


# Getting the prediction data in a format ready for competition submission:

# In[ ]:


y_submit[np.isnan(y_submit)]=0
sample_submission = submit_template
sample_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_submit
sample_submission.to_csv('submission.csv', index=False)


# I hope this notebook serves as a good start for beginners who are interested in tackling NLP problems using the CNN angle. There are some ideas which you could use to push the performance further, such as :
# 1. Tweak CNN parameters such as number of strides, different padding settings, window size.
# 2. Hyper-parameter tunings
# 3. Experiment with different architecture layers
# 
# Thank you for your time in reading and if you like what I wrote, support me by upvoting my notebook..
# 
# With the toxic competition coming to an end in a month, I wish everyone godspeed!
