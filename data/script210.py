
# coding: utf-8

# # Using FastText models (not vectors) for robust embeddings
# 
# I'd like to explain my approach of using pretrained FastText models as input to Keras Neural Networks. FastText is a word embedding not unlike Word2Vec or GloVe, but the cool thing is that each word vector is based on sub-word character n-grams. That means that even for previously unseen words (e.g. due to typos), the model can make an educated guess towards its meaning. To find out more about FastText, check out both their [Github](https://github.com/facebookresearch/fastText/) and [website](https://fasttext.cc/).
# 
# To do this, we won't be using the classic Keras embedding layer and instead hand-craft the embedding for each example. As a result, we need to  write more code and invest some time into preprocessing, but that is easily justified by the results.
# 
# **Disclaimer: Loading the FastText model will take some serious memory! I recommend having at least 60 GB of RAM. EC2's p2.xlarge instance should have no problems with this, but you can always [add some swap](https://stackoverflow.com/questions/17173972/how-do-you-add-swap-to-an-ec2-instance) for good measure. I also added a section below to build a training generator for this.**

# ## Preparations: Getting FastText and the model
# 
# First, build FastText from sources as described [here](https://github.com/facebookresearch/fastText#requirements). Don't worry, there's nothing crazy you have to do and it will finish in less than a minute. Next, install the Python package in your virtualenv following [these instructions](https://github.com/facebookresearch/fastText/tree/master/python).
# 
# For the model, I use the one pretrained on English Wikipedia. I'd love to have one trained on Twitter or similar, since it might be more internet-slangy, but I haven't found any yet and don't feel like pretraining one myself. Download the model [here](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md). Make sure you get the bin, not just the vec (text) file. I'll assume you placed it (or a symlink to it) into your code directory and named it `ft_model.bin`.

# ## Preparations: Exploring the model
# 
# Let's explore the model! Go to your FastText directory and run `./fasttext nn <path_to_ft_model>`. Now you can enter some terms and see the nearest neighbors to this word in the embedding space. Here are some examples:
# 
# ```
# Query word? queen
# —queen 0.719091
# ‘queen 0.692849
# #queen 0.656498
# queena 0.650313
# king 0.64931
# queen`s 0.63954
# king/queen 0.634855
# s/queen 0.627386
# princess 0.623889
# queeny 0.620919
# ```
# 
# Ok that looks pretty ugly. I suppose Facebook was not very exact in their cleaning of the input data. But some sensible suggestions are there: `king` and `princess`! Let's try a typo that is unlikely to have appeared in the original data:
# 
# ```
# Query word? dimensionnallity
# dimension, 0.722278
# dimensionality 0.708645
# dimensionful 0.698573
# codimension 0.689754
# codimensions 0.67555
# twodimensional 0.674745
# dimension 0.67258
# \,kdimensional 0.668848
# ‘dimensions 0.665725
# two–dimensional 0.665109
# ```
# 
# Sweet! Even though it has never seen that word, it recognizes it to be related with "dimensionality". Let's try some something mean:
# 
# ```
# Query word? dumb
# stupid 0.746051
# dumber 0.732965
# clueless 0.662594
# idiotic 0.64993
# silly 0.632314
# stupidstitious 0.628875
# stupidly 0.622968
# moronic 0.621633
# ignorant 0.620475
# stupider 0.617377
# ```
# 
# Nice! Even though this was trained on Wikipedia, we're getting at least some basic insults. I'll leave it to you to explore the really hateful words. They all seem to be there ;)
# 
# **Note:** Keep in mind that exploring the nearest neighbors is a very superficial approach to understanding the model! The embedding space has 300 dimensions, and we boil them down to a single distance metric. We can't be sure in which dimensions these words are related to each other, but we can trust in the model to have learnt something sensible.
# 
# **Pro tip:** Our data should be cleaned and normalized in a similar way as Facebook did before they trained this model. We can query the model to get some insights into what they did, e.g.
# 
# ```
# Query word? 1
# insel 0.483141
# inseln 0.401125
# ...
# Query word? one
# two 0.692744
# three 0.676568
# ...
# ```
# 
# This tells us they converted all numbers to their text equivalent, and so should we!

# ## Loading and cleaning the data
# 
# We define a method `normalize` to clean and prepare a single string. We will use it later to prepare our string data. Also, we load the data as we're used to:

# In[ ]:


import re
import numpy as np
import pandas as pd
from fastText import load_model

window_length = 200 # The amount of words we look at per example. Experiment with this.

def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s

print('\nLoading data')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['comment_text'] = train['comment_text'].fillna('_empty_')
test['comment_text'] = test['comment_text'].fillna('_empty_')


# Ok next, let's load the FastText model and define methods that convert text to a sequence of vectors. Note that I'm just considering the last n words of each text. You could play with this, too.

# In[ ]:


classes = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]

print('\nLoading FT model')
ft_model = load_model('ft_model.bin')
n_features = ft_model.get_dimension()

def text_to_vector(text):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = normalize(text)
    words = text.split()
    window = words[-window_length:]
    
    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x

def df_to_data(df):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in enumerate(df['comment_text'].values):
        x[i, :] = text_to_vector(comment)

    return x


# To convert an input dataframe to an input vector, just call `df_to_data`. This will result in the shape `(n_examples, window_length, n_features)`. Here, for each row we would have 200 words a 300 features each.
# 
# **EDIT/NOTE:** This will probably not fit into your memory, so don't bother executing it :) Instead, read my generator guide below.

# In[ ]:


x_train = df_to_data(train)
y_train = train[classes].values

x_test = df_to_data(test)
y_test = test[classes].values


# And now you should be good to go! Train this as usual. You don't need an `EmbeddingLayer`, but you need to pass `input_shape=(window_length, n_features)` to the first layer in your NN.
# 
# I'm still in the process of experimenting, but I already achieved a single-model LB score of `0.9842` with something very simple. Bagging multiple of these models got me into the top 100 easily. Good luck!

# ### PS: Using a generator so you don't have to keep the whole damn thing in memory
# As @liujilong pointed out, not even the p2.xlarge machine with 64 GB can hold both the training and test set for window sizes longer than ~100 words. It seems I underestimated how much memory this monster model eats! Also, locally I had long [added swap space](https://stackoverflow.com/questions/17173972/how-do-you-add-swap-to-an-ec2-instance) and switched to generators so I wouldn't have to keep the whole thing memory. Let me show you how to implement the generator part. This is also useful to add some randomization later on.
# 
# The idea is that instead of converting the whole training set to one large array, we can write a function that just spits out one batch of data at a time, infinitely. Keras can automaticaly spin up a separate thread for this method (note though that "threads" in Python are ridiculous and do not give any speedup whatsoever). This means that we have to write some more code and training will be slightly slower, but we need only a fraction of the memory and we can add some cool randomization to each batch later on (see ideas section below).
# 
# We can keep all the code from above. This generator method works only for training data, not for validation data, so you will need to split by hand. Let's do that now:

# In[ ]:


# Split the dataset:
split_index = round(len(train) * 0.9)
shuffled_train = train.sample(frac=1)
df_train = shuffled_train.iloc[:split_index]
df_val = shuffled_train.iloc[split_index:]

# Convert validation set to fixed array
x_val = df_to_data(df_val)
y_val = df_val[classes].values

def data_generator(df, batch_size):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    batch_i = 0 # Counter inside the current batch vector
    batch_x = None # The current batch's x data
    batch_y = None # The current batch's y data
    
    while True: # Loop forever
        df = df.sample(frac=1) # Shuffle df each epoch
        
        for i, row in df.iterrows():
            comment = row['comment_text']
            
            if batch_x is None:
                batch_x = np.zeros((batch_size, window_length, n_features), dtype='float32')
                batch_y = np.zeros((batch_size, len(classes)), dtype='float32')
                
            batch_x[batch_i] = text_to_vector(comment)
            batch_y[batch_i] = row[classes].values
            batch_i += 1

            if batch_i == batch_size:
                # Ready to yield the batch
                yield batch_x, batch_y
                batch_x = None
                batch_y = None
                batch_i = 0


# Alright, now we can use this generator to train the network. To make sure that one epoch has approxamitely the same number of examples as are in the training set, we need to set the `steps_per_epoch` to the number of batches we expect to cover the whole dataset. Here's the code:

# In[ ]:


model = build_model()  # TODO: Implement

batch_size = 128
training_steps_per_epoch = round(len(df_train) / batch_size)
training_generator = data_generator(df_train, batch_size)

# Ready to start training:
model.fit_generator(
    training_generator,
    steps_per_epoch=training_steps_per_epoch,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    ...
)


# And there you go, this should work on p2.xlarge even for long window lengths!

# ### More stuff to try:
# Some suggestions. I've tried most of these and found them helpful:
# 
# * Add random but common typos to strings before converting to FT vectors. That way, the model can learn in which way typos affect the embeddings. Use the training generator so you can adjust this over time.
# * Add more string preprocessing to our `normalize` function
# * Randomize the windows instead of using the end (great that we already have a generator!)
# * Use FastText's sentence vector feature to summarize parts of the text outside the window
# * Add other features ontop of the FT ones, e.g. capitalization etc.
