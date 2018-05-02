
# coding: utf-8

# # <center>Beginner's Guide to Capsule Networks</center>
# 
# _Author: Zafar_
# 
# _Last Updated: 03/30/18_
# 
# --------
# 
# In the recently concluded [Toxic Comments Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), Capsule Network (aka CapsNet) proved to be a huge success. This notebook introduces and implements a Capsule Network in Keras and evaluates its performance in the DonorsChoose.Org Application Screening Competition.
# 
# ## Contents
# 1. [Introduction to Capsule Networks](#introduction)
#     * 1.1 [Human Visual Recognition](#human)
#     * 1.2 [Capsules](#capsules)
#     * 1.3 [Routing by Agreement](#routing)
#     * 1.4 [Mathematics behind CapsNet](#maths)
#     * 1.5 [The Dyanmic Routing Algorithm](#algo)
#     * 1.6 [A word about squash function](#squash)
#     * 1.7 [The advantage of Capsule Networks](#advantage)
# 2. [Boilerplate Code](#boilerplate)
# 3. [CapsNet implementation](#capsnet_model)
# 4. [Training](#training)
# 5. [Submission](#submission)
# 6. [Conclusion](#conclusion)
# 7. [References](#references)
# 
# Before you read further, it is important to note that the examples presented here are just for the purpose of understanding.

# <a id="introduction"></a>
# ## Introduction to Capsule Networks
# <a id="human"></a>
# #### Human Visual Recognition
# Any real object is made up of several smaller objects. For example, a tree consists of a *trunk*, a *crown* and *roots*. These parts form a hierarchy. The crown of a tree further consists of branches and branches have leaves.
# 
# ![Parts of a tree](https://study.com/cimages/multimages/16/tree_parts_diagram.png)
# 
# Whenever we see some object, our eyes make some **fixation points** and the relative positions and natures of these fixation points help our brain in recognizing that object. By doing so, our brain does not have to process every detail. Just by seeing some leaves and branches, our brain recognizes there is a crown of a tree. And the crown is standing on a trunk below which are some roots. Combining this hierarchical information, our brain knows that there is a tree. From now on, we will call **the parts of the objects** as entities.
# 
# ![Parts of a tree](https://raw.githubusercontent.com/zaffnet/images/master/images/tree.png)
# 
# >*Each complex object can be thought of a hierarchy of simpler objects.*
# <a id="capsules"></a>
# #### Capsules
# 
# The assumption behind CapsNet is that there are capsules (groups of neurons) that tell whether certain objects (**entities**) are present in an image. Corresponding to each entity, there is a capsule which gives:
# 1. the probability that the entity exists
# 2. The **instantiation parameters** of that entity.
# 
# Instantiation parameters are the properties of that entity in an image (like "position", "size", "position", "hue", etc). For example, a **rectangle** is a simple geometric object. The capsule corresponding to a rectangle will tell us about its instantiation parameters. 
# 
# ![Rectangle capsule](https://raw.githubusercontent.com/zaffnet/images/master/images/rectangle.png)
# 
# 
# From the figure above, our imaginary capsule consists of 6 neurons each corresponding to some property of the rectangle. The length of this vector will give us the probability of the presence of a rectangle. So, the probability that a rectangle is present will be: $$\sqrt[]{1.3^2 + 0.6^2 + 7.4^2 + 6.5^2 + 0.5^2 + 1.4^2} = 10.06$$
# 
# But wait a minute! If the length of the output vector represents the probability of the existence of an entity, shouldn't it be less than or equal to 1 (i.e., $0 \leq P \leq 1$)? Yes, and that is why we transform the capsule output $s$ like this:
# 
# ![squashing function](https://raw.githubusercontent.com/zaffnet/images/master/images/squash.png)
# 
# 
# This non-linear transformation is called **squashing** function and it serves as an activation function for capsule networks (just like ReLU is used in CNNs).
# 
# >*A capsule is a group of neurons whose activation $v = <v_1, v_2, ..., v_n>$ represents the instantiation parameters of an entity and whose length represents the probability of the existence of that entity.*
# <a id="routing"></a>
# #### Routing by agreement
# A CapsNet consists of several layers. Capsules in the lower layer correspond to simple entities (like rectangles, triangles, circles, etc). These low-level capsules bet on the presence of more complex entities and their bets are "combined" to get the output of high-level capsules (doors, windows, etc). For example, the presence of a *rectangle* (angle with x-axis = 0, size = 5, position = 0,...) and a *triangle* (angle with x-axis = 6, size = 5, position = 5,...) work together to bet on the presence of a *house* (a higher-level entity). 
# 
# There is a **coupling effect** too. When some low-level capsules agree on the presence of a high-level entity, the high-level capsule corresponding to that entity sends a feedback to these low-level capsules which *increases* their bet on that high-level capsule. To understand this, let's assume we have two levels of capsules: 
# 1. Lower level corresponds to rectangles, triangles and circles
# 2. High level corresponds to houses, boats, and cars
# 
# If there is an image of a house, the capsules corresponding to rectangles and triangles will have large activation vectors. Their relative positions (coded in their instantiation parameters) will bet on the presence of high-level objects. Since they will agree on the presence of house, the output vector of the house capsule will become large. This, in turn, will make the predictions by the rectangle and the traingle capsules larger. This cycle will repeat 4-5 times after which the bets on the presence of a house will be considerably larger than the bets on the presence of a boat or a car.
# <a id="maths"></a>
# #### Mathematics behind CapsNet
# Suppose layer $l$ and $l+1$ have $m$ and $n$ capsules respectively. Our task is to calculate the activations of the capsules at layer $l+1$ given the activations at layer $l$. Let $u$ denotes the activations of capsules at layer $l$. We have to calculate $v$, the activations of capsules at layer $l+1$. 
# 
# For a capsule $j$ at layer $l+1$, 
# 
# 1. We first calculate the **prediction vectors** by the capsules at layer $l$. The prediction vector by a capsule $i$ (of layer $l$) for the capsule $j$ (of layer $l+1$) is given by:
#     $$\boldsymbol{\hat{\textbf{u}}}_{j|i} = \boldsymbol{\textbf{W}}_{ij}\boldsymbol{\textbf{u}}_i$$ $\textbf{W}_{ij}$ is the weight matrix.
# 
# 2. We then calculate the **output vector** for the capsule $j$. The output vector is the weighted sum of all the prediction vectors given by the capsules of layer $l$ for the capsule $j$:
#     $$s_j = \sum_{i=1}^{m}{c_{ij}\boldsymbol{\hat{\textbf{u}}}_{j|i}}$$ The scalar $\textbf{c}_{ij}$ is called **coupling coefficient** between capsule $i$ (of layer $l$) and $j$ (of layer $l+1$). These coefficients are determied by an algorithm called the **iterative dynamic routing algorithm**.
# 
# 3. We apply the **squashing** function on the output vector to get the activation $\textbf{v}_j$ of the capsule $j$:
#     $$\textbf{v}_j = \textbf{squash}(\textbf{s}_j)$$
#     
# <a id="algo"></a>
# #### The dynamic routing algorithm
# The activation vectors of layer $l+1$ send feedback signals to the capsules at layer $l$. If the prediction vector of capsule $i$ (of layer $l$) for a capsule $j$ (of layer $l+1$) is in agreement with the activation vector of capsule $j$, their dot product should be high. Hence the "weight" of the prediction vector $\boldsymbol{\hat{\textbf{u}}}_{j|i}$ is increased in the output vector of $j$. In other words, those prediction vectors that helped the activation vector have a lot more weight in the output vector (and consequently the activation vector). This cycle of mutual help continues for 4-5 rounds. 
# 
# But the predictions of a low-level capsule for high-level capsules should sum to one. That is why for a capsule $i$ (of layer $l$), 
# $$c_{ij} = \frac{\exp(b_{ij})}{\sum_{k}{\exp(b_{ik})}}$$ Clearly, $$\sum_{k}{c_{ik}} = 1$$ The logit $b_{ij}$ indicates whether capsules $i$ (of layer $l$) and $j$ (of layer $l+1$) have strong coupling. In other words, it is a measure of how much the presence of the capsule $j$ is explained by the capsule $i$. Initially, all $b_{ij}$ should be equal.
# 
# **Routing algorithm:**
# >Given: Prediction vectors $\boldsymbol{\hat{\textbf{u}}}_{j|i}$, number of routing iterations $r$
# 
# >for all capsule $i$ in layer $l$ and capsule $j$ in layer $l+1$: $b_{ij} = 0$ 
# 
# >for $r$ iterations do:
# 
# >>for all capsules $i$ in the layer $l$: $c_i = softmax(b_i)$ 
# >>**(the bets of a capsule on high-level capsules should sum to 1)**
# 
# >>for all capsules $j$ in the layer $l+1$: $s_j = \sum_{i=1}^{m}{c_{ij}\boldsymbol{\hat{\textbf{u}}}_{j|i}}$
# >>**(the output vector is the weighted sum of prediction vectors)**
# 
# >>for all capsules $j$ in the layer $l+1$: $\textbf{v}_j = \textbf{squash}(\textbf{s}_j)$
# >>**(apply the activation function)**
# 
# >> for all capsule $i$ in layer $l$ and capsule $j$ in layer $l+1$: $b_{ij} = b_{ij} + \boldsymbol{\hat{\textbf{u}}}_{j|i} \cdot \textbf{v}_j$
# 
# > return $\textbf{v}_j$
# 
# The last line in the loop is very important. It is here that the routing happens. If the product has $\boldsymbol{\hat{\textbf{u}}}_{j|i} \cdot \textbf{v}_j$ is large, it will increase $b_{ij}$ which will increase the corresponding coupling coefficient $c_{ij}$, which in turn, will make the product $\boldsymbol{\hat{\textbf{u}}}_{j|i} \cdot \textbf{v}_j$ even larger.
# 
# This is how CapsNet works. At this point, you will find no difficulty in reading the [original paper](https://arxiv.org/pdf/1710.09829.pdf) by Hinton.
# <a id="squash"></a>
# #### A word about squashing function:
# The derivative of $\|\mathbf{s}\|$ is undefined when $\|\mathbf{s}\|=0$, and it may blow up during training: if a vector is zero, the gradients will be `nan`, so when the optimizer updates the variables, they will also become `nan`. The solution is to implement the norm manually by computing the square root of the sum of squares plus a tiny epsilon value: $\|\mathbf{s}\| \approx \sqrt{\sum\limits_i{{s_i}^2}\,\,+ \epsilon}$.
# <a id="advantage"></a>
# #### What is the advantage?
# In a CNN, there are pooling layers. We generally use MaxPool which is a very primitive type of routing mechanism. The most active feature in a local pool (say 4x4 grid) is routed to the higher layer and the higher-level detectors don't have a say in the routing. Compare this with the routing-by-agreement mechanism introduced in the CapsNet. Only those features that agree with high-level detectors are routed. This is the advantage of CapsNet over CNN. It has a superior dynamic routing mechanism (dynamic because the information to be routed is determined in real time).

# <a id="boilerplate"></a>
# ## Boilerplate Code
# #### Essential imports

# In[ ]:


import gc
import os
import nltk
import tqdm
import numpy as np
import pandas as pd
nltk.download("punkt")


# In[ ]:


def tokenize_sentences(sentences, words_dict):
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict


# In[ ]:


def read_embedding_list(file_path):
    embedding_word_dict = {}
    embedding_list = []
    f = open(file_path)

    for index, line in enumerate(f):
        if index == 0:
            continue
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            continue
        embedding_list.append(coefs)
        embedding_word_dict[word] = len(embedding_word_dict)
    f.close()
    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


# In[ ]:


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


# In[ ]:


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))
        words_train.append(current_words)
    return words_train


# <a id="capsnet_model"></a>
# ### Capsule Network Model
# The Architecture of our CapsNet is very similar to general architecture, except for an addition Capsule Layer.
# 
# ![Text Classification](https://raw.githubusercontent.com/zaffnet/images/master/images/comparison.jpg)
# 
# 
# #### Advantage of Capsule Layer in Text Classification
# As you can see, we have used Capsule layer instead of Pooling layer. Capsule Layer eliminates the need for forced pooling layers like MaxPool. In many cases, this is desired because we get translational invariance without losing minute details.

# In[ ]:


from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.engine import Layer
from keras.layers import Activation, Add, Bidirectional, Conv1D, Dense, Dropout, Embedding, Flatten
from keras.layers import concatenate, GRU, Input, K, LSTM, MaxPooling1D
from keras.layers import GlobalAveragePooling1D,  GlobalMaxPooling1D, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks


# #### CapsNet parameters

# In[ ]:


gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.3
rate_drop_dense = 0.3


# In[ ]:


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# #### Capsule Layer

# In[ ]:


class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


# In[ ]:


def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input1 = Input(shape=(sequence_length,))
    embed_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(1, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


# In[ ]:


def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    num_labels = train_y.shape[1]
    patience = 5
    best_loss = -1
    best_weights = None
    best_epoch = 0
    
    current_epoch = 0
    
    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict(val_x, batch_size=batch_size)

        total_loss = 0
        for j in range(num_labels):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= num_labels

        print("Epoch {0} loss {1} best_loss {2}".format(current_epoch, total_loss, best_loss))

        current_epoch += 1
        if total_loss < best_loss or best_loss == -1:
            best_loss = total_loss
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == patience:
                break

    model.set_weights(best_weights)
    return model


# In[ ]:


def train_folds(X, y, X_test, fold_count, batch_size, get_model_func):
    print("="*75)
    fold_size = len(X) // fold_count
    models = []
    result_path = "predictions"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = np.array(X[fold_start:fold_end])
        val_y = np.array(y[fold_start:fold_end])

        model = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y)
        train_predicts_path = os.path.join(result_path, "train_predicts{0}.npy".format(fold_id))
        test_predicts_path = os.path.join(result_path, "test_predicts{0}.npy".format(fold_id))
        train_predicts = model.predict(X, batch_size=512, verbose=1)
        test_predicts = model.predict(X_test, batch_size=512, verbose=1)
        np.save(train_predicts_path, train_predicts)
        np.save(test_predicts_path, test_predicts)

    return models


# <a id="training"></a>
# ### Training
# 
# #### IMPORTANT
# Due to time limit in Kaggle kernels, I have restricted the model size and trained it on a small part of the  dataset. The commented values are those for which this model is trained.
# 
# 

# In[ ]:


# train_file_path = "../input/donorschooseorg-preprocessed-data/train_preprocessed.csv"
train_file_path = "../input/donorschooseorg-preprocessed-data/train_small.csv"

# test_file_path = "../input/donorschooseorg-preprocessed-data/test_preprocessed.csv"
test_file_path = "../input/donorschooseorg-preprocessed-data/test_small.csv"

# embedding_path = "../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec"
embedding_path = "../input/donorschooseorg-preprocessed-data/embeddings_small.vec"

batch_size = 128 # 256
recurrent_units = 16 # 64
dropout_rate = 0.3 
dense_size = 8 # 32
sentences_length = 10 # 300
fold_count = 2 # 10


# In[ ]:


UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"
CLASSES = ["project_is_approved"]


# In[ ]:


# Load data
print("Loading data...")
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)
list_sentences_train = train_data["application_text"].fillna(NAN_WORD).values
list_sentences_test = test_data["application_text"].fillna(NAN_WORD).values
y_train = train_data[CLASSES].values


# In[ ]:


print("Tokenizing sentences in train set...")
tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})
print("Tokenizing sentences in test set...")
tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)


# In[ ]:


# Embedding
words_dict[UNKNOWN_WORD] = len(words_dict)
print("Loading embeddings...")
embedding_list, embedding_word_dict = read_embedding_list(embedding_path)
embedding_size = len(embedding_list[0])


# In[ ]:


print("Preparing data...")
embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
embedding_list.append([0.] * embedding_size)
embedding_word_dict[END_WORD] = len(embedding_word_dict)
embedding_list.append([-1.] * embedding_size)

embedding_matrix = np.array(embedding_list)

id_to_word = dict((id, word) for word, id in words_dict.items())
train_list_of_token_ids = convert_tokens_to_ids(
    tokenized_sentences_train,
    id_to_word,
    embedding_word_dict,
    sentences_length)
test_list_of_token_ids = convert_tokens_to_ids(
    tokenized_sentences_test,
    id_to_word,
    embedding_word_dict,
    sentences_length)
X_train = np.array(train_list_of_token_ids)
X_test = np.array(test_list_of_token_ids)


# In[ ]:


get_model_func = lambda: get_model(
    embedding_matrix,
    sentences_length,
    dropout_rate,
    recurrent_units,
    dense_size)


# In[ ]:


del train_data, test_data, list_sentences_train, list_sentences_test
del tokenized_sentences_train, tokenized_sentences_test, words_dict
del embedding_list, embedding_word_dict
del train_list_of_token_ids, test_list_of_token_ids
gc.collect();


# In[ ]:


print("Starting to train models...")
models = train_folds(X_train, y_train, X_test, fold_count, batch_size, get_model_func)


# <a id="submission"></a>
# ### Submission
# 
# We trained the model for 10 folds using default parameters. We will make a rank-averaged submission.

# In[ ]:


from scipy.stats import rankdata

LABELS = ["project_is_approved"]

base = "../input/donorschooseorg-application-screening-predictions/predictions/predictions/"
predict_list = []
for j in range(10):
    predict_list.append(np.load(base + "predictions_001/test_predicts%d.npy"%j))
    
print("Rank averaging on ", len(predict_list), " files")
predcitions = np.zeros_like(predict_list[0])
for predict in predict_list:
    predcitions = np.add(predcitions.flatten(), rankdata(predict)/predcitions.shape[0])  
predcitions /= len(predict_list)

submission = pd.read_csv('../input/donorschoose-application-screening/sample_submission.csv')
submission[LABELS] = predcitions
submission.to_csv('submission.csv', index=False)


# <a id="conclusion"></a>
# ### Conclusion
# We can see that a Capsule Network can be helpful in Text Classification. Even without any hyperparameter tuning, one can build a strong baseline using CapsNet. 
# <a id="references"></a>
# ### References
# * [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
# * [Understanding Hinton’s Capsule Networks](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b)
# * [Capsule Networks (CapsNets) – Tutorial](https://www.youtube.com/watch?v=pPN8d0E3900)

# In[ ]:


from IPython.lib.display import YouTubeVideo
YouTubeVideo('pPN8d0E3900', width=800, height=450)

