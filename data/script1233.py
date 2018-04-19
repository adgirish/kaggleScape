
# coding: utf-8

# In[ ]:


import numpy as np
np.random.seed(666)
import pandas as pd
from sklearn.cross_validation import train_test_split
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#LOAD DATA

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv("../input/sample_submission.csv")
sample.head(2)


# In[ ]:


#CREATE TARGET VARIABLE
train["EAP"] = (train.author=="EAP")*1
train["HPL"] = (train.author=="HPL")*1
train["MWS"] = (train.author=="MWS")*1
train.drop("author", 1, inplace=True)
target_vars = ["EAP", "HPL", "MWS"]
train.head(2)


# In[ ]:


#STEMMING WORDS
import nltk.stem as stm
import re
stemmer = stm.SnowballStemmer("english")
train["stem_text"] = train.text.apply(lambda x: (" ").join([stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]"," ", x).split(" ")]))
test["stem_text"] = test.text.apply(lambda x: (" ").join([stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]"," ", x).split(" ")]))
train.head(3)


# In[ ]:


#PROCESS TEXT: RAW
from keras.preprocessing.text import Tokenizer
tok_raw = Tokenizer()
tok_raw.fit_on_texts(train.text.str.lower())
tok_stem = Tokenizer()
tok_stem.fit_on_texts(train.stem_text)
train["seq_text_stem"] = tok_stem.texts_to_sequences(train.stem_text)
test["seq_text_stem"] = tok_stem.texts_to_sequences(test.stem_text)
train.head(3)


# In[ ]:


#EXTRACT DATA FOR KERAS MODEL
from keras.preprocessing.sequence import pad_sequences
def get_keras_data(dataset, maxlen=20):
    X = {
        "stem_input": pad_sequences(dataset.seq_text_stem, maxlen=maxlen)
    }
    return X


maxlen = 60
dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.85)
X_train = get_keras_data(dtrain, maxlen)
y_train = np.array(dtrain[target_vars])
X_valid = get_keras_data(dvalid, maxlen)
y_valid = np.array(dvalid[target_vars])
X_test = get_keras_data(test, maxlen)

n_stem_seq = np.max( [np.max(X_valid["stem_input"]), np.max(X_train["stem_input"])])+1


# In[ ]:


import keras.backend as K
import tensorflow as tf
from keras import initializers, layers

class CapsuleLayer(Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_vector] and output shape = \
    [None, num_capsule, dim_vector]. For Dense Layer, input_dim_vector = dim_vector = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_vector: dimension of the output vectors of the capsules in this layer
    :param num_routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
                                 initializer=self.kernel_initializer,
                                 name='W')

        # Coupling coefficient. The redundant dimensions are just to facilitate subsequent matrix calculation.
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_vector]
        # Expand dims to [None, input_num_capsule, 1, 1, input_dim_vector]
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now it has shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        """  
        # Compute `inputs * W` by expanding the first dim of W. More time-consuming and need batch_size.
        # Now W has shape  = [batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector]
        w_tiled = K.tile(K.expand_dims(self.W, 0), [self.batch_size, 1, 1, 1, 1])
        
        # Transformed vectors, inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = K.batch_dot(inputs_tiled, w_tiled, [4, 3])
        """
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0. This is faster but requires Tensorflow.
        # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
        """
        # Routing algorithm V1. Use tf.while_loop in a dynamic way.
        def body(i, b, outputs):
            c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))
            b = b + K.sum(inputs_hat * outputs, -1, keepdims=True)
            return [i-1, b, outputs]
        cond = lambda i, b, inputs_hat: i > 0
        loop_vars = [K.constant(self.num_routing), self.bias, K.sum(inputs_hat, 1, keepdims=True)]
        _, _, outputs = tf.while_loop(cond, body, loop_vars)
        """
        # Routing algorithm V2. Use iteration. V2 and V1 both work without much difference on performance
        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            # outputs.shape=[None, 1, num_capsule, 1, dim_vector]
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            # last iteration needs not compute bias which will not be passed to the graph any more anyway.
            if i != self.num_routing - 1:
                # self.bias = K.update_add(self.bias, K.sum(inputs_hat * outputs, [0, -1], keepdims=True))
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
            # tf.summary.histogram('BigBee', self.bias)  # for debugging
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])
    
class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
    Output shape: [None, d2]
    """
    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of vectors of capsules
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])


# In[ ]:


#KERAS MODEL DEFINITION
from keras.layers import Dense, Dropout, Embedding
from keras.layers import Flatten, Input, SpatialDropout1D, Reshape
from keras.models import Model
from keras.optimizers import Adam 

def get_model():
    embed_dim = 50
    dropout_rate = 0.9
    emb_dropout_rate = 0.9
   
    input_text = Input(shape=[maxlen], name="stem_input")
    
    emb_lstm = SpatialDropout1D(emb_dropout_rate) (Embedding(n_stem_seq, embed_dim
                                                ,input_length = maxlen
                                                               ) (input_text))
    dense = Dropout(dropout_rate) (Dense(1024) (Flatten() (emb_lstm)))
    dense = Reshape((128, 8)) (dense)
    dense = Flatten() (Mask()(CapsuleLayer(128, 8))(dense))
    
    output = Dense(3, activation="softmax")(dense)

    model = Model([input_text], output)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

model = get_model()
model.summary()
    


# In[ ]:


#TRAIN KERAS MODEL
model = get_model()
model.fit(X_train, y_train, epochs=27
          , validation_data=[X_valid, y_valid]
         , batch_size=1024)


# In[ ]:


#MODEL EVALUATION
from sklearn.metrics import log_loss

preds_train = model.predict(X_train)
preds_valid = model.predict(X_valid)

print(log_loss(y_train, preds_train))
print(log_loss(y_valid, preds_valid))


# In[ ]:


#PREDICTION
preds = pd.DataFrame(model.predict(X_test), columns=target_vars)
submission = pd.concat([test["id"],preds], 1)
submission.to_csv("./submission.csv", index=False)
submission.head()

