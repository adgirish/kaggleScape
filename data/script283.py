
# coding: utf-8

# # Description
# 
# This is an associated model using RNN and Ridge to solve Mercari Price Suggestion Challenge competition.
# 
# You can find description of the competition here https://www.kaggle.com/c/mercari-price-suggestion-challenge

# ## Import packages
# 
# Import all needed packages for constructing models and solving the competition

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K


# ## Define RMSL Error Function

# In[ ]:


def rmsle(Y, Y_pred):
    # Y and Y_red have already been in log scale.
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))


# ## Load train and test data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain_df = pd.read_table('../input/train.tsv')\ntest_df = pd.read_table('../input/test.tsv')\nprint(train_df.shape, test_df.shape)")


# ## Prepare data for processing by RNN and Ridge

# In[ ]:


# Handle missing data.
def fill_missing_values(df):
    df.category_name.fillna(value="Other", inplace=True)
    df.brand_name.fillna(value="missing", inplace=True)
    df.item_description.fillna(value="None", inplace=True)
    return df

train_df = fill_missing_values(train_df)
test_df = fill_missing_values(test_df)


# In[ ]:


# Scale target variable to log.
train_df["target"] = np.log1p(train_df.price)

# Split training examples into train/dev examples.
train_df, dev_df = train_test_split(train_df, random_state=347, train_size=0.99)

Y_train = train_df.target.values.reshape(-11, 1)
Y_dev = dev_df.target.values.reshape(-1, 1)

# Calculate number of train/dev/test examples.
n_trains = train_df.shape[0]
n_devs = dev_df.shape[0]
n_tests = test_df.shape[0]
print("Training on", n_trains, "examples")
print("Validating on", n_devs, "examples")
print("Testing on", n_tests, "examples")


# # RNN Model
# 
# This section will use RNN Model to solve the competition with following steps:
# 
# 1. Preprocessing data
# 1. Define RNN model
# 1. Fitting RNN model on training examples
# 1. Evaluating RNN model on dev examples
# 1. Make prediction for test data using RNN model

# In[ ]:


# Concatenate train - dev - test data for easy to handle
full_df = pd.concat([train_df, dev_df, test_df])


# ## Process categorical data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint("Processing categorical data...")\nle = LabelEncoder()\n\nle.fit(full_df.category_name)\nfull_df.category_name = le.transform(full_df.category_name)\n\nle.fit(full_df.brand_name)\nfull_df.brand_name = le.transform(full_df.brand_name)\n\ndel le')


# ## Process text data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint("Transforming text data to sequences...")\nraw_text = np.hstack([full_df.item_description.str.lower(), full_df.name.str.lower()])\n\nprint("   Fitting tokenizer...")\ntok_raw = Tokenizer()\ntok_raw.fit_on_texts(raw_text)\n\nprint("   Transforming text to sequences...")\nfull_df[\'seq_item_description\'] = tok_raw.texts_to_sequences(full_df.item_description.str.lower())\nfull_df[\'seq_name\'] = tok_raw.texts_to_sequences(full_df.name.str.lower())\n\ndel tok_raw')


# In[ ]:


# Define constants to use when define RNN model
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_TEXT = np.max([
    np.max(full_df.seq_name.max()),
    np.max(full_df.seq_item_description.max()),
]) + 4
MAX_CATEGORY = np.max(full_df.category_name.max()) + 1
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1


# ## Get data for RNN model

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef get_keras_data(df):\n    X = {\n        \'name\': pad_sequences(df.seq_name, maxlen=MAX_NAME_SEQ),\n        \'item_desc\': pad_sequences(df.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),\n        \'brand_name\': np.array(df.brand_name),\n        \'category_name\': np.array(df.category_name),\n        \'item_condition\': np.array(df.item_condition_id),\n        \'num_vars\': np.array(df[["shipping"]]),\n    }\n    return X\n\ntrain = full_df[:n_trains]\ndev = full_df[n_trains:n_trains+n_devs]\ntest = full_df[n_trains+n_devs:]\n\nX_train = get_keras_data(train)\nX_dev = get_keras_data(dev)\nX_test = get_keras_data(test)')


# ## Define RNN model

# In[ ]:


def new_rnn_model(lr=0.001, decay=0.0):    
    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_name = Embedding(MAX_TEXT, 20)(name)
    emb_item_desc = Embedding(MAX_TEXT, 60)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)

    # rnn layers
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)

    # main layers
    main_l = concatenate([
        Flatten() (emb_brand_name),
        Flatten() (emb_category_name),
        item_condition,
        rnn_layer1,
        rnn_layer2,
        num_vars,
    ])

    main_l = Dense(256)(main_l)
    main_l = Activation('elu')(main_l)

    main_l = Dense(128)(main_l)
    main_l = Activation('elu')(main_l)

    main_l = Dense(64)(main_l)
    main_l = Activation('elu')(main_l)

    # the output layer.
    output = Dense(1, activation="linear") (main_l)

    model = Model([name, item_desc, brand_name , category_name, item_condition, num_vars], output)

    optimizer = Adam(lr=lr, decay=decay)
    model.compile(loss="mse", optimizer=optimizer)

    return model

model = new_rnn_model()
model.summary()
del model


# ## Fit RNN model to train data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Set hyper parameters for the model.\nBATCH_SIZE = 1024\nepochs = 2\n\n# Calculate learning rate decay.\nexp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1\nsteps = int(n_trains / BATCH_SIZE) * epochs\nlr_init, lr_fin = 0.007, 0.0005\nlr_decay = exp_decay(lr_init, lr_fin, steps)\n\nrnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)\n\nprint("Fitting RNN model to training examples...")\nrnn_model.fit(\n        X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,\n        validation_data=(X_dev, Y_dev), verbose=2,\n)')


# ## Evaluate RNN model on dev data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint("Evaluating the model on validation data...")\nY_dev_preds_rnn = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)\nprint(" RMSLE error:", rmsle(Y_dev, Y_dev_preds_rnn))')


# ## Make prediction for test data

# In[ ]:


rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
rnn_preds = np.expm1(rnn_preds)


# # Ridge Model
# 
# This section will solve the competition using Ridge model with following steps:
# 
# 1. Preprocessing data
# 1. Fitting Ridge model on training examples
# 1. Evaluating Ridge model on dev examples
# 1. Make prediction for test data using Ridge model

# In[ ]:


# Concatenate train - dev - test data for easy to handle
full_df = pd.concat([train_df, dev_df, test_df])


# ## Convert data type to string

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Convert data type to string\nfull_df['shipping'] = full_df['shipping'].astype(str)\nfull_df['item_condition_id'] = full_df['item_condition_id'].astype(str)")


# ## Extract features from data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint("Vectorizing data...")\ndefault_preprocessor = CountVectorizer().build_preprocessor()\ndef build_preprocessor(field):\n    field_idx = list(full_df.columns).index(field)\n    return lambda x: default_preprocessor(x[field_idx])\n\nvectorizer = FeatureUnion([\n    (\'name\', CountVectorizer(\n        ngram_range=(1, 2),\n        max_features=50000,\n        preprocessor=build_preprocessor(\'name\'))),\n    (\'category_name\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'category_name\'))),\n    (\'brand_name\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'brand_name\'))),\n    (\'shipping\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'shipping\'))),\n    (\'item_condition_id\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'item_condition_id\'))),\n    (\'item_description\', TfidfVectorizer(\n        ngram_range=(1, 3),\n        max_features=100000,\n        preprocessor=build_preprocessor(\'item_description\'))),\n])\n\nX = vectorizer.fit_transform(full_df.values)\n\nX_train = X[:n_trains]\nX_dev = X[n_trains:n_trains+n_devs]\nX_test = X[n_trains+n_devs:]\n\nprint(X.shape, X_train.shape, X_dev.shape, X_test.shape)')


# ## Fitting Ridge model on training data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint("Fitting Ridge model on training examples...")\nridge_model = Ridge(\n    solver=\'auto\', fit_intercept=True, alpha=0.5,\n    max_iter=100, normalize=False, tol=0.05,\n)\nridge_model.fit(X_train, Y_train)')


# ## Evaluating Ridge model on dev data

# In[ ]:


Y_dev_preds_ridge = ridge_model.predict(X_dev)
Y_dev_preds_ridge = Y_dev_preds_ridge.reshape(-1, 1)
print("RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridge))


# ## Make prediction for test data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nridge_preds = ridge_model.predict(X_test)\nridge_preds = np.expm1(ridge_preds)')


# # Evaluating for associated model on dev data

# In[ ]:


def aggregate_predicts(Y1, Y2):
    assert Y1.shape == Y2.shape
    ratio = 0.63
    return Y1 * ratio + Y2 * (1.0 - ratio)

Y_dev_preds = aggregate_predicts(Y_dev_preds_rnn, Y_dev_preds_ridge)
print("RMSL error for RNN + Ridge on dev set:", rmsle(Y_dev, Y_dev_preds))


# # Creating Submission

# In[ ]:


preds = aggregate_predicts(rnn_preds, ridge_preds)
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
submission.to_csv("./rnn_ridge_submission.csv", index=False)


# # Something can be tried to improve the model
# 
# - Change aggregation ratio for aggregate_predicts
# - Change learning rate and learning rate decay RNN model
# - Descrease the batch size for RNN model
# - Increase the embedding output dimension for RNN model
# -  Add more Dense layers for RNN model
# - Add Batch Normalization layers for RNN model
# - Try LSTM, Bidirectional RNN, stack RNN for RNN model
# - Using other optimizer for RNN model
# - Change parameters for Ridge model
# - Something else that can help to improve the model

# # References
# 
# 1. https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl
# 1. https://www.kaggle.com/isaienkov/rnn-gru-with-keras-512-64-relu-0-43758
# 1. https://www.kaggle.com/lopuhin/eli5-for-mercari
