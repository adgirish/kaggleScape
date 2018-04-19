
# coding: utf-8

# # Description
# 
# This is an associated model using RNN ,Ridge, and a RidgeCV to solve Mercari Price Suggestion Challenge competition. It currently gets a RMSLE of ~0.419 in the development set and around ~0.427 in the competition, which is better than the currently public Kernals for the competition. I figure I don't have time to keep working on this for the next few weeks so I will simply share this model, along with some notes and thoughts, so that it can help others with their own models or function as a good jumping off point. After this kernal, I will probably bow out for the rest of the comp. If you find this kernal useful, please give me some of them sweet sweet kernel likes.
# 
# I left some old code attempts in comments as well as a few repeated lines. This is just more matterial to play with if they change the code around.
# 
# (Referenced Kernal links at the bottom.)

# ## Import packages
# 
# Import all needed packages for constructing models and solving the competition

# In[ ]:


from datetime import datetime 
start_real = datetime.now()
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
# from keras.layers import Bidirectional
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from nltk.corpus import stopwords
import math
# set seed
np.random.seed(123)


# ## Define RMSL Error Function
# This is for checking the predictions at the end. Note that the Y and Y_pred will already be in log scale by the time this is used, so no need to log them in the function.

# In[ ]:


def rmsle(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))


# ## Load train and test data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain_df = pd.read_table('../input/train.tsv')\ntest_df = pd.read_table('../input/test.tsv')\nprint(train_df.shape, test_df.shape)")


# ## Preprossing the data for RNN and Ridge models
# This is the preprocessing that can be done for both models at the same time

# Remove low prices, anything below 3. Mercari does not allow postings below 3 so below that is an error. Removing them helps the models.

# In[ ]:


# remove low prices
train_df = train_df.drop(train_df[(train_df.price < 3.0)].index)
train_df.shape


# Mercari also does not allow postings over 2000. Could get rid of those, but only 3 and they are only a few dollars higher so that is probably just shipping fees.

# In[ ]:


# train_df = train_df.drop(train_df[(train_df.price > 2000)].index)
# train_df.shape


# The following block removes stopwords. The models are robust against stopwords but it might help to remove them when looking for such small improvements with such a strict time limit. However, this did not seem to help enough to merit the 1-2 minutes it takes for this block to run.

# In[ ]:


# %%time

# stop = stopwords.words('english')
# train_df.item_description.fillna(value='No description yet', inplace=True)
# train_df['item_description'] = train_df['item_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
# train_df.name.fillna(value="missing", inplace=True)
# train_df['name'] = train_df['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# test_df.item_description.fillna(value='No description yet', inplace=True)
# test_df['item_description'] = test_df['item_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
# test_df.name.fillna(value="missing", inplace=True)
# test_df['name'] = test_df['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# train_df.head()


# The length of the description, that is the raw number of words used, does have some correlation with price. The RNN might find this out on it's own, but since a max depth is used to save computations, it does not always know. Description length clearly helps the model, name length maybe not so much. Does not hurt the models so leaving name length in.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# get name and description lengths\ndef wordCount(text):\n    try:\n        if text == \'No description yet\':\n            return 0\n        else:\n            text = text.lower()\n            words = [w for w in text.split(" ")]\n            return len(words)\n    except: \n        return 0\ntrain_df[\'desc_len\'] = train_df[\'item_description\'].apply(lambda x: wordCount(x))\ntest_df[\'desc_len\'] = test_df[\'item_description\'].apply(lambda x: wordCount(x))\ntrain_df[\'name_len\'] = train_df[\'name\'].apply(lambda x: wordCount(x))\ntest_df[\'name_len\'] = test_df[\'name\'].apply(lambda x: wordCount(x))\ntrain_df.head()')


# Here we split the category_name into 3 parts. Our models can get more information this way. I tried making a small 3 part RNN layer for this instead which does worse than this method *but* is occasionally faster.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# split category name into 3 parts\ndef split_cat(text):\n    try: return text.split("/")\n    except: return ("No Label", "No Label", "No Label")\ntrain_df[\'subcat_0\'], train_df[\'subcat_1\'], train_df[\'subcat_2\'] = \\\nzip(*train_df[\'category_name\'].apply(lambda x: split_cat(x)))\ntest_df[\'subcat_0\'], test_df[\'subcat_1\'], test_df[\'subcat_2\'] = \\\nzip(*test_df[\'category_name\'].apply(lambda x: split_cat(x)))')


# The brand name data is sparse, missing over 600,000 values. This gets *some* of those values back by checking their names. However, It does not seem to help the models either way at this point. An *exact* name match against all_brand names will find about 3000 of these. We can be pretty confident in these. At the other extreme, we can search for *any* matches throughout all words in name. This finds over 200,000 but a lot of these are incorrect. Can land somewhere in the middle by either keeping cases or trimming out some of the 5000 brand names.
# 
# For example, PINK is a brand by victoria secret. If we remove case, then almost all *pink* items are labeled as PINK brand. The other issue is that some of the "brand names" are not brands but really categories like "Boots" or "Keys". 
# 
# Currently, checking every word in name of a case-sensitive match does best. This gets around 137,000 finds while avoiding the problems with brands like PINK.

# In[ ]:


# %%time
# attempt to find missing brand names
# train_df['name'] = train_df.name.str.lower()
# train_df['brand_name'] = train_df.brand_name.str.lower()
# test_df['name'] = test_df.name.str.lower()
# test_df['brand_name'] = test_df.brand_name.str.lower()
full_set = pd.concat([train_df,test_df])
all_brands = set(full_set['brand_name'].values)
train_df.brand_name.fillna(value="missing", inplace=True)
test_df.brand_name.fillna(value="missing", inplace=True)

# get to finding!
premissing = len(train_df.loc[train_df['brand_name'] == 'missing'])
def brandfinder(line):
    brand = line[0]
    name = line[1]
    namesplit = name.split(' ')
    if brand == 'missing':
        for x in namesplit:
            if x in all_brands:
                return name
    if name in all_brands:
        return name
    return brand
train_df['brand_name'] = train_df[['brand_name','name']].apply(brandfinder, axis = 1)
test_df['brand_name'] = test_df[['brand_name','name']].apply(brandfinder, axis = 1)
found = premissing-len(train_df.loc[train_df['brand_name'] == 'missing'])
print(found)


# Standard split the train test for validation and log the price

# In[ ]:


# Scale target variable to log.
train_df["target"] = np.log1p(train_df.price)

# Split training examples into train/dev examples.
train_df, dev_df = train_test_split(train_df, random_state=123, train_size=0.99)

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


# ## Fill missing data
# Note that replacing 'No description yet' with "missing" helps the model a bit by treating it the same as the NA values

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Filling missing values\ndef fill_missing_values(df):\n    df.category_name.fillna(value="missing", inplace=True)\n    df.brand_name.fillna(value="missing", inplace=True)\n    df.item_description.fillna(value="missing", inplace=True)\n    df.item_description.replace(\'No description yet\',"missing", inplace=True)\n    return df\n\nprint("Filling missing data...")\nfull_df = fill_missing_values(full_df)\nprint(full_df.category_name[1])')


# ## Process categorical data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint("Processing categorical data...")\nle = LabelEncoder()\n# full_df.category = full_df.category_name\nle.fit(full_df.category_name)\nfull_df[\'category\'] = le.transform(full_df.category_name)\n\nle.fit(full_df.brand_name)\nfull_df.brand_name = le.transform(full_df.brand_name)\n\nle.fit(full_df.subcat_0)\nfull_df.subcat_0 = le.transform(full_df.subcat_0)\n\nle.fit(full_df.subcat_1)\nfull_df.subcat_1 = le.transform(full_df.subcat_1)\n\nle.fit(full_df.subcat_2)\nfull_df.subcat_2 = le.transform(full_df.subcat_2)\n\ndel le')


# ## Process text data
# From here til the end of the RNN model are some commented out code lines when I used a short RNN layer to process category_name. Using the 3 subcats makes a better model but can sometimes be *slightly* slower.

# In[ ]:


# %%time
# # Break category_name into parts
# def catgsub(col):
#     col = col.str.replace(' ','')
#     col = col.str.replace('/',' ')
#     col = col.str.replace('&','')
#     return col
# full_df['category_name'] = catgsub(full_df['category_name'])
# print(full_df.category_name[1])


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint("Transforming text data to sequences...")\nraw_text = np.hstack([full_df.item_description.str.lower(), full_df.name.str.lower(), full_df.category_name.str.lower()])\n\nprint("   Fitting tokenizer...")\ntok_raw = Tokenizer()\ntok_raw.fit_on_texts(raw_text)\n\nprint("   Transforming text to sequences...")\nfull_df[\'seq_item_description\'] = tok_raw.texts_to_sequences(full_df.item_description.str.lower())\nfull_df[\'seq_name\'] = tok_raw.texts_to_sequences(full_df.name.str.lower())\n# full_df[\'seq_category\'] = tok_raw.texts_to_sequences(full_df.category_name.str.lower())\n\ndel tok_raw')


# In[ ]:


full_df['seq_name'][:5]


# # Define constants to use when define RNN model
# Note the comments next to the first few lines indicate the longest entry in that column. Just for reference.

# In[ ]:


MAX_NAME_SEQ = 10 #17
MAX_ITEM_DESC_SEQ = 75 #269
MAX_CATEGORY_SEQ = 8 #8
MAX_TEXT = np.max([
    np.max(full_df.seq_name.max()),
    np.max(full_df.seq_item_description.max()),
#     np.max(full_df.seq_category.max()),
]) + 100
MAX_CATEGORY = np.max(full_df.category.max()) + 1
MAX_BRAND = np.max(full_df.brand_name.max()) + 1
MAX_CONDITION = np.max(full_df.item_condition_id.max()) + 1
MAX_DESC_LEN = np.max(full_df.desc_len.max()) + 1
MAX_NAME_LEN = np.max(full_df.name_len.max()) + 1
MAX_SUBCAT_0 = np.max(full_df.subcat_0.max()) + 1
MAX_SUBCAT_1 = np.max(full_df.subcat_1.max()) + 1
MAX_SUBCAT_2 = np.max(full_df.subcat_2.max()) + 1


# ## Get data for RNN model

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef get_rnn_data(dataset):\n    X = {\n        \'name\': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),\n        \'item_desc\': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),\n        \'brand_name\': np.array(dataset.brand_name),\n        \'category\': np.array(dataset.category),\n#         \'category_name\': pad_sequences(dataset.seq_category, maxlen=MAX_CATEGORY_SEQ),\n        \'item_condition\': np.array(dataset.item_condition_id),\n        \'num_vars\': np.array(dataset[["shipping"]]),\n        \'desc_len\': np.array(dataset[["desc_len"]]),\n        \'name_len\': np.array(dataset[["name_len"]]),\n        \'subcat_0\': np.array(dataset.subcat_0),\n        \'subcat_1\': np.array(dataset.subcat_1),\n        \'subcat_2\': np.array(dataset.subcat_2),\n    }\n    return X\n\ntrain = full_df[:n_trains]\ndev = full_df[n_trains:n_trains+n_devs]\ntest = full_df[n_trains+n_devs:]\n\nX_train = get_rnn_data(train)\nY_train = train.target.values.reshape(-1, 1)\n\nX_dev = get_rnn_data(dev)\nY_dev = dev.target.values.reshape(-1, 1)\n\nX_test = get_rnn_data(test)')


# Here are some unused RMSE and RMSLE functions. They can be used as a loss function in the model but the built in 'mse' works just as well. Worth having around just in case. There needs to be a small non-zero number added to the means or a zero value might sneak in and cause it to return NaN. As is, they work fine as a loss function just not special.

# In[ ]:


def root_mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1)+0.0000001)
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)+0.0000001)


# ## Define RNN model
# Now to build the model. Old category stuff is commented out but left in case of revist. (other adjustment notes in comments)

# In[ ]:


# set seed again in case testing models adjustments by looping next 2 blocks
np.random.seed(123)

def new_rnn_model(lr=0.001, decay=0.0):
    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
#     category = Input(shape=[1], name="category")
#     category_name = Input(shape=[X_train["category_name"].shape[1]], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    desc_len = Input(shape=[1], name="desc_len")
    name_len = Input(shape=[1], name="name_len")
    subcat_0 = Input(shape=[1], name="subcat_0")
    subcat_1 = Input(shape=[1], name="subcat_1")
    subcat_2 = Input(shape=[1], name="subcat_2")

    # Embeddings layers (adjust outputs to help model)
    emb_name = Embedding(MAX_TEXT, 20)(name)
    emb_item_desc = Embedding(MAX_TEXT, 60)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
#     emb_category_name = Embedding(MAX_TEXT, 20)(category_name)
#     emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    emb_desc_len = Embedding(MAX_DESC_LEN, 5)(desc_len)
    emb_name_len = Embedding(MAX_NAME_LEN, 5)(name_len)
    emb_subcat_0 = Embedding(MAX_SUBCAT_0, 10)(subcat_0)
    emb_subcat_1 = Embedding(MAX_SUBCAT_1, 10)(subcat_1)
    emb_subcat_2 = Embedding(MAX_SUBCAT_2, 10)(subcat_2)
    

    # rnn layers (GRUs are faster than LSTMs and speed is important here)
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)
#     rnn_layer3 = GRU(8) (emb_category_name)

    # main layers
    main_l = concatenate([
        Flatten() (emb_brand_name)
#         , Flatten() (emb_category)
        , Flatten() (emb_item_condition)
        , Flatten() (emb_desc_len)
        , Flatten() (emb_name_len)
        , Flatten() (emb_subcat_0)
        , Flatten() (emb_subcat_1)
        , Flatten() (emb_subcat_2)
        , rnn_layer1
        , rnn_layer2
#         , rnn_layer3
        , num_vars
    ])
    # (incressing the nodes or adding layers does not effect the time quite as much as the rnn layers)
    main_l = Dropout(0.1)(Dense(512,kernel_initializer='normal',activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(256,kernel_initializer='normal',activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(128,kernel_initializer='normal',activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(64,kernel_initializer='normal',activation='relu') (main_l))

    # the output layer.
    output = Dense(1, activation="linear") (main_l)
    
    model = Model([name, item_desc, brand_name , item_condition, 
                   num_vars, desc_len, name_len, subcat_0, subcat_1, subcat_2], output)

    optimizer = Adam(lr=lr, decay=decay)
    # (mean squared error loss function works as well as custom functions)  
    model.compile(loss = 'mse', optimizer = optimizer)

    return model

model = new_rnn_model()
model.summary()
del model


# ## Fit RNN model to train data
# This is where most of the time is spent. It takes around 35-40 minutes to run the RNN model. 2 epochs with smaller batches tends to do better than more epochs with larger batches. Trimming time off here will be important if adding more models.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Set hyper parameters for the model.\nBATCH_SIZE = 512 * 3\nepochs = 2\n\n# Calculate learning rate decay.\nexp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1\nsteps = int(len(X_train['name']) / BATCH_SIZE) * epochs\nlr_init, lr_fin = 0.005, 0.001\nlr_decay = exp_decay(lr_init, lr_fin, steps)\n\n# Create model and fit it with training dataset.\nrnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)\nrnn_model.fit(\n        X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,\n        validation_data=(X_dev, Y_dev), verbose=1,\n)")


# ## Evaluate RNN model on dev data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint("Evaluating the model on validation data...")\nY_dev_preds_rnn = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)\nprint(" RMSLE error:", rmsle(Y_dev, Y_dev_preds_rnn))')


# ## Make prediction for test data

# In[ ]:


rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
rnn_preds = np.expm1(rnn_preds)


# # Ridge Models
# 
# Now onto the Ridge models. Less to play with in the Ridge models but it is faster than the RNN. 

# In[ ]:


# Concatenate train - dev - test data for easy to handle
full_df = pd.concat([train_df, dev_df, test_df])


# ## Handle missing data and convert data type to string
# All inputs must be strings in a ridge model. The other note here is that filling NAs for item_description use 'No description yet' so it is read the same as the 'No description yet' entries.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint("Handling missing values...")\nfull_df[\'category_name\'] = full_df[\'category_name\'].fillna(\'missing\').astype(str)\nfull_df[\'subcat_0\'] = full_df[\'subcat_0\'].astype(str)\nfull_df[\'subcat_1\'] = full_df[\'subcat_1\'].astype(str)\nfull_df[\'subcat_2\'] = full_df[\'subcat_2\'].astype(str)\nfull_df[\'brand_name\'] = full_df[\'brand_name\'].fillna(\'missing\').astype(str)\nfull_df[\'shipping\'] = full_df[\'shipping\'].astype(str)\nfull_df[\'item_condition_id\'] = full_df[\'item_condition_id\'].astype(str)\nfull_df[\'desc_len\'] = full_df[\'desc_len\'].astype(str)\nfull_df[\'name_len\'] = full_df[\'name_len\'].astype(str)\nfull_df[\'item_description\'] = full_df[\'item_description\'].fillna(\'No description yet\').astype(str)')


# ## Vectorizing all the data
# Takes around 8-10 minutes depending on the inputs used.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint("Vectorizing data...")\ndefault_preprocessor = CountVectorizer().build_preprocessor()\ndef build_preprocessor(field):\n    field_idx = list(full_df.columns).index(field)\n    return lambda x: default_preprocessor(x[field_idx])\n\nvectorizer = FeatureUnion([\n    (\'name\', CountVectorizer(\n        ngram_range=(1, 2),\n        max_features=50000,\n        preprocessor=build_preprocessor(\'name\'))),\n#     (\'category_name\', CountVectorizer(\n#         token_pattern=\'.+\',\n#         preprocessor=build_preprocessor(\'category_name\'))),\n    (\'subcat_0\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'subcat_0\'))),\n    (\'subcat_1\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'subcat_1\'))),\n    (\'subcat_2\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'subcat_2\'))),\n    (\'brand_name\', CountVectorizer(\n        token_pattern=\'.+\',\n        preprocessor=build_preprocessor(\'brand_name\'))),\n    (\'shipping\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'shipping\'))),\n    (\'item_condition_id\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'item_condition_id\'))),\n    (\'desc_len\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'desc_len\'))),\n    (\'name_len\', CountVectorizer(\n        token_pattern=\'\\d+\',\n        preprocessor=build_preprocessor(\'name_len\'))),\n    (\'item_description\', TfidfVectorizer(\n        ngram_range=(1, 3),\n        max_features=100000,\n        preprocessor=build_preprocessor(\'item_description\'))),\n])\n\nX = vectorizer.fit_transform(full_df.values)\n\nX_train = X[:n_trains]\nY_train = train_df.target.values.reshape(-1, 1)\n\nX_dev = X[n_trains:n_trains+n_devs]\nY_dev = dev_df.target.values.reshape(-1, 1)\n\nX_test = X[n_trains+n_devs:]\nprint(X.shape, X_train.shape, X_dev.shape, X_test.shape)')


# ## Fitting Ridge model on training data
# A Ridge model with cross validation does *slighly* better than one without, but even with the minimum of 2 CV, it still takes 4-5 minutes. Any more and it becomes impractical for our narrow time limit. A regular ridge model will only take ~30 seconds. So, for the purposes of having another model to predict on, might as well make both.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint("Fitting Ridge model on training examples...")\nridge_model = Ridge(\n    solver=\'auto\', fit_intercept=True, alpha=1.0,\n    max_iter=100, normalize=False, tol=0.05, random_state = 1,\n)\nridge_modelCV = RidgeCV(\n    fit_intercept=True, alphas=[5.0],\n    normalize=False, cv = 2, scoring=\'neg_mean_squared_error\',\n)\nridge_model.fit(X_train, Y_train)\nridge_modelCV.fit(X_train, Y_train)')


# ## Evaluating Ridge model on dev data

# In[ ]:


Y_dev_preds_ridge = ridge_model.predict(X_dev)
Y_dev_preds_ridge = Y_dev_preds_ridge.reshape(-1, 1)
print("RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridge))


# In[ ]:


Y_dev_preds_ridgeCV = ridge_modelCV.predict(X_dev)
Y_dev_preds_ridgeCV = Y_dev_preds_ridgeCV.reshape(-1, 1)
print("CV RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridgeCV))


# ## Make prediction for test data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nridge_preds = ridge_model.predict(X_test)\nridge_preds = np.expm1(ridge_preds)\nridgeCV_preds = ridge_modelCV.predict(X_test)\nridgeCV_preds = np.expm1(ridgeCV_preds)')


# # Evaluating for associated model on dev data
# This combines the 3 predicts into one. Rather than take a simple average, aggregate predicts will use ratios to vary the weights of the 3 models. It also use a simple loop to run through all the possible ratios to find the best ratio on the dev set. It is not the most computationally efficient loop but it only takes 2 seconds to run so no big deal.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def aggregate_predicts3(Y1, Y2, Y3, ratio1, ratio2):\n    assert Y1.shape == Y2.shape\n    return Y1 * ratio1 + Y2 * ratio2 + Y3 * (1.0 - ratio1-ratio2)\n\n# Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_ridgeCV, Y_dev_preds_ridge, 0.4, 0.3)\n# print("RMSL error for RNN + Ridge + RidgeCV on dev set:", rmsle(Y_dev, Y_dev_preds))')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#ratio optimum finder for 3 models\nbest1 = 0\nbest2 = 0\nlowest = 0.99\nfor i in range(100):\n    for j in range(100):\n        r = i*0.01\n        r2 = j*0.01\n        if r+r2 < 1.0:\n            Y_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_ridgeCV, Y_dev_preds_ridge, r, r2)\n            fpred = rmsle(Y_dev, Y_dev_preds)\n            if fpred < lowest:\n                best1 = r\n                best2 = r2\n                lowest = fpred\n#             print(str(r)+"-RMSL error for RNN + Ridge + RidgeCV on dev set:", fpred)\nY_dev_preds = aggregate_predicts3(Y_dev_preds_rnn, Y_dev_preds_ridgeCV, Y_dev_preds_ridge, best1, best2)')


# In[ ]:


print(best1)
print(best2)
print("(Best) RMSL error for RNN + Ridge + RidgeCV on dev set:", rmsle(Y_dev, Y_dev_preds))


# Here is the commented out version for when I was working with 2 models instead of 3. Left here just in case.

# In[ ]:


# %%time
# def aggregate_predicts2(Y1, Y2,ratio):
#     assert Y1.shape == Y2.shape
#     return Y1 * ratio + Y2 * (1.0 - ratio)

# #ratio optimum finder
# best = 0
# lowest = 0.99
# for i in range(100):
#     r = i*0.01
#     Y_dev_preds = aggregate_predicts2(Y_dev_preds_rnn, Y_dev_preds_ridge, r)
#     fpred = rmsle(Y_dev, Y_dev_preds)
#     if fpred < lowest:
#         best = r
#         lowest = fpred
#     print(str(r)+"-RMSL error for RNN + Ridge on dev set:", fpred)
# Y_dev_preds = aggregate_predicts2(Y_dev_preds_rnn, Y_dev_preds_ridge, best)


# # Creating Submission
# I create 4 submissions with every run of the code. Might as well if the notebook takes one hour to run. They are all just variations of the aggregate prediction ratios (best, average, and 2 variations). I don't normally submit them all but sometimes a variation will do just a smidge better than the best prediction. When having extra submissions in a day, it could not hurt.

# In[ ]:


# best predicted submission
preds = aggregate_predicts3(rnn_preds, ridgeCV_preds, ridge_preds, best1, best2)
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
submission.to_csv("./rnn_ridge_submission_best.csv", index=False)


# In[ ]:


# mean submission
preds = aggregate_predicts3(rnn_preds, ridgeCV_preds, ridge_preds, 0.334, 0.333)
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
submission.to_csv("./rnn_ridge_submission_mean.csv", index=False)


# In[ ]:


# variation 1 submission
preds = aggregate_predicts3(rnn_preds, ridgeCV_preds, ridge_preds, best1-0.1, best2+0.1)
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
submission.to_csv("./rnn_ridge_submission_var_1.csv", index=False)


# In[ ]:


# variation 2 submission
preds = aggregate_predicts3(rnn_preds, ridgeCV_preds, ridge_preds, best1+0.1, best2-0.1)
submission = pd.DataFrame({
        "test_id": test_df.test_id,
        "price": preds.reshape(-1),
})
submission.to_csv("./rnn_ridge_submission_var_2.csv", index=False)


# # Time keeper
# Time keeping is really important in this challenge, as any Kernel that runs more than one hour when published will be rejected and a waste of time. Make sure that the following printout is less than that before publishing, hopefully with some wiggle room.

# In[ ]:


stop_real = datetime.now()
execution_time_real = stop_real-start_real 
print(execution_time_real)


# # Ideas that might help the models
# 
# ### RNN model
# NNs always have lots of paramiters to play with, it is a time for predictive power trade. (learning rate/decay, batch size, embedding output dimension, adding or removing layers, ect.) The tricky part is balancing those while keeping the total run time under 1 hour. Less epochs with smaller batch size tend to do better than more epochs with larger batches.  If a comperable RNN could be run in under 15-20 minutes then 2 RNN models might work better for fitting. Either way, if you want to fit in more and/or different models into the aggregate, probably need to trim some fat here. Note that the RNN, since it cannot run that long, is more sensive to any random seed changes.
# 
# ### Ridge models
# The more inputs the better with Ridge, but it does cost a bit of time. benifits from weak inputs, like name length, are very marginal but still there. Trimming them out might be better to save time. Aside from changing the inputs, I can only think that changing CV will improve the model notably more but at the cost of much more time. Cutting out the CV model could save a few minutes that could be better spent elsewhere. 
# ### General
# Using more and/or different models tends to do better, but the models need to be both good and different from each other. Hard to do with limited time frame.

# # References
# 
# This was originally based off of this Kernal: https://www.kaggle.com/nvhbk16k53/associated-model-rnn-ridge
# 
# With ideas gained from the visualizations here: https://www.kaggle.com/thykhuely/mercari-interactive-eda-topic-modelling
# 
# You can find description of the competition here https://www.kaggle.com/c/mercari-price-suggestion-challenge
