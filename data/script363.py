
# coding: utf-8

# I start with [Alex Papiu's](https://www.kaggle.com/apapiu/ridge-script) script where he preprocesses the data into a ```scipy.sparse``` matrix and train a neural network. Keras does not like ```scipy.sparse``` matrices and converting the entire training set to a matrix will lead to computer memory issues; so the model is trained in batches: 32 samples at a time, and these few samples can be converted to matrices and fed into the network. 
#               
# Also, thanks to Pavel (Pasha) Gyrya's contribution for improving the model. Now there is no need to convert batches to np matrices. 
# 
# This requieres a batch generator, which I pieced together from this [stack overflow question](https://stackoverflow.com/questions/41538692/using-sparse-matrices-with-keras-and-tensorflow) and I set up an iterator to make it threadsafe for parallelization. ~~Kagle allows the use of 32 cores which speeds up the training~~. Seems like kaggle only allows four cores.  
# 
# I have been tuning the network and it seems like a smaller network with longer epochs yields better results. Currently I have a two hidden layers with 25  and 10 nodes. This is quite small but, with the input layer considered, this network still yields approximately 1.5M parameters!
# 
# Give it a try and let me know what you think. There are still plenty of things on can try:
# * Add a validation set for early stopping. 
# * Tune `batch_size`, `samples_per_epoch`, and nodes in hidden layers.
# * Add dropout.
# * Add L1 and/or L2 regularization.
#    
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import scipy
import time
import gc

from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Input


start_time = time.time()


def avg_predictions(L):
    N = len(L)
    f = L[0]
    for p in range(1,N):
        f = f + L[p]
        f = (1/N)*f
    return f

def define_model(data, nodes1, nodes2, drop1, drop2):
    x = Input(shape = (data.shape[1], ), dtype = 'float32', sparse = True)     
    d1 = Dense(nodes1, activation='relu')(x)
    d2 = Dropout(drop1)(d1)
    d3 = Dense(nodes2, activation='sigmoid')(d2)
    d4 = Dropout(drop2)(d3)
    out= Dense(1, activation = 'linear')(d4)
    model = Model(x,out)
    return model
    

def preprocess(num_brands, name_min, max_feat_desc, ngrams):
    print("Preprocessing Data...")
    NUM_BRANDS = num_brands
    NAME_MIN_DF = name_min
    MAX_FEAT_DESCP = max_feat_desc
    
    df_train = pd.read_csv('../input/train.tsv', sep='\t')
    df_train = df_train.reindex(np.random.permutation(df_train.index))
    df_train.reset_index(inplace=True, drop=True)
    
    df_test = pd.read_csv('../input/test.tsv', sep='\t')
    
    df = pd.concat([df_train, df_test], 0)
    nrow_train = df_train.shape[0]
    Y = np.log1p(df_train["price"])
    
    del df_train
    gc.collect()
    
    df["category_name"] = df["category_name"].fillna("Other").astype("category")
    df["brand_name"] = df["brand_name"].fillna("unknown")
    
    pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]
    df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"
    
    df["item_description"] = df["item_description"].fillna("None")
    df["item_condition_id"] = df["item_condition_id"].astype("category")
    df["brand_name"] = df["brand_name"].astype("category")
    
    
    #print("Encodings...")
    count = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = count.fit_transform(df["name"])
    
    #print("Category Encoders...")
    unique_categories = pd.Series("/".join(df["category_name"].unique().astype("str")).split("/")).unique()
    count_category = CountVectorizer()
    X_category = count_category.fit_transform(df["category_name"])
    
    #print("Descp encoders...")
    count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 
                                  ngram_range = (1, ngrams),
                                  stop_words = "english")
    X_descp = count_descp.fit_transform(df["item_description"])
    
    #print("Brand encoders...")
    vect_brand = LabelBinarizer(sparse_output=True)
    X_brand = vect_brand.fit_transform(df["brand_name"])
    
    #print("Dummy Encoders...")
    X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[
        "item_condition_id", "shipping"]], sparse = True).values)
    
    X = scipy.sparse.hstack((X_dummies, 
                             X_descp,
                             X_brand,
                             X_category,
                             X_name)).tocsr()

    
    return X[:nrow_train], Y, X[nrow_train:], df_test
    
    
def set_split(X_data, y_data, test_size):
    
    N = int(X_data.shape[0]*(1-test_size))
    
    return(X_data[:N], X_data[N:], y_data[:N], y_data[N:])


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)




start_time = time.time()

X, Y, X_test, df_test = preprocess(2500, 10, 50000, 3)

x_train, x_val, y_train, y_val = set_split(X, Y, test_size=0.10)

elapsed_time = time.time() - start_time
print("Preprocessing Time: {}".format(hms_string(elapsed_time)))

tpoint1 = time.time()
print("Fitting Model...")

nodes1 = 64
nodes2 = 32
drop1 =  0.30
drop2 =  0.25

print("Training Model...")
    
model = define_model(x_train, nodes1, nodes2, drop1, drop2)
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=3, verbose=1, mode='auto')
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x=x_train, y=y_train,
          batch_size=600,
          callbacks=[monitor],
          validation_data=(x_val, y_val),
          epochs=10, verbose=0)
    
tpoint2 = time.time()
print("Time Training: {}".format(hms_string(tpoint2-tpoint1)))
    
pred = model.predict(x=X_test, batch_size=8000, verbose=0)


tpoint3 = time.time()
print("Time for Predicting: {}".format(hms_string(tpoint3-tpoint2)))

df_test["price"] = np.expm1(pred)
df_test[["test_id", "price"]].to_csv("submission_NN.csv", index = False)

elapsed_time = time.time() - start_time
print("Total Time: {}".format(hms_string(elapsed_time)))





