import pyximport
pyximport.install()
import pandas as pd
import numpy as np
import os
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=1)
from keras import backend
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_extraction import stop_words
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, normalize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Merge, BatchNormalization, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding,Flatten, GRU,concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import activations
import lightgbm as lgb
import time
import gc
import math
import re
import string
import itertools
from nltk.stem import WordNetLemmatizer
lemma  = WordNetLemmatizer()

class LemmaVectorizer(CountVectorizer):
    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()
        l_adder = self.lemma_adder()
        return lambda doc: self._word_ngrams(l_adder(tokenize(preprocess(self.decode(doc)))), stop_words)

    def lemma_adder(self):
        def lemmatizer(tokens):
            return list(set([lemma.lemmatize(w) for w in tokens] + tokens))
            
        return lemmatizer

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens"""
        def tokenizer(doc):
            token_pattern = re.compile(self.token_pattern)
            return token_pattern.findall(doc)
            
        return tokenizer 

def rmsle(y, h): 
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

def rmse(y, h): 
    return np.sqrt(np.square(h-y).mean())


def rmsle_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    
    y_pred = np.expm1(preds)
    y_tar  = np.expm1(y)
    
    score = rmsle(y_tar, y_pred) 
    return 'rmsle', score, False


def cleanName(text):
    # convert to lower case and strip regex
    try:
        # convert to lower case and strip regex
        textProc = text.lower()
        textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        regex = re.compile(u'[^A-Za-z0-9]+')
        textProc = regex.sub(" ", textProc)
        textProc = " ".join(textProc.split())
        
        return textProc
    except: 
        return "name error"

def getLastTwo(text):
    # convert to lower case and strip regex
    try:
         # convert to lower case and strip regex
        text = text.lower()
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text)
        text = text.split()
        
        if len(text)==1:
            return text[0]
        
        text = text[-2]+" "+text[-1]

        return text
    except: 

        return " "

def split_cat(text):
    try: return text.split("/")
    except: return ("None", "None", "None")

start = time.time()

if not ('X_train' in globals()):    

    NUM_BRANDS         = 5000
    DESC_MAX_FEAT      = 500000
    LGBM_NAME_MIN_DF   = 100
    NAME_MIN_DF        = 2
    DESC_MIN_DF        = 2
    
    MAX_NUM_OF_WORDS = 50000
    NAME_MAX_SEQUENCE_LENGTH  = 20
    DESC_MAX_SEQUENCE_LENGTH  = 100
    EMBEDDING_DIM = 30    

    print("Reading in Data")
    
    df     = pd.read_csv('../input/train.tsv', sep='\t')
    dfTest = pd.read_csv('../input/test.tsv', sep='\t')
    submission: pd.DataFrame = dfTest[['test_id']]
    
    n = 350000  #chunk row size
    dfTestList = [dfTest[i:i+n] for i in range(0,dfTest.shape[0],n)]
    cSize = len(dfTestList)
    del dfTest
    
    n_trains = df.shape[0]
    y = np.log1p(df["price"].values)
    y2 = np.reciprocal(df["price"].values+1.0)
    
    print(df.shape)
    
    print("Data Cleaning Training")
    
    # Clean all the mess
    df["brand_name"]        = df["brand_name"].fillna("unknown")
    df["item_description"]  = df["item_description"].fillna("None")
    df["name"]              = df["name"].fillna("None")
    df["item_condition_id"] = df["item_condition_id"].fillna(0)
    df["shipping"]          = df["shipping"].fillna(0)
    df['category_name']     = df['category_name'].fillna("None/None/None")    
        
    pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]
    df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"

    df['general_cat'], df['subcat_1'], df['subcat_2'] = \
    zip(*df['category_name'].apply(lambda x: split_cat(x)))
    
    df['general_cat'].fillna(value='None', inplace=True)
    df['subcat_1'].fillna(value='None', inplace=True)
    df['subcat_2'].fillna(value='None', inplace=True)
    
    df['general_cat']        = df['general_cat'].astype('category')
    df['subcat_1']           = df['subcat_1'].astype('category')
    df['subcat_2']           = df['subcat_2'].astype('category')     
    df["item_condition_id"]  = df["item_condition_id"].astype("category")

    df['name']               = df['name'].apply(lambda x: cleanName(x))
    df["item_description"]   = df["item_description"].apply(lambda x: cleanName(x))
    df['category_name']      = df['category_name'].apply(lambda x: cleanName(x))
    df['object']             = df['name'].apply(lambda x: getLastTwo(x))

    def dataCleanChunk(dfTest):
        dfTest["brand_name"]        = dfTest["brand_name"].fillna("unknown")
        dfTest["item_description"]  = dfTest["item_description"].fillna("None")
        dfTest["name"]              = dfTest["name"].fillna("None")
        dfTest["item_condition_id"] = dfTest["item_condition_id"].fillna(0)
        dfTest["shipping"]          = dfTest["shipping"].fillna(0)
        dfTest['category_name']     = dfTest['category_name'].fillna("None/None/None")    
            
        dfTest.loc[~dfTest["brand_name"].isin(pop_brands), "brand_name"] = "Other"
    
        dfTest['general_cat'], dfTest['subcat_1'], dfTest['subcat_2'] = \
        zip(*dfTest['category_name'].apply(lambda x: split_cat(x)))
        
        dfTest['general_cat'].fillna(value='None', inplace=True)
        dfTest['subcat_1'].fillna(value='None', inplace=True)
        dfTest['subcat_2'].fillna(value='None', inplace=True)
        
        dfTest['general_cat']        = dfTest['general_cat'].astype('category')
        dfTest['subcat_1']           = dfTest['subcat_1'].astype('category')
        dfTest['subcat_2']           = dfTest['subcat_2'].astype('category')     
        dfTest["item_condition_id"]  = dfTest["item_condition_id"].astype("category")
    
        dfTest['name']               = dfTest['name'].apply(lambda x: cleanName(x))
        dfTest["item_description"]   = dfTest["item_description"].apply(lambda x: cleanName(x))
        dfTest['category_name']      = dfTest['category_name'].apply(lambda x: cleanName(x))
        dfTest['object']             = dfTest['name'].apply(lambda x: getLastTwo(x))
        return dfTest
        
    print("Data Cleaning Testing")
    for index, dfChunk in enumerate(dfTestList):
        dfTestList[index] = dataCleanChunk(dfChunk)
        
    print('Generate CNN Name Sequences')
    tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS)
    tokenizer.fit_on_texts(df['name'].tolist())
    sequences = tokenizer.texts_to_sequences(df['name'].tolist())
    nameSequences = pad_sequences(sequences, maxlen=NAME_MAX_SEQUENCE_LENGTH)

    nameSequencesTest = []
    for index, dfChunk in enumerate(dfTestList):
        sequences = tokenizer.texts_to_sequences(dfChunk['name'].tolist())
        nameSequencesTest.append(pad_sequences(sequences, maxlen=NAME_MAX_SEQUENCE_LENGTH))

    print('Generate CNN Item Desc Sequences')
    tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS)
    tokenizer.fit_on_texts(df['item_description'].tolist())
    sequences = tokenizer.texts_to_sequences(df['item_description'].tolist())
    descSequences = pad_sequences(sequences, maxlen=DESC_MAX_SEQUENCE_LENGTH)

    descSequencesTest = []
    for index, dfChunk in enumerate(dfTestList):
        sequences = tokenizer.texts_to_sequences(dfChunk['item_description'].tolist())
        descSequencesTest.append(pad_sequences(sequences, maxlen=DESC_MAX_SEQUENCE_LENGTH))

    print("Processing Category Data")
    tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS, filters='.',split=".")
    tokenizer.fit_on_texts(df['general_cat'].tolist())
    generalCat = pad_sequences(tokenizer.texts_to_sequences(df['general_cat'].tolist()), maxlen=1)

    generalCatTest = []
    for index, dfChunk in enumerate(dfTestList):
        generalCatTest.append(pad_sequences(tokenizer.texts_to_sequences(dfChunk['general_cat'].tolist()), maxlen=1))

    tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS, filters='.',split=".")
    tokenizer.fit_on_texts(df['subcat_1'].tolist())
    sub1Cat = pad_sequences(tokenizer.texts_to_sequences(df['subcat_1'].tolist()), maxlen=1)
    
    sub1CatTest = []
    for index, dfChunk in enumerate(dfTestList):
        sub1CatTest.append(pad_sequences(tokenizer.texts_to_sequences(dfChunk['subcat_1'].tolist()), maxlen=1))

    tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS, filters='.',split=".")
    tokenizer.fit_on_texts(df['subcat_2'].tolist())
    sub2Cat = pad_sequences(tokenizer.texts_to_sequences(df['subcat_2'].tolist()), maxlen=1)
    
    sub2CatTest = []
    for index, dfChunk in enumerate(dfTestList):
        sub2CatTest.append(pad_sequences(tokenizer.texts_to_sequences(dfChunk['subcat_2'].tolist()), maxlen=1))
    
    tokenizer = Tokenizer(num_words=MAX_NUM_OF_WORDS, filters='.',split=".")
    tokenizer.fit_on_texts(df['brand_name'].tolist())
    brandCat = pad_sequences(tokenizer.texts_to_sequences(df['brand_name'].tolist()), maxlen=1)
    
    brandCatTest = []
    for index, dfChunk in enumerate(dfTestList):
        brandCatTest.append(pad_sequences(tokenizer.texts_to_sequences(dfChunk['brand_name'].tolist()), maxlen=1))
    
    priceLabels   = y
    itCond        = df["item_condition_id"].values
    shipping      = df["shipping"].values

    itCondTest = []
    shippingTest  = []
    for index, dfChunk in enumerate(dfTestList):
        itCondTest.append(dfChunk["item_condition_id"].values)
        shippingTest.append(dfChunk["shipping"].values)

    def get_keras_data(filterOne,filterTwo = None, filterThree = None):
        if filterThree is not None:
            X = {
                'name' : nameSequences[filterOne][filterTwo][filterThree],
                'desc' : descSequences[filterOne][filterTwo][filterThree],
                'general' : generalCat[filterOne][filterTwo][filterThree],
                'sub1Cat' : sub1Cat[filterOne][filterTwo][filterThree],
                'sub2Cat' : sub2Cat[filterOne][filterTwo][filterThree],
                'brandCat': brandCat[filterOne][filterTwo][filterThree],
                'itemCond': np.array(itCond, dtype=np.float32)[filterOne][filterTwo][filterThree],
                'shipping': np.array(shipping, dtype=np.float32)[filterOne][filterTwo][filterThree],
                }
            return X  
        elif filterTwo is not None:
            X = {
                'name' : nameSequences[filterOne][filterTwo],
                'desc' : descSequences[filterOne][filterTwo],
                'general' : generalCat[filterOne][filterTwo],
                'sub1Cat' : sub1Cat[filterOne][filterTwo],
                'sub2Cat' : sub2Cat[filterOne][filterTwo],
                'brandCat': brandCat[filterOne][filterTwo],
                'itemCond': np.array(itCond, dtype=np.float32)[filterOne][filterTwo],
                'shipping': np.array(shipping, dtype=np.float32)[filterOne][filterTwo],
                }
            return X  
        else :
            X = {
                'name' : nameSequences[filterOne],
                'desc' : descSequences[filterOne],
                'general' : generalCat[filterOne],
                'sub1Cat' : sub1Cat[filterOne],
                'sub2Cat' : sub2Cat[filterOne],
                'brandCat': brandCat[filterOne],
                'itemCond': np.array(itCond, dtype=np.float32)[filterOne],
                'shipping': np.array(shipping, dtype=np.float32)[filterOne],
                }
            return X  

    def get_keras_dataTest(cNum):
        X = {
            'name' : nameSequencesTest[cNum],
            'desc' : descSequencesTest[cNum],
            'general' : generalCatTest[cNum],
            'sub1Cat' : sub1CatTest[cNum],
            'sub2Cat' : sub2CatTest[cNum],
            'brandCat': brandCatTest[cNum],
            'itemCond': np.array(itCondTest[cNum], dtype=np.float32),
            'shipping': np.array(shippingTest[cNum], dtype=np.float32),
            }
        return X  
        
    def rnn_model():
        
        nameInput      = Input(shape=(NAME_MAX_SEQUENCE_LENGTH,), dtype='int32', name="name")
        descInput      = Input(shape=(DESC_MAX_SEQUENCE_LENGTH,), dtype='int32', name="desc")
        geneInput      = Input(shape=(1,), dtype='int32', name="general")
        sub1Input      = Input(shape=(1,), dtype='int32', name="sub1Cat")
        sub2Input      = Input(shape=(1,), dtype='int32', name="sub2Cat")
        branInput      = Input(shape=(1,), dtype='int32', name="brandCat")
        condInput      = Input(shape=[1], dtype='float32', name="itemCond")
        shipInput      = Input(shape=[1], dtype='float32', name="shipping")
        nameEmbedded   = Embedding(MAX_NUM_OF_WORDS,EMBEDDING_DIM,input_length=NAME_MAX_SEQUENCE_LENGTH,trainable=True)(nameInput)
        descEmbedded   = Embedding(MAX_NUM_OF_WORDS,EMBEDDING_DIM,input_length=DESC_MAX_SEQUENCE_LENGTH,trainable=True)(descInput)
        geneEmbedded   = Embedding(15,5)(geneInput)
        sub1Embedded   = Embedding(150,10)(sub1Input)
        sub2Embedded   = Embedding(1000,10)(sub2Input)
        branEmbedded   = Embedding(5000,10)(branInput)
    
        rnn_layer1     = GRU(10) (descEmbedded)
        rnn_layer2     = GRU(8) (nameEmbedded)
        x = concatenate(
            [rnn_layer1,
            rnn_layer2,
            Flatten()(geneEmbedded),
            Flatten()(sub1Embedded),
            Flatten()(sub2Embedded),
            Flatten()(branEmbedded),
            shipInput,
            condInput])
        x = Dense(50, activation='relu')(x)
        x = BatchNormalization()(x)
        preds = Dense(1, activation='linear')(x)
    
        model = Model([nameInput,descInput,geneInput,sub1Input,sub2Input,branInput,condInput,shipInput], preds)
    
        BATCH_SIZE = 512
        epochs = 2
    
        exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1

        steps = int(n_trains / BATCH_SIZE) * epochs
        lr_init, lr_fin = 0.007, 0.0005
        lr_decay = exp_decay(lr_init, lr_fin, steps)
        optimizer = Adam(lr=lr_init, decay=lr_decay)
        
        model.compile(loss="mse", optimizer=optimizer)
        
        del nameInput
        del descInput
        del geneInput
        del sub1Input
        del sub2Input
        del branInput
        del condInput
        del shipInput
        del nameEmbedded
        del descEmbedded
        del geneEmbedded
        del sub1Embedded
        del sub2Embedded
        del branEmbedded
        del rnn_layer1
        del rnn_layer2
        del x
        del optimizer
        gc.collect()
        
        return model

    del tokenizer
    del sequences
    gc.collect()

    print("Name Features 1")
    count = LemmaVectorizer(min_df=LGBM_NAME_MIN_DF,
                            decode_error = 'replace',
                            ngram_range = (1,1),
                            token_pattern = r"(?u)\b\w+\b",
                            strip_accents = 'unicode')
    X_name_1 = count.fit_transform(df["name"])

    X_name_1_Test = []
    for index, dfChunk in enumerate(dfTestList):
        X_name_1_Test.append(count.transform(dfChunk["name"]))

    del count    
    
    print("Name Features 234")
    count = CountVectorizer(min_df=NAME_MIN_DF,
                            decode_error = 'replace',
                            ngram_range = (2,3),
                            token_pattern = r"(?u)\b\w+\b",
                            strip_accents = 'unicode')
    X_name_234 = count.fit_transform(df["name"])
    
    X_name_234_Test = []
    for index, dfChunk in enumerate(dfTestList):
        X_name_234_Test.append(count.transform(dfChunk["name"]))
        
    del count    

    print("Name Features char")
    count = TfidfVectorizer(min_df=NAME_MIN_DF,
                            decode_error = 'replace',
                            ngram_range = (1,4),
                            token_pattern = r"(?u)\b\w+\b",
                            strip_accents = 'unicode',
                            analyzer      = 'char')
    X_name_char = count.fit_transform(df["name"])
    X_name_char_Test = []
    for index, dfChunk in enumerate(dfTestList):
        X_name_char_Test.append(count.transform(dfChunk["name"]))

    del count 

    print("category Features")
    count = CountVectorizer(ngram_range = (1,1),
                            decode_error = 'replace',
                            token_pattern = r"(?u)\b\w+\b",
                            strip_accents = 'unicode')
    X_category = count.fit_transform(df["category_name"])
    X_category_Test = []
    for index, dfChunk in enumerate(dfTestList):        
        X_category_Test.append(count.transform(dfChunk["category_name"]))
    
    del count    

    print('Object Features')
    df['object'] = df['name'].apply(lambda x: getLastTwo(x))
    count = CountVectorizer(min_df=LGBM_NAME_MIN_DF,decode_error = 'replace',)
    X_object = count.fit_transform(df["object"])
    X_object_Test = []
    for index, dfChunk in enumerate(dfTestList):    
        X_object_Test.append(count.transform(dfChunk["object"]))

    del count 
    
    print("Brand Features")
    count = LemmaVectorizer(ngram_range = (1,1),
                            decode_error = 'replace',
                            token_pattern = r"(?u)\b\w+\b",
                            strip_accents = 'unicode')
    X_brand = count.fit_transform(df["brand_name"])
    X_brand_Test = []
    for index, dfChunk in enumerate(dfTestList):    
        X_brand_Test.append(count.transform(dfChunk["brand_name"]))

    del count
    gc.collect()
    
    print("Item Cond & Shipping Features")
    ohe = OneHotEncoder(dtype=np.float32, handle_unknown='ignore')
    X_dummies = ohe.fit_transform(np.array([df["item_condition_id"].tolist(),df["shipping"].tolist()]).T)
    X_dummies_Test = []
    for index, dfChunk in enumerate(dfTestList):    
        X_dummies_Test.append(ohe.transform(np.array([dfChunk["item_condition_id"].tolist(),dfChunk["shipping"].tolist()]).T))

    gc.collect()
    
    print("Description Features")
    count_descp = CountVectorizer(max_features = DESC_MAX_FEAT,
                                  decode_error = 'replace',
                                  min_df=LGBM_NAME_MIN_DF,
                                  ngram_range = (1,1),
                                  token_pattern = r"(?u)\b\w+\b",
                                  strip_accents = 'unicode')
    X_descp = count_descp.fit_transform(df["item_description"])
    X_descp_add = count_descp.transform(df["name"])
    X_descp = X_descp + X_descp_add

    X_descp_Test = []
    for index, dfChunk in enumerate(dfTestList):
        X_descp_Test.append(count_descp.transform(dfChunk["item_description"]) + count_descp.transform(dfChunk["name"]))

    del count_descp
    del X_descp_add
    gc.collect()

    print("Description Features Tf-Idf")
    count_descp = TfidfVectorizer(max_features = DESC_MAX_FEAT,
                                  decode_error = 'replace',
                                  min_df=DESC_MIN_DF,
                                  ngram_range = (2,3),
                                  token_pattern = r"(?u)\b\w+\b",
                                  strip_accents = 'unicode')
    X_descp_tfidf = count_descp.fit_transform(df["item_description"])

    X_descp_tfidf_Test = []
    for index, dfChunk in enumerate(dfTestList):
        X_descp_tfidf_Test.append(count_descp.transform(dfChunk["item_description"]))

    del count_descp
    gc.collect()

    del df
    del dfTestList
    gc.collect()

    print("X_name_1 {}".format(X_name_1.shape)) 
    print("X_name_234 {}".format(X_name_234.shape))
    print("X_name_char {}".format(X_name_char.shape))    
    print("X_category {}".format(X_category.shape))    
    print("X_object {}".format(X_object.shape))    
    print("X_brand {}".format(X_brand.shape))    
    print("X_dummies {}".format(X_dummies.shape))    
    print("X_descp {}".format(X_descp.shape))    
    print("X_descp_tfidf {}".format(X_descp_tfidf.shape))    


print("Concatenate X_1")
X_1 = hstack((X_dummies,            
              X_descp,
              X_brand,              
              X_category,
              X_name_1,              
              X_object,              
              )).tocsr()

print("Concatenate X_2")
X_2 = hstack((X_name_234,
              X_descp_tfidf,
              X_name_char
              )).tocsr()

del X_dummies            
del X_descp
del X_brand              
del X_category
del X_object
del X_name_1
del X_name_234
del X_descp_tfidf
del X_name_char
gc.collect()

X_1.data = X_1.data.astype(np.float32)
X_2.data = X_2.data.astype(np.float32)

X_1_Test = []         
for index in range(cSize):
    print('Concat X1 Batch {}'.format(index+1))
    X_1_Test.append(hstack((X_dummies_Test[0],X_descp_Test[0],X_brand_Test[0],X_category_Test[0],X_name_1_Test[0],X_object_Test[0])).tocsr())
    del X_dummies_Test[0]
    del X_descp_Test[0]
    del X_brand_Test[0]
    del X_category_Test[0]
    del X_name_1_Test[0]
    del X_object_Test[0]
    gc.collect()
    X_1_Test[index].data = X_1_Test[index].data.astype(np.float32)
    
X_2_Test = []         
for index in range(cSize):
    print('Concat X2 Batch {}'.format(index+1))
    X_2_Test.append(hstack((X_name_234_Test[0],X_descp_tfidf_Test[0],X_name_char_Test[0])).tocsr())
    del X_name_234_Test[0]
    del X_descp_tfidf_Test[0]
    del X_name_char_Test[0]
    gc.collect()
    X_2_Test[index].data = X_2_Test[index].data.astype(np.float32)




print("X_1 {}".format(X_1.shape)) 
print("X_2 {}".format(X_2.shape))


np.random.seed(0)

filterTrain     = np.where((np.expm1(y) > 1) )
y     = y[filterTrain[0]]
y2    = y2[filterTrain[0]]
X_1   = X_1[filterTrain[0]]  
X_2   = X_2[filterTrain[0]]  

# OOF Features for LGBM
numSplit2     = 2
numFold2      = 1
kf2           = RepeatedKFold(n_splits=numSplit2, random_state=None, n_repeats=1)
ridgeHelp     = np.zeros([X_1.shape[0], 3])
ridgeValid    = []         
for index in range(cSize):
    ridgeValid.append(np.zeros([X_1_Test[index].shape[0], 3]))

for train_index2, test_index2 in kf2.split(X_1):
    oofTrainData1, oofTestData1     = X_2[train_index2], X_2[test_index2]
    oofTrainTarget, oofTestTarget   = y[train_index2], y[test_index2]
    oofTrainTarget2, oofTestTarget2 = y2[train_index2], y2[test_index2]
    

    print("Ridge Y1 OOF Fold_{}".format(numFold2))        
    model = Ridge(
            solver='auto',
            fit_intercept=True,
            alpha=0,
            max_iter=100,
            normalize=False,
            tol=0.05)
         
    model.fit(oofTrainData1, oofTrainTarget)
    y_pred_1   = model.predict(oofTestData1)
    
    val_pred_1 = []         
    for index in range(cSize):
        val_pred_1.append(model.predict(X_2_Test[index]))

    del model
    gc.collect()

    print("Ridge Y2 OOF Fold_{}".format(numFold2))        
    model = Ridge(
            solver='auto',
            fit_intercept=True,
            alpha=0,
            max_iter=100,
            normalize=False,
            tol=0.05)
         
    model.fit(oofTrainData1, oofTrainTarget2)
    y_pred_3   = model.predict(oofTestData1)
    y_pred_3   = np.clip(y_pred_3,0,10000)

    val_pred_3 = []         
    for index in range(cSize):
        val_pred_3.append(np.clip(model.predict(X_2_Test[index]),0,10000))

    del model
    del oofTrainData1
    del oofTestData1
    gc.collect()

    rnnTrainData   = get_keras_data(filterTrain,train_index2)

    print("RNN OOF Fold_{}".format(numFold2))
    model = rnn_model()
    model.fit(rnnTrainData,oofTrainTarget,
              batch_size=512,
              epochs=2,
              verbose=2)

    rnnValidData   = get_keras_data(filterTrain,test_index2)        
    y_pred_2   = model.predict(rnnValidData, batch_size=70000)[:,0]

    val_pred_2 = []         
    for index in range(cSize):
        val_pred_2.append(model.predict(get_keras_dataTest(index), batch_size=70000)[:,0])

    del model
    del rnnTrainData
    del rnnValidData
    gc.collect()

    ridgeHelp[test_index2,0] = y_pred_1
    ridgeHelp[test_index2,1] = y_pred_2
    ridgeHelp[test_index2,2] = y_pred_3

    for index in range(cSize):
        ridgeValid[index][:,0] = ridgeValid[index][:,0] + val_pred_1[0]
        ridgeValid[index][:,1] = ridgeValid[index][:,1] + val_pred_2[0]
        ridgeValid[index][:,2] = ridgeValid[index][:,2] + val_pred_3[0]
        del val_pred_1[0]
        del val_pred_2[0]
        del val_pred_3[0]

    numFold2= numFold2+1

del X_2_Test
del nameSequencesTest
del descSequencesTest
del generalCatTest
del sub1CatTest
del sub2CatTest
del brandCatTest
del itCondTest
del shippingTest

for index in range(cSize):
    ridgeValid[index] = ridgeValid[index] / numSplit2


print("Training LGBM")
d_train = lgb.Dataset(hstack((X_1,ridgeHelp)).tocsr(), label=y)

del X_1
del ridgeHelp

params = {
    'max_bin':255,
    'min_data_in_leaf':1,
    'learning_rate': 0.15,
    'application': 'regression',
    'max_depth': 20,
    'num_leaves': 90,
    'verbosity': -1,
    'metric': 'RMSE',
    'data_random_seed': 1,
    'bagging_freq' : 0, 
    'bagging_fraction' : 0.5,
    'feature_fraction' : 1,
    'lambda_l1' : 2, 
    'lambda_l2' : 0,        
    'nthread': 8,
    'bin_construct_sample_cnt': 50000
}

print("Training Start")
model      = lgb.train(params, train_set=d_train, num_boost_round=1500, verbose_eval=100) 
print("Training Finished")

y_pred_4 = []         
for index in range(cSize):
    print('LGBM Predict Batch {}'.format(index+1))
    y_pred_4.append(model.predict(hstack((X_1_Test[0],ridgeValid[0])).tocsr()))
    del X_1_Test[0]
    del ridgeValid[0]

print("Write Submission")
del model
submission['price'] = np.clip(np.expm1(np.clip(list(itertools.chain.from_iterable(y_pred_4)),0,10)),0,10000)
submission.to_csv("submission_12Feb_From_v5.csv", index=False)