# Modified from Bojan LGBM Script
# Ensamble with Ridge and NN can reach my current LB position

import pyximport
pyximport.install()
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_extraction import stop_words
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, normalize
import lightgbm as lgb
import time
import gc
import math
import re
import string
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
    try:
        textProc = text.lower()
        textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        regex = re.compile(u'[^A-Za-z0-9]+')
        textProc = regex.sub(" ", textProc)
        textProc = " ".join(textProc.split())
        
        return textProc
    except: 
        return "name error"

def getLastTwo(text):
    try:
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

NUM_BRANDS         = 5000
DESC_MAX_FEAT      = 500000
LGBM_NAME_MIN_DF   = 20
NAME_MIN_DF        = 2
DESC_MIN_DF        = 2

print("Reading in Data")
df     = pd.read_csv('../input/train.tsv', sep='\t')
dfTest = pd.read_csv('../input/test.tsv', sep='\t')
n_trains = df.shape[0]
y = np.log1p(df["price"].values)
submission: pd.DataFrame = dfTest[['test_id']]

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

print("Data Cleaning Testing")
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

print("Name Features 1")
count = LemmaVectorizer(min_df=LGBM_NAME_MIN_DF,
                        decode_error = 'replace',
                        ngram_range = (1,1),
                        token_pattern = r"(?u)\b\w+\b",
                        strip_accents = 'unicode')
X_name_1 = count.fit_transform(df["name"])
X_name_1_Test = count.transform(dfTest["name"])
del count    

print("category Features")
count = CountVectorizer(ngram_range = (1,1),
                        decode_error = 'replace',
                        token_pattern = r"(?u)\b\w+\b",
                        strip_accents = 'unicode')
X_category = count.fit_transform(df["category_name"])
X_category_Test = count.transform(dfTest["category_name"])
del count    

print('Object Features')
df['object'] = df['name'].apply(lambda x: getLastTwo(x))
count = CountVectorizer(min_df=LGBM_NAME_MIN_DF,decode_error = 'replace',)
X_object = count.fit_transform(df["object"])
X_object_Test = count.transform(dfTest["object"])
del count 

print("Brand Features")
count = LemmaVectorizer(ngram_range = (1,1),
                        decode_error = 'replace',
                        token_pattern = r"(?u)\b\w+\b",
                        strip_accents = 'unicode')
X_brand = count.fit_transform(df["brand_name"])
X_brand_Test = count.transform(dfTest["brand_name"])
del count
gc.collect()

print("Item Cond & Shipping Features")
ohe = OneHotEncoder(dtype=np.float32, handle_unknown='ignore')
X_dummies = ohe.fit_transform(np.array([df["item_condition_id"].tolist(),df["shipping"].tolist()]).T)
X_dummies_Test = ohe.transform(np.array([dfTest["item_condition_id"].tolist(),dfTest["shipping"].tolist()]).T)
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

X_descp_Test = count_descp.transform(dfTest["item_description"])
X_descp_add_Test = count_descp.transform(dfTest["name"])
X_descp_Test = X_descp_Test + X_descp_add_Test

del count_descp
del X_descp_add
del X_descp_add_Test
gc.collect()

del df
gc.collect()

print("X_name_1 {}".format(X_name_1.shape)) 
print("X_category {}".format(X_category.shape))    
print("X_object {}".format(X_object.shape))    
print("X_brand {}".format(X_brand.shape))    
print("X_dummies {}".format(X_dummies.shape))    
print("X_descp {}".format(X_descp.shape))    


print("Concatenate X_1")
X_1 = hstack((X_dummies,            
              X_descp,
              X_brand,              
              X_category,
              X_name_1,              
              X_object,              
              )).tocsr()
X_1_Test = hstack((X_dummies_Test,            
              X_descp_Test,
              X_brand_Test,              
              X_category_Test,
              X_name_1_Test,              
              X_object_Test,              
              )).tocsr()

del X_dummies            
del X_descp
del X_brand              
del X_category
del X_object
del X_name_1
del X_dummies_Test            
del X_descp_Test
del X_brand_Test              
del X_category_Test
del X_object_Test
del X_name_1_Test

gc.collect()

X_1.data = X_1.data.astype(np.float32)
X_1_Test.data = X_1_Test.data.astype(np.float32)

print("X_1 {}".format(X_1.shape)) 

np.random.seed(0)

filterTrain     = np.where((np.expm1(y) > 1) )
y     = y[filterTrain[0]]
X_1   = X_1[filterTrain[0]]  

print("Training LGBM")
d_train = lgb.Dataset(X_1, label=y)
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

model      = lgb.train(params, train_set=d_train, num_boost_round=2500) 
y_pred_4   = model.predict(X_1_Test)    

submission['price'] = np.clip(np.expm1(y_pred_4),0,10000)
submission.to_csv("Submission_Single_LGBM.csv", index=False)