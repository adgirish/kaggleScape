
# coding: utf-8

# <h1>Introduction<h1>
# 
# Hello, everyone! I am going to try to predict probabilities using the magic of distributional semantics, and I propose a baseline solution based on Logistic Regression and Word2Vec. I hope this notebook will be helpful, and I will highly appreciate any critique or feedback. Feel free to write your thoughts at the comments section!

# In[ ]:


import numpy as np
import pandas as pd 
from subprocess import check_output
from gensim.models import Word2Vec

from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import f1_score, accuracy_score

import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')
lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_id = test['id'].values

author_mapping = {'EAP':0, 'HPL':1, 'MWS':2}
y_train = train['author'].map(author_mapping).values


# <h1>Pre-process the data<h1>
# 
# In order to use Word2Vec, you need to pre-process the data. It's very simple: you just need to split sentences to words (**tokenization**), bring the words to their basic form (**lemmatization**), and remove some very common words like articles or prepositions (**stop-word removal**). I'm using RegexpTokenizer, WordNetLemmatizer and NLTK stop word list. You could start experimenting already at this step and try to extend the stop word list or to use another lemmatizer! It will be interesting to know what will happen!

# In[ ]:


# data = [[lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop] for sent in train.text.values]


# ## Some initial experiemnts with simple vectorizers
# 
# I will write more about it very soon.

# In[ ]:


vectorizers = [ # ('3-gram TF-IDF Vectorizer on words', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                # ('3-gram Count Vectorizer on words', CountVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                # ('3-gram Hashing Vectorizer on words', HashingVectorizer(ngram_range=(1, 5), analyzer='word', binary=False)),
                ('TF-IDF + SVD', Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                                 ('svd', TruncatedSVD(n_components=150)),
                                ])),
                ('TF-IDF + SVD + Normalizer', Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', binary=False)),
                                 ('svd', TruncatedSVD(n_components=150)),
                                 ('norm', Normalizer()),
                                ]))
              ]


# In[ ]:


estimators = [
              (KNeighborsClassifier(n_neighbors=3), 'K-Nearest Neighbors', 'yellow'),
              (SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False), 'Support Vector Machine', 'red'),
              (LogisticRegression(tol=1e-8, penalty='l2', C=0.1), 'Logistic Regression', 'green'),
              (MultinomialNB(), 'Naive Bayes', 'magenta'),
              (RandomForestClassifier(n_estimators=10, criterion='gini'), 'Random Forest', 'gray'),
              (None, 'XGBoost', 'pink')
]


# In[ ]:


params = {}
params['objective'] = 'multi:softprob'
params['eta'] = 0.1
params['max_depth'] = 3
params['silent'] = 1
params['num_class'] = 3
params['eval_metric'] = 'mlogloss'
params['min_child_weight'] = 1
params['subsample'] = 0.8
params['colsample_bytree'] = 0.3
params['seed'] = 0


# In[ ]:


def vectorize():
    
    test_size = 0.3

    train_split, test_split = train_test_split(train, test_size=test_size)

    y_train_split = train_split['author'].map(author_mapping).values
    y_test_split = test_split['author'].map(author_mapping).values
    
    for vectorizer in vectorizers:
        print(vectorizer[0] + '\n')
        X = vectorizer[1].fit_transform(train.text.values)
        X_train, X_test = train_test_split(X, test_size=test_size)
        for estimator in estimators:
            if estimator[1] == 'XGBoost': 
                xgtrain = xgb.DMatrix(X_train, y_train_split)
                xgtest = xgb.DMatrix(X_test)
                model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)
                predictions = model.predict(xgtest, ntree_limit=model.best_ntree_limit).argmax(axis=1)
            else:
                estimator[0].fit(X_train, y_train_split)
                predictions = estimator[0].predict(X_test)
            print(accuracy_score(predictions, y_test_split), estimator[1])


# ### Baseline

# In[ ]:


train['num_words'] = train.text.apply(lambda x: len(str(x).split()))
test['num_words'] = test.text.apply(lambda x: len(str(x).split()))

train['num_unique_words'] = train.text.apply(lambda x: len(set(str(x).split())))
test['num_unique_words'] = test.text.apply(lambda x: len(set(str(x).split())))

train['num_chars'] = train.text.apply(lambda x: len(str(x)))
test['num_chars'] = test.text.apply(lambda x: len(str(x)))

train['num_stopwords'] = train.text.apply(lambda x: len([w for w in str(x).lower().split() if w in stop]))
test['num_stopwords'] = test.text.apply(lambda x: len([w for w in str(x).lower().split() if w in stop]))

train['mean_word_len'] = train.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test['mean_word_len'] = test.apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# In[ ]:


train_text = [' '.join([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop]) for sent in train.text.values]
test_text = [' '.join([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop]) for sent in test.text.values]


# In[ ]:


vectorizer = CountVectorizer(ngram_range=(1,7), analyzer='char')

full = vectorizer.fit_transform(train_text + test_text)
X_train = vectorizer.transform(train_text)
X_test = vectorizer.transform(test_text)

pred_full_test = 0
pred_train = np.zeros([train.shape[0], 3])

for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(train.drop(['id', 'author'], axis=1)):
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    model = MultinomialNB()
    model.fit(dev_X, dev_y)
    pred_full_test = pred_full_test + model.predict_proba(X_test)
    pred_train[val_index,:] = model.predict_proba(val_X)

pred_full_test = pred_full_test / 5.

train['CH_EAP'] = pred_train[:,0]
train['CH_HPL'] = pred_train[:,1]
train['CH_MWS'] = pred_train[:,2]
test['CH_EAP'] = pred_full_test[:,0]
test['CH_HPL'] = pred_full_test[:,1]
test['CH_MWS'] = pred_full_test[:,2]


# In[ ]:


vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
full = vectorizer.fit_transform(train_text + test_text)
X_train = vectorizer.transform(train_text)
X_test = vectorizer.transform(test_text)

pred_full_test = 0
pred_train = np.zeros([train.shape[0], 3])

for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(train.drop(['id', 'author'], axis=1)):
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    model = MultinomialNB()
    model.fit(dev_X, dev_y)
    pred_full_test = pred_full_test + model.predict_proba(X_test)
    pred_train[val_index,:] = model.predict_proba(val_X)

pred_full_test = pred_full_test / 5.

train['C_EAP'] = pred_train[:,0]
train['C_HPL'] = pred_train[:,1]
train['C_MWS'] = pred_train[:,2]
test['C_EAP'] = pred_full_test[:,0]
test['C_HPL'] = pred_full_test[:,1]
test['C_MWS'] = pred_full_test[:,2]


# In[ ]:


vectorizer = TfidfVectorizer(ngram_range=(1,5), analyzer='char')
full = vectorizer.fit_transform(train_text + test_text)
X_train = vectorizer.transform(train_text)
X_test = vectorizer.transform(test_text)

pred_full_test = 0
pred_train = np.zeros([train.shape[0], 3])

for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(train.drop(['id', 'author'], axis=1)):
    dev_X, val_X = X_train[dev_index], X_train[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    model = MultinomialNB()
    model.fit(dev_X, dev_y)
    pred_full_test = pred_full_test + model.predict_proba(X_test)
    pred_train[val_index,:] = model.predict_proba(val_X)

pred_full_test = pred_full_test / 5.

train['T_EAP'] = pred_train[:,0]
train['T_HPL'] = pred_train[:,1]
train['T_MWS'] = pred_train[:,2]
test['T_EAP'] = pred_full_test[:,0]
test['T_HPL'] = pred_full_test[:,1]
test['T_MWS'] = pred_full_test[:,2]


# In[ ]:


svd = TruncatedSVD(n_components=20, algorithm='arpack')
svd.fit(full)
train_svd = pd.DataFrame(svd.transform(X_train))
test_svd = pd.DataFrame(svd.transform(X_test))
    
train_svd.columns = ['SVD_' + str(i) for i in range(20)]
test_svd.columns = ['SVD_' + str(i) for i in range(20)]
train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)


# In[ ]:


train = train.drop(['id', 'text', 'author'], axis=1)
test = test.drop(['id', 'text'], axis=1)


# <h1>Distributional semantics<h1>
# 
# **Distributional semantic models** are frameworks that can represent words of natural language through real-valued vectors of fixed dimensions (the so-called **word embeddings**). The word "distributional" here is a reference to a distributional hypothesis that says that word semantics is distributed along all of its contexts. Such models able to capture various functional or topical relations between words through words context for for each word observed in a given corpus. Predicting words given their contexts (like **continuous bag-of-words** (CBOW) works) and  predicting the contexts from the words (like **continuous skip-gram** (SG) works) are two possible options of capturing the context, and this is how the distributional semantic model Word2Vec works. In short, with skip gram, you can create a lot more training instances from limited amount of data. We will set paramater sg to 1. It defines the training algorithm, and if sg=1, skip-gram is employed (and CBOW is employed otherwise).
# 
# About some other parameters:
# *min_count *= ignore all words with total frequency lower than this.
# *size* is the dimensionality of the feature vectors.
# *window* is the maximum distance between the current and predicted word within a sentence.

# In[ ]:


# train = pd.read_csv('../input/train.csv')
# test = pd.read_csv('../input/test.csv')
# test_id = test['id'].values

# author_mapping = {'EAP':0, 'HPL':1, 'MWS':2}
# y_train = train['author'].map(author_mapping).values


# In[ ]:


NUM_FEATURES = 100

model = Word2Vec(train_text + test_text, min_count=2, size=NUM_FEATURES, window=4, sg=1, alpha=1e-4, workers=4)


# Now we have 10852 words in our model, and we could try to find most similar words for some examples. Let's try the word "raven".

# In[ ]:


len(model.wv.vocab)


# In[ ]:


model.most_similar('raven')


# <h1>Compositional distributional semantics<h1>
# 
# We are able to represent each word in a form of a vector, but how to represent the whole sentence? Well ,semantics of sentences and phrases can be also captured as a composition of the word embeddings -- for instance, through **compositional distributional semantics** (CDS). CDS is a nominal notion of a method of capturing semantics of composed linguistic units like sentences and phrases by composing the distributional representations of the words that these units contain. The semantics of a whole sentence can be represented as a composition of words embeddings of the words constituting the sentence. An averaged unordered composition (or an arithmetic mean) is a one of the most popular methods of capturing semantics of a sentence since it is an effective solution despite its simplicity. Since one could claim that word embeddings are the building blocks of compositional representation, and while it has been shown that semantic relations can be mapped to translations in the learned vector space, the claim could be made for sentence representations of the embeddings.

# In[ ]:


def get_feature_vec(tokens, num_features, model):
    featureVec = np.zeros(shape=(1, num_features), dtype='float32')
    missed = 0
    for word in tokens:
        try:
            featureVec = np.add(featureVec, model[word])
        except KeyError:
            missed += 1
            pass
    if len(tokens) - missed == 0:
        return np.zeros(shape=(num_features), dtype='float32')
    return np.divide(featureVec, len(tokens) - missed).squeeze()


# In[ ]:


train_vectors = []
for i in train_text:
    train_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))


# <h1>Training the classifier<h1>
# 
# We are representing the labels of the authors in a form of numeric class labels, and then we are ready to train the classifier. I picked Logistic Regression, but you could use another one.

# <h1>Making predictions<h1>
# 
# And we are ready to make predictions! We will use the ability of the classifier to predict the probabilities of given classes.

# In[ ]:


test_vectors = []
for i in test_text:
    test_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))


# In[ ]:


full_vectors = []
for i in train_text + test_text:
    full_vectors.append(get_feature_vec([lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop], NUM_FEATURES, model))


# In[ ]:


svd = TruncatedSVD(n_components=30, algorithm='arpack')

svd.fit(full_vectors)
train_svd = pd.DataFrame(svd.transform(np.array(train_vectors)))
test_svd = pd.DataFrame(svd.transform(np.array(test_vectors)))
    
train_svd.columns = ['W2V_' + str(i) for i in range(30)]
test_svd.columns = ['W2V_' + str(i) for i in range(30)]

train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)


# In[ ]:


pred_full_test = 0
pred_train = np.zeros([train.shape[0], 3])
for dev_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(train):
    dev_X, val_X = train.loc[dev_index], train.loc[val_index]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    xgtrain = xgb.DMatrix(dev_X, dev_y)
    xgtest = xgb.DMatrix(test)
    model = xgb.train(params=list(params.items()), dtrain=xgtrain, num_boost_round=1000)
    predictions = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
    pred_full_test = pred_full_test + predictions
pred_full_test = pred_full_test / 5.


# In[ ]:


# xgtrain = xgb.DMatrix(train_vectors, y_train)
# xgtest = xgb.DMatrix(test_vectors)
# model = xgb.train(params=list(params.items()), dtrain=xgtrain,  num_boost_round=40)
# probs = model.predict(xgtest, ntree_limit=model.best_ntree_limit)


# <h1>Submission<h1>
# 
# One final step: make a dataframe to submit our results!

# In[ ]:


author = pd.DataFrame(pred_full_test)

final = pd.DataFrame()
final['id'] = test_id
final['EAP'] = author[0]
final['HPL'] = author[1]
final['MWS'] = author[2]

final.to_csv('submission.csv', sep=',',index=False)


# That's all for now! Thanks for reading this notebook. I'm glad if it helped you to learn something new. This is a very first version of this small tutorial, and I'm working hard to make it better. I plan to introduce some other methods of word vectors composing and to try to use some syntax features
# 
# Witch-ing you a spook-tacular Halloween! Do not let ghouls and spooks to ruin your models, and don't fear the curse of dimensionality!
