
# coding: utf-8

# # Data reading

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from functools import lru_cache
from tqdm import tqdm as tqdm
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from scipy import sparse


# In[2]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[4]:


train['comment_text'] = train['comment_text'].fillna('nan')


# In[5]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[6]:


test['comment_text'] = test['comment_text'].fillna('nan')


# In[7]:


submission = pd.read_csv('../input/sample_submission.csv')
submission.head()


# # Basic analysis
# We have multilabel classification task. So let's check proportion of each label:

# In[8]:


for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    print(label, (train[label] == 1.0).sum() / len(train))


# and correlation between target variables (maybe we'l could build some kind of hierarchy classification or something like it).

# In[9]:


train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].corr()


# # Text postprocessing
# 
# I'll try models with:
# - text as is
# - stemmed text
# - lemmatized text

# In[10]:


stemmer = EnglishStemmer()

@lru_cache(30000)
def stem_word(text):
    return stemmer.stem(text)


lemmatizer = WordNetLemmatizer()

@lru_cache(30000)
def lemmatize_word(text):
    return lemmatizer.lemmatize(text)


def reduce_text(conversion, text):
    return " ".join(map(conversion, wordpunct_tokenize(text.lower())))


def reduce_texts(conversion, texts):
    return [reduce_text(conversion, str(text))
            for text in tqdm(texts)]


# In[11]:


train['comment_text_stemmed'] = reduce_texts(stem_word, train['comment_text'])
test['comment_text_stemmed'] = reduce_texts(stem_word, test['comment_text'])
train['comment_text_lemmatized'] = reduce_texts(lemmatize_word, train['comment_text'])
test['comment_text_lemmatized'] = reduce_texts(lemmatize_word, test['comment_text'])


# In[14]:


train.head()


# In[15]:


test.head()


# # Validation
# 
# Our metric is collumn-average of collumn log_loss values. So let's define custom metric based on binary log loss and define cross-validation function:

# In[16]:


def metric(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    columns = y_true.shape[1]
    column_losses = []
    for i in range(0, columns):
        column_losses.append(log_loss(y_true[:, i], y_pred[:, i]))
    return np.array(column_losses).mean()


# ## Cross-validation
# 
# I don't found quickly a way to stratified split for multilabel case.
# 
# So I used next way for stratified splitting:
# 
# - define ordered list of all possible label combinations. E.g.
# 
#     - 0 = ["toxic"=0, "severe_toxic"=0, "obscene"=0, "threat"=0, "insult"=0, "identity_hate"=0]
#     - 1 = ["toxic"=0, "severe_toxic"=0, "obscene"=0, "threat"=0, "insult"=1, "identity_hate"=0]
#     - 2 = ["toxic"=0, "severe_toxic"=0, "obscene"=0, "threat"=0, "insult"=1, "identity_hate"=1]
# 
# - for each row replace label combination with combination index 
# - use StratifiedKFold on this
# - train and test model by train/test indices from StratifiedKFold
# 
# Basic idea is next:
# - we can present label combination as class for multiclass classification - at least for some cases
# - we can stratified split by combination indices
#     - so in each split distribution of combination indices will be similar to full set
#     - so source label distribution also will be similar
#     
# But I don't sure that all my assumpions are fully correct - at least, for common case.

# In[17]:


def cv(model, X, y, label2binary, n_splits=3):
    def split(X, y):
        return StratifiedKFold(n_splits=n_splits).split(X, y)
    
    def convert_y(y):
        new_y = np.zeros([len(y)])
        for i, val in enumerate(label2binary):
            idx = (y == val).max(axis=1)
            new_y[idx] = i
        return new_y
    
    X = np.array(X)
    y = np.array(y)
    scores = []
    for train, test in tqdm(split(X, convert_y(y)), total=n_splits):
        fitted_model = model(X[train], y[train])
        scores.append(metric(y[test], fitted_model(X[test])))
    return np.array(scores)


# Let's define possible label combinations:

# In[18]:


label2binary = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1, 1],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 1, 1],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1],
    [1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 0],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1],
])


# # Dummy model
# 
# Let's build dummy model that always return 0.5 and compare score on cross-validation with test-set public leatherboard "All 0.5s Benchmark" (score - 0.693)

# In[19]:


def dummy_model(X, y):
    def _predict(X):
        return np.ones([X.shape[0], 6]) * 0.5
    
    return _predict

cv(dummy_model,
   train['comment_text'],
   train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],
   label2binary)


# seems like we built metric correctly, so let's go to baseline building
# 
# # Baseline (binary logistic regression over word-based tf-idf)
# 
# Let's build model that:
# - compute tf-idf for given train texts
# - train 6 logistic regressions (one for each label)
# - compute tf-idf on test texts
# - compute probability of "1" class for all 6 regressions

# In[20]:


def regression_baseline(X, y):
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X)
    columns = y.shape[1]
    regressions = [
        LogisticRegression().fit(X_tfidf, y[:, i])
        for i in range(columns)
    ]
    
    def _predict(X):
        X_tfidf = tfidf.transform(X)
        predictions = np.zeros([len(X), columns])
        for i, regression in enumerate(regressions):
            regression_prediction = regression.predict_proba(X_tfidf)
            predictions[:, i] = regression_prediction[:, regression.classes_ == 1][:, 0]
        return predictions
    
    return _predict


# Now let's check model on source texts/stemmed texts/lemmatized texts

# In[21]:


cv(regression_baseline,
   train['comment_text'],
   train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],
   label2binary)


# In[22]:


cv(regression_baseline,
   train['comment_text_stemmed'],
   train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],
   label2binary)


# In[23]:


cv(regression_baseline,
   train['comment_text_lemmatized'],
   train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],
   label2binary)


# As you can see - this baseline gives best score on stemmed texts.
# Anyway - let's  try to add character-level features:
# 
# # Regressions over tfidf over words and character n-grams
# 
# Let's build model that:
# - compute tfidf of words of stemmed texts
# - compute tfidf of character n-grams from source text
# - train/predict regressions on computed tfidf-s.

# In[25]:


def regression_wordchars(X, y):
    tfidf_word = TfidfVectorizer()
    X_tfidf_word = tfidf_word.fit_transform(X[:, 1])
    tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)
    X_tfidf_char = tfidf_char.fit_transform(X[:, 0])
    X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char])
    
    columns = y.shape[1]
    regressions = [
        LogisticRegression().fit(X_tfidf, y[:, i])
        for i in range(columns)
    ]
    
    def _predict(X):
        X_tfidf_word = tfidf_word.transform(X[:, 1])
        X_tfidf_char = tfidf_char.transform(X[:, 0])
        X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char])
        predictions = np.zeros([len(X), columns])
        for i, regression in enumerate(regressions):
            regression_prediction = regression.predict_proba(X_tfidf)
            predictions[:, i] = regression_prediction[:, regression.classes_ == 1][:, 0]
        return predictions
    
    return _predict


# In[26]:


cv(regression_wordchars,
   train[['comment_text', 'comment_text_stemmed']],
   train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']],
   label2binary)


# # Prediction
# 
# Let's use our best model - regression over word&chars tfidf to build submission:

# In[27]:


get_ipython().run_cell_magic('time', '', "model = regression_wordchars(np.array(train[['comment_text', 'comment_text_stemmed']]),\n                             np.array(train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]))")


# In[28]:


get_ipython().run_cell_magic('time', '', "prediction = model(np.array(test[['comment_text', 'comment_text_stemmed']]))")


# In[30]:


for i, label in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
    submission[label] = prediction[:, i]
submission.head()


# In[29]:


submission.to_csv('output.csv', index=None)

