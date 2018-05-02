
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# <h1>Introduction</h1>
# 
# In this tutorial I'd like to illustrate some advanced uses of pipelines. Some readers might have used them in work already, or could be totally unfamiliar with them - no worries, I'll cover basic uses as well as some advanced tricks.

# <h2>Advantages of pipelines</h2>
# 1. Use of pipelines gives you a kind of meta-language to describe your model and abstract from some implementation details.
# 2. With pipelines, you don't need to carry test dataset transformation along with your train features - this is taken care of automatically.
# 3. Hyperparameter tuning made easy - set new parameters on any estimator in the pipeline, and refit - in 1 line. Or use GridSearchCV on the pipeline.

# <h2>Simple illustrations</h2>
# 
# Let's start with simple illustrations.

# <h3>Data preparation</h3>
# 
# I assume you are familiar with the data structure from other hot tutorials, so I'll be brief here.

# In[ ]:


#read the data in
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


#encode labels to integer classes
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder().fit(train['author'])

#Original labels are stored in a class property
#and binarized labels correspond to indexes of this array - 0,1,2 in our case of three classes
lb.classes_


# In[ ]:


#after transformation the label will look like an array of integer taking values 0,1,2
lb.transform(train['author'])


# Split the thain dataset into two parts: the large one is used for training models,
# the smaller one serves as a validation dataset - which is not seen during training, but has labels.
# 
# Here our new testing data set will be 0.7 of original train dataset (test_size=0.3),
# we want proportion of classes to be kept in the new test (stratify=train['author']), 
# and we set the random state for reproducability (random_state=17).

# In[ ]:


from sklearn.model_selection import train_test_split

X_train_part, X_valid, y_train_part, y_valid =    train_test_split(train['text'], 
                     lb.transform(train['author']), 
                test_size=0.3,random_state=17, stratify=train['author'])


# <h3>Preparing a pipeline</h3>
# 
# Let's create our fist model.

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

pipe1 = Pipeline([
    ('cv', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('logit', LogisticRegression()),
])


# In this pipeline, input data comes first to `CountVectorizer`, which creates a sparse matrix of word counts in each sentence. This matrix then serves as input to `TfidfTransformer` which massages the data and handles it to the LogisticRegression estimator for training and prediction.

# <h3>Fitting the Model</h3>
# 
# Our pipe1 object has all the properties of an estimator, so we can treat it as such. Hence, we call the `fit()` method.
# 
# Note that Pipeline "knows" that the first tho steps are transformers, so it will only call `fit()` and `transform()` on them, or just `fit_transform()` if it's defined for the class. For the `LogisticRegression` instance - our final model - only `fit()` will be called.

# In[ ]:


pipe1.fit(X_train_part, y_train_part)


# Let's stop here for a moment and check what we've got. 
# 
# 
# We can look up all the steps of the pipeline, and all the parameters of the steps: 

# In[ ]:


pipe1.steps


# We can access each step's parameters by name, as well as any of its methods and properties:

# In[ ]:


pipe1.named_steps['logit'].coef_


# <h3>Making predictions</h3>
# 
# This is as easy as with a 'regular' model. We just call `predict()` or `predict_proba()`.
# Let's use our hold-out data for validation:

# In[ ]:


from sklearn.metrics import log_loss

pred = pipe1.predict_proba(X_valid)
log_loss(y_valid, pred)


# <h3>Playing with parameters</h3>
# That was not a winner! But hold on, we are not there yet!
# We can improve the score by tuning some parameters. As I have shown earlier, we can check every step's paramters by its name:

# In[ ]:


pipe1.named_steps['logit'].get_params()


# But we can also check and set them all at once:

# In[ ]:


pipe1.get_params()


# You can see here, that the Pipeline class has all steps' parameters with their respective names prepended. We can set them as well and fit the model.

# In[ ]:


#set_params(cv__lowercase=True)
pipe1.set_params(cv__min_df=6, 
                 cv__lowercase=False).fit(X_train_part, y_train_part)
pred = pipe1.predict_proba(X_valid)
log_loss(y_valid, pred)


# A little bit better! You get the idea. Thinking `GridSearchCV` or `cross_val_score`? Yes, will work on the pipeline too. Fork this kernel and implement it yourself!

# <h3>Playing with a model</h3>
# Would you like to try another classifier? Naive Bayes seems to be in favor across winning kernels. Replacing a pipeline step is easy:

# In[ ]:


from sklearn.naive_bayes import MultinomialNB, BernoulliNB

pipe1 = Pipeline([
    ('cv', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #('logit', LogisticRegression()),
    ('bnb', BernoulliNB()),
   
])


# In[ ]:


pipe1.fit(X_train_part, y_train_part)
pred = pipe1.predict_proba(X_valid)
log_loss(y_valid, pred)


# Best score so far! Have more ideas? Fork this kernel and try them out!
# 
# Also, for more examples and a gentle intro, read another great Spooky pipeline tutorial: [pipeline for the beginners](https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines)
# 

# <h3>Feature Union</h3>
# 
# Another strong side of pipelines come from its brother class - `FeatureUnion`. It will help us to combine together some new features that we create as part of EDA. Let's, for example, take a statistics on parts of speech used in each sentence,  and see if it can help to improve the score.

# <h3>NLTK Part-of-Speech tagger</h3>
# 
# Suppose we assume that authors could be distinguished by some statistics of use of some parts of speech. May be frequency of conjugatoin is a significant feature? Or use of punctuation?
# 
# NLTK can help to tag words in sentences.

# In[ ]:


import nltk

text = "And now we are up for 'something' completely different;"
tokens = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokens)
tagged


# Puzzled about all the tags? `CC` means conjunction, coordinated. Take a look at the complete description with `nltk.help.upenn_tagset()` that I don't run here to keep the clutter down.
# 

# So, in order to tag our text in the pipeline, we will create an estimator class of our own. Don't be afraid - this is simple. We just have to inherit some base classes and overload very few functions that we are actually going to use:

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

class PosTagMatrix(BaseEstimator, TransformerMixin):
    #normalise = True - devide all values by a total number of tags in the sentence
    #tokenizer - take a custom tokenizer function
    def __init__(self, tokenizer=lambda x: x.split(), normalize=True):
        self.tokenizer=tokenizer
        self.normalize=normalize

    #helper function to tokenize and count parts of speech
    def pos_func(self, sentence):
        return Counter(tag for word,tag in nltk.pos_tag(self.tokenizer(sentence)))

    # fit() doesn't do anything, this is a transformer class
    def fit(self, X, y = None):
        return self

    #all the work is done here
    def transform(self, X):
        X_tagged = X.apply(self.pos_func).apply(pd.Series).fillna(0)
        X_tagged['n_tokens'] = X_tagged.apply(sum, axis=1)
        if self.normalize:
            X_tagged = X_tagged.divide(X_tagged['n_tokens'], axis=0)

        return X_tagged


# Now, our new pipeline:

# In[ ]:


from sklearn.pipeline import FeatureUnion

pipe2 = Pipeline([
    ('u1', FeatureUnion([
        ('tfdif_features', Pipeline([
            ('cv', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
        ])),
        ('pos_features', Pipeline([
            ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize) ),
        ])),
    ])),
    ('logit', LogisticRegression()),

])


# In[ ]:


pipe2.fit(X_train_part, y_train_part)
pred = pipe2.predict_proba(X_valid)
log_loss(y_valid, pred)


# Not an improvements, but hey, we learned somthing new!
# 
# By this cell, the reader may already feel the power on the new instruments. Are there downsides? Read on.

# <h2>What gets stuck in the pipes?</h2>
# 
# Ok, someone may comment - the way `CounterVectorizer` is usually fit is on the combined train+test datasets, so that the entire vocabulary is learnt. And pipeline accepts one dataset at a time. Yes, this is a problem. The `vocabulary` option won't help us, in case we want to play with ngrams. Is there a work around? Yes. Read on the advanced section
# 
# What about stacking? There are some complex worklfows, for example in [Simple Feature Engg Notebook - Spooky Author](https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author) it is proposed to stack 7 models! Yes, we can do so with piplines, read on.
# 
# 
# It could be slow to run all transformations all over again! True, and I'll show you the way to save time.
# 

# <h3>Overloading CountVectorizer class</h3>

# In[ ]:


class CountVectorizerPlus(CountVectorizer):
    def __init__(self, *args, fit_add=None, **kwargs):
        #this will store a reference to an extra data to include for fitting only
        self.fit_add = fit_add
        super().__init__(*args, **kwargs)
    
    def transform(self, X):
        U = super().transform(X)
        return U
    
    def fit_transform(self, X, y=None):
        if self.fit_add is not None:
            X_new = pd.concat([X, self.fit_add])
        else:
            X_new = X
        #calling CountVectorizer.fit_transform()
        super().fit_transform(X_new, y)

        U = self.transform(X)
        return U
    


# In[ ]:


pipe1a = Pipeline([
    ('cv', CountVectorizerPlus(fit_add=test['text'])),
    #('cv', CountVectorizerPlus()),
    ('tfidf', TfidfTransformer()),
    #('logit', LogisticRegression()),
    ('bnb', BernoulliNB()),
   
])


# In[ ]:


pipe1a.fit(X_train_part, y_train_part)
pred = pipe1a.predict_proba(X_valid)
print(log_loss(y_valid, pred))


# <h3>Stacking with Pipelines</h3>
# 
# If you now try to fit the following pipeline below, with intermediate classifiers, whose output you would like to combine and pass onto the final Classifier, it's going to fail. Why? Pipeline does not like to have more than one final estimator. After all, it's called final because, well, it run as a final step in the pipeline.
# 

# _This cell is Markdown, so the kernel won't stop here._
# ```
# pipe3 = Pipeline([
#     ('u1', FeatureUnion([
#         ('tfdif_features', Pipeline([
#             ('cv', CountVectorizer()),
#             ('tfidf', TfidfTransformer()),
#             ('tfidf_logit', LogisticRegression()),
#         ])),
#         ('pos_features', Pipeline([
#             ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize) ),
#             ('pos_logit', LogisticRegression()),
#         ])),
#     ])),
#     ('xgb', XGBClassifier()),
# 
# ])
# ```

# Happily, there is a solution. We can _pretend_ that our classifier is a transformer class, while it will 'transform' the input data into class predictions. For this, we make a wrapper around an estimator class:

# In[ ]:


#stacking trick
from sklearn.metrics import get_scorer
class ClassifierWrapper(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator, verbose=None, fit_params=None, use_proba=True, scoring=None):
        self.estimator = estimator
        self.verbose = verbose #True = 1, False = 0, 1 - moderately verbose, 2- extra verbose    
        if verbose is None:
            self.verbose=0
        else:
            self.verbose=verbose
        self.fit_params= fit_params
        self.use_proba = use_proba #whether to use predict_proba in transform
        self.scoring = scoring # calculate validation score, takes score function name
        #TODO check if scorer imported?
        self.score = None #variable to keep the score if scoring is set.

    def fit(self,X,y):
        fp=self.fit_params
        if self.verbose==2: print("X: ", X.shape, "\nFit params:", self.fit_params)
        
        if fp is not None:
            self.estimator.fit(X,y, **fp)
        else:
            self.estimator.fit(X,y)
        
        return self
    
    def transform(self, X):
        if self.use_proba:
            return self.estimator.predict_proba(X) #[:, 1].reshape(-1,1)
        else:
            return self.estimator.predict(X)
    
    def fit_transform(self,X,y,**kwargs):
        self.fit(X,y)
        p = self.transform(X)
        if self.scoring is not None:
            self.score = eval(self.scoring+"(y,p)")
            #TODO print own instance name?
            if self.verbose >0: print("score: ", self.score) 
        return p
    
    def predict(self,X):
        return self.estimator.predict(X)
    
    def predict_proba(self,X):
        return self.estimator.predict_proba(X)


# In[ ]:


from xgboost import XGBClassifier
#params are from the above mentioned tutorial
xgb_params={
    'objective': 'multi:softprob',
    'eta': 0.1,
    'max_depth': 3,
    'silent' :1,
    'num_class' : 3,
    'eval_metric' : "mlogloss",
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.3,
    'seed':17,
    'num_rounds':2000,
}


# In[ ]:


pipe3 = Pipeline([
    ('u1', FeatureUnion([
        ('tfdif_features', Pipeline([
            ('cv', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('tfidf_logit', ClassifierWrapper(LogisticRegression())),
        ])),
        ('pos_features', Pipeline([
            ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize) ),
            ('pos_logit', ClassifierWrapper(LogisticRegression())),
        ])),
    ])),
    ('xgb', XGBClassifier(**xgb_params)),
])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pipe3.fit(X_train_part, y_train_part)\npred = pipe3.predict_proba(X_valid)\nprint(log_loss(y_valid, pred))')


# <h3>Caching pipeline results</h3>
# 
# This is possible with the `memory` parameter of the `Pipeline()` constructor. The argument is either path to a directory, or a `joblib` object.

# In[ ]:


pipe4 = Pipeline([
    ('u1', FeatureUnion([
        ('tfdif_features', Pipeline([
            ('cv', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('tfidf_logit', ClassifierWrapper(LogisticRegression())),
        ], memory="/tmp")),
        ('pos_features', Pipeline([
            ('pos', PosTagMatrix(tokenizer=nltk.word_tokenize) ),
            ('pos_logit', ClassifierWrapper(LogisticRegression())),
        ], memory="/tmp")),
    ])),
    ('xgb', XGBClassifier(**xgb_params)),
])


# **I run the same code twice - first time to fit&cache, second time to use cache only**
# 
# Notice the difference! I'd like to warn you however. The cache may not always get invalidated when you think it should. You may want to manually remove the directory of the cache.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'pipe4.fit(X_train_part, y_train_part)\npred = pipe4.predict_proba(X_valid)\nprint(log_loss(y_valid, pred))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pipe4.fit(X_train_part, y_train_part)\npred = pipe4.predict_proba(X_valid)\nprint(log_loss(y_valid, pred))')


# <h2>Submission</h2>
# 
# Here we show how easy is it to train the model on the full train dataset and generate predictions for the test one.

# In[ ]:


#refit on the full train dataset
pipe4.fit(train['text'], lb.transform(train['author']))

# obtain predictions
pred = pipe4.predict_proba(test['text'])

#id,EAP,HPL,MWS
#id07943,0.33,0.33,0.33
#...
pd.DataFrame(dict(zip(lb.inverse_transform(range(pred.shape[1])),
                      pred.T
                     )
                 ),index=test.id).to_csv("submission.csv", index_label='id')


# <h2>Conclusions and further reading</h2>
# 
# Ok, we didn't win, but I din't promice. :) I'll stop here and let the reader add his/her own features and models, stack them and hopefully, rocket to the top!
# 
# What else can you learn about pipelines?
# 
# - Go to the doc page, http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html, and check out some examples linked to at the end of the page.
# 
# - Take a look at these tutorial and vote them up, if you like them.
#     - [pipeline for the beginners](https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines)
#     - [Simple Feature Engg Notebook - Spooky Author](https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author) - try to implement all the models in one pipeline.
# - Fork this kernel and explore your own ideas!
# 
# <h2>Good luck!</h2>
# 
