
# coding: utf-8

# Once you've gotten your feet wet in basic sklearn modeling, you might find yourself doing the same few steps over and over again in the same anaysis. To get to the next level, pipelines are your friend!
# 
# Pipelines are a way to streamline a lot of the routine processes, encapsulating little pieces of logic into one function call, which makes it easier to actually do modeling instead just writing a bunch of code. Pipelines allow for experiments, and for a dataset like this that only has the text as a feature, you're going to need to do a lot of experiments. Plus, when your modeling gets really complicated, it's sometimes hard to see if you have any data leakage hiding somewhere. Pipelines are set up with the fit/transform/predict functionality, so you can fit a whole pipeline to the training data and transform to the test data, without having to do it individually for each thing you do. Super convenienent, right??
# 
# This notebook is going to break down the pipeline process to make it easier to see how they all fit together. While not exhaustive, it should get you started on building your own pipelines so you can spend more time on the good stuff, thinking.
# 
# But first, we get the data:

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/train.csv')

df.dropna(axis=0)
df.set_index('id', inplace = True)

df.head()


# ## Preprocessing and Feature Engineering
# 
# To begin, let's do some basic feature engineering. To make it easier to replicate on the submission data, we will encapsulate the logic into a function.
# 
# Note, all of this preprocessing is standard stuff, and does not depend on the data it's processing on, so it's ok to do this now. Things like count vectorization and numeric scaling depend on the data it's run on, so that part must be done differently. We will get to that later.
# 
# For now, we will count the number of words in each row, the number of characters, the number of non stop words, and the number of commas, since who knows, maybe using commas helps build suspense??

# In[2]:


import re
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

#creating a function to encapsulate preprocessing, to mkae it easy to replicate on  submission data
def processing(df):
    #lowering and removing punctuation
    df['processed'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]','', x.lower()))
    
    #numerical feature engineering
    #total length of sentence
    df['length'] = df['processed'].apply(lambda x: len(x))
    #get number of words
    df['words'] = df['processed'].apply(lambda x: len(x.split(' ')))
    df['words_not_stopword'] = df['processed'].apply(lambda x: len([t for t in x.split(' ') if t not in stopWords]))
    #get the average word length
    df['avg_word_length'] = df['processed'].apply(lambda x: np.mean([len(t) for t in x.split(' ') if t not in stopWords]) if len([len(t) for t in x.split(' ') if t not in stopWords]) > 0 else 0)
    #get the average word length
    df['commas'] = df['text'].apply(lambda x: x.count(','))

    return(df)

df = processing(df)

df.head()


# ### Creating a Pipeline
# 
# Sklearn's pipeline functionality makes it easier to repeat commonly occuring steps in your modeling process. Similar to the processing function I made above, it provides a way to take code, fit it to the training data, apply it to the test data without having to copy and paste everything.
# 
# Super easy, but I find the documentation a little hard to piece through. So let's build the pipelines up from the bottom. Plus, since pipelines are made from pipelines, it's useful to see how they build on each other.
# 
# First step, split your data into training and testing.

# In[3]:


from sklearn.model_selection import train_test_split

features= [c for c in df.columns.values if c  not in ['id','text','author']]
numeric_features= [c for c in df.columns.values if c  not in ['id','text','author','processed']]
target = 'author'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33, random_state=42)
X_train.head()


# Now for the tricky parts.
# 
# First thing I want to do is define how to process my variables. The standard preprocessing apply the same preprocessing to the whole dataset, but in cases where you have heterogeneous data, this doesn't quite work. So first thing I'm going to do is create a selector transformer that simply returns the one column in the dataset by the key value I pass. 
# 
# I was having difficulty getting the selector to play nicely, so I made two different selectors for either text or numeric columns. The return type is different, but other than that they work the same.

# In[4]:


from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]
    


# To see how this is used, let's actually run it on one column.
# 
# I'm going to call it on the text column and transform it with another step. But again, pipelines are all about encapsulating several steps, so I'm going to make a mini pipeline that consists of two steps: first grab just that column from the dataset, then perform tf-idf on just that column and return the results.
# 
# To make a pipeline, just pass an array of tuples of the format (name, object). The first part is the name of the action, and the second is the actual object. So this pipeline consists of "selecting" and then "tfidf-ing" a column.
# 
# To execute, use it just like any other transformer. You can call text.fit() to fit to training data, text.transform() to apply it to training data, or text.fit_transform() to do both. 
# 
# Since it's text, it will return a sparse matrix, but we can see that it works:

# In[17]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

text = Pipeline([
                ('selector', TextSelector(key='processed')),
                ('tfidf', TfidfVectorizer( stop_words='english'))
            ])

text.fit_transform(X_train)


# Since our data is heterogeneous, we might want to do something else on numeric data, so let's build a mini pipeline for that too.
# 
# This transformer will be a simple scaler. Since our data is mixed, we must apply it column by column. Let's make one to process the "length" variable I made above. Just like the text one, we combine two steps, first selecting the column, then transforming the column, like so:

# In[6]:


from sklearn.preprocessing import StandardScaler

length =  Pipeline([
                ('selector', NumberSelector(key='length')),
                ('standard', StandardScaler())
            ])

length.fit_transform(X_train)


# We can see that the transformer pipeline returns a matrix for the column it's called on, so now all that's left to do is join the results from several transformed variables into a single dataset. I'll go ahead and make a pipeline for every variable in the data, then join them all together. 
# 
# First, I'll transform all the numeric columns with the standard scaler, but of course you can change the scaler for any column as you desire.

# In[7]:


words =  Pipeline([
                ('selector', NumberSelector(key='words')),
                ('standard', StandardScaler())
            ])
words_not_stopword =  Pipeline([
                ('selector', NumberSelector(key='words_not_stopword')),
                ('standard', StandardScaler())
            ])
avg_word_length =  Pipeline([
                ('selector', NumberSelector(key='avg_word_length')),
                ('standard', StandardScaler())
            ])
commas =  Pipeline([
                ('selector', NumberSelector(key='commas')),
                ('standard', StandardScaler()),
            ])


# To make a pipeline from all of our pipelines, we do the same thing, but now we use a FeatureUnion to join the feature processing pipelines.
# 
# The syntax is the same as a regular pipeline, it's just an array of tuple, with the (name, object) format. 
# 
# The feature union itself is not a pipeline, it's just a union, so you need to do *one more step* to make it useable: pass it to a pipeline, with the same structure, an array of tuples, with the simple (name, object) format. . As you can see, we get a pipeline-ception going on the more complex you get! 
# 
# You can then apply all those transformations at once with a single fit, transform, or fit_transform call. Nice, right?

# In[8]:


from sklearn.pipeline import FeatureUnion

feats = FeatureUnion([('text', text), 
                      ('length', length),
                      ('words', words),
                      ('words_not_stopword', words_not_stopword),
                      ('avg_word_length', avg_word_length),
                      ('commas', commas)])

feature_processing = Pipeline([('feats', feats)])
feature_processing.fit_transform(X_train)


# To add a model to the mix and generate predictions as well, you can add a model at the end of the pipeline. The syntax is, you guessed it, an array of tuples, merging the transformations with a model. 
# 
# We can see the raw accuracy is at 63%. Not bad for a start.
# 

# In[12]:


from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('features',feats),
    ('classifier', RandomForestClassifier(random_state = 42)),
])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
np.mean(preds == y_test)


# # Bringing It All Together
# 
# Preprocessing is great, but most likely, you actually want to model something too. To do that, we just need to add a classifier at the end of the pipeline. Here I'm going to make a pipeline that does all the processing I made above, then passes it to a Random Forest. As you might guess, this requires passing an array of tuples to the pipeline, with the (name, object) structure. 
# 
# To put it to use, we just fit and predict as if it was a regular classifier. First we fit on the training, then predict on the test data, and then see how well it did:

# # Cross Validation To Find The Best Pipeline
# 
# That alone should give you enough flexibility to create some rather complex pipelines. But we're on a role, let's keep going.
# 
# What if I wanted to do cross validation on my pipeline? How many trees should I use on my classifier? How deep should I go? Or even more complicated, how many words should I use in my tf-idf transform? Should I include stop words? Pipelines allow you to do that with just a few more lines.
# 
# Cross validation is all about figuring out what the best hyperparameters of the data set is. To see the list of all the possible things you could fine tune, call get_params().keys() on your pipeline.

# In[13]:


pipeline.get_params().keys()


# Obviously don't be crazy, cross validation takes a while to run, and the more options you select, the longer it takes. But to give you an idea on how these work together, to test out the different combinations, define a dictionary with the settings you want, with the key being the pipeline's parameter key name, and the value being an array of all the settings you want to apply.
# 
# After the dictionary is made, call GridSearchCV on your pipeline, passing the dictionary and the number of folds you want to use. 
# 
# Here's an example with a few settings on 5 fold cross validation.

# In[22]:


from sklearn.model_selection import GridSearchCV

hyperparameters = { 'features__text__tfidf__max_df': [0.9, 0.95],
                    'features__text__tfidf__ngram_range': [(1,1), (1,2)],
                   'classifier__max_depth': [50, 70],
                    'classifier__min_samples_leaf': [1,2]
                  }
clf = GridSearchCV(pipeline, hyperparameters, cv=5)
 
# Fit and tune model
clf.fit(X_train, y_train)


# If you want to see which settings won, you can do so:

# In[23]:


clf.best_params_


# What's really convenient is you can call refit to automatically fit the pipeline on all of the training data with the best_params_setting applied!
# 
# Then applying it to the test data is the same as before.
# 
# Not much of an improvement, but at least now we can go back and easily change out the individual pieces.

# In[24]:


#refitting on entire training data using best settings
clf.refit

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)

np.mean(preds == y_test)


# # Final Predictions
# 
# To generate submission results, you just need to do the preprocessing on the submission data, then call the pipeline with the predict_proba call, since we want to know all the probabilities, not just the label.
# 
# The only tricky part for the submission is we need the class names as the column values. To access it, you must call clf.best_estimator_.named_steps['classifier'].classes_

# In[28]:


submission = pd.read_csv('../input/test.csv')

#preprocessing
submission = processing(submission)
predictions = clf.predict_proba(submission)

preds = pd.DataFrame(data=predictions, columns = clf.best_estimator_.named_steps['classifier'].classes_)

#generating a submission file
result = pd.concat([submission[['id']], preds], axis=1)
result.set_index('id', inplace = True)
result.head()


# ### Wrapping Up
# 
# I hope this helps shed some light on the inner workings of pipelines. Using pipelines effectively can really help elevate you to the next level of data scientist, so once you've mastered the algorithms themselves, I strongly recommend mastering pipelines as well! You can even create a pipeline to test out different classifiers and pick the best one too! It's one step closer to automating away the boring stuff, letting you focus on what matters, the creativity and feature engineering.
# 
# More to come, so stay tuned!
