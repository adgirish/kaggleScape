
# coding: utf-8

# We'll take the ideas introduced in Rachael Tatman's [Beginner Tutorial: Python](https://www.kaggle.com/rtatman/beginner-s-tutorial-python), scale them up into a complete machine learning pipeline, and add some basic feature engineering.
# 
# First, we need to make quite a few  imports. It's a long list but everything important is a component of either [pandas](http://pandas.pydata.org/pandas-docs/stable/) for data manipulation, [spaCy](https://spacy.io/docs/usage/lightning-tour) for text processing, or [scikit-learn](http://scikit-learn.org/stable/) for machine learning. I'll assume you have general familiarity with both pandas and scikit-learn.

# In[ ]:


import pandas as pd
import spacy

from multiprocessing import cpu_count
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from spacy import attrs
from spacy.symbols import VERB, NOUN, ADV, ADJ


# Next we'll declare important constants, per the python style guide, [PEP8](https://www.python.org/dev/peps/pep-0008). This isn't strictly necessary, but makes for cleaner code.

# In[ ]:


TEXT_COLUMN = 'text'
Y_COLUMN = 'author'


# We're going to run a couple of different models with different sets of features, so it's worth taking a moment to set up our model evaluation process as its own function.
# 
# For evaulation, we need to do several things:
# 1. Split the input dataframe into the a feature dataframe and a label dataframe (X and Y).
# - Conduct  feature engineering.
# - Train the model.
# - Perform cross validation.
# - Report the relevant score. In this case, we'll use log loss to match the competition's evaluation.
# 
# Integrating [Scikit-learn pipelines](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) into our evaluation makes this straightforward to repeat with different models and features.

# In[ ]:


def test_pipeline(df, nlp_pipeline, pipeline_name=''):
    y = df[Y_COLUMN].copy()
    X = pd.Series(df[TEXT_COLUMN])
    # If you've done EDA, you may have noticed that the author classes aren't quite balanced.
    # We'll use stratified splits just to be on the safe side.
    rskf = StratifiedKFold(n_splits=5, random_state=1)
    losses = []
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nlp_pipeline.fit(X_train, y_train)
        losses.append(metrics.log_loss(y_test, nlp_pipeline.predict_proba(X_test)))
    print(f'{pipeline_name} kfolds log losses: {str([str(round(x, 3)) for x in sorted(losses)])}')
    print(f'{pipeline_name} mean log loss: {round(pd.np.mean(losses), 3)}')


# We're ready to load the data and run our first model. We'll start with the exact same model,
# a naive bayes classifer on unigram probabilities, as in Rachael's tutorial. Using sklearn instead of implementing everything ourselves will make this both easier to code up and faster to run.
# 
# The `Id` column doesn't actually help us (or if it does, isn't really in the spirit of an NLP competition), so we'll skip over it.

# In[ ]:


train_df = pd.read_csv("../input/train.csv", usecols=[TEXT_COLUMN, Y_COLUMN])


# In[ ]:


unigram_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('mnb', MultinomialNB())
                        ])
test_pipeline(train_df, unigram_pipe, "Unigrams only")


# Since we want to turn this into a nice clean pipeline, we'll do all of our feature engineering using custom transformers.  This first transformer takes the unigram pipeline that we built above and returns the predicted probabilities as features. We could use the raw CountVectorizer output and let our final model deal with the unigram features directly, but that would create two issues:
# 
# - CountVectorizer returns [a sparse format](https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.sparse.csr_matrix.html) that is a pain to integrate with the rest of our pipeline. 
# - Using CountVectorizer and MultinomialNB allows us to skip converting the word counts to probabilities, and to skip ensuring that probabilities are never exactly zero. See the `alpha` parameter in the [MultinomialNB documentation?](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB) As long as we use the default input of one, the model will peform this task (a [Laplace transform](https://en.wikipedia.org/wiki/Additive_smoothing)) for us.

# In[ ]:


class UnigramPredictions(TransformerMixin):
    def __init__(self):
        self.unigram_mnb = Pipeline([('text', CountVectorizer()), ('mnb', MultinomialNB())])

    def fit(self, x, y=None):
        # Every custom transformer requires a fit method. In this case, we want to train
        # the naive bayes model.
        self.unigram_mnb.fit(x, y)
        return self
    
    def add_unigram_predictions(self, text_series):
        # Resetting the index ensures the indexes equal the row numbers.
        # This guarantees nothing will be misaligned when we merge the dataframes further down.
        df = pd.DataFrame(text_series.reset_index(drop=True))
        # Make unigram predicted probabilities and label them with the prediction class, aka 
        # the author.
        unigram_predictions = pd.DataFrame(
            self.unigram_mnb.predict_proba(text_series),
            columns=['naive_bayes_pred_' + x for x in self.unigram_mnb.classes_]
                                           )
        # We only need 2 out of 3 columns, as the last is always one minus the 
        # sum of the other two. In some cases, that colinearity can actually be problematic.
        del unigram_predictions[unigram_predictions.columns[0]]
        df = df.merge(unigram_predictions, left_index=True, right_index=True)
        return df

    def transform(self, text_series):
        # Every custom transformer also requires a transform method. This time we just want to 
        # provide the unigram predictions.
        return self.add_unigram_predictions(text_series)


# It's time to start adding new features with spaCy. We'll flag the main parts of speech used in each sentence, average word length, and overall sentence length.
# 
# The single slowest step of working with spaCy is often loading the model in the first place, so we'll ensure this step is only done once. By default, spaCy will tag each word, build a dependency model, and perform entity recognition. We only need the part of speech tags, so we'll restrict the pipeline accordingly. In tests on my local machine, this sped up the parse by 5-10x.

# In[ ]:


NLP = spacy.load('en', disable=['parser', 'ner'])


# In[ ]:


class PartOfSpeechFeatures(TransformerMixin):
    def __init__(self):
        self.NLP = NLP
        # Store the number of cpus available for when we do multithreading later on
        self.num_cores = cpu_count()

    def part_of_speechiness(self, pos_counts, part_of_speech):
        if eval(part_of_speech) in pos_counts:
            return pos_counts[eval(part_of_speech).numerator]
        return 0

    def add_pos_features(self, df):
        text_series = df[TEXT_COLUMN]
        """
        Parse each sentence with part of speech tags. 
        Using spaCy's pipe method gives us multi-threading 'for free'. 
        This is important as this is by far the single slowest step in the pipeline.
        If you want to test this for yourself, you can use:
            from time import time 
            start_time = time()
            (some code)
            print(f'Code took {time() - start_time} seconds')
        For faster functions the timeit module would be standard... but that's
        meant for situations where you can wait for the function to be called 1,000 times.
        """
        df['doc'] = [i for i in self.NLP.pipe(text_series.values, n_threads=self.num_cores)]
        df['pos_counts'] = df['doc'].apply(lambda x: x.count_by(attrs.POS))
        # We get a very minor speed boost here by using pandas built in string methods
        # instead of df['doc'].apply(len). String processing is generally slow in python,
        # use the pandas string methods directly where possible.
        df['sentence_length'] = df['doc'].str.len()
        # This next step generates the fraction of each sentence that is composed of a 
        # specific part of speech.
        # There's admittedly some voodoo in this step. Math can be more highly optimized in python
        # than string processing, so spaCy really stores the parts of speech as numbers. If you
        # try >>> VERB in the console you'll get 98 as the result.
        # The monkey business with eval() here allows us to generate several named columns
        # without specifying in advance that {'VERB': 98}.
        for part_of_speech in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            df[f'{part_of_speech.lower()}iness'] = df['pos_counts'].apply(
                lambda x: self.part_of_speechiness(x, part_of_speech))
            df[f'{part_of_speech.lower()}iness'] /= df['sentence_length']
        df['avg_word_length'] = (df['doc'].apply(
            lambda x: sum([len(word) for word in x])) / df['sentence_length'])
        return df

    def fit(self, x, y=None):
        # since this transformer doesn't train a model, we don't actually need to do anything here.
        return self

    def transform(self, df):
        return self.add_pos_features(df.copy())


# Finally, sklearn models generally don't accept strings as inputs, so we'll need to drop all string columns. This includes the original
# 'text' column that we read from the csv!

# In[ ]:


class DropStringColumns(TransformerMixin):
    # You may have noticed something odd about this class: there's no __init__!
    # It's actually inherited from TransformerMixin, so it doesn't need to be declared again.
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        for col, dtype in zip(df.columns, df.dtypes):
            if dtype == object:
                del df[col]
        return df


# If you want to experiment with different combinations of features, try writing your own transformers and adding them to the pipeline.
# 
# If you're running this at home, expect this next step to take ~30 seconds or so as we're retraining the model several times during the cross validation.

# In[ ]:


logit_all_features_pipe = Pipeline([
        ('uni', UnigramPredictions()),
        ('nlp', PartOfSpeechFeatures()),
        ('clean', DropStringColumns()), 
        ('clf', LogisticRegression())
                                     ])
test_pipeline(train_df, logit_all_features_pipe)


# This pipeline is better... but only just barely. I'll leave it as an exercise for you to add better features and more powerful models. However, if we did want to submit this, we'd just feed `logit_all_features_pipe` into the `generate_submission_df` function.

# In[ ]:


def generate_submission_df(trained_prediction_pipeline, test_df):
    predictions = pd.DataFrame(
        trained_prediction_pipeline.predict_proba(test_df.text),
        columns=trained_prediction_pipeline.classes_
                               )
    predictions['id'] = test_df['id']
    predictions.to_csv("submission.csv", index=False)
    return predictions


# Exercises:
# 1. Update the `PartOfSpeechFeatures` transformer to record all parts of speech, not just the original four.
# - Can you generate a useful feature with spaCy's dependency parser? Fair warning,I haven't tried yet so the answer may well be no!
# - More challenging: Kevin Schiroo figured out that [sentences for MWS are missing exclamation marks](https://www.kaggle.com/c/spooky-author-identification/discussion/42135). A simple regex based on capital letters like `re.sub(r'\b (?=[A-Z])', '! ', sentence)` would insert too many exclamation points by treating names as ends of sentences. Can you use spaCy's entity recognition model to clean those up?
