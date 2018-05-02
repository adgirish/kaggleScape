
# coding: utf-8

# #### Last Update: 2017-10-27
# ***
# 
# # Tidy Notebook
# ***
# The idea for this notebook, is to also try to keep it as tidy as possible, in order to make it easy to understand for the reader. This includes trying to write every analysis we do on the markdown cells, and explain the code as much as possible on the code comments. ** COMMENTS ARE EVERY PROGRAMMERS FRIENDS **
# 
# **# About me
# ***
# I've been a developer for more than 12 years and recently, just a few months ago, I got captivated by the magic of machine learning, and how easy it got for us programmers to get into it. My main motivation is learning and grow in the Machine Learning area, but in the process, I'd love to help anyone who's not been on the programming field, to make their career path easier. <br>
# > Finally we get to make AI!
# 
# I'm from Buenos Aires, Argentina, 33 years old.<br>
# If you want; you can reach me on:
# * Linked In: https://www.linkedin.com/in/juanuhaedo/
# * Twitter: https://www.twitter.com/juanumusic/
# * GitHub: https://github.com/HarkDev/
# 
# **Don't forget to upvote if you liked this notebook!**

# # 1. First Steps
# ***
# ## 1.1 Import Required Libraries
# ***
# The first steps on every python script/notebook is to import the required libraries.<br>
# **What are libraries?**
# > Libraries are like pieces of code that do a lot of stuff, and that we use, so that we do not need to write them all over. 
# 
# We'll import the following libraries:
# * **numpy**: for linear algebra operations.
# * **pandas**: for data processing, reading csv files, etc.[](http://)
# * **matplotlib**: most used library for plotting on python.
# * **seaborn**: A library that used matplotlib and wraps functionality, so that drawing plots is even easier.
# * **nltk**: Natural Language Toolkit. We will use some of the functions from the library, but not all of them.
# * **string**: Python's string library

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Library for drawing plots.
import seaborn as sns # An extension of matplotlib that helps make plots easier in less code
import nltk
import string


# ## 1.2 Read CSV files.
# ***
# Using pandas we will read the datasets provided, which are in CSV format.<br>
# **What is CSV format?**
# > CSV stands for: *"comma separated values* and is one of the most common ways to store information on plain text file.
# 
# Let's read the csv into two variables:
# * **df_train** will contain the train information
# * **df_test** will contain the test information**

# In[ ]:


df_train = pd.read_csv('../input/train.csv') # Train dataset
df_test = pd.read_csv('../input/test.csv') # Test dataset


# ## 1.3 Take a peek at the data
# ***
# It's always important to actually look at the data loaded, in order to know that we are dealing with what we had in mind. In this case, we can see that the data contained, are fragments of books, identified by their author. <br>
# * The `id` column is a unique value that identified the row. We will discard this for our predictions.
# * The `text` column contains some text written by a specific author.
# * The `author` column, contains a string, that's identifies the author that wrote the text. This is the value that we will use to train our model and let it know to whom it corresponds, so that when me make the predictions, our model will know what values to predict.
# 
# The `.sample(10)` function of pandas, displays a random sample from the entire dataframe, with the size passed. In this case, we are going to take a peek at 10 samples.

# In[ ]:


df_train.sample(10) # Look at 10 random samples of the dataframe


# # 2 Data Munging
# ***
# We will now process the data, in order to try to make the best features for when we need to make our predictions.
# 
# There are great libraries for working with language, and text. The actual state of the art is NLTK(Natural Language Toolkit). This tutorial will cover the basics of treating text, without the help of NLTK, in order to understand what happens behind this library, although not everything that it does.
# 
# ## 1.2 Feature Engineering
# ***
# ### 1.2.1 Remove Punctuation
# ***
# Removing punctiation is important when you work with text, since you don't want your predictions to take special characters like '.', ',' or ':' into account.
# 
# On this cell, I'll show you how to make this removal by hand, even though there are many libaries to do this (including NLTK).
# 
# The punctuation attribute in `string.punctuation` contains a list of punctuations so we don't have to type them by hand.

# In[ ]:



# We create a function to do the punctuation removal
def remove_punctuation(text):

    # For each punctuation in our list
    for punct in string.punctuation:
        # Replace the actual punctuation with a space.
        text = text.replace(punct,'')

    # Return the new text
    return text

# Now, we will apply the remove punctiation to all our text
df_train.text = df_train.text.apply(remove_punctuation)
df_test.text = df_test.text.apply(remove_punctuation)


# ### 1.2.2 Stemming
# ***
# Stemming is the process of converting words that mean the same. For example, verb conjugation. Words like run, running, ran, all are the same word on different conjugations.
# 
# We will use NLTK for this technique.
# 
# ***
# A side note here. You will see that the way I apply stemming is in one line of code with a lot of things going on. I will try to explain the best I can what's going on...
# 
# 1.  In every one-liner code, you need to try to understand whats happening first in all of that. In our case, the first thing thats happening is that we are splitting our text by spaces :`text.split(" ")`. This returns a list of words.
# 2. We create a for loop (list comprehension), looping on each word and appying the stem using `stemmer.stem(word)`.
# 3. We rejoin the words, with a space, to get the sentence stemmed using `" ".join(...)`
# 
# **NOTE:** If the explanation above is not clear, please, leave a comment with the stuff you don't understand and I will try to correct the best possible way.
# 
# # STEMMING WARNING
# ***
# For this challenge, the idea is to classify to whom the text belongs

# In[ ]:


APPLY_STEMMING = False

if APPLY_STEMMING:
    import nltk.stem as stm # Import stem class from nltk
    import re
    stemmer = stm.PorterStemmer()

    # Crazy one-liner code here...
    # Explanation above...
    df_train.text = df_train.text.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))
    df_test.text = df_test.text.apply(lambda text: " ".join([stemmer.stem(word) for word in text.split(" ")]))


# ### 1.2.3 Vectorization
# One of the most common feature engineering when dealing with text, it's vectorization.<br>
# On this notebook we will make use of two different implementations: **Count Vectorizer** and **TfidfVectorizer¶**
# 
# Both work by counting each word on a corpus of text, so here I will show code for each one, but both have the same following parameters which are important to understand:
# 
# #### NGram Range
# ***
# Ngram range is a parameter that will do the vectorization for each word, and for each combination of n words. For example, if `ngram_range =(1,2)` the vectorization will be:
# 
# |the|cat|is|on|table|the_cat|cat_is|is_on|on_the|the_table|
# |---|---|--|--|-----|-|-|-|-|-|
# |2|1|1|1|1|1|1|1|1|1|
# 
# ### Stop Words
# ***
# Finally, the stop words parameter will let us remove the stop words such as *the, a, and, or, etc...*. We can pass the name of the language (scikit learn only comes with english by default) or a list of words, on our example, we will use **english**.
# 
# ###  1.2.3.1 Count Vectorizer
# ***
# CountVectorizer takes each text, and creates a column for each word that exists on the corpus, and sets the number of times that that word repeats, on the column, for a given text.<br>
# 
# For example, the text *"the cat is on the table"* will be vectorized as:
# 
# |the|cat|is|on|table|
# |---|---|--|--|-----|
# |2|1|1|1|1|

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer # Import the library to vectorize the text

# Instantiate the count vectorizer with an NGram Range from 1 to 3 and english for stop words.
count_vect = CountVectorizer(ngram_range=(1,3),stop_words='english')

# Fit the text and transform it into a vector. This will return a sparse matrix.
count_vectorized = count_vect.fit_transform(df_train.text)


# ### 1.2.3.2 TfidfVectorizer
# ***
# TfidfVectorizer is another way of treating each word. The name *TfIdf* stands for *"Term Frequency Inverse Document Frequency*. Since [SciKit's learn documentation explains it so well](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html), I'm just going to cite what this does:
# > Scales down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.
# 
# As you can see, her we set a value of (1,1) for ngram_range. This is because, after trying a few values, I found that the best accuracy on predictions, using TfIdf, is to use an ngram_range of 1. <br>
# The process of trying different values is called **Hyperparameter Tuning**.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer # Import the library to vectorize the text

# Instantiate the count vectorizer with an NGram Range from 1 to 3 and english for stop words.
tfidf_vect = TfidfVectorizer(ngram_range=(1,3), stop_words='english')

# Fit the text and transform it into a vector. This will return a sparse matrix.
tfidf_vectorized = tfidf_vect.fit_transform(df_train.text)


# # 3 Model
# ***
# It's now time to prepare our model. We will prepare the data, validate itm and finally make the predictions.
# 
# ## 3.1 Split Train and Test for validation
# ***
# It's important to validate our model, so that we know what we are working with. In order to validate it, we need to train it with some values, and then, predict other values that we know the real answer.
# 
# To do this, we will split the train data, since it contains the target values (author). We will use a piece of the data to train the model, and the other piece, to make the predictions and then validate the predictions.
# 
# Remember that we played with both **CountVectorizer** and **TfidfVectorizer** so from now on we will create two models and see which one performs better.

# In[ ]:


from sklearn.model_selection import train_test_split # Import the function that makes splitting easier.

# Split the vectorized data. Here we pass the vectorized values and the author column.
# Also, we specify that we want to use a 75% of the data for train, and the rest for test.

###########################
# COUNT VECTORIZED TOKENS #
###########################
X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(count_vectorized, df_train.author, train_size=0.75)

###########################
# TFIDF VECTORIZED TOKENS #
###########################
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_vectorized, df_train.author, train_size=0.75)


# ## 3.2 Multinomial Naive Bayes
# ***
# Multinomial Naive Bayes is one of the most common algorithm used in text clasification, so we will work with it.

# In[ ]:


# First, import the Multinomial Naive bayes library from sklearn 
from sklearn.naive_bayes import MultinomialNB

# Instantiate the model.
# One for Count Vectorized words
model_count_NB = MultinomialNB()
# One for TfIdf vectorized words
model_tfidf_NB = MultinomialNB()

# Train the model, passing the x values, and the target (y)
model_count_NB.fit(X_train_count, y_train_count)
model_tfidf_NB.fit(X_train_tfidf, y_train_tfidf)


# ## 3.3. Predict test values
# ***
# Once we have our model trained, we can predict the test values, and then compare them to the real values. 

# In[ ]:


# Predict the values, using the test features for both vectorized data.
predictions_count = model_count_NB.predict(X_test_count)
predictions_tfidf = model_tfidf_NB.predict(X_test_tfidf)


# ## 3.4 Validate the model
# ***
# Next step is validate the model. We will import the `accuracy_score` function of sklearn, to see how is our accuracy.

# In[ ]:


# Primero calculamos el accuracy general del modelo
from sklearn.metrics import accuracy_score
accuracy_count = accuracy_score(y_test_count, predictions_count)
accuracy_tfidf = accuracy_score(y_test_tfidf, predictions_tfidf)
print('Count Vectorized Words Accuracy:', accuracy_count)
print('TfIdf Vectorized Words Accuracy:', accuracy_tfidf)


# The accuracy seems pretty good, let's look at the confusion matrix.
# ** What is confusion matrix?**
# > A confusion matrix is a matrix where we can see where the predicted values are, and where they should be.

# In[ ]:


# Import the confusion matrix method from sklearn
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix passing the real values and the predicted ones
# Count
conf_mat_count = confusion_matrix(y_test_count, predictions_count)
# tfIdf
conf_mat_tfidf = confusion_matrix(y_test_tfidf, predictions_tfidf)

# Set plot size
plt.figure(figsize=(12,10))
# Use 2 subplots.
plt.subplot(1,2,1)

# Finally, plot the confusion matrix using seaborn's heatmap.
sns.heatmap(conf_mat_count.T, square=True, annot=True, fmt='d', cbar=True,
            xticklabels=y_test_count.unique(), yticklabels=y_test_count.unique())
plt.xlabel('True values')
plt.ylabel('Predicted Values');
plt.title('Count Vectorizer', fontsize=16)

plt.subplot(1,2,2)
# Finally, plot the confusion matrix using seaborn's heatmap.
sns.heatmap(conf_mat_tfidf.T, square=True, annot=True, fmt='d', cbar=True,
            xticklabels=y_test_tfidf.unique(), yticklabels=y_test_tfidf.unique())
plt.xlabel('True values')
plt.ylabel('Predicted Values');
plt.title('TfIdf Vectorizer', fontsize=16)


# #### Count Vectorizer
# We can see that:
# * For EAP, we correctly predicted 1309 cases, and missed 155 + 78
# * For MWS, we correctly predicted 1117 cases, and missed 145 + 106
# * For HPL, we correctly predicted 1613 cases, and missed 138 + 234
# 
# #### TfIdf Vectorizer
# We can see that:
# * For EAP, we correctly predicted 1742 cases, and missed 83 + 167
# * For MWS, we correctly predicted 999 cases, and missed 301 + 85
# * For HPL, we correctly predicted 1241 cases, and missed 238 + 39
# 
# > **Our best model seems to be the one with CountVectorizer, so we will make our submission with that model.**

# # 4 Final prediction and submission
# ***
# We have our model trained, with a somehow good accuracy. We will now train the model with the entire dataset, in order to show most of the data we have, and then we'll make the predictions wih the test dataset.
# 
# ## 4.1 Train with full dataset
# ***

# In[ ]:


# Instantiate the model.
model_NB = MultinomialNB()

# Train the model, passing the x values, and the target (y)
# the vectorized variable contains all the test data.
model_NB.fit(count_vectorized, df_train.author)


# ## 4.2 Remove punctiation and vectorize test with the same vectorizer
# ***
# Our predictive model needs to receive as inputs, the same features with which it was trained<br>
# To do this, we will be using the CountVectorizer we fitted at the begining of this notebook.

# In[ ]:


# Transform the text to a vector, with the same shape of the trained data.
X_test = count_vect.transform(df_test.text)


# ## 4.3 Predict the values
# ***
# Now it's time to make the predictions. We wil lnow use predict_proba instead of predict. This will return an array with the probabilities of being a certain author, instead of the class itself.

# In[ ]:


# Run the prediction
predicted_values = model_NB.predict_proba(X_test)


# ## 4.4 Generate Submission File
# ***
# With our values predicted, we can generate our submission file.  The submission file sholud contain the following columns:
# * id
# * EAP
# * HPL
# * MWS
# 
# If we look at the `classes_` attribute of the model, we can see that these are in the same order needed.

# In[ ]:


model_NB.classes_


# Finally, we can now create our sumissions file. What I like to do, is to automatically set the date and time in which this file was generated and the accuracy. All on the filename.

# In[ ]:


# Import the time library
import time

# Create the submission dataframe
df_submission = pd.DataFrame({
    'id': df_test.id.values,
    'EAP': predicted_values[:,0],
    'HPL': predicted_values[:,1],
    'MWS': predicted_values[:,2]
})


# Create the date and time string. (year month day _ hours minutes seconds)
datetime = time.strftime("%Y%m%d_%H%M%S")

# generate the file name with the date, the time and the accuracy of the count vectorized test.
filename = 'submission_' + datetime + '_acc_' + str(accuracy_count) + '.csv'

# Finally, convert it to csv. Index=True tells pandas not to include the index as a column
df_submission.to_csv(filename, index=False)

print('File',filename,'created.')


# # Foreword
# ***
# Hope this tutorial has elped anyone starting with machine learning with text corpuses. If you use this notebook, or fork it, please, cite me anywhere, or at least, leave a comment!<br>
# If you find any corrections, mistakes, or things to add, please, feel free to let me know on the comments too!

# * **Don't forget to upvote and/or comment if you liked this notebook!**
