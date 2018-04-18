
# coding: utf-8

# #**Hey Kagglers, this notebook will go over a couple of basic Natural Language Processing techniques using the [scikit-learn](http://scikit-learn.org/stable/) library. Special thanks to [Aaron7sun](https://www.kaggle.com/aaron7sun) for providing the [dataset](https://www.kaggle.com/aaron7sun/stocknews) and "assignment" for us to do!**  
# #Thanks for reading and feel free to leave feedback. I'd like to make more tutorial notebooks like this, so let me know if you think there is anything I could improve.

# ----------

# # Notebook Prep

# First things first, let's import the libraries we'll be using.  

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# [Pandas](http://pandas.pydata.org/) will make our data easy to look at and work with.  
# [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), part of scikit-learn, will take care of our NLP tasks.  
# [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), also part of scikit-learn, will train and test our predictive models.

# ----------

# # Data Import

# Now, let's [read](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) in the data with Pandas.  
# If you're working in something other than a Kaggle notebook, be sure to change the file location.  
# For this tutorial, we're just going to use the combined dataset that Aaron prepared for us, but you're welcome to import the other two CSV files if you want to combine them in a different way.

# In[ ]:


data = pd.read_csv('../input/Combined_News_DJIA.csv')


# Next, let's take a look at the data with the [head](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html) method.

# In[ ]:


data.head()


# We've got a lot of vaiables here, but the layout is pretty straight-forward.  
# As a reminder, the Label variable will be a **1** if the DJIA **stayed the same or rose** on that date or 
#  **0** if the DJIA **fell** on that date.

# And finally, before we get started on the rest of the notebook, we need to split our data into a training set and a testing set. Per Aaron's instructions, we'll use all of the dates up to the end of 2014 as our training data and everything after as testing data.

# In[ ]:


train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']


# ----------

# # Text Preprocessing

# Now that our data is loaded in, we need to clean it up just a little bit to prepare it for the rest of our analysis.  
# To illustrate this process, look at how the example headline below changes from cell to cell.  
# Don't worry about the code too much here, since this example is only meant to be visual.

# In[ ]:


example = train.iloc[3,10]
print(example)


# In[ ]:


example2 = example.lower()
print(example2)


# In[ ]:


example3 = CountVectorizer().build_tokenizer()(example2)
print(example3)


# In[ ]:


pd.DataFrame([[x,example3.count(x)] for x in set(example3)], columns = ['Word', 'Count'])


# Were you able to see everything that changed?  
# The process involved:  
# - Converting the headline to lowercase letters  
# - Splitting the sentence into a list of words  
# - Removing punctuation and meaningless words  
# - Transforming that list into a table of counts

# What started as a relatively "messy" sentence has now become an neatly organized table!  
# And while this may not be exactly what goes on behind the scenes with scikit-learn, this example should give you a pretty good idea about how it works.

# So now that you've seen what the text processing looks like, let's get started on the fun part, modeling!

# ----------

# # Basic Model Training and Testing

# As mentioned previously, scikit-learn is going to take care of all of our preprocessing needs.  
# The tool we'll be using is CountVectorizer, which takes a single list of strings as input, and produces word counts for each one.

# You might be wondering if our dataframe meets this "single list of strings" criteria, and the answer to that is... it doesn't!  
# In order to meet this criteria, we'll use the following [for loop](https://wiki.python.org/moin/ForLoop) to iterate through each row of our dataset, [combine](https://docs.python.org/3.5/library/stdtypes.html#str.join) all of our headlines into a single string, then [add](https://docs.python.org/3.5/tutorial/datastructures.html) that string to the list we need for CountVectorizer.

# In[ ]:


trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))


# With our headlines formatted, we can set up our CountVectorizer.  
# To start, let's just use the default settings and see how it goes!  
# Below, we'll name our default vectorizer, then [use](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.fit_transform) it on our list of combined headlines.  
# After that, we'll take a look at the size of the result to see how many words we have.

# In[ ]:


basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
print(basictrain.shape)


# Wow! Our resulting table contains counts for 31,675 different words!

# Now, let's train a logistic regression model using this data.  
# In the cell below, we're simply naming our model, then [fitting](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit) the model based on our X and Y values.

# In[ ]:


basicmodel = LogisticRegression()
basicmodel = basicmodel.fit(basictrain, train["Label"])


# Our model is ready to go, so let's set up our test data.  
# Here, we're just going to repeat the steps we used to prep our training data, then [predict](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict) whether the DJIA increased or decreased for each day in the test dataset.

# In[ ]:


testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
basictest = basicvectorizer.transform(testheadlines)
predictions = basicmodel.predict(basictest)


# The predictions are set, so let's use a [crosstab](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html) to take a look at the results!

# In[ ]:


pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])


# Prediction accuracy is just over 42%. It seems like this model isn't too reliable.  
# Now, let's also take a look at the coefficients of our model. (Excellent request from [Lucie](https://www.kaggle.com/luciegattepaille)!)

# The cell below will get a list of the names from our CountVectorizer and a list of the coefficients from our model, then combine the two lists into a Pandas dataframe.  
# Once that's made, we can sort it and check out the top 10 positive and negative coefficients.

# In[ ]:


basicwords = basicvectorizer.get_feature_names()
basiccoeffs = basicmodel.coef_.tolist()[0]
coeffdf = pd.DataFrame({'Word' : basicwords, 
                        'Coefficient' : basiccoeffs})
coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
coeffdf.head(10)


# In[ ]:


coeffdf.tail(10)


# Our most positive words don't seem particularly interesting, however there are some negative sounding words within our bottom 10, such as "sanctions," "low," and "hacking."  
# Maybe the saying "no news is good news" is true here?

# ----------

# # Advanced Modeling

# The technique we just used is known as a **bag-of-words** model. We essentially placed all of our headlines into a "bag" and counted the words as we pulled them out.  
# However, most people would agree that a single word doesn't always have enough meaning by itself.  
# Obviously, we need to consider the rest of the words in the sentence as well!  

# This is where the **n-gram** model comes in.  
# In this model, n represents the length of a sequence of words to be counted.  
# This means our bag-of-words model was the same as an n-gram model where n = 1.  
# So now, let's see what happens when we run an n-gram model where n = 2.

# Below, we'll create a new CountVectorizer with the n-gram parameter set to 2 instead of the default value of 1.

# In[ ]:


advancedvectorizer = CountVectorizer(ngram_range=(2,2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)


# Now that we've run our vectorizer, let's see what our data looks like this time around.

# In[ ]:


print(advancedtrain.shape)


# This time we have 366,721 unique variables representing two-word combinations!  
# And here I thought last time was big...

# So, just like last time, let's name and fit our regression model.

# In[ ]:


advancedmodel = LogisticRegression()
advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])


# And again like last time, let's transform our test data and make some predictions!

# In[ ]:


testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advancedtest = advancedvectorizer.transform(testheadlines)
advpredictions = advancedmodel.predict(advancedtest)


# Crosstab says...!

# In[ ]:


pd.crosstab(test["Label"], advpredictions, rownames=["Actual"], colnames=["Predicted"])


# This time we're up to nearly 57% prediction accuracy.  
# We might only consider this a slight improvement, but keep in mind that we've barely scratched the surface of NLP here, and we haven't even touched more advanced machine learning techniques.  
# Let's check out our coefficients again as well!

# In[ ]:


advwords = advancedvectorizer.get_feature_names()
advcoeffs = advancedmodel.coef_.tolist()[0]
advcoeffdf = pd.DataFrame({'Words' : advwords, 
                        'Coefficient' : advcoeffs})
advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
advcoeffdf.head(10)


# In[ ]:


advcoeffdf.tail(10)


# It seems that the results this time were fairly similar. Most of the positive bigrams are unremarkable, while a few of the negative ones like "bin laden" and "threatens to" could be considered to carry some negative meaning.

# ----------

# # What's next?

# If you'd like to keep going forward with this notebook, here are a couple of project ideas:  
# - Experiment with different n values using the n-gram model  
# - Use previous days' headlines to truly "predict" whether the DJIA will rise or fall  
# - Try a machine learning algorithm instead of the basic logistic regression used in this notebook

# Thanks again for reading! I hope you found this helpful!
