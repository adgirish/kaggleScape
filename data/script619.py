
# coding: utf-8

# In this tutorial we're going to get started with some basic natural language processing (NLP) tasks. We're going to:
# 
# * Read in some helpful NLP libraries & our dataset
# * Find out how often each author uses each word
# * Use that to guess which author wrote a sentence
# 
# Ready? Let's get started! :D

# ## General intuition 
# 
# For this tutorial, we're going to be guessing which author wrote a string of text based on the normalized unigram frequency. That's just a fancy way of saying that we're going to count how often each author uses every word in our training data and then divide by the number of total words they wrote. Then, if our test sentence has words that we've seen one author use a lot more than the others, we will guess that that person is probably the author.
# 
# Let's imagine that this is our training corpus:
# 
# * Author one: "A very spooky thing happened. The thing was so spooky I screamed."
# * Author two: "I ate a tasty candy apple. It was delicious"
# 
# And that this is our test sentence that we want to figure out who wrote:
# 
# * Author ???: "What a spooky thing!"
# 
# Just looking at it, it seems more likely that author one wrote this sentence. Author ones says both "spooky" and "thing" a lot, while author two does not (at least, based on our training data). Since we see both "spooky" and "thing" in our test sentence, it seems more likely that it was written by author one than author two--even though the test sentence does have the word "a" in it, which we have seen author two use too.
# 
# In the rest of this tutorial we're going to figure out how to translate this intution into code.

# ## Read in some helpful NLP libraries & our dataset
# 
# For this tutorial, I'm going to be using the Natural Language Toolkit, also called the "NLTK". It's an open-source Python library for analyzing language data. The really nice thing about the NLTK is that it has a really helpful book that goes step-by-step through a lot of the common NLP tasks. Even better: you can get the book for free [here](http://www.nltk.org/book/).

# In[ ]:


# read in some helpful libraries
import nltk # the natural langauage toolkit, open-source NLP
import pandas as pd # dataframes

### Read in the data

# read our data into a dataframe
texts = pd.read_csv("../input/train.csv")

# look at the first few rows
texts.head()


# ## Find out how often each author uses each word
# 
# A lot of NLP applications rely on counting how often certain words are used. (The fancy term for this is "word frequency".) Let's look at the word frequency for each of the authors in our dataset. The NLTK has lots of nice built-in functions and data structures for this that we can make use of.

# In[ ]:


### Split data

# split the data by author
byAuthor = texts.groupby("author")

### Tokenize (split into individual words) our text

# word frequency by author
wordFreqByAuthor = nltk.probability.ConditionalFreqDist()

# for each author...
for name, group in byAuthor:
    # get all of the sentences they wrote and collapse them into a
    # single long string
    sentences = group['text'].str.cat(sep = ' ')
    
    # convert everything to lower case (so "The" and "the" get counted as 
    # the same word rather than two different words)
    sentences = sentences.lower()
    
    # split the text into individual tokens    
    tokens = nltk.tokenize.word_tokenize(sentences)
    
    # calculate the frequency of each token
    frequency = nltk.FreqDist(tokens)

    # add the frequencies for each author to our dictionary
    wordFreqByAuthor[name] = (frequency)
    
# now we have an dictionary where each entry is the frequency distrobution
# of words for a specific author.     


# Now we can look at how often each writer uses specific words. Since this is a Halloween competition, how about "blood", "scream" and "fear"? 👻😨🧛‍♀️

# In[ ]:


# see how often each author says "blood"
for i in wordFreqByAuthor.keys():
    print("blood: " + i)
    print(wordFreqByAuthor[i].freq('blood'))

# print a blank line
print()

# see how often each author says "scream"
for i in wordFreqByAuthor.keys():
    print("scream: " + i)
    print(wordFreqByAuthor[i].freq('scream'))
    
# print a blank line
print()

# see how often each author says "fear"
for i in wordFreqByAuthor.keys():
    print("fear: " + i)
    print(wordFreqByAuthor[i].freq('fear'))


# ## Use word frequency to guess which author wrote a sentence
# 
# The general idea is is that different people tend to use different words more or less often. (I had a beloved college professor that was especially fond of "gestalt".) If you're not sure who said something but it has a lot of words one person uses a lot in it, then you might guess that they were the one who wrote it. 
# 
# Let's use this general principle to guess who might have been more likely to write the sentence "It was a dark and stormy night."

# In[ ]:


# One way to guess authorship is to use the joint probabilty that each 
# author used each word in a given sentence.

# first, let's start with a test sentence
testSentence = "It was a dark and stormy night."

# and then lowercase & tokenize our test sentence
preProcessedTestSentence = nltk.tokenize.word_tokenize(testSentence.lower())

# create an empy dataframe to put our output in
testProbailities = pd.DataFrame(columns = ['author','word','probability'])

# For each author...
for i in wordFreqByAuthor.keys():
    # for each word in our test sentence...
    for j  in preProcessedTestSentence:
        # find out how frequently the author used that word
        wordFreq = wordFreqByAuthor[i].freq(j)
        # and add a very small amount to every prob. so none of them are 0
        smoothedWordFreq = wordFreq + 0.000001
        # add the author, word and smoothed freq. to our dataframe
        output = pd.DataFrame([[i, j, smoothedWordFreq]], columns = ['author','word','probability'])
        testProbailities = testProbailities.append(output, ignore_index = True)

# empty dataframe for the probability that each author wrote the sentence
testProbailitiesByAuthor = pd.DataFrame(columns = ['author','jointProbability'])

# now let's group the dataframe with our frequency by author
for i in wordFreqByAuthor.keys():
    # get the joint probability that each author wrote each word
    oneAuthor = testProbailities.query('author == "' + i + '"')
    jointProbability = oneAuthor.product(numeric_only = True)[0]
    
    # and add that to our dataframe
    output = pd.DataFrame([[i, jointProbability]], columns = ['author','jointProbability'])
    testProbailitiesByAuthor = testProbailitiesByAuthor.append(output, ignore_index = True)

# and our winner is...
testProbailitiesByAuthor.loc[testProbailitiesByAuthor['jointProbability'].idxmax(),'author']


# So based on what we've seen in our training data, it looks like of our three authors, H.P. Lovecraft was the most likely to write the sentence "It was a dark and stormy night".

# ## Ready for more?
# 
# Now that you've got your feet wet, why not head over to [Sohier's intermediate tutorial](https://www.kaggle.com/sohier/intermediate-tutorial-python/), which includes lots of tips on optimizing your code and getting ready to submit to the competition. 
