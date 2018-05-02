
# coding: utf-8

# # Explaining the NLP terms   <br>
# <div class="section" id="the-bag-of-words-representation">
# <h3>1 The Bag of Words representation<a class="headerlink" href="#the-bag-of-words-representation" title="Permalink to this headline">¶</a></h3>
# <p>Text Analysis is a major application field for machine learning
# algorithms. However the raw data, a sequence of symbols cannot be fed
# directly to the algorithms themselves as most of them expect numerical
# feature vectors with a fixed size rather than the raw text documents
# with variable length.</p>
# <p>In order to address this, scikit-learn provides utilities for the most
# common ways to extract numerical features from text content, namely:</p>
# <ul class="simple">
# <li><strong>tokenizing</strong> strings and giving an integer id for each possible token,
# for instance by using white-spaces and punctuation as token separators.</li>
# <li><strong>counting</strong> the occurrences of tokens in each document.</li>
# <li><strong>normalizing</strong> and weighting with diminishing importance tokens that
# occur in the majority of samples / documents.</li>
# </ul>
# <p>In this scheme, features and samples are defined as follows:</p>
# <ul class="simple">
# <li>each <strong>individual token occurrence frequency</strong> (normalized or not)
# is treated as a <strong>feature</strong>.</li>
# <li>the vector of all the token frequencies for a given <strong>document</strong> is
# considered a multivariate <strong>sample</strong>.</li>
# </ul>
# <p>A corpus of documents can thus be represented by a matrix with one row
# per document and one column per token (e.g. word) occurring in the corpus.</p>
# <p>We call <strong>vectorization</strong> the general process of turning a collection
# of text documents into numerical feature vectors. This specific strategy
# (tokenization, counting and normalization) is called the <strong>Bag of Words</strong>
# or “Bag of n-grams” representation. Documents are described by word
# occurrences while completely ignoring the relative position information
# of the words in the document.</p>
# </div>
# <p><a class="reference internal" href="generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer" title="sklearn.feature_extraction.text.CountVectorizer"><code class="xref py py-class docutils literal"><span class="pre">CountVectorizer</span></code></a> implements both tokenization and occurrence
# counting in a single class:</p>

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


vect = CountVectorizer()
vect


# <br><p>Let’s use it to tokenize and count the word occurrences of a minimalistic
# corpus of text documents:</p>

# In[ ]:


corpus = ['Hi my name is kanav.','I love reading.','Kanav loves reading scripts.']
X= vect.fit_transform(corpus)
X # note the dimensions of X(3X9) means 3 rows and 9 columns. 


# <br>Note the dimensions of X (3X9) means 3 rows and 9 columns. <br>
# as there are three documents and 9 unique words<br>
# See

# In[ ]:


vect.get_feature_names()


# ### See this is the frequency matrix in the given documents

# Each term found by the analyzer during the fit is assigned a unique integer index corresponding to a column in the resulting matrix. This interpretation of the columns can be retrieved as follows:

# In[ ]:


X.toarray()


# Hence words that were not seen in the training corpus will be completely ignored in future calls to the transform method:

# In[ ]:


vect.transform(['hi,whats your name?.']).toarray()


# ### Normalization and stemming
# Since the words like love and loves has same meaning so, why not we treat them same?
# 

# In[ ]:


import nltk
porter = nltk.PorterStemmer()
[porter.stem(t) for t in vect.get_feature_names()]


# See the loves has now become love.

# Now we have total 8 unique features

# In[ ]:


list(set([porter.stem(t) for t in vect.get_feature_names()]))


# In[ ]:


WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(t) for t in list(set([porter.stem(t) for t in vect.get_feature_names()]))]


# # Lemmatization
# A very similar operation to stemming is called lemmatizing. The major difference between these is, as you saw earlier, stemming can often create non-existent words, whereas lemmas are actual words.
# 
# So, your root stem, meaning the word you end up with, is not something you can just look up in a dictionary, but you can look up a lemma.
# 
# Some times you will wind up with a very similar word, but sometimes, you will wind up with a completely different word. Let's see some examples.

# In[ ]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))


# ## Please upvote!
# i will be keep on updating !
