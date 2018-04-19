
# coding: utf-8

# Thanks Aaron7sun for base code with. explanation. I tried ngram model and Topic modeling additionally. 
# 
# 

# In[ ]:


import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd


# In[ ]:


data = pd.read_csv("../input/Combined_News_DJIA.csv")


# In[ ]:


train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']


# In[ ]:


len(test)


# In[ ]:


len(train)


# In[ ]:


trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))


# In[ ]:


trainvect = CountVectorizer()


# In[ ]:


Trainfeature = trainvect.fit_transform(trainheadlines)


# In[ ]:


####Detailed view of Document Count Matrix
DTM_With_Colm = pd.DataFrame(Trainfeature.toarray(),columns= trainvect.get_feature_names())


# In[ ]:


Trainfeature.shape


# Model Logistic Regression

# In[ ]:


Logis = LogisticRegression()


# In[ ]:


Model1 = Logis.fit(Trainfeature,train['Label'])


# In[ ]:


Model1


# In[ ]:


testheadlines =[]
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))


# In[ ]:


len(testheadlines)


# In[ ]:


Testfeature = trainvect.transform(testheadlines)


# In[ ]:


Testfeature.shape


# In[ ]:


Predicted = Model1.predict(Testfeature)


# In[ ]:


Predicted.shape


# In[ ]:


pd.crosstab(test["Label"], Predicted , rownames=["Actual"] , colnames= ["Predict"])


# Model Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

Nb = MultinomialNB()


# In[ ]:


Model2 = Nb.fit(Trainfeature,train['Label'])


# In[ ]:


Nbpredicted = Model2.predict(Testfeature)


# In[ ]:


Nbpredicted.shape


# In[ ]:


pd.crosstab(test["Label"], Nbpredicted, rownames= ["Acutal"],colnames=["Predict"])


# In[ ]:



import numpy as np
from sklearn.metrics import accuracy_score
y_NaviBayes = Nbpredicted
y_true = test["Label"]
accuracy_score(y_NaviBayes,y_true)
x_Logist = Predicted
x_true = test["Label"]
accuracy_score(x_Logist,x_true)


# **Ngram Model**

# In[ ]:


advvect = CountVectorizer(ngram_range=(1,2))
get_ipython().run_line_magic('time', '')
advancedtrain = advvect.fit_transform(trainheadlines)
advancedtrain.shape


# Model Naive Bayes - Ngram

# In[ ]:


advmodel = MultinomialNB()
advancemodel = advmodel.fit(advancedtrain,train["Label"])
advancetest = advvect.transform(testheadlines)
advNBprediction = advmodel.predict(advancetest) 
advNBprediction.shape




# In[ ]:


pd.crosstab(test["Label"],advNBprediction, rownames=["Acutal"],colnames=["Predicted"])


# In[ ]:


x_adNB = advNBprediction
x_test = test["Label"]


# In[ ]:


accuracy_score(x_test,x_adNB)


# In[ ]:


from sklearn import metrics
metrics.accuracy_score(x_test,x_adNB)


# **Latent Dirichlet Allocation**
# Topic Modeling 
# 

# In[ ]:


from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim


# In[ ]:


get_ipython().run_line_magic('time', '')
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
#Our Document
trainheadlines

# list for tokenized documents in loop
texts = []

# loop through document list
for i in trainheadlines:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]


# In[ ]:


get_ipython().run_line_magic('time', '')
#generate LDA
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=1,chunksize=10000,update_every=1)


# In[ ]:


import pyLDAvis.gensim%time
print(ldamodel.print_topics(num_topics=10, num_words=3))


# In[ ]:


ldamodel.print_topics(5)


# In[ ]:


import pyLDAvis.gensim
pyLDAvis.enable_notebook()
news = pyLDAvis.gensim.prepare(ldamodel,corpus, dictionary)



# In[ ]:


news

