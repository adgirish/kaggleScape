
# coding: utf-8

# *Hello everyone, this is my first Kernel and I'm very motivated to work on this project! ***
# 
# # **Introduction** : sngrams
# 
# In this tutorial, we will try to use the**  [#SNGRAMS ](https://pdfs.semanticscholar.org/4e5a/778e0e45a4bddb81916a33d8e3b380fbb836.pdf) **to find which author wrote which sentences ! Grigori Sidorov, Francisco Velasquez, Efstathios Stamatatos, Alexander Gelbukh and Liliana Chanona did the same research with the same number of authors : THREE ! Strange, isn't it ? 
# 
# # ** 1. What are sngrams ? **
# ##  1.1. Stanford Parser
# 
# Syntactic N-grams are using the output of the ** [ #Stanford Parser](http://nlp.stanford.edu:8080/parser/) ** which is a probabilistic parser that use knowledge of language. The parser exists in English and extrats groups of words that have a grammatical relation. The model is based on the structure of the sentences which depends of the language. 
# 
# The output of this parser for a sentence like ' It never once occurred to me that the fumbling might be a mere mistake.' (which is the second sentence of our train set) would look like : 
# 
# nsubj(occurred-4, It-1)<br>
# neg(occurred-4, never-2)<br>
# advmod(occurred-4, once-3)<br>
# root(ROOT-0, occurred-4)<br>
# case(me-6, to-5)<br>
# nmod(occurred-4, me-6)<br>
# mark(mistake-14, that-7)<br>
# det(fumbling-9, the-8)<br>
# nsubj(mistake-14, fumbling-9)<br>
# aux(mistake-14, might-10)<br>
# cop(mistake-14, be-11)<br>
# det(mistake-14, a-12)<br>
# amod(mistake-14, mere-13)<br>
# ccomp(occurred-4, mistake-14)<br>
# 
# The following tree is also defined :
# ![](https://raw.githubusercontent.com/BoltMaud/Kaggle_images/master/graphstanford.bmp)
# 
# *This graphe was designed by the librairy : nltk.draw.tree. * 
# *The name 'nsubj', 'neg' .. is the relation between the words in the parenthesis. The number indicates the position of the words in the sentence.   
# 
# ## 1.2. n-grams using the stanford parser
# 
# ### 1.2.1 Normal n-grams
# Using the sentence 'It never once occurred to me that the fumbling might be a mere mistake' the normal 2-grams is : 
# 
# It never ;  never once ;  once occurred ; occured to ;  to me ; me that ;  that the ;  the fumbling ; fumbling might ; might be ; be a ; a mere ;  mere mistake
# 
# For a 3-grams : 
# 
# It never once ;never once occurred ; once occurred to ; occurred to me ;to me that ; me that the ;that the fumbling ;the fumbling  might ; fumbling  might be ; might be a ; be a mere ; a mere mistake
#  
#  ### 1.2.2 Sn-grams 
#  The sn-grams use the result of the stanford parser and gives :
#  
#  For a s2grams, the couples are all the couple from the roots :
#  occured once, occured never, occured it , occured to, occured mistake, mistake mere,mistake a, mistake be, mistake might, mistake fumbling, mistake that, fumbing the, me to 
#  
#  For a s3grams the (occured to) and (to me) become (occured to me) and (mistake fumbling) and (fumbling the) become (mistake fumbling the) 
#  
#  # ** 2. How to use sngrams **
#  
#  I tried this solution on my computer but to generate all the features, the programme needed 5 hours. I decided to try without the syntaxic ngrams.
#  
#  # ** 3. Programme with ngrams ** 
#  
#  

# First, we import the dataset and the libs : 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

dfTrain = pd.read_csv("../input/train.csv") # importing train dataset
dfTest = pd.read_csv("../input/test.csv") # importing test dataset


# Then we create a matrix with the ngrams using TiDf. We delete the stop-word and accepte ngrams from 1 word to 3. 
# The function fit_tranform will tranform the data into a matrix and fit the model. 

# In[ ]:


vectorizer = CountVectorizer(stop_words="english",analyzer='word', ngram_range=(1,3))
train_counts = vectorizer.fit_transform(dfTrain.text)


# We fit the model with the Multinomiale method because it's the best one for the problems using TiDf and ngrams.

# In[ ]:


classifier = MultinomialNB()
classifier.fit(train_counts, dfTrain.author)


# The test set need to be transform too and the model is ready to predict : 

# In[ ]:


tests_counts = vectorizer.transform(dfTest.text)
predicted = pd.DataFrame(classifier.predict_proba(tests_counts) )


# Finally, we prepare the submit file :

# In[ ]:


submit=pd.DataFrame({})
submit["id"]=dfTest.id
submit["EAP"]=predicted[0]
submit["HPL"]=predicted[1]
submit["MWS"]=predicted[2]

print(submitfinal)
submit.to_csv("submit.csv", sep=',',index=False)


# # ** 4. Vizualisation ** 
# 
# In this part, I'm not sure to keep the rules but I didn't see anything that forbidd to use everything we know. 
# 
# ## 4.1 LDA Vizualisation 
# 
# Firstly, I used knime to extract the words topics for each authors. I created a vizualisation with the colors of Halloween. The size of a word depend of its weight at the output of LDA algorithm. 
# 
# ![](https://raw.githubusercontent.com/BoltMaud/Kaggle_images/master/viz1.png)
# 
# Then I used [#Tropes](http://www.tropes.fr/) to get some interesting information about the grammar.
# 
# ** THE PRONOUNS **
# ![](https://raw.githubusercontent.com/BoltMaud/Kaggle_images/master/pronouns.png)
# 
# ** THE CONNECTORS **
# ![](https://raw.githubusercontent.com/BoltMaud/Kaggle_images/master/connectors_.png)
# 
# ** THE MODALIZATION ** 
# ![](https://raw.githubusercontent.com/BoltMaud/Kaggle_images/master/modalities_.png)
# 
# 
# 
# 
