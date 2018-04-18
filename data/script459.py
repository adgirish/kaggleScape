
# coding: utf-8

# <h1><center> How to use content to come up with relevant Recommendations</center></h1>
# <h2><center>How to use text mining to decide which ted talk to watch</center></h2>
# 
# Well, I love ted talks, who doesn't? When I first looked at this dataset, a couple of things struck, me, first, the fact that, since this dataset contains transcripts of many ted talks, by default we have a corpus which is very rich and linguistically very well structured. Second, since this corpus has nice linguistic properties, this is probably as good a dataset as Reuters 20 News Group or any version of Guttenberg corpus. This got me thinking:
# <img src="https://github.com/Gunnvant/ted_talks/blob/master/thinking.gif?raw=true">
# 
# ***I have the data on all the transcripts across many ted talks, can I try to come up with a way to recommend ted talks based on their similarity, just as is done by official Ted page?***
# 
# Ofcourse, the recommendation engines used by the official ted page, will be a degrees of magnitude more sophisticated than what I can demonstrate here and would also involve use of some sort of historical user-item interaction data.
# 
# The idea is to demonstrate here how one can generate recommendations just using content. This becomes essentially important when you don't have any user-item interaction data, essentially when you are starting out new and still want to provide the consumers of your content relevant contextual recommendations.
# 
# <h2><center>Meet the data</center></h2>
# The data we will use comes in as a tabular flat file, the transcript for each talk is stored in a row across a column named transcript. Here is how the file looks like

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
transcripts=pd.read_csv("../input/transcripts.csv")
transcripts.head()


# After examining how the data looks like, its quite easy to extract the titles from the url. My eventual goal is to use the text in the transcript column to create a measure of similarity. And then recommend the 4 most similar titles to a given talk. It is quite straightforward to separate the title from url using a simple string split operation as shown below

# In[ ]:


transcripts['title']=transcripts['url'].map(lambda x:x.split("/")[-1])
transcripts.head()


# At this point I am ready to begin piecing together the components that will help me build a talk recommender. In order to achieve this I will have to:
# 
# 1. Create a vector representation of each transcript
# 2. Create a similarity matrix for the vector representation created above
# 3. For each talk, based on some similarity metric, select 4 most similar talks

# <h2><center>Creating word vectors using Tf-Idf</center></h2>
# Since my final goal is to recommend talks based on the similarity of their content, the first thing I will have to do is to, create a representation of the transcripts that are amenable to comparison. One way of doing this is to create a tfidf vector for each transcript. But what is this tfidf business anyway? Let's discuss that first.
# 
# <h3>Corpus, Document and Count Matrix</h3>
# To represent text, we will think of each transcript as one "Document" and the set of all documents as a "Corpus". Then we will create a vector representing the count of words that occur in each document, something like this:
# 
# <img src="https://github.com/Gunnvant/ted_talks/blob/master/count_matrix.png?raw=true">
# 
# As you can see for each document, we have created a vector for count of times each word occurs. So the vector  (1,1,1,1,0,0) , represents the count of words "This","is","sentence","one","two","three" in document 1.This is known as a count matrix. There is one issue with such a representation of text though, it doesn't take into account the importance of words in a document. For example, the word "one" occurs only once in the document 1 but is missing in the other documents, so from the point of view of its importance, "one" is an important word for document 1 as it characterises it, but if we look at the count vector for document 1, we can see that "one" gets of weight of 1 so do words like "This", "is" etc. Issues, regarding the importance of words in a document can be handled using what is known as Tf-Idf.
# 
# <h3>Term Frequency-Inverse Document Frequency (Tf-Idf)</h3>
# 
# In order to understand how Tf-Idf helps in identifying the importance of the words, let's do a thought experiment and ask our-selves a couple of questions, what determines if a word is important?
# 
# 1.  If the word occurs a lot in document?
# 2.  If the word occurs rarely in the corpus?
# 3.  Both 1 and 2?
# 
# A word is important in a document if, it occurs a lot in the document, but rarely in other documents in the corpus. Term Frequency measures how often the word appears in a given document, while Inverse document frequency measures how rare the word is in a corpus. The product of these two quantities, measures the importance of the word and is known as Tf-Idf. Creating a tf-idf representation is fairly straightforward, if you are working with a machine learning frame-work, such as scikit-learn, it's fairly straighforward to create a matrix representation of text data

# In[ ]:


Text=transcripts['transcript'].tolist()
tfidf=text.TfidfVectorizer(input=Text,stop_words="english")
matrix=tfidf.fit_transform(Text)
print(matrix.shape)


# So once we sort the issue of representing word vectors by taking into account the importance of the words, we are all set to tackle the next issue, how to find out which documents (in our case Ted talk transcripts) are similar to a given document?
# <h2><center>Finding similar documents</center></h2>
# To find out similar documents among different documents, we will need to compute a measure of similarity. Usually when dealing with Tf-Idf vectors, we use  $cosine$  similarity. Think of  $cosine$  similarity as measuring how close one TF-Idf vector is from the other. Now if you remember from the previous discussion, we were able to represent each transcript as a vector, so the $cosine$  similarity will become a means for us to find out how similar the transcript of one Ted Talk is to the other.
# So essentially, I created a $cosine$ matrix from Tf-Idf vectors to represent how similar each document was to the other, schematically, something like:
# 
# <img src="https://github.com/Gunnvant/ted_talks/blob/master/cosine.png?raw=true">
# 
# Again, using sklearn, doing this was very straighforward

# In[ ]:


### Get Similarity Scores using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_unigram=cosine_similarity(matrix)


# All I had to do now was for, each Transcript, find out the 4 most similar ones, based on cosine similarity. Algorithmically, this would amount to finding out, for each row in the cosine matrix constructed above, the index of five columns, that are most similar to the document (transcript in our case) corresponding to the respective row number. This was accomplished using a few lines of code

# In[ ]:


def get_similar_articles(x):
    return ",".join(transcripts['title'].loc[x.argsort()[-5:-1]])
transcripts['similar_articles_unigram']=[get_similar_articles(x) for x in sim_unigram]


# Let's check how we faired, by examining the recommendations. Let's pickup, any Ted Talk Title from, the list, let's say we pick up:

# In[ ]:


transcripts['title'].str.replace("_"," ").str.upper().str.strip()[1]


# Then, based on our analysis, the four most similar titles are

# In[ ]:


transcripts['similar_articles_unigram'].str.replace("_"," ").str.upper().str.strip().str.split("\n")[1]


# You can clearly, see that by using Tf-Idf vectors to compare transcripts of the talks, we were able to pick up, talks that were on similar themes. You can try a couple of more things:
# 
# 1. I have created a tf-idf with unigrams, you can try using bigrams and see if you get better results.
# 2. Try using pre-trained word vectors such as word2vec to create vector representation of just the Titles and try to find similarity using cosine distance
# 3. Take this analysis and create a sqlite db and write a flask web app
#  
#  Hope you find this useful. If yes, please upvote
#  
#  PS: I also created a medium post on this here https://towardsdatascience.com/how-i-used-text-mining-to-decide-which-ted-talk-to-watch-dfe32e82bffd
# 
