
# coding: utf-8

# # TED-Talks topic models

# ![](http://greatperformersacademy.com/images/images/Articles_images/10-best-ted-talks-ever.jpg)

# In this notebook we will study text processing using TED transcripts, passing through feature extraction to topic modeling in order to (1) have a first meet with text processing techniques and (2) analyze briefly some TED-Talks patterns.
# 
# In the amazing TED-Talks dataset, we have two files, one (ted_main.csv) with meta information about the talks, as # of comment, rating, related TEDs and so on; the other file has the transcripts which we'll care about in this tutorial. Even so, we'll use the ted_main.csv file to evaluate our topic modeling implementation, because it has a columns of talks' tags, useful as our "ground truth topics".

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
from time import time


# ### 0.1. Transcripts loading

# In[ ]:


ted_main_df = pd.read_csv('../input/ted_main.csv', encoding='utf-8')
transcripts_df = pd.read_csv('../input/transcripts.csv', encoding='utf-8')
transcripts_df.head()


# ## 1. Text feature extraction with TFIDF
# 
# First,  consider the term-frequency (TF) matrix above, that can be extracted from a list of documents and the universe of terms in such documents.
# 
# |        | Document 1 | Document 2 | ... | Document N |
# |--------|------------|------------|-----|------------|
# | Term 1 | 3          | 0          | ... | 1          |
# | Term 2 | 0          | 1          | ... | 2          |
# | Term 3 | 2          | 2          | ... | 1          |
# | ...    | ...        | ...        | ... | ...        |
# | Term N | 1          | 0          | ... | 0          |
# 
# 
# This is a huge matrix with all elements' frequency in all documents. Now consider de idf (inverse document frequency) as an operation to transform this frequency into word importance, calculated by:
# 
# $$ tfidf_{i,j} = tf_{i,j}  \times log(\frac{N}{df_{i}}) $$
# 
# Where $i$ refers to term index and $j$ document index. $N$ is the total number of documents and $df_{i}$ is the number of documents containing $i$.
# 
# 

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english",
                        use_idf=True,
                        ngram_range=(1,1), # considering only 1-grams
                        min_df = 0.05,     # cut words present in less than 5% of documents
                        max_df = 0.3)      # cut words present in more than 30% of documents 
t0 = time()

tfidf = vectorizer.fit_transform(transcripts_df['transcript'])
print("done in %0.3fs." % (time() - t0))


# Keeping that in mind, we'll want to see the 'most important' words in our matrix...

# In[ ]:


# Let's make a function to call the top ranked words in a vectorizer
def rank_words(terms, feature_matrix):
    sums = feature_matrix.sum(axis=0)
    data = []
    for col, term in enumerate(terms):
        data.append( (term, sums[0,col]) )
    ranked = pd.DataFrame(data, columns=['term','rank']).sort_values('rank', ascending=False)
    return ranked

ranked = rank_words(terms=vectorizer.get_feature_names(), feature_matrix=tfidf)
ranked.head()


# In[ ]:


# Let's visualize a word cloud with the frequencies obtained by idf transformation
dic = {ranked.loc[i,'term'].upper(): ranked.loc[i,'rank'] for i in range(0,len(ranked))}

import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white',
                      max_words=100,
                      colormap='Reds').generate_from_frequencies(dic)
fig = plt.figure(1,figsize=(12,10))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()


# ## 2. Topic modeling
# 
# Not recently decomposition techniques have been used to extract topics from text data. A topic is a mixture of words and a document should be pertinent to a topic if these words are present in there. Look at the diagram to understand better.
# 
# ![alt text](https://image.ibb.co/kH9t87/d.png)
# 
# Here we will extract topics from NMF and LDA to check for better results. First, let's try LDA.

# In[ ]:


from sklearn.decomposition import LatentDirichletAllocation

n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics,random_state=0)

topics = lda.fit_transform(tfidf)
top_n_words = 5
t_words, word_strengths = {}, {}
for t_id, t in enumerate(lda.components_):
    t_words[t_id] = [vectorizer.get_feature_names()[i] for i in t.argsort()[:-top_n_words - 1:-1]]
    word_strengths[t_id] = t[t.argsort()[:-top_n_words - 1:-1]]
t_words


# In[ ]:


fig, ax = plt.subplots(figsize=(7,15), ncols=2, nrows=5)
plt.subplots_adjust(
    wspace  =  0.5,
    hspace  =  0.5
)
c=0
for row in range(0,5):
    for col in range(0,2):
        sns.barplot(x=word_strengths[c], y=t_words[c], color="red", ax=ax[row][col])
        c+=1
plt.show()


# Not so bad, but there are topics that has words fairly inappropriate. We could say that these topics are uncohesive for humans. That is common difficult when applying topic modeling to real problems. 
# 
# Another problem is the optimal number of topics. There are several ways to validate it, as with perplexity or log-likelyhood. However, let's keep with 10 topics! :)

# In[ ]:


from sklearn.decomposition import NMF

n_topics = 10
nmf = NMF(n_components=n_topics,random_state=0)

topics = nmf.fit_transform(tfidf)
top_n_words = 5
t_words, word_strengths = {}, {}
for t_id, t in enumerate(nmf.components_):
    t_words[t_id] = [vectorizer.get_feature_names()[i] for i in t.argsort()[:-top_n_words - 1:-1]]
    word_strengths[t_id] = t[t.argsort()[:-top_n_words - 1:-1]]
t_words


# In[ ]:


fig, ax = plt.subplots(figsize=(7,15), ncols=2, nrows=5)
plt.subplots_adjust(
    wspace  =  0.5,
    hspace  =  0.5
)
c=0
for row in range(0,5):
    for col in range(0,2):
        sns.barplot(x=word_strengths[c], y=t_words[c], color="red", ax=ax[row][col])
        c+=1
plt.show()


# Hmm. Now you see that with NMF things get better. So, we'll use it testing for a document and see what topics are extracted.

# In[ ]:


# Formulating a pipeline to insert a document and extract the topics pertinency
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('tfidf', vectorizer),
    ('nmf', nmf)
])

document_id = 4
t = pipe.transform([transcripts_df['transcript'].iloc[document_id]]) 
print('Topic distribution for document #{}: \n'.format(document_id),t)
print('Relevant topics for document #{}: \n'.format(document_id),np.where(t>0.01)[1])
print('\nTranscript:\n',transcripts_df['transcript'].iloc[document_id][:500],'...')

talk = ted_main_df[ted_main_df['url']==transcripts_df['url'].iloc[document_id]]
print('\nTrue tags from ted_main.csv: \n',talk['tags'])


# Seems nice! The transcript #4 really talk about topic #5.
# 
# Now we would do a exploratory analysis about our topics and extract descriptitve statistics and visualizations for the transcripts.

# In[ ]:


t = pipe.transform(transcripts_df['transcript']) 
t = pd.DataFrame(t, columns=['#{}'.format(i) for i in range(0,10)])


# In[ ]:


import seaborn as sns

new_t = pd.DataFrame({'value':t['#0'].values,'topic':['#0']*len(t)})
for tid in t.columns[1:]:
    new_t = pd.concat([new_t, pd.DataFrame({'value':t[tid].values,'topic':[tid]*len(t)})])

fig = plt.figure(1,figsize=(12,6))
sns.violinplot(x="topic", y="value", data=new_t, palette='Reds')
plt.show()


# Through analyzing this plot, I would say that in general the topic distribution behaves evenly, excepts for #0. Things to be concluded:
# 
# 1. Topic #0 (['god', 'book', 'stories', 'oh', 'art']) is very general per si in terms of words meaning and perhaps explain this result. 
# 2. Topic #2 is only about music and it has high incidences because of its specificity, in contrast to #0.
# 3. Excluding #0 (open meaning issues), topics **#4 (about earth), #5 (about government), #7 (about data&information) and #9 (about education)** have higher quartiles, meaning that they are the most frequent topics that TED Talks carry on.
# 
# I believe that it effectively summarizes what TEDs are about: **ideas that really matter and worth spreading**.
# 
# Hope that it could help NLP beginners!
