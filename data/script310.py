
# coding: utf-8

# # About the dataset
# This dataset is about the show about "nothing". Yeah, you guessed it right. I am talking about Seinfeld, one of the greatest sitcoms of all time. This dataset provides episodic analysis of the series including the entire script and significant amount of information about each episode.

# # What have I done here?
# I have tried to create a classifier to predict the name of the speaker of a given dialogue. I have used the scripts database for this purpose. So, if a line is given to the predictor, it returns the person who said or might say that line.

# In[ ]:


# Importing libraries and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# importing data
df = pd.read_csv("../input/scripts.csv")
del df["Unnamed: 0"]

df.head()


# Now, we don't really require the EpisodeNo, SEID and Season columns so we remove them.

# In[ ]:


dial_df = df.drop(["EpisodeNo","SEID","Season"],axis=1)
dial_df.head()


# Time for some EDA

# # Plot by number of dialogues spoken

# In[ ]:


dial_df["Character"].value_counts().head(12).plot(kind="bar")


# For creating a corpus out of the data, we will create a datframe concatenating all dialogues of a character. We are choosing 12 characters(by number of dialogues) 

# In[ ]:


def corpus_creator(name):
    st = "" 
    for i in dial_df["Dialogue"][dial_df["Character"]==name]:
        st = st + i
    return st

corpus_df = pd.DataFrame()
corpus_df["Character"] = list(dial_df["Character"].value_counts().head(12).index)

li = []
for i in corpus_df["Character"]:
    li.append(corpus_creator(i))

corpus_df["Dialogues"] = li

corpus_df


# # Preparing stopwords(words that are obsolete for NLP or words that hinder modelling). Very helpful in human speech but useless when trained for modelling

# In[ ]:


from sklearn.feature_extraction import text
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)


# Now, we create a text_processor function to tokenize concatenated dialogues and removing stop words

# In[ ]:


from nltk.tokenize import word_tokenize
def text_processor(dialogue):
    dialogue = word_tokenize(dialogue)
    nopunc=[word.lower() for word in dialogue if word not in stop_words]
    nopunc=' '.join(nopunc)
    return [word for word in nopunc.split()]


# Now, we apply this method

# In[ ]:


corpus_df["Dialogues"] = corpus_df["Dialogues"].apply(lambda x: text_processor(x))
corpus_df


# Adding a length column to the new dataframe which contains length of the concatenated dialogues

# In[ ]:


corpus_df["Length"] = corpus_df["Dialogues"].apply(lambda x: len(x))
corpus_df


# # Who has spoken the most in Seinfeld?

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(ax=ax,y="Length",x="Character",data=corpus_df)


# Obviously, Jerry speaks the most followed by George, Elaine and Kramer

# Now, we do some **correlation analysis**  to find out how similar are the dialogues of different characters to each other.
# (For this, we will use a library called **gensim**. Next cell is the most important yet.)

# In[ ]:


import gensim
# Creating a dictionary for mapping every word to a number
dictionary = gensim.corpora.Dictionary(corpus_df["Dialogues"])
print(dictionary[567])
print(dictionary.token2id['cereal'])
print("Number of words in dictionary: ",len(dictionary))

# Now, we create a corpus which is a list of bags of words. A bag-of-words representation for a document just lists the number of times each word occurs in the document.
corpus = [dictionary.doc2bow(bw) for bw in corpus_df["Dialogues"]]

# Now, we use tf-idf model on our corpus
tf_idf = gensim.models.TfidfModel(corpus)

# Creating a Similarity objectr
sims = gensim.similarities.Similarity('',tf_idf[corpus],num_features=len(dictionary))

# Creating a dataframe out of similarities
sim_list = []
for i in range(12):
    query = dictionary.doc2bow(corpus_df["Dialogues"][i])
    query_tf_idf = tf_idf[query]
    sim_list.append(sims[query_tf_idf])
    
corr_df = pd.DataFrame()
j=0
for i in corpus_df["Character"]:
    corr_df[i] = sim_list[j]
    j = j + 1   


# # Heatmap to detect similarity between characters' dialogues

# In[ ]:


fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corr_df,ax=ax,annot=True)
ax.set_yticklabels(corpus_df.Character)
plt.savefig('similarity.png')
plt.show()


# This plot can actually depict how different the cahracters are from each other. Like **Jerry, George, Elaine and Kramer** speak highly similar lines. Maybe, that's why they are friends(This might have happened because they are usually talking to each other and also because their dialogues are more than others). **[Setting] has lowest correlation scores** well because it is the odd one out because it is not a person. **Jerry and George have highly similar dialogues with 81%  correlation.**

# # An awesome way to use classification
# I am predicting the dialogues said by Elaine, George and Kramer only, so we will choose only their dialogues(I wanted to include Jerry, I mean what is Seinfeld without Jerry Seinfeld but I will later tell you why I didn't do that. Look for clues in markdown)

# In[ ]:


dial_df = dial_df[(dial_df["Character"]=="ELAINE") | (dial_df["Character"]=="GEORGE") | (dial_df["Character"]=="KRAMER")]
dial_df.head(8)


# Way too many dialogues by george will certainly affect the classifier.

# # A text processor for processing dialogues

# In[ ]:


def text_process(dialogue):
    nopunc=[word.lower() for word in dialogue if word not in stop_words]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split()]


# # Preparation for classifier

# In[ ]:


X = dial_df["Dialogue"]
y = dial_df["Character"]


# # TF-IDF Vectorizer
# Convert a collection of raw documents to a matrix of TF-IDF features. Equivalent to CountVectorizer followed by TfidfTransformer.
# 
# In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The tf-idf value increases proportionally to the number of times a word appears in the document and is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general. Nowadays, tf-idf is one of the most popular term-weighting schemes; 83% of text-based recommender systems in the domain of digital libraries use tf-idf.
# 

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer=text_process).fit(X)


# In[ ]:


print(len(vectorizer.vocabulary_))
X = vectorizer.transform(X)


# In[ ]:


# Splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# # Creating a voting classifier with Multinomial Naive Bayes, logistic regression and random forest classifier
# The EnsembleVoteClassifier is a meta-classifier for combining similar or conceptually different machine learning classifiers for classification via majority or plurality voting.
# 
# The EnsembleVoteClassifier implements "hard" and "soft" voting. In hard voting, we predict the final class label as the class label that has been predicted most frequently by the classification models. In soft voting, we predict the class labels by averaging the class-probabilities (only recommended if the classifiers are well-calibrated).

# In[ ]:


from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import VotingClassifier as VC
mnb = MNB(alpha=10)
lr = LR(random_state=101)
rfc = RFC(n_estimators=80, criterion="entropy", random_state=42, n_jobs=-1)
clf = VC(estimators=[('mnb', mnb), ('lr', lr), ('rfc', rfc)], voting='hard')


# In[ ]:


# Fitting and predicting
clf.fit(X_train,y_train)

predict = clf.predict(X_test)


# In[ ]:


# Classification report
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predict))
print('\n')
print(classification_report(y_test, predict))


# So we get about 51% precision which is not bad considering the limited vocablury. George is dominant here due to high recall value of 0.80. So, unless a dialogue has words that don't exist at all in George's vocablury, there is a high chance George will the speaker of most lines. The situation was worse when Jerry was inclued. **This is why I decided to drop Jerry's dialogues from the dataset.** 

# # The Predictor

# In[ ]:


def predictor(s):
    s = vectorizer.transform(s)
    pre = clf.predict(s)
    print(pre)


# # Now, we predict...

# In[ ]:


# Answer should be Kramer
predictor(['I\'m on the Mexican, whoa oh oh, radio.'])


# In[ ]:


# Answer should be Elaine
predictor(['Do you have any idea how much time I waste in this apartment?'])


# In[ ]:


# Answer should be George 
predictor(['Yeah. I figured since I was lying about my income for a couple of years, I could afford a fake house in the Hamptons.'])


# In[ ]:


# Now, a random sentence
predictor(['Jerry, I\'m gonna go join the circus.'])


# In[ ]:


# A random sentence
predictor(['I wish we can find some way to move past this.'])


# In[ ]:


# Answer should be Kramer
predictor(['You’re becoming one of the glitterati.'])


# In[ ]:


# Answer should be Elaine
predictor(['Jerry, we have to have sex to save the friendship.'])


# See, this "george effect" can lead to awkward results. That's why I am trying to find a way to get rid of that high recall value and also improve the precision of classifier. I am open to suggestions.

# This is not the end. More is yet to come.
# Coming soon:
# 1. Alternative deep learning approach using Keras
# 2. Sentimental Analysis
