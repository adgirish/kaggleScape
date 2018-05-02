
# coding: utf-8

# If I'm ever given a problem to solve, i start with the simplest and stupidest solutions :) 
# Here's my crack at it -- no machine learning, plain old similarity matrices. Still better than word matching!
# 
# Given the question pair, i strip the stopwords, and calculate wordnet similarities between each word pair. Now each term has an array of similarities(with all terms in the other sentence) -- Get the maximum similarity value(for each term), average it across the terms, and BAM, you've got your duplicate score. 
# 
# *Pretty naive ;)*
# 
# Inspired from [this research paper][1]
# 
# 
# ----------
# 
# 
#   [1]: http://staffwww.dcs.shef.ac.uk/people/S.Fernando/pubs/clukPaper.pdf

# Let's have a look at the code

# In[ ]:


import pandas as pd

from nltk.corpus import wordnet as wn

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

df_train = pd.read_csv('../input/train.csv')


# In[ ]:


def get_terms(sentence):
    return [i for i in sentence.lower().split() if i not in stop]


# I'm leaving out the iterator for-loop, for better readability. 

# In[ ]:


row = df_train.iloc[0] #Just taking the first row, you can put a loopover 

res = row["is_duplicate"]
terms1 = get_terms(row["question1"])
terms2 = get_terms(row["question2"])

sims = []


# In[ ]:


for word1 in terms1:
    word1_sim = []

    try:
        syn1 = wn.synsets(word1)[0]
    except:  #if wordnet is not able to find a synset for word1
        sims.append([0 for i in range(0, len(terms2))])
        continue


    for word2 in terms2:
        try:
            syn2 = wn.synsets(word2)[0]
        except: #if wordnet is not able to find a synset for word2
            word1_sim.append(0)
            continue

        word_similarity = syn1.wup_similarity(syn2)
        word1_sim.append(word_similarity)

    sims.append(word1_sim)


# Here, i loop over all word pairs, and write the similarities in a list of lists. Basically sims[i][j] represents the wordnet similarity between "Term i" of question1 and "Term j" of question2. 
# 
# In case wordnet doesn't have the definition, the similarity is considered as 0. (Possibly slang/non-english words)

# Now that we have our similarity matrix, let's calculate the pair match score. 

# In[ ]:


word1_score = 0
for i in range(0, len(terms1), 1):
    try:
        word1_score += max(sims[i])
    except:
        continue
word1_score /= len(terms1) #Averaging over all terms


# We have to do the similar score calculation for words in Question2. Since the matrix is row-wise(from Question1), the maximum score is calculated column-wise.

# In[ ]:


word2_score = 0

for i in range(0, len(terms2), 1):
    try:
        word2_score += max([j[i] for j in sims])
    except:
        continue
word2_score /= len(terms2)


# Taking the average of the two word scores, 

# In[ ]:


pair_score = (word1_score + word2_score)/2


# The code is super-slow, and this was just a trial run. I ran this over around 4000 question pairs, and got an accuracy of 65%. 
# 
# Pretty good for a naive similarity test, eh?

# *Hit the upvote, if you learned somthing out of this :)*
