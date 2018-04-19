
# coding: utf-8

# # EDA for En Text Normalization
# 
# Here you can find a fast rush through the data in a short-question-answer style to get an impression of the data. It goes not into so much details but perhaps helps to get some ideas where to continue.
#  
# Work in progress ;-) .

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### What data files are given in this competition?

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ### How many training samples do we have and how many attributes are given per sample?

# In[ ]:


train = pd.read_csv("../input/en_train.csv")
train.shape


# ### How do these 5 training attributes look like?

# In[ ]:


train.head()


# * Ok, each sentence and each token has an id. 
# * Each token has a class to show its type.
# * We are asked to transform the tokens of the column "before" to those of the column "after"

# ### How many test samples do we have?

# In[ ]:


test = pd.read_csv("../input/en_test.csv")
test.shape


# In[ ]:


test.head()


# The test set misses the class column. 

# ### How does the submission example look like?

# In[ ]:


sample_submission = pd.read_csv("../input/en_sample_submission.csv")
sample_submission.head()


# The sample submission has an id that contains the sentence and the token id as a sentence_token combination. And we can see, that we are asked to predict the "after"-column. 
# 
# ### Last question: How big is the test size based on number of rows?
# 

# In[ ]:


test.shape[0]/(train.shape[0] + test.shape[0])


# As we are transforming sentences it seems to me that it makes more sense to look at the number of sentences instead:

# In[ ]:


num_train_sentences = len(train.sentence_id.unique())
num_train_sentences


# In[ ]:


num_test_sentences = len(test.sentence_id.unique())
num_test_sentences


# In[ ]:


num_test_sentences / (num_train_sentences + num_test_sentences)


# ## Sentences

# ### How long are the sentences?

# In[ ]:


train_sentences = train.groupby("sentence_id")["sentence_id"].count()
train_sentences.describe()


# Ok, most of the sentences in the train data are 8 to 18 tokens long. Let's have a look at the distribution:

# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(20,10))
sns.set_style("whitegrid")
count_length_fig = sns.countplot(train_sentences, ax=ax)
for item in count_length_fig.get_xticklabels():
    item.set_rotation(90)


# Amazing! There is a peak for sentences that are 7 tokens long. I will go back to them later on! 

# ### And the test set?

# In[ ]:


test_sentences = test.groupby("sentence_id")["sentence_id"].count()
test_sentences.describe()


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(20,10))
sns.set_style("whitegrid")
count_length_fig = sns.countplot(test_sentences, ax=ax)
for item in count_length_fig.get_xticklabels():
    item.set_rotation(90)


# Mean and quantiles are shifted a bit to higher values. This is probably caused by the low test size of roughly 9 %.

# ### How do the smallest and longest sentences look like? 

# #### Longest train sentence:

# In[ ]:


max_id = train_sentences[train_sentences == train_sentences.max()].index.values
max_id


# In[ ]:


long_example = train[train.sentence_id==max_id[0]].before.values.tolist()
long_example= ' '.join(long_example)
long_example


# Looks like a literature reference.

# #### Some smallest train sentences

# In[ ]:


min_id = train_sentences[train_sentences == train_sentences.min()].index.values
min_id


# In[ ]:


for n in range(5):
    small_example = train[train.sentence_id==min_id[n]].before.values.tolist()
    small_example= ' '.join(small_example)
    print(small_example)


# Dates :-)

# #### And the median sentences?

# In[ ]:


median_id = train_sentences[train_sentences == train_sentences.median()].index.values
median_id


# In[ ]:


for n in range(5):
    median_example = train[train.sentence_id==median_id[n]].before.values.tolist()
    median_example= ' '.join(median_example)
    print(median_example)


# ## Tokens

# ### How many unique tokens do we have?

# In[ ]:


len(train.token_id.unique())


# Ok, reading the explanations for the data in detail: Each token within a sentence has a token_id. Consequently the longest sentence has token ids ranging from 0 to 255 (inclusive), and one of the smallest from 0 to 1. 

# ## Token classes

# #### How many token classes do we have? And how many counts per class?

# In[ ]:


len(train["class"].unique())


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(10,12))
#sns.set_style("whitegrid")
count_classes_fig = sns.countplot(y="class", data=train, ax=ax)
for item in count_classes_fig.get_xticklabels():
    item.set_rotation(45)


# In[ ]:


train.groupby("class")["class"].count()


# Ok, most are plain. But there are also some exotic classes.
# 
# ### What is meant by electronic or verbatim?

# #### ELECTRONIC class examples

# In[ ]:


most_electronic_cases = train[train["class"]=='ELECTRONIC'].groupby("before")["before"].count(
).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x=most_electronic_cases.index, y=most_electronic_cases.values)


# #### VERBATIM class examples

# In[ ]:


most_verbatim_cases = train[train["class"]=='VERBATIM'].groupby("before")["before"].count(
).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x=most_verbatim_cases.index, y=most_verbatim_cases.values)


# ## Before words

# ### How many unique before words do we have and how do the most common look like?

# In[ ]:


len(train.before.unique())


# In[ ]:


train_word_counts = train.groupby("before")["before"].count().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x=train_word_counts.index, y=train_word_counts.values)


# In[ ]:


len(test.before.unique())


# In[ ]:


test_word_counts = test.groupby("before")["before"].count().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x=test_word_counts.index, y=test_word_counts.values)


# ## Pattern of changes

# ### How many before words changed after normalization in the train set?

# In[ ]:


train["change"] = 0
train.loc[train.before!=train.after, "change"] = 1
train["change"].value_counts()


# ### Which words changed most often?

# In[ ]:


most_changed_words = train[train.change==1].groupby("before")["before"].count(
).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x=most_changed_words.index, y=most_changed_words.values)


# ### To which class do the changed words belong most often?

# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(15,5))
changes_classes_fig = sns.countplot(x="class", data=train[train.change==1])
for item in changes_classes_fig.get_xticklabels():
    item.set_rotation(45)


# ### Do we have overlapping class contents?
# 
# Actually I find it a bit strange that we have a digit class and a decimal, date, cardinal, ordinal, fraction class etc.. Are the tokens in these classes really all of a different kind or can we find the same token for example in date AND digit? And if so, why are they classified this or that way (does this depend on the sentence content?).

# In[ ]:


unique_digits = set(train[train["class"]=="DIGIT"].before.unique().tolist())
unique_dates = set(train[train["class"]=="DATE"].before.unique().tolist())


# In[ ]:


overlap = unique_digits.intersection(unique_dates)
len(overlap)


# In[ ]:


list(overlap)[0:10]


# Yes, as we can see by this exampke: We  have overlapping class contents!!
# 
# This could be good or bad. On the one hand it could depend on the sentence content to which class a token was assigned to. Then the class would also contain this context information implicitly. But on the other hand what if this is not true and the class assignment is somehow dirty or not finetuned enough? Then this opens the door for class feature engineering... :-)

# ### New DataFrame: Sentence related information
# 
# 
# Before I continue, I like to create a new dataframe that holds some sentence related information: 
# * the sentence length
# * the number of chanced tokens 

# In[ ]:


train_sentences_info = pd.DataFrame(index=train.sentence_id.unique())
test_sentences_info = pd.DataFrame(index=test.sentence_id.unique())
train_sentences_info["length"] = train_sentences
test_sentences_info["length"] = test_sentences

train_sentences_info["num_changes"] = train.groupby("sentence_id")["change"].sum()


# In[ ]:


train_sentences_info.head()


# ### How many mean changes do we have per sentence? What is the maximum number of changes?

# In[ ]:


train_sentences_info["num_changes"].describe()


# That's fascinating: Up to 75 % of sentences only changed in one token or even didn't change at all. But there is one sentence with 94 changes!! Uff :-D How does it look like?

# ### The sentence that changed most... 

# In[ ]:


train_sentences_info[train_sentences_info.num_changes==94]


# In[ ]:


most_changed_sentence_id = train_sentences_info[train_sentences_info.num_changes==94].index.values[0]
most_changed_sentence = train[train.sentence_id==most_changed_sentence_id].before.values.tolist()
most_changed_sentence = ' '.join(most_changed_sentence)
most_changed_sentence


# Hui!

# In[ ]:


most_changed_sentence_after = train[train.sentence_id==most_changed_sentence_id].after.values.tolist()
most_changed_sentence_after = ' '.join(most_changed_sentence_after)
most_changed_sentence_after


# ### Do longer sentences have more changes than short ones?

# In[ ]:


plt.figure(figsize=(15,5))
#sns.jointplot(x="length", y="num_changes", data=train_sentences_info, kind="kde")
sns.jointplot(x="length", y="num_changes", data=train_sentences_info)


# For those that changed more than 1, I would say yes. Let's have a closer look for them:

# In[ ]:


plt.figure(figsize=(15,5))
sns.jointplot(x="length", y="num_changes", data=train_sentences_info[train_sentences_info.num_changes > 1])


# We can find a weak correlation between sentence lengths and number of changes. But this looks still like a diffuse cloud. We can't see a nice pattern.

# ### Which position in a sentence changed most often?
# 
# The token id tells us the position of a token in the sentence. I would guess that there are positions in a *specific kind of sentence* (whatever that means... must be specified) whose tokens changes more often. This would also relate to the language grammar structure. 
# 
# 

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x="token_id", data=train[(train.change==1) & (train.token_id <=30)])
plt.xlabel("Token ID")
plt.ylabel("Number of changes")


# Ok, we can see that sentences less or equal than 30 tokens (which corresponds to the major part of sentence length in this competition) changed most in the first and forth position. 

# ### Given sentences less or equal than 30 tokens, which positions in these sentences changed most often? 

# In[ ]:


collected = train[train.change==1][["sentence_id", "token_id"]]
collected["sentence_length"] = collected["sentence_id"].apply(lambda l: train_sentences_info.loc[l, "length"])
collected = collected[collected.sentence_length <= 30]
collected.head()


# In[ ]:


changed_positions = collected.groupby("sentence_length")["token_id"].value_counts().unstack()
changed_positions.describe()


# In[ ]:


changed_positions.fillna(0.0, inplace=True)
changed_positions = changed_positions.applymap(lambda l: np.log10(l+1))


# In[ ]:


mask = np.zeros_like(changed_positions.values)
mask[np.triu_indices_from(mask, k=2)] = True


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(changed_positions, mask=mask, cmap="magma")
plt.xlabel("position / token_id")
plt.ylabel("sentence length")
plt.title("Frequency of changed positions in sentences (log10-scale)")


# Very interesting! We can see...
# 
# * A lot of sentences changed in the second position or at the previous to last. The last is probably just a "." as it is the most common token but not the most changed one. 
# * Sentences that are 7 tokens long are somehow special: They occur very often (see distribution of sentence length in the train data) and they changed with very high frequency on their last positions. How do they look like?

# ## The train-7 sentences

# In[ ]:


train7_info = train_sentences_info[train_sentences_info.length==7]


# ### How do the changed  7-sentences look like with a change at position 4?

# In[ ]:


train7_sentence_ids = train7_info[train7_info.num_changes > 0].index.values
train7 = train[train.sentence_id.isin(train7_sentence_ids)]


# In[ ]:


train7_pos4_examples = train7[(train7.token_id==4) & (train7.change==1)].iloc[0:10,:]
sentence_ids = train7_pos4_examples.sentence_id.values
for idx in sentence_ids:
    before_sentence = train7[train7.sentence_id==idx].before.values.tolist()
    after_sentence = train7[train7.sentence_id==idx].after.values.tolist()
    before_sentence = ' '.join(before_sentence)
    after_sentence = ' '.join(after_sentence)
    print("before:" + before_sentence + "\n" + "_____after:" + after_sentence)


# To be continued :-)
