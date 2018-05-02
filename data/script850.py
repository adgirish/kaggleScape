
# coding: utf-8

# # Feature engineering
# 
# In this notebook i want try hand-crafting some features that could help to create a model. I want to see what creative ideas i can come up with - and if they indeed seem to work.

# In[16]:


import pandas as pd

df = pd.read_csv('../input/train.csv')


# In[17]:


df.head()


# Below i'm adding features to the dataset that are computed from the comment text. Some i've seen in discussions for this competition, others i came up with while looking at the data. Right now, they are:
# 
# * Length of the comment - my initial assumption is that angry people write short messages
# * Number of capitals - observation was many toxic comments being ALL CAPS
# * Proportion of capitals - see previous
# * Number of exclamation marks - i observed several toxic comments with multiple exclamation marks
# * Number of question marks - assumption that angry people might not use question marks
# * Number of punctuation symbols - assumption that angry people might not use punctuation
# * Number of symbols - assumtion that words like f*ck or $#* or sh*t mean more symbols in foul language (Thx for tip!)
# * Number of words - angry people might write short messages?
# * Number of unique words - observation that angry comments are sometimes repeated many times
# * Proportion of unique words - see previous
# * Number of (happy) smilies - Angry people wouldn't use happy smilies, right?

# In[18]:


df['total_length'] = df['comment_text'].apply(len)
df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                axis=1)
df['num_exclamation_marks'] = df['comment_text'].apply(lambda comment: comment.count('!'))
df['num_question_marks'] = df['comment_text'].apply(lambda comment: comment.count('?'))
df['num_punctuation'] = df['comment_text'].apply(
    lambda comment: sum(comment.count(w) for w in '.,;:'))
df['num_symbols'] = df['comment_text'].apply(
    lambda comment: sum(comment.count(w) for w in '*&$%'))
df['num_words'] = df['comment_text'].apply(lambda comment: len(comment.split()))
df['num_unique_words'] = df['comment_text'].apply(
    lambda comment: len(set(w for w in comment.split())))
df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
df['num_smilies'] = df['comment_text'].apply(
    lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))


# Let's inspect data - did this work?

# In[19]:


df.head()


# Now we'll calculation correlation between the added features and the to-be-predicted columns, this should be an indication of whether a model could use these features:

# In[20]:


features = ('total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks',
            'num_question_marks', 'num_punctuation', 'num_words', 'num_unique_words',
            'words_vs_unique', 'num_smilies', 'num_symbols')
columns = ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

rows = [{c:df[f].corr(df[c]) for c in columns} for f in features]
df_correlations = pd.DataFrame(rows, index=features)


# Let's output the data:

# In[21]:


df_correlations


# I'll also output the data as a heatmap - that's slightly easier to read.

# In[22]:


import seaborn as sns

ax = sns.heatmap(df_correlations, vmin=-0.2, vmax=0.2, center=0.0)


# So, what have we learned? Some of the feature ideas i had make sense: They correlate with the to-be-predicted data, so a model should be able to use them. Other feature ideas don't correlate - so they look less promising.
# 
# For now these feature seem the best candidates:
# * Proportion of capitals 
# * Number of unique words
# * Number of exclamation marks
# * Number of punctuations
# 
# Hope this could be usefull to someone! If you have more (feature) ideas or feedback - please comment, then i can add them here.
