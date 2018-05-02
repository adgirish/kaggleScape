
# coding: utf-8

# - "I want candy."
# - "What's the magic word?"
# - "Pleeaaaaase!"
# 
# This is what many parents teach to their children. But is this really the best way to get what you want? Is "please" really the magic word?
# 
# More precisely, we are going to look at forum messages and try to answer 2 questions:
#  - What are the best words to use in order to get replies?
#  - What are the best words to use in order to have +1s?
#  
# 

# In[ ]:


# We select messages that have gotten replies
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
fm = pd.read_csv('../input/ForumMessages.csv')
rep_ids = fm['ReplyToForumMessageId'].unique().tolist()
rep_fm = fm[fm['Id'].isin(rep_ids)]

# We store the number of replies for each message
nb_replies = fm['ReplyToForumMessageId'].value_counts()
rep_fm['NbReplies'] = rep_fm['Id'].map(nb_replies)

# We keep the words that are not quotes from other messages
import re
def words_wo_quotes(mes):
    mes_wo_q = re.sub(r'\[quote=.*\[/quote\]', '', mes, flags=re.DOTALL)
    return set([w for w in re.split('[^a-z]', mes_wo_q.lower()) if len(w)>=2])
rep_fm['Message'] = rep_fm['Message'].astype(str).apply(words_wo_quotes)

# For each word, we store in a dict the number of replies to messages where this word appears
from collections import defaultdict
dict_rep = defaultdict(int)
def fill_dict_rep(x):
    for w in x[0]:
        dict_rep[w] += x[1]
rep_fm[['Message', 'NbReplies']].apply(fill_dict_rep, axis=1)

# For each word, we count the number of messages where it appears
all_messages = fm['Message'].astype(str).apply(words_wo_quotes)
dict_all = dict.fromkeys(dict_rep.keys(), 0)
def fill_dict_all(x):
    for w in x:
        if w in dict_rep:
            dict_all[w] += 1
all_messages.apply(fill_dict_all)

# We will only consider words that appear in at least 1% of the messages
min_proportion = fm.shape[0]/100

# We sort the words according to the ratio (number of replies)/(number of messages where it appears)
lwords = []
for w in dict_all:
    if dict_all[w]>min_proportion:
        lwords.append([w, dict_rep[w]/dict_all[w]])
lwords = sorted(lwords, key=lambda h:h[1], reverse=True)

# We plot the 20 "best" words
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["figure.figsize"] = (10, 12)
x_axis = [wx[1] for wx in lwords[:20]]
y_axis = [wx[0] for wx in lwords[:20]]
plt.barh(range(60, 0, -3), [1]*20, height=1.5, alpha=0.4)
plt.barh(range(60, 0, -3), x_axis, height=1.5, alpha=0.4)
plt.yticks(np.arange(60.5, 0.5, -3), y_axis, fontsize=20)
plt.xlabel('Proportion of replies', fontsize=16)
plt.ylabel('Word used', fontsize=16)
plt.title('The 20 best words to use to get replies', fontsize=22)
plt.show()


# Some remarks:
# - Meta informations like "prize", "rules", "leaderboard" or "contest" have good success.
# - "Feeling curious" seems to be a good idea :)
# 
# The word "please" ranks only 509th of 640!!
# 
# Now what about the relationship between the words we use and the score we get for a message?
# 
# 

# In[ ]:


# We reload the csv
fm = pd.read_csv('../input/ForumMessages.csv')
fm = fm[['Message', 'Score']]
fm['Message'] = fm['Message'].astype(str).apply(words_wo_quotes)

# For each word, we count the number of messages where it appears and the corresponding scores 
score_dict = defaultdict(int)
count_dict = defaultdict(int)
def fill_dict(x):
    for w in x[0]:
        score_dict[w] += x[1]
        count_dict[w] += 1
fm.apply(fill_dict, axis=1)

# We sort the words according to the ratio (scores of the messages where it appears)/(number of messages where it appears)
lwords = []
for de in score_dict:
    if count_dict[de]>min_proportion:
        lwords.append([de, score_dict[de]/count_dict[de]])
lwords = sorted(lwords, key=lambda h:h[1], reverse=True)

# Once again, we plot the 20 "best" words
plt.rcParams["figure.figsize"] = (10, 12)
x_axis = [wx[1] for wx in lwords[:20]]
y_axis = [wx[0] for wx in lwords[:20]]
plt.barh(range(60, 0, -3), x_axis, height=1.5, alpha=0.6)
plt.yticks(np.arange(60.5, 0.5, -3), y_axis, fontsize=20)
plt.xlabel('Average score', fontsize=16)
plt.ylabel('Word used', fontsize=16)
plt.title('The 20 best words to use to have a good score', fontsize=22)
plt.show()


# Some remarks:
# - Since I did not drop HTML tags, we can see that it is very good practice to use unordered lists!
# - It seems that people like to see github repos.
# - Several words attracting interest are about the techniques that are used: engineering, linear, tree, ensemble, neural, logistic...
# 
# And to answer our former question, with a rank of 620th of 640, the word "please" does not appear to be the magic word!! 
# 
# - "So what's the magic word?"
# - "uuuuuuuuuul!"
# 
# :)
# 
# 
