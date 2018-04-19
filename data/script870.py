
# coding: utf-8

# In[ ]:


import json, datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


from datetime import datetime, timedelta
from collections import OrderedDict


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Files
# ### `data.json` / `data.csv`
# 
#  * **date (str)** : the date of publication (or last update) of the deck.
#  * **user (str)** : the user who uploaded the deck.
#  * **deck_class (str)** : one of the nine character class in Hearthstone (`Druid`, `Priest`, ...).
#  * **deck_archetype (str)** : the theme of deck labelled by the user (`Aggro Druid`, `Dragon Priest`, ...).
#  * **deck_format (str)** : the game format of the deck on the day data was recorded (`W` for "Wild" or `S` for "Standard").
#  * **deck_set (str)** : the latest expansion published prior the deck publication (`Naxxramas`, `TGT Launch`, ...).
#  * **deck_id (int)** : the ID of the deck.
#  * **deck_type (str)** : the type of the deck labelled by the user :
#     - *Ranked Deck* : a deck played on ladder.
#     - *Theorycraft* : a deck built with unreleased cards to get a gist of the future metagame.
#     - *PvE Adventure* : a deck built to beat the bosses in adventure mode.
#     - *Arena* : a deck built in arena mode.
#     - *Tavern Brawl* : a deck built for the weekly tavern brawl mode.
#     - *Tournament* : a deck brought at tournament by a pro-player.
#     - *None* : the game type was not mentioned.
#     
#  * **rating (int)** : the number of upvotes received by that deck.
#  * **title (str)** : the name of the deck.
#  * **craft_cost (int)** : the amount of dust (in-game craft material) required to craft the deck.
#  * **cards (list)** : a list of 30 card ids. Each ID can be mapped to the card description using the reference file.
# 
# ### `refs.json` 
# Contains the reference to the cards played in Hearthstone. This file was originally proposed on [HearthstoneJSON](https://hearthstonejson.com/). Each record features a lot of informations about the cards, I'll list the most important :
# 
# * **dbfId (int)** : the id of the card (the one used in `data.json`).
# * **rarity (str)** : the rarity of the card (`EPIC`, `RARE`, ...).
# * **cardClass (str)** : the character class (`WARLOCK`, `PRIEST`, ...).
# * **artist (str)** : the artist behind the card's art.
# * **collectible (bool)** : whether or not the card can be collected.
# * **cost (int)** : the card play cost.
# * **health (int)** : the card health (if it's a minion).
# * **attack (int)** : the card attack (if it's a minion).
# * **name (str)** : the card name.
# * **flavor (str)** : the card's flavor text.
# * **set (str)** : the set / expansion which featured this card.
# * **text (int)** : the card's text.
# * **type (str)** : the card's type (`MINION`, `SPELL`, ...).
# * **race (str)** : the card's race (if it's a minion).
# * **set (str)** : the set / expansion which featured this card.
# * ...

# In[ ]:


with open('../input/data.json') as file:
    data = json.load(file)


# In[ ]:


with open('../input/refs.json') as file:
    refs = json.load(file)


# In[ ]:


# to dataframe
decks = pd.DataFrame(data)


# In[ ]:


# transform date strings to datetime objects
decks['date'] = pd.to_datetime(decks['date'])


# In[ ]:


# reformat the card column
card_col = ['card_{}'.format(str(i)) for i in range(30)]
cards = pd.DataFrame([c for c in decks['cards']], columns=card_col)
cards = cards.apply(np.sort, axis=1)
decks = pd.concat([decks, cards], axis=1)
decks = decks.drop('cards', axis=1)


# In[ ]:


# remove tabs and newlines from user names
decks['user'] = decks['user'].apply(str.strip)


# In[ ]:


# delete unnecessary variables
del(data)
del(cards)


# # 2. Cleaning
# I included a few safety nets in the scraping pipeline in order to collect playable decks only (decks containing exactly 30 cards), but I might have downloaded noisy data without noticing. Let's check that out!

# In[ ]:


raw_length = len(decks)
print ('Number of decks :', raw_length)


# ### 2.1. Dates
# First thing we want to check is the consistency in the publication date of the decks. Let's see when the dataset starts and when it ends :

# In[ ]:


print ('First deck :', min(decks['date']))
print ('Last deck :', max(decks['date']))


# Given that the game was released worldwide on March 11, 2014, it seems a bit weird to have decks built prior to this date. I guess that they were built during the beta by testers. Because who knows what happened back then, it safer to get rid of these decks for now :

# In[ ]:


release_date = datetime(2014, 3, 11)
decks = decks[decks['date'] > release_date]


# In[ ]:


# now let's check the dates
print ('First deck :', min(decks['date']))
print ('Last deck :', max(decks['date']))


# In[ ]:


prc_left = round(len(decks) / raw_length * 100)
print ('Decks removed :', raw_length - len(decks))
print ('Original dataset left :', prc_left, '%')


# If we have unique ids, we don't have duplicates. Let's look for potential duplicates :

# In[ ]:


assert len(decks['deck_id'].unique()) == len(decks)


# ### 2.2. Game type
# Hearthstone has two main types of ladder games : "Standard" (where only recent and basic cards are allowed) and "Wild" (where all cards are allowed). Expansion cards rotate every year when new material is proposed by Blizzard. This means that an expansion card that was part of the "Standard" set last year has moved to the "Wild" set now. 
# 
# Here, we will just assume that all decks are "Standard" :

# In[ ]:


# decks = decks[decks['deck_format'] == 'W']


# ### 2.3. Deck Archetypes
# A few decks are marked as "Edit" ; I don't know what it means so I prefer to change it back to "Unknown" :

# In[ ]:


decks.loc[(decks['deck_archetype'] == 'Edit'), 'deck_archetype'] = 'Unknown'


# In[ ]:


# duplicates check
assert len(decks['deck_id'].unique()) == len(decks)


# ### 2.4. Deck Types
# Now let's have a look a the type of decks included in the dataset :

# In[ ]:


decks['deck_type'].value_counts()


# At the moment, we are interested in decks played on ladder only, so I think we can keep "Ranked Deck" and "Tournament". Tournament decks rarely differ from decks played on ladder (because everyone looks for the most competitive decks).
# 
# By dropping the "None" and "Theorycraft" categories, we lose a lot of data, which is annoying. There are two ways to save a few observations :
# 
# * Looking for keywords in the titles (like "Ranked") and keep the records with these words.
# * Keep the decks that are already duplicates of "Ranked Decks".
# 
# I think that the later option is safer because we are less likely to include garbage into the final data set. Let's give it a try :

# In[ ]:


# get the none and ranked decks
none_decks = decks[decks['deck_type'] == 'None']
ranked_decks = decks[decks['deck_type'].isin(['Ranked Deck', 'Tournament'])]


# In[ ]:


# looks for none ids with cards already in ranked
none_ids = pd.merge(none_decks, ranked_decks, on=card_col, how='inner')['deck_id_x']


# In[ ]:


# add the none_ids decks to ranked
none_could_be_ranked = none_decks[none_decks['deck_id'].isin(none_ids)]
ranked_decks = pd.concat([none_could_be_ranked, ranked_decks])


# In[ ]:


# the same for theorycraft decks
theory_decks = decks[decks['deck_type'] == 'Theorycraft']
theory_decks_ids = pd.merge(theory_decks, ranked_decks, on=card_col, how='inner')['deck_id_x']
theory_could_be_ranked = theory_decks[theory_decks['deck_id'].isin(theory_decks_ids)]
decks = pd.concat([theory_could_be_ranked, ranked_decks])


# In[ ]:


# duplicates check
assert len(decks['deck_id'].unique()) == len(decks)


# In[ ]:


prc_left = round(decks.shape[0] / raw_length * 100)
print ('Decks removed :', raw_length - decks.shape[0])
print ('Original dataset left :', prc_left, '%')


# So, after cleaning our set contains "None" and "Theorycraft" decks that could also be found in "Ranked" (so we consider them as ranked decks). We got rid of 38% of the original set. 

# # 3. Exploration
# In order to get insights from this data set, I'll be mosting looking at the number of submissions. Posting a deck on Hearthpwn doesn't mean that this deck was actually played in game (we don't have access to Blizzard's data), but I'll assume that it constitutes a reasonable proxy.

# ### 3.1. Timely Deck Submissions
# Let's have a look at the weekly submissions recorded on Hearthpwn. I wonder how the release of a new expansion motivates players at trying new decks. It should be possible to retrieve the release dates from the data set, but I did a little bit of research to find the official dates.

# In[ ]:


release_dates = {
    'Explorers' : datetime(2015, 11, 12),
    'Old Gods' : datetime(2016, 4, 26),
    'Classic Nerfs' : datetime(2016, 3, 14),
    'Yogg Nerf' : datetime(2016, 10, 3),
    'Karazhan' : datetime(2016, 8, 11),
    'Gadgetzan' : datetime(2016, 12, 1),
    'Naxx Launch' : datetime(2014, 7, 22),
    'Live Patch 5506' : datetime(2014, 5, 28),
    'Undertaker Nerf' : datetime(2015, 1, 29),
    'Blackrock Launch' : datetime(2015, 4, 2),
    'GvG Launch' : datetime(2014, 12, 8),
    'TGT Launch' : datetime(2015, 8, 24),
    'Warsong Nerf' : datetime(2016, 1, 16),
    'Live Patch 4973' : datetime(2014, 3, 14),
    'Aggro Downfall' : datetime(2017, 2, 28),
    # 'Beta Patch 4944' : datetime(2014, 3, 11),
    # 'GvG Prelaunch' : datetime(2014, 12, 5)
}


# In[ ]:


date_decks = decks.set_index(pd.DatetimeIndex(decks['date'])).sort_index()


# In[ ]:


weekly_submissions = date_decks.resample('W')['date'].count()


# In[ ]:


fig = plt.figure()
ax = weekly_submissions.plot(figsize=(25, 10), fontsize=15)

for key, date in release_dates.items():
    ax.axvline(date, color='green', alpha=.35)
    ax.text(date, 12000, key, rotation=90, fontsize=15)


# **Observation** : each expansion/adventure seems to be followed by a burst in deck submissions. Expansions contain much more news cards (~130) than adventures (~30-40), which may explain why more decks are submitted on expansions launch compared to adventures. *Wispers of the Old Gods* seem to have been particularily inspiring! Notice that each peak is followed by a sharp decrease in submissions over the next weeks, which suggests that the meta game stabilizes over time and new content is required to maintain the players involved.

# ### 3.2. Most played class
# We can also have a look at the character classes favored by the players. Hearthstone has nine different character classes : *Mage*, *Priest*, *Paladin*, *Warrior*, *Shaman*, *Druid*, *Rogue*, *Hunter* and *Warlock* :

# In[ ]:


class_count = decks['deck_class'].value_counts()
class_count_df = class_count.to_frame().reset_index()


# In[ ]:


colors = {
    'Druid' : 'sandybrown',
    'Hunter' : 'green',
    'Mage' : 'royalblue',
    'Paladin' : 'gold',
    'Priest' : 'lightgrey',
    'Rogue' : 'darkgrey',
    'Shaman' : 'darkblue',
    'Warlock' : 'purple',
    'Warrior' : 'firebrick',
}

# sort colors to make plotting easier
colors = OrderedDict(sorted(colors.items()))


# In[ ]:


class_count_df['color'] = class_count_df['index'].replace(colors)


# In[ ]:


class_count_df.plot.pie(
    y='deck_class', 
    labels=class_count_df['index'], 
    colors=class_count_df['color'],
    autopct='%.2f', 
    fontsize=15,
    figsize=(10, 10),
    legend=False,
)


# **Observation** : All classes seem to be equally represented in the past two years. There is a slight preference for *Mage* and *Priest* decks. However, this number is likely to have evolved over time. Some class might have been popular at so point, but not popular anymore after a given expansion.

# In[ ]:


weekly_classes = date_decks.groupby('deck_class').resample('W').size().T
weekly_classes_rf = weekly_classes.divide(weekly_classes.sum(axis=1), axis=0)

ax = weekly_classes_rf.plot(
    kind='area', 
    figsize=(20, 12), 
    color=colors.values(), 
    alpha=.35,
    legend=False,
    fontsize=15,
)

ax.set_ylim([0, 1])

for key, date in release_dates.items():
    ax.axvline(date, color='grey')
    ax.text(date, 1.2, key, rotation=90, fontsize=15)


# **Observation** : it's difficult to identify clear trends from this visualization without looking at the rise and fall of specific deck archetypes. But it is still possible to spot popularity bursts in certain classes after a given content update, for instance :
# 
# * *Warrior* got very popular right after *Blackrock Mountain* launch and dominated the meta after *Whispers of the Old Gods*
# * *Paladin* was trending after *TGT* was released and knew a peak after *Karazhan*.
# * *Shaman* inspired players after *Gadgetzan*.
# * *Warlock* was mostly played around *Naxxramas*.

# ### 3.3. Most rated decks / deck builders
# Hearthpwn offers the possibility to rate players' decks. A quick look at the rating distribution relative frequencies suggests that the vast majority of the decks do not get any ratings :

# In[ ]:


rating_count = decks['rating'].value_counts(normalize=True).sort_index()


# In[ ]:


rating_count.head(n=10)


# Let's have a look at the deck builders. Maybe the number of submissions is more informative :

# In[ ]:


users = decks.groupby('user')['rating'].count().sort_values(ascending=False)


# In[ ]:


users.head(n=30).plot(kind='bar', figsize=(20, 10), fontsize=15, color=sns.color_palette())


# **Observation** : the top submitters are pro-players, streamers & popular YouTubers. This is not really surprising since publishing competitive (or fun) decks is a way to increase popularity and attract new viewers. Hearthpwn has also a feature allowing users to assign a deck to another user. So it is possible that many of these decks were compiled and assigned by fans or followers.

# ### 3.4. Most played archetype

# In[ ]:


# total of differents archetypes
decks['deck_archetype'].value_counts().size


# It seems that we have a total of 74 deck archetypes (75 minus the "unknown" category). Note that Hearthpwn seems to have implemented the feature allowing users to specify the deck archetype around *Whispers of the Old Gods*, which means that, unless a player updates his old decks, deck archetype remains "Unknown" for pre-*Old Gods* decks. So the data we will explore now represent about a year of deck submissions. Let's have a look at the average rating for each archetype :

# In[ ]:


known = decks[decks['deck_archetype'] != 'Unknown']
counts = known.groupby(['deck_class', 'deck_archetype']).size().reset_index()


# In[ ]:


for i, group in counts.groupby('deck_class'):
    fig = plt.figure()
    group.sort_values(0, ascending=False).plot(
        kind='bar', 
        x='deck_archetype', 
        title=str(i),
        color=colors[str(i)],
        legend=False,
        figsize=(15, 6),
        fontsize=15,
    )


# **Observation** : this overview of the decks archetypes shows that some classes tend to generate more diverse decks than others. For instance, *Rogue*, *Priest* and *Druid* show at least 10 differents archetypes, whereas *Hunter* has only 6. Some archetypes seem to dominate their classes, like *Control Warrior* which is clearly over-represented  or *Midrange Shaman*, and some archetypes represent fringe or meme types of decks, like *Mech Shaman* who almost never saw play. This means that the data we have to deal with is highly imbalanced.
# 
# Moreover, this overview does not inform us about the evolution of archetypes in time. Some archetypes that seem to be less popular now might have been very popular earlier one (*Oil Rogue* for instance) but we don't have that information since a lot of the decks archetypes are marked as "Unknown". In order to get a better idea of the evolution of the trends, we might want to try to infer the archetype from the cards in the deck.
# 
# I'm going to try to classify the decks using a Random Forest algorithm. The 30 cards of the decks constitute the features we are going to work with.

# # 4. Archetype Classification

# ### 4.1. Random Forest (first-attempt)

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split


# In[ ]:


def predict_archetype(class_name, decks, card_col, n_trees=500, max_feats=5, log=True):

    # known / unknown split
    dclass = decks[decks['deck_class'] == class_name]
    known = dclass[dclass['deck_archetype'] != 'Unknown']
    unknown = dclass[dclass['deck_archetype'] == 'Unknown']

    # data / target split
    X = known[card_col]
    y = known['deck_archetype']

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)
    
    # random forest
    clf = RandomForestClassifier(
        n_estimators=n_trees, 
        max_features=max_feats, 
        class_weight=None
    )
    
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    
    # metrics
    conf_matrix = pd.DataFrame(
        confusion_matrix(y_test, pred), 
        columns=clf.classes_, 
        index=clf.classes_
    )
    
    if log:
        print (classification_report(y_test, pred))
        # print (cohen_kappa_score(y_test, pred))
        # print (conf_matrix)
    
    return clf


# In[ ]:


# save the classifiers each class
clfs = {}
for c in decks['deck_class'].unique():
    clfs[c] = predict_archetype(c, decks, card_col)


# **Observation** : Meh. Even though Random Forests might be the most well suited algorithm for this task, the accuracy of the prediction is not satisfactory. Let's have a closer look the predictions produced by one of our classifiers (paladin, for instance) :

# In[ ]:


paladin_clf = clfs['Paladin']


# In[ ]:


paladin_set = decks[decks['deck_class'] == 'Paladin']
paladin_ukn = paladin_set[paladin_set['deck_archetype'] == 'Unknown']
paladin_ukn.is_copy = False


# In[ ]:


pred = paladin_clf.predict(paladin_ukn[card_col])


# In[ ]:


paladin_ukn['deck_archetype'] = pred


# In[ ]:


paladin_ukn = paladin_ukn.set_index(pd.DatetimeIndex(paladin_ukn['date'])).sort_index()
weekly_paladin = paladin_ukn.groupby('deck_archetype').resample('W').size().reset_index()
weekly_paladin_piv = weekly_paladin.pivot(index='date', columns='deck_archetype', values=0).fillna(0)
weekly_paladin_rf = weekly_paladin_piv.divide(weekly_paladin_piv.sum(axis=1), axis=0)

ax = weekly_paladin_rf.plot(
    kind='area', 
    figsize=(20, 5), 
    legend=True,
    fontsize=15,
    color=sns.color_palette('Set3', 10)
)

ax.set_ylim([0, 1])

for key, date in release_dates.items():
    ax.axvline(date, color='grey')
    ax.text(date, 1.5, key, rotation=90, fontsize=15)


# **Observation** : well, indeed, it's not really accurate. *Reno Pally* was only introduced in the *Explorers* pack (so all archetypes labelled as such prior to this adventure are errors) and *N'Zoth* is a card from... *Old Gods*. Yet, those who have a little experience of the game can aknownledge that *N'Zoth Paladin* was indeed a big deal after *Old Gods* and *Secret Pally* became the plague after TGT (which is also correct).
# 
# So it seems that our algorithm does not put enough weight on the cards that constitute the core of the archetypes. There are probably several explanations for that : 
# 
# * **Imbalance** : the archetypes are just too imbalanced to yield good results. We can try to over-sample the classes with a very low number of observations (using SMOTE algorithm) to improve our model and boost recall scores.
# 
# * **Card Rotation** : as we have noticed earlier, all expansions / adventure cards rotate from the "Standard" format to the "Wild" format. This between-expansion variance might contribute to miss-classification. For instance a *Fatigue Mage* played in "Standard" today might very well contain more cards from the *Freeze Mage* in the *Wild* format than it's current *Wild* version, so both get confused easily (I'm making this up, but you get the idea). This could be particularily important since most of the labeled decks appeared after *Old Gods* on Hearthpwn. I think giving our algorithm information about the card rotation might improve our prediction. One way to achieve this is to build new features based on the number of card in the deck belonging to each expansion. The logic here is that associating *N'Zoth* decks (for instance) to post-*Old Gods* cards could reduce miss-identification in pre-*Old Gods* archetypes. Let's try this.

# ### 4.2. Random Forest (with resampling & feature engineering)

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


# building a card id - set mapping from refs file
refs_dict = {c.get('dbfId') : c.get('set') for c in refs}


# In[ ]:


def multi_smote(X, y, kind='svm'):
    
    # regroup observations by classes
    full = pd.concat([X, y], axis=1)
    by_arch = full.groupby('deck_archetype')

    samples = []
    
    for name, group in by_arch:
        
        # create a 2-classes dataset
        all_but_one = full[full['deck_archetype'] != name]
        all_but_one.is_copy = False
        all_but_one['deck_archetype'] = 'Other'
        
        toSMOTE = pd.concat([group, all_but_one])
        _X = toSMOTE[X.columns]
        _y = toSMOTE['deck_archetype']
        
        # resample with 2 classes
        sm = SMOTE(kind=kind)
        X_re, y_re = sm.fit_sample(_X, _y)
        re = np.column_stack([X_re, y_re])
        
        # remove reference to other
        re = re[~(re == 'Other').any(axis=1)]
    
        samples.append(re)
        
    resampled = np.concatenate(samples)
    
    return resampled[:, :len(X.columns)], resampled[:, -1]


# Here is the updated function including SMOTE resampling and set features :

# In[ ]:


def predict_archetype(class_name, decks, card_col, refs, n_trees=500, max_feats=5, log=True):

    # known / unknown split
    dclass = decks[decks['deck_class'] == class_name]
    known = dclass[dclass['deck_archetype'] != 'Unknown']
    unknown = dclass[dclass['deck_archetype'] == 'Unknown']

    # data / target split
    X = known[card_col]
    y = known['deck_archetype']
    
    # adding expansions counts
    set_df = known[card_col].apply(pd.Series.replace, to_replace=refs_dict)
    counts = set_df.apply(pd.value_counts, axis=1).fillna(0)
    X = pd.concat([X, counts], axis=1)

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5)
    
    # over-sampling the training set
    X_train, y_train = multi_smote(X_train, y_train)
    
    # random forest
    clf = RandomForestClassifier(
        n_estimators=n_trees, 
        max_features=max_feats, 
        class_weight=None
    )
    
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    
    # metrics
    conf_matrix = pd.DataFrame(
        confusion_matrix(y_test, pred), 
        columns=clf.classes_, 
        index=clf.classes_
    )
    
    if log:
        print (classification_report(y_test, pred))
        # print (conf_matrix)
        # print (cohen_kappa_score(y_test, pred))
        # print (clf.feature_importances_)
    
    return clf


# In[ ]:


# save the classifiers each class
clfs = {}
for c in decks['deck_class'].unique():
    clfs[c] = predict_archetype(c, decks, card_col, refs_dict)


# **Observation** : resampling and adding new features slighly improved the recall score for small classes as expected, but this improvement is rather small (and it sometimes reduces overall performances on some subsets of the data). There are are other approachs that we can try :
# 
# * **Disregard fringe deck archetypes** : who plays *Water Rogue* anyway? To improve the prediction, we could just get rid of the archetypes with less than, say, 100 decks.
# 
# * **Group the data by expansion** : to reduce the variance induced by new cards, we can train a different model for each expansion time-span ; but this is obviously less interesting than a general model.
# 
# * **Go unsupervised** : we can consider that the labels reported by Hearthpwn users do not represent accuratelly the variety of archetypes in Hearthstone. So we could group the decks the way we want to see other trends emerge, using LDA algorithm, for instance. Let's try this option.

# ### 4.3. LDA
# LDA is generally used to discover topics among text samples. When you think about it, our decks can also be considered like texts with similar or different words (cards). Decks that share the same cards represent a same topic (archetype). 

# In[ ]:


from gensim import corpora, models


# In[ ]:


# building a card id - set mapping from refs file
names_dict = {c.get('dbfId') : c.get('name') for c in refs}

# we'll test paladin, as an example
subset = decks[decks['deck_class'] == 'Paladin']
names_df = subset[card_col].apply(pd.Series.replace, to_replace=names_dict)
lists = names_df.values.tolist()
dictionary = corpora.Dictionary(lists)
corpus = [dictionary.doc2bow(l) for l in lists]

# we'll consider the 10 most important topics, as a starter
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)

topics = ldamodel.print_topics(num_topics=10, num_words=30)

def get_topic(row):
    '''This function returns the most likely archetype'''
    topics = ldamodel.get_document_topics(row)
    best = max(topics, key=lambda x: x[1])
    return best[0]

# get the archetypes based on card BOW
pred = [get_topic(d) for d in corpus]
subset.is_copy = False
subset['deck_archetype'] = pred


# In[ ]:


paladin_ukn = subset.set_index(pd.DatetimeIndex(subset['date'])).sort_index()
weekly_paladin = paladin_ukn.groupby('deck_archetype').resample('W').size().reset_index()
weekly_paladin_piv = weekly_paladin.pivot(index='date', columns='deck_archetype', values=0).fillna(0)
weekly_paladin_rf = weekly_paladin_piv.divide(weekly_paladin_piv.sum(axis=1), axis=0)

ax = weekly_paladin_rf.plot(
    kind='area', 
    figsize=(20, 8), 
    legend=True,
    fontsize=15,
    color=sns.color_palette('Set3', 10)
)

ax.set_ylim([0, 1])

for key, date in release_dates.items():
    ax.axvline(date, color='grey')
    ax.text(date, 1.3, key, rotation=90, fontsize=15)


# **Observation** : now we're talking! Although not perfect, the categories found by the LDA model make more sense. Upon closer inspection, a Hearthstone player will recognize popular decks among the numbers :

# In[ ]:


# The infamous secret-paladin from TGT.
topics[5]


# In[ ]:


# A mix of several types of aggressive paladin decks
topics[0]


# In[ ]:


# a classic post-Naxxramas midrange paladin
topics[9]


# In[ ]:


# Buff paladin
topics[8]


# In[ ]:


# Murloc paladin
topics[3]


# In[ ]:


# N'zoth paladin
topics[4]


# In[ ]:


# Dragon paladin (starting with Blackrock)
topics[2]


# # 5. Conclusion
# We have seen that archetypes selected by Hearthpwn users represent only the more recent versions of the decks, which means that older decks hardly felt into theses categories. LDA gave us a better idea of the life and death of various archetypes, but requires also some knowledge of the game.
# 
# This data set is a lot of fun and there is still a lot to do (regarding the deck ratings, the craft cost, etc.). Thanks for the read!
