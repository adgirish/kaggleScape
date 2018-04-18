
# coding: utf-8

# # **Film recommendation engine**
# *Fabien Daniel (July 2017)*
# ___
# This notebook aims at building a recommendation engine from the content of the TMDB dataset that contains around 5000 movies and TV series.Basically, the engine will work as follows: after the user has provided the name of a film he liked, the engine should be able to select in the database a list of 5 films that the user will enjoy. In practice, recommendation engines are of three kinds:
# - **popularity-based** engines: usually the most simple to implement be also the most impersonal
# - **content-based** engines: the recommendations are based on the description of the products
# - **collaborative filtering** engines: records from various users provide recommendations based on user similarities
# 
# In the current case, since the dataset only describe the content of the films and TV series, collaborative filtering is excluded and I will thus build an engine that uses both the content and the popularity of the entries.
# ___
# _**Acknowledgement**: many thanks to [J. Abécassis](https://www.kaggle.com/judithabk6) for the advices and help provided during the writing of this notebook_ <br>
# _In september 2017, the original dataset used by this kernel was replaced. [Sohier Dane](https://www.kaggle.com/sohier) did all the job to adapt the original kernel to the new format and the modifications are described in [this kernel](https://www.kaggle.com/sohier/film-recommendation-engine-converted-to-use-tmdb). Many thanks to Sohier for doing this !_
# ___
# This notebook is organized as follows:
# 
# **1. Exploration**
# - 1.1 Keywords
# - 1.2 Filling factor: missing values
# - 1.3 Number of films per year
# - 1.4 Genres
# 
# ** 2. Cleaning**
# - 2.1 Cleaning of the keywords
#     * 2.1.1 Grouping by roots
#     * 2.1.2 Groups of synonyms
# - 2.2 Correlations
# - 2.3 Missing values
#     * 2.3.1 Setting missing title years
#     * 2.3.2 Extracting keywords from the title
#     * 2.3.3 Imputing from regressions
#     
# **3. Recommendation Engine**
# - 3.1 Basic functioning of the engine 
#     * 3.1.1 Similarity
#     * 3.1.2 Popularity
# - 3.2 Definition of the recommendation engine functions
# - 3.3 Making meaningfull recommendations
# - 3.4 Exemple of recommendation: test-case
# 
# **4. Conclusion: possible improvements and points to adress**
# 
# 

# ___
# ## 1. Exploration
# 
#  First, we define a few functions to create an interface with the new structure of the dataset.
#  The code below is entirely taken from [Sohier's kernel](https://www.kaggle.com/sohier/film-recommendation-engine-converted-to-use-tmdb):

# In[ ]:


import json
import pandas as pd
#___________________________
def load_tmdb_movies(path):
    df = pd.read_csv(path)
    df['release_date'] = pd.to_datetime(df['release_date']).apply(lambda x: x.date())
    json_columns = ['genres', 'keywords', 'production_countries',
                    'production_companies', 'spoken_languages']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df
#___________________________
def load_tmdb_credits(path):
    df = pd.read_csv(path)
    json_columns = ['cast', 'crew']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    return df
#___________________
LOST_COLUMNS = [
    'actor_1_facebook_likes',
    'actor_2_facebook_likes',
    'actor_3_facebook_likes',
    'aspect_ratio',
    'cast_total_facebook_likes',
    'color',
    'content_rating',
    'director_facebook_likes',
    'facenumber_in_poster',
    'movie_facebook_likes',
    'movie_imdb_link',
    'num_critic_for_reviews',
    'num_user_for_reviews']
#____________________________________
TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES = {
    'budget': 'budget',
    'genres': 'genres',
    'revenue': 'gross',
    'title': 'movie_title',
    'runtime': 'duration',
    'original_language': 'language',
    'keywords': 'plot_keywords',
    'vote_count': 'num_voted_users'}
#_____________________________________________________
IMDB_COLUMNS_TO_REMAP = {'imdb_score': 'vote_average'}
#_____________________________________________________
def safe_access(container, index_values):
    # return missing value rather than an error upon indexing/key failure
    result = container
    try:
        for idx in index_values:
            result = result[idx]
        return result
    except IndexError or KeyError:
        return pd.np.nan
#_____________________________________________________
def get_director(crew_data):
    directors = [x['name'] for x in crew_data if x['job'] == 'Director']
    return safe_access(directors, [0])
#_____________________________________________________
def pipe_flatten_names(keywords):
    return '|'.join([x['name'] for x in keywords])
#_____________________________________________________
def convert_to_original_format(movies, credits):
    tmdb_movies = movies.copy()
    tmdb_movies.rename(columns=TMDB_TO_IMDB_SIMPLE_EQUIVALENCIES, inplace=True)
    tmdb_movies['title_year'] = pd.to_datetime(tmdb_movies['release_date']).apply(lambda x: x.year)
    # I'm assuming that the first production country is equivalent, but have not been able to validate this
    tmdb_movies['country'] = tmdb_movies['production_countries'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['language'] = tmdb_movies['spoken_languages'].apply(lambda x: safe_access(x, [0, 'name']))
    tmdb_movies['director_name'] = credits['crew'].apply(get_director)
    tmdb_movies['actor_1_name'] = credits['cast'].apply(lambda x: safe_access(x, [1, 'name']))
    tmdb_movies['actor_2_name'] = credits['cast'].apply(lambda x: safe_access(x, [2, 'name']))
    tmdb_movies['actor_3_name'] = credits['cast'].apply(lambda x: safe_access(x, [3, 'name']))
    tmdb_movies['genres'] = tmdb_movies['genres'].apply(pipe_flatten_names)
    tmdb_movies['plot_keywords'] = tmdb_movies['plot_keywords'].apply(pipe_flatten_names)
    return tmdb_movies


# Then, I load in a single place all the packages that will be used throughout the notebook and then load the dataset. Then, I give some information on the columns types and the number of missing values.

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input/tmdb-movie-metadata/"]).decode("utf8"))


# In[ ]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math, nltk, warnings
from nltk.corpus import wordnet
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from wordcloud import WordCloud, STOPWORDS
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"
pd.options.display.max_columns = 50
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
PS = nltk.stem.PorterStemmer()
#__________________
# load the dataset
credits = load_tmdb_credits("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")
movies = load_tmdb_movies("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")
df_initial = convert_to_original_format(movies, credits)
print('Shape:',df_initial.shape)
#__________________________________________
# info on variable types and filling factor
tab_info=pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values'}))
tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))
tab_info


# ___
# ### 1.1 Keywords

# To develop the recommendation engine, I plan to make an extensive use of the keywords that describe the films. Indeed, a basic assumption is that films described by similar keywords should have similar contents. Hence, I plan to have a close look at the way keywords are defined and as a first step, I quickly characterize what's already in there. To do so, I first list the keywords which are in the dataset:

# In[ ]:


set_keywords = set()
for liste_keywords in df_initial['plot_keywords'].str.split('|').values:
    if isinstance(liste_keywords, float): continue  # only happen if liste_keywords = NaN
    set_keywords = set_keywords.union(liste_keywords)
#_________________________
# remove null chain entry
set_keywords.remove('')


# and then define a function that counts the number of times each of them appear:

# In[ ]:


def count_word(df, ref_col, liste):
    keyword_count = dict()
    for s in liste: keyword_count[s] = 0
    for liste_keywords in df[ref_col].str.split('|'):        
        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue        
        for s in [s for s in liste_keywords if s in liste]: 
            if pd.notnull(s): keyword_count[s] += 1
    #______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count


# Note that this function will be used again in other sections of this notebook, when exploring the content of the *'genres'* variable and subsequently, when cleaning the keywords. Finally, calling this function gives access to a list of keywords which are sorted by decreasing frequency:

# In[ ]:


keyword_occurences, dum = count_word(df_initial, 'plot_keywords', set_keywords)
keyword_occurences[:5]


# At this stage, the list of keywords has been created and we know the number of times each of them appear in the dataset. In fact, this list can be used  to have a feeling of the content of the *most popular movies*. A fancy manner to give that information makes use of the *wordcloud* package. In this kind of representation, all the words are arranged in a figure with sizes that depend on their respective frequencies. Instead of a wordcloud, we can use histograms to give the same information. This allows to have a figure where the keywords are ordered by occurence and most importantly, this gives the number of times they appear, an information that can not be retrieved from the wordcloud representation. In the following figure, I compare both types of representations:

# In[ ]:


#_____________________________________________
# Function that control the color of the words
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# WARNING: the scope of variables is used to get the value of the "tone" variable
# I could not find the way to pass it as a parameter of "random_color_func()"
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)
#_____________________________________________
# UPPER PANEL: WORDCLOUD
fig = plt.figure(1, figsize=(18,13))
ax1 = fig.add_subplot(2,1,1)
#_______________________________________________________
# I define the dictionary used to produce the wordcloud
words = dict()
trunc_occurences = keyword_occurences[0:50]
for s in trunc_occurences:
    words[s[0]] = s[1]
tone = 55.0 # define the color of the words
#________________________________________________________
wordcloud = WordCloud(width=1000,height=300, background_color='black', 
                      max_words=1628,relative_scaling=1,
                      color_func = random_color_func,
                      normalize_plurals=False)
wordcloud.generate_from_frequencies(words)
ax1.imshow(wordcloud, interpolation="bilinear")
ax1.axis('off')
#_____________________________________________
# LOWER PANEL: HISTOGRAMS
ax2 = fig.add_subplot(2,1,2)
y_axis = [i[1] for i in trunc_occurences]
x_axis = [k for k,i in enumerate(trunc_occurences)]
x_label = [i[0] for i in trunc_occurences]
plt.xticks(rotation=85, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(x_axis, x_label)
plt.ylabel("Nb. of occurences", fontsize = 18, labelpad = 10)
ax2.bar(x_axis, y_axis, align = 'center', color='g')
#_______________________
plt.title("Keywords popularity",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 25)
plt.show()


# ___
# ### 1.2 Filling factor: missing values
# 
# The dataset consists in 5043 films or TV series which are described by 28 variables. As in every analysis, at some point, we will have to deal with the missing values and as a first step, I determine the amount of data which is missing in every variable:

# In[ ]:


missing_df = df_initial.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['filling_factor'] = (df_initial.shape[0] 
                                - missing_df['missing_count']) / df_initial.shape[0] * 100
missing_df.sort_values('filling_factor').reset_index(drop = True)


# We can see that most of the variables are well filled since only 2 of them have a filling factor below 93%.

# ___
# ### 1.3 Number of films per year

# The **title_year** variable indicates when films were released. In order to have a global look at the way films are distributed according to this variable, I group the films by decades:

# In[ ]:


df_initial['decade'] = df_initial['title_year'].apply(lambda x:((x-1900)//10)*10)
#__________________________________________________________________
# function that extract statistical parameters from a grouby objet:
def get_stats(gr):
    return {'min':gr.min(),'max':gr.max(),'count': gr.count(),'mean':gr.mean()}
#______________________________________________________________
# Creation of a dataframe with statitical infos on each decade:
test = df_initial['title_year'].groupby(df_initial['decade']).apply(get_stats).unstack()


# and represent the results in a pie chart:

# In[ ]:


sns.set_context("poster", font_scale=0.85)
#_______________________________
# funtion used to set the labels
def label(s):
    val = (1900 + s, s)[s < 100]
    chaine = '' if s < 50 else "{}'s".format(int(val))
    return chaine
#____________________________________
plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(11, 6))
labels = [label(s) for s in  test.index]
sizes  = test['count'].values
explode = [0.2 if sizes[i] < 100 else 0.01 for i in range(11)]
ax.pie(sizes, explode = explode, labels=labels,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow=False, startangle=0)
ax.axis('equal')
ax.set_title('% of films per decade',
             bbox={'facecolor':'k', 'pad':5},color='w', fontsize=16);
df_initial.drop('decade', axis=1, inplace = True)


# ___
# ### 1.4 Genres

# The **genres** variable will surely be important while building the recommendation engines since it describes the content of the film (i.e. Drama, Comedy, Action, ...). To see exactly which genres are the most popular, I use the same approach than for the keywords (hence using similar lines of code), first making a census of the genres:

# In[ ]:


genre_labels = set()
for s in df_initial['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))


# and then counting how many times each of them occur:

# In[ ]:


keyword_occurences, dum = count_word(df_initial, 'genres', genre_labels)
keyword_occurences[:5]


# Finally, the result is shown as a wordcloud:

# In[ ]:


words = dict()
trunc_occurences = keyword_occurences[0:50]
for s in trunc_occurences:
    words[s[0]] = s[1]
tone = 100 # define the color of the words
f, ax = plt.subplots(figsize=(14, 6))
wordcloud = WordCloud(width=550,height=300, background_color='black', 
                      max_words=1628,relative_scaling=0.7,
                      color_func = random_color_func,
                      normalize_plurals=False)
wordcloud.generate_from_frequencies(words)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# ___
# ## 2. Cleaning
# ___

# In[ ]:


df_duplicate_cleaned = df_initial


# ___
# ### 2.1 Cleaning of the keywords
# 
# Keywords will play an important role in the functioning of the engine. Indeed, recommendations will be based on similarity between films and to gauge such similarities, I will look for films described by the same keywords. Hence, the content of the **plot_keywords** variable deserves some attention since it will be extensively used.
# 

# ___
# #### 2.1.1 Grouping by *roots*
# 
# I collect the keywords that appear in the **plot\_keywords** variable. This list is then cleaned using the NLTK package. Finally, I look for the number of occurence of the various keywords.

# In[ ]:


# Collect the keywords
#----------------------
def keywords_inventory(dataframe, colonne = 'plot_keywords'):
    PS = nltk.stem.PorterStemmer()
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys = []
    icount = 0
    for s in dataframe[colonne]:
        if pd.isnull(s): continue
        for t in s.split('|'):
            t = t.lower() ; racine = PS.stem(t)
            if racine in keywords_roots:                
                keywords_roots[racine].add(t)
            else:
                keywords_roots[racine] = {t}
    
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:  
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)            
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
                   
    print("Nb of keywords in variable '{}': {}".format(colonne,len(category_keys)))
    return category_keys, keywords_roots, keywords_select


# In[ ]:


keywords, keywords_roots, keywords_select = keywords_inventory(df_duplicate_cleaned,
                                                               colonne = 'plot_keywords')


# In[ ]:


# Plot of a sample of keywords that appear in close varieties 
#------------------------------------------------------------
icount = 0
for s in keywords_roots.keys():
    if len(keywords_roots[s]) > 1: 
        icount += 1
        if icount < 15: print(icount, keywords_roots[s], len(keywords_roots[s]))


# In[ ]:


# Replacement of the keywords by the main form
#----------------------------------------------
def remplacement_df_keywords(df, dico_remplacement, roots = False):
    df_new = df.copy(deep = True)
    for index, row in df_new.iterrows():
        chaine = row['plot_keywords']
        if pd.isnull(chaine): continue
        nouvelle_liste = []
        for s in chaine.split('|'): 
            clef = PS.stem(s) if roots else s
            if clef in dico_remplacement.keys():
                nouvelle_liste.append(dico_remplacement[clef])
            else:
                nouvelle_liste.append(s)       
        df_new.set_value(index, 'plot_keywords', '|'.join(nouvelle_liste)) 
    return df_new


# In[ ]:


# Replacement of the keywords by the main keyword
#-------------------------------------------------
df_keywords_cleaned = remplacement_df_keywords(df_duplicate_cleaned, keywords_select,
                                               roots = True)


# In[ ]:


# Count of the keywords occurences
#----------------------------------
keywords.remove('')
keyword_occurences, keywords_count = count_word(df_keywords_cleaned,'plot_keywords',keywords)
keyword_occurences[:5]


# ___
# #### 2.2.2 Groups of *synonyms*
# 
# I clean the list of keywords in two steps. As a first step, I suppress the keywords that appear less that 5 times and replace them by a synomym of higher frequency. As a second step, I suppress all the keywords that appear in less than 3 films

# In[ ]:


# get the synomyms of the word 'mot_cle'
#--------------------------------------------------------------
def get_synonymes(mot_cle):
    lemma = set()
    for ss in wordnet.synsets(mot_cle):
        for w in ss.lemma_names():
            #_______________________________
            # We just get the 'nouns':
            index = ss.name().find('.')+1
            if ss.name()[index] == 'n': lemma.add(w.lower().replace('_',' '))
    return lemma   


# In[ ]:


# Exemple of a list of synonyms given by NLTK
#---------------------------------------------------
mot_cle = 'alien'
lemma = get_synonymes(mot_cle)
for s in lemma:
    print(' "{:<30}" in keywords list -> {} {}'.format(s, s in keywords,
                                                keywords_count[s] if s in keywords else 0 ))


# In[ ]:


# check if 'mot' is a key of 'key_count' with a test on the number of occurences   
#----------------------------------------------------------------------------------
def test_keyword(mot, key_count, threshold):
    return (False , True)[key_count.get(mot, 0) >= threshold]


# In[ ]:


keyword_occurences.sort(key = lambda x:x[1], reverse = False)
key_count = dict()
for s in keyword_occurences:
    key_count[s[0]] = s[1]
#__________________________________________________________________________
# Creation of a dictionary to replace keywords by higher frequency keywords
remplacement_mot = dict()
icount = 0
for index, [mot, nb_apparitions] in enumerate(keyword_occurences):
    if nb_apparitions > 5: continue  # only the keywords that appear less than 5 times
    lemma = get_synonymes(mot)
    if len(lemma) == 0: continue     # case of the plurals
    #_________________________________________________________________
    liste_mots = [(s, key_count[s]) for s in lemma 
                  if test_keyword(s, key_count, key_count[mot])]
    liste_mots.sort(key = lambda x:(x[1],x[0]), reverse = True)    
    if len(liste_mots) <= 1: continue       # no replacement
    if mot == liste_mots[0][0]: continue    # replacement by himself
    icount += 1
    if  icount < 8:
        print('{:<12} -> {:<12} (init: {})'.format(mot, liste_mots[0][0], liste_mots))    
    remplacement_mot[mot] = liste_mots[0][0]

print(90*'_'+'\n'+'The replacement concerns {}% of the keywords.'
      .format(round(len(remplacement_mot)/len(keywords)*100,2)))


# In[ ]:


# 2 successive replacements
#---------------------------
print('Keywords that appear both in keys and values:'.upper()+'\n'+45*'-')
icount = 0
for s in remplacement_mot.values():
    if s in remplacement_mot.keys():
        icount += 1
        if icount < 10: print('{:<20} -> {:<20}'.format(s, remplacement_mot[s]))

for key, value in remplacement_mot.items():
    if value in remplacement_mot.keys():
        remplacement_mot[key] = remplacement_mot[value]                    


# In[ ]:


# replacement of keyword varieties by the main keyword
#----------------------------------------------------------
df_keywords_synonyms =             remplacement_df_keywords(df_keywords_cleaned, remplacement_mot, roots = False)   
keywords, keywords_roots, keywords_select =             keywords_inventory(df_keywords_synonyms, colonne = 'plot_keywords')


# In[ ]:


# New count of keyword occurences
#-------------------------------------
keywords.remove('')
new_keyword_occurences, keywords_count = count_word(df_keywords_synonyms,
                                                    'plot_keywords',keywords)
new_keyword_occurences[:5]


# In[ ]:


# deletion of keywords with low frequencies
#-------------------------------------------
def remplacement_df_low_frequency_keywords(df, keyword_occurences):
    df_new = df.copy(deep = True)
    key_count = dict()
    for s in keyword_occurences: 
        key_count[s[0]] = s[1]    
    for index, row in df_new.iterrows():
        chaine = row['plot_keywords']
        if pd.isnull(chaine): continue
        nouvelle_liste = []
        for s in chaine.split('|'): 
            if key_count.get(s, 4) > 3: nouvelle_liste.append(s)
        df_new.set_value(index, 'plot_keywords', '|'.join(nouvelle_liste))
    return df_new


# In[ ]:


# Creation of a dataframe where keywords of low frequencies are suppressed
#-------------------------------------------------------------------------
df_keywords_occurence =     remplacement_df_low_frequency_keywords(df_keywords_synonyms, new_keyword_occurences)
keywords, keywords_roots, keywords_select =     keywords_inventory(df_keywords_occurence, colonne = 'plot_keywords')    


# In[ ]:


# New keywords count
#-------------------
keywords.remove('')
new_keyword_occurences, keywords_count = count_word(df_keywords_occurence,
                                                    'plot_keywords',keywords)
new_keyword_occurences[:5]


# In[ ]:


# Graph of keyword occurences
#----------------------------
font = {'family' : 'fantasy', 'weight' : 'normal', 'size'   : 15}
mpl.rc('font', **font)

keyword_occurences.sort(key = lambda x:x[1], reverse = True)

y_axis = [i[1] for i in keyword_occurences]
x_axis = [k for k,i in enumerate(keyword_occurences)]

new_y_axis = [i[1] for i in new_keyword_occurences]
new_x_axis = [k for k,i in enumerate(new_keyword_occurences)]

f, ax = plt.subplots(figsize=(9, 5))
ax.plot(x_axis, y_axis, 'r-', label='before cleaning')
ax.plot(new_x_axis, new_y_axis, 'b-', label='after cleaning')

# Now add the legend with some customizations.
legend = ax.legend(loc='upper right', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
for label in legend.get_texts():
    label.set_fontsize('medium')
            
plt.ylim((0,25))
plt.axhline(y=3.5, linewidth=2, color = 'k')
plt.xlabel("keywords index", family='fantasy', fontsize = 15)
plt.ylabel("Nb. of occurences", family='fantasy', fontsize = 15)
#plt.suptitle("Nombre d'occurences des mots clés", fontsize = 18, family='fantasy')
plt.text(3500, 4.5, 'threshold for keyword delation', fontsize = 13)
plt.show()


# ___
# ### 2.2 Correlations

# In[ ]:


f, ax = plt.subplots(figsize=(12, 9))
#_____________________________
# calculations of correlations
corrmat = df_keywords_occurence.dropna(how='any').corr()
#________________________________________
k = 17 # number of variables for heatmap
cols = corrmat.nlargest(k, 'num_voted_users')['num_voted_users'].index
cm = np.corrcoef(df_keywords_occurence[cols].dropna(how='any').values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size': 10}, linewidth = 0.1, cmap = 'coolwarm',
                 yticklabels=cols.values, xticklabels=cols.values)
f.text(0.5, 0.93, "Correlation coefficients", ha='center', fontsize = 18, family='fantasy')
plt.show()


# According to the values reported above, I delete a few variables from the dataframe and then re-order the columns.

# In[ ]:


df_var_cleaned = df_keywords_occurence.copy(deep = True)


# ___
# ###  2.3 Missing values
# I examine the number of missing values in each variable and then choose a methodology to complete the dataset.

# In[ ]:


missing_df = df_var_cleaned.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['filling_factor'] = (df_var_cleaned.shape[0] 
                                - missing_df['missing_count']) / df_var_cleaned.shape[0] * 100
missing_df = missing_df.sort_values('filling_factor').reset_index(drop = True)
missing_df


# The content of this table is now represented:

# In[ ]:


y_axis = missing_df['filling_factor'] 
x_label = missing_df['column_name']
x_axis = missing_df.index

fig = plt.figure(figsize=(11, 4))
plt.xticks(rotation=80, fontsize = 14)
plt.yticks(fontsize = 13)

N_thresh = 5
plt.axvline(x=N_thresh-0.5, linewidth=2, color = 'r')
plt.text(N_thresh-4.8, 30, 'filling factor \n < {}%'.format(round(y_axis[N_thresh],1)),
         fontsize = 15, family = 'fantasy', bbox=dict(boxstyle="round",
                   ec=(1.0, 0.5, 0.5),
                   fc=(0.8, 0.5, 0.5)))
N_thresh = 17
plt.axvline(x=N_thresh-0.5, linewidth=2, color = 'g')
plt.text(N_thresh, 30, 'filling factor \n = {}%'.format(round(y_axis[N_thresh],1)),
         fontsize = 15, family = 'fantasy', bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(0.5, 0.8, 0.5)))

plt.xticks(x_axis, x_label,family='fantasy', fontsize = 14 )
plt.ylabel('Filling factor (%)', family='fantasy', fontsize = 16)
plt.bar(x_axis, y_axis);


# ___
# #### 2.3.1 Setting missing title years
# 
#  To infer the title year, I use the list of actors and the director. For each of them, I determine the mean year of activity, using the current dataset. I then average the values obtained to estimate the title year.

# In[ ]:


df_filling = df_var_cleaned.copy(deep=True)
missing_year_info = df_filling[df_filling['title_year'].isnull()][[
            'director_name','actor_1_name', 'actor_2_name', 'actor_3_name']]
missing_year_info[:10]


# In[ ]:


df_filling.iloc[4553]


# In[ ]:


def fill_year(df):
    col = ['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name']
    usual_year = [0 for _ in range(4)]
    var        = [0 for _ in range(4)]
    #_____________________________________________________________
    # I get the mean years of activity for the actors and director
    for i in range(4):
        usual_year[i] = df.groupby(col[i])['title_year'].mean()
    #_____________________________________________
    # I create a dictionnary collectinf this info
    actor_year = dict()
    for i in range(4):
        for s in usual_year[i].index:
            if s in actor_year.keys():
                if pd.notnull(usual_year[i][s]) and pd.notnull(actor_year[s]):
                    actor_year[s] = (actor_year[s] + usual_year[i][s])/2
                elif pd.isnull(actor_year[s]):
                    actor_year[s] = usual_year[i][s]
            else:
                actor_year[s] = usual_year[i][s]
        
    #______________________________________
    # identification of missing title years
    missing_year_info = df[df['title_year'].isnull()]
    #___________________________
    # filling of missing values
    icount_replaced = 0
    for index, row in missing_year_info.iterrows():
        value = [ np.NaN for _ in range(4)]
        icount = 0 ; sum_year = 0
        for i in range(4):            
            var[i] = df.loc[index][col[i]]
            if pd.notnull(var[i]): value[i] = actor_year[var[i]]
            if pd.notnull(value[i]): icount += 1 ; sum_year += actor_year[var[i]]
        if icount != 0: sum_year = sum_year / icount 

        if int(sum_year) > 0:
            icount_replaced += 1
            df.set_value(index, 'title_year', int(sum_year))
            if icount_replaced < 10: 
                print("{:<45} -> {:<20}".format(df.loc[index]['movie_title'],int(sum_year)))
    return 


# In[ ]:


fill_year(df_filling)


# ___
# #### 2.3.2 Extracting keywords from the title
# 
# As previously outlined, keywords will play an important role in the functioning of the engine. Hence, I try to fill missing values in the **plot_keywords** variable using the words of the title. To do so, I create the list of synonyms of all the words contained in the title and I check if any of these synonyms are already in the keyword list. When it is the case, I add this keyword to the entry:

# In[ ]:


icount = 0
for index, row in df_filling[df_filling['plot_keywords'].isnull()].iterrows():
    icount += 1
    liste_mot = row['movie_title'].strip().split()
    new_keyword = []
    for s in liste_mot:
        lemma = get_synonymes(s)
        for t in list(lemma):
            if t in keywords: 
                new_keyword.append(t)                
    if new_keyword and icount < 15: 
        print('{:<50} -> {:<30}'.format(row['movie_title'], str(new_keyword)))
    if new_keyword:
        df_filling.set_value(index, 'plot_keywords', '|'.join(new_keyword)) 


# #### 2.3.3 Imputing from regressions
# 
# In Section 2.4, I had a look at the correlation between variables and found that a few of them showed some degree of correlation, with a Pearson's coefficient > 0.5:

# In[ ]:


cols = corrmat.nlargest(9, 'num_voted_users')['num_voted_users'].index
cm = np.corrcoef(df_keywords_occurence[cols].dropna(how='any').values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# I will use this finding to fill the missing values of the **gross**, **num_critic_for_reviews**, **num\_voted\_users** , and **num_user_for_reviews** variables. To do so, I will make regressions on pairs of correlated variables:

# In[ ]:


sns.set(font_scale=1.25)
cols = ['gross', 'num_voted_users']
sns.pairplot(df_filling.dropna(how='any')[cols],diag_kind='kde', size = 2.5)
plt.show();


# First, I define a function that impute the missing value from a linear fit of the data:

# In[ ]:


def variable_linreg_imputation(df, col_to_predict, ref_col):
    regr = linear_model.LinearRegression()
    test = df[[col_to_predict,ref_col]].dropna(how='any', axis = 0)
    X = np.array(test[ref_col])
    Y = np.array(test[col_to_predict])
    X = X.reshape(len(X),1)
    Y = Y.reshape(len(Y),1)
    regr.fit(X, Y)
    
    test = df[df[col_to_predict].isnull() & df[ref_col].notnull()]
    for index, row in test.iterrows():
        value = float(regr.predict(row[ref_col]))
        df.set_value(index, col_to_predict, value)


# This function takes the dataframe as input, as well as the names of two columns. A linear fit is performed between those two columns which is used to fill the holes in the first column that was given:

# In[ ]:


variable_linreg_imputation(df_filling, 'gross', 'num_voted_users')


# Finally, I examine which amount of data is still missing in the dataframe:

# In[ ]:


df = df_filling.copy(deep = True)
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['filling_factor'] = (df.shape[0] 
                                - missing_df['missing_count']) / df.shape[0] * 100
missing_df = missing_df.sort_values('filling_factor').reset_index(drop = True)
missing_df


# and we can see that in the worst case, the filling factor is around 96% (excluding the **homepage** and **tagline** variables).

# In[ ]:


df = df_filling.copy(deep=True)
df.reset_index(inplace = True, drop = True)


# ___
# ## 3. RECOMMENDATION ENGINE

# ___
# ### 3.1 Basic functioning of the engine 
#  order to build the recommendation engine, I will basically proceed in two steps:
# - 1/ determine $N$ films with a content similar to the entry provided by the user
# - 2/ select the 5 most popular films among these $N$ films
# 
# #### 3.1.1 Similarity
# When builing the engine, the first step thus consists in defining a criteria that would tell us how close two films are. To do so, I start from the description of the film that was selected by the user: from it, I get the director name, the names of the actors and a few keywords. I then build a matrix where each row corresponds to a film of the database and where the columns correspond to the previous quantities (director + actors + keywords) plus the *k* genres that were described in section 1.4:
# 

# |  movie title |director   |actor 1   |actor 2   |actor 3   | keyword 1  | keyword 2   | genre 1 | genre 2 | ... | genre k |
# |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
# |Film 1   | $a_{11}$  |  $a_{12}$ |   |   |  ... |   |   |   |   | $a_{1q}$  |
# |...   |   |   |   |   | ...  |   |   |   |   |   |
# |Film i   |  $a_{i1}$ | $a_{i2}$ |   |   | $a_{ij}$  |   |   |   |   |  $a_{iq}$ |
# |...   |   |   |   |   | ...  |   |   |   |   |   |
# | Film p   |$a_{p1}$   | $a_{p2}$  |   |   | ...  |   |   |   |   | $a_{pq}$  |

# In this matrix, the $a_{ij}$ coefficients take either the value 0 or 1 depending on the correspondance between the significance of column $j$ and the content of film $i$. For exemple, if "keyword 1" is in film $i$, we will have $a_{ij}$ = 1 and 0 otherwise. Once this matrix has been defined, we determine the distance between two films according to:
# 
# \begin{eqnarray}
# d_{m, n} = \sqrt{  \sum_{i = 1}^{N} \left( a_{m,i}  - a_{n,i} \right)^2  } 
# \end{eqnarray}
# 
# At this stage, we just have to select the N films which are the closest from the entry selected by the user.
# 
# #### 3.1.2 Popularity
# 
# According to similarities between entries, we get a list of $N$ films. At this stage, I select 5 films from this list and, to do so, I give a score for every entry. I decide de compute the score according to 3 criteria:
# - the IMDB score
# - the number of votes the entry received
# - the year of release
# 
# The two first criteria will be a direct measure of the popularity of the various entries in IMDB. For the third criterium, I introduce the release year since the database spans films from the early $XX^{th}$ century up to now. I assume that people's favorite films will be most of the time from the same epoch.
# 
# Then, I calculate the score according to the formula:

# 
# \begin{eqnarray}
# \mathrm{score} = IMDB^2 \times \phi_{\sigma_1, c_1} \times  \phi_{\sigma_2, c_2}
# \end{eqnarray}
# 
# where $\phi$ is a gaussian function:
# 
# \begin{eqnarray}
# \phi_{\sigma, c}(x) \propto \mathrm{exp}\left(-\frac{(x-c)^2}{2 \, \sigma^2}\right)
# \end{eqnarray}

# For votes, I get the maximum number of votes among the $N$ films and I set $\sigma_1 = c_1 = m$. For years, I put $\sigma_1 = 20$ and I center the gaussian on the title year of the film selected by the user. With the gaussians, I put more weight to the entries with a large number of votes and to the films whose release year is close to the title selected by the user.

# ___
# ### 3.2 Definition of the engine functions

# In[ ]:


gaussian_filter = lambda x,y,sigma: math.exp(-(x-y)**2/(2*sigma**2))


#  **Function collecting some variables content**: the *entry\_variables()* function returns the values taken by the variables *'director\_name', 'actor\_N\_name'* (N $\in$ [1:3]) and *'plot\_keywords'* for the film selected by the user.

# In[ ]:


def entry_variables(df, id_entry): 
    col_labels = []    
    if pd.notnull(df['director_name'].iloc[id_entry]):
        for s in df['director_name'].iloc[id_entry].split('|'):
            col_labels.append(s)
            
    for i in range(3):
        column = 'actor_NUM_name'.replace('NUM', str(i+1))
        if pd.notnull(df[column].iloc[id_entry]):
            for s in df[column].iloc[id_entry].split('|'):
                col_labels.append(s)
                
    if pd.notnull(df['plot_keywords'].iloc[id_entry]):
        for s in df['plot_keywords'].iloc[id_entry].split('|'):
            col_labels.append(s)
    return col_labels


#  **Function adding variables to the dataframe**: the function *add\_variables()* add a list of variables to the dataframe given in input and initialize these variables at 0 or 1 depending on the correspondance with the description of the films and the content of the REF_VAR variable given in input.

# In[ ]:


def add_variables(df, REF_VAR):    
    for s in REF_VAR: df[s] = pd.Series([0 for _ in range(len(df))])
    colonnes = ['genres', 'actor_1_name', 'actor_2_name',
                'actor_3_name', 'director_name', 'plot_keywords']
    for categorie in colonnes:
        for index, row in df.iterrows():
            if pd.isnull(row[categorie]): continue
            for s in row[categorie].split('|'):
                if s in REF_VAR: df.set_value(index, s, 1)            
    return df


#  **Function creating a list of films**: the *recommand()* function create a list of N (= 31) films similar to the film selected by the user.

# In[ ]:


def recommand(df, id_entry):    
    df_copy = df.copy(deep = True)    
    liste_genres = set()
    for s in df['genres'].str.split('|').values:
        liste_genres = liste_genres.union(set(s))    
    #_____________________________________________________
    # Create additional variables to check the similarity
    variables = entry_variables(df_copy, id_entry)
    variables += list(liste_genres)
    df_new = add_variables(df_copy, variables)
    #____________________________________________________________________________________
    # determination of the closest neighbors: the distance is calculated / new variables
    X = df_new.as_matrix(variables)
    nbrs = NearestNeighbors(n_neighbors=31, algorithm='auto', metric='euclidean').fit(X)

    distances, indices = nbrs.kneighbors(X)    
    xtest = df_new.iloc[id_entry].as_matrix(variables)
    xtest = xtest.reshape(1, -1)

    distances, indices = nbrs.kneighbors(xtest)

    return indices[0][:]
    


#  **Function extracting some parameters from a list of films**: the *create\_film\_selection()* function extracts some variables of the dataframe given in input and returns this list for a selection of N films. This list is ordered according to criteria established in the *critere\_selection()* function.

# In[ ]:


def extract_parameters(df, liste_films):     
    parametres_films = ['_' for _ in range(31)]
    i = 0
    max_users = -1
    for index in liste_films:
        parametres_films[i] = list(df.iloc[index][['movie_title', 'title_year',
                                        'imdb_score', 'num_user_for_reviews', 
                                        'num_voted_users']])
        parametres_films[i].append(index)
        max_users = max(max_users, parametres_films[i][4] )
        i += 1
        
    title_main = parametres_films[0][0]
    annee_ref  = parametres_films[0][1]
    parametres_films.sort(key = lambda x:critere_selection(title_main, max_users,
                                    annee_ref, x[0], x[1], x[2], x[4]), reverse = True)

    return parametres_films 


#  **Function comparing 2 film titles**: the sequel *sequel()* function compares the 2 titles passed in input and defines if these titles are similar or not.

# In[ ]:


def sequel(titre_1, titre_2):    
    if fuzz.ratio(titre_1, titre_2) > 50 or fuzz.token_set_ratio(titre_1, titre_2) > 50:
        return True
    else:
        return False


#  **Function giving marks to films**: the *critere\_selection()* function gives a mark to a film depending on its IMDB score,  the title year and the number of users who have voted for this film.

# In[ ]:


def critere_selection(title_main, max_users, annee_ref, titre, annee, imdb_score, votes):    
    if pd.notnull(annee_ref):
        facteur_1 = gaussian_filter(annee_ref, annee, 20)
    else:
        facteur_1 = 1        

    sigma = max_users * 1.0

    if pd.notnull(votes):
        facteur_2 = gaussian_filter(votes, max_users, sigma)
    else:
        facteur_2 = 0
        
    if sequel(title_main, titre):
        note = 0
    else:
        note = imdb_score**2 * facteur_1 * facteur_2
    
    return note


#  **Function adding films**: the *add\_to\_selection()* function complete the *film\_selection* list which contains 5 films that will be recommended to the user. The films are selected from the *parametres\_films* list and are taken into account only if the title is different enough from other film titles. 

# In[ ]:


def add_to_selection(film_selection, parametres_films):    
    film_list = film_selection[:]
    icount = len(film_list)    
    for i in range(31):
        already_in_list = False
        for s in film_selection:
            if s[0] == parametres_films[i][0]: already_in_list = True
            if sequel(parametres_films[i][0], s[0]): already_in_list = True            
        if already_in_list: continue
        icount += 1
        if icount <= 5:
            film_list.append(parametres_films[i])
    return film_list


#  **Function filtering sequels**: the *remove\_sequels()* function remove sequels from the list if more that two films from a serie are present. The older one is kept.

# In[ ]:


def remove_sequels(film_selection):    
    removed_from_selection = []
    for i, film_1 in enumerate(film_selection):
        for j, film_2 in enumerate(film_selection):
            if j <= i: continue 
            if sequel(film_1[0], film_2[0]): 
                last_film = film_2[0] if film_1[1] < film_2[1] else film_1[0]
                removed_from_selection.append(last_film)

    film_list = [film for film in film_selection if film[0] not in removed_from_selection]

    return film_list   


# **Main function**: create a list of 5 films that will be recommended to the user.

# In[ ]:


def find_similarities(df, id_entry, del_sequels = True, verbose = False):    
    if verbose: 
        print(90*'_' + '\n' + "QUERY: films similar to id={} -> '{}'".format(id_entry,
                                df.iloc[id_entry]['movie_title']))
    #____________________________________
    liste_films = recommand(df, id_entry)
    #__________________________________
    # Create a list of 31 films
    parametres_films = extract_parameters(df, liste_films)
    #_______________________________________
    # Select 5 films from this list
    film_selection = []
    film_selection = add_to_selection(film_selection, parametres_films)
    #__________________________________
    # delation of the sequels
    if del_sequels: film_selection = remove_sequels(film_selection)
    #______________________________________________
    # add new films to complete the list
    film_selection = add_to_selection(film_selection, parametres_films)
    #_____________________________________________
    selection_titres = []
    for i,s in enumerate(film_selection):
        selection_titres.append([s[0].replace(u'\xa0', u''), s[5]])
        if verbose: print("nº{:<2}     -> {:<30}".format(i+1, s[0]))

    return selection_titres


# ___
# ### 3.3 Making meaningful recommendations

# While building the recommendation engine, we are quickly faced to a big issue: the existence of sequels make that some recommendations may seem quite dumb ... As an exemple, somebody who enjoyed *"Pirates of the Caribbean: Dead Man's Chest"* would probably not like to be adviced to watch this: 

# In[ ]:


dum = find_similarities(df, 12, del_sequels = False, verbose = True)


# Unfortunately, if we build the engine according to the functionalities described in Section 3.1, this is what we are told !!
# 
# The origin of that issue is quite easily understood: many blockbusters have sequels that share the same director, actors and keywords ... Most of the time, the fact that sequels exist mean that it was a "fair" box-office success, which is a synonym of a good IMDB score. Usually, there's an inheritence of success among sequels which entail that according to the way the current engine is built, it is quite probable that if the engine matches one film of a serie, it will end recommending various of them. In the previous exemple, we see that the engine recommends the three films of the *"Lord of the ring"* trilogy, as well as *"Thor"* and *"Thor: the dark world"*. Well, I would personnaly not make that kind of recommendations to a friend ... 
# 
# Hence, I tried to find a way to prevent that kind of behaviour and I concluded that the quickest way to do it would be to work on the film's titles. To do so, I used the **fuzzywuzzy** package to build the *remove_sequels()* function. This function defines the degree of similarity of two film titles and if too close, the most recent film is removed from the list of recommendations. Using this function on the previous exemple, we end with the following recommendations:

# In[ ]:


dum = find_similarities(df, 12, del_sequels = True, verbose = True)


# which seems far more reasonable !! 
# 
# But, well, nothing is perfect. This way of discarding some recommendations assumes that there is a a continuity in the names of films pertaining to a serie. This is however not always the case:

# In[ ]:


dum = find_similarities(df, 2, del_sequels = True, verbose = True)


# Here, the user selected a film from the James Bond serie, *'Spectre'*, and the engine recommends him two other James Bond films, *'Casino Royale'* and *'Skyfall'*. Well, I guess that people who enjoyed *'Spectre'* will know that there is not a unique film featuring James Bond, and the current recommendation thus looks a bit irrelevant ...

# ___
# ### 3.4 Exemple of recommendation: test-case

# In[ ]:


selection = dict()
for i in range(0, 20, 3):
    selection[i] = find_similarities(df, i, del_sequels = True, verbose = True)


# ___
# ## 4. Conclusion: possible improvements and points to adress
# 
# Finally a few things were not considered when building the engine and they should deserve some attention:
# - the language of the film was not checked: in fact, this could be important to get sure that the films recommended are in the same language than the one choosen by the user
# - another point concerns the replacement of the keywords by more frequent synonyms. In some cases, it was shown that the synonyms selected had a different meaning that the original word. Definitely, the whole process might deserve more attention and be improved.
# - another improvement could be to create a list of connections between actors to see which are the actors that use to play in similar movies (I started an analysis in that direction in [another notebook](https://www.kaggle.com/fabiendaniel/categorizing-actors)). Hence, rather than only looking at the actors who are in the film selected by the user, we could enlarge this list by a few more people. Something similar could be done also with the directors.
# - extend the detections of sequels to films that don't share similar titles (e.g. James Bond serie)

# *Thanks a lot for reaching this point of the notebook !! <br>
# If you see anything wrong or something that could be improved, please, tell me !!* <br>
# 
# **If you found some interest in this notebook, thanks for upvoting !!**
