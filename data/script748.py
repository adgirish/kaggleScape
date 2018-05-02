
# coding: utf-8

# <h1><center>What India reads about</center></h1>
# <img src="http://blog.iefa.org/wp-content/uploads/2014/10/indian-woman-thinking-477531487.jpg">
# This is a great data set to practice some text mining and visualization skills. Although we all know what newspapers usually talk about, but its a lot of fun to go back to a dataset and uncover some interesting insights. For example if you are surprised to find out that BJP got mentioned almost 1.5 times more than Congress in the past 16, years of headlines in a leading national daily, then you should read on.....
#    
#    ## TOC
# 1. Persistent themes
#    *    Unigram, bigrams and trigrams
#    *    Common n-grams across years
#    *   Are we a country who loves to read about crime?
#    * Are suicides on a rise in India?
# 2. Indian Political Scene: BJP vs Congress
#    - Congress: Good, bad and ugly
#    - BJP: Good, bad and ugly
#    - NaMo vs RaGa
# 3. Why does india love Shah Rukh?
#    - Word frequency and co-occurence analysis
# 4. General trivia:
#    - Startups, when did the country catch the train? 
#    - Analytics, does mainstream media care?
#    - Kingfisher: How the decline was chronicled?
#   

# In[1]:


import numpy as np 
import pandas as pd 
import spacy
from wordcloud import WordCloud
data=pd.read_csv("../input/india-headlines-news-dataset/india-news-headlines.csv")
data=data[['publish_date','headline_text']].drop_duplicates()
data['publish_date']=pd.to_datetime(data['publish_date'],format="%Y%M%d")
data['year']=data['publish_date'].dt.year
nlp=spacy.load("en_core_web_lg")


#  <h1><center> Persistent Themes </center></h1>
#  To get a gist of what are the themes, that are being talked about, I followed a general approach of doing frequency counts, of unigrams, bigrams and trigrams on the whole dataset as well as for each year. Find below the code I used to create some simple visuals

# In[2]:


#The following code takes a really long time, so have created pickled versions of these objects and reading them loc
'''### Get imp words by year
import sklearn.feature_extraction.text as text
def get_imp(bow,mf,ngram):
    tfidf=text.CountVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')
    matrix=tfidf.fit_transform(bow)
    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(100)
### Global trends
bow=data['headline_text'].tolist()
total_data=get_imp(bow,mf=5000,ngram=1)
total_data_bigram=get_imp(bow=bow,mf=5000,ngram=2)
total_data_trigram=get_imp(bow=bow,mf=5000,ngram=3)
### Yearly trends
imp_terms_unigram={}
for y in data['year'].unique():
    bow=data[data['year']==y]['headline_text'].tolist()
    imp_terms_unigram[y]=get_imp(bow,mf=5000,ngram=1)
imp_terms_bigram={}
for y in data['year'].unique():
    bow=data[data['year']==y]['headline_text'].tolist()
    imp_terms_bigram[y]=get_imp(bow,mf=5000,ngram=2)
imp_terms_trigram={}
for y in data['year'].unique():
    bow=data[data['year']==y]['headline_text'].tolist()
    imp_terms_trigram[y]=get_imp(bow,mf=5000,ngram=3)
'''
import pickle
total_data=pd.read_pickle('../input/total-datapkl/total_data.pkl')
total_data_bigram=pd.read_pickle("../input/total-datapkl/total_data_bigram.pkl")
total_data_trigram=pd.read_pickle("../input/total-data-trigrampkl/total_data_trigram.pkl")
f=open("../input/total-data-trigrampkl/imp_terms_unigram.pkl","rb")
d=f.read()
imp_terms_unigram=pickle.loads(d)
f.close()
f=open("../input/total-data-trigrampkl/imp_terms_biigram.pkl","rb")
d=f.read()
imp_terms_bigram=pickle.loads(d)
f.close()
f=open("../input/total-data-trigrampkl/imp_terms_triigram.pkl","rb")
d=f.read()
imp_terms_trigram=pickle.loads(d)
f.close()
### Common unigrams across all the years
common_unigram={}
for y in np.arange(2001,2017,1):
    if y==2001:       
        common_unigram[y]=set(imp_terms_unigram[y].index).intersection(set(imp_terms_unigram[y+1].index))
    else:
        common_unigram[y]=common_unigram[y-1].intersection(set(imp_terms_unigram[y+1].index))
### Common bigrams across all the years
common_bigram={}
for y in np.arange(2001,2017,1):
    if y==2001:
         common_bigram[y]=set(imp_terms_bigram[y].index).intersection(set(imp_terms_bigram[y+1].index))
    else:
        common_bigram[y]=common_bigram[y-1].intersection(set(imp_terms_bigram[y+1].index))
### Common trigrams, 1 year window
common_trigram_1yr={}
for y in np.arange(2001,2017,1):
    common_trigram_1yr[str(y)+"-"+str(y+1)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index))
### Commin trigrams, 2 year window
common_trigram_2yr={}
for y in np.arange(2001,2015,3):
    if y==2001:
        common_trigram_2yr[str(y)+"-"+str(y+1)+"-"+str(y+2)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index)).intersection(set(imp_terms_trigram[y+2].index))
    else:
        common_trigram_2yr[str(y)+"-"+str(y+1)+"-"+str(y+2)]=set(imp_terms_trigram[y].index).intersection(set(imp_terms_trigram[y+1].index)).intersection(set(imp_terms_trigram[y+2].index))


# <h2>Count of top 20 unigrams, bigrams and trigrams</h2>

# In[3]:


import matplotlib.pyplot as plt
plt.subplot(1,3,1)
total_data.head(20).plot(kind="bar",figsize=(25,10),colormap='Set2')
plt.title("Unigrams",fontsize=30)
plt.yticks([])
plt.xticks(size=20)
plt.subplot(1,3,2)
total_data_bigram.head(20).plot(kind="bar",figsize=(25,10),colormap='Set2')
plt.title("Bigrams",fontsize=30)
plt.yticks([])
plt.xticks(size=20)
plt.subplot(1,3,3)
total_data_trigram.head(20).plot(kind="bar",figsize=(25,10),colormap='Set2')
plt.title("Trigrams",fontsize=30)
plt.yticks([])
plt.xticks(size=20)


# Some observations that one can make here, it seems bollywood, particularly, Shah Rukh Khan is very famous (look at the trigrams). Also, you can notice that Narendra Modi, has had a fair share of headlines mentioning him in the past 16 years. Also, if you look at some of the bigrams and trigrams, you will find mention of **year old**, **year old girl**, **year old woman**. We will look at these tokens in more detail later. A final comment, if you look at unigrams, you will notice that **BJP** gets mentioned quite often. We will look at this also in detail, later

# <h2>Bigrams and Trigrams across years</h2>
# To get a sense of trend across years, I also plotted bigrams and trigrams across the years

# <h2> Top 5 Bigrams across years</h2>

# In[4]:


for i in range(1,18,1):
    plt.subplot(9,2,i)
    imp_terms_bigram[2000+i].head(5).plot(kind="barh",figsize=(20,25),colormap='Set2')
    plt.title(2000+i,fontsize=20)
    plt.xticks([])
    plt.yticks(size=20,rotation=5)


# <h2> Top 5 Trigrams across years</h2>

# In[5]:


for i in range(1,18,1):
    plt.subplot(9,2,i)
    imp_terms_trigram[2000+i].head(5).plot(kind="barh",figsize=(20,25),colormap="Set2")
    plt.title(2000+i,fontsize=20)
    plt.xticks([])
    plt.yticks(size=15,rotation=5)


# If you look at the trigrams and bigrams closely, you will realize, that reporting of crime, sports (cricket in particular) and Shah Rukh Khan is persistent!!!
# <img src="http://www.indiantelevision.com/sites/drupal7.indiantelevision.co.in/files/styles/smartcrop_800x800/public/images/tv-images/2017/05/31/SRK-KKR_0.jpg?itok=rIWW5rMD">
# 

# In[6]:


## Count of common tokens across the years
count_common_bi={}
for year in range(2001,2017,1):
    count_common_bi[year]=pd.Series()
    for word in common_bigram[year]:
        if year==2001:
            count_common_bi[year][word]=imp_terms_bigram[year][word]+imp_terms_bigram[year+1][word]
        else:
            count_common_bi[year][word]=count_common_bi[year-1][word]+imp_terms_bigram[year+1][word]


# <h2>Which bigrams have been conistently reported over years?</h2>
# The previous couple of plots, capture, what was reported the most overall in the past 16 years and what happened in each of these 16 years. The next question to ask is which stories were reported consistently every year in the past 16 years. The way I tackled this was by finding common bigrams for year 2001 to 2002, then for 2003 and 2001 and 2002 combined and so on. This was the result:
# <h3>Top 10 bigrams common across years<h3>

# In[7]:


for i in range(1,17,1):
    plt.subplot(9,2,i)
    count_common_bi[2000+i].sort_values(ascending=False).head(10).plot(kind="barh",figsize=(20,35),colormap="Set2")
    if (2000+i)==2001:
        plt.title(str(2000+i)+"-"+str(2000+i+1),fontsize=30)
    else:
        plt.title("upto-"+str(2000+i+1),fontsize=30)
    plt.xticks([])
    plt.yticks(size=20,rotation=5)


# <h2>Do we love to read about crime a lot?</h2>
# While looking at the plot created above, one thing that strikes you is that crime reporting is very persistent. All the tokens in the above figure are actually, telling you the common bigrams from one year to another. One thing that strikes you is the fact that token **year old** and **commits suicide** are very prominent across years.
# 
# <h3>Let's first fgure out the story behind the token <i>year old</i></h3>
# In order to figure out the **context** around *year old*,
# -  I found out which headlines contained this token
# - Extracted the noun and verbs that occur with this token
# 

# In[8]:


## Story of 'year old'
index=data['headline_text'].str.match(r'(?=.*\byear\b)(?=.*\bold\b).*$')
texts=data['headline_text'].loc[index].tolist()
noun=[]
verb=[]
for doc in nlp.pipe(texts,n_threads=16,batch_size=10000):
    try:
        for c in doc:
            if c.pos_=="NOUN":
                noun.append(c.text)
            elif c.pos_=="VERB":
                verb.append(c.text)            
    except:
        noun.append("")
        verb.append("")


# In[9]:


plt.subplot(1,2,1)
pd.Series(noun).value_counts().head(10).plot(kind="bar",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns in context of 'Year Old'",fontsize=30)
plt.xticks(size=20,rotation=80)
plt.yticks([])
plt.subplot(1,2,2)
pd.Series(verb).value_counts().head(10).plot(kind="bar",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs in context of 'Year Old'",fontsize=30)
plt.xticks(size=20,rotation=80)
plt.yticks([])


# You can clearly see that the context around **year old** is crime/violence (rape,kills,arrested) against women, mostly. Just to confim, our results, that we got via an NLP parse, let's look at some of the news items, where **year old** token actually occured
# 

# In[10]:


data['headline_text'].loc[index].tolist()[0:20]


# <h2>Are sucides on a rise in India?</h2>
# Another, common bigram pattern that we had observed was *commits suicide*. For the records, the **suicide rate in India is lower than the world average**, source https://en.wikipedia.org/wiki/Suicide_in_India. However, it is surprising to note that instances of suicides, constantly make headlines in Indian daily. **Famer suicdes** are a big *political issue* in the country, is that the reason why instances of suicide get reported so much?
# 
# To get some sort of directional answer here, I again, reverted to an NLP based parse, extracting the nouns from the headlines which contained the token "commits suicide".
# 
# 

# In[11]:


index_s=data['headline_text'].str.match(r'(?=.*\bcommits\b)(?=.*\bsuicide\b).*$')
text_s=data['headline_text'].loc[index].tolist()
noun_s=[]
for doc in nlp.pipe(text_s,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_s.append(c.text)
    except:
        for c in doc:
            noun_s.append("") 


# In[12]:


pd.Series(noun_s).value_counts().head(20).plot("bar",figsize=(15,5),colormap="Set2")
plt.xticks(fontsize=20)
plt.yticks([])
plt.ylabel("Frequency")
plt.title("Frequency of Nouns in the context of 'Commits Suicide'",fontsize=30)


# Its very surprising to not see "Famer" in the context of suicides. As a matter of fact, farm suicides are the top most reasons of sucides in India (https://data.gov.in/catalog/stateut-wise-professional-profile-suicide-victim). Farmer suicides were 37080 in 2014 compared to 24204 student suicides. Let's use regular expressions to find out, the trend in suicide reporting, we will:
# - First find out instances where "commits suicide" pattern occurs
# - Then figure out out of these, in how many instaces student and farmer ocucr respectively

# In[13]:


index_s=data['headline_text'].str.match(r'(?=.*\bcommits\b)(?=.*\bsuicide\b).*$',case=False)
index_farmer=data.loc[index_s]['headline_text'].str.match(r'farmer',case=False)
index_stu=data.loc[index_s]['headline_text'].str.match(r'student',case=False)


# In[14]:


print("Approximately {} percent of suicides reported were student related".format(round(np.sum(index_stu)/np.sum(index_s),2)*100))


# In[15]:


print("Approximately {} percent of suicides reported were farmer related".format(round(np.sum(index_farmer)/np.sum(index_s),2)*100))


# Clearly, instances of farmer suicides are actually more in number than the ones reported by Times of India. Let's see, what are the keywords mentioned, when a "farmer" is mentioned in a headline?

# In[16]:


ind_farmer=data['headline_text'].str.match(r'farmer|farmers',case=False)


# In[17]:


text_f=data.loc[ind_farmer]['headline_text'].tolist()
noun_f=[]
verb_f=[]
for doc in nlp.pipe(text_f,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_f.append(c.text)
            elif c.pos_=="VERB":
                verb_f.append(c.text)
    except:
        for c in doc:
            noun_f.append("") 
            verb_f.append("")


# In[18]:


plt.subplot(1,2,1)
pd.Series(noun_f).value_counts()[2:].head(10).plot(kind="bar",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns in the context of 'Farmer(s)'",fontsize=25)
plt.xticks(size=20,rotation=80)
plt.yticks([])
plt.subplot(1,2,2)
pd.Series(verb_f).value_counts().head(10).plot(kind="bar",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs in the context of 'Farmer(s)'",fontsize=25)
plt.xticks(size=20,rotation=80)
plt.yticks([])


# As can be seen the headlines about farmers were mostly about them "protesting", "demanding" or "oppising" for "land", "water" or suicides. 

# <h1><center>Indian Political Scene: BJP vs Congress</center></h1>
# <img src="https://akm-img-a-in.tosshub.com/indiatoday/images/story/201703/congress-bjp_647_033117014707_0.jpg">
# <h2>Relative Frequency</h2>
# I calculated the frequency of how many times BJP and Congress occured in the corpus , here were the results

# In[19]:


index_bjp=data['headline_text'].str.match(r"bjp.*$",case=False)
index_con=data['headline_text'].str.match(r"congress.*$",case=False)
print("BJP was mentioned {} times".format(np.sum(index_bjp)))
print("Congress was mentioned {} times".format(np.sum(index_con)))
print("BJP was mentioned {} times more than Congress".format(np.round(np.sum(index_bjp)/np.sum(index_con),2)))


# <h2>What were the headlines about BJP?</h2>

# In[20]:


import textblob
data_bjp=data.loc[index_bjp].copy()
data_bjp['polarity']=data_bjp['headline_text'].map(lambda x: textblob.TextBlob(x).sentiment.polarity)
pos=" ".join(data_bjp.query("polarity>0")['headline_text'].tolist())
neg=" ".join(data_bjp.query("polarity<0")['headline_text'].tolist())
text=" ".join(data_bjp['headline_text'].tolist())


# In[21]:


from wordcloud import WordCloud,STOPWORDS
import PIL
bjp_mask=np.array(PIL.Image.open("../input/image-masks/bjp.png"))
wc = WordCloud(max_words=500, mask=bjp_mask,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(text)
plt.figure( figsize=(30,15) )
plt.imshow(wc)
plt.yticks([])
plt.xticks([])
plt.axis("off")


# <h3>Let's look at some syntactic and symantic elements in these headlines</h3>

# In[22]:


## top trigrams
from sklearn.feature_extraction import text
def get_imp(bow,mf,ngram):
    tfidf=text.CountVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')
    matrix=tfidf.fit_transform(bow)
    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(100)
bow=data_bjp['headline_text'].tolist()
bjp_trigrams=get_imp(bow,mf=5000,ngram=3)


# In[23]:


text_bjp=data_bjp['headline_text'].tolist()
noun_bjp=[]
verb_bjp=[]
for doc in nlp.pipe(text_bjp,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_bjp.append(c.text)
            elif c.pos_=="VERB":
                verb_bjp.append(c.text)
    except:
        for c in doc:
            noun_bjp.append("") 
            verb_bjp.append("")


# In[24]:


plt.subplot(1,3,1)
bjp_trigrams.head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Trigrams (BJP)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,2)
pd.Series(noun_bjp).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns (BJP)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,3)
pd.Series(verb_bjp).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs (BJP)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])


# <h2>What did positive and negative headlines about BJP contain?</h2>

# In[25]:


thumbs_up=np.array(PIL.Image.open("../input/image-masks/thumbsup.jpg"))
wc = WordCloud(max_words=500, mask=thumbs_up,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(pos)
thumbs_dn=np.array(PIL.Image.open("../input/image-masks/thumbsdown.jpg"))
wc1=WordCloud(max_words=500, mask=thumbs_dn,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(neg)
fig=plt.figure(figsize=(30,15))
ax=fig.add_subplot(1,2,1)
ax.imshow(wc)
ax.axis('off')
ax.set_title("Positive Headlines",fontdict={'fontsize':20})
ax=fig.add_subplot(1,2,2)
ax.imshow(wc1)
ax.axis('off')
ax.set_title("Negative Headlines",fontdict={'fontsize':20})


# In[26]:


bow=data_bjp.query("polarity>0")['headline_text'].tolist()
bjp_trigrams_pos=get_imp(bow,mf=5000,ngram=3)
bow=data_bjp.query("polarity<0")['headline_text'].tolist()
bjp_trigrams_neg=get_imp(bow,mf=5000,ngram=3)


# In[27]:


text_bjp_pos=data_bjp.query("polarity>0")['headline_text'].tolist()
noun_bjp_pos=[]
verb_bjp_pos=[]
for doc in nlp.pipe(text_bjp_pos,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_bjp_pos.append(c.text)
            elif c.pos_=="VERB":
                verb_bjp_pos.append(c.text)
    except:
        for c in doc:
            noun_bjp_pos.append("") 
            verb_bjp_pos.append("")


# In[28]:


text_bjp_neg=data_bjp.query("polarity<0")['headline_text'].tolist()
noun_bjp_neg=[]
verb_bjp_neg=[]
for doc in nlp.pipe(text_bjp_neg,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_bjp_neg.append(c.text)
            elif c.pos_=="VERB":
                verb_bjp_neg.append(c.text)
    except:
        for c in doc:
            noun_bjp_neg.append("") 
            verb_bjp_neg.append("")


# <h3><h3>Let's look at the most common verbs, nouns and trigrams used when positive news about BJP was reported</h3>

# In[29]:


plt.subplot(1,3,1)
bjp_trigrams_pos.head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Trigrams (BJP+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,2)
pd.Series(noun_bjp_pos).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns (BJP+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,3)
pd.Series(verb_bjp_pos).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs (BJP+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])


# <h3><h3>Let's look at the most common verbs, nouns and trigrams used when negative news about BJP was reported</h3>

# In[30]:


plt.subplot(1,3,1)
bjp_trigrams_neg.head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Trigrams (BJP-)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,2)
pd.Series(noun_bjp_neg).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns (BJP-)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,3)
pd.Series(verb_bjp_neg).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs (BJP-)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])


# <h2>What were the headlines about Congress?</h2>

# In[31]:


data_con=data.loc[index_con].copy()
data_con['polarity']=data_con['headline_text'].map(lambda x: textblob.TextBlob(x).sentiment.polarity)
pos=" ".join(data_con.query("polarity>0")['headline_text'].tolist())
neg=" ".join(data_con.query("polarity<0")['headline_text'].tolist())
text=" ".join(data_con['headline_text'].tolist())


# In[32]:


con_mask=np.array(PIL.Image.open('../input/image-masks/congress.png'))
wc = WordCloud(max_words=500, mask=con_mask,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(text)
plt.figure( figsize=(30,15))
plt.imshow(wc)
plt.yticks([])
plt.xticks([])
plt.axis("off")


# In[33]:


from sklearn.feature_extraction import text
bow=data_con['headline_text'].tolist()
con_trigrams=get_imp(bow,mf=5000,ngram=3)


# In[34]:


text_con=data_con['headline_text'].tolist()
noun_con=[]
verb_con=[]
for doc in nlp.pipe(text_con,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_con.append(c.text)
            elif c.pos_=="VERB":
                verb_con.append(c.text)
    except:
        for c in doc:
            noun_con.append("") 
            verb_con.append("")


# In[35]:


plt.subplot(1,3,1)
con_trigrams.head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Trigrams (Congress)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,2)
pd.Series(noun_con).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns (Congress)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,3)
pd.Series(verb_con).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs (Congress)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])


# <h2>What did positive and negative headlines about Congess contain?</h2>

# In[36]:


wc = WordCloud(max_words=500, mask=thumbs_up,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(pos)
wc1=WordCloud(max_words=500, mask=thumbs_dn,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(neg)
fig=plt.figure(figsize=(30,15))
ax=fig.add_subplot(1,2,1)
ax.imshow(wc)
ax.axis('off')
ax.set_title("Positive Headlines",fontdict={'fontsize':20})
ax=fig.add_subplot(1,2,2)
ax.imshow(wc1)
ax.axis('off')
ax.set_title("Negative Headlines",fontdict={'fontsize':20})


# In[37]:


bow=data_con.query("polarity>0")['headline_text'].tolist()
con_trigrams_pos=get_imp(bow,mf=5000,ngram=3)
bow=data_con.query("polarity<0")['headline_text'].tolist()
con_trigrams_neg=get_imp(bow,mf=5000,ngram=3)


# In[38]:


text_con_pos=data_con.query("polarity>0")['headline_text'].tolist()
noun_con_pos=[]
verb_con_pos=[]
for doc in nlp.pipe(text_con_pos,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_con_pos.append(c.text)
            elif c.pos_=="VERB":
                verb_con_pos.append(c.text)
    except:
        for c in doc:
            noun_con_pos.append("") 
            verb_con_pos.append("")


# In[39]:


text_con_neg=data_con.query("polarity<0")['headline_text'].tolist()
noun_con_neg=[]
verb_con_neg=[]
for doc in nlp.pipe(text_con_neg,n_threads=16,batch_size=1000):
    try:
        for c in doc:
            if c.pos_=='NOUN':
                noun_con_neg.append(c.text)
            elif c.pos_=="VERB":
                verb_con_neg.append(c.text)
    except:
        for c in doc:
            noun_con_neg.append("") 
            verb_con_neg.append("")


# <h3>Let's look at the most common verbs, nouns and trigrams used when positive news about Congress was reported</h3>

# In[40]:


plt.subplot(1,3,1)
con_trigrams_pos.head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Trigrams (Congress+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,2)
pd.Series(noun_con_pos).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns (Congress+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,3)
pd.Series(verb_con_pos).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs (Congress+)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])


# <h3>Let's look at the most common verbs, nouns and trigrams used when negative news about Congress was reported</h3>

# In[41]:


plt.subplot(1,3,1)
con_trigrams_neg.head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Trigrams (Congress-)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,2)
pd.Series(noun_con_neg).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Nouns (Congress-)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])
plt.subplot(1,3,3)
pd.Series(verb_con_neg).value_counts().head(10).plot(kind="barh",figsize=(20,5),colormap="Set2")
plt.title("Top 10 Verbs (Congress-)",fontsize=20)
plt.yticks(size=20)
plt.xticks([])


# <h2>Modi vs Rahul Gandhi</h2>
# <h3>Relative Frequency</h3>
# Let's look at the number of times Modi and Rahul Gandhi occurs

# In[42]:


index_nm=data['headline_text'].str.match("narendra modi",case=False)
index_rahul=data['headline_text'].str.match("rahul gandhi",case=False)
print("Modi has been mentioned {} times".format(np.sum(index_nm)))
print("Rahul Gandhi has been mentioned {} times".format(np.sum(index_rahul)))


# In[43]:


nm=pd.DataFrame(data['year'].loc[index_nm].value_counts())
r=pd.DataFrame(data['year'].loc[index_rahul].value_counts())
n_r=pd.concat([nm,r],axis=1)
n_r.columns=["Modi","Rahul"]


# In[44]:


n_r.plot(figsize=(10,10))
plt.title("Mentions of Modi and Rahul over time",fontsize=20)


# In[ ]:


'''
text=" ".join(data.loc[index_rahul]['headline_text'].tolist())
rahul=np.array(PIL.Image.open("../input/image-masks/raga.jpg"))
wc = WordCloud(max_words=5000, mask=rahul,width=5000,height=2500,background_color="white",stopwords=STOPWORDS,max_font_size=100).generate(text)
plt.figure( figsize=(30,15))
plt.imshow(wc)
plt.yticks([])
plt.xticks([])
plt.title("WordCloud of Headlines about Rahul",fontsize=15)
plt.axis("off")
''''''


# <h1>What  headlines Shah Rukh appeared in  ?</h1>
# We can try to answer this question by, carefully looking at the headlines where Shah Rukh gets mentioned

# In[45]:


index_shah=data['headline_text'].str.match(r'(?=.*\bshah\b)(?=.*\brukh\b).*$',case=False)
data_shah=data.loc[index_shah].copy()
data_shah['polarity']=data_shah['headline_text'].map(lambda x: textblob.TextBlob(x).sentiment.polarity)


# In[46]:


pos=data_shah.query("polarity>0")['headline_text']
neg=data_shah.query("polarity<0")['headline_text']
print("The number of positve headlines were {} times the negative headlines".format(round(len(pos)/len(neg),2)))


# In[47]:


plt.figure(figsize=(8,8))
plt.bar(["Positive","Negative"],[len(pos),len(neg)])
plt.title("Frequency of Positive and Negative News about Shah Rukh",fontsize=20)


# In[48]:


bow=data_shah['headline_text'].str.replace(r'shah|rukh|khan',"",case=False).tolist()
shah_uni=get_imp(bow,mf=5000,ngram=1)
shah_bi=get_imp(bow,mf=5000,ngram=2)
shah_tri=get_imp(bow,mf=5000,ngram=3)


# In[49]:


shah_bi.head(10).plot(kind="barh",figsize=(8,8),colormap="Set2")
plt.title("Most Frequent Bigrams in the Context of Shah Rukh",fontsize=15)


# It seems like Shah Rukh is mentioned in the context of his movies, his co-stars and the performance of his movies at box office. As a tribute to the great Star, let's put-together a wordcloud.

# In[51]:


shah_text=" ".join(bow)
con_mask=np.array(PIL.Image.open('../input/image-masks/shah.jpg'))
wc = WordCloud(max_words=500, mask=con_mask,width=5000,height=2500,background_color="white",stopwords=STOPWORDS).generate(shah_text)
plt.figure( figsize=(30,15))
plt.imshow(wc)
plt.axis("off")
plt.yticks([])
plt.xticks([])
plt.savefig('./shahrukh.png', dpi=50)
plt.show()


# This is still a work in progress.  Will continue adding more to this kernel in coming days. Please comment below to let me know, your views and how can I improve this further, Please upvote the kernel if you enjoyed reading this.
