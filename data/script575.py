
# coding: utf-8

# # Whose questions you can answer and which questions you might be interested in? 

# ## Outline

# The content of this kernel will cover two parts.
# 
# Part 1: Finding the users who always ask the similar questions with the specific user. 
# Part 2: Finding the users who always provide similar answers with the specific user.
# 
# Both parts will be finished with a two-step process: NLP and KNN model fitting. While the first part will be  analyzed with the text of questions while the second part will use the text of answers to solve and analyze.

# ## Fire up

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cross_validation import train_test_split
from wordcloud import WordCloud,STOPWORDS

Questions=pd.read_csv('../input/Questions.csv',encoding = 'iso-8859-1')
Answers=pd.read_csv('../input/Answers.csv',encoding = 'iso-8859-1')


# In order to shrink the volume of the data-set, I just use the data of users who both post questions on stack-overflow and answer other's questions.   

# In[ ]:


User_id_inQ= Questions['OwnerUserId'].unique()
User_id_inA= Answers['OwnerUserId'].unique()


# In[ ]:


All_id=set(User_id_inQ).intersection(User_id_inA)


# In[ ]:


print('So we have '+str(len(All_id))+' users that post both questions and answers on StackOverFlow')


# ## Natural Language Processing and Dataframe Transformation

# Before moving to the next step, I first select the top 10000 users in terms of the total number of questions and answers posted on the website. If I perform the analysis on all data, the kernel will kill itself.... So let us finish the selection first.

# In[ ]:


users=pd.DataFrame({'idUser':list(All_id)})
users['Quantity']=users['idUser'].apply(lambda x: len(Questions[Questions['OwnerUserId']==x]['Body'])+len(Answers[Answers['OwnerUserId']==x]['Body']))


# In[ ]:


users_final=users.sort(['Quantity'],ascending=0).reset_index(drop=True)


# In[ ]:


users_final=users_final.iloc[0:10000,]


# In[ ]:


All_id=list(users_final['idUser'])


# Firstly, create a function that can clean the body of questions and answers. Only the main body of questions will be used.

# In[ ]:


from html.parser import HTMLParser
import re
import nltk
from nltk.corpus import stopwords


# In[ ]:


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def clean(text):
    removed_html=strip_tags(text)
    letters_only = re.sub("[^a-zA-Z]", " ", removed_html) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return(meaningful_words)


# The clean function works. Great. Next, I will shrink the dataset and only keep the data of users who both ask and answer questions.

# In[ ]:


Q_data=Questions[['OwnerUserId','Body']]
A_data=Answers[['OwnerUserId','Body']]
Question=Q_data[Q_data['OwnerUserId'].isin(All_id)]
Answer=A_data[A_data['OwnerUserId'].isin(All_id)]


# In[ ]:


Question['Non_html_body']=Question['Body'].apply(lambda x:strip_tags(str(x)))
Answer['Non_html_body']=Answer['Body'].apply(lambda x:strip_tags(str(x)))


# Then, we create a new data frame with each row containing all questions and answers the specific user has posted on Stack-overflow.

# In[ ]:


Question['Q_words']=Question['Body'].apply(lambda x:clean(str(x)))
Answer['A_words']=Answer['Body'].apply(lambda x:clean(str(x)))


# In[ ]:


User_id=All_id
Question_corpus=[]
Answer_corpus=[]
for id in All_id:
    Q_frame=Question[Question['OwnerUserId']==id].reset_index(drop=True)
    A_frame=Answer[Answer['OwnerUserId']==id].reset_index(drop=True)
    for i in range(len(Q_frame['OwnerUserId'])):
        if i==0:
            tmp=Q_frame['Q_words'][i]
        else:
            tmp=tmp+Q_frame['Q_words'][i]
    Question_corpus.append(tmp)
    
    for j in range(len(A_frame['OwnerUserId'])):
        if j==0:
            tmp2=A_frame['A_words'][j]
        else:
            tmp2=tmp2+A_frame['A_words'][j]
    Answer_corpus.append(tmp2)


# In[ ]:


User_id[0]


# In[ ]:


Question_C=[]
Answer_C=[]
for i in range(10000):
    tmp1=" ".join(Question_corpus[i])
    tmp2=" ".join(Answer_corpus[i])
    Question_C.append(tmp1)
    Answer_C.append(tmp2)


# In[ ]:


Final_frame=pd.DataFrame({'User_id':User_id,'Question':Question_C,'Answer':Answer_C})


# Finally, create tfidf matrix for both questions and answers

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
Q_features=tfidf.fit_transform(Question_C)
A_features=tfidf.fit_transform(Answer_C)


# ## KNN Models

# In[ ]:


from sklearn.neighbors import NearestNeighbors
knn_1=NearestNeighbors(n_neighbors=30,algorithm='brute',metric='cosine')
knn_2=NearestNeighbors(n_neighbors=30,algorithm='brute',metric='cosine')
Question_fit=knn_1.fit(Q_features)
Answer_fit=knn_2.fit(A_features)


# ### Answer question 1: Whose questions will also get your attention?

# I tend to use the KNN model that focuses on question text to answer this question. Namely, the model will calculate the nearest neighbor of the user based on the question text and see which user also ask the same question. The process is enclosed in a single function.

# In[ ]:


def Question_f(user_id):
    ###Find the corresponding features##
    index=Final_frame[Final_frame['User_id']==user_id].index.tolist()
    ### Word Cloud##
    Question_word=Question_C[index[0]]
    wordcloud = WordCloud(background_color='black',
                      width=3000,
                      height=2500
                     ).generate(Question_word)
    Neighbors = Question_fit.kneighbors(Q_features[index[0]])[1].tolist()[0][1:]
    Users=np.array(User_id)[Neighbors].tolist()
    print('= ='*50)
    print("User's question is always featured by words below")
    plt.figure(1,figsize=(8,8))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    print('= ='*50)
    print('The top five users that always ask similar questions are: '+str(Users[0])+' '+str(Users[1])+' '+str(Users[2])+' '+str(Users[3])+' '+str(Users[4]))
    print('= ='*50)
    User_question=Question[Question['OwnerUserId']==user_id].reset_index(drop=True)['Non_html_body'][0]
    print('Example: Question of user:\n'+User_question)
    print('= ='*50)
    Question2=Question[Question['OwnerUserId']==Users[0]].reset_index(drop=True)['Non_html_body'][0]
    print('Example: Question from a similar user:\n'+Question2)


# **An example**

# In[ ]:


Question_f(100297)


# ### Answer question 2: Which users share similar expertise with you?

# In[ ]:


def Answer_f(user_id):
    ###Find the corresponding features##
    index=Final_frame[Final_frame['User_id']==user_id].index.tolist()
    ### Word Cloud##
    Answer_word=Answer_C[index[0]]
    wordcloud = WordCloud(background_color='blue',
                      width=3000,
                      height=2500
                     ).generate(Answer_word)
    Neighbors = Answer_fit.kneighbors(A_features[index[0]])[1].tolist()[0][1:]
    Users=np.array(User_id)[Neighbors].tolist()
    print('= ='*50)
    print("User's answer is always featured by words below")
    plt.figure(1,figsize=(8,8))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    print('= ='*50)
    print('The top five users that always provide similar answers are: '+str(Users[0])+' '+str(Users[1])+' '+str(Users[2])+' '+str(Users[3])+' '+str(Users[4]))
    print('= ='*50)
    User_answer=Answer[Answer['OwnerUserId']==user_id].reset_index(drop=True)['Non_html_body'][0]
    print('Example: Answer from user:\n'+User_answer)
    print('= ='*50)
    Answer2=Answer[Answer['OwnerUserId']==Users[0]].reset_index(drop=True)['Non_html_body'][0]
    print('Example: Answer from a similar user:\n'+Answer2)


# **An example**

# In[ ]:


Answer_f(100297)


# **Both the questions and answers are quite technical.... So I am really not sure whether the implementation of model is good or bad.**

# ## Conclusion

# 1. The script takes a really long time to run. I need more elegant coding skill as well as the NLP technique.
# 
# 2. I am really not sure the result of clustering. The context is quite technical.
# 
# 3. For the part of grouping users who provide similar answer, it is also scientific to use the question the user answered. But it needs more codes for data frame transformation.
