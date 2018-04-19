
# coding: utf-8

# In[ ]:


"""
this is a good competition to predict the project is approved or rejected
the question is what is the criteria that make the project approve or rejected
in the data set we have some text columns then the project approves column 
Suppose that the column project_resource_summary will make the teacher approve or reject the project 
so I suppose that using naive bayes classifier is good fot that 
but the problem is that sample submission file the score in decimal numbers 
MultinomialNB have method predict_proba that get the output n probability score 
the output in training set is decimal number 
I hope this model solve this competition
"""


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#from wordcloud import WordCloud,STOPWORDS
from numpy import nan
from bs4 import BeautifulSoup    
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt


# In[ ]:


train_data = pd.read_csv("../input/train1csv/train.csv")
test_data = pd.read_csv("../input/train1csv/test.csv")


# In[ ]:


#methode for cleaning the data


# In[ ]:


def review_to_words(raw_review): 
    review =raw_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))
    


# In[ ]:


#clean the two columns in train and test data set


# In[ ]:


corpus1= []
for i in range(0, 182080):
    corpus1.append(review_to_words(train_data['project_resource_summary'][i]))


# In[ ]:


corpus2= []
for i in range(0, 78035):
    corpus2.append(review_to_words(test_data['project_resource_summary'][i]))


# In[ ]:


#making new columns due to the new cleaning data set then put it in data set and delete the old columns 


# In[ ]:


train_data['new_project_resource_summary']=corpus1
test_data['new_project_resource_summary']=corpus2
train_data.drop(['project_resource_summary'],axis=1,inplace=True)
test_data.drop(['project_resource_summary'],axis=1,inplace=True)


# In[ ]:


X = train_data.new_project_resource_summary
y = train_data.project_is_approved

print(X.shape)
print(y.shape)


# In[ ]:


#cross validation for training  data set  


# In[ ]:


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


#CountVectorizer for most 2000 word in the data set


# In[ ]:


vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 2000)


# In[ ]:



vectorizer.fit(X_train)

X_train_d = vectorizer.transform(X_train)
X_train_d

X_test_d = vectorizer.transform(X_test)
X_test_d


# In[ ]:


#using the naive_bayes with  MultinomialNB


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


classifier = MultinomialNB()

classifier.fit(X_train_d,y_train)

y_pred_class = classifier.predict(X_test_d)


# In[ ]:


#the accuracy_score of the Prediction 


# In[ ]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

print(y_test.value_counts())


# In[ ]:


#the confusion_matrix of the Prediction


# In[ ]:


metrics.confusion_matrix(y_test, y_pred_class)


# In[ ]:


#making Prediction with decimal score  as submission file 


# In[ ]:


y_pred_prob = classifier.predict_proba(X_test_d)[:, 1]
y_pred_prob


# In[ ]:


#Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
#source from scikit-learn   http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html


# In[ ]:


metrics.roc_auc_score(y_test, y_pred_prob)


# In[ ]:


#Repeat the previous steps in the test data set


# In[ ]:


x=test_data.new_project_resource_summary
print(x.shape)
X_test_real = vectorizer.transform(x)
X_test_real


# In[ ]:


#the out put arra here is for 0 or 1 but in Decimal  out but


# In[ ]:


real_y1=classifier.predict_proba(X_test_real)
real_y1


# In[ ]:


#convert the out put array to data frame


# In[ ]:


import pandas
d=real_y1
d.tolist()
df = pandas.DataFrame(d)
df = pandas.DataFrame(d)
my_columns = ["0", "1"]
df.columns = my_columns
df.head()


# In[ ]:


#Choose only the column that Contain  the number nearest to 1


# In[ ]:


real_y=classifier.predict_proba(X_test_real)[:, 1]
real_y


# In[ ]:


#convert the out put array to data frame


# In[ ]:


c=real_y
c.tolist()
df_final = pandas.DataFrame(c)
my_columns = [ "project_is_approved"]
df_final.columns = my_columns
df_final.head()


# In[ ]:


#convert the out put array to data frame and contact the data frame with id column in test data frame 
#to get the out put like submission file


# In[ ]:



id_test=test_data.id
type(id_test)

a=id_test
a.tolist()

df_final_id = pandas.DataFrame(a)
my_columns = [ "id"]
df_final_id.columns = my_columns

submission = pd.concat([df_final_id, df_final],axis=1)
submission.head(10)


# In[ ]:



submission.to_csv('submission.csv', sep='\t',encoding='utf8')


# In[ ]:


#thank for all have a nice day

