
# coding: utf-8

# Goal of this notebook to test several classifiers on the data set with different features 

# And beforehand i want to thank Jose Portilla for his magnificent "Python for Data Science and Machine Learning" course on Udemy , which helped me to dive into ML =)

# ### Let's begin

# First of all neccesary imports

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's read the data from csv file

# In[ ]:


sms = pd.read_csv('../input/spam.csv', encoding='latin-1')
sms.head()


# Now drop "unnamed" columns and rename v1 and v2 to "label" and "message"

# In[ ]:


sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
sms = sms.rename(columns = {'v1':'label','v2':'message'})


# Let's look into our data

# In[ ]:


sms.groupby('label').describe()


# Intresting that "Sorry, I'll call later" appears only 30 times here =)

# Now let's create new feature "message length" and plot it to see if it's of any interest

# In[ ]:


sms['length'] = sms['message'].apply(len)
sms.head()


# In[ ]:


mpl.rcParams['patch.force_edgecolor'] = True
plt.style.use('seaborn-bright')
sms.hist(column='length', by='label', bins=50,figsize=(11,5))


# Looks like the lengthy is the message, more likely it is a spam. Let's not forget this

# ### Text processing and vectorizing our meddages

# Let's create new data frame. We'll need a copy later on

# In[ ]:


text_feat = sms['message'].copy()


# Now define our tex precessing function. It will remove any punctuation and stopwords aswell.

# In[ ]:


def text_process(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    
    return " ".join(text)


# In[ ]:


text_feat = text_feat.apply(text_process)


# In[ ]:


vectorizer = TfidfVectorizer("english")


# In[ ]:


features = vectorizer.fit_transform(text_feat)


# ###  Classifiers and predictions

# First of all let's split our features to test and train set

# In[ ]:


features_train, features_test, labels_train, labels_test = train_test_split(features, sms['label'], test_size=0.3, random_state=111)


# Now let's import bunch of classifiers, initialize them and make a dictionary to itereate through

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier(n_neighbors=49)
mnb = MultinomialNB(alpha=0.2)
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=31, random_state=111)
abc = AdaBoostClassifier(n_estimators=62, random_state=111)
bc = BaggingClassifier(n_estimators=9, random_state=111)
etc = ExtraTreesClassifier(n_estimators=9, random_state=111)


# Parametres are based on notebook:
# [Spam detection Classifiers hyperparameter tuning][1]
# 
# 
#   [1]: https://www.kaggle.com/muzzzdy/d/uciml/sms-spam-collection-dataset/spam-detection-classifiers-hyperparameter-tuning/

# In[ ]:


clfs = {'SVC' : svc,'KN' : knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc, 'AdaBoost': abc, 'BgC': bc, 'ETC': etc}


# Let's make functions to fit our classifiers and make predictions

# In[ ]:


def train_classifier(clf, feature_train, labels_train):    
    clf.fit(feature_train, labels_train)


# In[ ]:


def predict_labels(clf, features):
    return (clf.predict(features))


# Now iterate through classifiers and save the results

# In[ ]:


pred_scores = []
for k,v in clfs.items():
    train_classifier(v, features_train, labels_train)
    pred = predict_labels(v,features_test)
    pred_scores.append((k, [accuracy_score(labels_test,pred)]))


# In[ ]:


df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])
df


# In[ ]:


df.plot(kind='bar', ylim=(0.9,1.0), figsize=(11,6), align='center', colormap="Accent")
plt.xticks(np.arange(9), df.index)
plt.ylabel('Accuracy Score')
plt.title('Distribution by Classifier')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Looks like ensemble classifiers are not doing as good as expected.

# ### Stemmer

# It is said that stemming short messages does no goot or even harm predictions. Let's try this out.

# Define our stemmer function

# In[ ]:


def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words


# Stem, split, fit - repeat... Predict!

# In[ ]:


text_feat = text_feat.apply(stemmer)


# In[ ]:


features = vectorizer.fit_transform(text_feat)


# In[ ]:


features_train, features_test, labels_train, labels_test = train_test_split(features, sms['label'], test_size=0.3, random_state=111)


# In[ ]:


pred_scores = []
for k,v in clfs.items():
    train_classifier(v, features_train, labels_train)
    pred = predict_labels(v,features_test)
    pred_scores.append((k, [accuracy_score(labels_test,pred)]))


# In[ ]:


df2 = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score2'])
df = pd.concat([df,df2],axis=1)
df


# In[ ]:


df.plot(kind='bar', ylim=(0.85,1.0), figsize=(11,6), align='center', colormap="Accent")
plt.xticks(np.arange(9), df.index)
plt.ylabel('Accuracy Score')
plt.title('Distribution by Classifier')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Looks like mostly the same . Ensemble classifiers doing a little bit better, NB still got the lead.

# ### What have we forgotten? Message length!

# Let's append our message length feature to the matrix we fit into our classifiers

# In[ ]:


lf = sms['length'].as_matrix()
newfeat = np.hstack((features.todense(),lf[:, None]))


# In[ ]:


features_train, features_test, labels_train, labels_test = train_test_split(newfeat, sms['label'], test_size=0.3, random_state=111)


# In[ ]:


pred_scores = []
for k,v in clfs.items():
    train_classifier(v, features_train, labels_train)
    pred = predict_labels(v,features_test)
    pred_scores.append((k, [accuracy_score(labels_test,pred)]))


# In[ ]:


df3 = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score3'])
df = pd.concat([df,df3],axis=1)
df


# In[ ]:


df.plot(kind='bar', ylim=(0.85,1.0), figsize=(11,6), align='center', colormap="Accent")
plt.xticks(np.arange(9), df.index)
plt.ylabel('Accuracy Score')
plt.title('Distribution by Classifier')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# This time everyone are doing a little bit worse, except for LinearRegression and RandomForest. But the winner is still MultinominalNaiveBayes.

# ### Voting classifier

# We are using ensemble algorithms here, but what about ensemble of ensembles? Will it beat NB?

# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


eclf = VotingClassifier(estimators=[('BgC', bc), ('ETC', etc), ('RF', rfc), ('Ada', abc)], voting='soft')


# In[ ]:


eclf.fit(features_train,labels_train)


# In[ ]:


pred = eclf.predict(features_test)


# In[ ]:


print(accuracy_score(labels_test,pred))


# Better but nope.

# ### Final verdict - well tuned NaiveBayes is your friend in spam detection.
