
# coding: utf-8

# # 1. Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# # 2. Import data

# In[ ]:


data = pd.read_csv("../input/spam.csv",encoding='latin-1')


# In[ ]:


data.head()


# Let's drop the unwanted columns, and rename the column name appropriately.

# In[ ]:


#Drop column and name change
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})


# In[ ]:


data.tail()


# In[ ]:


#Count observations in each label
data.label.value_counts()


# In[ ]:


# convert label to a numerical variable
data['label_num'] = data.label.map({'ham':0, 'spam':1})


# In[ ]:


data.head()


# # 3. Train Test Split
# Before performing text transformation, let us do train test split. Infact, we can perform k-Fold cross validation. However, due to simplicity, I am doing train test split.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label"], test_size = 0.2, random_state = 10)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # 4.Text Transformation
# Various text transformation techniques such as stop word removal, lowering the texts, tfidf transformations, prunning, stemming can be performed using sklearn.feature_extraction libraries. Then, the data can be convereted into bag-of-words. <br> <br>
# For this problem, Let us see how our model performs without removing stop words.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


vect = CountVectorizer()


# Note : We can also perform tfidf transformation.

# In[ ]:


vect.fit(X_train)


# vect.fit function learns the vocabulary. We can get all the feature names from vect.get_feature_names( ). <br> <br> Let us print first and last twenty features

# In[ ]:


print(vect.get_feature_names()[0:20])
print(vect.get_feature_names()[-20:])


# In[ ]:


X_train_df = vect.transform(X_train)


# Now, let's transform the Test data.

# In[ ]:


X_test_df = vect.transform(X_test)


# In[ ]:


type(X_test_df)


# # 5. Visualisations 

# In[ ]:


ham_words = ''
spam_words = ''
spam = data[data.label_num == 1]
ham = data[data.label_num ==0]


# In[ ]:


import nltk
from nltk.corpus import stopwords


# In[ ]:


for val in spam.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    #tokens = [word for word in tokens if word not in stopwords.words('english')]
    for words in tokens:
        spam_words = spam_words + words + ' '
        
for val in ham.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        ham_words = ham_words + words + ' '


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


# Generate a word cloud image
spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)
ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)


# In[ ]:


#Spam Word cloud
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[ ]:


#Ham word cloud
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# # 6. Machine Learning models:

# ### 6.1 Multinomial Naive Bayes
# Generally, Naive Bayes works well on text data. Multinomail Naive bayes is best suited for classification with discrete features. 

# In[ ]:


prediction = dict()
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_df,y_train)


# In[ ]:


prediction["Multinomial"] = model.predict(X_test_df)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


accuracy_score(y_test,prediction["Multinomial"])


# ### 6.2 Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_df,y_train)


# In[ ]:


prediction["Logistic"] = model.predict(X_test_df)


# In[ ]:


accuracy_score(y_test,prediction["Logistic"])


# ### 6.3 $k$-NN classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_df,y_train)


# In[ ]:


prediction["knn"] = model.predict(X_test_df)


# In[ ]:


accuracy_score(y_test,prediction["knn"])


# ### 6.4 Ensemble classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train_df,y_train)


# In[ ]:


prediction["random_forest"] = model.predict(X_test_df)


# In[ ]:


accuracy_score(y_test,prediction["random_forest"])


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train_df,y_train)


# In[ ]:


prediction["adaboost"] = model.predict(X_test_df)


# In[ ]:


accuracy_score(y_test,prediction["adaboost"])


# # 7. Parameter Tuning using GridSearchCV

# Based, on the above four ML models, Naive Bayes has given the best accuracy. However, Let's try to tune the parameters of $k$-NN using GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


k_range = np.arange(1,30)


# In[ ]:


k_range


# In[ ]:


param_grid = dict(n_neighbors=k_range)
print(param_grid)


# In[ ]:


model = KNeighborsClassifier()
grid = GridSearchCV(model,param_grid)
grid.fit(X_train_df,y_train)


# In[ ]:


grid.best_estimator_


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_score_


# In[ ]:


grid.grid_scores_


# # 8. Model Evaluation

# In[ ]:


print(classification_report(y_test, prediction['Multinomial'], target_names = ["Ham", "Spam"]))


# In[ ]:


conf_mat = confusion_matrix(y_test, prediction['Multinomial'])
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]


# In[ ]:


sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# # 9. Future works

# In[ ]:


print(conf_mat)


# By seeing the above confusion matrix, it is clear that 5 Ham are mis classified as Spam, and 8 Spam are misclassified as Ham. Let'see what are those misclassified text messages. Looking those messages may help us to come up with more advanced feature engineering.

# In[ ]:


pd.set_option('display.max_colwidth', -1)


# I increased the pandas dataframe width to display the misclassified texts in full width. 

# ### 9.1 Misclassified as Spam

# In[ ]:


X_test[y_test < prediction["Multinomial"] ]


# ### 9.2 Misclassfied as Ham

# In[ ]:


X_test[y_test > prediction["Multinomial"] ]


# It seems length of the spam text is much higher than the ham. Maybe we can include length as a feature.  In addition to unigram, we can also try bigram features. 
