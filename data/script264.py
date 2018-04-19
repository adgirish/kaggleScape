
# coding: utf-8

# Upvote ! if you find it useful 
# 
# 
# **TUTORIAL for BEGINNERS ON HOW TO SOLVE ALMOST ANY  TEXT CLASSIFICATION PROBLEM USING NAIVE BAYES**
# **So This is a very basic TUTORIAL, For newcomers. I will be using simple naive bayes along with Count Vectorization to solve this problem**
# 
# So Lets Start.
# 
# Lets Understand the hot terms first:
# 
# **Naive Bayes Classifier**
# 
# **It is a simple classifier based on bayes theorem with full independence between different features.**  I know it passed right from above the head.
# 
# Lets find a simple explanation:
# * So we have a dataset with [text, label] e.g.["This process, however, afforded me no means of..."	,EAP] where EAP represents the author so we have a text and we have to find out which author out of three wrote that text.
# * Now First of all computer does not understand alphabets or words, so we convert the words into numbers so make computer understand it
# * For that we simply assign a number for each unique word, e.g.  "to be or not to be" will be assigned numbers like to:1, be:2, or:3, not:4.
# * Now we can send input to the computer like [1,2,3,4,1,2] 
# * But naive bayes does not want this it just wants the count of each words.
# * Naive bayes like the count of "to" or "1" in document1 or row1 and so on
# * So we give naive bayes input like this [0,1]:2   [0,2]:2   [0,3]:1 and so on. We are giving it the count of each word and each document so zero represents first document.
# * So we are going to have a big two dimensional matrix with lots of columns and rows equal to the number of documents given
# 
# Now lets look at naive bayes again, Naive bayes considers each words as independant and does not value the position of each word: (the postion of word have no role in classifying or training in naive bayes)
# 
# Secondly it is based on probability and bayes theorem, I am not going to explain it here as it will take more than necessary space so Chapter 13 of **An Introduction to information retreival by Christopher D manning , Prabhakar Raghavan
# Hinrich Schütze ** Section 13.2 will explain us the naive bayes in detail
# 
# https://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
# 
# 
# Now lets come back to the dataset
# 
# 
# So lets see what we got, we got a training data set with author labels and testing datasets without labels. In this we just take the training data set, Split it by using 70% for training the Naive Bayes classifier and other 30% for testing it.
# Here is the basic code and easiest one.
# 
# 
# So first of all we import libraries then we load the csv file into a pandas dataframe, or in layman terms, into a python variable so that we can process it.
# 
# Then we see a sample of the data using training_data.head().
# 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#READING INPUT
training_data = pd.read_csv("../input/train.csv")
testing_data=pd.read_csv("../input/test.csv")
training_data.head()


# Now by just looking at the top 4 entries, it must be clear that does id have any role in determining who the author is?
# No. Id is just used as a identifier for text NOT for author. So one thing is clear over here, that id is useless and would not help in any way to our model to learn.
# 
# So now we simple omit the id.
# Similarly to avoid clutterness we map "EAP" to 0 "HPL" to 1 and "MWS" to 2 as it will be more convenient for our classifier. 
# In other words we are just telling our computer that if classifier predicts 0 for the text then it means that it is preicting "EAP", if 1 then it means that it is predicting "HPL", if 2 then it means that it is predicting 2.
# 
# Next we take all the rows under the column named "text" and put it in X ( a variable in python)
# 
# Similarly we take all rows under the column named "author_num" and put it in y (a variable in python)

# In[ ]:


training_data['author_num'] = training_data.author.map({'EAP':0, 'HPL':1, 'MWS':2})
X = training_data['text']
y = training_data['author_num']
print (X.head())
print (y.head())



# Now we got the data, we got the text and the corresponing label, Now we need to split the data into training set and testing set.
# Testing set is the one which we will never show to the computer, we will take it and keep it in a safe and only use it to test the model.
# 
# So we are going to split it into 70% for training and 30% for testing.

# In[ ]:


per=int(float(0.7)* len(X))
X_train=X[:per]
X_test=X[per:]
y_train=y[:per]
y_test=y[per:]


# Here are some libraries we are going to need

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# **Now comes the most important part.
# Vectorization**
# 
# We see that computers get crazy with text, It only understands numbers, but we have got to classify text. Now what do we do?
# We do tokenization and vectorization to save the count of each word. Confused right.
# keep on reading believe me you will get it in the end.
# Now let say you have got a text like "My name is computer My life". so what exactly does the vectorizer do.??
# 
# Lets see,
# first it breaks it into tokens something like this. 
# 
# My
# 
# name
# 
# is
# 
# computer
# 
# My
# 
# life
# 
# Pretty easy right
# Now it first creates a vocabulary from it
# e.g
# My:0
# computer:1
# is:2
# life:3
# name:4
# 
# Still very easy
# 
# 
# 
# Now we see that we create a sparse matrix out of it,  
# 
# 
# 
# Which have 1 row and 5 columns are there were 5 unique tokens in our text
# 
# 
# 
# we see that matrix[0,0] is 2 which specifies that 0 item in dictionary which is My came 2 times and so on.
# 

# In[ ]:


#toy example
text=["My name is computer My life"]
toy = CountVectorizer(lowercase=False, token_pattern=r'\w+|\,')
toy.fit_transform(text)
print (toy.vocabulary_)
matrix=toy.transform(text)
print (matrix[0,0])
print (matrix[0,1])
print (matrix[0,2])
print (matrix[0,3])
print (matrix[0,4])



# Exacly like the above toy example we vectorize the text

# In[ ]:


vect = CountVectorizer(lowercase=False, token_pattern=r'\w+|\,')
X_cv=vect.fit_transform(X)
X_train_cv = vect.transform(X_train)
X_test_cv = vect.transform(X_test)
print (X_train_cv.shape)


# Here comes the final step
# We give the data to the clf.fit for training and test it for score.
# We have not used log score over here for simplicity.

# In[ ]:


clf=MultinomialNB()
clf.fit(X_train_cv, y_train)
clf.score(X_test_cv, y_test)


# Now we saw the accuracy on our training data we made for ourself, Now we will let the kaggle test our accuracy. So first of all we will update our vocabulary and transform raw TEST data from kaggle into vectorized form

# In[ ]:


X_test=vect.transform(testing_data["text"])




# Now we have successfully vectorized the data given by kaggle Now we fit the whole training data without any split into our Naive Bayes Model
# Next we give it the testing vectorized data to predict the probabilities

# In[ ]:


clf=MultinomialNB()
clf.fit(X_cv, y)
predicted_result=clf.predict_proba(X_test)
predicted_result.shape


# We see that we got a result with 8392 rows presenting each text and 3 columns each column representing probability of each author.

# In[ ]:


#NOW WE CREATE A RESULT DATA FRAME AND ADD THE COLUMNS NECESSARY TO SUBMIT HERE
result=pd.DataFrame()
result["id"]=testing_data["id"]
result["EAP"]=predicted_result[:,0]
result["HPL"]=predicted_result[:,1]
result["MWS"]=predicted_result[:,2]
result.head()


# FINALLY WE SUBMIT THE RESULT TO KAGGLE FOR EVALUATION

# In[ ]:


result.to_csv("TO_SUBMIT.csv", index=False)


# ****PLEASE UPVOTE IF YOU FIND IT USEFUL****
