
# coding: utf-8

# In this notebook, let us build our model using LinearSVC which is best for text classification problems.

# **Objective:**

# Develop algorithms to classify genetic mutations based on clinical evidence (text)

# In[ ]:


import numpy as np 
import pandas as pd

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Training Data

# In[ ]:


training_variants_df = pd.read_csv("../input/training_variants")


# In[ ]:


training_variants_df.head(5)


# In[ ]:


training_text_df = pd.read_csv("../input/training_text",sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])


# In[ ]:


training_text_df.head(5)


# In[ ]:


training_text_df["Text"][0]


# In[ ]:


training_merge_df = training_variants_df.merge(training_text_df,left_on="ID",right_on="ID")


# In[ ]:


training_merge_df.head(5)


# In[ ]:


training_merge_df.columns


# Testing Data

# In[ ]:


testing_variants_df = pd.read_csv("../input/test_variants")


# In[ ]:


testing_variants_df.head(5)


# In[ ]:


testing_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])


# In[ ]:


testing_text_df.head(5)


# In[ ]:


testing_merge_df = testing_variants_df.merge(testing_text_df,left_on="ID",right_on="ID")


# In[ ]:


testing_merge_df.head(5)


# In[ ]:


training_merge_df["Class"].unique()


# Describing both Training and Testing data

# In[ ]:


training_merge_df.describe()


# In[ ]:


testing_merge_df.describe()


# Check for missing values in both training and testing data columns

# In[ ]:


import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')
msno.bar(training_merge_df)


# In[ ]:


msno.bar(testing_merge_df)


# Split the training data to train and test for checking the model accuracy

# In[ ]:


from sklearn.model_selection import train_test_split

train ,test = train_test_split(training_merge_df,test_size=0.2) 
np.random.seed(0)
train


# In[ ]:


X_train = train['Text'].values
X_test = test['Text'].values
y_train = train['Class'].values
y_test = test['Class'].values


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm


# Set pipeline to build a complete text processing model with Vectorizer, Transformer and LinearSVC

# In[ ]:


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', svm.LinearSVC())
])
text_clf = text_clf.fit(X_train,y_train)


# Getting 65% accuracy with only LinearSVC. Try different ensemble models to get more accurate model. 

# In[ ]:


y_test_predicted = text_clf.predict(X_test)
np.mean(y_test_predicted == y_test)


# Predicting values for test data

# In[ ]:


X_test_final = testing_merge_df['Text'].values


# In[ ]:


predicted_class = text_clf.predict(X_test_final)


# In[ ]:


testing_merge_df['predicted_class'] = predicted_class


# Appended the predicted values to the testing data

# In[ ]:


testing_merge_df.head(5)


# Onehot encoding to get the predicted values as columns

# In[ ]:


onehot = pd.get_dummies(testing_merge_df['predicted_class'])
testing_merge_df = testing_merge_df.join(onehot)


# In[ ]:


testing_merge_df.head(5)


# Preparing submission data

# In[ ]:


submission_df = testing_merge_df[["ID",1,2,3,4,5,6,7,8,9]]
submission_df.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']
submission_df.head(5)


# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# If you really feel this will help you. Please upvote this and encourage me to write more. 
