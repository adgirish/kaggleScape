
# coding: utf-8

# This tutorial is aimed at beginners, especially those who are both new to machine learning/data science as well as python. 
# 
# In this tutorial I would walk you through the process of building a predictive model, namely a decision tree. 
# 
# The pipeline consists of several steps:
# 1. Exploratory Data Analysis (EDA) - understanding the data and the underlying interactions between the different variables
# 2. Data Pre-processing - preparing the data for modelling
# 3. Building the model
# 4. Evaluating the performance of the model, and possibly fine-tune and tweak it if necessary
# 
# The goal of the model is to predict whether a passenger survived the Titanic disaster, given their age, class and a few other features.
# 
# Almost every line of code would be explained, so those who are more familiar with python (and especially with the numpy and pandas libraries) are welcome to skip the first parts

# # Loading Libraries
# Not all python capabilities are loaded to your working environment by default (even if they exist and are installed on your computer), Therefore, we would need to import every library we are going to use. numpy and pandas are probably the most commonly used libraries. 
# 
# Numpy is requried whenever calculations are required (calculating means, medians, sqaure root, etc.).
# pandas is a great module for data processing and data frames. 
# 
# We can choose alias names to our modules for the sake of convenience (numpy --> np, pandas --> pd)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Laoding the data
# We would use the pandas module to read the files. using the "read_csv" function. the files format is.csv (similar to .xls)
# 
# In the round brackets we have the path to where the files are saved on Kaggle's server.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# We now have two data frames: train and test. Let's examine them:

# In[ ]:


train.head()


# In[ ]:


test.head()


# The "head" fucntion displays the first 5 rows of the data frame. 
# 
# Let us explore the columns: 
# * PassengerId - this is a just a generated Id
# * Pclass - which class did the passenger ride - first, second or third
# * Name - self explanatory
# * Sex - male or female
# * Age
# * SibSp - were the passenger's spouse or siblings  with them on the ship
# * Partch -  were the passenger's parents or children  with them on the ship
# * Ticket - ticket number
# * Fare - ticker price
# * Cabin
# * Embarked - port of embarkation
# * Survived - did the passenger  survive the sinking of the Titanic?
# 
# Notice the difference between the test and train data frames: the "survived" column is missing in the test data frame. this is intentional - the whole goal is building a model that would predict the survival probability of a person, given their basic features.

# # Exploratory Data Analysis (EDA)
# 
# After loading the data, let us examine it. It is usually not recommended to throw all the data into a predictive model without first understanding the data. this would often help us improving our model.

# In[ ]:


print('Total number of passangers in the training data...', len(train))
print('Number of passangers in the training data who survived...', len(train[train['Survived'] == 1]))


# The "len" function gives the length of whatever is the input, in that case our train data frame.
# 
# We can access a specific column in a data frame by specifying its name in square brackets. we can also filter the values we are interested in with a condition.
# 
# So the second line in the above cell should be understood as following:
# "give me the length of the train data frame, if we only count the rows where the value of the "Survived" column id 1"
# or, simply - "give me the number of people (rows) who survived ("survived = 1")
# 
# When we want to use the "equals" symbol in the context of a comparison\condition statement, we use "==" instead of "="

# Now, similarly, let's see what is the % of men and women who survived, and then by the same token with class and age:

# In[ ]:


print('% of men who survived', 100*np.mean(train['Survived'][train['Sex'] == 'male']))
print('% of women who survived', 100*np.mean(train['Survived'][train['Sex'] == 'female']))


# Here we first use the numpy module, namely the "mean" function".
# Let's try to understand the logics of the first line:
# * from left to right -   "print 100 multiplied by the mean of the "survived" column but only where the sex is "male" "
# * might be easier from right to left: "let us only look at rows where the sex is "male", now from this reduced data frame let us only regard the "survived" column. let's take the mean of the column vales. and multiply by 100"

# In[ ]:


print('% of passengers who survived in first class', 100*np.mean(train['Survived'][train['Pclass'] == 1]))
print('% of passengers who survived in third class', 100*np.mean(train['Survived'][train['Pclass'] == 3]))


# In[ ]:


print('% of children who survived', 100*np.mean(train['Survived'][train['Age'] < 18]))
print('% of adults who survived', 100*np.mean(train['Survived'][train['Age'] > 18]))


# As we can see, at least in terms of survival statistics, the "Titanic" movie (spoiler alert) was an accurate representation of reality

# # Data Pre-processing
# ## Non numeric features
# 
# We are going to use a decision tree model. The model requires only numeric values, but one of our features is categorical: "female" or "male". this can easily be fixed by encoding this feature: "male" = 1, "female" = 0

# In[ ]:


train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)


# The "apply" means "do that for each value in the column". the statement in the brackets should be read as following :"for every value in the column ("lambda x:") if it's a male then replace with 1, otherwise replace with 0

# ## Missing Values
# Another common problem which has to be addressed is missing values. We can simply delete rows with missing values, but usually we would 'want to take advantage of as many data points as possible. Replacing missing values with zeros would not be a good idea - as age 0 or price 0 have actual meanings and that would change our data.
# 
# Therefore a good replacement value would be something that doesn't affect the data too much, such as the median or mean. the "fillna" function replaces every NaN (not a number) entry with the given input (the mean of the column in our case):

# In[ ]:


train['Age'] = train['Age'].fillna(np.mean(train['Age']))
train['Fare'] = train['Fare'].fillna(np.mean(train['Fare']))


# ## Omit irrelevant columns
# Let us only take the columns we find relevant. ID columns are never relevant (or at least should not be, if the data was sampled randomly). As our model is very simple, let us also omit the port, cabin and name columns although more sophisticated models can definitely take advantage of them

# In[ ]:


train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]


# ## X and Y
# Y has the labels, our answers column. X is all the rest of the data - the features, without the labels (The survived column). This sepration would hoepfully be clearer in a few cells

# In[ ]:


X = train.drop('Survived', axis = 1)
y = train['Survived']


# As explained above, X now has the train data without the "Survived" column (this is acheived with the "drop" function). Y, on the other hand, has only the "Survived" column.

# # Predict
# 
# We have our training data, and we have our test data. but in order to evaluate our model we need to split the training dataset into a train dataset and an evaluation dataset (validation). The validation data would be used to evaluate the model, while the training data would be used to train the data. 
# 
# To do that, we can use the function "train_test_split" from the sklearn module. the sklean module is probably the most commonly used library in most simple machine learning tasks (this does not include deep learning where other libraries can be more popular)

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## Training the model
# Now we are finally ready, and we can train the model.
# 
# First, we need to import our model - A decision tree classifier (again, using the sklearn library).
# 
# Then we would feed the model both with the data (X_train) and the answers for that data (y_train)

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# The training happens in the third line (the "fit" function). During the training, the algorithm is trying to build the optimal decision tree. The way it works is very similar to the game "21 Questions". In the game we always start with general questions that give the best partition of the data ("is the character you are thinking of a male? are them alive?" etc.). It tries to acheive the best score on the training data - that is, building a model that would predict the survival outcome of as many passengers as possible. 

# ## Evaluate the model

# Now we have a model. Let's evaulate it with using the accuracy_score function. This output of the function is the number of right answers (passengers survival/death was predicted correctly) divided by the total number of passengers 

# In[ ]:


from sklearn.metrics import accuracy_score
print('Training accuracy...', accuracy_score(y_train, classifier.predict(X_train)))
print('Validation accuracy', accuracy_score(y_test, classifier.predict(X_test)))


# The accuracy function compares between the actual results (our ground truth - y_train or y_test) with the prediction of the model (given by "classifier.predict(X_train)" or "classifier.predict(X_test)" respectrively.

# The large difference between the training score and the validation score suggets that our model **overfits**. That is, instead of leraning general rules that can be applied on unseen data, it does something that is more similar to memorize the training data. So our model performs really well on the training data (98% accuracy) but not remotely as well on the validation data.
# 
# It is clear once we plot the tree. the next bulk of code would not be explained and can be regarded as a useful magic that plots decision trees.

# In[ ]:


from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont


with open("tree1.dot", 'w') as f:
    f = tree.export_graphviz(classifier,
                                  out_file=f,
                                  impurity = False,
                                  feature_names = X_test.columns.values,
                                  class_names = ['No', 'Yes'],
                                  rounded = True,
                                  filled= True )

    #Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

    # Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# This is a very complex model, and if you zoom in you would see that in many leaves we have only 1 sample. this means that the model learned many complex rules to memorize the survival or death of each passenger in the training data. 
# 
# Let's take Mr. Owen Harris	(the first row of the training data),  having such a complex model is similar to having a rule that says that if a passenger is in class 3, and is a male, and his age is more than 21, and his age is less than 23, and his fare is more than 7, and his fare is less than 8, and he has a sibling/spouse on board with him but no parents/children, then he would not survive. But this is obviously a rule that was tailor made to Mr Harris, and is equivalent to simply saying "Mr Harris did not survive", a rule that cannot be generalized to new unseen passengers who are not Mr Harris.

# ## Improve the model
# We can reduce overfitting by limiting the number of "questions" that the model is allowed to ask. as each node in the tree is a question, by limiting the depth of the tree we can limit the number of questions. So let us again create an instance of a decision tree, but this one cannot produce trees deeper than 3 (3 questions):

# In[ ]:


classifier = DecisionTreeClassifier(max_depth = 3)
classifier.fit(X_train, y_train)


# In[ ]:


print('train score...' , accuracy_score(y_train, classifier.predict(X_train)))
print('test score...', accuracy_score(y_test, classifier.predict(X_test)))


# We can see that while the train score went down, the test score has improved and it is now almost as high as the train score. This means that the model does not overfit as badly anymore. 82% accuracy with such a simple model is quite impressive in my opinion. 
# 
# Let's visualize the tree again using the same code snippet from above:

# In[ ]:


with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(classifier,
                              out_file=f,
                              impurity = False,
                              feature_names = X_test.columns.values,
                              class_names = ['No', 'Yes'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# The training is the process of finding the most important features, and then use them to split the data. The training algorithm found that the most important features is the Sex. secondly, the class for females, and age for males. the bluer the block is, the higher the survival rate is, and opoositely with browner blocker. 
