
# coding: utf-8

# <h1> Welcome to my Titanic Kernel! </h1>
# <h2>This kernel will provide a analysis through the Titanic Disaster to understand the Survivors patterns</h2><br>
# 
# I will handle with data (<i>transform, missings, manipulation</i>), explore the data (<i>descritive and visual</i>) and also create a Deep Learning model

# Are you looking for another interesting Kernels? <a href="https://www.kaggle.com/kabure/kernels">CLICK HERE</a> <br>
# Give me your feedback and if yo like this kernel, votes up

# <i>*I'm from Brazil, so english is not my first language, sorry about some mistakes</i>

# # Table of Contents:
# 
# **1. [Introduction](#Introduction)** <br>
# **2. [Librarys](#Librarys)** <br>
# **3. [Knowning the data](#Known)** <br>
# **4. [Exploring some Variables](#Explorations)** <br>
# **5. [Preprocessing](#Prepocess)** <br>
# **6. [Modelling](#Model)** <br>
# **7. [Validation](#Validation)** <br>
# 

# <a id="Introduction"></a> <br> 
# # **1. Introduction:** 
# <h3> The data have 891 entries on train dataset and 418 on test dataset</h3>
# - 10 columns in train_csv and 9 columns in train_test
# 

# <h2>Competition Description: </h2>
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# <h3>Data Dictionary</h3><br>
# Variable	Definition	Key<br>
# <b>survival</b>	Survival	0 = No, 1 = Yes<br>
# <b>pclass</b>	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd<br>
# <b>sex</b>	Sex	<br>
# <b>Age</b>	Age in years	<br>
# <b>sibsp</b>	# of siblings / spouses aboard the Titanic	<br>
# <b>parch</b>	# of parents / children aboard the Titanic	<br>
# <b>ticket</b>	Ticket number	<br>
# <b>fare</b>	Passenger fare	<br>
# <b>cabin</b>	Cabin number	<br>
# <b>embarked	</b>Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton<br>
# <h3>Variable Notes</h3><br>
# <b>pclass: </b>A proxy for socio-economic status (SES)<br>
# 1st = Upper<br>
# 2nd = Middle<br>
# 3rd = Lower<br>
# <b>age: </b>Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5<br>
# <b>sibsp:</b> The dataset defines family relations in this way...<br>
# - <b>Sibling </b>= brother, sister, stepbrother, stepsister<br>
# - <b>Spouse </b>= husband, wife (mistresses and fiancés were ignored)<br>
# 
# <b>parch: </b>The dataset defines family relations in this way...<br>
# - <b>Parent</b> = mother, father<br>
# - <b>Child </b>= daughter, son, stepdaughter, stepson<br>
# 
# Some children travelled only with a nanny, therefore parch=0 for them.<br>

# I am using the beapproachs as possible but if you think I can do anything another best way, please, let me know.

# <a id="Librarys"></a> <br> 
# # **2. Librarys:** 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 10,8


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# <a id="Known"></a> <br> 
# # **3. First look at the data:** 

# I will start looking the type and informations of the datasets

# In[ ]:


#Looking data format and types
print(df_train.info())
print(df_test.info())


# In[ ]:


#Some Statistics
df_train.describe()


# In[ ]:


#Take a look at the data
print(df_train.head())


# <a id="Known"></a> <br> 
# # **4. Exploring the data:** 

# <h2>To try a new approach in the data, I will start the data analysis by the Name column

# In[ ]:


#Looking how the data is and searching for a re patterns
df_train["Name"].head()


# In[ ]:


#GettingLooking the prefix of all Passengers
df_train['Title'] = df_train.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

#Plotting the result
sns.countplot(x='Title', data=df_train, palette="hls")
plt.xticks(rotation=45)
plt.show()


# In[ ]:


#Do the same on df_test
df_test['Title'] = df_test.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))


# In[ ]:


#Now, I will identify the social status of each title

Title_Dictionary = {
        "Capt":       "Officer",
        "Col":        "Officer",
        "Major":      "Officer",
        "Dr":         "Officer",
        "Rev":        "Officer",
        "Jonkheer":   "Royalty",
        "Don":        "Royalty",
        "Sir" :       "Royalty",
        "the Countess":"Royalty",
        "Dona":       "Royalty",
        "Lady" :      "Royalty",
        "Mme":        "Mrs",
        "Ms":         "Mrs",
        "Mrs" :       "Mrs",
        "Mlle":       "Miss",
        "Miss" :      "Miss",
        "Mr" :        "Mr",
        "Master" :    "Master"
                   }
    
# we map each title to correct category
df_train['Title'] = df_train.Title.map(Title_Dictionary)
df_test['Title'] = df_test.Title.map(Title_Dictionary)


# In[ ]:


print("Chances to survive based on titles: ")
print(df_train.groupby("Title")["Survived"].mean())

#Plotting the results
sns.countplot(x='Title', data=df_train, palette="hls",hue="Survived")
plt.xticks(rotation=45)
plt.show()


# It's interesting... Children's and ladys first, huh?

# <h1> Now I will handle the Age variable that has a high number of NaN's, using some columns to correctly input he missing Age's

# In[ ]:


#First I will look my distribuition without NaN's
#I will create a df to look distribuition 
age_high_zero_died = df_train[(df_train["Age"] > 0) & 
                              (df_train["Survived"] == 0)]
age_high_zero_surv = df_train[(df_train["Age"] > 0) & 
                              (df_train["Survived"] == 1)]


sns.distplot(age_high_zero_surv["Age"], bins=24, color='g')
sns.distplot(age_high_zero_died["Age"], bins=24, color='r')
plt.title("Distribuition and density by Age",fontsize=15)
plt.xlabel("Age",fontsize=12)
plt.ylabel("Density Died and Survived",fontsize=12)
plt.show()


# In[ ]:


#Let's group the median age by sex, pclass and title, to have any idea and maybe input in Age NAN's

age_group = df_train.groupby(["Sex","Pclass","Title"])["Age"]

print(age_group.median())


# This might show us a better way to input the NAN's 
# 
# <b>For example: </b> an male in 2 class that is a Officer the median Age is 42. <br>
# And we will use that to complete the missing data
# 

# In[ ]:


#inputing the values on Age Na's
df_train.loc[df_train.Age.isnull(), 'Age'] = df_train.groupby(['Sex','Pclass','Title']).Age.transform('median')

print(df_train["Age"].isnull().sum())


# In[ ]:


#Let's see the result of the inputation
sns.distplot(df_train["Age"], bins=24)
plt.title("Distribuition and density by Age")
plt.xlabel("Age")
plt.show()


# In[ ]:


#separate by survivors or not
g = sns.FacetGrid(df_train, col='Survived',size=5)
g = g.map(sns.distplot, "Age")
plt.show()


# Now let's categorize them 

# In[ ]:


#df_train.Age = df_train.Age.fillna(-0.5)

interval = (0, 5, 12, 18, 25, 35, 60, 120)
cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']

df_train["Age_cat"] = pd.cut(df_train.Age, interval, labels=cats)

df_train["Age_cat"].head()


# In[ ]:


#Do the same to df_test

interval = (0, 5, 12, 18, 25, 35, 60, 120)
cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']

df_test["Age_cat"] = pd.cut(df_test.Age, interval, labels=cats)


# In[ ]:


#Describe of categorical Age
print(pd.crosstab(df_train.Age_cat, df_train.Survived))

#Plotting the result
sns.countplot("Age_cat",data=df_train,hue="Survived", palette="hls")
plt.xlabel("Categories names")
plt.title("Age Distribution ")


# It look's better

# In[ ]:


#Looking the Fare distribuition to survivors and not survivors

sns.distplot(df_train[df_train.Survived == 0]["Fare"], 
             bins=50, color='r')
sns.distplot(df_train[df_train.Survived == 1]["Fare"], 
             bins=50, color='g')
plt.title("Fare Distribuition by Survived", fontsize=15)
plt.xlabel("Fare", fontsize=12)
plt.ylabel("Density",fontsize=12)
plt.show()


# <br>
# Description of Fare variable<br>
# - Min: 0<br>
# - Median: 14.45<br>
# - Mean: 32.20<br>
# - Max: 512.32<br> 
# - Std: 49.69<br>
# 
# <h3>I will create a categorical variable to treat the Fare expend</h3><br>
# I will use the same technique used in Age but now I will use the quantiles to binning
# 
# 

# In[ ]:


#Filling the NA's with -0.5
df_train.Fare = df_train.Fare.fillna(-0.5)

#intervals to categorize
quant = (-1, 0, 8, 15, 31, 600)

#Labels without input values
label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']

#doing the cut in fare and puting in a new column
df_train["Fare_cat"] = pd.cut(df_train.Fare, quant, labels=label_quants)

#Description of transformation
print(pd.crosstab(df_train.Fare_cat, df_train.Survived))

#Plotting the new feature
sns.countplot(x="Fare_cat", hue="Survived", data=df_train, palette="hls")
plt.title("Count of survived x Fare expending")


# In[ ]:


# Replicate the same to df_test
df_test.Fare = df_test.Fare.fillna(-0.5)

quant = (-1, 0, 8, 15, 31, 1000)
label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']

df_test["Fare_cat"] = pd.cut(df_test.Fare, quant, labels=label_quants)


# <h2>To complete this part, I will now work on "Names"

# In[ ]:


#Now lets drop the variable Fare, Age and ticket that is irrelevant now
del df_train["Fare"]
del df_train["Ticket"]
del df_train["Age"]
del df_train["Cabin"]
del df_train["Name"]

#same in df_test
del df_test["Fare"]
del df_test["Ticket"]
del df_test["Age"]
del df_test["Cabin"]
del df_test["Name"]


# In[ ]:


#Looking the result of transformations
df_train.head()


# <h1>It's looking ok

# Now, lets start explore the data

# In[ ]:


# Let see how many people die or survived
print("Total of Survived or not: ")
print(df_train.groupby("Survived")["PassengerId"].count())
sns.countplot(x="Survived", data=df_train,palette="hls")
plt.title('Total Distribuition by survived or not')


# In[ ]:


print(pd.crosstab(df_train.Survived, df_train.Sex))
sns.countplot(x="Sex", data=df_train, hue="Survived",palette="hls")
plt.title('Sex Distribuition by survived or not')


# <h2>We can look that % dies to mens are much higher than female

# <h1>Now, lets do some exploration in Pclass and Embarked to see if might have some information to build the model

# In[ ]:


# Distribuition by class
print(pd.crosstab(df_train.Pclass, df_train.Embarked))
sns.countplot(x="Embarked", data=df_train, hue="Pclass",palette="hls")
plt.title('Embarked x Pclass')


# In[ ]:


#lets input the NA's with the highest frequency
df_train["Embarked"] = df_train["Embarked"].fillna('S')


# In[ ]:


# Exploring Survivors vs Embarked
print(pd.crosstab(df_train.Survived, df_train.Embarked))
sns.countplot(x="Embarked", data=df_train, hue="Survived",palette="hls")
plt.title('Class Distribuition by survived or not')


# In[ ]:


# Exploring Survivors vs Pclass
print(pd.crosstab(df_train.Survived, df_train.Pclass))
sns.countplot(x="Pclass", data=df_train, hue="Survived",palette="hls")
plt.title('Class Distribuition by survived or not')


# <b>Looking the graphs, is clear that 3st class and Embarked at Southampton have a high probabilities to not survive</b>

# To finish the analysis I let's look the Sibsp and Parch variables

# In[ ]:


g = sns.factorplot(x="SibSp",y="Survived",data=df_train,kind="bar", size = 6, palette = "hls")
g = g.set_ylabels("Probability(Survive)")


# Interesting. With 1 or 2 siblings/spouses have more chance to survived the disaster

# In[ ]:


# Explore Parch feature vs Survived
g  = sns.factorplot(x="Parch",y="Survived",data=df_train, kind="bar", size = 6,palette = "hls")
g = g.set_ylabels("survival probability")


# We can see a high standard deviation in the survival with 3 parents/children person's <br>
# Also that small families (1~2) have more chance to survival than single or big families

# So to Finish our exploration I will create a new column to with familiees size

# In[ ]:


#Create a new column and sum the Parch + SibSp + 1 that refers the people self
df_train["FSize"] = df_train["Parch"] + df_train["SibSp"] + 1

df_test["FSize"] = df_test["Parch"] + df_test["SibSp"] + 1


# In[ ]:


print(pd.crosstab(df_train.FSize, df_train.Survived))
sns.factorplot(x="FSize",y="Survived", data=df_train, kind="bar",size=6)
plt.show()


# In[ ]:


del df_train["SibSp"]
del df_train["Parch"]

del df_test["SibSp"]
del df_test["Parch"]


# OK, its might be enough to start with the preprocess and builting the model
# 

# <a id="Preprocess"></a> <br> 
# # **5. Preprocessing :** 

# In[ ]:


df_train.head()


# Now we might have information enough to think about the model structure

# In[ ]:


df_train = pd.get_dummies(df_train, columns=["Sex","Embarked","Age_cat","Fare_cat","Title"],                          prefix=["Sex","Emb","Age","Fare","Prefix"], drop_first=True)

df_test = pd.get_dummies(df_test, columns=["Sex","Embarked","Age_cat","Fare_cat","Title"],                         prefix=["Sex","Emb","Age","Fare","Prefix"], drop_first=True)


# In[ ]:


#Finallt, lets look the correlation of df_train
plt.figure(figsize=(15,12))
plt.title('Correlation of Features for Train Set')
sns.heatmap(df_train.astype(float).corr(),vmax=1.0,  annot=True)
plt.show()


# In[ ]:


df_train.shape


# In[ ]:


train = df_train.drop(["Survived","PassengerId"],axis=1)
train_ = df_train["Survived"]

test_ = df_test.drop(["PassengerId"],axis=1)

X_train = train.values
y_train = train_.values

X_test = test_.values
X_test = X_test.astype(np.float64, copy=False)


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# <a id="Model"></a> <br> 
# # **6. Modelling : ** 

# <h3>Titanic survivors prediction: <br>
# a binary classification example</h3>
# Two-class classification, or binary classification, may be the most widely applied kind of machine-learning problem.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras
from keras.optimizers import SGD
import graphviz


# <h1>Anatomy of a neural network: </h1>
# 
# As you saw in the previous chapters, training a neural network revolves around the following
# objects:
# - Layers, which are combined into a network (or model)
# - The input data and corresponding targets
# - The loss function, which defines the feedback signal used for learning
# - The optimizer, which determines how learning proceeds
# 
# 
# 
# 
# <h2> Layers: the building blocks of deep learning</h2>
# from keras import layers<br>
# layer = layers.Dense(32, input_dim=data_dimension)) 
# 
# - We can think of layers as the LEGO bricks of deep learning, a metaphor that is
# made explicit by frameworks like Keras. Building deep-learning models in Keras is
# done by clipping together compatible layers to form useful data-transformation pipelines.
# 
# 
# <h2>What are activation functions, and why are they necessary?</h2>
# Without an activation function like relu (also called a non-linearity), the Dense layer would consist of two linear operations—a dot product and an addition: <br><br>
# <i>output = dot(W, input) + b</i><br><br>
# 
# So the layer could only learn linear transformations (affine transformations) of the
# input data: the hypothesis space of the layer would be the set of all possible linear
# transformations of the input data into a 16-dimensional space. 
# 
# 
# <h2>Loss functions and optimizers:<br>
# keys to configuring the learning process</h2>
# Once the network architecture is defined, you still have to choose two more things:
# - <b>Loss function (objective function) </b>- The quantity that will be minimized during
# training. It represents a measure of success for the task at hand.
# - <b>Optimizer</b> - Determines how the network will be updated based on the loss function.
# It implements a specific variant of stochastic gradient descent (SGD).

# In[ ]:


# Creating the model
model = Sequential()

# Inputing the first layer with input dimensions
model.add(Dense(18, 
                activation='relu',  
                input_dim=20,
                kernel_initializer='uniform'))
#The argument being passed to each Dense layer (18) is the number of hidden units of the layer. 
# A hidden unit is a dimension in the representation space of the layer.

#Stacks of Dense layers with relu activations can solve a wide range of problems
#(including sentiment classification), and you’ll likely use them frequently.

# Adding an Dropout layer to previne from overfitting
model.add(Dropout(0.50))

#adding second hidden layer 
model.add(Dense(12,
                kernel_initializer='uniform',
                activation='relu'))

# Adding another Dropout layer
model.add(Dropout(0.50))

# adding the output layer that is binary [0,1]
model.add(Dense(1,
                kernel_initializer='uniform',
                activation='sigmoid'))
#With such a scalar sigmoid output on a binary classification problem, the loss
#function you should use is binary_crossentropy

#Visualizing the model
model.summary()


# Stacks of Dense layers with relu activations can solve a wide range of problems (including sentiment classification), and you’ll likely use them frequently.

# Finally, we need to choose a loss function and an optimizer. 

# In[ ]:


#Creating an Stochastic Gradient Descent
sgd = SGD(lr = 0.01, momentum = 0.9)

# Compiling our model
model.compile(optimizer = sgd, 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
#optimizers list
#optimizers['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

# Fitting the ANN to the Training set
model.fit(X_train, y_train, 
               batch_size = 60, 
               epochs = 30, verbose=2)


# Because you’re facing a binary classification problem and the output of your network is a probability (you end your network with a single-unit layer with a sigmoid activation), it’s best to use the <i>binary_crossentropy</i> loss.

# <h1>Evaluating the model</h1>

# In[ ]:


scores = model.evaluate(X_train, y_train, batch_size=30)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Not bad result to a simple model! Let's now verify the validation of our model, to see and understand the learning curve

# <a id="Validation"></a> <br> 
# # **7. Validation: ** 

# In[ ]:


# Fit the model
history = model.fit(X_train, y_train, validation_split=0.20, 
                    epochs=180, batch_size=10, verbose=0)

# list all data in history
print(history.history.keys())


# Let's look this keys values further

# In[ ]:


# summarizing historical accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Why this occurs and how to solve this problem in graph? it's a overffiting? 

# In[ ]:



# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


y_pred = model.predict(X_test)


# <h1>It's my first Deep Learning implementation... I am studying about this and I will continue editing this Kernel to improve the results</h1>

# Give me your feedback how can I increase this model =) 

# In[ ]:


# Trying to implementing the TensorBoard to evaluate the model

callbacks = [
    keras.callbacks.TensorBoard(log_dir='my_log_dir',
                                histogram_freq=1,
                                embeddings_freq=1,
                               )
]

#history = classifier.fit(X_train, y_train,
#                         epochs=80,
#                         batch_size=10,
#                         validation_split=0.2,
#                         callbacks=callbacks)

#Its backing an error 
#ValueError: No variables to save

#How to solve this ?

