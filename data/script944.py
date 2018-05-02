
# coding: utf-8

# **This is my first attempt on Kaggle ever. I've learned a lot from the tutorials and some other kernels. Here I incorporate my ideas and what I learned into this kernel. Please upvote if you find this useful. Feel free to leave comments below.**
# 
# **(09/12/2017 update) Neural network results in a score > 0.81**
# 
# **Let's begin now!**

# In[ ]:


# Import all the libraries I need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# ignore Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV

import keras 
from keras.models import Sequential # intitialize the ANN
from keras.layers import Dense      # create layers

# load the data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df = df_train.append(df_test , ignore_index = True)

# some quick inspections
df_train.shape, df_test.shape, df_train.columns.values


# Let's look at individual features one by one:
# 
# ## Pclass

# In[ ]:


# check if there is any NAN
df['Pclass'].isnull().sum(axis=0)


# In[ ]:


# inspect the correlation between Pclass and Survived
df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# Pclass should be a very useful featrue

# ## Name

# In[ ]:


df.Name.head(10)


# Each name has a title, which is clearly what matters since it contains information of gender or status. Let's extract the titles from these names.

# In[ ]:


df['Title'] = df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())

# inspect the amount of people for each title
df['Title'].value_counts()


# Looks like the main ones are "Master", "Miss", "Mr", "Mrs". Some of the others can be be merged into some of these four categories. For the rest, I'll just call them 'Others'

# In[ ]:


df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')
df.Title.loc[ (df.Title !=  'Master') & (df.Title !=  'Mr') & (df.Title !=  'Miss') 
             & (df.Title !=  'Mrs')] = 'Others'

# inspect the correlation between Title and Survived
df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


# inspect the amount of people for each title
df['Title'].value_counts()


# Now we can use dummy encoding for these titles and drop the original names

# In[ ]:


df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1).drop(labels=['Name'], axis=1)


# ## Sex

# In[ ]:


# check if there is any NAN
df.Sex.isnull().sum(axis=0)


# In[ ]:


# inspect the correlation between Sex and Survived
df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()


# In[ ]:


# map the two genders to 0 and 1
df.Sex = df.Sex.map({'male':0, 'female':1})


# ## Age

# In[ ]:


# check if there is any NAN
df.Age.isnull().sum(axis=0)


# There are 263 missing values! Age can probably be inferred from other features such as Title, Fare, SibSp, Parch. So I decide to come back to Age after I finish inspecting the other features.

# ## SibSp and Parch 

# In[ ]:


# check if there is any NAN
df.SibSp.isnull().sum(axis=0), df.Parch.isnull().sum(axis=0)


# In[ ]:


# create a new feature "Family"
df['Family'] = df['SibSp'] + df['Parch'] + 1

# inspect the correlation between Family and Survived
df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()


# In[ ]:


# inspect the amount of people for each Family size
df['Family'].value_counts()


# We can see that the survival rate increases with the family size, but not beyond Family = 4. Also, the amount of people in big families is much lower than those in small families. I will combine all the data with Family > 4 into one category. Since people in big families have an even lower survival rate (0.161290) than those who are alone, I decided to map data with Family > 4 to Family = 0, such that the survival rate always increases as Family increases.

# In[ ]:


df.Family = df.Family.map(lambda x: 0 if x > 4 else x)
df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()


# In[ ]:


df['Family'].value_counts()


# ## Ticket

# In[ ]:


# check if there is any NAN
df.Ticket.isnull().sum(axis=0)


# In[ ]:


df.Ticket.head(20)


# It looks like there are two types of tickets: (1) number (2) letters + number
# 
# Ticket names with letters probably represent some special classes. For the numbers, the majority of tickets have their first digit = 1, 2, or 3, which probably also represent different classes. So I just keep the first element (a letter or a single-digit  number) of these ticket names

# In[ ]:


df.Ticket = df.Ticket.map(lambda x: x[0])

# inspect the correlation between Ticket and Survived
df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()


# In[ ]:


# inspect the amount of people for each type of tickets
df['Ticket'].value_counts()


# We can see that the majority of tickets are indeed "3", "2", and "1", and their corresponding survival rates are "1" > "2" > "3". For the others, the survival rate is low except for "9","C","F","P", and "S". However, "9" and "F" are very small sample. The correlation we see here probably comes from Pclass or Fare, so let's check.

# In[ ]:


df[['Ticket', 'Fare']].groupby(['Ticket'], as_index=False).mean()


# In[ ]:


df[['Ticket', 'Pclass']].groupby(['Ticket'], as_index=False).mean()


# Indeed, there is some relation between Ticket and Fare, or between Ticket and Pclass. Also, one interesting thing is that "P" corresponds to very high Fare and Pcalss (better than "1"). This is an additional information that cannot be seen in other features. So keeping Ticket as a feature might be useful.
# 
# I'll come back to Ticket later. I don't further transform it now because its current form might be useful for guessing some missing values in other features. 

# ## Fare

# In[ ]:


# check if there is any NAN
df.Fare.isnull().sum(axis=0)


# Only one missing value. Fare can probably be inferred from Ticket, Pclass, Cabin and Embarked. Let's see the corresponding values of for these features.

# In[ ]:


df.Ticket[df.Fare.isnull()]


# In[ ]:


df.Pclass[df.Fare.isnull()]


# In[ ]:


df.Cabin[df.Fare.isnull()]


# In[ ]:


df.Embarked[df.Fare.isnull()]


# There is no corresponding value for Cabin, so let's look at the relation between Fare and the three features.  

# In[ ]:


# use boxplot to visualize the distribution of Fare for each Pclass
sns.boxplot('Pclass','Fare',data=df)
plt.ylim(0, 300) # ignore one data point with Fare > 500
plt.show()


# In[ ]:


# inspect the correlation between Pclass and Fare
df[['Pclass', 'Fare']].groupby(['Pclass']).mean()


# In[ ]:


# divide the standard deviation by the mean. A lower ratio means a tighter 
# distribution of Fare in each Pclass
df[['Pclass', 'Fare']].groupby(['Pclass']).std() / df[['Pclass', 'Fare']].groupby(['Pclass']).mean()


# In[ ]:


# use boxplot to visualize the distribution of Fare for each Ticket
sns.boxplot('Ticket','Fare',data=df)
plt.ylim(0, 300) # ignore one data point with Fare > 500
plt.show()


# In[ ]:


# inspect the correlation between Ticket and Fare 
# (we saw this earlier)
df[['Ticket', 'Fare']].groupby(['Ticket']).mean()


# In[ ]:


# divide the standard deviation by the mean. A lower ratio means a tighter 
# distribution of Fare in each Ticket type
df[['Ticket', 'Fare']].groupby(['Ticket']).std() /  df[['Ticket', 'Fare']].groupby(['Ticket']).mean()


# In[ ]:


# use boxplot to visualize the distribution of Fare for each Embarked
sns.boxplot('Embarked','Fare',data=df)
plt.ylim(0, 300) # ignore one data point with Fare > 500
plt.show()


# In[ ]:


# inspect the correlation between Embarked and Fare
df[['Embarked', 'Fare']].groupby(['Embarked']).mean()


# In[ ]:


# divide the standard deviation by the mean. A lower ratio means a tighter 
# distribution of Fare in each Embarked
df[['Embarked', 'Fare']].groupby(['Embarked']).std() /  df[['Embarked', 'Fare']].groupby(['Embarked']).mean()


# Looks like Fare indeed has correlation with these three features. I'll guess the missing value using the median value of (Pcalss = 3) & (Ticket = 3) & (Embarked = S)

# In[ ]:


guess_Fare = df.Fare.loc[ (df.Ticket == '3') & (df.Pclass == 3) & (df.Embarked == 'S')].median()
df.Fare.fillna(guess_Fare , inplace=True)

# inspect the mean Fare values for people who died and survived
df[['Fare', 'Survived']].groupby(['Survived'],as_index=False).mean()


# In[ ]:


# visualize the distribution of Fare for people who survived and died
grid = sns.FacetGrid(df, hue='Survived', size=4, aspect=1.5)
grid.map(plt.hist, 'Fare', alpha=.5, bins=range(0,210,10))
grid.add_legend()
plt.show()


# In[ ]:


# visualize the correlation between Fare and Survived using a scatter plot
df[['Fare', 'Survived']].groupby(['Fare'],as_index=False).mean().plot.scatter('Fare','Survived')
plt.show()


# We can see that people with lower Fare are less likely to survive. But this is certainly not a smooth curve if we don't bin the data. It would be better to feed machine learning algorithms with intervals of Fare, because using the original Fare values would likely cause over-fitting. 

# In[ ]:


# bin Fare into five intervals with equal amount of people
df['Fare-bin'] = pd.qcut(df.Fare,5,labels=[1,2,3,4,5]).astype(int)

# inspect the correlation between Fare-bin and Survived
df[['Fare-bin', 'Survived']].groupby(['Fare-bin'], as_index=False).mean()


# Now the correlation is clear after binning the data!
# 
# ## Cabin

# In[ ]:


# check if there is any NAN
df.Cabin.isnull().sum(axis=0)


# This is highly incomplete. We have two choices: (1) map the missing ones to a new cabin category "unknown" (2) just drop this feature. I have tried both and I decided to choose (2)

# In[ ]:


df = df.drop(labels=['Cabin'], axis=1)


# ## Embarked

# In[ ]:


# check if there is any NAN
df.Embarked.isnull().sum(axis=0)


# In[ ]:


df.describe(include=['O']) # S is the most common


# In[ ]:


# fill the NAN
df.Embarked.fillna('S' , inplace=True )


# In[ ]:


# inspect the correlation between Embarked and Survived as well as some other features
df[['Embarked', 'Survived','Pclass','Fare', 'Age', 'Sex']].groupby(['Embarked'], as_index=False).mean()


# The survival rate does change between different Embarked values. However, it is due to the changes of other features. For example, people from Embarked = C are more likely to survive because they are generally richer (Pclass, Fare). People from Embarked = S has the lowest survival rate because it has the lowest fraction of female passengers, even though they are a bit richer than people from Embarked = Q.
# 
# I therefore decided to drop this feature as well.

# In[ ]:


df = df.drop(labels='Embarked', axis=1)


# Now we can go back to **Age** and try to fill the missing values. Let's see how it relates to other features

# In[ ]:


# visualize the correlation between Title and Age
grid = sns.FacetGrid(df, col='Title', size=3, aspect=0.8, sharey=False)
grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))
plt.show()


# In[ ]:


# inspect the mean Age for each Title
df[['Title', 'Age']].groupby(['Title']).mean()


# In[ ]:


# inspect the standard deviation of Age for each Title
df[['Title', 'Age']].groupby(['Title']).std()


# In[ ]:


# visualize the correlation between Fare-bin and Age
grid = sns.FacetGrid(df, col='Fare-bin', size=3, aspect=0.8, sharey=False)
grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))
plt.show()


# In[ ]:


# inspect the mean Age for each Fare-bin
df[['Fare-bin', 'Age']].groupby(['Fare-bin']).mean()


# In[ ]:


# inspect the standard deviation of Age for each Fare-bin
df[['Fare-bin', 'Age']].groupby(['Fare-bin']).std()


# In[ ]:


# visualize the correlation between SibSp and Age
grid = sns.FacetGrid(df, col='SibSp', col_wrap=4, size=3.0, aspect=0.8, sharey=False)
grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))
plt.show()


# In[ ]:


# inspect the mean Age for each SibSp
df[['SibSp', 'Age']].groupby(['SibSp']).mean()


# In[ ]:


# inspect the standard deviation of Age for each SibSp
df[['SibSp', 'Age']].groupby(['SibSp']).std()


# In[ ]:


# visualize the correlation between Parch and Age
grid = sns.FacetGrid(df, col='Parch', col_wrap=4, size=3.0, aspect=0.8, sharey=False)
grid.map(plt.hist, 'Age', alpha=.5, bins=range(0,105,5))
plt.show()


# In[ ]:


# inspect the mean Age for each Parch
df[['Parch', 'Age']].groupby(['Parch']).mean()


# In[ ]:


# inspect the standard deviation of Age for each Parch
df[['Parch', 'Age']].groupby(['Parch']).std() 


# The change of Age as a function of Title, Fare-bin, or SibSp is quite significant, so I'll use them to guess the missing values. I use a random forest regressor to do this. 

# In[ ]:


# notice that instead of using Title, we should use its corresponding dummy variables 
df_sub = df[['Age','Master','Miss','Mr','Mrs','Others','Fare-bin','SibSp']]

X_train  = df_sub.dropna().drop('Age', axis=1)
y_train  = df['Age'].dropna()
X_test = df_sub.loc[np.isnan(df.Age)].drop('Age', axis=1)

regressor = RandomForestRegressor(n_estimators = 300)
regressor.fit(X_train, y_train)
y_pred = np.round(regressor.predict(X_test),1)
df.Age.loc[df.Age.isnull()] = y_pred

df.Age.isnull().sum(axis=0) # no more NAN now


# And then we still need to bin the data into different Age intervals, for the same reason as Fare

# In[ ]:


bins = [ 0, 4, 12, 18, 30, 50, 65, 100] # This is somewhat arbitrary...
age_index = (1,2,3,4,5,6,7)
#('baby','child','teenager','young','mid-age','over-50','senior')
df['Age-bin'] = pd.cut(df.Age, bins, labels=age_index).astype(int)

df[['Age-bin', 'Survived']].groupby(['Age-bin'],as_index=False).mean()


# Now we can look at **Ticket** again 

# In[ ]:


df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()


# In[ ]:


df['Ticket'].value_counts()


# The main categories of Ticket are "1", "2", "3", "P", "S", and "C", so I will combine all the others into "4"

# In[ ]:


df['Ticket'] = df['Ticket'].replace(['A','W','F','L','5','6','7','8','9'], '4')

# check the correlation again
df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()


# In[ ]:


# dummy encoding
df = pd.get_dummies(df,columns=['Ticket'])


# ## Modeling and Prediction
# Now we can drop the features we don't need and split the data into training and test sets

# In[ ]:


df = df.drop(labels=['SibSp','Parch','Age','Fare','Title'], axis=1)
y_train = df[0:891]['Survived'].values
X_train = df[0:891].drop(['Survived','PassengerId'], axis=1).values
X_test  = df[891:].drop(['Survived','PassengerId'], axis=1).values


# (09/12/2017 update) Using NN gives better result than XGBoost and Random Forest do. 

# In[ ]:


# Initialising the NN
model = Sequential()

# layers
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
model.fit(X_train, y_train, batch_size = 32, epochs = 200)


# We can now get the prediction. I got a public score of 0.81339 using the output from my laptop (python 2.7), which is different from what is generated here.

# In[ ]:


y_pred = model.predict(X_test)
y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])

output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})
output.to_csv('prediction-ann.csv', index=False)

