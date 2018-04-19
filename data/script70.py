
# coding: utf-8

# # Introduction
# #### *This notebook describes and implements a basic approach to solving the Titanic Survival Prediction problem. The prediction is made using a Random Forest Classifier.*

# ## 1. Exploring training and test sets

# First, load required packages.

# In[1]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
plt.style.use('ggplot')


# Read training and test sets. Both datasets will be used in exploring and predicting.

# In[2]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[3]:


train.sample(frac=1).head(3)


# In[4]:


test.sample(frac=1).head(3)


# ## 2. Exploring missing data

# Looks like there are missing (NaN) values among both datasets.

# In[5]:


train.info()


# In[6]:


test.info()


# ### Non-numeric data

# *Cabin* column stores quite a lot of different qualitative values and has a relatively large amount of missing data.

# In[7]:


missing_val_df = pd.DataFrame(index=["Total", "Unique Cabin", "Missing Cabin"])
for name, df in zip(("Training data", "Test data"), (train, test)):
    total = df.shape[0]
    unique_cabin = len(df["Cabin"].unique())
    missing_cabin = df["Cabin"].isnull().sum()
    missing_val_df[name] = [total, unique_cabin, missing_cabin]
missing_val_df


# We shall remove _Cabin_ columns from our dataframes.
# 
# Also, we can exclude _PassengerId_ from the training set, since IDs are unnecessary for classification.

# In[8]:


train.drop("PassengerId", axis=1, inplace=True)
for df in train, test:
    df.drop("Cabin", axis=1, inplace=True)


# Fill in missing rows in _Embarked_ column with __S__ (Southampton Port), since it's the most frequent.

# In[9]:


non_empty_embarked = train["Embarked"].dropna()
unique_values, value_counts = non_empty_embarked.unique(), non_empty_embarked.value_counts()
X = np.arange(len(unique_values))
colors = ["brown", "grey", "purple"]

plt.bar(left=X,
        height=value_counts,
        color=colors,
        tick_label=unique_values)
plt.xlabel("Port of Embarkation")
plt.ylabel("Amount of embarked")
plt.title("Bar plot of embarked in Southampton, Queenstown, Cherbourg")


# ### Quantitative data

# Consider the distributions of passenger ages and fares (excluding NaN values).

# In[10]:


survived = train[train["Survived"] == 1]["Age"].dropna()
perished = train[train["Survived"] == 0]["Age"].dropna()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(12, 6)
fig.subplots_adjust(hspace=0.5)
ax1.hist(survived, facecolor='green', alpha=0.75)
ax1.set(title="Survived", xlabel="Age", ylabel="Amount")
ax2.hist(perished, facecolor='brown', alpha=0.75)
ax2.set(title="Dead", xlabel="Age", ylabel="Amount")


# In[11]:


survived = train[train["Survived"] == 1]["Fare"].dropna()
perished = train[train["Survived"] == 0]["Fare"].dropna()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(12, 8)
fig.subplots_adjust(hspace=0.5)
ax1.hist(survived, facecolor='darkgreen', alpha=0.75)
ax1.set(title="Survived", xlabel="Age", ylabel="Amount")
ax2.hist(perished, facecolor='darkred', alpha=0.75)
ax2.set(title="Dead", xlabel="Age", ylabel="Amount")


# We can clean up *Age* and *Fare* columns filling in all of the missing values with **median **of all values in the training set.

# In[12]:


for df in train, test:
    df["Embarked"].fillna("S", inplace=True)
    for feature in "Age", "Fare":
        df[feature].fillna(train[feature].mean(), inplace=True)


# _Ticket_ column has a lot of various values. It will have no significant impact on our ensemble model.

# In[13]:


for df in train, test:
    df.drop("Ticket", axis=1, inplace=True)


# ## 3. Feature engineering

# ### Converting non-numeric columns

# All of the non-numeric features except _Embarked_ aren't particularly informative.
# 
# We shall convert _Embarked_ and _Sex_ columns to numeric because we can't feed non-numeric columns into a Machine Learning algorithm.

# In[14]:


for df in train, test:
    df["Embarked"] = df["Embarked"].map(dict(zip(("S", "C", "Q"), (0, 1, 2))))
    df["Sex"] = df["Sex"].map(dict(zip(("female", "male"), (0, 1))))


# ### Generating new features

# $SibSp$ + $Parch$ + $1$ gives the total number of people in a family.

# In[15]:


for df in train, test:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1


# Extract the passengers' titles (Mr., Mrs., Rev., etc.) from their names.

# In[16]:


for df in train, test:
    titles = list()
    for row in df["Name"]:
        surname, title, name = re.split(r"[,.]", row, maxsplit=2)
        titles.append(title.strip())
    df["Title"] = titles
    df.drop("Name", axis=1, inplace=True)


# In[17]:


title = train["Title"]
unique_values, value_counts = title.unique(), title.value_counts()
X = np.arange(len(unique_values))

fig, ax = plt.subplots()
fig.set_size_inches(18, 10)
ax.bar(left=X, height=value_counts, width=0.5, tick_label=unique_values)
ax.set_xlabel("Title")
ax.set_ylabel("Count")
ax.set_title("Passenger titles")
ax.grid(color='g', linestyle='--', linewidth=0.5)


# Looks like some titles are very rare. Let's map them into related titles.

# In[18]:


for df in train, test:
    for key, value in zip(("Mr", "Mrs", "Miss", "Master", "Dr", "Rev"),
                          np.arange(6)):
        df.loc[df["Title"] == key, "Title"] = value
    df.loc[df["Title"] == "Ms", "Title"] = 1
    for title in "Major", "Col", "Capt":
        df.loc[df["Title"] == title, "Title"] = 6
    for title in "Mlle", "Mme":
        df.loc[df["Title"] == title, "Title"] = 7
    for title in "Don", "Sir":
        df.loc[df["Title"] == title, "Title"] = 8
    for title in "Lady", "the Countess", "Jonkheer":
        df.loc[df["Title"] == title, "Title"] = 9
test["Title"][414] = 0


# Nominal features of our model.

# In[19]:


nominal_features = ["Pclass", "Sex", "Embarked", "FamilySize", "Title"]
for df in train, test:
    for nominal in nominal_features:
        df[nominal] = df[nominal].astype(dtype="category")


# Finally, we get

# In[20]:


train.sample(frac=1).head(10)


# ## 4. Prediction

# Choose the most informative predictors and randomly split the training data.

# In[21]:


from sklearn.model_selection import train_test_split

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch",
              "Fare", "Embarked", "FamilySize", "Title"]
X_train, X_test, y_train, y_test = train_test_split(train[predictors], train["Survived"])


# Build a Random Forest model from the training set and evaluate the mean accuracy on the given test set.

# In[22]:


forest = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth=5,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)
forest.fit(X_train, y_train)
print("Random Forest score: {0:.2}".format(forest.score(X_test, y_test)))


# Examine the feature importances.

# In[23]:


plt.bar(np.arange(len(predictors)), forest.feature_importances_)
plt.xticks(np.arange(len(predictors)), predictors, rotation='vertical')


# Pick the best features and make a submission.

# In[24]:


predictors = ["Title", "Sex", "Fare", "Pclass", "Age", "FamilySize"]
clf = RandomForestClassifier(n_estimators=100,
                             criterion='gini',
                             max_depth=5,
                             min_samples_split=10,
                             min_samples_leaf=5,
                             random_state=0)
clf.fit(train[predictors], train["Survived"])
prediction = clf.predict(test[predictors])

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction})
submission.to_csv("submission.csv", index=False)

