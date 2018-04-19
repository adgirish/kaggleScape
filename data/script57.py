
# coding: utf-8

# # Competition description
# > The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# > 
# > One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# > 
# > In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# # Objectives
# This is my first Kaggle competition and the first project that I will do by myself. I've completed the Andrew Ng course on ML and watched a bunch of videos about numpy, pandas and sci-kit learn. My objectives with this notebook is to keep learning putting into practice what I already "know" as well as what I will learn during participating in this competition. This notebook is highly inspired in other kernels that I saw as well. I hope I can learn writing this kernel.

# **This is a kernel under construction, any tips and comments are appreciated**

# # Index
# #### 0. Setting Environment  (Complete)  
#     0.1 Loading Libraries  
#     0.2 Loading Data  
# #### 1. Preprocessing  (Very complete)  
#     1.1 Missing Values  
#     1.2 Feature Engineering  
#         1.2.1 Family Name  
#         1.2.2 Title  
#         1.2.3 Fare and Age bins  
#         1.2.4 IsChild, IsOld  
#         1.2.5 FamilySize  
#         1.2.6 IsAlone and LargeFamily  
#         1.2.7 Tickets  
#         1.2.8 Cabins  
#     1.3 Converting Data  
#     1.4 Creating Masks  
# #### 2. Visualization  (In Progress)  
#     2.1 Survival rates per features  
#     2.2 Age, Fare, Sex Analysis  
#     2.3 Fare and Class Analysis
#     2.4 Tickets
#     2.5 Correlation  
# #### 3. Machine Learning  (In Progress - Need help with model selection)  
#     3.1 Choosing a model  
#     3.2 Learning  
#     3.4 Predicting  

# # 0. Setting Environment

# ## 0.1 Loading Libraries

# In[ ]:


# Linear Algebra
import numpy as np

# Dataframes and Series
import pandas as pd

# Preprocessing
from sklearn.preprocessing import LabelEncoder

# Model Selection and Learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin 

#Visualization
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 14,10 # Big graphs unless specified
sns.set(color_codes=True) # Set style and color of graphs


# ## 0.2 Loading Data

# I will create **copies** of the input data as I will modify it at cleaning and feature engineering stages. For easy manipulation of train and test sets I will create a list with both (**references** to the actual dataframes).

# In[ ]:


# Raw input dataframes
raw_train = pd.read_csv("../input/train.csv")
raw_test = pd.read_csv("../input/test.csv")

# Copy for the preprocessing
df_train = raw_train.copy()
df_test = raw_test.copy()
df_total = pd.concat([df_train, df_test])

# Easy application of cleaning
data_cleaner = [df_train, df_test]


# In[ ]:


df_train.head()


# # 1. Preprocessing

# We know that the features are:
# * Survived: target binary value
# * PassengerId:  identifier of passengers - can be discarded for the analysis
# * Pclass: ordinal feature that probrably correlates with social class (can be used as is or as dummy variables)
# * Name: text data, we can extract family name and title from here
# * Sex: binary value (male, female)
# * SibSp, Parch: we can use these as they are or create intervals
# * Ticket: can have some hidden importance (maybe correlated with fare or pclass)
# * Fare, Age: continuous values, can be used as is or as intervals
# * Cabin: can be important for the analysis as it correlates with the localization of the passager in the ship
# * Embarked: Where embarked. Categorical variable (C, Q or S)

# We can discard PassengerId since we assume the passengers are sorted at random.

# In[ ]:


for df in data_cleaner:
    df.drop(["PassengerId"], axis=1, inplace=True)


# In[ ]:


# Dataframe information
print('-'*20, 'Train Set')
df_train.info()
print('-'*20, 'Test Set')
df_test.info()


# ## 1.1 Missing Values

# We can see from the data loaded that there is missing values in Age, Fare, Embarked and (more critically) in Cabin features. For Fare and Age it is reasonable to pick the mean or the median (last will be used as it handles outliers better), for Embarked mode will be used. We couldn't use something similar with Cabin as it has so many missing data, so it will be assigned a letter M for missing in the examples without Cabin data for further analysis.

# In[ ]:


# Simple imputation
for df in data_cleaner:
    df["Age"].fillna(df_train.Age.median(), inplace=True)
    df["Fare"].fillna(df_train.Fare.median(), inplace=True)
    df["Embarked"].fillna(df_train.Embarked.mode()[0], inplace=True)
    
    # Cant impute anything because of number of missing values
    df["Cabin"].fillna('M', inplace=True)


# In[ ]:


# Check if there are still any missing value
print('-'*20, 'Train Set')
print(df_train.isnull().any())
print('-'*20, 'Test Set')
print(df_test.isnull().any())


# # 1.2 Feature Engineering

# We can extract title and family names from the Name feature

# In[ ]:


def extract_name(f):
    return f.split(', ')[0]

def extract_title(f):
    return (f.split(", ")[1]).split(".")[0]


# In[ ]:


for df in data_cleaner:
    df["FamilyName"] = df.Name.apply(extract_name)
    df["Title"] = df.Name.apply(extract_title)


# ### 1.2.1 Family Name

# In[ ]:


df_train.FamilyName.value_counts().head(10)


# We can pick categories for names with at least 2 individuals:

# In[ ]:


family_names = df_train.FamilyName

for df in data_cleaner:
    df["FamilyName"] = df.FamilyName.apply(lambda f: f if f in family_names.unique() and
                                     family_names.value_counts()[f] > 1 else 'Unique')


# In[ ]:


df_train['FamilyName'].value_counts()


# In[ ]:


plt.figure(figsize=(14,6))
df_names = df_train.copy()
df_names['UniqueName'] = df_names['FamilyName'] == 'Unique'
sns.barplot("UniqueName", "Survived", data=df_names)
plt.show()


# In[ ]:


# For now I think these numbers has little to contribute
for df in data_cleaner:
    df.drop(["FamilyName"], axis=1, inplace=True)


# ### 1.2.2 Title

# In[ ]:


titles = df_train.Title


# In[ ]:


titles.value_counts()


# We first can group some of the classes togheter and then realize the same analysis as the above.

# In[ ]:


miss = ["Ms", "Mlle"]
mrs = ["Mme"]

for df in data_cleaner:
    df["Title"] = df.Title.apply(lambda f: 'Miss' if f in miss else f)
    df["Title"] = df.Title.apply(lambda f: 'Mrs' if f in mrs else f)


# In[ ]:


titles = df_train.Title

for df in data_cleaner:
    df["Title"] = df.Title.apply(lambda f: f if f in titles.unique() and
                                     titles.value_counts()[f] > 10 else 'Rare')


# In[ ]:


df_train['Title'].value_counts()


# In[ ]:


plt.figure(figsize=(14,6))
sns.barplot('Title', 'Survived', data=df_train)
plt.show()


# From Mr, Mrs and Miss we see some intersection with the Sex feature. Master seems to indicate correlation with age, so maybe can be used to impute age.

# ### 1.2.3 Age and Fare bins

# In[ ]:


plt.subplot(211)
sns.distplot(df_train['Age'][df_train.Survived == 1], color='b')
sns.distplot(df_train['Age'][df_train.Survived == 0], color='r')
plt.legend(['Survived', 'Died']), plt.xticks(range(0,92,2))
plt.subplot(212)
sns.distplot(df_train['Fare'][df_train.Survived == 1], color='b')
sns.distplot(df_train['Fare'][df_train.Survived == 0], color='r')
plt.legend(['Survived', 'Died'])
plt.show()


# For age we can create equally spaced bins, but for fare it makes more sense to divide into parts with same amount of individuals:

# In[ ]:


_, fare_bins = pd.qcut(df_total['Fare'], 4, retbins=True)
fare_bins[0] -= 0.001 # dirty fix
_, age_bins = pd.cut(df_total['Age'], 5, retbins=True)
for df in data_cleaner:
    df["FareBin"] = pd.cut(df['Fare'], fare_bins)
    df["AgeBin"] = pd.cut(df['Age'], age_bins)


# ### 1.2.4 IsChild

# In[ ]:


plt.figure(figsize=(14,5))
sns.distplot(df_train['Age'][df_train.Survived == 1], range(26), color='b')
sns.distplot(df_train['Age'][df_train.Survived == 0], range(26), color='r')
plt.xlim((0,26)), plt.xticks(range(0,26,1))
plt.legend(['Survived', 'Died'])
plt.show()


# From the plot 16 seems to be a good point for determining as a child

# In[ ]:


for df in data_cleaner:
    df['IsChild'] = df['Age'] < 16


# In[ ]:


sns.barplot('IsChild', 'Survived', data=df_train)
plt.show()


# ### 1.2.5 Family Size

# In[ ]:


plt.subplot(121)
sns.barplot('SibSp', 'Survived', data=df_train)
plt.subplot(122)
sns.barplot('Parch', 'Survived', data=df_train)
plt.show()


# We can easily create a new feature called family size for Parch + SibSp + Himself/Herself

# In[ ]:


for df in data_cleaner:
    df["FamilySize"] = df.Parch + df.SibSp +1


# In[ ]:


sns.distplot(df_train['FamilySize'][df_train.Survived == 1], range(26), color='b')
sns.distplot(df_train['FamilySize'][df_train.Survived == 0], range(26), color='r')
plt.legend(['Survived', 'Died'])
plt.show()


# We can also define binary variables for alone individuals and individuals with a large family as these seems to be related with less chance of surviving.

# ### 1.2.6 - IsAlone and LargeFamily

# In[ ]:


for df in data_cleaner:
    df["IsAlone"] = df["FamilySize"] == 1
    df["LargeFamily"] = df["FamilySize"] >= 5


# In[ ]:


plt.subplot(121)
sns.barplot('IsAlone', 'Survived', data=df_train)
plt.subplot(122)
sns.barplot('LargeFamily', 'Survived', data=df_train)
plt.show()


# ### 1.2.7 Exploring Tickets

# In[ ]:


ticket_values = pd.concat([df_train['Ticket'], df_test['Ticket']]).value_counts()
ticket_values.head()


# In[ ]:


for df in data_cleaner:
    df['N_ticket'] = df['Ticket'].apply(lambda f: ticket_values[f])


# In[ ]:


sns.distplot(df_train['N_ticket'][df_train.Survived == 1], range(26), color='b')
sns.distplot(df_train['N_ticket'][df_train.Survived == 0], range(26), color='r')
plt.legend(['Survived', 'Died'])
plt.show()


# ### 1.2.8 Exploring Cabin

# I will limit my analysis to the letter of the Cabin.

# In[ ]:


for df in data_cleaner:
    df["Cabin"] = df["Cabin"].apply(lambda f: f[0])


# Let's see if there is hiding information in the missingness of Cabin

# In[ ]:


# Mask for cabin missing
cabin_M = df_train["Cabin"] == 'M'
cabin_Mn = df_train["Cabin"] != 'M'

plt.subplot(211)
sns.barplot("Cabin", "Survived", data=df_train)
plt.subplot(223)
plt.bar('Missing', df_train.Cabin[cabin_M].count())
plt.bar('Not Missing', df_train.Cabin[cabin_Mn].count())
plt.subplot(224)
df_cabin = df_train.copy()
df_cabin["Cabin_Missing"] = df_train["Cabin"].apply(lambda f: 1 if f == 'M' else 0)
sns.barplot("Cabin_Missing", "Survived", data=df_cabin)
plt.show()


# As missing the value seems to be related to surviving, let's create a feature for this as it may be more relevant than the Cabin itself as there is such a large number of missing Cabins.

# In[ ]:


for df in data_cleaner:
    df['CabinMissing'] = df.Cabin == 'M'


# ## 1.3 Cleaning

# In[ ]:


# Double check missing values
for df in data_cleaner:
    print('-'*20)
    print(df.isnull().any())


# Create code variables for the categorical data

# In[ ]:


# Encode categorical data
label = LabelEncoder()
for df in data_cleaner:    
    df['Sex_Code'] = label.fit_transform(df['Sex'])
    df['Embarked_Code'] = label.fit_transform(df['Embarked'])
    df['Title_Code'] = label.fit_transform(df['Title'])
    df['AgeBin_Code'] = label.fit_transform(df['AgeBin'])
    df['FareBin_Code'] = label.fit_transform(df['FareBin'])
    df['Cabin_Code'] = label.fit_transform(df["Cabin"])
    
     # Pclass can be viewed as categorical or simply an integer variable, which is best?
    df['Pclass'] = df.Pclass.astype('category')
    
    # Bool to int
    df['IsChild'] = df['IsChild'].apply(int)
    df['IsAlone'] = df['IsAlone'].apply(int)
    df['LargeFamily'] = df['LargeFamily'].apply(int)
    df['CabinMissing'] = df['CabinMissing'].apply(int)


# In[ ]:


df_train.columns


# In[ ]:


# define masks

# target variable
target = ['Survived']

# pretty categorical names
data_pretty = ['Age', 'Pclass', 'Title', 'Sex', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked',
                'FamilySize', 'FareBin', 'AgeBin', 'IsAlone', 'IsChild', 'LargeFamily',
                'N_ticket', 'CabinMissing']

# continuous and integer variables
data_numbers = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'N_ticket']

# age and fare bins
data_bins = ['AgeBin_Code', 'FareBin_Code']

# categorical variables
data_cat = ['Pclass', 'Title', 'Sex', 'Cabin', 'Embarked' 'IsAlone',
            'CabinMissing', 'IsChild', 'LargeFamily']

# encoded categorical variables
data_code = ['Pclass', 'Title_Code', 'Sex_Code', 'Cabin_Code', 'Embarked_Code',
             'IsAlone', 'CabinMissing', 'IsChild', 'LargeFamily']


# Create a dummy df of the categorical data

# In[ ]:


dummy_train = pd.get_dummies(df_train[data_pretty + target])
dummy_test = pd.get_dummies(df_test[data_pretty])

# mask for dummy variables
dummy_labels = dummy_test.columns.tolist()

# easily assign things to train and testing
dummy = [dummy_train, dummy_test]


# In[ ]:


dummy_train.head()


# In[ ]:


df_train[data_numbers].describe()


# # 2. Visualization

# ## 1.2 Surviving rates per feature

# In[ ]:


# Histogram of each feature count and survival count
df_train[data_numbers + data_code].hist(color='g')
df_train[data_numbers + data_code][df_train["Survived"] == 1].hist()
plt.show()


# In[ ]:


# Who survived per group (Pivot tables)
for x in data_pretty:
    if df_train[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(df_train[[x, target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')


# In[ ]:


# % Survivors per groups
plt.figure(figsize=(14,10))
plt.suptitle("Percentage of survivors per group", fontsize=18)
plt.subplot(231)
sns.barplot('Sex', 'Survived', data=df_train)
plt.subplot(232)
sns.barplot('AgeBin_Code', 'Survived', data=df_train)
plt.subplot(233)
sns.barplot('Embarked', 'Survived', data=df_train)
plt.subplot(234)
sns.barplot('FareBin_Code', 'Survived', data=df_train)
plt.subplot(235)
sns.barplot('FamilySize', 'Survived', data=df_train)
plt.subplot(236)
sns.barplot('Pclass', 'Survived', data=df_train)
plt.show()


# ## 2.2 Sex, Age and Fare analysis

# In[ ]:


# Violin plots for sex-age-fare analysis

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,10))
sns.violinplot(x = 'Sex', y = 'Age', hue = 'Survived', data = df_train, split = True, ax = axis1, palette="Set1")
axis1.set_title('Sex vs Age Survival Comparison')
sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = df_train, split = True, ax = axis2, palette="Set1")
axis2.set_title('Pclass vs Age Survival Comparison')
sns.violinplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = df_train, split = True, ax = axis3, palette="Set1")
axis3.set_title('Sex vs Fare Survival Comparison')
fig.suptitle("Sex, Age, Fare Analysis", fontsize=18)
plt.show()


# In[ ]:


survived = df_train['Survived'] == 1
died = df_train['Survived'] == 0
sns.factorplot('Sex', 'Age', 'Survived', data=df_train, col='Pclass')
sns.factorplot('Sex', 'Fare', 'Survived', data=df_train, col='Pclass')
plt.show()


# # 2.3 Fare analysis

# In[ ]:


# Embarked - Fare
sns.factorplot('Embarked', 'Fare', 'Survived', data=df_train, col='Sex')


# In[ ]:


# Fare analysis
plt.suptitle("Fare Analysis", fontsize=18)
plt.subplot(121)
sns.boxplot('Pclass', 'Fare', 'Survived', df_train, orient='v')
plt.ylim((0,150))
ax1 = plt.subplot(122)
tab = pd.crosstab(df_train['Pclass'], df_train['FareBin'])
tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax=ax1)
plt.xlabel('Pclass')
plt.ylabel('FareBin')
plt.legend(frameon=True)

plt.show()


# ## 2.4 Tickets

# In[ ]:


sns.factorplot('N_ticket', 'Survived', data=df_train, col='Sex')
sns.factorplot('N_ticket', 'Survived', data=df_train, col='Pclass')
plt.show()


# ## 3.5 Correlation of features

# In[ ]:


# Correlation Matrix
df_train.corr()


# In[ ]:


# Correlation Heatmap

def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    colormap = sns.color_palette("coolwarm", 100)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(df_train)


# # 3. Learning: Logistic Regression

# In[ ]:


# Check columns
for e in dummy_train.columns:
    if e not in dummy_test.columns:
        print(e)
print('-'*10)
for e in dummy_test.columns:
    if e not in dummy_train.columns:
        print(e)


# In[ ]:


dummy_test['Cabin_T'] = 0 #dirty fix


# ## 3.1 Model Selection

# In[ ]:


dummy_test.columns


# In[ ]:


# Features mask (Selected by intuition)

features = ['IsChild',
            'IsAlone',
            'LargeFamily',
            'SibSp', 'Parch',
            'FamilySize',
            'N_ticket',
            'Title_Rare',
            'Sex_female',
            'Age',
            'Fare',
            "Pclass_1", "Pclass_2",
            "Embarked_C", "Embarked_S",
            'AgeBin_(16.136, 32.102]', 'AgeBin_(32.102, 48.068]',
            'AgeBin_(48.068, 64.034]', 'AgeBin_(64.034, 80.0]',
            "FareBin_(7.896, 14.454]", "FareBin_(14.454, 31.275]", "FareBin_(31.275, 512.329]",
            'CabinMissing']

features1 = ['IsChild', 'IsAlone', "LargeFamily", 'SibSp', 'Parch', 'Sex_female',"Pclass_1", "Pclass_2",
            "Embarked_C", "Embarked_S", 'CabinMissing', 'Age', 'Fare']

features2 = ['IsChild', 'IsAlone', 'SibSp', 'Parch', 'Sex_female',"Pclass_1", "Pclass_2",
            "Embarked_C", "Embarked_S", 'CabinMissing', 'Age', "LargeFamily",
            "FareBin_(7.896, 14.454]", "FareBin_(14.454, 31.275]", "FareBin_(31.275, 512.329]"]

features3 = ['IsChild', 'IsAlone', 'SibSp', 'Parch', 'Sex_female',"Pclass_1", "Pclass_2",
            "Embarked_C", "Embarked_S", 'CabinMissing', "LargeFamily"]

features4 = ['Sex_female',"Pclass_1", "Pclass_2", 'IsChild', 'FamilySize', 'Fare',
            "Embarked_C", "Embarked_S", 'CabinMissing', "LargeFamily"]

feature_options = [features1, features2, features3, features4]


# In[ ]:


# Create train and test set
train_set = dummy_train[features + target].copy()
test_set = dummy_test[features].copy()

# Scaling
for e in train_set.columns:
    if e in data_numbers:
        test_set[e] = StandardScaler().fit_transform(test_set[e].values.reshape(-1,1)).ravel()


# In[ ]:


from sklearn.linear_model import LogisticRegressionCV
import statsmodels.api as sm
from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

for f in feature_options:
    print("\n", "Features: ", f)
    logit_model=sm.Logit(train_set[target], train_set[f])
    result=logit_model.fit()
    print(result.summary())
    print("-"*20)


# In[ ]:


features_ = features2

logit_model=sm.Logit(train_set[target], train_set[features_])
result=logit_model.fit()
print(result.summary())
print("-"*20)


# In[ ]:


# Another approuch
logreg = LogisticRegression()
rfe = RFECV(logreg, 1, 10, verbose=3)
X_rfe = rfe.fit_transform(train_set[features], train_set[target])


# In[ ]:


print(rfe.support_)
print(rfe.ranking_)


# In[ ]:


features_rfe = train_set.iloc[:,[0,2,8,11,12,14,17,18,20,21,22]].columns.tolist()
train_set[features_rfe].head()


# In[ ]:


logit_model=sm.Logit(train_set[target], train_set[features_rfe])
result=logit_model.fit()
print(result.summary())
print("-"*20)


# In[ ]:


logit_model=sm.Logit(train_set[target], train_set[features2])
result=logit_model.fit()
print(result.summary())
print("-"*20)


# ## 3.2 Learning

# In[ ]:


X, X_test, y, y_test = train_test_split(train_set[features_].values, train_set[target].values,
                                        test_size=0.25, stratify=train_set[target].values, random_state=42)
X.shape, X_test.shape, y.shape, y_test.shape


# In[ ]:


params = {
    #'polynomialfeatures__degree': [1, 2, 3],
    'classification__penalty': ['l1', 'l2'],
    'classification__C': np.linspace(0.05,50,100),
    'classification__random_state': [42]
}

pipe = Pipeline([
    #('polynomialfeatures', PolynomialFeatures()),
    ('classification', LogisticRegression())
])
    
grid = RandomizedSearchCV(pipe, params, 100, n_jobs=-1, cv=10, verbose=3, random_state=42)


# In[ ]:


grid.fit(X, y.ravel())


# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# In[ ]:


grid.score(X_test, y_test)


# In[ ]:


lg = grid.best_estimator_


# ## 3.3 Predicting

# In[ ]:


lg.fit(train_set[features_], train_set[target])


# In[ ]:


pred = lg.predict(test_set[features_])


# In[ ]:


df_pred = pd.concat([raw_test["PassengerId"], pd.Series(pred, name="Survived")], axis=1)


# In[ ]:


df_pred['Survived'] = df_pred.Survived.astype(int)
df_pred.head()


# In[ ]:


df_pred.to_csv("out.csv", index=False)

