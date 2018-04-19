
# coding: utf-8

# In[ ]:


from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# # Deep Sea Dive into the Top 10% - Machine Learning Survival Guide
# ### Arun Godwin Patel

# ***

# # Introduction
# 
# At the start of any Data Science project, understanding of the context and case at hand is key. In an enterprise setting, this would be business and domain understanding.
# 
# Before I started working with the data, I decided to do some research on the Titanic.

# In[ ]:


from IPython.display import Image
Image(filename='../input/titanic-images/titanic.png')


#  ### Here is it, the "Unsinkable" RMS Titanic in all its glory.

# In[ ]:


Image(filename='../input/titanic-images/thecause.png')


# ### The facts above outline some interesting theories as to why the disaster occured. Was it a coincidence? Was somebody to blame?...
# 
# ### Below is a timeline based on the local time of the ship, from construction to survival - for some. The speed at which the disaster developed once they were hit,  was astonishing.

# In[ ]:


Image(filename='../input/titanic-images/timeline.png')


# In[ ]:


Image(filename='../input/titanic-images/passengers.png')


# - From the information presented above, it is clear that there are certain factors that may have caused the disaster. However, there are also clearly factors that helped to improve a passengers chance of survival, such as **social standing** and **gender**.
# - Hence, I have created this kernel to demonstrate a **'Deep Sea Dive'** into the range of **Machine Learning** techniques available to **predict survival**. I will walk you through the entire Data Science process, providing a wide variety of techniques with explanation, so that you can follow along too.

# ***

# # Content
# 
# 1. **Import packages**
# 2. **Load data**
# 3. **Feature Analysis**
#     - 3.1 - Identify and treat missing values
#         - 3.1.1 - Age
#         - 3.1.2 - Cabin
#         - 3.1.3 - Embarked
#         - 3.1.4 - Fare
#     - 3.2 - Numerical values
#         - 3.2.1 - Age
#         - 3.2.2 - Fare
#         - 3.2.3 - Parch
#         - 3.2.4 - Pclass
#         - 3.2.5 - SibSp
#     - 3.3 - Categorical values
#         - 3.3.1 - Cabin
#         - 3.3.2 - Embarked
#         - 3.3.3 - Sex
# 4. **Feature Engineering**
#     - 4.1 - Age
#     - 4.2 - Cabin
#     - 4.3 - Fare
#     - 4.4 - Title
#     - 4.5 - Family Size
#     - 4.6 - Sex
#     - 4.7 - Ticket
#     - 4.8 - Correlation of all features
# 5. **Modeling**
#     - 5.1 - Preparation of data
#     - 5.2 - Model comparison
#         - 5.2.1 - Algorithm A-Z
#         - 5.2.2 - Performance comparison
#     - 5.3 - Model selection
#     - 5.4 - Model feature reduction
#     - 5.5 - Optimisation of selected models
#     - 5.6 - Ensemble voting
#     - 5.7 - Output predictions
# 6. **Conclusion**

# ***

# # 1. 
# ## Import packages

# In[ ]:


# This first set of packages include Pandas, for data manipulation, numpy for mathematical computation
# and matplotlib & seaborn, for visualisation.
import pandas as pd
import numpy as np
from numpy import sort
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', palette='deep')
print('Data Manipulation, Mathematical Computation and Visualisation packages imported!')

# Next, these packages are from the scikit-learn library, including the algorithms I plan to use.
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
print('Algorithm packages imported!')

# These packages are also from scikit-learn, but these will be used for model selection and cross validation.
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
print('Model Selection packages imported!')

# Once again, these are from scikit-learn, but these will be used to assist me during feature reduction.
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
print('Feature Reduction packages imported!')

# Finally, a metrics package from scikit-learn to be used when comparing the results of the cross validated models. 
from sklearn import metrics
print('Cross Validation Metrics packages imported!')

# Set visualisation colours
mycols = ["#76CDE9", "#FFE178", "#9CE995", "#E97A70"]
sns.set_palette(palette = mycols, n_colors = 4)
print('My colours are ready! :)')

# Ignore future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
print('Future warning will be ignored!')


# ***

# # 2. 
# ## Load data
# 
# - The Pandas packages helps us work with our datasets. We start by acquiring the training and testing datasets into Pandas DataFrames. 
# - We also combine these datasets to run certain operations on both datasets together.

# In[ ]:


# Read the raw training and test data into DataFrames.
raw_train = pd.read_csv('../input/titanic/train.csv')
raw_test = pd.read_csv('../input/titanic/test.csv')

# Append the test dataset onto train to allow for easier feature engineering across both datasets.
full = raw_train.append(raw_test, ignore_index=True)

print('Datasets \nfull: ', full.shape, '\ntrain: ', raw_train.shape, '\ntest: ', raw_test.shape)
full.head(3)


# - Here we can see that we have 891 records to train our models on and 418 to output predictions for. Due to the small size of the dataset, we will use techniques to attempt to ensure that we can build models in a robust way.
# 
# - We also have 12 columns:
#     - **Age** - The Age of the passenger in years.
#     - **Cabin** - The Cabin number that the passenger was assigned to. If NaN, this means they had no cabin and perhaps were not assigned one due to the cost of their ticket.
#     - **Embarked** - Port of embarkation (S = Southampton, C = Cherbourg, Q = Queenstown).
#     - **Fare** - The fare of the ticket purchased by the passenger.
#     - **Name** - Full name and title of the passenger.
#     - **Parch** - Number of parents and children associated with the passenger aboard.
#     - **PassengerId** - Unique ID assigned to each passenger.
#     - **Pclass** - Class of ticket purchased (1 = 1st class, 2 = 2nd class, 3 = 3rd class).
#     - **Sex** - Gender of the passenger.
#     - **SibSp** - Number of siblings and spouses associated with the passenger aboard.
#     - **Survived** - Survival flag of passenger (1 = Survived, 0 = did not survive). Note that this column is only present within the training dataset, it is missing from the test dataset because this is what we are going to predict.
#     - **Ticket** - Ticket number.
#     
# ## Note:
# 
# **Pclass** can be viewed as a proxy for socio-economic status (SES):
# - 1st = Upper class
# - 2nd = Middle class
# - 3rd = Lower class
# 
# **Age**: Age is fractional if the passenger is less than 1 year old. If the age is estimated, is it in the form of xx.5
# 
# **SibSp & Parch**: These 2 columns represent family relations, which are defined in the following ways:
# - Sibling = Brother, sister, stepbrother, stepsister.
# - Spouse = Husband, wife (mistresses and fiancés were ignored).
# - Parent = Mother, father.
# - Child = daughter, son, stepdaughter, stepson.
# - Some children travelled only with a nanny, therefore Parch = 0 for them.

# ***

# # 3. 
# ## Feature Analysis
# ### 3.1 - Identify and treat missing values
# 
# ***Why are missing values so important?***
# 
# First of all, some algorithms do not like missing values. Some are capable of handling them, but others are not. Therefore since we are using a variety of algorithms, it's best to treat them in an appropriate way.
# 
# **If you have missing values, you have two options**:
# - Delete the entire row
# - Fill the missing entry with an imputed value
# 
# Unfortunately, our dataset for training is very small. Therefore we want to preserve as much of the data as we can, so that the algorithms can learn from more examples. This means that deleting entire rows is not a suitable option. Hence, we must fill these missing values.
# 
# 
# A missing value is an entry in a column that has no assigned value. This can mean multiple things:
# - A missing value may be the **result of an error during the production of the dataset**. This could be a human error, or machinery error depending on where the data comes from. 
#     - In the context of the Titanic dataset, it is likely that this data was originally recorded on pen and paper, since we are talking about the year 1912. Therefore, perhaps the missing values were the result of information being lost over time? Or maybe the clerk that kept hold of this data, did not record it properly as the ship was being boarded?
# - A missing value in some cases, may just mean a that a **'zero'** should be present. In which case, it can be replaced by a 0.
# - However, missing values represent no information. Therefore, **does the fact that you don't know what value to assign an entry, mean that filling it with a '0' is always a good fit?** 
# 
# Missing values must be treated with a lot of care, as whatever you fill it with comes with assumptions. These assumptions must be outlined and taken into consideration when working with the data.
# 
# To decide how we should fill these values, let's see which columns contain them.

# In[ ]:


full.isnull().sum()


# So we have missing values for **Age**, **Cabin**, **Embarked** and **Fare**.
# The 418 missing values for **Survived** represent the 418 rows from the test dataset that we appended to the training data.
# 
# The following table presents some descriptive statistics to help us fill the missing values.

# In[ ]:


full.describe(include='all')


# And the following correlation matrix can give us guidance on how to fill the numerical columns with missing values.

# In[ ]:


# Correlation matrix between numerical values (SibSp, Parch, Age and Fare) with Survived 
plt.subplots(figsize=(20, 15))
g = sns.heatmap(full[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "YlGnBu", linewidths = 1)


# ### 3.1.1 - Age

# From the descriptive statistics above we can see that:
# - Age has a mean of approx. 29
# - Standard deviation of approx. 14
# - Minimum of approx. 0.17
# - Maximum of 80
# 
# So now we want to find a strategy for imputing the missing values of Age. Looking at the correlation matrix, we can see that Age correlates highly with:
# - **Fare**
# - **Parch**
# - **SibSp**
# 
# Hence, we will use these features to impute the missing values of Age. We will impute the values with the median age of similar rows according to these 3 features. If there are no similar rows, the missing value will be filled with the median age of the dataset.

# In[ ]:


# Creates a list of all the rows with missing values for age, ordered by Index
missing_age = list(full["Age"][full["Age"].isnull()].index)

# Using a for loop to iterate through the list of missing values, and filling each entry with the imputed value
for i in missing_age:
    age_med = full["Age"].median() # median age for entire dataset
    age_pred = full["Age"][((full['SibSp'] == full.iloc[i]["SibSp"]) & (full['Parch'] == full.iloc[i]["Parch"]) & (full['Pclass'] == full.iloc[i]["Pclass"]))].median() # median age of similar passengers
    if not np.isnan(age_pred):
        full['Age'].iloc[i] = age_pred # if there are similar passengers, fill with predicted median
    else:
        full['Age'].iloc[i] = age_med # otherwise, fill with median of entire dataset

print('Number of missing values for Age: ', full['Age'].isnull().sum())


# **Most statistical models rely on a normal distribution**, a distribution that is symmetric and has a characteristic bell shape.
# 
# A machine learning algorithm doesn’t need to know beforehand the type of data distribution it will work on, but learns it directly from the data used for training. In this kernel, we are using K-Nearest-Neighbours which is based on a distance measure. This is quite sensitive to the scale of numeric values provided. Therefore, in order for the algorithm to converge faster or to provide a more exact solution, I have investigated the distribution.
# 
# However, even if transforming the actual distributions isn’t necessary for a machine learning algorithm to work properly, it can still be beneficial for these reasons:
# 
# - To make the **cost function minimize** better the error of the predictions
# - To make the algorithm **converge properly and faster**
# 
# Looking at the following distribution plot, we can see that it follows a fairly normal distribution, so we don't need to rescale or transform this column.

# In[ ]:


plt.subplots(figsize=(15, 10))
g = sns.distplot(full["Age"], color="#76CDE9", label="Skewness : %.2f"%(full["Age"].skew()))
g = g.legend(loc="best");


# ### 3.1.2 - Cabin
# 
# Cabin is an interesting feature, because Cabin may act as a proxy to many things such as:
# - Ticket fare
# - Social-economic status
# - Location on board, perhaps some people may have been located in a beneficial position for escape
# 
# However, we have a lot of missing values, meaning that as it is right now, it will be unable to tell us much about Survival.
# 
# To fill these missing values with the exact Cabin numbers would be almost impossible. Firstly because this information is unknown, and it was also take a long time to try and figure out likely Cabin's for each passenger. 
# 
# Therefore a trade-off is necessary. The trade off I went for was to **fill the missing values with 'X'**, to indicate that this passenger was not assigned to a Cabin. 
# 
# For the remainder of the present values, we have a lot of categories. Each value for the cabin contains a letter, followed by a number, as seen below. Therefore, I will take simply the first letter from the cabin, to create more concise categories within this column.

# In[ ]:


# Crosstab to show the number of unique categories within the Cabin feature
pd.crosstab(full['Sex'], full['Cabin'])


# In[ ]:


# Cabin
full["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in full['Cabin'] ])
full.head()


# ### 3.1.3 - Embarked
# 
# From the missing values table above, we can see that we only have 2 missing values for Embarked. Therefore, since we can from the descriptive statistics that "Southampton" is the most frequent port of embarkation in the dataset, **I will fill the missing values for this column with "S".**

# In[ ]:


#  Filling the missing values in Embarked with 'S'
full['Embarked'].fillna("S", inplace = True)

# full = pd.get_dummies(full, columns = ["Cabin"], prefix="Cabin")
print('Number of missing values for Embarked: ', full['Embarked'].isnull().sum())


# ### 3.1.4 - Fare
# 
# Similarly to Embarked, Fare has a very low number of missing values, in this case 1. Therefore **I will fill the missing values for this column with the median.**

# In[ ]:


# Fill Fare missing values with the median value
full["Fare"] = full["Fare"].fillna(full["Fare"].median())

# full = pd.get_dummies(full, columns = ["Cabin"], prefix="Cabin")
print('Number of missing values for Fare: ', full['Fare'].isnull().sum())


# Now I want to check the distribution of this feature, since it is the only other continuous numerical feature along with Age.

# In[ ]:


plt.subplots(figsize=(15, 10))
g = sns.distplot(full["Fare"], color="#76CDE9", label="Skewness : %.2f"%(full["Fare"].skew()))
g = g.legend(loc="best");


# - Here we can see something very different to what we saw with Age. We can see that the distribution for Fare is highly **positively skewed**, which means that the most common value is always less than the mean and the median. From the distribution plot, we can see that most of the values are concentrated to the left, but we have several large values that are 'tailing' the distribution. 
# 
# - This can also be seen in the descriptive statistics, where we can see the Lower Quartile, Median and Upper Quartile are fairly low, but the maximum value is very high.
# 
# - When you predict values and some of the values are too extreme with respect to the majority of values, you can apply transformations that tend to shorten the distance between values. So we want to transform this feature to minimise the weight of any **extreme cases**. 
# 
# - When trying to minimise the error, the algorithm therefore won't focus too much on the extreme values but it will obtain a generalised solution. The **logarithmic** transformation is usually a good choice and this requires positive values, so this is fine.

# In[ ]:


# Apply log to Fare to reduce skewness distribution
full["Fare"] = full["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

plt.subplots(figsize=(15, 10))
g = sns.distplot(full["Fare"], color="#76CDE9", label="Skewness : %.2f"%(full["Fare"].skew()))
g = g.legend(loc="best")


# As you can see, the skewness is much lower now and it follows a more closely now a normal distribution.
# 
# ### 3.2 - Numerical values
# 
# Now I will analyse the numerical features in the dataset. This exploratory analysis will help me to construct an idea of which features to create in the next stage, and also whether to create categorical or dummy variables from these features.
# 
# ### 3.2.1 - Age
# 
# - From the correlation matrix shown above, with all numerical features, only Fare seems to have a significant correlation with Survival.
# 
# - However, there are many subpopulations and features that we can engineer from these existing features, which may reveal some more significant correlations.
# 
# - First, I will investigate Age against Survival.

# In[ ]:


# Age distibution vs Survived
plt.subplots(figsize=(15, 10))
g = sns.kdeplot(full["Age"][(full["Survived"] == 0)], color="#76CDE9",  shade = True)
g = sns.kdeplot(full["Age"][(full["Survived"] == 1)], ax = g, color="#FFDA50", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# - The distribution of Age, as seen earlier, follows a fairly normal distribution with a slight tail.
# 
# - However, the distribution of Age varies slightly from passengers that survived and did not survive.
# 
# - We can see that there is a peak relating to young passengers, **under the age of 10**, that survived. Also you can see that older passengers, **60-80**, had a lower chance of survival. In general, younger passengers had a greater chance of survival than older ones.
# 
# - From this plot, we can say that there are some age categories of passengers that have a greater or worse chance of survival.
#     - **Therefore, this could be a good feature to bin into categories.**
#     
# ### 3.2.2 - Fare
# 
# From the newly logarithmic transformed Fare feature, we can see that the feature has a much lower skewness. But it is not following exactly a normal distribution. Lets examine how this feature changes with Survival.

# In[ ]:


# Age distibution vs Survived
plt.subplots(figsize=(15, 10))
g = sns.kdeplot(full["Fare"][(full["Survived"] == 0)], color="#76CDE9",  shade = True)
g = sns.kdeplot(full["Fare"][(full["Survived"] == 1)], ax = g, color="#FFDA50", shade= True)
g.set_xlabel("Fare")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# - The distributions of Fare between survivors and non-survivors differs greatly. This leads me to believe that Fare has a very important influence on survival, as backed up by the correlation matrix.
# - We can see from the above plot that passengers who paid a lower fare, had less of a chance of survival.
# - Whereas passengers that paid a mid to higher fare, had a significantly higher chance of survival.
#     - Therefore, **Fare lends itself to being a good candidate for binning into categories as well.**
#     
# ### 3.2.3 - Parch
# 
# - Parch represents the number of parents and children associated with the passenger aboard.
# - Hence, although this is a numerical feature, it is a discrete one. A distribution plot will not be suitable here, instead we will use a factorplot to display the survival chances of each discrete value.

# In[ ]:


# Explore Parch feature vs Survived
g  = sns.factorplot(x = "Parch", y = "Survived", data = full, kind = "bar", palette = mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")


# - Here we can see that generally, the **lower the number of parents and children aboard, the better the chance of a passengers survival**. 
# - This may have been for several reasons, for example looking for fellow family members during the tragedy.
# - One value to take note of is Parch = 3, this shows the highest chance of survival for this feature. However, it also has a very high standard deviation, meaning that the survival of passengers within the category, vary largely from the mean survival rate of people with 3 parents and children on board. 
# - Finally, although this is a numerical feature, since we are dealing with a finite number of discrete values we will treat is from now as a **categorical variable**.
# - This feature also gives indication to the size of the family travelling with passengers, therefore we will use it later to construct a new feature, **"Family Size"**.
# 
# ### 3.2.4 - Pclass
# 
# - Similarly to Parch, Pclass is a numerical feature, but it consists of a finite number of discrete values - 1, 2, and 3. Hence, a distribution plot will not be suitable here.
# - We will use a factorplot to show how survival rate was affected by the class.

# In[ ]:


# Explore Pclass feature vs Survived
g  = sns.factorplot(x = "Pclass", y = "Survived", data = full, kind = "bar", palette = mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")


# - We can see here that the chances of **survival decrease consistently as you move down the classes**.
# - 1st class had the best chance of survival - perhaps they were prioritised due to higher socio-economic status, or they were located in a beneficial position (cabin closer to an exit) for escape.
# - 3rd class had the worst chance of survival.

# In[ ]:


# Explore Pclass vs Survived by Sex
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=full,
                    kind="bar", palette=mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")


# - The survival chances discussed above also hold across Genders. 
# - Both **males and females have decreasing chances of survival, the lower class they were in**.
# - However, **females clearly have a much better chance of survival than males**.
# - Similarly to Parch, although Pclass is numerical, it will be better treated as a **categorical feature**.
# 
# ### 3.2.5 - SibSp

# In[ ]:


# Explore SibSp vs Survived
g = sns.factorplot(x="SibSp", y="Survived", data=full,
                    kind="bar", palette=mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")


# - It looks as though passengers **having many siblings or spouses aboard, had a lower chance of survival**. Perhaps this ties up with the reasoning why passengers with a large value for "Parch" had a lower chance of survival.
# - This feature, along with "Parch" can be later used to create a new variable, indicating **"Family Size"**.
# 
# ### 3.3 - Categorical values
# 
# Now I will analyse the categorical features in the dataset. This exploratory analysis will help me to construct an idea of which features to create in the next stage, and whether there are any significant features in predicting survival.
# 
# ### 3.3.1 - Cabin
# 
# As seen earlier when replacing the missing values for this column, this is an interesting feature as it may act as a proxy for many things such as:
# 
# - Ticket fare
# - Social-economic status
# - Location on board, perhaps some people may have been located in a beneficial position for escape
# 
# In order to tackle the missing values, we assigned the value 'X' to signal that this passenger did not have a cabin.
# 
# Let's have a look at how the passengers were distributed across cabins and also how survival rate was affected by cabin letter.

# In[ ]:


plt.subplots(figsize=(15, 10))
g = sns.countplot(x = "Cabin", data = full, palette = mycols)


# - We can see from this countplot that the **majority of passengers, did not have a cabin**. The letter associated with the most passengers after this were "B" and "C".
# - We may be seeing a lot of passengers in cabin "X" because **most of the passengers were travelling 3rd class**. So the Cabins, may have been assigned to higher ranking or higher paying passengers, whereas lower fare passengers may have been on the deck.

# In[ ]:


g = sns.factorplot(y = "Survived", x ="Cabin", data = full, kind = "bar", size = 10, aspect = 1.5,
                   order=['A','B','C','D','E','F','G','T','X'], palette = mycols)
g.despine(left=True)
g = g.set_ylabels("Survival Probability")


# - Interestingly, we can see here that **passengers that were travelling in a cabin, had a much better chance of survival than passengers who weren't**. The only exception to this was cabin "T", of which you can see from the previous chart, there was a very low amount of passengers travelling in.
# - The standard deviation for "X" is okay, whereas the standard deviation for the cabins vary. Some are acceptable but some are very high. 
# - This tells me that although they had a better chance of survival, the **results varied highly between cabins**.
# - This gives me an indication that **creating dummy variables from these categories, may be a good choice**.
# 
# ### 3.3.2 - Embarked

# In[ ]:


# Explore Embarked vs Survived
g = sns.factorplot(x="Embarked", y="Survived", data=full,
                    kind="bar", palette=mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")


# - Here we can see that the **highest survival rates came from passengers embarking from Cherbourg**, then Queenstown and finally Southampton.
# - We know that we had the most amount of passengers embarking from Southampton but **maybe the "type" of people embarking from these ports had an influence on their survival**.

# In[ ]:


htmp = pd.crosstab(full['Embarked'], full['Pclass'])

plt.subplots(figsize=(15, 10))
g = sns.heatmap(htmp, cmap="YlGnBu", vmin = 0, vmax = 500, linewidths = 1, annot = True, fmt = "d");


# - This confirms that **Southampton had the most number of 3rd class passengers, but also had the most number of first class passengers**. 
# - However, Cherbourg also had a large number of first class passengers too.
# - Perhaps first class passengers were prioritised during the tragedy, or they were in a beneficial position to evacuate. 
# - Interestingly, looking back at the initial research... The first port of embarkation was Southampton, then Cherbourg, then Queenstown. Southampton had the most number of 1st class passengers, but the lowest rate of survival. Could this be because they were packed into the lower decks and cabins on the boats, in order to leave space and to allow for easier loading of passengers to the upper decks at Cherbourg and Queenstown? Just a thought that may be worth investigating. 
# 
# ### 3.3.3 - Sex
# 
# - We all remember how the saying went... **"Women and children first!"**
# - Therefore, without visualising anything it's easy to understand what we may see.
# - I will confirm these assumptions below, and then explore how the survival of males and females changed, across the other features.

# In[ ]:


# Explore Sex vs Survived
g = sns.factorplot(x="Sex", y="Survived", data=full,
                    kind="bar", palette=mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")


# - As expected, the survival rates of **females were signficantly higher than males.**
# - **Note**: in order for the algorithms to be able to process this column. During feature engineering I will encode this variable with a 1 and 0 instead of male/female.
# - Now to explore the survival rates of males and females across the other features.

# In[ ]:


sns.factorplot(x="Sex", col = "Pclass", hue = "Survived", data=full, kind="count");


# - We see that we have an **even number of female survivors across classes**. However for men, **3rd class passengers did not have a good chance of survival**. 
# - More male survivors actually came from 2nd class, than 1st class.
# - The most number of female deaths came from 3rd class

# In[ ]:


sns.factorplot(x="Sex", y = "Age", hue = "Survived", data=full, kind="violin",
               split = True, size = 10, aspect = 1.5);


# - Now focusing on "Age". We can see here that for **males, of the non-survivors, there was a slightly positively skewed normal distribution**. However, for surviving males, there is a large number of survivors of a **very young age** - perhaps indicating young children or babies. Then we have another peak at around **30 years old**.
# - For females, there was a much larger proportion of **young females that died than survived**, aged approximately 10 and under.
# - However when looking at the survivors, the most number of female survivors were around **25 years old**, and there were also a large number of females around 40 years old that survived.
# - **Age clearly had a large influence on survival across gender**.

# In[ ]:


sns.factorplot(x="Sex", y="Fare", hue="Survived", col = "Pclass", row="Embarked", data=full, kind="bar");


# - Finally, this shows that across Genders - the **fare paid decreased as you go down the classes**. 
# - Also, it is unclear whether survivors paid more or less than non-survivors, as this varies inconsistently.
# - Interestingly, of the passengers that embarked from Queenstown across 1st and 2nd class. There were no male survivors, and no females that died.
# 
# This completed the **Feature Analysis**, and has given me a good foundation to decide on how to conduct **Feature Engineering**.

# ***

# # 4. 
# ## Feature Engineering
# 
# - During this section I will create a variety of new variables with a multitude of methods.
# - For categorical variables, you can **encode each of the categories to have a numerical representation**.
#     - e.g. for the column Embarked, we could encode "S", "C", and "Q" to be represented as 0, 1 and 2 respectively.
# - However, I have decided to create **dummy variables** for each category within a column. This will replace the original column with a seperate column for each category, with a **flag** representing whether that passenger is associated with that category.
# - For numerical variables, we have seen that there are **certain values that indicate a better chance of survival**. So I will convert these into **binned categories** and create **dummy variables** from them.
# - I will also create some completely new features, by combining existing ones.
# 
# 
# ### 4.1 - Age
# 
# - As discussed when analysing this feature, the survival rates differed greatly with Age.
# - Therefore we will **bin this feature into categories**.

# In[ ]:


full.head()


# In[ ]:


# Creating a new feature by cutting the Age column into 5 bins
full['Age Band'] = pd.cut(full['Age'], 5)

# Now lets see the age bands that pandas has created and how survival is affected by each
full[['Age Band', 'Survived']].groupby(['Age Band'], as_index=False).mean().sort_values(by='Age Band', 
                                                                                          ascending=True)


# - These age bands look like suitable bins for natural categorisation. However, I now want to replace these bins with meaningful values and create dummy variables from it.

# In[ ]:


# Locate and replace values
full.loc[full['Age'] <= 16.136, 'Age'] = 1
full.loc[(full['Age'] > 16.136) & (full['Age'] <= 32.102), 'Age'] = 2
full.loc[(full['Age'] > 32.102) & (full['Age'] <= 48.068), 'Age']   = 3
full.loc[(full['Age'] > 48.068) & (full['Age'] <= 64.034), 'Age']   = 4
full.loc[ full['Age'] > 63.034, 'Age'] = 5
full['Fare'] = full['Fare'].astype(int)

# Replace with categorical values
full['Age'] = full['Age'].replace(1, '0-16')
full['Age'] = full['Age'].replace(2, '16-32')
full['Age'] = full['Age'].replace(3, '32-48')
full['Age'] = full['Age'].replace(4, '48-64')
full['Age'] = full['Age'].replace(5, '64+')


# Creating dummy columns for each new category, with a 1/0
full = pd.get_dummies(full, columns = ["Age"], prefix="Age")

# Deleting the Age and Age Band column as they are no longer needed
drop = ['Age Band']
full.drop(drop, axis = 1, inplace = True)

full.head(n=3)


# ### 4.2 - Cabin
# 
# - Luckily, we have already done the leg work during the analysis stage. All we have to do now is create dummy variables for each cabin letter.

# In[ ]:


full = pd.get_dummies(full, columns = ["Cabin"], prefix="Cabin")
full.head(3)


# ### 4.3 - Embarked
# 
# - For this feature, we have simply 3 categories. So all we need to do is create 3 dummy columns to represent each category.

# In[ ]:


full = pd.get_dummies(full, columns = ["Embarked"], prefix="Embarked")
full.head(3)


# ### 4.4 - Fare
# 
# - Like Age, we want to investigate how we can **bin this feature into categories**. We originally had a few extreme values creating a positively skewed distribution. However we cleaned this up using a **logarithmic transformation**. 
# - Normally for a skewed distribution I would use **"quantile cut"** to bin a feature.
#     - The reason for this is because **"cut" creates bins of equal spacing according to the values themselves and not the frequences of those values**.
#     - Hence, because Age is normally distributed, you'll see higher frequencies in the inner bins and fewer in the outer. This essentially will reflect the bell shaped curve of a normal distribution.
#         - However with "qcut", the bins will be chosen so that you have the same number of records in each bin.
# - But, the newly transformed Fare column has a skewness similar to Age, so I will use **"cut"** to create the bins.

# In[ ]:


full['Fare Band'] = pd.cut(full['Fare'], 4)
full[['Fare Band', 'Survived']].groupby(['Fare Band'], as_index=False).mean().sort_values(by='Fare Band', 
                                                                                          ascending=True)


# - Now I will locate each of the bins using ".loc" and replace it with an encoded number from 1-4.
# - After this, I will replace this with a categorical value to create dummy variables.

# In[ ]:


# Locate and replace values
full.loc[ full['Fare'] <= 1.56, 'Fare'] = 1
full.loc[(full['Fare'] > 1.56) & (full['Fare'] <= 3.119), 'Fare'] = 2
full.loc[(full['Fare'] > 3.119) & (full['Fare'] <= 4.679), 'Fare']   = 3
full.loc[ full['Fare'] > 4.679, 'Fare'] = 4
full['Fare'] = full['Fare'].astype(int)

# Replace with categorical values
full['Fare'] = full['Fare'].replace(1, 'Very low')
full['Fare'] = full['Fare'].replace(2, 'Low')
full['Fare'] = full['Fare'].replace(3, 'Medium')
full['Fare'] = full['Fare'].replace(4, 'High')

# Create dummy variables
full = pd.get_dummies(full, columns = ["Fare"], prefix="Fare")

# Drop the un-needed Fare Band column
drop = ['Fare Band']
full.drop(drop, axis = 1, inplace = True)

full.head(n=3)


# ### 4.5 - Title
# 
# - As mentioned before, the "Name" column itself does not hold that much insight.
# - However, what may be interesting is if we extract the **title** from the name, and categorise this.

# In[ ]:


# Extracting Title from the Name feature
full['Title'] = full['Name'].str.extract(' ([A-Za-z]+)\.', expand=True)

# Displaying number of males and females for each Title
pd.crosstab(full['Sex'], full['Title'])


# - The goal here is to create categories from this "Title" column. As seen from the crosstab, we have some Title's which have few very passengers. 
# - Some of these can be **merged** with other Titles, such as Mlle and Ms with Miss.
# - Some titles are very rare too so I can categorise these under a **"Rare"** categorisation.

# In[ ]:


# Merge titles with similar titles
full['Title'] = full['Title'].replace('Mlle', 'Miss')
full['Title'] = full['Title'].replace('Ms', 'Miss')
full['Title'] = full['Title'].replace('Mme', 'Mrs')

# Now lets see the crosstab again
pd.crosstab(full['Sex'], full['Title'])


# In[ ]:


# Now I will replace the rare titles with the value "Rare"
full['Title'] = full['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev',
                                       'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
# Let's see how these new Title categories vary with mean survival rate
full[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# - We now have 5 meaningful categories. We can see that "Mrs" has the highest chance of survival, and "Mr" the lowest.
# - The final thing to do is to create some dummy variables with these new "Title" categories.

# In[ ]:


# Create dummy variables for Title
full = pd.get_dummies(full, columns = ["Title"], prefix="Title")

# Drop unnecessary name column
drop = ['Name']
full.drop(drop, axis = 1, inplace = True)

full.head(3)


# ### 4.6 - Family Size
# 
# - The next feature I will construct is using "Parch" and "SibSp" to show the size of family associated with the family on board.
# - This is a simple summation of **Parch + SibSp + 1 **
# - We add 1 to include the passenger in the family size feature.
# 
# - Also from our analysis earlier of SibSp and Parch, we saw that the chance of **survival varied greatly between each of the discrete values**. 
# - Generally, the lower the number of Parch and SibSp, the better the chance of survival.
# - Therefore, we will use this Family Size feature to construct some **binned features**.

# In[ ]:


# First creating the Family Size feature
full['Family Size'] = full['SibSp'] + full['Parch'] + 1

# Explore Family Size vs Survived
g = sns.factorplot(x="Family Size", y="Survived", data=full,
                    kind="bar", palette=mycols, size = 10, aspect = 1.5)
g.despine(left=True)
g = g.set_ylabels("Survival probability")


# - As expected from our analysis of Parch and SibSp, **Family Size follows a similar pattern. **
# - Interestingly, after a value of 4, the survival rate decreases, but then spikes at 7. However, these larger values have a high standard deviation.

# In[ ]:


# Now to bin the Family Size feature into bins
full['Lone Traveller'] = full['Family Size'].map(lambda s: 1 if s == 1 else 0)
full['Party of 2'] = full['Family Size'].map(lambda s: 1 if s == 2  else 0)
full['Medium Family'] = full['Family Size'].map(lambda s: 1 if 3 <= s <= 4 else 0)
full['Large Family'] = full['Family Size'].map(lambda s: 1 if s >= 5 else 0)

# Delete the no longer needed Family Size
drop = ['Family Size']
full.drop(drop, axis = 1, inplace = True)

full.head(n=3)


# ### 4.7 - Sex
# 
# - All we have to do for Sex is to convert the categorical string values into a 1/0.

# In[ ]:


# Convert Sex into categorical value - 0 for male and 1 for female
full["Sex"] = full["Sex"].map({"male": 0, "female":1})
full.head()


# ### 4.8 - Ticket
# 
# - Ticket is an interesting column because we have **lots of unique categories, but within the ticket number is some encoded information**.
# - Some tickets have letters at the start, indicating some association with the port.
#     - Some are purely numbers.
# - However, I believe that there may be some useful information within this feature. What I will do is **create a flag to indicate whether the ticket includes a prefix including letters**.
# - I could have created dummy variables for each individual prefix, or the prefix's that just had letters. But I thought that wrapping all of the prefixes including letters into one flag would be better in this case, since there are many prefixes.

# In[ ]:


# First, I want to create a new column that extracts the prefix from the ticket.
# In the case where there is no prefix, but a number... It will return the whole number.

# I will do this by first of all creating a list of all the values in the column
tickets = list(full['Ticket'])

# Using a for loop I will create a list including the prefixes of all the values in the list
prefix = []
for t in tickets:
    split = t.split(" ", 1) # This will split each value into a list of 2 values, surrounding the " ". For example, the ticker "A/5 21171" will be split into [A/5, 21171]
    prefix.append(split)
    
# Now I want to take the first value within these lists for each value. I will put these into another list using a for loop.
tickets = []
for t in prefix:
    ticket = t[0]
    tickets.append(ticket)
    
full['Ticket_Prefix'] = pd.Series(tickets)
full.head(3)


# - Now we have the prefix, or in the case of tickets that had no prefix, we have numbers.
# - Final thing to do is create a **flag to indicate the passengers that had a ticket with a prefix or not**.
# - The way I will do this is by creating a function to allocate a 0 to all entries that start with 0-9, and allocate a 1 to everything else, i.e. the values with a prefix.

# In[ ]:


# Create the function
def TicketPrefix(row):
    for t in row['Ticket_Prefix']:
        if t[0] == '0':
            val = 0
        elif t[0] == '1':
            val = 0
        elif t[0] == '2':
            val = 0
        elif t[0] == '3':
            val = 0
        elif t[0] == '4':
            val = 0
        elif t[0] == '5':
            val = 0
        elif t[0] == '6':
            val = 0
        elif t[0] == '7':
            val = 0
        elif t[0] == '8':
            val = 0
        elif t[0] == '9':
            val = 0
        else:
            val = 1
        return val
    
# Create a new column that appolies the above function to create its values        
full['Ticket Has Prefix'] = full.apply(TicketPrefix, axis = 1)

# Clean up variables not needed anymore
drop = ['Ticket', 'Ticket_Prefix', 'PassengerId'] # We delete passenger ID here as it is no use for the modeling
full.drop(drop, axis = 1, inplace = True)

full.head(3)


# - We have now finished feature engineering. I tried many different methods of this, including **creating polynomials and also interaction variables**. 
# - If you wish to create interaction variables, you'll have to encode the initial features differently, as **you don't want to multiply two variables that contain a 0 within them**.
#     - Let's take an example - we have two columns with 3 categories, represented by 0, 1 and 2. If you create an interaction variable from these two features, 5 of the interactions will all = 0, and it will be **impossible to distinguish between whihc combination of categories created it**.
# - The reason why I have decided to create as many features as possible, is because I plan to **reduce these features specifically to each algorithm** as they will give different importances to each feature.
# 
# ### 4.8 - Correlation of all features

# In[ ]:


corr = full.corr()
plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(150, 250, as_cmap=True)
sns.heatmap(corr, cmap="YlGnBu", vmin = -1.0, vmax = 1.0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True);


# - We can see that we have many features, and we have some nice correlations between survived and the features.
# - One thing to always be careful of is **multicollinearity**.
#     - ***This is where a predictor feature within our model can be linearly predicted from other features with a substantial degree of accuracy***.
#         - i.e. where we have high correlations between predictor features, this is what we want to avoid, as this suggests that one can accurately predict the other, or the value of one may be encoded within the other, essentially telling us the same thing.
#         - Multicollinearity can cause problems not because it will reduce the predictive power of the model, but it **affects the calculations in regards to the individual features**.
#             - Meaning that if we have a model with colinear predictors, it can indicate how well the entire bundle of predictors can predict the outcome we are trying to find. But it may not give valid results about any individual feature. **When we use algorithms that depend on assigning varying weights to features, this can become a problem**.
# - Looking at the correlation heatmap, we dont have any pressing issues of this. However, we do have some examples where we have high correlations. In order to overcome this, we will **reduce the features of each individual model in the next section**.

# ***

# # 5.
# ## Modeling
# ### 5.1 - Preparation of data
# 
# - We have created all the features we desire for modeling. Now we need to prepare the training and test datasets for training and prediction.

# In[ ]:


# First I will take a copy of the full dataset, in case we need to change or use the full dataset for something else.
copy = full.copy()

# Remember that the training dataset had only 891 rows, and the test had 418. 
# At the start we simply concatenated the two datasets together to form our "full" dataset
# Hence, we can simply split the two datasets apart again to extract the training and test data.
train_setup = copy[:891]
test_setup = copy[891:]

# I'll take another copy just to be safe
train = train_setup.copy()
test = test_setup.copy()

# And print the shape of these dataframes to confirm it has done what I intended
print('train: ', train.shape, '\ntest: ', test.shape)


# - You'll notice that from the tables above, "Survived" appears as a float.
# - Hence, in order for the output files later to be accepted by Kaggle, **I will convert this to an integer.**
# - Note, that our test dataset should have 35 features, as the "Survived" feature was originally missing.
# - You'll find that all the values for "Survived" in this new test dataset will be filled with NaN.
# - Hence, we can remove this column with no issues.

# In[ ]:


# Convert the "Survived" column in the training data to an integere
train['Survived'] = train['Survived'].astype(int)

# Drop the "Survived" column from the test data
drop = ['Survived']
test.drop(drop, axis=1, inplace=True)

print('train: ', train.shape, '\ntest: ', test.shape)


# - Now I will **separate the predictor features of the training dataset, from the "Survived" column which is what we want to predict**.
# - The reason why we do this is because the algorithms that we will run later take two inputs, **X** and **Y**, representing the features of the model, and the desired predicted outcome.

# In[ ]:


# Take another copy for the training features, because why not
train_x = train.copy()

# Now delete the "Survived" column from the training features
del train_x['Survived']

# And create a new column called train_y representing what we are trying to predict for the training dat
train_y = train['Survived']

# Confirm this all worked
print('train_x - features for training: ', train_x.shape, '\ntrain_y - target variable for training: ', train_y.shape,)


# - We can see that we have 35 features for training for 891 rows, which is exactly what we want.
# - Our target column is simply 1 column spanning 891 rows, with a 1/0.
# - Next, we will take the training dataset and target column that we have just built, and we will **create samples of this**.
#     - The reason why we are doing this is so that we can have an **estimation, validation and test dataset**.
#     - We already have our test dataset, this is the dataset that we will apply our final model to and submit our predictions to Kaggle.
#     - However, we will now **split the training dataset into 2 smaller sets. So that we can train our model, and then test it against the validation sample**.
#         - We do this so that we can be certain that the **models we select are robust!**
#         - We do not want to produce a model that predicts the training dataset with 100% accuracy, but is unable to generalise to new data because it has learnt the training data too closely.
#         - Hence, in order to **create models that are able to handle new data** that may vary from the training data - we must test out model before we let it loose on our final test data.
#         - What we are looking for is a model that performs well in predicting the estimation dataset, but also one that is able to predict with a high accuracy the results of the validation dataset, without seeing the target column for validation.

# In[ ]:


# To do this, we will use the neat package train_test_split
# This allows you to split your dataset into smaller training and test samples, controlling how much goes into each.
# For this, we will assign 25% to the validation dataset
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_x, train_y, test_size=0.25, random_state=42)

# X_train = predictor features for estimation dataset
# X_test = predictor variables for validation dataset
# Y_train = target variable for the estimation dataset
# Y_test = target variable for the estimation dataset

print('X_train: ', X_train.shape, '\nX_test: ', X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)


# - Now we are ready for modeling with our algorithms! Hooray!

# ## 5.2 - Model comparison
# 
# - What we are trying to predict is "Survival". We have seen that this is a **binary feature**, meaning that it can only take the values 1/0.
# - This immediately narrows our problem to which algoriths we can choose. The class of algorithms that we must choose from are called **"Classification"** algorithms, a supervised learning class of algorithms that are capable of predicting 1's and 0's.
# - Luckily, we have lots of classification algorithms to choose from. I have decided to take a variety of algorithms from the scikit-learn package.
# 
# The algorithms that I have chosen are the following:
# 
# ***Tree Based:***
# - Decision Tree Classifier
# 
# ***Ensemble:***
# - Random Forest Classifier
# - Extra Trees Classifier
# - AdaBoost Classifier
# - XGBoost
# - Gradient Boosting Classifier
# 
# ***Support Vector Machines:***
# - Support Vector Machine Classifier
# - Linear Support Vector Machine Classifier
# 
# ***Neighbours:***
# - K-Nearest Neighbours Classifier
# 
# ***Neural Networks:***
# - Multi Layer Perceptron Classifier
# 
# ***Linear***
# - Perceptron
# - Logistic Regression
# - Stochastic Gradient Descent Classifier
# 
# ***Naive Bayes:***
# - Gaussian Naive Bayes
# 
# ***Voting:***
# - Ensemble Voting Classifier
# 
# *My explanations below are not intended to be exhaustive. I created the graphics to introduce the algorithms in a light, easy to understand way. This will give you a solid foundation to understand how and why they work. For more detailed information please refer to the references provided.*
# 
# ### 5.2.1 - Algorithm A-Z
# 
# ### ***Tree Based***

# In[ ]:


Image(filename='../input/titanic-images/DT.png')


# - Decision trees have a lot of parameters to tune. The way they split, the depth and how many observations have been placed into each terminal node, as well as much more, can be tuned with optimisation.
# - Decision trees also form the basis of various other modeling techniques, as described below.
# 
# ### ***Ensemble***

# In[ ]:


Image(filename='../input/titanic-images/randomforest.png')


# In[ ]:


Image(filename='../input/titanic-images/ETC.png')


# In[ ]:


Image(filename='../input/titanic-images/ada.png')


# In[ ]:


Image(filename='../input/titanic-images/xgb.png')


# In[ ]:


Image(filename='../input/titanic-images/GBC.png')


# ### ***Support Vector Machines***

# In[ ]:


Image(filename='../input/titanic-images/SVM.png')


# ### ***Neighbours***

# In[ ]:


Image(filename='../input/titanic-images/KNN.png')


# ### ***Neural Networks***

# In[ ]:


Image(filename='../input/titanic-images/MLP.png')


# ### ***Linear***

# In[ ]:


Image(filename='../input/titanic-images/logreg.png')


# ### **Naive Bayes**

# In[ ]:


Image(filename='../input/titanic-images/GNB.png')


# ### ***Voting***

# In[ ]:


Image(filename='../input/titanic-images/voting.png')


# ### 5.2.2 - Performance comparison
# 
# - Now I will run all of these models against the **estimation dataset**, and then cross validate the results to ensure robust results. 
# 
# - To ensure good quality performance, I will then **test the trained model against the validation dataset**.
# - I will run the "voting" classifier once I have optimised the selected algorithms.

# In[ ]:


# First I will use ShuffleSplit as a way of randomising the cross validation samples.
shuff = ShuffleSplit(n_splits=3, test_size=0.2, random_state=50)

# Decision Tree
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, Y_train)
dt_scores = cross_val_score(dt, X_train, Y_train, cv = shuff)
dt_scores = dt_scores.mean()
dt_apply_acc = metrics.accuracy_score(Y_test, dt.predict(X_test))

# Random Forest
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, Y_train)
rf_scores = cross_val_score(rf, X_train, Y_train, cv = shuff)
rf_scores = rf_scores.mean()
rf_apply_acc = metrics.accuracy_score(Y_test, rf.predict(X_test))

# Extra Trees
etc = ExtraTreesClassifier(random_state=0)
etc.fit(X_train, Y_train)
etc_scores = cross_val_score(etc, X_train, Y_train, cv = shuff)
etc_scores = etc_scores.mean()
etc_apply_acc = metrics.accuracy_score(Y_test, etc.predict(X_test))

# Adaboost classifier
ada = AdaBoostClassifier(random_state=0)
ada.fit(X_train, Y_train)
ada_scores = cross_val_score(ada, X_train, Y_train, cv = shuff)
ada_scores = ada_scores.mean()
ada_apply_acc = metrics.accuracy_score(Y_test, ada.predict(X_test))

# xgboost
xgb = XGBClassifier(random_state=0)
xgb.fit(X_train, Y_train)
xgb_scores = cross_val_score(xgb, X_train, Y_train, cv = shuff)
xgb_scores = xgb_scores.mean()
xgb_apply_acc = metrics.accuracy_score(Y_test, xgb.predict(X_test))

#gradient boosting classifier
gbc = GradientBoostingClassifier(random_state=0)
gbc.fit(X_train, Y_train)
gbc_scores = cross_val_score(gbc, X_train, Y_train, cv = shuff)
gbc_scores = gbc_scores.mean()
gbc_apply_acc = metrics.accuracy_score(Y_test, gbc.predict(X_test))

# Support Vector Machine Classifier
svc = SVC(random_state=0)
svc.fit(X_train, Y_train)
svc_scores = cross_val_score(svc, X_train, Y_train, cv = shuff)
svc_scores = svc_scores.mean()
svc_apply_acc = metrics.accuracy_score(Y_test, svc.predict(X_test))

# Linear Support Vector Machine Classifier
lsvc = LinearSVC(random_state=0)
lsvc.fit(X_train, Y_train)
lsvc_scores = cross_val_score(lsvc, X_train, Y_train, cv = shuff)
lsvc_scores = lsvc_scores.mean()
lsvc_apply_acc = metrics.accuracy_score(Y_test, lsvc.predict(X_test))

# K-Nearest Neighbours
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knn_scores = cross_val_score(knn, X_train, Y_train, cv = shuff)
knn_scores = knn_scores.mean()
knn_apply_acc = metrics.accuracy_score(Y_test, knn.predict(X_test))

# Multi Layer Perceptron Classifier
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train, Y_train)
mlp_scores = cross_val_score(mlp, X_train, Y_train, cv = shuff)
mlp_scores = mlp_scores.mean()
mlp_apply_acc = metrics.accuracy_score(Y_test, mlp.predict(X_test))

# Perceptron
pcn = Perceptron(random_state=0)
pcn.fit(X_train, Y_train)
pcn_scores = cross_val_score(pcn, X_train, Y_train, cv = shuff)
pcn_scores = pcn_scores.mean()
pcn_apply_acc = metrics.accuracy_score(Y_test, pcn.predict(X_test))

#Logistic regression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, Y_train)
lr_scores = cross_val_score(lr, X_train, Y_train, cv = shuff)
lr_scores = lr_scores.mean()
lr_apply_acc = metrics.accuracy_score(Y_test, lr.predict(X_test))

# Stochastic Gradient Descent
sgd = SGDClassifier(random_state=0)
sgd.fit(X_train, Y_train)
sgd_scores = cross_val_score(sgd, X_train, Y_train, cv = shuff)
sgd_scores = sgd_scores.mean()
sgd_apply_acc = metrics.accuracy_score(Y_test, sgd.predict(X_test))

# Gaussian Naive Bayes
gss = GaussianNB()
gss.fit(X_train, Y_train)
gss_scores = cross_val_score(gss, X_train, Y_train, cv = shuff)
gss_scores = gss_scores.mean()
gss_apply_acc = metrics.accuracy_score(Y_test, gss.predict(X_test))


models = pd.DataFrame({
    '1_Model': ['Gradient Boosting Classifier',
              'Logistic Regression',             
              'Support Vector Machine',
              'Linear SVMC',
              'Random Forest', 
              'KNN',
              'Gaussian Naive Bayes',
              'Perceptron',
              'Stochastic Gradient Descent',
              'Decision Tree',
              'XGBoost',
              'Adaboost',
              'Extra Trees', 
              'Multi Layer Perceptron'],
    '2_Mean Cross Validation Score': [gbc_scores,
                                      lr_scores, 
                                      svc_scores, 
                                      lsvc_scores,
                                      rf_scores, 
                                      knn_scores, 
                                      gss_scores, 
                                      pcn_scores, 
                                      sgd_scores, 
                                      dt_scores,
                                      xgb_scores, 
                                      ada_scores, 
                                      etc_scores, 
                                      mlp_scores], 
    '3_Accuracy when applied to Test': [gbc_apply_acc,
                                      lr_apply_acc, 
                                      svc_apply_acc, 
                                      lsvc_apply_acc,
                                      rf_apply_acc, 
                                      knn_apply_acc, 
                                      gss_apply_acc, 
                                      pcn_apply_acc, 
                                      sgd_apply_acc, 
                                      dt_apply_acc,
                                      xgb_apply_acc, 
                                      ada_apply_acc, 
                                      etc_apply_acc, 
                                      mlp_apply_acc]
                                                    })

# Finally I will plot the scores for cross validation and test, to see the top performers
g = sns.factorplot(x="2_Mean Cross Validation Score", y="1_Model", data = models,
                    kind="bar", palette=mycols, orient = "h", size = 5, aspect = 2.5,
                  order = ['Adaboost', 'Logistic Regression', 'Support Vector Machine',
                          'Linear SVMC', 'Gradient Boosting Classifier', 'XGBoost', 
                          'Multi Layer Perceptron', 'Decision Tree', 'Random Forest', 
                          'Extra Trees', 'KNN', 'Gaussian Naive Bayes', 'Perceptron', 
                          'Stochastic Gradient Descent'])
g.despine(left = True);

h = sns.factorplot(x="3_Accuracy when applied to Test", y="1_Model", data = models,
                    kind="bar", palette=mycols, orient = "h", size = 5, aspect = 2.5,  
                  order = ['KNN', 'Support Vector Machine', 'XGBoost', 'Logistic Regression', 
                          'Linear SVMC', 'Stochastic Gradient Descent', 'Extra Trees', 
                          'Gradient Boosting Classifier', 'Multi Layer Perceptron', 
                          'Random Forest', 'Decision Tree', 'Adaboost', 'Gaussian Naive Bayes', 
                          'Perceptron'])
h.despine(left = True);


# ## 5.3 - Model selection
# 
# Using a combination of these 2 plots, I decided to focus on the following models for optimisation:
# 
# - ***KNN***
# - ***Support Vector Machine***
# - ***Gradient Boosting Classifier***
# - ***XGBoost***
# - ***Multi Layer Perceptron***
# - ***Linear SVMC***
# - ***Random Forest***
# - ***Logistic Regression***
# - ***Decision Tree***
# - ***Adaboost***
# - ***Extra Tree***
# 
# The low performing models will be ignored from now on. Below, you can also view the raw for cross validation and testing.

# In[ ]:


models


# ## 5.4 - Model feature reduction
# 
# - Now that we have chosen the models to use going forward, I now want to **reduce the features of the dataset.**
# 
# ***The reasons why dimensionality reduction is important are:***
# - High dimensionality can lead to **high computational cost** to perform learning and inference.
# - It often leads to **overfitting**, this is "the production of an analysis that corresponds too closely or exactly to a particular set of data, and therefore may fail to fit additional data or predict future observations reliably".
# 
# Furthermore, **some features may be more important than others with different models**. Hence, we much approach feature reduction carefully for each chosen model. No two reduced datasets will look the same for each model, unless we get lucky.
# 
# There are various ways to attempt to reduce the dimensionality of a dataset. The methods that I chose to use were:
# - **Chi-squared** test of independance. This test measures dependence between stochastic variables, so using this function "weeds out" the features that are most likely to be independant of class and therefore irrelevant for classification.
# - **Extra Trees** feature importance. This is a built in attribute of trees based estimators, the higher to score of a feature, the more important it is. This can be used to discard irrelevant features when couples with SelectFromModel.
# 
# *These first two methods of feature reduction were used for the algorithms that did not have the in-built feature_importances attribute.*
# 
# - **"feature__importances_"**. This is a built in attribute for some estimators that will indicate the higher the feature, the more important it is.
# - **Recursive Feature Elimination**. Given an estimator that assigns weights to features, such as coefficients, RFE is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained by the "coef_" or the "feature__importances_" attribute. Then, the least important features are pruned from the data. RFECV performs RFE in a cross-validation loop to find the optimal number of featurs.
# 
# The first two methods, **chi-squared and extra trees**, will provide us a reduced dataset without depending on the algorithm. Hence, for estimators without the "feature__importances_" attribute built in, we can re-use the reduced datasets produced. I will first create these reusable datasets to make life easier later.
# 
# **Note**: Moving forward I will use the reduced feature set that produces the best average training score and test accuracy for each model. In the case where two reduced feature sets produce comparable results, a trade-off will be necessary. Also consider that during optimisation, these scores are likely to increase.
# 
# ### Chi-squared reduction

# In[ ]:


# Because the chi2 test will be the same for each model, we only need to run this test once.
# When we have established which features are the most important via this test, we can re-use this 
# reduced dataset for all other estimators without "feature_importances_"

# chi2 test of independence

# fit a model using a score function of chi2
Kbest = SelectKBest(score_func=chi2, k=10)
fit = Kbest.fit(X_train, Y_train)

# Create a table with the results and score for each features
scores = pd.DataFrame({'Columns': X_test.columns.values, 'Score': fit.scores_})

# Visualise the scores of dependence with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot('Score','Columns',data = scores, palette=mycols,orient = "h")
g.set_xlabel("Importance")
g = g.set_title("Feature Importances using Chi-squared")


# - From this, we can see that there are only a **small number of features that have a high importance**.
# - There are a lot of features with very low importance and we can get rid of.

# In[ ]:


# First I will take some copies of the original estimation, validation and test datasets
# so that we don't mess with these
chi2_reduced_train = X_train.copy()
chi2_reduced_test = X_test.copy()
chi2_reduced_final_test = test.copy()

# Now I will drop all of the columns that I deemed to be irrelevant
# I played with a variety of options here, but this is what I found to work best
drop = ['SibSp', 'Age_0-16', 'Age_16-32', 'Age_32-48', 'Age_48-64', 'Cabin_A', 'Cabin_F',
        'Cabin_G', 'Embarked_Q', 'Embarked_S', 'Fare_Low', 'Fare_Medium', 'Title_Rare',
        'Title_Master','Title_Rare', 'Ticket Has Prefix',
        'Age_64+', 'Cabin_A', 'Cabin_F', 'Cabin_G', 'Cabin_T']

# Reduce features of estimation, validation and test for use in modelling
chi2_reduced_train.drop(drop, axis = 1, inplace = True)
chi2_reduced_test.drop(drop, axis = 1, inplace = True)
chi2_reduced_final_test.drop(drop, axis = 1, inplace = True)

# You'll see that we now have just 18 features
print('X_train: ', chi2_reduced_train.shape, '\nX_test: ', chi2_reduced_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)


# ### Extra Trees reduction

# In[ ]:


# Now I will do the same for the extra trees method of feature reduction
# Which once again, can be reused for estimators that do not have the feature_importances_ attribute

# First, let's fit a model using the extra trees classifier
model = ExtraTreesClassifier()
model.fit(X_train, Y_train)

# And create a table with the importances
scores = pd.DataFrame({'Columns': X_test.columns.values, 'Score': model.feature_importances_})
scores.sort_values(by='Score', ascending=False)

# Finally let's visualise this
plt.subplots(figsize=(15, 10))
g = sns.barplot('Score','Columns',data = scores, palette=mycols,orient = "h")
g.set_xlabel("Importance")
g = g.set_title("Feature Importances using Trees")


# - In comparison to the chi-squared test, we see a slightly **increased number of features that are important**. Some features are consistent with chi-squared, such as "Sex" and "Title".

# In[ ]:


# In a similar fashion, I will reduce the estimation, validation and test datasets according to
# the extra trees feature importances.

# Take another copy
etc_reduced_train = X_train.copy()
etc_reduced_test = X_test.copy()
etc_reduced_final_test = test.copy()

# Once again, I tried a few options here of which features the drop. I decided these were the best choice.
drop = ['Age_0-16', 'Age_16-32', 'Age_48-64', 'Age_64+', 'Cabin_A', 'Cabin_B',
        'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G','Cabin_T',
        'Embarked_Q', 'Title_Rare']

# Reduce features of estimation, validation and test datasets
etc_reduced_train.drop(drop, axis = 1, inplace = True)
etc_reduced_test.drop(drop, axis = 1, inplace = True)
etc_reduced_final_test.drop(drop, axis = 1, inplace = True)

# Let's see the new shape of the data
print('X_train: ', etc_reduced_train.shape, '\nX_test: ', etc_reduced_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)


# - The feature reduction methods: feature__importances_ and RFE, can be completed during modeling.
# - Now that we have prepared the chi-squared and extra trees reduced datasets for estimators without the feature__importances_ attribute, **we are ready to start modeling the selected algorithms.**
# 
# ### ***K-Nearest Neighbours***
# 
# - No feature__importances_ attribute available, so we will use the **chi-squared and extra trees reduced datasets**.

# In[ ]:


# KNN - chi-squared
knn = KNeighborsClassifier()

# Fit estimator to reduced dataset
knn.fit(chi2_reduced_train, Y_train)

# Compute cross validated scores and take the mean
knn_scores = cross_val_score(knn, chi2_reduced_train, Y_train, cv = shuff)
knn_scores = knn_scores.mean()

print('Chi2 - Mean Cross Validated Score: {:.2f}'. format(knn_scores*100))
knn_apply_acc = metrics.accuracy_score(Y_test, knn.predict(chi2_reduced_test))
print('Chi2 - Accuracy when applied to Test: {:.2f}'. format(knn_apply_acc*100))
print("-"*50)

##############################################################################

# KNN - extra trees
knn = KNeighborsClassifier()

# Fit estimator to reduced dataset
knn.fit(etc_reduced_train, Y_train)

# Compute cross validated scores and take the mean
knn_scores = cross_val_score(knn, etc_reduced_train, Y_train, cv = shuff)
knn_scores = knn_scores.mean()
print('Extra Trees - Mean Cross Validated Score: {:.2f}'. format(knn_scores*100))
knn_apply_acc = metrics.accuracy_score(Y_test, knn.predict(etc_reduced_test))
print('Extra Trees - Accuracy when applied to Test: {:.2f}'. format(knn_apply_acc*100))


# ### ***Support Vector Machine***
# 
# - No feature__importances_ attribute available, so we will use the **chi-squared and extra trees reduced datasets**.

# In[ ]:


# SVM - chi-squared
svc = SVC()

# Fit estimator to reduced dataset
svc.fit(chi2_reduced_train, Y_train)
        
# Compute cross validated scores and take the mean
svc_scores = cross_val_score(svc, chi2_reduced_train, Y_train, cv = shuff)
svc_scores = svc_scores.mean()
        
print('Chi2 - Mean Cross Validated Score: {:.2f}'. format(svc_scores*100))
svc_apply_acc = metrics.accuracy_score(Y_test, svc.predict(chi2_reduced_test))
print('Chi2 - Accuracy when applied to Test: {:.2f}'. format(svc_apply_acc*100))
print("-"*50)

##############################################################################

# SVM - extra trees
svc = SVC()

# Fit estimator to reduced dataset
svc.fit(etc_reduced_train, Y_train)
        
# Compute cross validated scores and take the mean
svc_scores = cross_val_score(svc, etc_reduced_train, Y_train, cv = shuff)
svc_scores = svc_scores.mean()
        
print('Extra Trees - Mean Cross Validated Score: {:.2f}'. format(svc_scores*100))
svc_apply_acc = metrics.accuracy_score(Y_test, svc.predict(etc_reduced_test))
print('Extra Trees - Accuracy when applied to Test: {:.2f}'. format(svc_apply_acc*100)) 


# ### ***Gradient Boosting Classifier***
# 
# - This estimator does include **"feature__importances_"** as an attribute, hence we will use this and **RFECV**.

# In[ ]:


# Sort feature importances from GBC model trained earlier
indices = np.argsort(gbc.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = gbc.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("GBC feature importance");


# In[ ]:


# Take some copies
gbc_red_train = X_train.copy()
gbc_red_test = X_test.copy()
gbc_final_test = test.copy()

# Fit a model to the estimation data
gbc = gbc.fit(gbc_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
gbc_feat_red = SelectFromModel(gbc, prefit = True)

# Reduce estimation, validation and test datasets
gbc_X_train = gbc_feat_red.transform(gbc_red_train)
gbc_X_test = gbc_feat_red.transform(gbc_red_test)
gbc_final_test = gbc_feat_red.transform(gbc_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', gbc_X_train.shape, '\nX_test: ', gbc_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
gbc_rfecv_train = X_train.copy()
gbc_rfecv_test = X_test.copy()
gbc_rfecv_final_test = test.copy()

# Initialise RFECV
gbc_rfecv = RFECV(estimator = gbc, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
gbc_rfecv.fit(gbc_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
gbc_rfecv_X_train = gbc_rfecv.transform(gbc_rfecv_train)
gbc_rfecv_X_test = gbc_rfecv.transform(gbc_rfecv_test)
gbc_rfecv_final_test = gbc_rfecv.transform(gbc_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(gbc_rfecv.support_)
print(gbc_rfecv.ranking_)

print('X_train: ', gbc_rfecv_X_train.shape, '\nX_test: ', gbc_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)


# In[ ]:


# GBC - feature_importances_
gbc = GradientBoostingClassifier(random_state=0)

# Fit estimator to reduced dataset
gbc.fit(gbc_X_train, Y_train)

# Compute cross validated scores and take the mean
gbc_scores = cross_val_score(gbc, gbc_X_train, Y_train, cv = shuff)
gbc_scores = gbc_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(gbc_scores*100))
gbc_apply_acc = metrics.accuracy_score(Y_test, gbc.predict(gbc_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(gbc_apply_acc*100))
print("-"*50)

##############################################################################

# GBC - RFECV
gbc = GradientBoostingClassifier(random_state=0)

# Fit estimator to reduced dataset
gbc.fit(gbc_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
gbc_scores = cross_val_score(gbc, gbc_rfecv_X_train, Y_train, cv = shuff)
gbc_scores = gbc_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(gbc_scores*100))
gbc_apply_acc = metrics.accuracy_score(Y_test, gbc.predict(gbc_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(gbc_apply_acc*100))


# ### ***XGBoost***
# 
# - This estimator does include **"feature__importances_"** as an attribute, hence we will use this and **RFECV**.

# In[ ]:


xgb = XGBClassifier()
xgb.fit(X_train, Y_train)

# Sort feature importances from GBC model trained earlier
indices = np.argsort(xgb.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = xgb.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("XGB feature importance");


# In[ ]:


# Take some copies
xgb_red_train = X_train.copy()
xgb_red_test = X_test.copy()
xgb_final_test = test.copy()

# Fit a model to the estimation data
xgb = xgb.fit(xgb_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
xgb_feat_red = SelectFromModel(xgb, prefit = True)

# Reduce estimation, validation and test datasets
xgb_X_train = xgb_feat_red.transform(xgb_red_train)
xgb_X_test = xgb_feat_red.transform(xgb_red_test)
xgb_final_test = xgb_feat_red.transform(xgb_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', xgb_X_train.shape, '\nX_test: ', xgb_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
xgb_rfecv_train = X_train.copy()
xgb_rfecv_test = X_test.copy()
xgb_rfecv_final_test = test.copy()

# Initialise RFECV
xgb_rfecv = RFECV(estimator = xgb, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
xgb_rfecv.fit(xgb_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
xgb_rfecv_X_train = xgb_rfecv.transform(xgb_rfecv_train)
xgb_rfecv_X_test = xgb_rfecv.transform(xgb_rfecv_test)
xgb_rfecv_final_test = xgb_rfecv.transform(xgb_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(xgb_rfecv.support_)
print(xgb_rfecv.ranking_)

print('X_train: ', xgb_rfecv_X_train.shape, '\nX_test: ', xgb_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)


# In[ ]:


# XGB - feature_importances_
xgb = XGBClassifier()

# Fit estimator to reduced dataset
xgb.fit(xgb_X_train, Y_train)

# Compute cross validated scores and take the mean
xgb_scores = cross_val_score(xgb, xgb_X_train, Y_train, cv = shuff)
xgb_scores = xgb_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(xgb_scores*100))
xgb_apply_acc = metrics.accuracy_score(Y_test, xgb.predict(xgb_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(xgb_apply_acc*100))
print("-"*50)

##############################################################################

# XGB - RFECV
xgb = XGBClassifier()

# Fit estimator to reduced dataset
xgb.fit(xgb_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
xgb_scores = cross_val_score(xgb, xgb_rfecv_X_train, Y_train, cv = shuff)
xgb_scores = xgb_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(xgb_scores*100))
xgb_apply_acc = metrics.accuracy_score(Y_test, xgb.predict(xgb_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(xgb_apply_acc*100))


# ### ***Multi Layer Perceptron***
# 
# - No feature__importances_ attribute available, so we will use the **chi-squared and extra trees reduced datasets.**

# In[ ]:


# MLP - chi-squared
mlp = MLPClassifier()

# Fit estimator to reduced dataset
mlp.fit(chi2_reduced_train, Y_train)

# Compute cross validated scores and take the mean
mlp_scores = cross_val_score(mlp, chi2_reduced_train, Y_train, cv = shuff)
mlp_scores = mlp_scores.mean()

print('Chi2 - Mean Cross Validated Score: {:.2f}'. format(mlp_scores*100))
mlp_apply_acc = metrics.accuracy_score(Y_test, mlp.predict(chi2_reduced_test))
print('Chi2 - Accuracy when applied to Test: {:.2f}'. format(mlp_apply_acc*100))
print("-"*50)

##############################################################################

# MLP - extra trees
mlp = MLPClassifier()

# Fit estimator to reduced dataset
mlp.fit(etc_reduced_train, Y_train)

# Compute cross validated scores and take the mean
mlp_scores = cross_val_score(mlp, etc_reduced_train, Y_train, cv = shuff)
mlp_scores = mlp_scores.mean()
print('Extra Trees - Mean Cross Validated Score: {:.2f}'. format(mlp_scores*100))
mlp_apply_acc = metrics.accuracy_score(Y_test, mlp.predict(etc_reduced_test))
print('Extra Trees - Accuracy when applied to Test: {:.2f}'. format(mlp_apply_acc*100))


# ### ***Linear Support Vector Machine***
# 
# - No feature__importances_ attribute available, so we will use the **chi-squared and extra trees reduced datasets.**

# In[ ]:


# LSVC - chi-squared
lsvc = LinearSVC()

# Fit estimator to reduced dataset
lsvc.fit(chi2_reduced_train, Y_train)

# Compute cross validated scores and take the mean
lsvc_scores = cross_val_score(lsvc, chi2_reduced_train, Y_train, cv = shuff)
lsvc_scores = lsvc_scores.mean()

print('Chi2 - Mean Cross Validated Score: {:.2f}'. format(lsvc_scores*100))
lsvc_apply_acc = metrics.accuracy_score(Y_test, lsvc.predict(chi2_reduced_test))
print('Chi2 - Accuracy when applied to Test: {:.2f}'. format(lsvc_apply_acc*100))
print("-"*50)

##############################################################################

# LSVC - extra trees
lsvc = LinearSVC()

# Fit estimator to reduced dataset
lsvc.fit(etc_reduced_train, Y_train)

# Compute cross validated scores and take the mean
lsvc_scores = cross_val_score(lsvc, etc_reduced_train, Y_train, cv = shuff)
lsvc_scores = lsvc_scores.mean()
print('Extra Trees - Mean Cross Validated Score: {:.2f}'. format(lsvc_scores*100))
lsvc_apply_acc = metrics.accuracy_score(Y_test, lsvc.predict(etc_reduced_test))
print('Extra Trees - Accuracy when applied to Test: {:.2f}'. format(lsvc_apply_acc*100))


# ### ***Random Forest***
# 
# - This estimator does include **"feature__importances_"** as an attribute, hence we will use this and **RFECV**.

# In[ ]:


rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

# Sort feature importances from GBC model trained earlier
indices = np.argsort(rf.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = rf.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("RF feature importance");


# In[ ]:


# Take some copies
rf_red_train = X_train.copy()
rf_red_test = X_test.copy()
rf_final_test = test.copy()

# Fit a model to the estimation data
rf = rf.fit(rf_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
rf_feat_red = SelectFromModel(rf, prefit = True)

# Reduce estimation, validation and test datasets
rf_X_train = rf_feat_red.transform(rf_red_train)
rf_X_test = rf_feat_red.transform(rf_red_test)
rf_final_test = rf_feat_red.transform(rf_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', rf_X_train.shape, '\nX_test: ', rf_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
rf_rfecv_train = X_train.copy()
rf_rfecv_test = X_test.copy()
rf_rfecv_final_test = test.copy()

# Initialise RFECV
rf_rfecv = RFECV(estimator = rf, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
rf_rfecv.fit(rf_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
rf_rfecv_X_train = rf_rfecv.transform(rf_rfecv_train)
rf_rfecv_X_test = rf_rfecv.transform(rf_rfecv_test)
rf_rfecv_final_test = rf_rfecv.transform(rf_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(rf_rfecv.support_)
print(rf_rfecv.ranking_)

print('X_train: ', rf_rfecv_X_train.shape, '\nX_test: ', rf_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)


# In[ ]:


# RF - feature_importances_
rf = RandomForestClassifier()

# Fit estimator to reduced dataset
rf.fit(rf_X_train, Y_train)

# Compute cross validated scores and take the mean
rf_scores = cross_val_score(rf, rf_X_train, Y_train, cv = shuff)
rf_scores = rf_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(rf_scores*100))
rf_apply_acc = metrics.accuracy_score(Y_test, rf.predict(rf_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(rf_apply_acc*100))
print("-"*50)

##############################################################################

# RF - RFECV
rf = RandomForestClassifier()

# Fit estimator to reduced dataset
rf.fit(rf_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
rf_scores = cross_val_score(rf, rf_rfecv_X_train, Y_train, cv = shuff)
rf_scores = rf_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(rf_scores*100))
rf_apply_acc = metrics.accuracy_score(Y_test, rf.predict(rf_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(rf_apply_acc*100))


# ### ***Logistic Regression***
# 
# - No feature__importances_ attribute available, so we will use the **chi-squared and extra trees reduced datasets**.

# In[ ]:


# LR - chi-squared
lr = LogisticRegression()

# Fit estimator to reduced dataset
lr.fit(chi2_reduced_train, Y_train)

# Compute cross validated scores and take the mean
lr_scores = cross_val_score(lr, chi2_reduced_train, Y_train, cv = shuff)
lr_scores = lr_scores.mean()

print('Chi2 - Mean Cross Validated Score: {:.2f}'. format(lr_scores*100))
lr_apply_acc = metrics.accuracy_score(Y_test, lr.predict(chi2_reduced_test))
print('Chi2 - Accuracy when applied to Test: {:.2f}'. format(lr_apply_acc*100))
print("-"*50)

##############################################################################

# LR - extra trees
lr = LogisticRegression()

# Fit estimator to reduced dataset
lr.fit(etc_reduced_train, Y_train)

# Compute cross validated scores and take the mean
lr_scores = cross_val_score(lr, etc_reduced_train, Y_train, cv = shuff)
lr_scores = lr_scores.mean()
print('Extra Trees - Mean Cross Validated Score: {:.2f}'. format(lr_scores*100))
lr_apply_acc = metrics.accuracy_score(Y_test, lr.predict(etc_reduced_test))
print('Extra Trees - Accuracy when applied to Test: {:.2f}'. format(lr_apply_acc*100))


# ### ***Decision Tree***
# 
# - This estimator does include **"feature__importances_"** as an attribute, hence we will use this and **RFECV**.

# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)

# Sort feature importances from GBC model trained earlier
indices = np.argsort(dt.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = dt.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("DT feature importance");


# In[ ]:


# Take some copies
dt_red_train = X_train.copy()
dt_red_test = X_test.copy()
dt_final_test = test.copy()

# Fit a model to the estimation data
dt = dt.fit(dt_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
dt_feat_red = SelectFromModel(dt, prefit = True)

# Reduce estimation, validation and test datasets
dt_X_train = dt_feat_red.transform(dt_red_train)
dt_X_test = dt_feat_red.transform(dt_red_test)
dt_final_test = dt_feat_red.transform(dt_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', dt_X_train.shape, '\nX_test: ', dt_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
dt_rfecv_train = X_train.copy()
dt_rfecv_test = X_test.copy()
dt_rfecv_final_test = test.copy()

# Initialise RFECV
dt_rfecv = RFECV(estimator = dt, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
dt_rfecv.fit(dt_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
dt_rfecv_X_train = dt_rfecv.transform(dt_rfecv_train)
dt_rfecv_X_test = dt_rfecv.transform(dt_rfecv_test)
dt_rfecv_final_test = dt_rfecv.transform(dt_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(dt_rfecv.support_)
print(dt_rfecv.ranking_)

print('X_train: ', dt_rfecv_X_train.shape, '\nX_test: ', dt_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)


# In[ ]:


# DT - feature_importances_
dt = DecisionTreeClassifier()

# Fit estimator to reduced dataset
dt.fit(dt_X_train, Y_train)

# Compute cross validated scores and take the mean
dt_scores = cross_val_score(dt, dt_X_train, Y_train, cv = shuff)
dt_scores = dt_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(dt_scores*100))
dt_apply_acc = metrics.accuracy_score(Y_test, dt.predict(dt_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(dt_apply_acc*100))
print("-"*50)

##############################################################################

# DT - RFECV
dt = DecisionTreeClassifier()

# Fit estimator to reduced dataset
dt.fit(dt_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
dt_scores = cross_val_score(dt, dt_rfecv_X_train, Y_train, cv = shuff)
dt_scores = dt_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(dt_scores*100))
dt_apply_acc = metrics.accuracy_score(Y_test, dt.predict(dt_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(dt_apply_acc*100))


# ### ***Adaboost***
# 
# - This estimator does include **"feature__importances_"** as an attribute, hence we will use this and **RFECV**.

# In[ ]:


ada = AdaBoostClassifier()
ada.fit(X_train, Y_train)

# Sort feature importances from GBC model trained earlier
indices = np.argsort(ada.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = ada.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("ADA feature importance");


# In[ ]:


# Take some copies
ada_red_train = X_train.copy()
ada_red_test = X_test.copy()
ada_final_test = test.copy()

# Fit a model to the estimation data
ada = ada.fit(ada_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
ada_feat_red = SelectFromModel(ada, prefit = True)

# Reduce estimation, validation and test datasets
ada_X_train = ada_feat_red.transform(ada_red_train)
ada_X_test = ada_feat_red.transform(ada_red_test)
ada_final_test = ada_feat_red.transform(ada_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', ada_X_train.shape, '\nX_test: ', ada_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
ada_rfecv_train = X_train.copy()
ada_rfecv_test = X_test.copy()
ada_rfecv_final_test = test.copy()

# Initialise RFECV
ada_rfecv = RFECV(estimator = ada, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
ada_rfecv.fit(ada_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
ada_rfecv_X_train = ada_rfecv.transform(ada_rfecv_train)
ada_rfecv_X_test = ada_rfecv.transform(ada_rfecv_test)
ada_rfecv_final_test = ada_rfecv.transform(ada_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(ada_rfecv.support_)
print(ada_rfecv.ranking_)

print('X_train: ', ada_rfecv_X_train.shape, '\nX_test: ', ada_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)


# In[ ]:


# ADA - feature_importances_
ada = AdaBoostClassifier()

# Fit estimator to reduced dataset
ada.fit(ada_X_train, Y_train)

# Compute cross validated scores and take the mean
ada_scores = cross_val_score(ada, ada_X_train, Y_train, cv = shuff)
ada_scores = ada_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(ada_scores*100))
ada_apply_acc = metrics.accuracy_score(Y_test, ada.predict(ada_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(ada_apply_acc*100))
print("-"*50)

##############################################################################

# ADA - RFECV
ada = AdaBoostClassifier()

# Fit estimator to reduced dataset
ada.fit(ada_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
ada_scores = cross_val_score(ada, ada_rfecv_X_train, Y_train, cv = shuff)
ada_scores = ada_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(ada_scores*100))
ada_apply_acc = metrics.accuracy_score(Y_test, ada.predict(ada_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(ada_apply_acc*100))


# ### ***Extra Trees***
# 
# - This estimator does include **"feature__importances_"** as an attribute, hence we will use this and **RFECV**.

# In[ ]:


etc = ExtraTreesClassifier()
etc.fit(X_train, Y_train)

# Sort feature importances from GBC model trained earlier
indices = np.argsort(etc.feature_importances_)[::-1]

# Visualise these with a barplot
plt.subplots(figsize=(15, 10))
g = sns.barplot(y=X_train.columns[indices], x = etc.feature_importances_[indices], orient='h', palette = mycols)
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
g.set_title("ETC feature importance");


# In[ ]:


# Take some copies
etc_red_train = X_train.copy()
etc_red_test = X_test.copy()
etc_final_test = test.copy()

# Fit a model to the estimation data
etc = etc.fit(etc_red_train, Y_train)

# Allow the feature importances attribute to select the most important features
etc_feat_red = SelectFromModel(etc, prefit = True)

# Reduce estimation, validation and test datasets
etc_X_train = etc_feat_red.transform(etc_red_train)
etc_X_test = etc_feat_red.transform(etc_red_test)
etc_final_test = etc_feat_red.transform(etc_final_test)

print("Results of 'feature_importances_':")
print('X_train: ', etc_X_train.shape, '\nX_test: ', etc_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)
print("-"*75)

##############################################################################

# Take come copies
etc_rfecv_train = X_train.copy()
etc_rfecv_test = X_test.copy()
etc_rfecv_final_test = test.copy()

# Initialise RFECV
etc_rfecv = RFECV(estimator = etc, step = 1, cv = shuff, scoring = 'accuracy')

# Fit RFECV to the estimation data
etc_rfecv.fit(etc_rfecv_train, Y_train)

# Now reduce estimation, validation and test datasets
etc_rfecv_X_train = etc_rfecv.transform(etc_rfecv_train)
etc_rfecv_X_test = etc_rfecv.transform(etc_rfecv_test)
etc_rfecv_final_test = etc_rfecv.transform(etc_rfecv_final_test)

print("Results of 'RFECV':")

# Let's see the results of RFECV
print(etc_rfecv.support_)
print(etc_rfecv.ranking_)

print('X_train: ', etc_rfecv_X_train.shape, '\nX_test: ', etc_rfecv_X_test.shape, '\nY_train: ', Y_train.shape, '\nY_test: ', Y_test.shape)


# In[ ]:


# ETC - feature_importances_
etc = ExtraTreesClassifier()

# Fit estimator to reduced dataset
etc.fit(etc_X_train, Y_train)

# Compute cross validated scores and take the mean
etc_scores = cross_val_score(etc, etc_X_train, Y_train, cv = shuff)
etc_scores = etc_scores.mean()

print('feature_importances_ - Mean Cross Validated Score: {:.2f}'. format(etc_scores*100))
etc_apply_acc = metrics.accuracy_score(Y_test, etc.predict(etc_X_test))
print('feature_importances_ - Accuracy when applied to Test: {:.2f}'. format(etc_apply_acc*100))
print("-"*50)

##############################################################################

# ETC - RFECV
etc = ExtraTreesClassifier()

# Fit estimator to reduced dataset
etc.fit(etc_rfecv_X_train, Y_train)

# Compute cross validated scores and take the mean
etc_scores = cross_val_score(etc, etc_rfecv_X_train, Y_train, cv = shuff)
etc_scores = etc_scores.mean()

print('RFECV - Mean Cross Validated Score: {:.2f}'. format(etc_scores*100))
etc_apply_acc = metrics.accuracy_score(Y_test, etc.predict(etc_rfecv_X_test))
print('RFECV - Accuracy when applied to Test: {:.2f}'. format(etc_apply_acc*100))


# **Now we have found the optimal feature set for each of the selected models!** This is a very powerful exercise and it is important to refine the dimensionality of the data for each individual model, as each estimator will work with the data differently and hence produce different importances.
# 
# - The next task, is to optimise our algorithms with their optimal feature sets.
# 
# ## 5.5 - Optimisation of selected models
# 
# ***Optimisation is the operation of fine tuning the parameters of each individual estimator to improve the performance.*** For detailed information, please see the documentation for the algorithms within the Acknowledgments.
# - **However, each estimator has many different parameters, so how do you know which to choose?**
# - Luckily, we have a really nice package at our disposal - **GridSearchCV**.
#     - This allows you to define a "grid" of parameters for the estimator to try, and it will tell you the best combination!
# - In order to find the best combination of parameters, you'll more than often want to try lots of different options and therefore your grid will be large.
# - This is quite simply, **a massive computational task**. Without gargantuan processing power, this can take days to complete - my computer is not powerful and it really did take that long.
# - **There are ways around this**:
#     - Get a more powerful computer (very expensive).
#     - Use Cloud infrastructure, such as GCP or AWS (very cheap and you get $300 free when you sign up).
#         - I tried GCP and this is an excellent option.
#         
# So, within this section **I will show and explain how to use GridSearchCV**. However for computational reasons, I will not include the full grid that I ran, and **I will leave it up to you to find the optimal combination of parameters**. Researching each algorithm and understanding it's inner workings will help guide you when choosing your grid.
# 
# We will optimise the models in the same order that we reduced the features:
# - ***KNN***
# - ***Support Vector Machine***
# - ***Gradient Boosting Classifier***
# - ***XGBoost***
# - ***Multi Layer Perceptron***
# - ***Linear SVMC***
# - ***Random Forest***
# - ***Logistic Regression***
# - ***Decision Tree***
# - ***Adaboost***
# - ***Extra Tree***
# 
# **Note**:  Some optimised models may show drastically improved results from the vanilla model, some may only be small or not at all. Nevertheless, moving forward I will continue to work with the optimised models. 
# 
# ### ***K-Nearest Neighbours***
# 
# - Using **chi-squared** reduced dataset.

# In[ ]:


# K-Nearest Neighbours
knn = KNeighborsClassifier()

# First I will present the original paramters and scores from the model before optimisation
print('BEFORE - Parameters: ', knn.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(knn_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(knn_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
knn_param_grid = {'n_neighbors': [11], 
                  'weights': ['uniform'], 
                  'algorithm': ['auto'], 
                  'leaf_size': [5],
                  'p': [1]
                 }

# Run the GridSearchCV against the above grid
gsKNN = GridSearchCV(knn, param_grid = knn_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsKNN.fit(etc_reduced_train, Y_train)

# Choose the best estimator from the GridSearch
KNN_best = gsKNN.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
KNN_pred_acc = metrics.accuracy_score(Y_test, gsKNN.predict(etc_reduced_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsKNN.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsKNN.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(KNN_pred_acc*100))


# ### ***Support Vector Machine***
# 
# - Using **extra trees** reduced dataset.

# In[ ]:


# Support Vector Machine
svc = SVC()

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', svc.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(svc_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(svc_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
svc_param_grid = {'C': [0.2],
                  'kernel': ['rbf'],
                  'degree': [1],
                  'gamma': [0.3],
                  'coef0': [0.1],
                  'max_iter': [-1],
                  'decision_function_shape': ['ovo'],
                  'probability': [True]
                 }

# Run the GridSearchCV against the above grid
gsSVC = GridSearchCV(svc, param_grid = svc_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsSVC.fit(etc_reduced_train, Y_train)

# Choose the best estimator from the GridSearch
SVC_best = gsSVC.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
SVC_pred_acc = metrics.accuracy_score(Y_test, gsSVC.predict(etc_reduced_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsSVC.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsSVC.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(SVC_pred_acc*100))


# ### ***Gradient Boosting Classifier***
# 
# - Using the **RFECV** reduced dataset for GBC.

# In[ ]:


# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=0)

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', gbc.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(gbc_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(gbc_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
gbc_param_grid = {'loss' : ["deviance"],
                  'n_estimators' : [50],
                  'learning_rate': [0.1],
                  'max_depth': [2],
                  'min_samples_split': [2],
                  'min_samples_leaf': [3]
                 }

# Run the GridSearchCV against the above grid
gsGBC = GridSearchCV(gbc, param_grid = gbc_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsGBC.fit(gbc_X_train, Y_train)

# Choose the best estimator from the GridSearch
GBC_best = gsGBC.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
GBC_pred_acc = metrics.accuracy_score(Y_test, gsGBC.predict(gbc_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsGBC.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsGBC.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(GBC_pred_acc*100))


# ### ***XGBoost***
# 
# - Using the **RFECV** reduced dataset for XGB.

# In[ ]:


# XGBoost
xgb = XGBClassifier()

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', xgb.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(xgb_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(xgb_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
xgb_param_grid = {'n_jobs': [-1],
                  'min_child_weight': [2], 
                  'max_depth': [3], 
                  'gamma': [0.1], 
                  'learning_rate': [0.05], 
                  'n_estimators': [200], 
                  'subsample': [0.75],
                  'colsample_bytree': [0.3],
                  'colsample_bylevel': [0.2], 
                  'booster': ['gbtree'], 
                  "reg_alpha": [0.1],
                  'reg_lambda': [0.6]
                 }

# Run the GridSearchCV against the above grid
gsXGB = GridSearchCV(xgb, param_grid = xgb_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsXGB.fit(xgb_X_train, Y_train)

# Choose the best estimator from the GridSearch
XGB_best = gsXGB.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
XGB_pred_acc = metrics.accuracy_score(Y_test, gsXGB.predict(xgb_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsXGB.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsXGB.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(XGB_pred_acc*100))


# ### ***Multi Layer Perceptron***
# 
# - Using the **extra trees** reduced dataset.

# In[ ]:


# Multi Layer Perceptron
mlp = MLPClassifier()

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', mlp.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(mlp_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(mlp_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters

mlp_param_grid = {'hidden_layer_sizes': [(100, )], 
                  'activation': ['relu'],
                  'solver': ['adam'], 
                  'alpha': [0.0001], 
                  'batch_size': ['auto'], 
                  'learning_rate': ['constant'], 
                  'max_iter': [300], 
                  'tol': [0.01],
                  'learning_rate_init': [0.01], 
                  'power_t': [0.7], 
                  'momentum': [0.7], 
                  'early_stopping': [True],
                  'beta_1': [0.9], 
                  'beta_2': [ 0.999], 
                  'epsilon': [0.00000001] 
                 }

# Run the GridSearchCV against the above grid
gsMLP = GridSearchCV(mlp, param_grid = mlp_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsMLP.fit(etc_reduced_train, Y_train)

# Choose the best estimator from the GridSearch
MLP_best = gsMLP.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
MLP_pred_acc = metrics.accuracy_score(Y_test, gsMLP.predict(etc_reduced_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsMLP.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsMLP.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(MLP_pred_acc*100))


# ### ***Linear Support Vector Machine***
# 
# - Using the **extra trees** reduced dataset.

# In[ ]:


# Linear Support Vector Machine
lsvc = LinearSVC()

# First I will present the original paramters and scores from the model before optimisation
print('BEFORE - Parameters: ', lsvc.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(lsvc_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(lsvc_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
lsvc_param_grid = {'tol': [0.0001],
                   'C': [0.1],
                   'fit_intercept': [True], 
                   'intercept_scaling': [0.2], 
                   'max_iter': [500]
                  }

# Run the GridSearchCV against the above grid
gsLSVC = GridSearchCV(lsvc, param_grid = lsvc_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsLSVC.fit(etc_reduced_train, Y_train)

# Choose the best estimator from the GridSearch
LSVC_best = gsLSVC.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
LSVC_pred_acc = metrics.accuracy_score(Y_test, gsLSVC.predict(etc_reduced_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsLSVC.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsLSVC.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(LSVC_pred_acc*100))


# ### ***Random Forest***
# 
# - Using the **RFECV** reduced dataset for RF.

# In[ ]:


# Random Forest
rf = RandomForestClassifier()

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', rf.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(rf_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(rf_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
rf_param_grid = {'n_estimators': [500], 
                 'criterion': ['gini'], 
                 'max_features': [None],
                 'max_depth': [5],
                 'min_samples_split': [3],
                 'min_samples_leaf': [5],
                 'max_leaf_nodes': [None], 
                 'random_state': [0,], 
                 'oob_score': [True],
                 'n_jobs': [-1] 
                 }

# Run the GridSearchCV against the above grid
gsRF = GridSearchCV(rf, param_grid = rf_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsRF.fit(rf_rfecv_X_train, Y_train)

# Choose the best estimator from the GridSearch
RF_best = gsRF.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
RF_pred_acc = metrics.accuracy_score(Y_test, gsRF.predict(rf_rfecv_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsRF.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsRF.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(RF_pred_acc*100))


# ### ***Logistic Regression***
# 
# - Using the **extra trees** reduced dataset.

# In[ ]:


# Logistic Regression
lr = LogisticRegression()

# First I will present the original paramters and scores from the model before optimisation
print('BEFORE - Parameters: ', lr.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(lr_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(lr_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
lr_param_grid = {'tol': [0.00001],
                 'C': [0.4],
                 'fit_intercept': [True], 
                 'intercept_scaling': [0.5], 
                 'max_iter': [500], 
                 'solver': ['liblinear']  
                  }

# Run the GridSearchCV against the above grid
gsLR = GridSearchCV(lr, param_grid = lr_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsLR.fit(etc_reduced_train, Y_train)

# Choose the best estimator from the GridSearch
LR_best = gsLR.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
LR_pred_acc = metrics.accuracy_score(Y_test, gsLR.predict(etc_reduced_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsLR.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsLR.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(LR_pred_acc*100))


# ### ***Decision Tree***
# 
# - Using the **feature__importances_** reduced dataset for DT.

# In[ ]:


# Decision Tree
dt = DecisionTreeClassifier()

# First I will present the original paramters and scores from the model before optimisation
print('BEFORE - Parameters: ', dt.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(dt_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(dt_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
dt_param_grid = {'criterion': ['gini'], 
                 'max_features': ['auto'],
                 'max_depth': [4],
                 'min_samples_split': [4],
                 'min_samples_leaf': [3],
                 'max_leaf_nodes': [15] 
                }

# Run the GridSearchCV against the above grid
gsDT = GridSearchCV(dt, param_grid = dt_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsDT.fit(dt_X_train, Y_train)

# Choose the best estimator from the GridSearch
DT_best = gsDT.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
DT_pred_acc = metrics.accuracy_score(Y_test, gsDT.predict(dt_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsDT.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsDT.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(DT_pred_acc*100))


# ### ***Adaboost***
# 
# - Using the **RFECV** reduced dataset for ADA.

# In[ ]:


# Adaboost
ada_dt = DecisionTreeClassifier()
ada = AdaBoostClassifier(ada_dt)

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', ada.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(ada_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(ada_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
ada_param_grid = {'n_estimators': [200], 
                  'learning_rate': [0.01],
                  'base_estimator__criterion' : ["gini"],
                  'base_estimator__max_depth': [3],
                  'base_estimator__min_samples_split': [10],
                  'base_estimator__min_samples_leaf': [1]
                 }

# Run the GridSearchCV against the above grid
gsADA = GridSearchCV(ada, param_grid = ada_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsADA.fit(ada_rfecv_X_train, Y_train)

# Choose the best estimator from the GridSearch
ADA_best = gsADA.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
ADA_pred_acc = metrics.accuracy_score(Y_test, gsADA.predict(ada_rfecv_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsADA.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsADA.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(ADA_pred_acc*100))


# ### ***Extra Trees***
# 
# - Using the **RFECV** reduced dataset for ETC.

# In[ ]:


# Extra Trees
etc = ExtraTreesClassifier()

# First I will present the original parameters and scores from the model before optimisation
print('BEFORE - Parameters: ', etc.get_params())
print("BEFORE - Mean Cross Validated Score: {:.2f}". format(etc_scores*100)) 
print("BEFORE - Test Score: {:.2f}". format(etc_apply_acc*100))
print('-'*50)

# Search grid for optimal parameters
etc_param_grid = {"max_depth": [3],
                  "min_samples_split": [3],
                  "min_samples_leaf": [5],
                  "n_estimators" :[50],
                  "criterion": ["gini"]
                 }

# Run the GridSearchCV against the above grid
gsETC = GridSearchCV(etc, param_grid = etc_param_grid, cv=shuff, scoring="accuracy", n_jobs= -1, verbose = 1)

# Fit the optimised model to the optimal reduced dataset that we found previously
gsETC.fit(etc_rfecv_X_train, Y_train)

# Choose the best estimator from the GridSearch
ETC_best = gsETC.best_estimator_

# Apply this optimised model to the validation dataset to assess prediction accuracy
ETC_pred_acc = metrics.accuracy_score(Y_test, ETC_best.predict(etc_rfecv_X_test))

# Print the best set of parameters selected by GridSearch
print('AFTER - Parameters: ', gsETC.best_params_)
print('AFTER - Best Training Score: {:.2f}'. format(gsETC.best_score_*100))
print('AFTER - Test Score: {:.2f}'. format(ETC_pred_acc*100))


# ## 5.6 - Ensemble Voting
# 
# - In the Algorithm A-Z, I explained how ensemble voting works. This works best when each of the estimators are **slightly different in terms of results**. 
# - First I will create a **correlation heatmap**, to show how closely each of the optimised estimators produce similar results.
#     - I'll do this by creating a "column" for each estimator, consisting of the model name at the top and the predictions underneath.
#     - I will then concatenate these together to create a DataFrame.

# In[ ]:


# First thing I want to do, is compare how similarly each of the optimised estimators predict the test dataset.
# I'll do this by creating "columns" using pd.Series to concatenate together
knn_ensemble = pd.Series(gsKNN.predict(etc_reduced_final_test), name = "KNN")
svc_ensemble = pd.Series(gsSVC.predict(etc_reduced_final_test), name = "SVC")
gbc_ensemble = pd.Series(gsGBC.predict(gbc_final_test), name = "GBC")
xgb_ensemble = pd.Series(gsXGB.predict(xgb_final_test), name = "XGB")
mlp_ensemble = pd.Series(gsMLP.predict(etc_reduced_final_test), name = "MLP")
lsvc_ensemble = pd.Series(gsLSVC.predict(etc_reduced_final_test), name = "LSVC")
rf_ensemble = pd.Series(gsRF.predict(rf_rfecv_final_test), name = "RF")
lr_ensemble = pd.Series(gsLR.predict(etc_reduced_final_test), name = "LR")
dt_ensemble = pd.Series(gsDT.predict(dt_final_test), name = "DT")
ada_ensemble = pd.Series(gsADA.predict(ada_rfecv_final_test), name = "ADA")
etc_ensemble = pd.Series(gsETC.predict(etc_rfecv_final_test), name = "ETC")

# Concatenate all classifier results
ensemble_results = pd.concat([knn_ensemble, svc_ensemble, gbc_ensemble, xgb_ensemble, 
                              mlp_ensemble, lsvc_ensemble, rf_ensemble, 
                              lr_ensemble, dt_ensemble, ada_ensemble, etc_ensemble], axis=1)


plt.subplots(figsize=(20, 15))
g= sns.heatmap(ensemble_results.corr(),annot=True, cmap = "YlGnBu", linewidths = 1.0)


# - From this heatmap we can see that a lot of the estimators correlate very highly, indicating that the **predicted results are very similar**.
# - However, we do have some models that show some differences.
# - This is what we want, therefore I will now go ahead and build a voting ensemble. **I will then fit this model to the original dataset with 35 columns**.
#     - The reason why I am doing this is because the **optimised models all used different datasets, of different sizes**. 
#     - For the voting ensemble to work, **we are required to fit the model on one feature set and one target feature**.
#     - Hence, I chose the **original estimation and validation set** to score the training data and output a testing accuracy.
#     
# **Note**: I will exclude the Linear Support Vector Machine from this Voting Classifier. The reason for this is because I am creating a voting ensemble from highly calibrated models after optimisation. Therefore, "soft" voting is best suited. Soft voting requires an output of probabilities, of which LSVC is incapable of producing.

# In[ ]:


# Set up the voting classifier with all of the optimised models I have built.
votingC = VotingClassifier(estimators=[('KNN', KNN_best), ('SVC', SVC_best),('GBC', GBC_best),
                                       ('XGB', XGB_best), ('MLP', MLP_best), ('RF', RF_best), 
                                       ('LR', LR_best), ('DT', DT_best), ('ADA', ADA_best), 
                                       ('ETC', ETC_best)], voting='soft', n_jobs=4)

# Fit the model to the training data
votingC = votingC.fit(X_train, Y_train)

# Take the cross validated training scores as an average
votingC_scores = cross_val_score(votingC, X_train, Y_train, cv = shuff)
votingC_scores = votingC_scores.mean()

# Print the results and include how accurately the voting ensemble was able to predict
# the validation dataset
print('Mean Cross Validated Score: {:.2f}'. format(votingC_scores*100))
votingC_apply_acc = metrics.accuracy_score(Y_test, votingC.predict(X_test))
print('Accuracy when applied to Test: {:.2f}'. format(votingC_apply_acc*100))


# ## 5.7 - Output predictions
# 
# - We are now at the final stage, **outputting our predictions into a CSV file and uploading them to Kaggle**.
# - For each of the optimised models fit to the estimation dataset, **I will now predict the survival values for the test dataset**.
# - Then, I will **combine this predicted column with the "PassengerId"** column from the original test dataset that we imported. These passengers are the 418 passengers that we wanted to predict survival for.
#     - These two columns will be concatenated together, as required by the upload requirements on Kaggle.

# In[ ]:


# Output predictions into a csv file for Kaggle upload
KNN_test_pred = KNN_best.predict(etc_reduced_final_test)
KNN_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": KNN_test_pred})
# KNN_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('K-Nearest Neighbours predictions uploaded to CSV!')


# Output predictions into a csv file for Kaggle upload
SVC_test_pred = SVC_best.predict(etc_reduced_final_test)
SVC_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": SVC_test_pred})
# SVC_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Support Vector Machine Classifier predictions uploaded to CSV!')


# Output predictions into a csv file for Kaggle upload
GBC_test_pred = GBC_best.predict(gbc_final_test)
GBC_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": GBC_test_pred})
# GBC_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Gradient Boosting Classifier predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
XGB_test_pred = XGB_best.predict(xgb_final_test)
XGB_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": XGB_test_pred})
# XGB_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('XGBoost predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
MLP_test_pred = MLP_best.predict(etc_reduced_final_test)
MLP_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": MLP_test_pred})
# MLP_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Multi Layer Perceptron predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
LSVC_test_pred = LSVC_best.predict(etc_reduced_final_test)
LSVC_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": LSVC_test_pred})
# LSVC_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Linear Support Vector Machine Classifier predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
RF_test_pred = RF_best.predict(rf_rfecv_final_test)
RF_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": RF_test_pred})
# RF_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Random Forest predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
LR_test_pred = LR_best.predict(etc_reduced_final_test)
LR_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": LR_test_pred})
# LR_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Logistic Regression predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
DT_test_pred = DT_best.predict(dt_final_test)
DT_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": DT_test_pred})
# DT_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Decision Tree predictions uploaded to CSV!')

# Output predictions into a csv file for Kaggle upload
ADA_test_pred = ADA_best.predict(ada_rfecv_final_test)
ADA_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": ADA_test_pred})
# ADA_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Adaboost predictions uploaded to CSV!')

# Extra Trees
ETC_test_pred = ETC_best.predict(etc_rfecv_final_test)
ETC_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": ETC_test_pred})
# ETC_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Extra Trees predictions uploaded to CSV!')

# Voting
votingC_test_pred = votingC.predict(test)
votingC_submission = pd.DataFrame({"PassengerId": raw_test["PassengerId"], "Survived": votingC_test_pred})
# votingC_submission.to_csv('[enter your directory here]', index=False)

print('-'*50)
print('Voting predictions uploaded to CSV!')

print('-'*50)
print("Here's a preview of what the CSV looks like...")
votingC_submission.head()


# - The final thing to do is simply **upload the CSV files to the Kaggle competition!** I suggest uploading a few of the highest performers , then go back to your script and **iterate to improve** them!
# - This iterative loop of improvement can be never-ending... this is where your judgement must **decide when the best time to stop is**.

# ***

# 
# # 6.
# ## Conclusion
# 
# - Throughout this journey, we have **explored the data thoroughly** and **conducted various tests** and experiments with the view of understanding it well enough to make an accurate prediction.
# - My aim for creating this notebook was to **share my insight**, **show you my workflow**, and to **educate**. I hope this has helped you!
# - **With Data Science, there are various ways to do any task**. This is just my solution, and my ideas. **I'd be keen to hear how you'd do things differently or how I could improve this notebook**.
# 
# ### ***Please leave a comment, follow me and upvote if you enjoyed it!***

# In[ ]:


Image(filename='../input/titanic-images/end2.png')


# ***

# ## Acknowledgements:
# 
# - **CNN**, for facts about the Titanic - https://edition.cnn.com/2013/09/30/us/titanic-fast-facts/index.html
# - **Scikit-learn documentation** - http://scikit-learn.org/stable/
# - **Dummies Guide to Transforming Distributions** - http://www.dummies.com/programming/big-data/data-science/transforming-distributions-machine-learning/
# - **Multicollinearity** - https://en.wikipedia.org/wiki/Multicollinearity
# - **Decision Trees**:
#     - http://mines.humanoriented.com/classes/2010/fall/csci568/portfolio_exports/lguo/decisionTree.html
#     - https://medium.com/machine-learning-101/chapter-3-decision-trees-theory-e7398adac567
# - **Random Forest**:
#     - https://medium.com/machine-learning-101/chapter-5-random-forest-classifier-56dc7425c3e1
# - **Extra Trees**:
#     - https://www.quora.com/What-is-the-extra-trees-algorithm-in-machine-learning
# - **Adaboost**:
#     - https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/
#     - https://medium.com/machine-learning-101/https-medium-com-savanpatel-chapter-6-adaboost-classifier-b945f330af06
# - **XGBoost**:
#     - https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
#     - http://xgboost.readthedocs.io/en/latest/model.html
# - **Gradient Boosting Classifier**:
#     - https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/
#     - https://www.analyticsvidhya.com/blog/2015/09/complete-guide-boosting-methods/
# - **Support Vector Machine & Linear Support Vector Machine**:
#     - https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/
#     - https://machinelearningmastery.com/support-vector-machines-for-machine-learning/
# - **K-Nearest Neighbours**:
#     - https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/
# - **Perceptron & Multi Layer Perceptron**:
#     - https://machinelearningmastery.com/neural-networks-crash-course/
#     - https://en.wikipedia.org/wiki/Multilayer_perceptron
# - **Logistic Regression**:
#     - http://www.statisticssolutions.com/conduct-interpret-logistic-regression/
# - **Gaussian Naive Bayes**:
#     - https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
#     - https://machinelearningmastery.com/naive-bayes-for-machine-learning/
# 
# - **Great Kernels that I found useful**:
#     - **Erik Bruin** - https://www.kaggle.com/erikbruin/titanic-2nd-degree-families-and-majority-voting
#     - **LD Freeman** - https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
#     - **Yassine Ghouzam** - https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
#     - **Manav Sehgal** - https://www.kaggle.com/startupsci/titanic-data-science-solutions
