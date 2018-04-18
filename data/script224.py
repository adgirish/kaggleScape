
# coding: utf-8

# ![title](https://media.giphy.com/media/4hN2NSrwKe23e/giphy.gif)

# # Introduction
# <a id="introduction" ></a>
# ***
# This kernel is mostly for beginners with detailed statistical analysis of Titanic data set along with Machine learning models. I am super excited to share my first kernel with the Kaggle community, and I think my journey of data science can take a leap from this community.  As I go on in this journey and learn new topics, I will incorporate them into new updates. So, check for them and please comment if you have a suggestion to make this kernel better!! I will gladly work with the comments cause I truly believe, Kaggle, is an awesome community and we learn the most through collaborating. Going back to the topics of this kernel, I will do more in-depth visualizations to explain the data, and the machine learning models will be used to predict passenger survival status. Let's get started.

# ## Table of contents
# ***
# - [Introduction](#introduction)
# - [Kernel Goals](#aboutthiskernel)
# - [Part 1: Importing Necessary Modules](#import_libraries)
#     - [1a. Libraries](#import_libraries)
#     - [1b. Load datasets](#load_data)
#     - [1c. About this dataset](#aboutthisdataset)
#     - [1d. Tableau Visualization](#tableau_visualization)
# - [Part 2: Overview and Cleaning the Data](#scrubbingthedata)
#     - [2a. Dealing with missing values](#dealwithnullvalues)
#     - [2b. Dealing with categorical variables(feature engineering)](#dealwithvariables)
# - [Part 3: Visualization and Feature Relations](#visualization_and_feature_relations)
#     - [3a. Gender and Survived](#gender_and_survived)
#     - [3b. Pclass and Survived](#pclass_and_survived)
#     - [3c. Fare and Survived](#fare_and_survived)
#     - [3d. Age and Survived](#age_and_survived)
#     - [3e. Combined Feature relations](#combined_feature_relations)
# - [Part 4: Statistical Overview](#statisticaloverview)
#     - [4a. Correlation Matrix and Heatmap](#heatmap)
#     - [4b. Statistical Test for Correlation](#statistical_test)
#     - [4c. The T-Test](#t_test)
# - [Part 5: Feature Engineering](#feature_engineering)
# - [Part 6: Pre-Modeling Tasks](#pre_model_tasks)
#     - [6a. Separating dependent and independent variables](#dependent_independent)
#     - [6b. Splitting the training data](#split_training_data)
#     - [6c. Feature Scaling](#feature_scaling)
# - [Part 7: Modeling the Data](#modelingthedata)
#     - [7a. Logistic Regression](#logistic_regression)
#     - [7b. K-Nearest Neighbors(KNN)](#knn)
#     - [7c. Gaussian Naive Bayes](#gaussian_naive)
#     - [7d. Support Vector Machines](#svm)
#     - [7e. Decision Tree Classifier](#decision_tree)
#     - [7f. Bagging on Decision Tree Classifier](#bagging_decision)
#     - [7g. Random Forest Classifier](#random_forest)
#     - [7h. Gradient Boosting Classifier](#gradient_boosting)
#     - [7i. XGBClassifier](#XGBClassifier)
#     - [7j. AdaBoost Classifier](#adaboost)
#     - [7k. Extra Tree Classifier](#extra_tree)
#     - [7l. Gaussian Process Classifier](#GaussianProcessClassifier)
#     - [7m. Voting Classifier](#voting_classifier)
# - [Part 8: Submit Test Predictions](#submit_predictions)
#     
# - [ Credits](#credits)

# # Kernel Goals
# <a id="aboutthiskernel"></a>
# ***
# There are two primary goals of this kernel.
# - To do a statistical and exploratory data analysis of how some group of people was survived more than others through visualization.  
# - And to create machine learning models that can predict the chances of passengers survival.

# # Part 1: Importing Necessary Libraries and datasets
# ***
# <a id="import_libraries**"></a>
# ## 1a. Libraries
# 

# In[24]:


# Import necessary modules for data analysis and data visualization. 
# Data analysis modules
import pandas as pd
import numpy as np

# Visualization libraries
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

## Machine learning libraries
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV



## Ignore warning
# import warnings
# warnings.filterwarnings('ignore')


# ## 1b. Load datasets
# <a id="load_data"></a>
# ***

# In[25]:


## Importing the datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ## 1c. About This Dataset
# <a id="aboutthisdataset"></a>
# ***
# **The following information about this dataset was derived from kaggle. **
# 
# The data has split into two groups:
# 
# - training set (train.csv)
# - test set (test.csv)
# 
# ***The training set includes our target variable, passenger survival status***(also known as the ground truth from the Titanic tragedy) along with other independent features like gender, class, fare, and Pclass. 
# 
# The test set should be used to see how well my model performs on unseen data. ***The test set does not provide passengers survival status***. We are going to use our model to predict passenger survival status.
# 
# Now let's go through the features and describe a little. There are couple of different type of features, They are..
# ***
#  **Nominal:** Variables are used to **name** or label a series of values.
# - **Sex**
#     - 0 = Female
#     - 1 = Male
# - **Survived** (Our outcome or dependent variable)
#     - 0 = No
#     - 1 = Yes
# ***
# **Ordinal:**  Variables that provide good information about the **order of choices**, such as Pclass in this dataset.
# - **Sibsp** (# of siblings/spouses aboard the Titanic)    
# - **Parch**(# of parents/children aboard the Titanic)   
# - **Pclass** (A proxy for socio-economic status (SES)) 
#     - 1 = 1st(Upper)
#     - 2 = 2nd(Middle) 
#     - 3 = 3rd(Lower)
# ***
# **Categorical:**
# - **Embarked**(Port of Embarkation)    
#     - C = Cherbourg, 
#     - Q = Queenstown, 
#     - S = Southampton
# ***
# **Numeric/Continous:** This type is just continous variable such as the **fare** feature in this dataset. 
# - **Age**
# - **Fare**
# ***
# 
# - **Passenger ID**(assumed to be random unique identifiers for each passengers. Doesn't have any relations to the dataset. Will be removed from the dataset.)
# - **Ticket** (Ticket number for passenger.)
# - **Cabin**( Cabin name for passenger.) 
# 

# ## 1d. Tableau visualization
# <a id='tableau_visualization'></a>
# ***
# I have incorporated a tableau visualization below of the training data. This visualization... 
# * is for us to have an overview and understand the dataset. 
# * is done without making any changes(including Null values) to any features of the dataset.
# ***
# Let's get a better perspective of the dataset through this visualization.
# 

# In[26]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1516349898238' style='position: relative'><noscript><a href='#'><img alt='An Overview of Titanic Training Dataset ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic_data_mining&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Titanic_data_mining&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic_data_mining&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1516349898238');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# # Part 2: Overview and Cleaning the Data
# <a id="cleaningthedata"></a>
# ***

# In[27]:


## saving passenger id in advance in order to submit later. 
passengerid = test.PassengerId


# In[28]:


## We will drop PassengerID and Ticket since it will be useless for our data. 
train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)


# This dataset is almost clean. However, before we jump into visualization and machine learning models, lets analyze and see what we have here.

# In[29]:


print (train.info())
print ("*"*40)
print (test.info())


# It looks like, the features have unequal amount of data entries and they have multiple types variables. This can happen for the following reasons...
# * We may have missing values in our features.
# * We may have categorical features. 
# * We may have alphanumerical or/and text features. 
# 

# ## 2a. Dealing with Missing values
# <a id="dealwithnullvalues"></a>
# ***

# In[30]:


print (train.isnull().sum())
print (''.center(20, "*"))
print (test.isnull().sum())


# We see that in both **train** and **test** dataset have missing values. Let's go through each of them and fix them. 

# ### Embarked feature
# ***

# In[31]:


print (train.Embarked.value_counts(dropna=False))
##print ((train.Embarked.value_counts(dropna=False)/len(train.Embarked)*100))
print ("Let's see the perecentage of each unique values including NA's..")
print (round(train.Embarked.value_counts(dropna=False, normalize=True)*100, 2))


# It looks like there are only two null values( ~ 0.22 %) in the Embarked feature which is equvaluet to only 0.22% of all the values, we can replace these with the mode value "S." 

# In[32]:


train[train.Embarked.isnull()]


# In[33]:


## Replacing the null values in the Embarked column with the mode. 
train.Embarked.fillna(train.Embarked.mode()[0], inplace=True)
## Checking back to see if we have done it right. 
train.Embarked.isnull().sum()


# ### Cabin Feature
# ***

# In[34]:


print(train.Cabin.isnull().sum()/len(train.Cabin))
print(test.Cabin.isnull().sum()/len(test.Cabin))


# Approximately 77% of Cabin feature is missing in the training data and 78% missing on the test data. We have two choices, we can either get rid of the whole feature, or we can brainstorm a little and find an appropriate way to put them in use. For example...
# * We may say passengers with cabin records had a higher socio-economic-status then others. 
# * We may also say passengers with cabin records were more likely to be taken into consideration for the rescue mission. 
# 
# I think it's would be wise to keep the data. We will assign all the null values as **"N"** for now and will put cabin column to good use in the feature engineering section.

# In[35]:


train.Cabin.fillna("N", inplace=True)
test.Cabin.fillna("N", inplace=True)


# All the cabin names start with an english alphabet following by digits. We can group these cabins by the alphabets. 

# In[36]:


train.Cabin = [i[0] for i in train.Cabin]
test.Cabin = [i[0] for i in test.Cabin]


# In[37]:


train.Cabin.value_counts()


# In[38]:


print (train.isnull().sum())
print(''.center(15,'*'))
print(test.isnull().sum())


# ### Fare Feature
# ***

# In[39]:


test[test.Fare.isnull()]


# In[40]:


## replace the test.fare null values with test.fare mean
test.Fare.fillna(test.Fare.mean(), inplace=True)


# In[41]:


test.Fare.isnull().sum()


# ### Age Feature
# ***

# In[42]:


((train.Age.isnull().sum()/len(train))*100)


# 
# 
# There are different ways to deal with Null values. Some standard approaches are mean, median and mode. However, we will take a different approach since **~20% data in the Age column is missing** and it would be unwise to replace the missing values with median, mean or mode. We will use pythons library **fancyimpute** where I can use **K Nearest neighbors(KNN)** machine learning model to impute nearest neighbor value instead of  Null value. In order to run the fancyimpute we will first have to convert categorical variables into numerical variables. We will keep the age column unchanged for now and work on that in the feature engineering section. 

# # Part 3. Visualization and Feature Relations
# <a id="visualization_and_feature_relations" ></a>
# ***
# Before dive into finding relations between different features and our dependent variable(survivor) let us create some predictions about how we think the relations might turnout among features.
# 
# **Predictions:**
# - Gender: More female survived than male
# - Pclass: Higher socio-economic status passenger survived more than others. 
# - Age: Younger passenger survived more than other passengers. 
# 
# Now, let's see how the features are related to each other by creating some visualizations. 
# 
# 

# ## 3a. Gender and Survived
# <a id="gender_and_survived"></a>
# ***

# In[43]:


import matplotlib.ticker as mtick

pal = {'male':"skyblue", 'female':"Pink"}
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex", 
            y = "Survived", 
            data=train, 
            palette = pal,
            linewidth=2 )
plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25)
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Sex",fontsize = 15);



# This bar plot above shows the distribution of female and male survived. The **x_label** shows gender and the **y_label** shows % of passenger survived. This bar plot shows that ~74% female passenger survived while only ~19% male passenger survived.

# In[44]:


pal = {1:"seagreen", 0:"gray"}
sns.set(style="darkgrid")
plt.subplots(figsize = (15,8))
ax = sns.countplot(x = "Sex", 
                   hue="Survived",
                   data = train, 
                   linewidth=2, 
                   palette = pal
)

## Fixing title, xlabel and ylabel
plt.title("Passenger Gender Distribution - Survived vs Not-survived", fontsize = 25)
plt.xlabel("Sex", fontsize = 15);
plt.ylabel("# of Passenger Survived", fontsize = 15)

## Fixing xticks
#labels = ['Female', 'Male']
#plt.xticks(sorted(train.Sex.unique()), labels)

## Fixing legends
leg = ax.get_legend()
leg.set_title("Survived")
legs = leg.texts
legs[0].set_text("No")
legs[1].set_text("Yes")
plt.show()


# This count plot shows the actual distribution of male and female passengers that survived and did not survive. It shows that among all the females ~ 230 survived and ~ 70 did not survive. While among male passengers ~110 survived and ~480 did not survive. 
# 
# **Summary**
# ***
# - As we suspected, female passengers have survived at a much better rate than male passengers. 
# - It seems about right since females and children were the priority. 

# ## 3b. Pclass and Survived
# <a id="pcalss_and_survived"></a>
# ***

# In[45]:


plt.subplots(figsize = (15,10))
sns.barplot(x = "Pclass", 
            y = "Survived", 
            data=train, 
            linewidth=2)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
plt.xlabel("Socio-Economic class", fontsize = 15);
plt.ylabel("% of Passenger Survived", fontsize = 15);
labels = ['Upper', 'Middle', 'Lower']
#val = sorted(train.Pclass.unique())
val = [0,1,2] ## this is just a temporary trick to get the label right. 
plt.xticks(val, labels);


# - It looks like ...
#     - ~ 63% first class passenger survived titanic tragedy, while 
#     - ~ 48% second class and 
#     - ~ only  24% third class passenger survived. 
# 
# 

# In[46]:


# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
## I have included to different ways to code a plot below, choose the one that suites you. 
ax=sns.kdeplot(train.Pclass[train.Survived == 0] , 
               color='gray',
               shade=True,
               label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'] , 
               color='g',
               shade=True, 
               label='survived')
plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Passenger Class", fontsize = 15)
## Converting xticks into words for better understanding
labels = ['Upper', 'Middle', 'Lower']
plt.xticks(sorted(train.Pclass.unique()), labels);


# This kde plot is pretty self explanatory with all the labels and colors. Something I have noticed that some readers might find questionable is that, the lower class passengers have survived more than second class passnegers. It is true since there were a lot more third class passengers than first and second. 
# 
# **Summary**
# ***
# First class passenger had the upper hand during the tragedy than second and third class passengers. You can probably agree with me more on this, when we look at the distribution of ticket fare and survived column. 
# 
# 

# ## 3c. Fare and Survived
# <a id="fare_and_survived"></a>
# ***

# In[47]:


# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')
plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Fare", fontsize = 15)



# This plot shows somethig really interesting..
# - The spike in the plot under 100 dollar represents that alot of passenger who bought ticket with in that range did not survive. 
# - When fare is approximately more than 280 dollars, there is no gray shade which mean, either everyone passed that fare point  survived or may be there is an outlier that clouds our judge ment. Let's check...

# In[48]:


train[train.Fare > 280]


# Yap, like we assumed, it was a outlier with fare of $512 which is so far away from other fare points. We can sure delete this point however, we will keep it for now. 

# ## 3d. Age and Survived
# <a id="age_and_survived"></a>
# ***

# In[49]:


# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')
plt.title('Age Distribution - Surviver V.S. Non Survivors', fontsize = 25)
plt.xlabel("Age", fontsize = 15)
plt.ylabel('Frequency', fontsize = 15);


# There is nothing out of the ordinary of about this plot, except the very left part of the distribution. It shows that children and infants were the priority. 

# ## 3e. Combined Feature Relations
# <a id='combined_feature_relations'></a>
# ***
# 

# In[50]:


pal = {1:"seagreen", 0:"gray"}
g = sns.FacetGrid(train,size=5, col="Sex", row="Survived", margin_titles=True, hue = "Survived",
                  palette=pal)
g = g.map(plt.hist, "Age", edgecolor = 'white');
g.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)


# Facetgrid is a just a good way to visualize multiple variables and how they are related to each other. From the charts above, we can get a quick recap of what we know so far, which is, "female passengers survived more than male passengers"

# In[51]:


g = sns.FacetGrid(train,size=5, col="Sex", row="Embarked", margin_titles=True, hue = "Survived",
                  palette = pal
                  )
g = g.map(plt.hist, "Age", edgecolor = 'white').add_legend();
g.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)


# In[52]:


g = sns.FacetGrid(train, size=5,hue="Survived", col ="Sex", margin_titles=True,
                palette=pal,)
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
g.fig.suptitle("Survived by Sex, Fare and Age", size = 25)
plt.subplots_adjust(top=0.85)


# In[53]:


g = sns.FacetGrid(train, size=8, aspect=0.5, hue="Survived", 
                  col ="Pclass", margin_titles=True,
                palette=pal)
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
g.fig.suptitle("Survived by Pclass, Fare and Age", size = 25)
plt.subplots_adjust(top=0.85)


# In[54]:


g = sns.factorplot(y = "Survived",
               x = "Sex",
               col = "Embarked",
               row = "Cabin",
              data = train, 
              kind = "bar",
              ci = None,
               margin_titles=True,
              size = 4)
g.fig.suptitle("Survived, Embarked and Cabin", fontsize =25)
plt.subplots_adjust(top=0.95)


# In[55]:


sns.factorplot(x = "Parch", y = "Survived", data = train,kind = "point",size = 8)
plt.title("Factorplot of Parents/Children survived", fontsize = 25)
plt.subplots_adjust(top=0.85)


# In[56]:


sns.factorplot(x = "SibSp", y = "Survived", data = train,kind = "point",size = 8)
plt.title('Factorplot of Sibilings/Spouses survived', fontsize = 25)
plt.subplots_adjust(top=0.85)


# In[57]:


# Placing 0 for female and 
# 1 for male in the "Sex" column. 
train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)


# So, we did a couple of steps in part 2, let's review...
# ***
# - We dropped **PassengerId** feature since it will not aid us in any way in creating our model. 
# - We assigned Null values in the **Cabin** feature into **N**. 
# - We used mean for **Fare** feature and mode for **Embarked** feature to fill up their missing values. 
# - We created dummy variables for **Sex** since they are categorical variable.

# # Part 4: Statistical Overview
# <a id="statisticaloverview"></a>
# ***

# ![title](https://cdn-images-1.medium.com/max/400/1*hFJ-LI7IXcWpxSLtaC0dfg.png)

# In[58]:


train.head()


# In[59]:


train.dtypes


# In[60]:


train[['Pclass', 'Survived']].groupby("Pclass").mean()


# In[61]:


# Overview(Survived vs non survied)
survived_summary = train.groupby("Survived")
survived_summary.mean()


# In[62]:


survived_summary = train.groupby("Sex")
survived_summary.mean()


# In[63]:


survived_summary = train.groupby("Pclass")
survived_summary.mean()


# 
# There are a couple of points provided below, which should be noted from this statistical overview.
# - This data set has 891 raw and 9 columns. 
# - only 38% passenger survived during that tragedy.
# - ~74% female passenger survived, while only ~19% male passenger survived. 
# - ~63% first class passengers survived, while only 24% lower class passenger survived.
# 
# 

# ## 4a. Correlation Matrix and Heatmap
# <a id="heatmap"></a>
# ***

# In[64]:


abs(train.corr()['Survived']).sort_values(ascending = False)


# In[65]:


## Let's see the correlation between target variables and other features.
train.corr()['Survived']


# In[66]:


## get the most important variables 
corr = train.corr()**2
corr.Survived.sort_values(ascending=False)


# In[67]:


## heatmeap to see the correlation between features. 
# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (15,12))
sns.heatmap(train.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20);


# #### Positive Correlation Features:
# - Fare and Survived: 0.26
# 
# #### Negative Correlation Features:
# - Fare and Pclass: -0.55
# - Gender and Survived: -0.54
# - Pclass and Survived: -0.34
# 
# 
# **So, Let's analyze these correlations a bit.** We have found some moderately strong correlations between different features. There is a positive correlation between Fare and Survived rated. This can be explained by saying that, the passenger who paid more money for their ticket were more likely to survive. This theory aligns with one other correlation which is the correlation between Fare and Pclass(-0.55). This relationship can be explained by saying that first class passenger(1) paid more for fare then second class passenger(2), similarly second class passenger paid more than the third class passenger(3). This theory can also be supported by mentioning another Pclass correlation with our dependent variable, Survived. The correlation between Pclass and Survived is -0.34. This can also be explained by saying that first class passenger had a better chance of surviving than the second or the third and so on.
# 
# However, the biggest correlation with our dependent variable is the Gender variable, which basically is the info of whether the passenger was male or female. this is a negative correlation with a magnitude of -0.54 which definitely points towards some undeniable insights. Let's do some statistics to see how undeniable this insight is. 

# ## 4b. Statistical Test for Correlation
# <a id="statistical_test"></a>
# ***
# #### One-Sample T-Test(Measuring male-female ratio)
# A one-sample t-test checks whether a sample mean differs from the population mean. Since *Gender* has the highest correlation with the dependent variable *Survived*, we can test to see if the mean *Gender* that survived differs from mean Gender that did not survive. 
# ***
# **Note:** There are two types of outcome in the Gender variable, **"0"** and **"1"**, ***"0" being the gender as female and "1" being the gender as "male"***. Therefore, while determining average of Gender, we should keep in mind that, **an increase in the average gender means an increase in male passengers, similarly a decrease in the average gender mean a reduction in male passengers, therefore increase in female passengers and so on**
# ***
# 
# ***Hypothesis Testing***: Is there a significant difference in the mean Gender between the passenger who survived and passenger who did not survive?
# 
# - ** Null Hypothesis(H0)** The null hypothesis would be that there is no difference in the mean Gender between the passenger who survived and passenger who did not survive. 
# - ** Alternative Hypothesis(H1):** The alternative hypothesis would be that there is a difference in the mean Gender between the passenger who survived and those who did not.
# 
# 
# 

# In[68]:


## Lets compare the means of gender for passengers who survived and the passengers who didnot survive. 

avg_survived = train[train["Survived"]==1]["Sex"].mean()
print ("The average Gender for the passengers who survived is: " + str(avg_survived))
avg_not_survived = train[train["Survived"]==0]["Sex"].mean()
print ("The average Gender for the passengers who did not survive is: " + str(avg_not_survived))


# ## 4c. The T-Test
# <a id='t_test'></a>
# ***
# Let's conduct a T-Test at **95% confidence level**. The T-Test statistics will tell us how much the sample of the survived passengers(in statistics languange this is the sample) alligns with the means of passenger(in statistics language this is population mean) who did not survive. if the test statistics(t-statistics) do not align(fall with in the critical value) 95% of the time, we reject the null hypothesis that **the sample comes from the same distribution as the passenger population**. In order to conduct a one sample t-test, we can use the **tats.ttest_1samp()** function. 

# In[69]:


import scipy.stats as stats
stats.ttest_1samp(a=  train[train['Survived']==1]['Sex'], # Sample of passenger who survived. 
                  popmean = avg_not_survived)  # Mean of passenger who did not survive.


# #### T-test Quantile
# ***
# T-test Quantile helps us to find that critical value area. In other words it helps us to draw a border between rejection area and non rejection area. If the t-statistics value we have acquired above fall outside the quantile or non-rejection area then we reject the null hypothesis. We can find the quantiles with **stats.t.ppf()**

# In[70]:


## Finding the t-test quantile. 
degree_freedom = len(train[train['Survived']==1])

LQ = stats.t.ppf(0.025,degree_freedom)  # Left Quartile

RQ = stats.t.ppf(0.975,degree_freedom)  # Right Quartile

print ('The left quartile range of t-distribution is: ' + str(LQ))
print ('The right quartile range of t-distribution is: ' + str(RQ))


# #### T-Test Result
# ***
# 
# Our T-test result shows that the **t-statistics is -21.15**, while our left quantile range for t-distribution is -1.9669 and our right quantile range for t-distribution is 1.9669. 

# #### One-Sample T-Test Summary
# ***
# - **Our t-test score is T-Test = -21.1516** 
# - **P-value = 4.97**
# - **We reject the null hypothesis**
# ***
# **We reject the null hypothesis because ..**
# - The T-statistics is outside the quantiles(Non-rejection Region)
# - The P-value is less than the alpha(.5) or confidence level of 95%. 
# ***
# Based on the statistics from One sample t-test, we decided to **reject the null hypothesis**. We rejected the null hypothesis based on two core components. The first one is the T-statistics of -21.15, which is far away from the quantile of -1.966. In addition to that the P-value of 4.97 which is lower than the chosen alpna value(confidence level) of .05 or 5%.
# 
# Even though we have some statistics to conclude the null hypothesis but this does not mean that there is a practical significance in mean of gender between the passenger who survived and the passenger who did not survive. A more detailed conducted experiment with more data analysis and visualization can help us find more insights about this tragedy. 
# 
# 

# # Part 5: Feature Engineering
# <a id="feature_engineering"></a>
# ***

# ## name_length

# In[71]:


train['name_length'] = [len(i) for i in train.Name]
test['name_length'] = [len(i) for i in test.Name]


# In[72]:


def name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'good'
    else:
        a = 'long'
    return a


# In[73]:


train['nLength_group'] = train['name_length'].map(name_length_group)
test['nLength_group'] = test['name_length'].map(name_length_group)


# In[74]:


## cuts the column by given bins based on the range of name_length
#group_names = ['short', 'medium', 'good', 'long']
#train['name_len_group'] = pd.cut(train['name_length'], bins = 4, labels=group_names)


# ## title

# In[75]:


## get the title from the name
train["title"] = [i.split('.')[0] for i in train.Name]
train["title"] = [i.split(',')[1] for i in train.title]
test["title"] = [i.split('.')[0] for i in test.Name]
test["title"]= [i.split(',')[1] for i in test.title]


# In[76]:


train.title.value_counts()


# In[77]:


#rare_title = ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col']
#train.Name = ['rare' for i in train.Name for j in rare_title if i == j]
train["title"] = [i.replace('Ms', 'Miss') for i in train.title]
train["title"] = [i.replace('Mlle', 'Miss') for i in train.title]
train["title"] = [i.replace('Mme', 'Mrs') for i in train.title]
train["title"] = [i.replace('Dr', 'rare') for i in train.title]
train["title"] = [i.replace('Col', 'rare') for i in train.title]
train["title"] = [i.replace('Major', 'rare') for i in train.title]
train["title"] = [i.replace('Don', 'rare') for i in train.title]
train["title"] = [i.replace('Jonkheer', 'rare') for i in train.title]
train["title"] = [i.replace('Sir', 'rare') for i in train.title]
train["title"] = [i.replace('Lady', 'rare') for i in train.title]
train["title"] = [i.replace('Capt', 'rare') for i in train.title]
train["title"] = [i.replace('the Countess', 'rare') for i in train.title]
train["title"] = [i.replace('Rev', 'rare') for i in train.title]



# In[78]:


#rare_title = ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col']
#train.Name = ['rare' for i in train.Name for j in rare_title if i == j]
test['title'] = [i.replace('Ms', 'Miss') for i in test.title]
test['title'] = [i.replace('Dr', 'rare') for i in test.title]
test['title'] = [i.replace('Col', 'rare') for i in test.title]
test['title'] = [i.replace('Dona', 'rare') for i in test.title]
test['title'] = [i.replace('Rev', 'rare') for i in test.title]


# In[79]:


train.title.value_counts()


# ## has_cabin

# In[80]:


train["has_cabin"] = [0 if i == 'N'else 1 for i in train.Cabin]
test["has_cabin"] = [0 if i == 'N'else 1 for i in test.Cabin]


# In[81]:


train.has_cabin.value_counts()


# ## Cabin feature

# In[82]:


print (sorted(train.Cabin.unique()))
print (''.center(45,'*'))
print(sorted(test.Cabin.unique()))


# It looks like there is one more unique values in the training data. This will complicate running machine learning models. therefore when you create dummy variables, we will have to make sure to drop **T** column from training data. 

# ## child feature

# In[83]:


## We are going to create a new feature "age" from the Age feature. 
train['child'] = [1 if i<16 else 0 for i in train.Age]
test['child'] = [1 if i<16 else 0 for i in test.Age]


# In[84]:


train.child.value_counts()


# ## family_size feature
# 

# In[85]:


## Family_size seems like a good feature to create
train['family_size'] = train.SibSp + train.Parch+1
test['family_size'] = test.SibSp + test.Parch+1


# In[86]:


def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a


# In[87]:


train['family_group'] = train['family_size'].map(family_group)
test['family_group'] = test['family_size'].map(family_group)


# ## is_alone feature

# In[88]:


train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]
test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]


# ## Ticket feature

# In[89]:


train.Ticket.value_counts().head(10)


# It looks like many passengers had travelled as a group during the 

# ## fare feature

# In[90]:


train.head()


# ## calculated_fare feature

# In[91]:


## 
train['calculated_fare'] = train.Fare/train.family_size
test['calculated_fare'] = test.Fare/test.family_size


# In[92]:


train.calculated_fare.mean()


# In[93]:


train.calculated_fare.mode()


# In[94]:


def fare_group(fare):
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a
        
    


# In[95]:


train['fare_group'] = train['calculated_fare'].map(fare_group)
test['fare_group'] = test['calculated_fare'].map(fare_group)


# In[96]:


#train['fare_group'] = pd.cut(train['calculated_fare'], bins = 4, labels=groups)


# Some people have travelled in groups like family or friends. It seems like Fare column kept a record of the total fare rather  than 

# ## Creating dummy variables

# In[97]:


train = pd.get_dummies(train, columns=['title',"Pclass", 'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=True)
test = pd.get_dummies(test, columns=['title',"Pclass",'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=True)
train.drop(['Cabin_T', 'family_size','Ticket','Name', 'Fare','name_length'], axis=1, inplace=True)
test.drop(['Ticket','Name','family_size',"Fare",'name_length'], axis=1, inplace=True)


# ## Age feature

# In[98]:


pd.options.display.max_columns = 99
train.head()


# In[99]:


front = train['Age']
train.drop(labels=['Age'], axis=1,inplace = True)
train.insert(0, 'Age', front)
train.head()


# In[100]:


front = test['Age']
test.drop(labels=['Age'], axis=1,inplace = True)
test.insert(0, 'Age', front)
test.head()


# In[101]:


train.Age.head()


# In[102]:


# importing missing values using KNN for age column. 
from fancyimpute import KNN
age_train = KNN(k=10).complete(train)

train = pd.DataFrame(age_train, columns = train.columns)
train.head(2)


# In[103]:


# importing missing values using KNN for age column. 
from fancyimpute import KNN
age_test = KNN(k=10).complete(test)

test = pd.DataFrame(age_test, columns = test.columns)
test.head(2)


# In[104]:


## create bins for age
def age_group_fun(age):
    a = ''
    if age <= 1:
        a = 'infant'
    elif age <= 4: 
        a = 'toddler'
    elif age <= 13:
        a = 'child'
    elif age <= 18:
        a = 'teenager'
    elif age <= 35:
        a = 'Young_Adult'
    elif age <= 45:
        a = 'adult'
    elif age <= 55:
        a = 'middle_aged'
    elif age <= 65:
        a = 'senior_citizen'
    else:
        a = 'old'
    return a
        


# In[105]:


train['age_group'] = train['Age'].map(age_group_fun)
test['age_group'] = test['Age'].map(age_group_fun)


# In[106]:


train = pd.get_dummies(train,columns=['age_group'], drop_first=True)
test = pd.get_dummies(test,columns=['age_group'], drop_first=True)


# In[107]:


"""train.drop('Age', axis=1, inplace=True)
test.drop('Age', axis=1, inplace=True)"""


# # Part 6: Pre-Modeling Tasks

# ## 6a. Separating dependent and independent variables
# <a id="dependent_independent"></a>
# ***

# In[108]:


# separating our independent and dependent variable
X = train.drop(['Survived'], axis=1)
y = train["Survived"]


# In[109]:


#age_filled_data_nor = NuclearNormMinimization().complete(df1)
#Data_1 = pd.DataFrame(age_filled_data, columns = df1.columns)
#pd.DataFrame(zip(Data["Age"],Data_1["Age"],df["Age"]))


# ## 6b. Splitting the training data
# <a id="split_training_data" ></a>
# ***
# 

# In[110]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = .33, random_state = 0)


# In[111]:


x_train.head()


# ## 6c. Feature Scaling
# <a id="feature_scaling" ></a>
# ***
# 

# In[112]:


x_train.head()


# In[113]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[114]:


test = sc.transform(test)


# # Part 7: Modeling the Data
# <a id="modelingthedata"></a>
# ***
# I will train the data with the following models:
# - Logistic Regression
# - K-Nearest Neighbors(KNN)
# - Gaussian Naive Bayes
# - Support Vector Machines
# - Decision Tree Classifier
# - Bagging on Decision Tree Classifier
# - Random Forest Classifier
# - Gradient Boosting Classifier
# 
# 

# In[115]:


## Necessary modules for creating models. 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix


# ## 7a. Logistic Regression
# <a id="logistic_regression"></a>
# ***

# Let's run the logistic regression model first with out any hyper parameter tuning. 

# In[116]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = logreg, X = x_train, y = y_train, cv = 10, n_jobs = -1)
logreg_accy = accuracies.mean()
print (round((logreg_accy),3))


# In[117]:


#note: this is an alternative to train_test_split
##from sklearn import model_selection
##cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
##cv_results = model_selection.cross_validate(logreg, X,y, cv  = cv_split)


# In[118]:


##print (cv_results)
##cv_results['train_score'].mean()


# ### Grid Search on Logistic Regression

# In[119]:


C_vals = [0.099,0.1,0.2,0.5,12,13,14,15,16,16.5,17,17.5,18]
penalties = ['l1','l2']

param = {'penalty': penalties, 
         'C': C_vals 
        }
grid_search = GridSearchCV(estimator=logreg, 
                           param_grid = param,
                           scoring = 'accuracy', 
                           cv = 10
                          )


# In[120]:


grid_search = grid_search.fit(x_train, y_train)


# In[121]:


print (grid_search.best_params_)
print (grid_search.best_score_)


# In[122]:


grid_search.grid_scores_


# In[123]:


logreg_grid = grid_search.best_estimator_


# In[124]:


logreg_accy = logreg_grid.score(x_test, y_test)
logreg_accy


# In[125]:


print (classification_report(y_test, y_pred, labels=logreg_grid.classes_))
print (confusion_matrix(y_pred, y_test))


# In[126]:


from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-pastel')
y_score = logreg_grid.decision_function(x_test)

FPR, TPR, _ = roc_curve(y_test, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Titanic survivors', fontsize= 18)
plt.show()


# In[127]:


plt.style.use('seaborn-pastel')

y_score = logreg_grid.decision_function(x_test)

precision, recall, _ = precision_recall_curve(y_test, y_score)
PR_AUC = auc(recall, precision)

plt.figure(figsize=[11,9])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Titanic survivors', fontsize=18)
plt.legend(loc="lower right")
plt.show()


# ## 7b. K-Nearest Neighbor classifier(KNN)
# <a id="knn"></a>
# ***

# In[128]:


from sklearn.neighbors import KNeighborsClassifier
## choosing the best n_neighbors
nn_scores = []
best_prediction = [-1,-1]
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='minkowski', p =2)
    knn.fit(x_train,y_train)
    score = accuracy_score(y_test, knn.predict(x_test))
    #print i, score
    if score > best_prediction[1]:
        best_prediction = [i, score]
    nn_scores.append(score)
    
print (best_prediction)
plt.plot(range(1,100),nn_scores)


# In[129]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
#n_neighbors: specifies how many neighbors will vote on the class
#weights: uniform weights indicate that all neighbors have the same weight while "distance" indicates
        # that points closest to the 
#metric and p: when distance is minkowski (the default) and p == 2 (the default), this is equivalent to the euclidean distance metric
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
knn_accy = round(accuracy_score(y_test, y_pred), 3)
print (knn_accy)


# ### Grid search on KNN classifier

# In[130]:


n_neighbors=[1,2,3,4,5,6,7,8,9,10]
weights=['uniform','distance']
param = {'n_neighbors':n_neighbors, 
         'weights':weights}
grid2 = GridSearchCV(knn, 
                     param,
                     verbose=False, 
                     cv=StratifiedKFold(n_splits=5, random_state=15, shuffle=True)
                    )
grid2.fit(x_train, y_train)


# In[131]:


print (grid2.best_params_)
print (grid2.best_score_)


# In[132]:


## using grid search to fit the best model.
knn_grid = grid2.best_estimator_


# In[133]:


##accuracy_score =(knn_grid.predict(x_test), y_test)
knn_accy = knn_grid.score(x_test, y_test)
knn_accy


# ## 7c. Gaussian Naive Bayes
# <a id="gaussian_naive"></a>
# ***

# In[134]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
gaussian_accy = round(accuracy_score(y_pred, y_test), 3)
print(gaussian_accy)


# ## 7d. Support Vector Machines
# <a id="svm"></a>
# ***

# In[135]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC(kernel = 'rbf', probability=True, random_state = 1, C = 3)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
svc_accy = round(accuracy_score(y_pred, y_test), 3)
print(svc_accy)


# ## 7e. Decision Tree Classifier
# <a id="decision_tree"></a>
# ***

# In[136]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier()
dectree.fit(x_train, y_train)
y_pred = dectree.predict(x_test)
dectree_accy = round(accuracy_score(y_pred, y_test), 3)
print(dectree_accy)


# ### Grid search on Decision Tree

# In[137]:


max_depth = range(1,30)
max_feature = [21,22,23,24,25,26,'auto']
criterion=["entropy", "gini"]

param = {'max_depth':max_depth, 
         'max_features':max_feature, 
         'criterion': criterion}
decisiontree_grid = GridSearchCV(dectree, 
                                param_grid = param, 
                                 verbose=False, 
                                 cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),
                                n_jobs = -1)
decisiontree_grid.fit(x_train, y_train) 


# In[138]:


print( decisiontree_grid.best_params_)
print (decisiontree_grid.best_score_)


# In[139]:


decisiontree_grid = decisiontree_grid.best_estimator_


# In[140]:


decisiontree_grid.score(x_test, y_test)


# ## 7f. Bagging Classifier
# <a id="bagging"></a>
# ***

# In[141]:


from sklearn.ensemble import BaggingClassifier
BaggingClassifier = BaggingClassifier()
BaggingClassifier.fit(x_train, y_train)
y_pred = BaggingClassifier.predict(x_test)
bagging_accy = round(accuracy_score(y_pred, y_test), 3)
print(bagging_accy)


# ## 7g. Random Forest Classifier
# <a id="random_forest"></a>

# In[142]:


from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators=100,max_depth=9,min_samples_split=6, min_samples_leaf=4)
#randomforest = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
random_accy = round(accuracy_score(y_pred, y_test), 3)
print (random_accy)


# In[143]:


n_estimators = [100,120]
max_depth = range(1,30)



parameters = {'n_estimators':n_estimators, 
         'max_depth':max_depth, 
        }
randomforest_grid = GridSearchCV(randomforest,
                                 param_grid=parameters,
                                 cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),
                                 n_jobs = -1
                                )


# In[144]:


randomforest_grid.fit(x_train, y_train) 


# In[145]:


randomforest_grid.score(x_test, y_test)


# ## 7h. Gradient Boosting Classifier
# <a id="gradient_boosting"></a>
# ***

# In[146]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gradient = GradientBoostingClassifier()
gradient.fit(x_train, y_train)
y_pred = gradient.predict(x_test)
gradient_accy = round(accuracy_score(y_pred, y_test), 3)
print(gradient_accy)


# ## 7i. XGBClassifier
# <a id="XGBClassifier"></a>
# ***

# In[147]:


from xgboost import XGBClassifier
XGBClassifier = XGBClassifier()
XGBClassifier.fit(x_train, y_train)
y_pred = XGBClassifier.predict(x_test)
XGBClassifier_accy = round(accuracy_score(y_pred, y_test), 3)
print(XGBClassifier_accy)


# ## 7j. AdaBoost Classifier
# <a id="adaboost"></a>
# ***

# In[148]:


from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier()
adaboost.fit(x_train, y_train)
y_pred = adaboost.predict(x_test)
adaboost_accy = round(accuracy_score(y_pred, y_test), 3)
print(adaboost_accy)


# ## 7k. Extra Trees Classifier
# <a id="extra_tree"></a>
# ***

# In[149]:


from sklearn.ensemble import ExtraTreesClassifier
ExtraTreesClassifier = ExtraTreesClassifier()
ExtraTreesClassifier.fit(x_train, y_train)
y_pred = ExtraTreesClassifier.predict(x_test)
extraTree_accy = round(accuracy_score(y_pred, y_test), 3)
print(extraTree_accy)


# ## 7l. Gaussian Process Classifier
# <a id="GaussianProcessClassifier"></a>
# ***

# In[150]:


from sklearn.gaussian_process import GaussianProcessClassifier
GaussianProcessClassifier = GaussianProcessClassifier()
GaussianProcessClassifier.fit(x_train, y_train)
y_pred = GaussianProcessClassifier.predict(x_test)
gau_pro_accy = round(accuracy_score(y_pred, y_test), 3)
print(gau_pro_accy)


# ## 7m. Voting Classifier
# <a id="voting_classifer"></a>
# ***

# In[153]:


from sklearn.ensemble import VotingClassifier

voting_classifier = VotingClassifier(estimators=[
    ('logreg_grid', logreg_grid),
    ('logreg',logreg), 
    ('svc', svc),
    ('random_forest', randomforest),
    ('gradient_boosting', gradient),
    ('decision_tree',dectree), 
    ('decision_tree_grid',decisiontree_grid), 
    ('knn',knn),
    ('knn_grid', knn_grid),
    ('XGB Classifier', XGBClassifier),
    ('BaggingClassifier', BaggingClassifier),
    ('ExtraTreesClassifier', ExtraTreesClassifier),
    ('gaussian',gaussian),
    ('gaussian process classifier', GaussianProcessClassifier)], voting='soft')

voting_classifier = voting_classifier.fit(x_train,y_train)


# In[154]:


y_pred = voting_classifier.predict(x_test)
voting_accy = round(accuracy_score(y_pred, y_test), 3)
print(voting_accy)


# In[155]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Decision Tree', 'Gradient Boosting Classifier', 'Voting Classifier', 'XGB Classifier','ExtraTrees Classifier','Bagging Classifier'],
    'Score': [svc_accy, knn_accy, logreg_accy, 
              random_accy, gaussian_accy, dectree_accy,
               gradient_accy, voting_accy, XGBClassifier_accy, extraTree_accy, bagging_accy]})
models.sort_values(by='Score', ascending=False)


# # Part 8: Submit test predictions
# <a id="submit_predictions"></a>
# ***

# In[156]:


test_prediction = voting_classifier.predict(test)
submission = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": test_prediction
    })

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv("titanic1_submission.csv", index=False)


# ##### Credits
# <a id="credits"></a>
# ***
# I have used Randy Lao's Predicting Employee Kernel as a starting point for this kernel; it's one of the best I have come across. So, Please check it out if you want.

# ***
# If you like to discuss any other projects or just have a chat about data science topics, I'll be more than happy to connect with you on:
# 
# **LinkedIn:** https://www.linkedin.com/in/masumrumi/ 
# 
# **My Website:** http://masumrumi.strikingly.com/ 
# 
# *** This kernel is a work in progress like all of my other notebooks. I will always incorporate new concepts of data science as I master them. This journey of learning is worth sharing as well as collaborating. Therefore any comments about further improvements would be genuinely appreciated.***
# ***
# ## If you have come this far, Congratulations!!
# 
# ## If this notebook helped you in anyway, please upvote!!
