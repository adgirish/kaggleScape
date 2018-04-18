
# coding: utf-8

# # **Introduction**
# 
# This is my first Kaggle, and my first foray into data analysis using python.  The following kernel contains the steps, including 3 approaches to this classification task, enumerated below for assessing the Titanic survival dataset:<br> <br> 
# 1. [Import Data & Python Packages](#1-bullet) <br>
# 2. [Assess Data Quality & Missing Values](#2-bullet)<br>
#     * [2.1 Age - Missing Values](#2.1-bullet) <br>
#     * [2.2 Cabin - Missing Values](#2.2-bullet) <br>
#     * [2.3 Embarked - Missing Values](#2.3-bullet) <br>
#     * [2.4 Final Adjustments to Data](#2.4-bullet) <br>
#     * [2.4.1 Additional Variables](#2.5-bullet) <br> 
# 3. [Exploratory Data Analysis](#3-bullet) <br>
# 4. [Logistic Regression](#4-bullet) <br>
#     * [4. 1 Hold-Out Testing & Logistic Model Assessment](#4.1-bullet) <br>
#     * [4.2 Kaggle "Test" Dataset](#4.2-bullet) <br>
#     * [4.3 Re-run Logistic Regression w/ 80-20 Split](#4.3-bullet) <br>
#     * [4.4 Out-of-sample test results](#4.4-bullet) <br>
#     * [4.5 Logistic Regression Conclusions](#4.5-bullet) <br>
# 6. [Alternate Approach 1 : Random Forest Estimation](#6-bullet) <br>
# 7. [Alternate Approach 2: Decision Tree](#7-bullet) <br>

# ## 1. Import Data & Python Packages <a class="anchor" id="1-bullet"></a>

# In[ ]:


import numpy as np 
import pandas as pd 

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

#sklearn imports source: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8


# In[ ]:


# get titanic & test csv files as a DataFrame

#developmental data (train)
titanic_df = pd.read_csv("../input/train.csv")

#cross validation data (hold-out testing)
test_df    = pd.read_csv("../input/test.csv")

# preview developmental data
titanic_df.head(5)


# In[ ]:


test_df.head(5)


# <font color=red>  Note: There is no target variable for the hold out data (i.e. "Survival" column is missing), so there's no way to use this as our cross validation sample.  Refer to Section 5.</font>

# ## 2. Data Quality & Missing Value Assessment <a class="anchor" id="2-bullet"></a>

# In[ ]:


# check missing values in train dataset
titanic_df.isnull().sum()


# ### 2.1    Age - Missing Values <a class="anchor" id="2.1-bullet"></a>

# In[ ]:


sum(pd.isnull(titanic_df['Age']))


# In[ ]:


# proportion of "Age" missing
round(177/(len(titanic_df["PassengerId"])),4)


# ~20% of entries for passenger age are missing. Let's see what the 'Age' variable looks like in general.

# In[ ]:


ax = titanic_df["Age"].hist(bins=15, color='teal', alpha=0.8)
ax.set(xlabel='Age', ylabel='Count')
plt.show()


# Since "Age" is (right) skewed, using the mean might give us biased results by filling in ages that are older than desired.  To deal with this, we'll use the median to impute the missing values. 

# In[ ]:


# median age is 28 (as compared to mean which is ~30)
titanic_df["Age"].median(skipna=True)


# ### 2.2 Cabin - Missing Values <a class="anchor" id="2.2-bullet"></a>

# In[ ]:


# proportion of "cabin" missing
round(687/len(titanic_df["PassengerId"]),4)


# 77% of records are missing, which means that imputing information and using this variable for prediction is probably not wise.  We'll ignore this variable in our model.

# ### 2.3 Embarked - Missing Values <a class="anchor" id="2.3-bullet"></a>

# In[ ]:


# proportion of "Embarked" missing
round(2/len(titanic_df["PassengerId"]),4)


# There are only 2 missing values for "Embarked", so we can just impute with the port where most people boarded.

# In[ ]:


sns.countplot(x='Embarked',data=titanic_df,palette='Set2')
plt.show()


# By far the most passengers boarded in Southhampton, so we'll impute those 2 NaN's w/ "S".

# *References for graph creation:*<br>
# https://matplotlib.org/1.2.1/examples/pylab_examples/histogram_demo.html <br>
# https://seaborn.pydata.org/generated/seaborn.countplot.html

# ### 2.4 Final Adjustments to Data (Train & Test) <a class="anchor" id="2.4-bullet"></a>

# Based on my assessment of the missing values in the dataset, I'll make the following changes to the data:
# * If "Age" is missing for a given row, I'll impute with 28 (median age).
# * If "Embark" is missing for a riven row, I'll impute with "S" (the most common boarding port).
# * I'll ignore "Cabin" as a variable.  There are too many missing values for imputation.  Based on the information available, it appears that this value is associated with the passenger's class and fare paid.

# In[ ]:


train_data = titanic_df
train_data["Age"].fillna(28, inplace=True)
train_data["Embarked"].fillna("S", inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)


# ### 2.4.1 Additional Variables <a class="anchor" id="2.4.1-bullet"></a>

# According to the Kaggle data dictionary, both SibSp and Parch relate to traveling with family.  For simplicity's sake (and to account for possible multicollinearity), I'll combine the effect of these variables into one categorical predictor: whether or not that individual was traveling alone.

# In[ ]:


## Create categorical variable for traveling alone

train_data['TravelBuds']=train_data["SibSp"]+train_data["Parch"]
train_data['TravelAlone']=np.where(train_data['TravelBuds']>0, 0, 1)


# In[ ]:


train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)
train_data.drop('TravelBuds', axis=1, inplace=True)


# I'll also create categorical variables for Passenger Class ("Pclass"), Gender ("Sex"), and Port Embarked ("Embarked"). 

# In[ ]:


#create categorical variable for Pclass

train2 = pd.get_dummies(train_data, columns=["Pclass"])


# In[ ]:


train3 = pd.get_dummies(train2, columns=["Embarked"])


# In[ ]:


train4=pd.get_dummies(train3, columns=["Sex"])
train4.drop('Sex_female', axis=1, inplace=True)


# In[ ]:


train4.drop('PassengerId', axis=1, inplace=True)
train4.drop('Name', axis=1, inplace=True)
train4.drop('Ticket', axis=1, inplace=True)
train4.head(5)


# In[ ]:


df_final = train4


# ### Now, apply the same changes to the test data. <br>
# I will apply to same imputation for "Age" in the Test data as I did for my Training data (if missing, Age = 28).  <br> I'll also remove the "Cabin" variable from the test data, as I've decided not to include it in my analysis. <br> There were no missing values in the "Embarked" port variable. <br> I'll add the dummy variables to finalize the test set.  <br> Finally, I'll impute the 1 missing value for "Fare" with the median, 14.45.

# In[ ]:


test_df["Age"].fillna(28, inplace=True)
test_df["Fare"].fillna(14.45, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)


# In[ ]:


test_df['TravelBuds']=test_df["SibSp"]+test_df["Parch"]
test_df['TravelAlone']=np.where(test_df['TravelBuds']>0, 0, 1)

test_df.drop('SibSp', axis=1, inplace=True)
test_df.drop('Parch', axis=1, inplace=True)
test_df.drop('TravelBuds', axis=1, inplace=True)

test2 = pd.get_dummies(test_df, columns=["Pclass"])
test3 = pd.get_dummies(test2, columns=["Embarked"])

test4=pd.get_dummies(test3, columns=["Sex"])
test4.drop('Sex_female', axis=1, inplace=True)

test4.drop('PassengerId', axis=1, inplace=True)
test4.drop('Name', axis=1, inplace=True)
test4.drop('Ticket', axis=1, inplace=True)
final_test = test4


# In[ ]:


final_test.head(5)


# *References for categorical variable creation: <br>
# http://pbpython.com/categorical-encoding.html <br>
# https://chrisalbon.com/python/data_wrangling/pandas_create_column_using_conditional/*

# ## 3. Exploratory Data Analysis <a class="anchor" id="3-bullet"></a>

# ## 3.1 Exploration of Age <a class="anchor" id="3.1-bullet"></a>

# In[ ]:


plt.figure(figsize=(15,8))
sns.kdeplot(titanic_df["Age"][df_final.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(titanic_df["Age"][df_final.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
plt.show()


# The age distribution for survivors and deceased is actually very similar.  One notable difference is that, of the survivors, a larger proportion were children.  The passengers evidently made an attempt to save children by giving them a place on the life rafts. 

# In[ ]:


plt.figure(figsize=(20,8))
avg_survival_byage = df_final[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")


# Considering the survival rate of passengers under 16, I'll also include another categorical variable in my dataset: "Minor"

# In[ ]:


df_final['IsMinor']=np.where(train_data['Age']<=16, 1, 0)


# In[ ]:


final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)


# ## 3.2 Exploration of Fare <a class="anchor" id="3.2-bullet"></a>

# In[ ]:


plt.figure(figsize=(15,8))
sns.kdeplot(df_final["Fare"][titanic_df.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(df_final["Fare"][titanic_df.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
# limit x axis to zoom on most information. there are a few outliers in fare. 
plt.xlim(-20,200)
plt.show()


# As the distributions are clearly different for the fares of survivors vs. deceased, it's likely that this would be a significant predictor in our final model.  Passengers who paid lower fare appear to have been less likely to survive.  This is probably strongly correlated with Passenger Class, which we'll look at next.

# ## 3.3 Exploration of Passenger Class <a class="anchor" id="3.3-bullet"></a>

# In[ ]:


sns.barplot('Pclass', 'Survived', data=titanic_df, color="darkturquoise")
plt.show()


# Unsurprisingly, being a first class passenger was safest.

# ## 3.4 Exploration of Embarked Port <a class="anchor" id="3.4-bullet"></a>

# In[ ]:


sns.barplot('Embarked', 'Survived', data=titanic_df, color="teal")
plt.show()


# Passengers who boarded in Cherbourg, France, appear to have the highest survival rate.  Passengers who boarded in Southhampton were marginally less likely to survive than those who boarded in Queenstown.  This is probably related to passenger class, or maybe even the order of room assignments (e.g. maybe earlier passengers were more likely to have rooms closer to deck). <br> It's also worth noting the size of the whiskers in these plots.  Because the number of passengers who boarded at Southhampton was highest, the confidence around the survival rate is the highest.  The whisker of the Queenstown plot includes the Southhampton average, as well as the lower bound of its whisker.  It's possible that Queenstown passengers were equally, or even more, ill-fated than their Southhampton counterparts.

# ## 3.5 Exploration of Traveling Alone vs. With Family <a class="anchor" id="3.5-bullet"></a>

# In[ ]:


sns.barplot('TravelAlone', 'Survived', data=df_final, color="mediumturquoise")
plt.show()


# Individuals traveling without family were more likely to die in the disaster than those with family aboard.  Given the era, it's likely that individuals traveling alone were likely male.

# ## 3.6 Exploration of Gender Variable <a class="anchor" id="3.6-bullet"></a>

# In[ ]:


sns.barplot('Sex', 'Survived', data=titanic_df, color="aquamarine")
plt.show()


# This is a very obvious difference.  Clearly being female greatly increased your chances of survival.

# References: <br>
# https://seaborn.pydata.org/generated/seaborn.barplot.html <br>
# https://seaborn.pydata.org/generated/seaborn.kdeplot.html

# ## 4. Logistic Regression and Results <a class="anchor" id="4-bullet"></a>

# In[ ]:


df_final.head(10)


# In[ ]:


cols=["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
X=df_final[cols]
Y=df_final['Survived']


# In[ ]:


import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary())


# Nearly all variables are significant at the 0.05 alpha level, but we'll run the model again without Fare and TravelAlone (removed one at a time, results didn't change much.  In the end removed both).  I also removed "IsMinor" from this regression, as the information provided is redundant to the Age variable.

# In[ ]:


cols2=["Age", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male"]  
X2=df_final[cols2]
Y=df_final['Survived']

logit_model=sm.Logit(Y,X2)
result=logit_model.fit()

print(result.summary())


# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X2, Y)

logreg.score(X2, Y)


# ## Model's Predictive Score: 0.7935

# *References:* <br>
# https://github.com/statsmodels/statsmodels/issues/3931 <br>
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

# ## 4.1 . Hold-Out Testing <a class="anchor" id="4.1-bullet"></a>

# ### 4.2 Using Kaggle's Titanic "Test" Data <a class="anchor" id="4.2-bullet"></a>

# In[ ]:


#from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
#logreg = LogisticRegression()
#logreg.fit(X2, Y)

#X_test = final_test[cols2]
#y_test = final_test['Survived']

#y_pred = logreg.predict(X_test)
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# </div>
#  <div class="alert alert-block alert-danger">
# <font color=red> **Cross Validation: Turns out the test data doesn't have "survived" information, so this isn't helpful for our out-of-sample analysis.** </font>
# 
# 

# ## 4.3 Using 80-20 Split for Cross Validation <a class="anchor" id="4.3-bullet"></a>

# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df_final, test_size=0.2)


# *References:* <br>
# https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas

# In[ ]:


#re-fit logistic regression on new train sample

cols2=["Age", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male"] 
X3=train[cols2]
Y3=train['Survived']
logit_model3=sm.Logit(Y3,X3)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()
logreg.fit(X3, Y3)
logreg.score(X3, Y3)


# The score for the new training sample (80% of original) is very close to the original performance, which is good!<br>
# Let's assess how well it scores on the 20% hold-out sample.

# In[ ]:


from sklearn import metrics
logreg.fit(X3, Y3)

X3_test = test[cols2]
Y3_test = test['Survived']

Y3test_pred = logreg.predict(X3_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X3_test, Y3_test)))


# The model's out of sample performance does not show any deterioration.<br>
# *Resources:* <br>
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html <br>
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

# # 4.4 Out-of-sample Assessment <br> <a class="anchor" id="4.4-bullet"></a>

# ### Assessing the model's performance based on Cross Validation ROC/AUC 

# In[ ]:


# Model's in sample AUC

from sklearn.metrics import roc_auc_score
logreg.fit(X3, Y3)
Y3_pred = logreg.predict(X3)

y_true = Y3
y_scores = Y3_pred
roc_auc_score(y_true, y_scores)


# In[ ]:


#Visualizing the model's ROC curve (**source for graph code given below the plot)
from sklearn.metrics import roc_curve, auc
logreg.fit(X3, Y3)

y_test = Y3_test
X_test = X3_test
 
# Determine the false positive and true positive rates
FPR, TPR, _ = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
 
# Calculate the AUC

roc_auc = auc(FPR, TPR)
print ('ROC AUC: %0.3f' % roc_auc )
 
# Plot of a ROC curve
plt.figure(figsize=(10,10))
plt.plot(FPR, TPR, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Sample Performance)')
plt.legend(loc="lower right")
plt.show()


# An AUC score of 0.5 is effectively as good as the flip of a coin, and means that the model really has no classification power at all between the positive and negative occurences. The AUC for both the test and train samples when run on my logistic regression demonstrates relatively strong power of separation between positive and negative occurences (survived - 1, died - 0).
# 
# > ### "AUC of a classifier is equivalent to the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance." -Majnik, Bosnic, 2011<br> 
# 
# <br> *References*: <br>
# ROC Analysis of Classifiers in Machine Learning: A Survey, Matjaz Majnik, Zoran Bosnic, 2011: <br>
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.301.969&rep=rep1&type=pdf<br>
# 
# https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
# http://www.ultravioletanalytics.com/2014/12/16/kaggle-titanic-competition-part-x-roc-curves-and-auc/

# # 4.5 Logistic Regression Conclusion<br> <a class="anchor" id="4.5-bullet"></a>
# <br> 
# Based on my analysis, if you were to be aboard the Titanic, your chances of survival were best if you fit the following criteria:<br>
# * Female
# * Young
# * In First Class 
# * Embarked in Cherbourg France
# 

# ## 5. Random Forest Estimation <a class="anchor" id="5-bullet"></a>

# Our Logistic Regression is effective and easy to interpret, but there are other ML techniques which could provide a more accurate prediction.  Random forests, a tree-based machine learning technique, often provide more accurate results than Logistic Regression classifier models.  With respect to tree growth, performance tends to taper off after a certain number of trees are grown. <br> <br>
# I conducted several iterations of a Random Forest model by adjusting the number of trees (n_estimators parameter) and submitted by results for scoring on Kaggle. I tested 40, 80, 100, and 120 trees, and the best out-of-sample predictictive power was achieved with 100 trees. <br> <br>
# *Note*: I used the same variables I utilized in my first logistic regression to train my random forest. If I were to start this again from scratch, I might have tried testing a wider range of variables (e.g. leaving in the two variables on travel companions which I had reduced to a single categorical variable around traveling along).  

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

cols=["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
X=df_final[cols]
Y=df_final['Survived']

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, Y)
random_forest.score(X, Y)


# *References*:<br>
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html <br>
# https://stats.stackexchange.com/questions/260460/optimization-of-a-random-forest-model<br>
# https://en.wikipedia.org/wiki/Random_forest <br>
# https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

# ## Final RF Submission

# In[ ]:


final_test_RF=final_test[cols]
Y_pred_RF = random_forest.predict(final_test_RF)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_RF
    })
submission.to_csv('titanic.csv', index=False)


# ## 6. Decision Tree <a class="anchor" id="6-bullet"></a>

# Let's try another method- a decision tree.  There is a tradeoff for the additional complexity of utilizing a decision tree as compared to a logistic regression: growing your number of trees too much can subject your model to overfitting and reduce the predictive power of the model.  I've set parameters within the DecisionTreeClassifier from sklearn to help make sure my model is not overfit (too many branches based on the train data).  Some trial and error went into this to determine the optimal number of branches to "prune" to achieve strong out-of-sample results.<br><br>
# *Note*: Again, I used the same variables for the decision tree as I did in my first logistic regression and in my random forest.

# In[ ]:


from sklearn import tree
import graphviz
tree1 = tree.DecisionTreeClassifier(criterion='gini', splitter='best',max_depth=3, min_samples_leaf=20)


# *Resources*:<br>
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# http://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/
# https://en.wikipedia.org/wiki/Pruning_(decision_trees)

# In[ ]:


cols=["Age", "Fare", "TravelAlone", "Pclass_1", "Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 
X_DT=df_final[cols]
Y_DT=df_final['Survived']

tree1.fit(X_DT, Y_DT)


# Let's see how our tree grew! What were the splits the model identified as being most significant in this classification task?

# In[ ]:


import graphviz 
tree1_view = tree.export_graphviz(tree1, out_file=None, feature_names = X_DT.columns.values, rotate=True) 
tree1viz = graphviz.Source(tree1_view)
tree1viz


# *Reference*:<br>http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

# In[ ]:


final_test_DT=final_test[cols]


# In[ ]:


Y_pred_DT = tree1.predict(final_test_DT)


# In[ ]:


# submission = pd.DataFrame({
#        "PassengerId": test_df["PassengerId"],
#        "Survived": Y_pred_DT
#    })
#submission.to_csv('titanic.csv', index=False)


# **Final References:** <br>
# *Editing Markdowns*: https://medium.com/ibm-data-science-experience/markdown-for-jupyter-notebooks-cheatsheet-386c05aeebed<br>
# *Matplotlib color library:* https://matplotlib.org/examples/color/named_colors.html
