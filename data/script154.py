
# coding: utf-8

# # Intro
# This kernel is **not for beginners** but for those who are already familiar with Titanic dataset. Maybe some day I'll write a kernel for beginners, but not now.
# 
# Now I'd like to point something out. Keep in mind that I'm talking just from *my experience on Titanic* so the following may not be true for you, so be cautious.
# 
#  - "Has_Cabin" feature does not help. I engineered a feature with 0 if a passenger has no Cabin (NaN) and 1 if he got one may make sense as cabin data of 1st class passengers was found (IRL) on the body of steward Herbert Cave, so I tried it. But that doesn't seem to help.
#  - "Deck" feature does not help. Based on letters found in Cabin column we may engineer a Deck feature, indicating which deck (A - G, T or U for Unknown) the passenger was on. But it's rather noisy, it doesn't help the score.
#  - "Embarked" does not help, I have no idea why people even include it in their kernels. It has no impact on survival chances.
#  - *Edit*: actually certain algorithms may perform better if you turn categorical features into ordinal ones (like turning Pclass to Pclass_1, Pclass_2 and Pclass_3 features with possible values {0, 1}). Pros are higher accuracy in certain cases, cons are - you lose relation between Pclasses (meaning the algorithm will think those are independent, unordered classes, when in fact they are ordered - Pclass=1 is "better" than Pclass=3) and you add dimensions which is not always good because of the curse of dimensionality. In my specific case turning Pclass into 3 features did not help, but as I learned it's a good idea to try both approaches and see what's better in your case.
#  - Making Bins in Fare and Age helps a little bit.
#  - Standard Scaler is our friend. It helps to boost the score. Scaling features is helpful for many ML algorithms like KNN for example, it really boosts their score. This kind of scaler may assume normal distribution of the features and sometimes MinMaxScaler may be better, but from what I've tried, the standard one leads to better results in this case.
#  - I don't know about feature scaling in R, maybe R methods scale them by default? If not, and if you're using R, try scaling, it may help.
#  - There is not much sence in scaling features that are already 0 or 1 like Sex, but for now I scale them all. You can try to pick features for scaling. If you don't use bins (if you use Age or Fare "as is"), scaling may help to boost your score a bit, try it.
# 
# # About Kaggle scoring
# I've spent a lot of time doggedly fighting to improve my score by even the tiny bit and got from 0.72 to 0.82-0.83 (as of now), but that was out of interest - just to see if it's possible and if I can do it. To tell you the truth, I think that all the models within 0.80-0.85 public score interval are really close and, almost equal.
# MLA builds and "returns" you a specific mathematical function, the "final hypothesis" g(x). I think that models within 0.80-0.85 interval return very similar functions, it just so happens that some of them yield a public score of 0.81 and others 0.82, this does really does not mean that the 0.82 one is better. On the private dataset, the first model of those two may prove to be better actually.
# So I think models with public scores of 0.80-0.85 (maybe even starting from 0.79) are more or less the same.
# 
# Don't try to chase the Public score too eagerly. As Kaggle states:
# 
# > Your final score may not be based on the same exact subset of data as the public leaderboard, but rather a different private data subset of your full submission — your public score is only a rough indication of what your final score is. You should thus choose submissions that will most likely be best overall, and not necessarily on the public subset.
# 
# Look at competitions where you've got 20% of data as public, and the rest is private. You see the guy on, say, the first place? Well, come the end of competition and private scoring starts, he may find himself in 12th position, while the 12th guy may become the best :)
# 
# Moreover, if you submit your results too many times, you subconsciously "bleed" public test set data into your models, and your models adapt to the public test set a little more. They may tend to overfit to the public test set, meaning your score on a private test set may drop significantly. In this kernel I'm trying to get the highest public score that I can, but again, you can say that in a way I'm overfitting my data to the test set.
# 
# So generally your aim should be constructing the most robust and generalizable model with a solid public score, but not necessarily the best score. On private dataset your model may actually turn out to be the best, keep that in mind.
# 
# Now let's roll.

# ## Loading a bunch of stuff
# Imports are from my Jupyter notebooks on my PC, in those notebooks I import 'em all so that later I don't have to bother with importing things.

# In[ ]:


# NumPy
import numpy as np

# Dataframe operations
import pandas as pd

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Models
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.linear_model import Perceptron
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate

# GridSearchCV
from sklearn.model_selection import GridSearchCV

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix


# ## Loading datasets

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
data_df = train_df.append(test_df) # The entire data: train + test.


# # Engineering features
# 
#  - **Imputing Age**
#  
# I make a Title feature for imputing ages more precisely. Median is used because ages distribution is not always normal, so it's generally preferred over mean. But I don't think this matters a lot, you can use mean too.
# I don't use Title feature for fitting models so it's discarded.

# In[ ]:


data_df['Title'] = data_df['Name']
# Cleaning name and extracting Title
for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

# Replacing rare titles with more common ones
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data_df.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute
    
# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Age'] = data_df['Age'][:891]
test_df['Age'] = data_df['Age'][891:]

# Dropping Title feature
data_df.drop('Title', axis = 1, inplace = True)


#  - **Adding Family_Size**
#  
# That's just Parch + SibSp.

# In[ ]:


data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']

# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Family_Size'] = data_df['Family_Size'][:891]
test_df['Family_Size'] = data_df['Family_Size'][891:]


#  - **Adding Family_Survival**
#  
#  This feature is from [S.Xu's kernel](https://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever), he groups families and people with the same tickets togerher and researches the info. I've cleaned the code a bit but it still does the same, I left it as is. For comments see the original kernel.

# In[ ]:


data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      data_df.loc[data_df['Family_Survival']!=0.5].shape[0])


# In[ ]:


for _, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))

# # Family_Survival in TRAIN_DF and TEST_DF:
train_df['Family_Survival'] = data_df['Family_Survival'][:891]
test_df['Family_Survival'] = data_df['Family_Survival'][891:]


#  - **Making FARE BINS**
#  
# It's ordinal. FareBin = 3 is indeed greater than FareBin = 1. I've seen people turning it into dummies for some reason...

# In[ ]:


data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)

# Making Bins
data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)

label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])

train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]
test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]

train_df.drop(['Fare'], 1, inplace=True)
test_df.drop(['Fare'], 1, inplace=True)


#  - **Making AGE BINS**
#  
# Note here that it is better to use the entire dataset for mean/median/mode calculation, otherwise we will miss out useful information. 

# In[ ]:


data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)

label = LabelEncoder()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])

train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]
test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]

train_df.drop(['Age'], 1, inplace=True)
test_df.drop(['Age'], 1, inplace=True)


#  - **Mapping SEX and cleaning data (dropping garbage) **

# In[ ]:


train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)

train_df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
               'Embarked'], axis = 1, inplace = True)
test_df.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
              'Embarked'], axis = 1, inplace = True)


# So now our datasets look like this:

# In[ ]:


train_df.head(3)


# # Training
# 
#  - **Creating X and y**

# In[ ]:


X = train_df.drop('Survived', 1)
y = train_df['Survived']
X_test = test_df.copy()


#  - **Scaling features**

# In[ ]:


std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)


#  - **Grid Search CV**
#  
#  Here I use KNN.

# In[ ]:


n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=10, scoring = "roc_auc")
gd.fit(X, y)
print(gd.best_score_)
print(gd.best_estimator_)


# 
# 
# In case you get a different result here (result may vary), what I got was:
# 
# > KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=18, p=2, weights='uniform')
# 
# This gave 0.884103388207 ROC_AUC score (not accuracy score!). I had a ton of models with roc_auc around 0.93-0.94 but when tested, they mostly showed lower results. Doesn't mean they are worse though.

#  - **Using a model found by grid searching**

# In[ ]:


gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(X_test)


# When I submitted the result, the model I've specified above yielded [0.82775] public score.

# - **Using another K**
# 
# This guy comes from empirical messing around with amount of neighbors in KNN. It's the same as the above one, but with another n:

# In[ ]:


knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 
                           weights='uniform')
knn.fit(X, y)
y_pred = knn.predict(X_test)


# Being a fan of simple models there's no way I couldn't try playing with n_neighbors lowering it (the lower it is --> the less complex the model is, though too simple model is bad news too).

# - **Making submission**

# In[ ]:


temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
temp['Survived'] = y_pred
temp.to_csv("../working/submission.csv", index = False)


#  ## Result
#  So when I submitted the score I got 0.83253.

# # Conclusion
# A couple of points here too.
#  - Of course KNN is not the only model I used. I used SVMs, AdaBoosting, GradientBoosting, RandomForests etc. etc., many many various models, all this is omited here. But KNN shows solid results on all my latest engineered datasets so I preferred it for simplicity. Feel free to try the dataset with other estimators, on particular you can try XGBoost with tuned hyperparams. There's a chance you may achieve a higher score on the same features as described in this kernel.
#  - If you engineer Family groups further, I think you may get better results. Check out [Finding the 'real' families on the Titanic by Eric Bruin](https://www.kaggle.com/erikbruin/finding-the-real-families-on-the-titanic) to gain insight into families.
#  - Generally, grouping passengers is a good way to improve your score. Try searching for groups.
#  - You can use interesting features like [SLogL in Oscar's kernel](https://www.kaggle.com/pliptor/divide-and-conquer-0-82296), give it a try.
# 
# However, do be prepared to invest a lot of time for only a small revenue. Oscar's 0.82296 score appears to be pretty much the limit, I think that's as far as we can get; if you get a score above that but below 0.86 or so, that means your model is more or less the same as mine or as Oscar's or as Erik's. The truly better model would yield 86-88+, but it's very tough to get there.
