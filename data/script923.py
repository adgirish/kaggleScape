
# coding: utf-8

# ## 1. Introduction
# 
# ### In this notebook, we  will show our method which can get a PB score <font color = red > 0.84211 </font>. This notebook only uses single model random forest classifier, so there is still room to improve the performance like  using ensemble methods.  
# 
# ### Because our method based on other great minds and the intuition behind the feature engineering can be found in the following notebooks. Hence, I will omit the details and only provide the codes.
# 
# #### 1. Titanic Random Forest: 82.78%
# #### 2. A Journey through Titanic
# #### 3. Titanic Data Science Solutions
# #### 4.Pytanic

# ## 2. Load Libraries and Raw Data

# In[1]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing 
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import cross_validation

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv("../input/train.csv")
train.head()


# In[3]:


test = pd.read_csv("../input/test.csv")
test.head()


# ## 2. Feature Engineering

# In[4]:


train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)


# In[5]:


def Name_Title_Code(x):
    if x == 'Mr.':
        return 1
    if (x == 'Mrs.') or (x=='Ms.') or (x=='Lady.') or (x == 'Mlle.') or (x =='Mme'):
        return 2
    if x == 'Miss':
        return 3
    if x == 'Rev.':
        return 4
    return 5

train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
test['Name_Title'] = test['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0]) 


# In[6]:


def Age_feature(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)  
        data = train.groupby(['Name_Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
#         i['Age'] = data.transform(lambda x: x.fillna(x.median()))
    return train, test


# In[7]:


def Family_feature(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test 


# In[8]:


def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                    np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test


# In[9]:


def Cabin_feature(train, test):
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test


# In[10]:


def cabin_num(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    del train['Cabin_num']
    del test['Cabin_num']
    del train['Cabin_num1']
    del test['Cabin_num1']
    return train, test


# In[11]:


def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test


# In[12]:


test['Fare'].fillna(train['Fare'].mean(), inplace = True)


# In[13]:


def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test


# In[14]:


def drop(train, test, bye = ['PassengerId']):
    for i in [train, test]:
        for z in bye:
            del i[z]
    return train, test


# In[15]:


train, test = Age_feature(train, test)
 
train['Name_Title'] = train['Name_Title'].apply(Name_Title_Code)
test['Name_Title'] = test['Name_Title'].apply(Name_Title_Code)
train = pd.get_dummies(columns = ['Name_Title'], data = train)
test = pd.get_dummies(columns = ['Name_Title'], data = test)

train, test = cabin_num(train, test)

train, test = Cabin_feature(train, test)

train, test = embarked_impute(train, test)

train, test = Family_feature(train, test)

test['Fare'].fillna(train['Fare'].mean(), inplace = True)

train, test = ticket_grouped(train, test)

train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Fam_Size','Cabin_Letter'])  

train, test = drop(train, test)


# In[16]:


train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)


# ## 4. Model training

# In[17]:



# rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
# param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}
# gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

# gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])

# print(gs.best_score_)
# print(gs.best_params_) 


# In[18]:


from sklearn.ensemble import RandomForestClassifier
 
rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(train.iloc[:, 1:], train.iloc[:, 0])
print("%.4f" % rf.oob_score_)


# In[19]:


pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]


# ## 5. Submit

# In[20]:


submit = pd.read_csv('../input/genderclassmodel.csv')
submit.set_index('PassengerId',inplace=True)

rf_res =  rf.predict(test)
submit['Survived'] = rf_res
submit['Survived'] = submit['Survived'].apply(int)
submit.to_csv('submit.csv')


# In[21]:


submit


# ## 6. Learning together
# ### If you have other techniques or expereriences to improve this PB score, I wish you could share with us. Let's learn together.
