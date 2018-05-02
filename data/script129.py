
# coding: utf-8

# Read the dataset, and print the field information.

# In[ ]:


import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.info())
print(test.info())


# We can see the Age and Cabin fields have many values missing both in the train and test dataset. For the age field, we can based the person's title's mean age to supplement it. For the Cabin field, I have no idea how to complement it, so just leave it and do nothing.

# In[ ]:


#Use the Regular Expression to get the title from the name field.
import re
pattern = re.compile(r'.*?,(.*?)\.')
def getTitle(x):
    result = pattern.search(x)
    if result:
        return result.group(1).strip()
    else:
        return ''

train['Title'] = train['Name'].map(getTitle)
test['Title'] = test['Name'].map(getTitle)

#Check how many rows missing the Age by Title
print(train['Title'][train['Age'].isnull()].value_counts())
print(test['Title'][test['Age'].isnull()].value_counts())

#Set the missing Age of Title 'Master' 
master_age_mean = train['Age'][(train['Title']=='Master')&(train['Age']>0)].mean()
train.loc[train[(train['Title']=='Master')&(train['Age'].isnull())].index, 'Age'] = master_age_mean
test.loc[test[(test['Title']=='Master')&(test['Age'].isnull())].index, 'Age'] = master_age_mean

#Set the missing Age of Title 'Mr' 
mr_age_mean = train['Age'][(train['Title']=='Mr')&(train['Age']>0)].mean()
train.loc[train[(train['Title']=='Mr')&(train['Age'].isnull())].index, 'Age'] = mr_age_mean
test.loc[test[(test['Title']=='Mr')&(test['Age'].isnull())].index, 'Age'] = mr_age_mean

#Set the missing Age of Title 'Miss' or 'Ms'
miss_age_mean = train['Age'][(train['Title']=='Miss')&(train['Age']>0)].mean()
train.loc[train[(train['Title']=='Miss')&(train['Age'].isnull())].index, 'Age'] = miss_age_mean
test.loc[test[((test['Title']=='Miss')|(test['Title']=='Ms'))&(test['Age'].isnull())].index, 'Age'] = miss_age_mean

#Set the missing Age of Title 'Mrs' 
mrs_age_mean = train['Age'][(train['Title']=='Mrs')&(train['Age']>0)].mean()
train.loc[train[(train['Title']=='Mrs')&(train['Age'].isnull())].index, 'Age'] = mrs_age_mean
test.loc[test[(test['Title']=='Mrs')&(test['Age'].isnull())].index, 'Age'] = mrs_age_mean

#Set the missing Age of Title 'Dr' 
dr_age_mean = train['Age'][(train['Title']=='Dr')&(train['Age']>0)].mean()
train.loc[train[(train['Title']=='Dr')&(train['Age'].isnull())].index, 'Age'] = dr_age_mean
test.loc[test[(test['Title']=='Mrs')&(test['Age'].isnull())].index, 'Age'] = dr_age_mean

print(train['Age'].describe())
print(test['Age'].describe())


# Now the Age field has no missing value. Let's do some data exploration work to see the value distribution by survival.

# In[ ]:


import matplotlib.pyplot as plt
alpha = 0.6
fig = plt.figure(figsize=(8, 12))
grouped = train.groupby(['Survived'])
group0 = grouped.get_group(0)
group1 = grouped.get_group(1)

plot_rows = 5
plot_cols = 2
ax1 = plt.subplot2grid((plot_rows,plot_cols), (0,0), rowspan=1, colspan=1)
plt.hist([group0.Age, group1.Age], bins=16, range=(0,80), stacked=True, 
        label=['Not Survived', 'Survived'], alpha=alpha)
plt.legend(loc='best', fontsize='x-small')
ax1.set_title('Survival distribution by Age')

ax2 = plt.subplot2grid((plot_rows,plot_cols), (0,1), rowspan=1, colspan=1)
n, bins, patches = plt.hist([group0.Pclass, group1.Pclass], bins=5, range=(0,5), 
        stacked=True, label=['Not Survived', 'Survived'], alpha=alpha)
plt.legend(loc='best', fontsize='x-small')
ax2.set_xticks([1.5, 2.5, 3.5])
ax2.set_xticklabels(['Class1', 'Class2', 'Class3'], fontsize='small')
ax2.set_yticks([0, 150, 300, 450, 600, 750])
ax2.set_title('Survival distribution by Pclass')

ax3 = plt.subplot2grid((plot_rows,plot_cols), (1,0), rowspan=1, colspan=2)
ax3.set_title('Survival distribution by Sex')
patches, l_texts, p_texts = plt.pie(train.groupby(['Survived', 'Sex']).size(), 
        labels=['Not Survived Female', 'Not Survived Male', 'Survived Female', 'Survived Male'],
        autopct='%3.1f', labeldistance = 1.1, pctdistance = 0.6)
plt.legend(loc='upper right', fontsize='x-small')
for t in l_texts:
    t.set_size(10)
for p in p_texts:
    p.set_size(10)
#plt.legend(loc='best', fontsize='x-small')
plt.axis('equal')

ax4 = plt.subplot2grid((plot_rows,plot_cols), (2,0), rowspan=1, colspan=1)
ax4.set_title('Survival distribution by SibSp')
plt.hist([group0.SibSp, group1.SibSp], bins=9, range=(0,9), stacked=True, 
        label=['Not Survived', 'Survived'], log=False, alpha=alpha)
plt.legend(loc='best', fontsize='x-small')

ax5 = plt.subplot2grid((plot_rows,plot_cols), (2,1), rowspan=1, colspan=1)
ax5.set_title('Survival distribution by SibSp')
plt.hist([group0[group0.SibSp>1].SibSp, group1[group1.SibSp>1].SibSp], bins=8, range=(1, 9), stacked=True, 
        label=['Not Survived', 'Survived'], log=False, alpha=alpha)
plt.legend(loc='best', fontsize='x-small')

ax6 = plt.subplot2grid((plot_rows,plot_cols), (3,0), rowspan=1, colspan=1)
ax6.set_title('Survival distribution by Parch')
plt.hist([group0.Parch, group1.Parch], bins=7, range=(0,7), stacked=True, 
        label=['Not Survived', 'Survived'], log=False, alpha=alpha)
plt.legend(loc='best', fontsize='x-small')

ax7 = plt.subplot2grid((plot_rows,plot_cols), (3,1), rowspan=1, colspan=1)
ax7.set_title('Survival distribution by Parch')
plt.hist([group0[group0.Parch>1].Parch, group1[group1.Parch>1].Parch], bins=6, range=(1, 7), stacked=True, 
        label=['Not Survived', 'Survived'], log=False, alpha=alpha)
plt.legend(loc='best', fontsize='x-small')

ax8 = plt.subplot2grid((plot_rows,plot_cols), (4,0), rowspan=1, colspan=1)
ax8.set_title('Survival distribution by Fare')
plt.hist([group0.Fare, group1.Fare], bins=11, range=(0, 550), stacked=True, 
        label=['Not Survived', 'Survived'], log=False, alpha=alpha)
plt.legend(loc='best', fontsize='x-small')

ax9 = plt.subplot2grid((plot_rows,plot_cols), (4,1), rowspan=1, colspan=1)
ax9.set_title('Survival distribution by Fare')
plt.hist([group0[group0.Fare>50].Fare, group1[group1.Fare>50].Fare], bins=11, range=(0, 550), stacked=True, 
        label=['Not Survived', 'Survived'], log=False, alpha=alpha)
plt.legend(loc='best', fontsize='x-small')
plt.subplots_adjust(wspace=0.3, hspace=0.3)


# Let's go deeper exploration to see if a child was survived or not, how will their parents survival?

# In[ ]:


childgrouped = train[train['Age']<19].groupby(['Survived'])
childgroup0 = childgrouped.get_group(0)
childgroup1 = childgrouped.get_group(1)
parent = train[(train['Age']>18)&(train['Parch']>0)]

merged0 = pd.merge(childgroup0, parent, how='left', on='Ticket')
merged0 = merged0[['Survived_x', 'Sex_x', 'Age_x', 'Survived_y', 'Sex_y', 'Age_y', 'Ticket']]
merged0 = merged0[merged0.Survived_y>=0]
fig = plt.figure(figsize=(8, 4))
plot_rows = 2
plot_cols = 1
ax1 = plt.subplot2grid((plot_rows,plot_cols), (0,0), rowspan=1, colspan=1)
bottom = merged0.Survived_y.value_counts().index
width1 = merged0[merged0['Sex_y']=='female'].Survived_y.value_counts()
plt.barh(bottom, width1, 0.8, 0.0, color='blue', label='mother', alpha=0.6)
width2 = merged0[merged0['Sex_y']=='male'].Survived_y.value_counts()
plt.barh(width2.index, width2, 0.8, width1[width2.index], color='green', label='father', alpha=0.6)
plt.legend(loc='best', fontsize='x-small')
ax1.set_yticks([0.4, 1.4])
ax1.set_yticklabels(['Not Survived Parents', 'Survived Parents'], fontsize='small')
ax1.set_title('Parents survival distribution by not survived child')

merged1 = pd.merge(childgroup1, parent, how='left', on='Ticket')
merged1 = merged1[['Survived_x', 'Sex_x', 'Age_x', 'Survived_y', 'Sex_y', 'Age_y', 'Ticket']]
merged1 = merged1[merged1.Survived_y>=0]
ax2 = plt.subplot2grid((plot_rows,plot_cols), (1,0), rowspan=1, colspan=1)
bottom = merged1.Survived_y.value_counts().index
width1 = merged1[merged1['Sex_y']=='female'].Survived_y.value_counts()
plt.barh(bottom, width1, 0.8, 0.0, color='blue', label='mother', alpha=0.6)
width2 = merged1[merged1['Sex_y']=='male'].Survived_y.value_counts()
plt.barh(width2.index, width2, 0.8, width1[width2.index], color='green', label='father', alpha=0.6)
plt.legend(loc='best', fontsize='x-small')
ax2.set_yticks([0.4, 1.4])
ax2.set_yticklabels(['Not Survived Parents', 'Survived Parents'], fontsize='small')
ax2.set_title('Parents survival distribution by survived child')

plt.subplots_adjust(hspace=1.0)


# From the above analysis, we can conclude that if a child is not survived, then the child's parents
# may not survived also. If a child is survived, the child's mother has big chance to survive than 
# father.

# Now let's take a look at the people's survival status if he/she had friends or Sibsp, based on the same ticket number. Note it's better to based on the ticket number to judge if a passenger had friends or SibSp in the boat than just look at the SibSp field. For example, take a look at the passengers of ticket number 1601, these people are most likely from the same area and had some relationship with each other, and I suspect most of their surname should be same, but mis-spelled. Don't ask me why I know that, I come from the Middle Kingdom, :-)

# In[ ]:


ticket = train['Ticket'][train['Parch']==0]
ticket_dup = ticket.duplicated(False)
index = ticket_dup[ticket_dup==True].index
new_train = train.loc[index]
new_train['FriendsSurvived'] = -1
for i in range(0, len(index)):
    ticketID = new_train.loc[index[i]]['Ticket']
    passengerID = new_train.loc[index[i]]['PassengerId']
    survived = new_train['Survived'][(new_train['Ticket']==ticketID)&(new_train['PassengerId']!=passengerID)]
    new_train.loc[index[i], 'FriendsSurvived'] = round(float(survived.sum())/len(survived))
print(new_train[(new_train['Sex']=='female')&(new_train['Survived']==0)].FriendsSurvived.value_counts())
print(new_train[(new_train['Sex']=='male')&(new_train['Survived']==0)].FriendsSurvived.value_counts())
print(new_train[(new_train['Sex']=='female')&(new_train['Survived']==1)].FriendsSurvived.value_counts())
print(new_train[(new_train['Sex']=='male')&(new_train['Survived']==1)].FriendsSurvived.value_counts())


# In[ ]:


fig = plt.figure(figsize=(8, 4))
plot_rows = 2
plot_cols = 1
ax1 = plt.subplot2grid((plot_rows,plot_cols), (0,0), rowspan=1, colspan=1)
width1 = new_train[(new_train['Sex']=='female')&(new_train['Survived']==0)].FriendsSurvived.value_counts()
plt.barh(width1.index, width1, 0.8, 0.0, color='blue', label='Not survived female', alpha=0.6)
width2 = new_train[(new_train['Sex']=='male')&(new_train['Survived']==0)].FriendsSurvived.value_counts()
plt.barh(width2.index, width2, 0.8, [width1, 0.0], color='green', label='Not survived male', alpha=0.6)
plt.legend(loc='best', fontsize='x-small')
ax1.set_yticks([0.4, 1.4])
ax1.set_yticklabels(['Friends not survived', 'Friends survived'], fontsize='small')
ax1.set_title('Not survived sex distribution by friends survival')

ax2 = plt.subplot2grid((plot_rows,plot_cols), (1,0), rowspan=1, colspan=1)
width1 = new_train[(new_train['Sex']=='female')&(new_train['Survived']==1)].FriendsSurvived.value_counts()
plt.barh(width1.index, width1, 0.8, 0.0, color='blue', label='Survived female', alpha=0.6)
width2 = new_train[(new_train['Sex']=='male')&(new_train['Survived']==1)].FriendsSurvived.value_counts()
plt.barh(width2.index, width2, 0.8, width1[width2.index], color='green', label='Survived male', alpha=0.6)
plt.legend(loc='best', fontsize='x-small')
ax2.set_yticks([0.4, 1.4])
ax2.set_yticklabels(['Friends not survived', 'Friends survived'], fontsize='small')
ax2.set_title('Survived sex distribution by friends survival')

plt.subplots_adjust(hspace=1.0)


# So we can conclude from the above figures:
# <p>If a woman has friends/SibSp, she was survived if her friends/SibSp survived, and not survived if her friends/SibSp not survived.</p>
# <p>If a man has friends/SibSp, he was about 19/(19+26)=42% chance to survive if his friends/SibSp survived, and 2/(2+38)=5% chance to survive if his friends/SibSp not survived.</p>

# Now let's convert the features into category integers. And we will create a new feature "FamilySize" which equals the sum of SibSp and Parch

# In[ ]:


sex_to_int = {'male':1, 'female':0}
train['SexInt'] = train['Sex'].map(sex_to_int)
embark_to_int = {'S': 0, 'C':1, 'Q':2}
train['EmbarkedInt'] = train['Embarked'].map(embark_to_int)
train['EmbarkedInt'] = train['EmbarkedInt'].fillna(0)
print(train.describe())
test['SexInt'] = test['Sex'].map(sex_to_int)
test['EmbarkedInt'] = test['Embarked'].map(embark_to_int)
test['EmbarkedInt'] = test['EmbarkedInt'].fillna(0)
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
train['FamilySize'] = train['SibSp'] + train['Parch']
test['FamilySize'] = test['SibSp'] + test['Parch']


# And let's create some more new features to indicate if a passenger has friends/SibSp and how many of them are survived or not survived by sex.

# In[ ]:


ticket = train[train['Parch']==0]
ticket = ticket.loc[ticket.Ticket.duplicated(False)]
grouped = ticket.groupby(['Ticket'])
#The Friends field indicate if the passenger has frineds/SibSp in the boat.
train['Friends'] = 0
#The below fields statistic how many are survived or not survived by sex.
train['Male_Friends_Survived'] = 0
train['Male_Friends_NotSurvived'] = 0
train['Female_Friends_Survived'] = 0
train['Female_Friends_NotSurvived'] = 0
for (k, v) in grouped.groups.items():
    for i in range(0, len(v)):
        train.loc[v[i], 'Friends'] = 1
        train.loc[v[i], 'Male_Friends_Survived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='male')&(train.Survived==1)].Survived.count()
        train.loc[v[i], 'Male_Friends_NotSurvived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='male')&(train.Survived==0)].Survived.count()
        train.loc[v[i], 'Female_Friends_Survived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='female')&(train.Survived==1)].Survived.count()
        train.loc[v[i], 'Female_Friends_NotSurvived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='female')&(train.Survived==0)].Survived.count()


# In[ ]:


test_ticket = test[test['Parch']==0]
test['Friends'] = 0
test['Male_Friends_Survived'] = 0
test['Male_Friends_NotSurvived'] = 0
test['Female_Friends_Survived'] = 0
test['Female_Friends_NotSurvived'] = 0

grouped = test_ticket.groupby(['Ticket'])
for (k, v) in grouped.groups.items():
    temp_df = train[train.Ticket==k]
    length = temp_df.shape[0]
    if temp_df.shape[0]>0:
        for i in range(0, len(v)):
            test.loc[v[i], 'Friends'] = 1
            test.loc[v[i], 'Male_Friends_Survived'] = temp_df[(temp_df.Sex=='male')&(temp_df.Survived==1)].shape[0]
            test.loc[v[i], 'Male_Friends_NotSurvived'] = temp_df[(temp_df.Sex=='male')&(temp_df.Survived==0)].shape[0]
            test.loc[v[i], 'Female_Friends_Survived'] = temp_df[(temp_df.Sex=='female')&(temp_df.Survived==1)].shape[0]
            test.loc[v[i], 'Female_Friends_NotSurvived'] = temp_df[(temp_df.Sex=='female')&(temp_df.Survived==0)].shape[0]


# And let's create some more new features to indicate if a passenger has Parents/Child and their survival status.

# In[ ]:


train['FatherOnBoard'] = 0
train['FatherSurvived'] = 0
train['MotherOnBoard'] = 0
train['MotherSurvived'] = 0
train['ChildOnBoard'] = 0
train['ChildSurvived'] = 0
train['ChildNotSurvived'] = 0
grouped = train[train.Parch>0].groupby('Ticket')
for (k, v) in grouped.groups.items():
    for i in range(0, len(v)):
        if train.loc[v[i], 'Age']<19:
            temp = train[(train.Ticket==k)&(train.Age>18)]
            if temp[temp.SexInt==1].shape[0] == 1:
                train.loc[v[i], 'FatherOnBoard'] = 1
                train.loc[v[i], 'FatherSurvived'] = temp[temp.SexInt==1].Survived.sum()
            if temp[temp.SexInt==0].shape[0] == 1:
                train.loc[v[i], 'MotherOnBoard'] = 1
                train.loc[v[i], 'MotherSurvived'] = temp[temp.SexInt==0].Survived.sum()
        else:
            temp = train[(train.Ticket==k)&(train.Age<19)]
            length = temp.shape[0]
            if length>0:
                train.loc[v[i], 'ChildOnBoard'] = 1
                train.loc[v[i], 'ChildSurvived'] = temp[temp.Survived==1].shape[0]
                train.loc[v[i], 'ChildNotSurvived'] = temp[temp.Survived==0].shape[0]
                


# In[ ]:


test['FatherOnBoard'] = 0
test['FatherSurvived'] = 0
test['MotherOnBoard'] = 0
test['MotherSurvived'] = 0
test['ChildOnBoard'] = 0
test['ChildSurvived'] = 0
test['ChildNotSurvived'] = 0
grouped = test[test.Parch>0].groupby('Ticket')
for (k, v) in grouped.groups.items():
    temp = train[train.Ticket==k]
    length = temp.shape[0]
    if length>0:
        for i in range(0, len(v)):
            if test.loc[v[i], 'Age']<19:
                if temp[(temp.SexInt==1)&(temp.Age>18)].shape[0] == 1:
                    test.loc[v[i], 'FatherOnBoard'] = 1
                    test.loc[v[i], 'FatherSurvived'] = temp[(temp.SexInt==1)&(temp.Age>18)].Survived.sum()
                if temp[(temp.SexInt==0)&(temp.Age>18)].shape[0] == 1:
                    test.loc[v[i], 'MotherOnBoard'] = 1
                    test.loc[v[i], 'MotherSurvived'] = temp[(temp.SexInt==0)&(temp.Age>18)].Survived.sum()
            else:
                length = temp[temp.Age<19].shape[0]
                if length>0:
                    test.loc[v[i], 'ChildOnBoard'] = 1
                    test.loc[v[i], 'ChildSurvived'] = temp[(temp.Age<19)&(temp.Survived==1)].shape[0]
                    test.loc[v[i], 'ChildNotSurvived'] = temp[(temp.Age<19)&(temp.Survived==0)].shape[0]


# Now let's take a look if the embarked port has impact to survival rate.

# In[ ]:


fig = plt.figure(figsize=(8, 1))
grouped = train.groupby(['Survived'])
group0 = grouped.get_group(0)
group1 = grouped.get_group(1)

ax1 = plt.subplot2grid((1,2), (0,0), rowspan=1, colspan=2)
bottom = group0.EmbarkedInt.value_counts().index
width1 = group0.EmbarkedInt.value_counts()
plt.barh(bottom, width1, 0.8, 0.0, color='blue', label='Not Survived', alpha=0.6)
width2 = group1.EmbarkedInt.value_counts()
plt.barh(bottom, width2, 0.8, width1, color='green', label='Survived', alpha=0.6)
plt.legend(loc='best', fontsize='x-small')
ax1.set_yticks([0.4, 1.4, 2.4])
ax1.set_yticklabels(['Southampton', 'Cherbourg', 'Queenstown'], fontsize='small')
ax1.set_title('Survival distribution by Embarked')


# More feature process work, get the title from the names and map to a new feature, cut the fare and age into category.

# In[ ]:


title_to_int = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':1, 'Dr':4, 'Rev':4, 'Mlle':2, 'Major':4, 'Col':4,
        'Ms':3, 'Lady':3, 'the Countess':4, 'Sir':4, 'Mme':3, 'Capt':4, 'Jonkheer':4, 'Don':1, 'Dona':3}
train['TitleInt'] = train['Title'].map(title_to_int)
test['TitleInt'] = test['Title'].map(title_to_int)
train.loc[train[train['Age']<13].index, 'TitleInt'] = 5
test.loc[test[test['Age']<13].index, 'TitleInt'] = 5

train['FareCat'] = pd.cut(train['Fare'], [-0.1, 50, 100, 150, 200, 300, 1000], right=True, 
        labels=[0, 1, 2, 3, 4, 5])
test['FareCat'] = pd.cut(test['Fare'], [-0.1, 50, 100, 150, 200, 300, 1000], right=True, 
        labels=[0, 1, 2, 3, 4, 5])
train['AgeCat'] = pd.cut(train['Age'], [-0.1, 12.1, 20, 30, 35, 40, 45, 50, 55, 65, 100], right=True, 
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test['AgeCat'] = pd.cut(test['Age'], [-0.1, 12.1, 20, 30, 35, 40, 45, 50, 55, 65, 100], right=True, 
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# After the new features created, let's check if a passenger's title has impact to survival rate.

# In[ ]:


fig = plt.figure(figsize=(8, 1))
grouped = train.groupby(['Survived'])
group0 = grouped.get_group(0)
group1 = grouped.get_group(1)

ax1 = plt.subplot2grid((1,2), (0,0), rowspan=1, colspan=2)
bottom = group0.TitleInt.value_counts().index
width1 = group0.TitleInt.value_counts()
plt.barh(bottom, width1, 0.8, 0.0, color='blue', label='Not Survived', alpha=0.6)
width2 = group1.TitleInt.value_counts()
plt.barh(bottom, width2, 0.8, width1, color='green', label='Survived', alpha=0.6)
plt.legend(loc='best', fontsize='x-small')
ax1.set_yticks([1.4, 2.4, 3.4, 4.4, 5.4])
ax1.set_yticklabels(['Mr', 'Miss', 'Mrs', 'Profession', 'Child'], fontsize='small')
ax1.set_title('Survival distribution by Title')


# Finally go to the prediction part. First we will choose which columns to trian and predict. Then we will split the train and test parts of the train dataset.

# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat']
#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeInt', 'TitleInt', 'Fare']
#columns = ['Pclass', 'FamilySize', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat']
#columns = ['Pclass', 'FamilySize', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat',
#        'Friends', 'FriendsSex', 'FriendsSurvived']
#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat',
#        'Friends', 'FriendsSex', 'FriendsSurvived', 'FatherSurvived', 'MotherSurvived', 'ChildSurvived']
#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat']
#        'Friends', 'FriendsSex', 'FriendsSurvived']
#columns = ['Pclass', 'SexInt', 'EmbarkedInt', 'Age', 'TitleInt','Fare', 'Friends', 'FriendsSex', 'FriendsSurvived', 'FatherSurvived', 'MotherSurvived', 'ChildSurvived']
columns = ['Pclass', 'SexInt', 'EmbarkedInt', 'Age', 'TitleInt','Fare', 
        'Friends', 'Male_Friends_Survived', 'Male_Friends_NotSurvived', 'Female_Friends_Survived', 'Female_Friends_NotSurvived',
        'MotherOnBoard', 'MotherSurvived', 'ChildOnBoard', 'ChildSurvived', 'ChildNotSurvived']
X_train, X_test, y_train, y_test = train_test_split(train[columns], train['Survived'], test_size=0.2, random_state=123)

#Check the features importance. 
#selected = SelectKBest(f_classif, 18)
#selected.fit(X_train, y_train)
#X_train_selected = selected.transform(X_train)
#X_test_selected = selected.transform(X_test)
#print(selected.scores_)
#print(selected.pvalues_)


# From the above result, the Feature importance are SexInt>TitleInt>PClass>Fare>Male_Friends_Survived>Female_Friends_Survived>

# Now let's first use RandomForest to train and predict. First we can use the GridSearch to find the best param.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [10, 50, 100, 150], 'min_samples_leaf': [1, 2, 4, 8], 
        'max_depth': [None, 5, 10, 50], 'max_features': [None, 'auto'], 'min_samples_split': [2, 4, 8]}
rfc = RandomForestClassifier(criterion='gini', min_weight_fraction_leaf=0.0, 
        max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, 
        n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None)
classifer = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1)
#classifer.fit(X_train, y_train)
#print(classifer.grid_scores_)
#print(classifer.best_params_)
#print(X_train.info())


# The Grid search Best params are:
# {'min_samples_split': 2, 'max_depth': 10, 'n_estimators': 150, 'min_samples_leaf': 1, 'max_features': 'auto'}, we can based these params to train and predict.

# In[ ]:


rfc = RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=10, 
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
        max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, 
        oob_score=False, n_jobs=1, random_state=232, verbose=0, warm_start=False, class_weight=None)

rfc.fit(X_train, y_train)
result = rfc.predict(X_test)
rightnum = 0

for i in range(0, result.shape[0]):
    if result[i] == y_test.iloc[i]:
        rightnum += 1
print(rightnum/result.shape[0])


rfc.fit(train[columns], train['Survived'])
predict_rf = rfc.predict(test[columns])

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predict_rf
    })
submission.to_csv("titanic_predict_RF.csv", index=False)


# The RF gives a 0.88268 accuracy rate on the test parts. And generate the prediction on the test dataset. The prediction score is 0.78947.

# Let's use XGB to do the train and prediton, to compare with RF. The accuracy rate of the test pars are 0.8715, close to RF. The prediction score is 0.76077

# In[ ]:


import xgboost as xgb
xgbclassifer = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic', nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
xgbclassifer.fit(X_train, y_train)
result = xgbclassifer.predict(X_test)
#print(result[:10])
rightnum = 0
for i in range(0, result.shape[0]):
    if result[i] == y_test.iloc[i]:
        rightnum += 1
print(rightnum/result.shape[0])

xgbclassifer.fit(train[columns], train['Survived'])
predict_xgb = xgbclassifer.predict(test[columns])

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predict_xgb
    })
submission.to_csv("titanic_predict_xgb.csv", index=False)


# Let's use Neural Network to do the train and prediton, to compare with RF. The accuracy rate of the test pars are 0.8603, not good as RF and XGB. But the prediction score is highest, 0.8134

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2, l1
from sklearn.preprocessing import StandardScaler

stdScaler = StandardScaler()
X_train_scaled = stdScaler.fit_transform(X_train)
X_test_scaled = stdScaler.transform(X_test)
model = Sequential()
#model.add(Dense(700, input_dim=7, init='normal', activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1600, input_dim=16, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, init='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.fit(X_train_scaled, y_train, nb_epoch=20, batch_size=32)
result = model.predict(X_test_scaled)
rightnum = 0
for i in range(0, result.shape[0]):
    if result[i] >= 0.5:
        result[i] = 1
    else:
        result[i] = 0
    if result[i] == y_test.iloc[i]:
        rightnum += 1
print(rightnum/result.shape[0])

train_scaled = stdScaler.fit_transform(train[columns])
test_scaled = stdScaler.transform(test[columns])
model.fit(train_scaled, train['Survived'], nb_epoch=20, batch_size=32, verbose=0)
predict_NN = model.predict(test_scaled)
print(predict_NN.shape)
for i in range(0, predict_NN.shape[0]):
    if predict_NN[i] >= 0.5:
        predict_NN[i] = 1
    else:
        predict_NN[i] = 0
        
predict_NN = predict_NN.reshape((predict_NN.shape[0]))
predict_NN = predict_NN.astype('int')
print(predict_NN.shape)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predict_NN
    })
submission.to_csv("titanic_predict_NN.csv", index=False)


# Test with SVM, the result is 0.7150, not a good result. Seems need to fine tune the parameters.

# In[ ]:


from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
result = clf.predict(X_test)
rightnum = 0
for i in range(0, result.shape[0]):
    if result[i] == y_test.iloc[i]:
        rightnum += 1
print(rightnum/result.shape[0])

predict_svm = clf.predict(test[columns])


# And let's try the genetic program, I am using the DEAP for it. As the genetic program needs a lot of time to train(I use 300 generation), it will exceed the notebook run time limit. I just post the genetic program result here. The prediction gives a score of 0.79904

# In[ ]:


from deap import gp
import itertools
import operator
import math
import numpy as np

train_GP = train[columns]
test_GP = test[columns]
train_GP = train_GP.astype('float')
test_GP = test_GP.astype('float')

pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 16), float, "IN")

pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# Define a protected division function
def protectedDiv(left, right):
    if right != 0.0:
        return left/right
    else:
        return 1
def protectedSqrt(x):
    return math.sqrt(abs(x))
pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(protectedDiv, [float,float], float)
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.tanh, [float], float)
pset.addPrimitive(math.hypot, [float, float], float)
pset.addPrimitive(max, [float, float], float)
pset.addPrimitive(min, [float, float], float)
pset.addPrimitive(protectedSqrt, [float], float)

# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

expr = 'sub(min(add(cos(cos(mul(8.791524057280029, add(0.8813426844222205, IN2)))), protectedDiv(sub(protectedDiv(IN14, IN10), IN11), protectedSqrt(sub(protectedDiv(hypot(protectedDiv(if_then_else(or_(True, and_(or_(True, True), True)), IN9, add(protectedDiv(protectedDiv(0.9363229575602429, 9.457393756367038), 68.60931271115085),sub(0.76609748747378, IN15))), 9.457393756367038), sin(0.22806884104520364)), if_then_else(or_(lt(protectedDiv(if_then_else(True, IN2, IN4), protectedDiv(hypot(0.8201212481183877, 81.83905300378048), IN0)), IN5), or_(True, True)), IN9,cos(mul(0.7027184429804209, add(protectedDiv(protectedDiv(0.9363229575602429, 9.457393756367038), 9.457393756367038), sub(if_then_else(or_(False, True), cos(0.23992695841396383), hypot(2.718281828459045, IN12)), IN15)))))), IN12)))), hypot(IN4, tanh(protectedSqrt(max(protectedDiv(hypot(add(0.9095718824824596, 360.49409062712783), sin(IN9)), min(mul(97.97893140946204, protectedDiv(IN0, 0.3160036101933281)), protectedDiv(protectedDiv(if_then_else(or_(True, True), IN9, cos(mul(8.791524057280029, add(protectedDiv(IN9, 9.457393756367038), sub(add(IN9, 31.792888528447644), IN15))))), 9.457393756367038), 2.0803290751797636))), protectedSqrt(cos(mul(8.791524057280029, IN6)))))))), min(protectedSqrt(max(protectedDiv(5.010370880110474, protectedDiv(IN15, add(336.6490971360746, protectedDiv(sub(if_then_else(True, max(IN3, IN5), cos(sin(0.22806884104520364))), hypot(sin(add(add(mul(tanh(max(IN4, 95.88858811362302)), 235.99768570024435), 2.0803290751797636),sin(sub(protectedDiv(IN0, 0.3160036101933281), 0.9466582602925034)))), add(cos(add(49.46712430568304, 4.937708033384823)), protectedDiv(mul(97.97893140946204, protectedDiv(IN0, 0.3160036101933281)), protectedSqrt(max(protectedDiv(IN3, IN1), if_then_else(True, protectedSqrt(0.8174130427244765), 0.7377841058934647))))))),  if_then_else(or_(and_(or_(or_(False, False), False), True), True), sub(IN14,  0.22806884104520364), IN9))))), protectedSqrt(max(protectedDiv(IN3, IN1),  cos(546.215547398933))))), IN3))'
gpfunc = gp.compile(expr, pset)

def output(x):
    try:
        return int(round(1.0/(1.0 + math.exp(-x))))
    except(OverflowError):
        return 0
    
predict_GP = np.zeros((test.shape[0]))
for i in range (0, test.shape[0]):
    predict_GP[i] = output(gpfunc(*test[columns].iloc[i, :]))
predict_GP = predict_GP.astype('int')
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predict_GP})
submission.to_csv("titanic_predict_gp.csv", index=False)


# And finally, let's combine the predictions by differnet models, and see if can impore the score. The result score is 0.81340

# In[ ]:


predict_combine = np.zeros((test.shape[0]))
for i in range(0, test.shape[0]):
    temp = predict_rf[i] + predict_NN[i] + predict_GP[i]
    if temp>=2:
        predict_combine[i] = 1
predict_combine = predict_combine.astype('int')

combination = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predict_combine
    })
combination.to_csv("titanic_predict_combine.csv", index=False)


# So the final result is, using neural network gets the highest score 0.8134
