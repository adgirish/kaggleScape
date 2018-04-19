
# coding: utf-8

# *This Kernel is under construction! More notes and plots will be added in the future *

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import sys, re
import seaborn as sns
from sklearn import preprocessing
from subprocess import check_output
#print(check_output(["ls", "input"]).decode("utf8"))
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)
matplotlib.rcParams['xtick.major.pad']='10'
matplotlib.rcParams['ytick.major.pad']='10'


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train_survived = df_train['Survived']
df_all = pd.concat([df_train.drop(['Survived'], axis=1), df_test])
#df_all.info()


# In[ ]:


fig, axes = plt.subplots(1,8, figsize=(15,8))
F_count = df_train.groupby(['Sex'])['Name'].count()[0]
M_count = df_train.groupby(['Sex'])['Name'].count()[1]

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch']
df_temp = df_train.groupby(['Survived','Sex']).size().to_frame(name='Count').reset_index()
df_temp.loc[(df_temp['Sex']=='male'),'Count'] = df_temp[(df_temp['Sex']=='male')]['Count']/M_count
df_temp.loc[(df_temp['Sex']=='female'),'Count'] = df_temp[(df_temp['Sex']=='female')]['Count']/F_count
df_temp2 = df_temp.pivot(index='Sex', columns='Survived', values='Count')

ax = df_temp2.plot(ax=axes[0], kind='bar', stacked=True,legend=False)#, figsize=(1,6))#, color=('black','limegreen'))
ax.set_title('gender')

gender_list = ['male','female']
for gender in gender_list:
    gender_count = df_train[df_train['Sex']==gender].groupby(['Pclass'])['Name'].count()
    df_temp = df_train[df_train['Sex']==gender].groupby(['Pclass','Survived']).size().to_frame(name='Count').reset_index()
    for i in range(1,4):
        df_temp.loc[(df_temp['Pclass']==i),'Count'] = df_temp[(df_temp['Pclass']==i)]['Count']/gender_count[i]
    df_temp
    df_temp2 = df_temp.pivot(index='Pclass', columns='Survived', values='Count')
    ax = df_temp2.plot(ax=axes[gender_list.index(gender)+1], kind='bar', stacked=True,legend=False)#, figsize=(1,6))#, color=('black','limegreen'))
    ax.set_title(gender)
    
    
df_train['AgeCat']=pd.cut(df_train['Age'], bins=[0, 18, 100], include_lowest=True, labels=[1, 2])
df_test['AgeCat']=pd.cut(df_test['Age'], bins=[0, 18, 100], include_lowest=True, labels=[1, 2])

for gender in gender_list:
    gender_count = df_train[df_train['Sex']==gender].groupby(['AgeCat'])['Name'].count()
    df_temp = df_train[df_train['Sex']==gender].groupby(['AgeCat','Survived']).size().to_frame(name='Count').reset_index()
    for i in range(1,3):
        df_temp.loc[(df_temp['AgeCat']==i),'Count'] = df_temp[(df_temp['AgeCat']==i)]['Count']/gender_count[i]
    df_temp
    df_temp2 = df_temp.pivot(index='AgeCat', columns='Survived', values='Count')
    ax = df_temp2.plot(ax=axes[gender_list.index(gender)+3], kind='bar', stacked=True,legend=False)#, figsize=(1,6))#, color=('black','limegreen'))
    ax.set_title(gender)

df_train['FareCat']=pd.cut(df_train['Fare'],bins=[0, 25, 90, 1000], include_lowest=True, labels=[1,2,3])
df_test['FareCat']=pd.cut(df_test['Fare'],bins=[0, 25, 90, 1000], include_lowest=True, labels=[1,2,3])

for gender in gender_list:
    gender_count = df_train[df_train['Sex']==gender].groupby(['FareCat'])['Name'].count()
    df_temp = df_train[df_train['Sex']==gender].groupby(['FareCat','Survived']).size().to_frame(name='Count').reset_index()
    for i in range(1,4):
        df_temp.loc[(df_temp['FareCat']==i),'Count'] = df_temp[(df_temp['FareCat']==i)]['Count']/gender_count[i]
    df_temp
    df_temp2 = df_temp.pivot(index='FareCat', columns='Survived', values='Count')
    ax = df_temp2.plot(ax=axes[gender_list.index(gender)+5], kind='bar', stacked=True,legend=False)#, figsize=(1,6))#, color=('black','limegreen'))
    ax.set_title(gender)

df_train['FamCat']=pd.cut(df_train['FamilySize'],bins=[0,1, 4, 20], include_lowest=True, labels=[1,2,3])


gender_count = df_train.groupby(['FamCat'])['Name'].count()
df_temp = df_train.groupby(['FamCat','Survived']).size().to_frame(name='Count').reset_index()
for i in range(1,4):
    df_temp.loc[(df_temp['FamCat']==i),'Count'] = df_temp[(df_temp['FamCat']==i)]['Count']/gender_count[i]
    df_temp
    df_temp2 = df_temp.pivot(index='FamCat', columns='Survived', values='Count')
ax = df_temp2.plot(ax=axes[7], kind='bar', stacked=True)#, figsize=(1,6))#, color=('black','limegreen'))

ax.set_title('family size')
gender_count
#df_temp2
leg = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
leg.set_title('Survived', prop={'size': 18, 'weight': 'normal'})

fig.tight_layout()


# In[ ]:


matplotlib.pyplot.figure(figsize=(12, 6))
df_temp = df_train.groupby(['Survived','Sex']).size().to_frame(name='Count').reset_index()
sns.swarmplot(x="Sex", y="Age", hue='Survived', data=df_train,dodge=True)#, jitter=True);


# In[ ]:


sns.factorplot(x="Pclass", y="Age", hue='Survived', data=df_train,
               col="Sex", kind="swarm", dodge=True, size=5, aspect=1); #, jitter=True);


# In[ ]:


sns.factorplot(x="Pclass", y="Fare", hue='Survived', data=df_train,
               col="Sex", kind="swarm",size=5, aspect=1, dodge=True); 


# In[ ]:


matplotlib.pyplot.figure(figsize=(12, 6))
df_train['Title'] = df_train.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.swarmplot(x="Title", y="Age", hue='Survived', data=df_train, size=5); 
plt.xticks(rotation=90)


# In[ ]:


df_train['Title'] = df_train['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
df_train['Title'] = df_train['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col',
                                            'Capt', 'Countess', 'Jonkheer'],'VIP')


# In[ ]:


matplotlib.pyplot.figure(figsize=(12, 6))
sns.swarmplot(x="Title", y="Age", hue='Survived', data=df_train, dodge=True); 
plt.xticks(rotation=90);


# In[ ]:


matplotlib.pyplot.figure(figsize=(12, 6))
sns.swarmplot(x="Title", y="Fare", hue='Survived', data=df_train, dodge=True); 
plt.xticks(rotation=90);


# In[ ]:


df_train['Has_Cabin'] = ~df_train.Cabin.isnull()

fig, axes = plt.subplots(1,5, figsize=(8,8))
title_list = df_train['Title'].unique().tolist()
cabin_list = [False,True]

for title in title_list:
    title_count = df_train[df_train['Title']==title].groupby(['Has_Cabin'])['Name'].count()
    df_temp = df_train[df_train['Title']==title].groupby(['Has_Cabin','Survived']).size().to_frame(name='Count').reset_index()
    for cabin in cabin_list:
        df_temp.loc[(df_temp['Has_Cabin']==cabin),'Count'] = df_temp[(df_temp['Has_Cabin']==cabin)]['Count']/int(title_count[cabin_list.index(cabin)])
        df_temp2 = df_temp.pivot(index='Has_Cabin', columns='Survived', values='Count')
    ax = df_temp2.plot(ax=axes[title_list.index(title)], kind='bar', stacked=True,legend=False)#, figsize=(1,6))#, color=('black','limegreen'))
    ax.set_title(title)

leg = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
leg.set_title('Survived', prop={'size': 18, 'weight': 'normal'})
fig.tight_layout()

