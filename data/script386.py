
# coding: utf-8

# #### Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns
sns.set_style('whitegrid')


# #### Read Data

# In[ ]:


database_df = pd.read_csv("../input/database.csv")

database_df = (database_df.drop(['Record ID', 'Agency Code','Victim Ethnicity','Perpetrator Ethnicity',
                                 'Record Source'],axis=1))

#print(database_df.head())
Victim_Sex = database_df['Victim Sex'].values
Victim_Age = database_df['Victim Age'].values
#Victim_Age = Victim_Age.astype('int16')
Perpetrator_Sex = database_df['Perpetrator Sex'].values
Perpetrator_Age = database_df['Perpetrator Age'].values
#Perpetrator_Age = Perpetrator_Age.astype('int16')
Relationship = database_df['Relationship'].values
Weapon = database_df['Weapon'].values
Solved = database_df['Crime Solved'].values
V_Race = database_df['Victim Race'].values
P_Race = database_df['Perpetrator Race'].values
Crime_Type = database_df['Crime Type'].values
Agency_Name = database_df['Agency Name'].values
Agency_Type = database_df['Agency Type'].values
P_Count =  database_df['Perpetrator Count'].values
City =  database_df['City'].values
State =  database_df['State'].values
Year =  database_df['Year'].values
Month =  database_df['Month'].values
Rel_Category = database_df['Relationship'].values
W_Category = database_df['Weapon'].values

homicide = pd.DataFrame(np.column_stack((Victim_Sex,Victim_Age,Perpetrator_Sex,Perpetrator_Age,
                    Relationship,Weapon,Solved,V_Race,P_Race,Crime_Type,Agency_Name,Agency_Type,
                    Rel_Category,P_Count,City,State,Year,Month,W_Category)))
homicide.columns = ['Victim_Sex','Victim_Age','Perpetrator_Sex','Perpetrator_Age','Relationship',
                    'Weapon','Solved','V_Race','P_Race','Crime_Type','Agency_Name','Agency_Type',
                    'Rel_Category','P_Count','City','State','Year','Month','W_Category']

homicide.loc[(homicide['Relationship'] == 'Wife') | (homicide['Relationship'] == 'Ex-Wife') |
             (homicide['Relationship'] == 'Girlfriend') |
             (homicide['Relationship'] == 'Common-Law Wife'), 'Rel_Category'] = 'Partner-F'

homicide.loc[(homicide['Relationship'] == 'Husband') | (homicide['Relationship'] == 'Ex-Husband') |
             (homicide['Relationship'] == 'Boyfriend') | 
             (homicide['Relationship'] == 'Common-Law Husband'), 'Rel_Category'] = 'Partner-M'

homicide.loc[(homicide['Relationship'] == 'Father') | (homicide['Relationship'] == 'In-Law') |
             (homicide['Relationship'] == 'Mother') | (homicide['Relationship'] == 'Stepfather') |
             (homicide['Relationship'] == 'Stepmother'), 'Rel_Category'] = 'Parent'

homicide.loc[(homicide['Relationship'] == 'Daughter') | (homicide['Relationship'] == 'Son') |
             (homicide['Relationship'] == 'Stepdaughter') | 
             (homicide['Relationship'] == 'Stepson'), 'Rel_Category'] = 'Children'

homicide.loc[(homicide['Relationship'] == 'Brother') | (homicide['Relationship'] == 'Sister'),
             'Rel_Category'] = 'Sibling'

homicide.loc[(homicide['Relationship'] == 'Employee') | (homicide['Relationship'] == 'Employer') ,
             'Rel_Category'] = 'Work'

homicide.loc[(homicide['Relationship'] == 'Boyfriend/Girlfriend') & (homicide['Victim_Sex'] == 'Female'),
             'Rel_Category'] = 'Partner-F'

homicide.loc[(homicide['Relationship'] == 'Boyfriend/Girlfriend') & ((homicide['Victim_Sex'] == 'Male') |
            (homicide['Victim_Sex'] == 'Unknown')) , 'Rel_Category'] = 'Partner-M'

FV_MP = homicide[(homicide.Victim_Sex == 'Female') & (homicide.Perpetrator_Sex == 'Male')]
FV_FP = homicide[(homicide.Victim_Sex == 'Female') & (homicide.Perpetrator_Sex == 'Female')]
MV_MP = homicide[(homicide.Victim_Sex == 'Male') & (homicide.Perpetrator_Sex == 'Male')]
MV_FP = homicide[(homicide.Victim_Sex == 'Male') & (homicide.Perpetrator_Sex == 'Female')]
FV_UP = homicide[(homicide.Victim_Sex == 'Female') & (homicide.Perpetrator_Sex == 'Unknown')]
MV_UP = homicide[(homicide.Victim_Sex == 'Male') & (homicide.Perpetrator_Sex == 'Unknown')]
UV_UP = homicide[(homicide.Victim_Sex == 'Unknown') & (homicide.Perpetrator_Sex == 'Unknown')]

homicide.head(2)


# #### Exploratory Analysis of Age, Sex of Victim and Perpetrator

# In[ ]:


pd.crosstab(homicide.Victim_Sex,homicide.Perpetrator_Sex)


# In[ ]:


plt.figure(figsize=(10,20),facecolor='#eeeeee')
plt.subplots_adjust(bottom=0, left=.05, right=1.5, top=0.9, hspace=.6,wspace=.5)

plt.subplot(12, 12, (1,5))
plt.xlim([0,100])
sns.distplot(FV_MP.Perpetrator_Age,color='blue',hist=False,kde=True)
plt.xlabel('')
plt.xticks([])
plt.yticks([])

plt.subplot(12, 12, (13,41))
plt.title('Female Victim, Male Perpetrator',fontsize=18,fontweight='bold',color='mediumorchid')
plt.ylim([0,100])
plt.xlim([0,100])
plt.scatter(FV_MP.Perpetrator_Age,FV_MP.Victim_Age, s=20, c='mediumorchid', alpha=0.08)
plt.ylabel('Victim Age')
plt.xlabel('Perpetrator Age')

plt.subplot(12, 12, (18,42))
plt.ylim([0,100])
sns.distplot(FV_MP.Victim_Age,color='hotpink',vertical=True,hist=False,kde=True)
plt.ylabel('')
plt.xticks([])
plt.yticks([])

plt.subplot(12, 12, (7,11))
plt.xlim([0,100])
sns.distplot(MV_FP.Perpetrator_Age,color='hotpink',hist=False,kde=True)
plt.xlabel('')
plt.xticks([])
plt.yticks([])

plt.subplot(12, 12, (19,47))
plt.title('Male Victim, Female Perpetrator',fontsize=18,fontweight='bold',color='mediumpurple')
plt.ylim([0,100])
plt.xlim([0,100])
plt.scatter(MV_FP.Perpetrator_Age,MV_FP.Victim_Age, s=20, c='mediumpurple', alpha=0.08)
plt.xlabel('Perpetrator Age')


plt.subplot(12, 12, (24,48))
plt.ylim([0,100])
sns.distplot(MV_FP.Victim_Age,color='blue',vertical=True,hist=False,kde=True)
plt.ylabel('')
plt.xticks([])
plt.yticks([])

plt.subplot(12, 12, (49,53))
plt.xlim([0,100])
sns.distplot(FV_FP.Perpetrator_Age,color='hotpink',hist=False,kde=True)
plt.xlabel('')
plt.xticks([])
plt.yticks([])

plt.subplot(12, 12, (61,89))
plt.title('Female Victim, Female Perpetrator',fontsize=18,fontweight='bold',color='hotpink')
plt.ylim([0,100])
plt.xlim([0,100])
plt.scatter(FV_FP.Perpetrator_Age,FV_FP.Victim_Age, s=20, c='hotpink', alpha=0.08)
plt.ylabel('Victim Age')
plt.xlabel('Perpetrator Age')

plt.subplot(12, 12, (66,90))
plt.ylim([0,100])
sns.distplot(FV_FP.Victim_Age,color='hotpink',vertical=True,hist=False,kde=True)
plt.ylabel('')
plt.xticks([])
plt.yticks([])


plt.subplot(12, 12, (55,59))
plt.xlim([0,100])
sns.distplot(MV_MP.Perpetrator_Age,color='blue',hist=False,kde=True)
plt.xlabel('')
plt.xticks([])
plt.yticks([])

plt.subplot(12, 12, (67,95))
plt.title('Male Victim, Male Perpetrator',fontsize=18,fontweight='bold',color='blue')
plt.ylim([0,100])
plt.xlim([0,100])
plt.scatter(MV_MP.Perpetrator_Age,MV_MP.Victim_Age, s=20, c='blue', alpha=0.08)
plt.xlabel('Perpetrator Age')

plt.subplot(12, 12, (72,96))
plt.ylim([0,100])
sns.distplot(MV_MP.Victim_Age,color='blue',vertical=True,hist=False,kde=True)
plt.ylabel('')
plt.xticks([])
plt.yticks([])

plt.subplot(12, 12, (97,101))
plt.xlim([0,100])
plt.xlabel('')
plt.xticks([])
plt.yticks([])

MV_UP_j = np.random.random(len(MV_UP)) * 12
FV_UP_j = np.random.random(len(FV_UP)) * 12
UV_UP_j = np.random.random(len(UV_UP)) * 12

plt.subplot(12, 12, (109,137))
plt.title('Unknown Perpetrator',fontsize=18,fontweight='bold',color='dimgrey')
plt.ylim([0,100])
plt.xlim([0,100])
plt.scatter((MV_UP.Victim_Age*0+20+MV_UP_j),MV_UP.Victim_Age, s=20, c='blue', alpha=0.08)
plt.scatter((FV_UP.Victim_Age*0+50+FV_UP_j),FV_UP.Victim_Age, s=20, c='hotpink', alpha=0.08)
plt.scatter((UV_UP.Victim_Age*0+80+UV_UP_j),UV_UP.Victim_Age, s=20, c='darkgrey', alpha=0.08)
plt.ylabel('Victim Age')
plt.xlabel('Perpetrator Age Unknown')
plt.xticks([])

plt.subplot(12, 12, (114,138))
plt.ylim([0,100])
sns.distplot(FV_UP.Victim_Age,color='hotpink',vertical=True,hist=False,kde=True)
sns.distplot(MV_UP.Victim_Age,color='blue',vertical=True,hist=False,kde=True)
sns.distplot(UV_UP.Victim_Age,color='darkgrey',vertical=True,hist=False,kde=True)
plt.ylabel('')
plt.xticks([])
plt.yticks([])

'''
plt.subplot(12, 12, (115,143))
plt.title('Histogram - Victims Age',fontsize=18,fontweight='bold',color='dimgrey')
plt.hist(FV_MP.Victim_Age,color='mediumorchid',orientation='horizontal',linewidth=6,alpha=0.4,histtype='step')
plt.hist(MV_FP.Victim_Age,color='mediumpurple',orientation='horizontal',linewidth=6,alpha=0.4,histtype='step')
plt.hist(MV_MP.Victim_Age,color='blue',orientation='horizontal',linewidth=6,alpha=0.4,histtype='step')
plt.hist(FV_FP.Victim_Age,color='hotpink',orientation='horizontal',linewidth=6,alpha=0.4,histtype='step')
plt.hist(np.hstack((FV_UP.Victim_Age.reshape(-1),MV_UP.Victim_Age.reshape(-1),UV_UP.Victim_Age.reshape(-1))),color='darkgrey',orientation='horizontal',linewidth=6,alpha=0.4,histtype='step')
plt.ylim([0,100])
plt.xticks([])
'''
plt.show()


# In[ ]:


#### KDE Plot


# In[ ]:


#### KDE Plot


# In[ ]:


'''
f, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

s = np.linspace(0, 3, 10)

cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
sns.kdeplot(FV_MP.Perpetrator_Age, FV_MP.Victim_Age, cmap=cmap, shade=True, ax=axes[0,0])
axes[0,0].set(xlim=(0, 100), ylim=(0, 100),  title = 'Female Victim, Male Perpetrator')

cmap = sns.cubehelix_palette(start=.5, light=1, as_cmap=True)
sns.kdeplot(FV_FP.Perpetrator_Age, FV_FP.Victim_Age, cmap=cmap, shade=True, ax=axes[0,1])
axes[0,1].set(xlim=(0, 100), ylim=(0, 100),  title = 'Female Victim, Female Perpetrator')

cmap = sns.cubehelix_palette(start=1, light=1, as_cmap=True)
sns.kdeplot(MV_FP.Perpetrator_Age, MV_FP.Victim_Age, cmap=cmap, shade=True, ax=axes[1,0])
axes[1,0].set(xlim=(0, 100), ylim=(0, 100),  title = 'Male Victim, Female Perpetrator')

cmap = sns.cubehelix_palette(start=1.5, light=1, as_cmap=True)
sns.kdeplot(MV_MP.Perpetrator_Age, MV_MP.Victim_Age, cmap=cmap, shade=True, ax=axes[1,1])
axes[1,1].set(xlim=(0, 100), ylim=(0, 100),  title = 'Male Victim, Male Perpetrator')

f.tight_layout()
'''


# ### Exploratory Analysis of Weapon, Relationship, Race, State

# In[ ]:


pd.crosstab(homicide.Rel_Category,homicide.Weapon)


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(12,10),facecolor='#eeeeee')
plt.subplots_adjust(bottom=0, left=.05, right=1.5, top=0.9, hspace=.35,wspace=.35)
homicide = homicide[(homicide.Victim_Age != ' ') & (homicide.Perpetrator_Age != ' ')]
R = ['Partner-F', 'Partner-M', 'Parent', 'Children', 'Sibling','Family','Friend','Work','Neighbor','Acquaintance','Stranger','Unknown']
W = ['Handgun', 'Firearm', 'Shotgun', 'Rifle', 'Gun','Knife','Blunt Object','Unknown','Strangulation','Suffocation','Fire','Explosives','Drugs','Poison','Drowning','Fall']
homicide.loc[(homicide['Weapon'] == 'Handgun') | (homicide['Weapon'] == 'Firearm') |
             (homicide['Weapon'] == 'Shotgun') | (homicide['Weapon'] == 'Rifle')   |
             (homicide['Weapon'] == 'Gun'), 'W_Category'] = 'yellow'
homicide.loc[(homicide['Weapon'] == 'Strangulation') | (homicide['Weapon'] == 'Suffocation') , 'W_Category'] = 'steelblue'
homicide.loc[(homicide['Weapon'] == 'Drowning') | (homicide['Weapon'] == 'Fall') , 'W_Category'] = 'olive'
homicide.loc[(homicide['Weapon'] == 'Fire') | (homicide['Weapon'] == 'Explosives') , 'W_Category'] = 'palegreen'
homicide.loc[(homicide['Weapon'] == 'Drugs') | (homicide['Weapon'] == 'Poison') , 'W_Category'] = 'mediumblue'
homicide.loc[(homicide['Weapon'] == 'Knife') , 'W_Category'] = 'red'
homicide.loc[(homicide['Weapon'] == 'Blunt Object') , 'W_Category'] = 'purple'
homicide.loc[(homicide['Weapon'] == 'Unknown') , 'W_Category'] = 'silver'

for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.title(R[i],fontsize=14,fontweight='bold',color='brown')
    plt.ylim([0,100])
    plt.xlim([0,100])
    plt.scatter(homicide[homicide.Rel_Category == R[i]].Perpetrator_Age,homicide[homicide.Rel_Category == R[i]].Victim_Age, marker='o',s=10, c=homicide.W_Category, alpha=0.08)
    plt.ylabel('Victim Age')
    plt.xlabel('Perpetrator Age')
    
plt.show()


# In[ ]:


plt.figure(figsize=(12,8),facecolor='#efefef')
sns.set()
cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
sns.heatmap(pd.crosstab(homicide.Rel_Category,homicide.Weapon), annot=True, fmt="d", linewidths=.5,cmap='Reds')


# In[ ]:


plt.figure(figsize=(12,5),facecolor='#efefef')
sns.set()
cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
sns.heatmap(pd.crosstab(homicide.P_Race,homicide.Weapon), annot=True, fmt="d", linewidths=.5,cmap='Reds')


# In[ ]:


plt.figure(figsize=(12,18),facecolor='#efefef')
sns.set()
cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
sns.heatmap(pd.crosstab(homicide.Year,homicide.Weapon), annot=True, fmt="d", linewidths=.5,cmap='Reds')


# In[ ]:


plt.figure(figsize=(12,20),facecolor='#efefef')
sns.set()
cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
sns.heatmap(pd.crosstab(homicide.State,homicide.Weapon), annot=True, fmt="d", linewidths=.5,cmap='Reds')


# In[ ]:


plt.figure(figsize=(6,20),facecolor='#efefef')
sns.set()
cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
sns.heatmap(pd.crosstab(homicide.Year,homicide.Solved), annot=True, fmt="d", linewidths=.5,cmap='Blues')


# In[ ]:


plt.figure(figsize=(24,18),facecolor='#efefef')
sns.set()
cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
sns.heatmap(pd.crosstab(homicide.Year,homicide.State), annot=True, fmt="d", linewidths=.5,cmap='Reds')


# So looks like there are few outliers in Florida, see 1988 to 1991 numbers.

# In[ ]:


#### Scatterplot by Weapons


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(12,12),facecolor='#eeeeee')
plt.subplots_adjust(bottom=0, left=.05, right=1.5, top=0.9, hspace=.35,wspace=.35)
homicide = homicide[(homicide.Victim_Age != ' ') & (homicide.Perpetrator_Age != ' ')]
W = ['Handgun', 'Firearm', 'Shotgun', 'Rifle', 'Gun','Knife','Blunt Object','Unknown','Strangulation','Suffocation','Fire','Explosives','Drugs','Poison','Drowning','Fall']
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.title(W[i],fontsize=14,fontweight='bold',color='brown')
    plt.ylim([0,100])
    plt.xlim([0,100])
    plt.scatter(homicide[homicide.Weapon == W[i]].Perpetrator_Age,homicide[homicide.Weapon == W[i]].Victim_Age, marker='o',s=10, c='red', alpha=0.08)
    plt.ylabel('Victim Age')
    plt.xlabel('Perpetrator Age')

plt.show()


# In[ ]:


#### Scatterplot by Relationship


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(12,10),facecolor='#eeeeee')
plt.subplots_adjust(bottom=0, left=.05, right=1.5, top=0.9, hspace=.35,wspace=.35)
homicide = homicide[(homicide.Victim_Age != ' ') & (homicide.Perpetrator_Age != ' ')]
R = ['Partner-F', 'Partner-M', 'Parent', 'Children', 'Sibling','Family','Friend','Work','Neighbor','Acquaintance','Stranger','Unknown']
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.title(R[i],fontsize=14,fontweight='bold',color='brown')
    plt.ylim([0,100])
    plt.xlim([0,100])
    plt.scatter(homicide[homicide.Rel_Category == R[i]].Perpetrator_Age,homicide[homicide.Rel_Category == R[i]].Victim_Age, marker='o',s=10, c='red', alpha=0.08)
    plt.ylabel('Victim Age')
    plt.xlabel('Perpetrator Age')

plt.show()


# *Predictive Model*

# In[ ]:


pd.pivot_table(homicide,columns = ['Solved'],index = ['P_Race'],values=['Rec_ID'],aggfunc='count',fill_value='')


# In[ ]:


# We will predict these Race+Sex Class for these 189995 cases where crime is not solved
# and neither Perpetrator Race or Sex is known

# Our traning data will be Solved cases where Race, Sex, Age of perpetrators are known. 
# 30% from these data will be used for validation of the model

test = homicide[(homicide.Solved == 'No') & (homicide.P_Race == 'Unknown') &
                (homicide.Perpetrator_Sex == 'Unknown')]

X_y = homicide[(homicide.Solved == 'Yes') & (homicide.P_Race != 'Unknown') &
                (homicide.Perpetrator_Sex != 'Unknown') & (homicide.Perpetrator_Age != 0)]

y = X_y['P_Race']

X_train, X_val, y_train, y_val = train_test_split(X_y, y, test_size=0.3,random_state=42)

test.info()
X_y.info()

X_train.info(), X_val.info()

# Train 2 Classifier models and use it to Predict Perpetrator Race, Perpetrator Sex

