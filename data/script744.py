
# coding: utf-8

# **INTRODUCTION**
# <a href="http://imgbb.com/"><img src="http://image.ibb.co/hUkmtk/int.jpg" alt="int" border="0"></a>

# In the United States, animal bites are often reported to law enforcement (such as animal control). The main concern with an animal bite is that the animal may be rabid. This dataset includes information on over 9,000 animal bites which occurred near Louisville, Kentucky from 1985 to 2017 and includes information on whether the animal was quarantined after the bite occurred and whether that animal was rabid.

# **Content:**
# 1. Features of Animal Bite Data
# 1. Animal Species
# 1. Animal Name  VS  Number of Bite
# 1. The Most Aggressive 10 Species
# 1. When Animals Bite
# 1. Male or Female is More Dangerous
# 1. Probability of Being Rabid
# 1.  Common Feature of 4 Rabid Animal
# 
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np #linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization library
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/Health_AnimalBites.csv')


# In[ ]:


# There are 15 features
data.columns


# In[ ]:


data.head()


# **Features of Animal Bite Data**

# * bite_date: The date the bite occurred
# * SpeciesIDDesc: The species of animal that did the biting
# * BreedIDDesc: Breed (if known)
# * GenderIDDesc: Gender (of the animal)
# * color: color of the animal
# * vaccination_yrs: how many years had passed since the last vaccination
# * vaccination_date: the date of the last vaccination
# * victim_zip: the zipcode of the victim
# * AdvIssuedYNDesc: whether advice was issued
# * WhereBittenIDDesc: Where on the body the victim was bitten
# * quarantine_date: whether the animal was quarantined
# * DispositionIDDesc: whether the animal was released from quarantine
# * head_sent_date: the date the animal’s head was sent to the lab
# * release_date: the date the animal was released
# * ResultsIDDesc: results from lab tests (for rabies)

# **Animal Species**

# In[ ]:


# There are 9 animals name 
species = data.SpeciesIDDesc
species = species.dropna() #drop nan values in species feature
speciesOfAnimal = species.unique()
print(speciesOfAnimal)


# **Animal Name  VS  Number of Bite **
# * Number of dog bite = 7029
# * Number of cat bite = 1568
# * Number of bat bite = 237
# * Number of raccoon bite = 27
# * Number of other bite = 11
# * Number of rabbit bite = 3
# * Number of horse bite = 5
# * Number of skunk bite = 1
# * Number of ferret bite = 4

# In[ ]:



animal_list = []
for  i in speciesOfAnimal:
    animal_list.append(len(species[species==i]))
ax = sns.barplot(x=speciesOfAnimal, y =animal_list)
plt.title('Number of Species Bite')
plt.xticks(rotation=90)
print(animal_list)


# **When Animals Bite**
# * Monthly distribution of dog, cat and bat bites are visualized. 

# In[ ]:


def animal_month(animal,data):
    month_list= ['01','02','03','04','05','06','07','08','09','10','11','12']
    numberOfAnimal = []
    for i in month_list:
        x = data.loc[(data['SpeciesIDDesc']==animal)&(data['bite_date'].str.split('-').str[1]==i)]
        numberOfAnimal.append(len(x))
    ax = sns.barplot(x=month_list,y=numberOfAnimal,palette  = "Blues")
    plt.title(animal + ' bite for 12 month')


# In[ ]:


# Dogs mostly bites at 5th month
animal_month('DOG',data)


# In[ ]:


# Cats mostly bites at 6th month
animal_month('CAT',data)


# In[ ]:


# Bats mostly bites at 8th month
animal_month('BAT',data)


# **The Most Aggressive 10 Species**
# 1. PIT BULL
# 1. GERM SHEPHERD
# 1. LABRADOR RETRIV
# 1. BOXER
# 1. CHICHAUHUA
# 1. SHIH TZU
# 1. BEAGLE   (that shocked me bc its my favourite :) )
# 1. ROTTWEILER
# 1. AAUST. TERR
# 1. DACHSHUND
# <a href="http://ibb.co/cGdjeQ"><img src="http://preview.ibb.co/denczQ/pit.jpg" alt="pit" border="0"></a>
# 

# In[ ]:


count = data.BreedIDDesc.value_counts()
plt.figure(figsize=(15,8))
ax = sns.barplot(x=count[0:10].index,y=count[0:10])
plt.xticks(rotation=20)
plt.ylabel("Number of Bite")
print(count[0:10].index)


# **Where the Animals Bite**
# * Where dogs, cats and bats bite people are visualized. While changing *bite_place* method, you can observe where other animals bite.

# In[ ]:


def bite_place(animal,data):
    bitePlaces = data.WhereBittenIDDesc.unique()
    #print(bitePlaces)
    head = data.loc[(data['SpeciesIDDesc']==animal)&(data['WhereBittenIDDesc']=='HEAD')]
    body = data.loc[(data['SpeciesIDDesc']==animal)&(data['WhereBittenIDDesc']=='BODY')]
    numberOfHead = len(head)
    numberOfBody = len(body)
    total = numberOfHead+numberOfBody
    fig1=plt.figure()
    ax1=fig1.add_subplot(111,aspect='equal')
    ax1.add_patch(
        patches.Rectangle((0.3,0.1),0.4,0.5,alpha=numberOfBody/float(total),color='r')
    )
    circle = plt.Circle((0.5,0.7),0.1,color='r',alpha=numberOfHead/float(total))
    ax1.add_artist(circle)
    plt.text(0.45,0.7,round(numberOfHead/float(total),2))
    plt.text(0.45,0.4,round(numberOfBody/float(total),2))
    plt.title(str(animal)+' Bite Probability of Head and Body')
    plt.axis('off')


# In[ ]:


#Dog bites 19% head and 81% body
bite_place('DOG',data)


# In[ ]:


#Cat bites 4% head and 96% body
bite_place('CAT',data)


# In[ ]:


#Bat bites 5% head and 95% body
bite_place('BAT',data)


# **Male or Female is More Dangerous**

# In[ ]:



gender = ['MALE','FEMALE']
count_gender = data.GenderIDDesc.value_counts()
plt.figure(figsize= (7,8))
x = sns.barplot(x=gender, y= count_gender[0:2])
plt.ylabel('Number of Bite ')
plt.xticks(rotation = 20)
plt.title('MALE VS FEMALE')
print(count_gender[0:2])


# **Probability of Being Rabid**
# 

# In[ ]:


def rabid_prob(animal,data):
    labels = ['POSITIVE','NEGATIVE']
    colors = ['red','green']
    explode = [0.1,0]
    p = data.loc[(data['SpeciesIDDesc']==animal)&(data['ResultsIDDesc']=='POSITIVE')]
    n = data.loc[(data['SpeciesIDDesc']==animal)&(data['ResultsIDDesc']=='NEGATIVE')]
    sizes = [len(p),len(n)]
    print(sizes)
    if len(p)==0:
        labels = ['','NEGATIVE']
    plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct ='%1.1f&&')
    plt.axis('equal')
    plt.title(animal + ' Rabid Probability')
    plt.show()


# In[ ]:


# Dog rabid probability is 1.7%
rabid_prob('DOG',data)


# In[ ]:


# Bat rabid probability is 1.8%
rabid_prob('BAT',data)


# There are total 4 rabid record.

# ** Common Feature of 4 Rabid Animal**
# * There are 4 rabid animal that are 3 bat and 1 dog.
# * Information of bats are nan
# * Information of dog is greatpyreneese, female and white.

# In[ ]:


a = data.loc[(data['ResultsIDDesc']=='POSITIVE')]
a = a.loc[:,['bite_date','SpeciesIDDesc','BreedIDDesc','GenderIDDesc','color','ResultsIDDesc']]
print(a)


# **CONCULUSION**
# * In my opinion that is not related to result of data analysis, people who have dog are more dangerous than others who have cats. Because animals do  what their owners say e
# * ** If you have any suggest or question, I will be happy to hear it.**
