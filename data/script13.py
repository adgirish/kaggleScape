
# coding: utf-8

# # INTRODUCTION
# * In this kernel, I am interested with the effect of the raising hands, visiting resources and viewing announcements.
# * Then found two of students have low level grade although they have higher values of raising hands, visiting resources and viewing announcements and start researching it **why !!!**
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import seaborn as sns
sns.set(style='ticks')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv('../input/xAPI-Edu-Data.csv')


# In[ ]:


# there are no nan values or missing something in 
print(data.info())      


# In[ ]:


# These are the features names
print(data.columns)     


# In[ ]:


melt = pd.melt(data,id_vars='Class',value_vars=['raisedhands','VisITedResources','AnnouncementsView'])


# * As it can be seen in swarm plot students who have higher values of raising hands, visiting resources and viewing announcements take high level values mostly.
# * However there are some students who take low level although they have higher values of raising hands, visiting resources and viewing announcements
# 

# In[ ]:


sns.swarmplot(x='variable',y='value',hue='Class' , data=melt,palette={'H':'lime','M':'grey','L':'red'})
plt.ylabel('Values from zero to 100')
plt.title('High, middle and low level students')


# Well, lets look at why these have low level grade.

# In[ ]:


ave_raisedhands = sum(data['raisedhands'])/len(data['raisedhands'])
ave_VisITedResources = sum(data['VisITedResources'])/len(data['VisITedResources'])
ave_AnnouncementsView = sum(data['AnnouncementsView'])/len(data['AnnouncementsView'])
unsuccess = data.loc[(data['raisedhands'] >= ave_raisedhands) & (data['VisITedResources']>=ave_VisITedResources) & (data['AnnouncementsView']>=ave_AnnouncementsView)  & (data['Class'] == 'L')]


# In[ ]:


# All features of these two students
print(unsuccess)


# * In order to find why these two are in low level although they raisedhands, VisITedResources and AnnouncementsView
# * Lets first give numeric value to Class feature so as to compare features more precisely
# 

# In[ ]:


data['numeric_class'] = [1 if data.loc[i,'Class'] == 'L' else 2 if data.loc[i,'Class'] == 'M' else 3 for i in range(len(data))]


# In[ ]:


# Then start with gender: These two are boy so they can be low level due to it :) Girls say YEEESS but lets look
grade_male_ave = sum(data[data.gender == 'M'].numeric_class)/float(len(data[data.gender == 'M']))
grade_female_ave = sum(data[data.gender == 'F'].numeric_class)/float(len(data[data.gender == 'F']))


# * Average of female is higher than average of male. Therefore two only reason taking low grade can be being male. YEAAA  girls are more intelligent than boys.
# * Actually I am joking :) Gender comparison cannot completely explain why these two students takes low level grades.
# 

# In[ ]:


# Now lets look at nationality
nation = data.NationalITy.unique()
nation_grades_ave = [sum(data[data.NationalITy == i].numeric_class)/float(len(data[data.NationalITy == i])) for i in nation]
ax = sns.barplot(x=nation, y=nation_grades_ave)
jordan_ave = sum(data[data.NationalITy == 'Jordan'].numeric_class)/float(len(data[data.NationalITy == 'Jordan']))
print('Jordan average: '+str(jordan_ave))
plt.xticks(rotation=90)


# * As it can be seen in bar plot Jordan is seventh country with average 2.09.
# * Not so bad. Jordan has positive impact on these two students actually.
# 

# In[ ]:


# now lets look at topic : chemistry
lessons = data.Topic.unique()
lessons_grade_ave=[sum(data[data.Topic == i].numeric_class)/float(len(data[data.Topic == i])) for i in lessons]
ax = sns.barplot(x=lessons, y=lessons_grade_ave)
plt.title('Students Success on different topics')
chemistry_ave = sum(data[data.Topic == 'Chemistry'].numeric_class)/float(len(data[data.Topic == 'Chemistry']))
print('Chemistry average: '+ str(chemistry_ave))
plt.xticks(rotation=90)


# * Chemistry is not hardest lesson. Even it can be concluded that it is one of the easiest lessons with its 2.08 average
# * **Come on why these students take low level grades !!!**
# 

# In[ ]:


# Lets look at relation with family members
relation = data.Relation.unique()
relation_grade_ave = [sum(data[data.Relation == i].numeric_class)/float(len(data[data.Relation == i])) for i in relation]
ax = sns.barplot(x=relation, y=relation_grade_ave)
plt.title('Relation with father or mother affects success of students')


# * Having relation with mum has positive effect on these students
# * Students who have relation with their mum is more successful

# In[ ]:


#Lets look at how many times the student participate on discussion groups
discussion = data.Discussion
discussion_ave = sum(discussion)/len(discussion)
ax = sns.violinplot(y=discussion,split=True,inner='quart')
ax = sns.swarmplot(y=discussion,color='black')
ax = sns.swarmplot(y = unsuccess.Discussion, color='red')
plt.title('Discussion group participation')


# * These two students are under the average of discussion.
# * Average is 43. Therefore, participating discussion groups can be important success of these two students

# In[ ]:


# Now lastly lets look at
absence_day = data.StudentAbsenceDays.unique()
absense_day_ave = [sum(data[data.StudentAbsenceDays == i].numeric_class)/float(len(data[data.StudentAbsenceDays == i])) for i in absence_day]
ax = sns.barplot(x=absence_day, y=absense_day_ave)
plt.title('Absence effect on success')


# * Find one more reason why these have low level grade because their absence days are above seven.

# # CONCLUSION
# * Positive and negative effects on success of these two students can be seen.
# <a href="https://imgbb.com/"><img src="https://image.ibb.co/gjHDr5/as.jpg" alt="as" border="0"></a><br /><a target='_blank' href='https://tr.imgbb.com/'>upload image</a><br />

# # If you have any suggest, comment or question, I will be happy to hear it.
