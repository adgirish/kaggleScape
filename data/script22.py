
# coding: utf-8

# # Novice to Grandmaster- What Data Scientists say?

# ![](https://www.kaggle.com/static/images/host-home/host-home-recruiting.png?v=daa83a72816a)

# Kaggle is the world's largest Data Science platform with more than 1 million users, and it is an excellent platform for students like me to learn and grow in the field of Data Science and Machine Learning. It has users from various domains,like statisticians,Data Scientists and Machine Learning Practitioners.This dataset published by Kaggle is a gem for people like me, who like to analyse and investigate data. In this notebook, we will try to find some trending or some common questions, each budding data scientist would like to know, like the most used tools, the resources to learn data science ,etc. 
# 
# The biggest problem that we might face is fake and bogus responses. As it is a survey, not everyone will answer with proper credentials, and thus I assume that there will be a lot many outlier. Let's dive in straight into the pool of data and gain some insights..

# # Introduction 

# ### Who are Data Scientists?
# 
# A data scientist is a statistician or a programmer, who cleans, manages and organizes data, perform descriptive statistics and analysis to develop insights,build predictive models and solve business related problems. Let's see what do Data Scientists on kaggle say..

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import base64
import io
from scipy.misc import imread
import codecs
from IPython.display import HTML
from matplotlib_venn import venn2
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


response=pd.read_csv('../input/multipleChoiceResponses.csv',encoding='ISO-8859-1')


# In[ ]:


response.head()


# ## Some Basic Analysis

# In[ ]:


print('The total number of respondents:',response.shape[0])
print('Total number of Countries with respondents:',response['Country'].nunique())
print('Country with highest respondents:',response['Country'].value_counts().index[0],'with',response['Country'].value_counts().values[0],'respondents')
print('Youngest respondent:',response['Age'].min(),' and Oldest respondent:',response['Age'].max())


# Seriously?? Youngest Rspondent is not even a year old. LOL!! And how come grandpa is still coding at the age of 100. It may be a fake response.

# ## Gender Split

# In[ ]:


plt.subplots(figsize=(22,12))
sns.countplot(y=response['GenderSelect'],order=response['GenderSelect'].value_counts().index)
plt.show()


# The graph clearly shows that there are a lot more male respondents as compared to female. It seems that Ladies were either busy with their coding, **or ladies don't code**...:p. Just Kidding.

# ## Respondents By Country

# In[ ]:


resp_coun=response['Country'].value_counts()[:15].to_frame()
sns.barplot(resp_coun['Country'],resp_coun.index,palette='inferno')
plt.title('Top 15 Countries by number of respondents')
plt.xlabel('')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()
tree=response['Country'].value_counts().to_frame()
squarify.plot(sizes=tree['Country'].values,label=tree.index,color=sns.color_palette('RdYlGn_r',52))
plt.rcParams.update({'font.size':20})
fig=plt.gcf()
fig.set_size_inches(40,15)
plt.show()


# **USA and India**, constitute maximum respondents, about 1/3 of the total. Similarly Chile has the lowest number of respondents. Is this graph sufficient enough to say that majority of Kaggle Users are from India and USA. I don't think so, as the total users on Kaggle are more than 1 million while the number of respondents are only 16k.

# ## Compensation
# 
# Data Scientists are one of the most highest payed indviduals. Lets check what the surveyors say..

# In[ ]:


response['CompensationAmount']=response['CompensationAmount'].str.replace(',','')
response['CompensationAmount']=response['CompensationAmount'].str.replace('-','')
rates=pd.read_csv('../input/conversionRates.csv')
rates.drop('Unnamed: 0',axis=1,inplace=True)
salary=response[['CompensationAmount','CompensationCurrency','GenderSelect','Country','CurrentJobTitleSelect']].dropna()
salary=salary.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')
salary['Salary']=pd.to_numeric(salary['CompensationAmount'])*salary['exchangeRate']
print('Maximum Salary is USD $',salary['Salary'].dropna().astype(int).max())
print('Minimum Salary is USD $',salary['Salary'].dropna().astype(int).min())
print('Median Salary is USD $',salary['Salary'].dropna().astype(int).median())


# Look at that humungous Salary!! Thats **even larger than GDP of many countries**. Another example of bogus response. The minimum salary maybe a case of a student. The median salary shows that Data Scientist enjoy good salary benefits.

# In[ ]:


plt.subplots(figsize=(15,8))
salary=salary[salary['Salary']<1000000]
sns.distplot(salary['Salary'])
plt.title('Salary Distribution',size=15)
plt.show()


# ### Compensation by Country

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sal_coun=salary.groupby('Country')['Salary'].median().sort_values(ascending=False)[:15].to_frame()
sns.barplot('Salary',sal_coun.index,data=sal_coun,palette='RdYlGn',ax=ax[0])
ax[0].axvline(salary['Salary'].median(),linestyle='dashed')
ax[0].set_title('Highest Salary Paying Countries')
ax[0].set_xlabel('')
max_coun=salary.groupby('Country')['Salary'].median().to_frame()
max_coun=max_coun[max_coun.index.isin(resp_coun.index)]
max_coun.sort_values(by='Salary',ascending=True).plot.barh(width=0.8,ax=ax[1],color=sns.color_palette('RdYlGn'))
ax[1].axvline(salary['Salary'].median(),linestyle='dashed')
ax[1].set_title('Compensation of Top 15 Respondent Countries')
ax[1].set_xlabel('')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()


# The left graph shows the Top 15 high median salary paying countries. It is good to see that these countries provide salary more than the median salary of the complete dataset. Similarly,the right graph shows median salary of the Top 15 Countries by respondents. The most shocking graph is for **India**. India has the 2nd highest respondents, but still it has the lowest median salary in the graph. Individuals in USA have a salary almost 10 times more than their counterparts in India. What may be the reason?? Are IT professionals in India really underpaid?? We will check that later.

# ### Salary By Gender

# In[ ]:


plt.subplots(figsize=(10,8))
sns.boxplot(y='GenderSelect',x='Salary',data=salary)
plt.ylabel('')
plt.show()


# The salary for males look to be high as compared to others.

# ## Age

# In[ ]:


plt.subplots(figsize=(15,8))
response['Age'].hist(bins=50,edgecolor='black')
plt.xticks(list(range(0,80,5)))
plt.title('Age Distribution')
plt.show() 


# The respondents are young people with majority of them being in the age bracket if 25-35.

# ## Profession & Major

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,15))
sns.countplot(y=response['MajorSelect'],ax=ax[0],order=response['MajorSelect'].value_counts().index)
ax[0].set_title('Major')
ax[0].set_ylabel('')
sns.countplot(y=response['CurrentJobTitleSelect'],ax=ax[1],order=response['CurrentJobTitleSelect'].value_counts().index)
ax[1].set_title('Current Job')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()


# Data Science and Machine Learning is used in almost every industry. This is evident from the left graph,as people from different areas of interest like Physics, Biology, etc are taking it up for better understanding of the data. The right side graph shows the Current Job of the respondents. A major portion of the respondents are Dats Scientists. But as it is survey data, we know that there may be many ambigious responses. Later on we will check are these respondents real datas-scientists or self proclaimed data-scientists.

# ## Compensation By Job Title

# In[ ]:


sal_job=salary.groupby('CurrentJobTitleSelect')['Salary'].median().to_frame().sort_values(by='Salary',ascending=False)
ax=sns.barplot(sal_job.Salary,sal_job.index,palette=sns.color_palette('inferno',20))
plt.title('Compensation By Job Title',size=15)
for i, v in enumerate(sal_job.Salary): 
    ax.text(.5, i, v,fontsize=10,color='white',weight='bold')
fig=plt.gcf()
fig.set_size_inches(8,8)
plt.show()


# Operations Research Practitioner has the highest median salary followed by Predictive Modeler and Data Scientist. Computer Scientist and Programmers have the lowest compensation. 

# ## Machine Learning

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,12))
skills=response['MLSkillsSelect'].str.split(',')
skills_set=[]
for i in skills.dropna():
    skills_set.extend(i)
plt1=pd.Series(skills_set).value_counts().sort_values(ascending=False).to_frame()
sns.barplot(plt1[0],plt1.index,ax=ax[0],palette=sns.color_palette('inferno_r',15))
ax[0].set_title('ML Skills')
tech=response['MLTechniquesSelect'].str.split(',')
techniques=[]
for i in tech.dropna():
    techniques.extend(i)
plt1=pd.Series(techniques).value_counts().sort_values(ascending=False).to_frame()
sns.barplot(plt1[0],plt1.index,ax=ax[1],palette=sns.color_palette('inferno_r',15))
ax[1].set_title('ML Techniques used')
plt.subplots_adjust(wspace=0.8)
plt.show()


# It is evident that most of the respondents are working with Supervised Learning, and Logistic Regression being the favorite among them. There is no algorithm that is the best for all classification domains. A way to select one algorithm for a particular domain is by means of crossvalidation on the training data.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,12))
ml_nxt=response['MLMethodNextYearSelect'].str.split(',')
nxt_year=[]
for i in ml_nxt.dropna():
    nxt_year.extend(i)
pd.Series(nxt_year).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[0])
tool=response['MLToolNextYearSelect'].str.split(',')
tool_nxt=[]
for i in tool.dropna():
    tool_nxt.extend(i)
pd.Series(tool_nxt).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[1])
plt.subplots_adjust(wspace=0.8)
ax[0].set_title('ML Method Next Year')
ax[1].set_title('ML Tool Next Year')
plt.show()


# It is evident that the next year is going to see a jump in number of **Deep Learning** practitioners. Deep Learning and neural nets or in short AI is a favorite hot-topic for the next Year. Also in terms of Tools, Python is preferred more over R. Big Data Tools like Spark and Hadoop also have a good share in the coming years.

# ## Best Platforms to Learn

# In[ ]:


plt.subplots(figsize=(6,8))
learn=response['LearningPlatformSelect'].str.split(',')
platform=[]
for i in learn.dropna():
    platform.extend(i)
pd.Series(platform).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter',15))
plt.title('Best Platforms to Learn',size=15)
plt.show()


# My personal favorite Kaggle, is the most sought after source for learning Data Science, as it has notebooks from really experienced Data Scientists. The next choice is Online Courses i.e MOOC's. Platforms like coursera,udacity provide interactive videos and exercises for learning. Similary Youtube channels like Siraj Raval and others offer a free medium to study. These all medium are above than Textbooks. The reason maybe that textbooks often have limited content, or people are more fond of watching videos and learning.

# ## Hardware Used

# In[ ]:


plt.subplots(figsize=(10,10))
hard=response['HardwarePersonalProjectsSelect'].str.split(',')
hardware=[]
for i in hard.dropna():
    hardware.extend(i)
pd.Series(hardware).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',10))
plt.title('Machines Used')
plt.show()


# Since majority of the respondents fall in the age category below 25, which is where a majority of students fall under, thus a basic Laptop is the most commonly used machine for work.

# ## Where Do I get Datasets From??

# In[ ]:


plt.subplots(figsize=(15,15))
data=response['PublicDatasetsSelect'].str.split(',')
dataset=[]
for i in data.dropna():
    dataset.extend(i)
pd.Series(dataset).value_counts().plot.pie(autopct='%1.1f%%',colors=sns.color_palette('Paired',10),startangle=90,wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
plt.title('Dataset Source')
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.ylabel('')
plt.show()


# With hundreds of Dataset available, Kaggle is the most sought after source for datasets.

# ## Code Sharing

# In[ ]:


plt.subplots(figsize=(15,15))
code=response['WorkCodeSharing'].str.split(',')
code_share=[]
for i in code.dropna():
    code_share.extend(i)
pd.Series(code_share).value_counts().plot.pie(autopct='%1.1f%%',shadow=True,colors=sns.color_palette('Set3',10),startangle=90,wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
plt.title('Code Sharing Medium')
my_circle=plt.Circle( (0,0), 0.65, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.ylabel('')
plt.show()


# Github is the most used platform for Code and project sharing. The advantages of using github are:
# 
# 1)Version control your projects.
# 
# 2)Explore other’s projects on GitHub, get inspired code more, or contribute to their project.
# 
# 3)Collaborate with others, by letting other people contribute to your projects or you contributing to other’s projects, and many more.
# 
# ## Challenges in Data Science

# In[ ]:


plt.subplots(figsize=(15,18))
challenge=response['WorkChallengesSelect'].str.split(',')
challenges=[]
for i in challenge.dropna():
    challenges.extend(i)
plt1=pd.Series(challenges).value_counts().sort_values(ascending=False).to_frame()
sns.barplot(plt1[0],plt1.index,palette=sns.color_palette('inferno',25))
plt.title('Challenges in Data Science')
plt.show()


# The main challenge in Data Science is **getting the proper Data**. The graph clearly shows that dirty data is the bigget challenge. Now what is dirty data?? Dirty data is a database record that contains errors. Dirty data can be caused by a number of factors including duplicate records, incomplete or outdated data, and the improper parsing of record fields from disparate systems. Luckily Kaggle datasets are pretty clean and standardised.
# 
# Some other major challenges are the **Lack of Data Science and machine learning talent, difficulty in getting data and lack of tools**. Thats why Data Science is the sexiest job in 21st century.With the increasing amount of data, this demand will substantially grow.

# ## Job Satisfaction

# In[ ]:


satisfy=response.copy()
satisfy['JobSatisfaction'].replace({'10 - Highly Satisfied':'10','1 - Highly Dissatisfied':'1','I prefer not to share':np.NaN},inplace=True)
satisfy.dropna(subset=['JobSatisfaction'],inplace=True)
satisfy['JobSatisfaction']=satisfy['JobSatisfaction'].astype(int)
satisfy_job=satisfy.groupby(['CurrentJobTitleSelect'])['JobSatisfaction'].mean().sort_values(ascending=False).to_frame()
ax=sns.barplot(y=satisfy_job.index,x=satisfy_job.JobSatisfaction,palette=sns.color_palette('inferno',20))
fig=plt.gcf()
fig.set_size_inches(8,10)
for i, v in enumerate(satisfy_job.JobSatisfaction): 
    ax.text(.1, i, v,fontsize=10,color='white',weight='bold')
plt.title('Job Satisfaction out of 10')
plt.show()


# Data Scientists and Machine Learning engineers are the most satisfied people(who won't be happy with so much money), while Programmers have the lowest job satisfaction. But the thing to note here is that even if Computer Scientists have a lower salary than Programmers, still they have a good job satisfaction level than programmers. Thus Salary is not the only criteria or job satisfaction.
# 
# ## Job Satisfication By Country

# In[ ]:


satisfy=response.copy()
satisfy['JobSatisfaction'].replace({'10 - Highly Satisfied':'10','1 - Highly Dissatisfied':'1','I prefer not to share':np.NaN},inplace=True)
satisfy.dropna(subset=['JobSatisfaction'],inplace=True)
satisfy['JobSatisfaction']=satisfy['JobSatisfaction'].astype(int)
satisfy_job=satisfy.groupby(['Country'])['JobSatisfaction'].mean().sort_values(ascending=True).to_frame()
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'Viridis',
        reversescale = True,
        showscale = True,
        locations = satisfy_job.index,
        z = satisfy_job['JobSatisfaction'],
        locationmode = 'country names',
        text = satisfy_job['JobSatisfaction'],
        marker = dict(
            line = dict(color = 'rgb(200,200,200)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Satisfaction')
            )
       ]

layout = dict(
    title = 'Job Satisfaction By Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(0,0,255)',
        projection = dict(
        type = 'chloropleth',
            
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap2010')


# The average Job Satisfaction level is between **6-7.5** for most of the countries. It is lower in Japan(where people work for about 14 hours) and China. It is higher in come countries like Sweden and Mexico.
# 
# 
# 
# # Python vs R or (Batman vs Superman)
# 
# ![](https://blog.webhose.io/wp-content/uploads/2017/08/python-versus-R.png)'
# 
# Python and R are the most widely used Open-Source languages for Data Science and Machine-Learning stuff. For a budding data scientist or analyst, the biggest and trickiest doubt is: **Which Language Should I Start With??** While both the languages have their own advantages and shortcomings, it depends on the individual's purpose while selecting a language of his/her choice. Both the languages cater the needs of different kinds of work. Python is a general purpose langauge, thus web and application integration is easier, while R is meant for pure statistical and analytics purpose. The area where R will completely beat Python is visualisations with the help of packages like **ggplot2 and shiny**. But Python has an upperhand in Machine Learning stuff. So lets see what the surveyers say..

# In[ ]:


resp=response.dropna(subset=['WorkToolsSelect'])
resp=resp.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')
python=resp[(resp['WorkToolsSelect'].str.contains('Python'))&(~resp['WorkToolsSelect'].str.contains('R'))]
R=resp[(~resp['WorkToolsSelect'].str.contains('Python'))&(resp['WorkToolsSelect'].str.contains('R'))]
both=resp[(resp['WorkToolsSelect'].str.contains('Python'))&(resp['WorkToolsSelect'].str.contains('R'))]


# ### Recommended Language For Begineers

# In[ ]:


response['LanguageRecommendationSelect'].value_counts()[:2].plot.bar()
plt.show()


# Clearly Python is the recommended language for begineers. The reason for this maybe due to its simple english-like syntax and general purpose functionality.

# ## Recommendation By Python and R users

# In[ ]:


labels1=python['LanguageRecommendationSelect'].value_counts()[:5].index
sizes1=python['LanguageRecommendationSelect'].value_counts()[:5].values

labels2=R['LanguageRecommendationSelect'].value_counts()[:5].index
sizes2=R['LanguageRecommendationSelect'].value_counts()[:5].values


fig = {
  "data": [
    {
      "values": sizes1,
      "labels": labels1,
      "domain": {"x": [0, .48]},
      "name": "Language",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": sizes2 ,
      "labels": labels2,
      "text":"CO2",
      "textposition":"inside",
      "domain": {"x": [.54, 1]},
      "name": "Language",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Language Recommended By Python and R users",
        "annotations": [
            {
                "font": {
                    "size": 30
                },
                "showarrow": False,
                "text": "Python",
                "x": 0.17,
                "y": 0.5
            },
            {
                "font": {
                    "size": 30
                },
                "showarrow": False,
                "text": "R",
                "x": 0.79,
                "y": 0.5}]}}
py.iplot(fig, filename='donut')


# This is a interesting find. About **91.6%** Python users recommend Python as the first language for begineers, whereas only **67.2%** R users recommend R as the first language. Also **20.6%** R users recommend Python but only **1.68%** Python users recommend R as the first language. One thing to note is that users of both recommend the same Languages i.e SQL, Matlab and C/C++. I have only considered the Top 5 recommended languages, so the percentage will change if we consider all of them. But the difference would be just 2-3%.

# ### Necessary or Not??

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
response['JobSkillImportancePython'].value_counts().plot.pie(ax=ax[0],autopct='%1.1f%%',explode=[0.1,0,0],shadow=True,colors=['g','lightblue','r'])
ax[0].set_title('Python Necessity')
ax[0].set_ylabel('')
response['JobSkillImportanceR'].value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',explode=[0,0.1,0],shadow=True,colors=['lightblue','g','r'])
ax[1].set_title('R Necessity')
ax[1].set_ylabel('')
plt.show()


# Clearly Python is a much more necessary skill compared to R.
# 
# Special Thanks to [Steve Broll](https://www.kaggle.com/stevebroll) for helping in the color scheme.

# ### Number Of Users By Language

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
pd.Series([python.shape[0],R.shape[0],both.shape[0]],index=['Python','R','Both']).plot.bar(ax=ax[0])
ax[0].set_title('Number of Users')
venn2(subsets = (python.shape[0],R.shape[0],both.shape[0]), set_labels = ('Python Users', 'R Users'))
plt.title('Venn Diagram for Users')
plt.show()


# The number of Python users are definetely more than R users. This may be due to the easy learning curve of Python. However there are more users who know both the languages. These responses might be from established Data Scientists,as they tend to have a knowledge in multiple languages and tools.

# ## Compensation

# In[ ]:


py_sal=(pd.to_numeric(python['CompensationAmount'].dropna())*python['exchangeRate']).dropna()
py_sal=py_sal[py_sal<1000000]
R_sal=(pd.to_numeric(R['CompensationAmount'].dropna())*R['exchangeRate']).dropna()
R_sal=R_sal[R_sal<1000000]
both_sal=(pd.to_numeric(both['CompensationAmount'].dropna())*both['exchangeRate']).dropna()
both_sal=both_sal[both_sal<1000000]
trying=pd.DataFrame([py_sal,R_sal,both_sal])
trying=trying.transpose()
trying.columns=['Python','R','Both']
print('Median Salary For Individual using Python:',trying['Python'].median())
print('Median Salary For Individual using R:',trying['R'].median())
print('Median Salary For Individual knowing both languages:',trying['Both'].median())


# In[ ]:


trying.plot.box()
plt.title('Compensation By Language')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()


# Python coders have a slightly higher median salary as that compared to their R counterparts. However, the people who know both these languages, have a pretty high median salary as compared to both of them.
# 
# ## Language Used By Professionals

# In[ ]:


py1=python.copy()
r=R.copy()
py1['WorkToolsSelect']='Python'
r['WorkToolsSelect']='R'
r_vs_py=pd.concat([py1,r])
r_vs_py=r_vs_py.groupby(['CurrentJobTitleSelect','WorkToolsSelect'])['Age'].count().to_frame().reset_index()
r_vs_py.pivot('CurrentJobTitleSelect','WorkToolsSelect','Age').plot.barh(width=0.8)
fig=plt.gcf()
fig.set_size_inches(10,15)
plt.title('Job Title vs Language Used',size=15)
plt.show()


# As I had mentioned earlier, R beats Python in visuals. Thus people with Job-Titles like Data Analyst, Business Analyst where graphs and visuals play a very prominent role, prefer R over Python. Similarly almost 90% of statisticians use R. Also as stated earlier, Python is better in Machine Learning stuff, thus Machine Learning engineers, Data Scientists and others like DBA or Programmers prefer Python over R. 
# 
# Thus for data visuals--->R else---->Python.
# 
# **Note: This graph is not for Language Recommended by professionals, but the tools used by the professionals.**

# ## Job Function vs Language

# In[ ]:


r_vs_py=pd.concat([py1,r])
r_vs_py=r_vs_py.groupby(['JobFunctionSelect','WorkToolsSelect'])['Age'].count().to_frame().reset_index()
r_vs_py.pivot('JobFunctionSelect','WorkToolsSelect','Age').plot.barh(width=0.8)
fig=plt.gcf()
fig.set_size_inches(10,15)
plt.title('Job Description vs Language Used')
plt.show()


# As I had already mentioned ** R excels in analytics, but Python beats in Machine Learning.** The graph shows that R has influence when it comes to pure analytics, but other ways python wins.

# ## Tenure vs Language Used

# In[ ]:


r_vs_py=pd.concat([py1,r])
r_vs_py=r_vs_py.groupby(['Tenure','WorkToolsSelect'])['Age'].count().to_frame().reset_index()
r_vs_py.pivot('Tenure','WorkToolsSelect','Age').plot.barh(width=0.8)
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.title('Job Tenure vs Language Used')
plt.show()


# As we had seen earlier, Python is highly recommended for beginners. Thus the proportion of Python users is more in the initial years of coding. The gap between the languages however reduces over the years, as the coding experience increases.
# 
# ## Industry vs Language Used

# In[ ]:


r_vs_py=pd.concat([py1,r])
r_vs_py=r_vs_py.groupby(['EmployerIndustry','WorkToolsSelect'])['Age'].count().to_frame().reset_index()
r_vs_py.pivot('EmployerIndustry','WorkToolsSelect','Age').plot.barh(width=0.8)
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.title('Industry vs Language Used')
plt.show()


# R beats Python in Government, Insurance and Non-Profits industries. Similarly, Python beats R with a very big margin in Technology and Military industry. In remaining other industries, the share of Python looks to be roughly 15-20% more than that of R.
# 
# ## Common Tools with Python and R

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,15))
py_comp=python['WorkToolsSelect'].str.split(',')
py_comp1=[]
for i in py_comp:
    py_comp1.extend(i)
plt1=pd.Series(py_comp1).value_counts()[1:15].sort_values(ascending=False).to_frame()
sns.barplot(plt1[0],plt1.index,ax=ax[0],palette=sns.color_palette('inferno_r',15))
R_comp=R['WorkToolsSelect'].str.split(',')
R_comp1=[]
for i in R_comp:
    R_comp1.extend(i)
plt1=pd.Series(R_comp1).value_counts()[1:15].sort_values(ascending=False).to_frame()
sns.barplot(plt1[0],plt1.index,ax=ax[1],palette=sns.color_palette('inferno_r',15))
ax[0].set_title('Commonly Used Tools with Python')
ax[1].set_title('Commonly Used Tools with R')
plt.subplots_adjust(wspace=0.8)
plt.show()


# **SQL** seems to be the most common complementory tool used with both the languages. SQL  is the primary language for querying large databases, thus knowing it well is a very big plus.

# # Asking the Data Scientists
# 
# 
# 
# ![](https://ewebdesign.com/wp-content/uploads/2017/01/zarget_banner2.gif)
# 
# If I successfully write a Hello World program, then does that make me programmer or a developer?? If I beat my friends in a race, then does that make me the fastest person on the earth?? The answer is pretty obviously **NO.** This is the problem with the emerging Computer Science and IT folks. Based on their limited skills and experience, they start considering themselves much more than they really are. Many of them start calling them Machine Learning Practitioners even if they haven't really worked on real life projects and have just worked on some simple datasets. Similarly many responses here must be a bluff response. Lets check how many of them are the real Data Science practitioners.

# In[ ]:


response['DataScienceIdentitySelect'].value_counts()


# So about 26% of the total respondents consider themselves as Data Scientist. What does Sort of mean?? Are they still learning or are they unemployed. For now lets consider them as a No.
# 
# ## Current Job Titles

# In[ ]:


plt.subplots(figsize=(10,8))
scientist=response[response['DataScienceIdentitySelect']=='Yes']
scientist['CurrentJobTitleSelect'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',15))
plt.title('Job Titles',size=15)
plt.show()


# Surprisingly there is **no entry for the Job Title Data Scientist**. There reasons for this could be that the people with CurrentJobTitleSelect as Data Scientist(who might be working as Data Scientist) might have not answered the question: **"Do you currently consider yourself a Data Scientist?"**
# 
# There are many overlapping and common skills between the jobs like Data Analyst,Data Scientist and Machine Learning experts, Statisticians,etc. Thus they too have similar skills and consider themselves as Data Scientists even though they are not labeled the same. Now lets check if the previous assumption was True.

# In[ ]:


true=response[response['CurrentJobTitleSelect']=='Data Scientist']


# It was indeed **True**. People with their CurrentJobTitle as Data Scientist did not answer the question **"Do you currently consider yourself a Data Scientist?"**. So I am considering them also to be real Data Scientists.

# In[ ]:


scientist=pd.concat([scientist,true])
scientist['CurrentJobTitleSelect'].shape[0]


# So out of the total respondents, about **40%** of them are Data Scientists or have skills for the same.
# 
# ## Country-Wise Split

# In[ ]:


plt.subplots(figsize=(10,8))
coun=scientist['Country'].value_counts()[:15].sort_values(ascending=False).to_frame()
sns.barplot(coun.Country,coun.index,palette='inferno')
plt.title('Countries By Number Of Data Scientists',size=15)
plt.show()


# The graph is similar to the demographic graph where we had shown number of users by country. The difference is that the numbers have reduced as we have only considered Data Scientists.
# 
# ## Employment Status & Education

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,10))
sns.countplot(y=scientist['EmploymentStatus'],ax=ax[0])
ax[0].set_title('Employment Status')
ax[0].set_ylabel('')
sns.countplot(y=scientist['FormalEducation'],order=scientist['FormalEducation'].value_counts().index,ax=ax[1],palette=sns.color_palette('viridis_r',15))
ax[1].set_title('Formal Eduaction')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()


# About **67%** of the data scientists are employed full-time, while about **11-12%** of them are unemployed but looking for job. On the education side it is evident that about **45-46%** of the data scientists hold a **master's degree**, while about **23-24%** of them have a **bachelor's degree or a doctoral degree**. Thus education seems to be an important factor for becoming a data scientist. Let's see how does the salary vary according to the education.
# 
# ## Compensation By Formal Education
# 

# In[ ]:


plt.subplots(figsize=(25,12))
comp_edu=scientist.merge(salary,left_index=True,right_index=True,how='left')
comp_edu=comp_edu[['FormalEducation','Salary']]
sns.boxplot(x='FormalEducation',y='Salary',data=comp_edu)
plt.title('Compensation vs Education')
plt.xticks(rotation=90)
plt.show()


# This is surprising as the salary ranges for Bachelor's, Master's and Doctoral degree look to very similar. The median for Bachelor's degree seems to be a bit high as compared to Master's and doctoral degree. I didn't expect this as many of the Data Scientists hold a masters degree. But I think **Work Experience** matters more than any degree. Maybe the Bachelor's degree holders have more experience as compared than the other two.
# 
# ## Previous Job and Salary Change

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(30,15))
past=scientist['PastJobTitlesSelect'].str.split(',')
past_job=[]
for i in past.dropna():
    past_job.extend(i)
pd.Series(past_job).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('summer',25),ax=ax[0])
ax[0].set_title('Previous Job')
sal=scientist['SalaryChange'].str.split(',')
sal_change=[]
for i in sal.dropna():
    sal_change.extend(i)
pd.Series(sal_change).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('summer',10),ax=ax[1])
ax[1].set_title('Salary Change')
plt.subplots_adjust(wspace=0.9)
plt.show()


# Clearly majority of people switching to Data Science get a salary hike about **6-20% or more**. With this increasing demand for Data Scientists, the salary may also increase with time.
# 
# ## Tools used at Work

# In[ ]:


plt.subplots(figsize=(8,8))
tools=scientist['WorkToolsSelect'].str.split(',')
tools_work=[]
for i in tools.dropna():
    tools_work.extend(i)
pd.Series(tools_work).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('RdYlGn',15))
plt.show()


# Similar observations, Python, R and SQL are the most used tools or languages in Data Science
# 
# 

# ## Where Did they Learn From??

# In[ ]:


course=scientist['CoursePlatformSelect'].str.split(',')
course_plat=[]
for i in course.dropna():
    course_plat.extend(i)
course_plat=pd.Series(course_plat).value_counts()
blogs=scientist['BlogsPodcastsNewslettersSelect'].str.split(',')
blogs_fam=[]
for i in blogs.dropna():
    blogs_fam.extend(i)
blogs_fam=pd.Series(blogs_fam).value_counts()
labels1=course_plat.index
sizes1=course_plat.values

labels2=blogs_fam[:5].index
sizes2=blogs_fam[:5].values


fig = {
  "data": [
    {
      "values": sizes1,
      "labels": labels1,
      "domain": {"x": [0, .48]},
      "name": "MOOC",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": sizes2 ,
      "labels": labels2,
      "text":"CO2",
      "textposition":"inside",
      "domain": {"x": [.54, 1]},
      "name": "Blog",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Blogs and Online Platforms",
        "showlegend":True,
        "annotations": [
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "MOOC's",
                "x": 0.18,
                "y": 0.5
            },
            {
                "font": {
                    "size": 12
                },
                "showarrow": False,
                "text": "BLOGS",
                "x": 0.83,
                "y": 0.5}]}}
py.iplot(fig, filename='donut')


# Coursera is the most favoured platform by Data Scientists for learning Data Science. My personal vote also goes for Coursera, where you can learn things from scratch to advanced on the same platform. It is not limited to a single language like Python or R, but also has courses covering other languages like Scala,etc. Similarly KDNuggets is the most preferred blog.
# 
# ## Time Spent on Tasks
# 
# A Data Scientist is not always building predictive models, he is also responsible for the data quality, gathering the right data, analytics,etc. Lets see how much time a data scientist spends on these differnt tasks.

# In[ ]:


import itertools
plt.subplots(figsize=(22,10))
time_spent=['TimeFindingInsights','TimeVisualizing','TimeGatheringData','TimeModelBuilding']
length=len(time_spent)
for i,j in itertools.zip_longest(time_spent,range(length)):
    plt.subplot((length/2),2,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    scientist[i].hist(bins=10,edgecolor='black')
    plt.axvline(scientist[i].mean(),linestyle='dashed',color='r')
    plt.title(i,size=20)
    plt.xlabel('% Time')
plt.show()


# The read line is the mean line.
# Lets do it stepwise:
# 
#   - **TimeGatheringData:** It is undoubtedly the most time consuming part. Getting the data is the most painstaking task in the entire process, which is followed by Data Cleaning(not shown as data not available) which is yet other time consuming process. Thus gathering right data and scrubing the data are the most time consuming process.
#   
#   - **TimeVisualizing:** It is probably the least time consuming process(and probably the most enjoyable one..:p), and it reduces even further if we use Enterprise Tools like Tableau,Qlik,Tibco,etc, which helps in building graphs and dashboards with simple drag and drop features.
#   
#   - **TimeFindingInsights:** It is followed after visualising the data, which involves finding facts and patterns in the data, slicing and dicing it to find insights for business processes.It looks to a bit more time consuming as compared to TimeVisualizing.
#   
#   - **TimeModelBuilding:** It is where the data scientists build predictive models, tune these models,etc. It is the 2nd most time consuming process after TimeDataGathering.
# 
# 

# ## Cloud Services
# 
# With the increasing size of the data, it is not possible to process the data and perform predictive analytics on the physical server infrastructures. Cloud thus takes predictive analytics to a next level, with scalabilty its main advantage. They manage service that enables you to easily build machine learning models that work on any type of data, of any size. Lets check what are the most used cloud platforms by Data Scientists

# In[ ]:


cloud=['WorkToolsFrequencyAmazonML','WorkToolsFrequencyAWS','WorkToolsFrequencyCloudera','WorkToolsFrequencyHadoop','WorkToolsFrequencyAzure']
plt.subplots(figsize=(30,15))
length=len(cloud)
for i,j in itertools.zip_longest(cloud,range(length)):
    plt.subplot((length/2+1),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    sns.countplot(i,data=scientist)
    plt.title(i,size=20)
    plt.ylabel('')
    plt.xlabel('')
plt.show()


# It is evident that **AmazonAWS**, which is a public cloud service provider is the most used cloud platform, followed by Hadoop. Hadoop is an open-source software framework used for distributed storage and processing of dataset of big data. For reading more about Hadoop, **<a href='https://www.analyticsvidhya.com/blog/2014/05/hadoop-simplified/'>Check this</a>**

# ## Importance Of Visualisations

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,12))
sns.countplot(scientist['JobSkillImportanceVisualizations'],ax=ax[0])
ax[0].set_title('Job Importance For Visuals')
ax[0].set_xlabel('')
scientist['WorkDataVisualizations'].value_counts().plot.pie(autopct='%2.0f%%',colors=sns.color_palette('Paired',10),ax=ax[1])
ax[1].set_title('Use Of Visualisations in Projects')
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.ylabel('')
plt.show()


# Visualisations are a very integral part of Data Science Projects, and the above graph also shows the same. Almost all data science projects i.e **99%** of the projects have visualisations in them, doesn't matter how big or small. About **95%** of Data Scientists say that Visualisations skills are nice to have or necessary.Visuals help to understand and comprehend the data faster not only to the professionals but also to target customers, who may not be technically skilled.
# 
# ## BI Tools
# 
# Business intelligence software is a type of application software designed to retrieve, analyze, transform and report data for business intelligence. They make data visualisation and analytics very simple as compared to normal coding way in Python or R. The only drawback is that they are **proprietory and costly**. Lets check which are the most frequently used enterprise BI tools used by Data Scientists.

# In[ ]:


BI=['WorkToolsFrequencyQlik','WorkToolsFrequencySAPBusinessObjects','WorkToolsFrequencyTableau','WorkToolsFrequencyTIBCO','WorkToolsFrequencyAngoss','WorkToolsFrequencyIBMCognos','WorkToolsFrequencyKNIMECommercial','WorkToolsFrequencyExcel']
plt.subplots(figsize=(30,25))
length=len(BI)
for i,j in itertools.zip_longest(BI,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    sns.countplot(i,data=scientist)
    plt.title(i,size=20)
    plt.ylabel('')
    plt.xlabel('')
plt.show()


# I had read or heard somewhere that **Excel is the bread and butter for analysts**. This thing still somewhat holds true, as Excel is still popular among Data Scientists. However, the most frequently used BI tool looks to be **Tableau**. I personally use Tableau and it is pretty user friendly, drag and drop and you have your graphs ready.
# 
# ## Knowledge Of Algorithms (Maths and Stats)

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(25,12))
sns.countplot(y=scientist['AlgorithmUnderstandingLevel'],order=scientist['AlgorithmUnderstandingLevel'].value_counts().index,ax=ax[0],palette=sns.color_palette('summer',15))
sns.countplot(scientist['JobSkillImportanceStats'],ax=ax[1])
ax[0].set_title('Algorithm Understanding')
ax[0].set_ylabel('')
ax[1].set_title('Knowledge of Stats')
ax[1].set_xlabel('')
plt.show()


# Data Scientists have a good knowledge of mathematical concepts like Statistics and Linear Algebra, which are the most important part of Machine Learning algorithms. But is this maths really required, as many standard libraries like scikit,tensorflow,keras etc have all these things already implemented. But the experienced data scientists say that we should have a good understanding of the maths behind the algorithms. About **95%** of the data scientists say the stats is an important asset in Data Science.
# 
# ## Learning Platform Usefullness

# In[ ]:


plt.subplots(figsize=(25,35))
useful=['LearningPlatformUsefulnessBlogs','LearningPlatformUsefulnessCollege','LearningPlatformUsefulnessCompany','LearningPlatformUsefulnessKaggle','LearningPlatformUsefulnessCourses','LearningPlatformUsefulnessProjects','LearningPlatformUsefulnessTextbook','LearningPlatformUsefulnessYouTube']
length=len(useful)
for i,j in itertools.zip_longest(useful,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.2)
    scientist[i].value_counts().plot.pie(autopct='%1.1f%%',colors=['g','lightblue','r'],wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
    plt.title(i,size=25)
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.xlabel('')
    plt.ylabel('')
plt.show()


# The above donut charts shows the opinion of Data Scientists about the various platforms to learn Data Science. The plot looks best for **Projects**,where the percentage for not useful is almost **0%**.According to my personal opinion too, projects are the best platform or way for learning anything in the IT industry. The other excellent platforms are **Online Courses and Kaggle**. The graphs for other platforms are quite similar to each other.
# 
# ## What should the Resume have??

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(22,8))
sns.countplot(y=scientist['ProveKnowledgeSelect'],order=scientist['ProveKnowledgeSelect'].value_counts().index,ax=ax[0],palette=sns.color_palette('inferno',15))
ax[0].set_title('How to prove my knowledge')
sns.countplot(scientist['JobSkillImportanceKaggleRanking'],ax=ax[1])
ax[1].set_title('Kaggle Rank')
plt.show()


# It is evident that Work experience in ML projects and Kaggle competitions reflects the knowledge of Data Science. Also a kaggle rank can be a good thing in one's resume. As I had mentioned earlier that relevant work experience might have a higher value as compared to any Master's or Doctoral degree. This statement thus holds good, as Data Scientists prefer work experience over degree, as seen in the above graph
# 
# 
# ## How did they search for Jobs?? 

# In[ ]:


plt.subplots(figsize=(10,8))
scientist.groupby(['EmployerSearchMethod'])['Age'].count().sort_values(ascending=True).plot.barh(width=0.8,color=sns.color_palette('winter',10))
plt.title('Job Search Method',size=15)
plt.ylabel('')
plt.show()


# Many Data Scientists get to know about the jobs through their friends or relatives or they were contacted by directly by the company. Thus we should properly maintain our professional profiles like Linkedin and keep on updating it, as such networking sites could help you get your dream job.

# ## Checking the Free Responses
# 
# This file contains the free form responses answered by the respondents. The problem with this one is that being a free form response, every user will answer in their own way. What I mean is we will have different answers for the same thing. An example of this that I observed is the library **Pandas is written as pandas, Pandas, panda and in many such differnt forms.** Thus I will try to analyse this file using **nltk(Natural Language Toolkit).**

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
free=pd.read_csv('../input/freeformResponses.csv')
stop_words=set(stopwords.words('english'))
stop_words.update(',',';','!','?','.','(',')','$','#','+',':','...')


# ## Motivation Behind Working on Kaggle

# In[ ]:


kaggle=b'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fxO6RtrnxF0mFj99LC2lfH0LAZ/KvVvCf/BFL4e2KB/EHjbX9Umz/q7VYoI8f98lv1FFwPx1or93/C//AASc/Z68PyiS88O6hrzgYAvtUnC59cRuoP8AKvU/D/7DfwD8MlDafCjwxMU5U32npdn/AMihs0XA/nUisbmf/V28sn+4hNdp4c+BHxE8XrG2jeC9a1FZMFDDZuQ304r+kTQPhd4N8Kbf7E8JaFo+3p9g02GDH/fCiumHFK4H8+PhP/gnH+0H4tiWWL4fXmmxMeG1N1g/QnP6V694S/4I2/GfW4hLrGreG9AQnGx7mSeTHrhUA/Wv2woouB+VnhH/AIIgfOJPE/xTJTvb6XpGCf8Ato8p/wDQa9x8Gf8ABH74GeGQj6lP4j8STZywv72JIz7BY4lIH4mvuKikB5P4C/ZP+EPwyFv/AMI78P8ARbGSADZM1v5smR/FufJz716pBaw2ibIIkhT+7GoUfpUtJQAUtJR2oAWkoooAKKWigApKWkoAWkoooAKKWigBKKKWgBKKKKAFpKKKACiiigBaSiigBaSiloAKSlpKAFopKWgAooooApa3H52jX8f963kX81NfzC/EGLyPHviWLps1O5X8pWr+oSWMTRPGejKVP41/Mt8d9Ek8N/Gzx9pcxBktNevoiV6HE74I/CmgOFoooqgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK9w/Y3/Z6uv2lvjronhNS0WmIftmpTqOY7ZCN2PckgD614fX68/wDBFb4SDSfAXjX4iXMOJ9Wu49KtGYcrHCu+Qj2ZpU/749qTA/R7QNDsfDGh6do+mQLa6dp9vHa20CfdjiRQqKPoABV4jIooqQGGGM9Y1P4Cmm0gPWGM/wDABU1JQBGLaFekSD6KKeEVegA+gpaKACiiloASlopKACiiigAooooAKWikoAKWkooAKKKKAFoopKACiiloAKSiigAooooAKWikoAKKKKACilooASlopKAClpKWgBKKKWgAopKKACiiigBaKSigBaSlpKAFpKKKAFooooASv5zf24tK/sX9rb4o2uNv/E4eXH++qv8A+zV/RlX4If8ABU/wwPDf7aXjGZIzHFqtvZX6D1zbpGxH1aNqaA+SaKKKoAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACv6Cf+CbXhQ+Ff2OfAQdPLk1GB9QZf8Afc4P4qqn8a/n2r+kf9kKy/s79mH4Z22MeXoduMf8BqWB69RRRSAWkoooAKKWkoAKKKKACiiloASiiloASiiigAooooAKWkooAKKKKACiiigAoopaAEpaSigBaKSigAooooAKKKKACiiigAooooAKKKKACiiigBaSiigAoopaAEooooAKKKKACiiloAKKTFFABX43/wDBazwkbL46eDfEartjvvD6WbED7zxXE7E/98yqPwr9kK/Ob/gtR4HGq/B/wZ4oSHMmkanJbvKM8LMq4B/FKAPxzoooqwCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK/pn+AOnNpXwR8DWjLsaLR7YEen7sGv5wvhR4Ybxr8UfB/h5IjM2q6xZ2PlqMlvMmRMf+PV/Tnp1olhp9tbRoESGJY1VRgAAAAAfhUsCxRRRSAKKKKACiiigAooooAKKKKACiiigAoopaACkopaAEooooAWkooxQAUUUUAFLRSUAFFFFAC0lFFABRRRQAUUUUALSUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUtFACV4L+3Z8PJPib+yj8RNIgRZLqLTZL+BG/ikgHmqB7nbge9e9VW1PToNX066sblBJb3MTQyIf4lYYI/I0Afyx0V1/wAXfh9d/Cn4n+KPCF6rLcaNqM9kS4+8EcqG+hAz+NchVgFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRUlvbyXc8cMSGSWRgqqo5JPQUAfZv8AwSk+CV18TP2ndK8Sy2/maH4RV9QuJG6GYoyQKPcOyt9Fr91K+ZP+Cfn7MKfs0fAnTrW/h2+K9bRNQ1ZmGGjkZcrD/wAAB2n3Br6bqACiiigAopaSgAopaSgAooooAKKKKACilpKACiiigAooooAKKKKACiiigAooooAWkoooAKKOKKACiiigAooooAKKKKAFpKKKAFpKKWgApKKKAFpKKKACiiigBaSiigAooooAKKKKAFpKKWgBMUUZooAKKKKAPxi/4LHfBo+EPjVo/juztfLsPEtr5c8in5ftMQAOR2LKc++DX571+/X/AAUs+BDfHD9l/XPsMJl1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'


# In[ ]:


motivation=free['KaggleMotivationFreeForm'].dropna().apply(nltk.word_tokenize)
motivate=[]
for i in motivation:
    motivate.extend(i)
motivate=pd.Series(motivate)
motivate=([i for i in motivate.str.lower() if i not in stop_words])
f1=open("kaggle.png", "wb")
f1.write(codecs.decode(kaggle,'base64'))
f1.close()
img1 = imread("kaggle.png")
hcmask1 = img1
wc = WordCloud(background_color="black", max_words=4000, mask=hcmask1, 
               stopwords=STOPWORDS, max_font_size= 60,width=1000,height=1000)
wc.generate(" ".join(motivate))
plt.imshow(wc)
plt.axis('off')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()


# ### Lets flex our data science strength...:p
# 
# The wordcloud shows the motivation of users for working on kaggle. Clearly Learning Data Science, Machine Learning, interest in the same,curiosity, fun and datasets retrieval are some of the relevant ones.
# 
# ## Most Frequently Used Libraries

# In[ ]:


library=free['WorkLibrariesFreeForm'].dropna().apply(nltk.word_tokenize)
lib=[]
for i in library:
    lib.extend(i)
lib=pd.Series(lib)
lib=([i for i in lib.str.lower() if i not in stop_words])
lib=pd.Series(lib)
lib=lib.value_counts().reset_index()
lib.loc[lib['index'].str.contains('Pandas|pandas|panda'),'index']='Pandas'
lib.loc[lib['index'].str.contains('Tensorflow|tensorflow|tf|tensor'),'index']='Tensorflow'
lib.loc[lib['index'].str.contains('Scikit|scikit|sklearn'),'index']='Sklearn'
lib=lib.groupby('index')[0].sum().sort_values(ascending=False).to_frame()
R_packages=['dplyr','tidyr','ggplot2','caret','randomforest','shiny','R markdown','ggmap','leaflet','ggvis','stringr','tidyverse','plotly']
Py_packages=['Pandas','Tensorflow','Sklearn','matplotlib','numpy','scipy','seaborn','keras','xgboost','nltk','plotly']
f,ax=plt.subplots(1,2,figsize=(18,10))
lib[lib.index.isin(Py_packages)].sort_values(by=0,ascending=True).plot.barh(ax=ax[0],width=0.9,color=sns.color_palette('viridis',15))
ax[0].set_title('Most Frequently Used Py Libraries')
lib[lib.index.isin(R_packages)].sort_values(by=0,ascending=True).plot.barh(ax=ax[1],width=0.9,color=sns.color_palette('viridis',15))
ax[1].set_title('Most Frequently Used R Libraries')
ax[1].set_ylabel('')
plt.show()


# Some brief information about the libraries:
# 
# ### Python:
# 
# 1) **Sklearn**- For Machine Learning algorithms. This library has almost all the important machine learning algorithms used for industries.
# 
# 2) **Pandas, Matlotlib & Seaborn-** Mostly used together for analytics and visualisation work.
# 
# 3) **TensorFlow and Keras-** Used for Deep Learning.
# 
# 4) **Numpy and Scipy-** Used for scientific computations.
# 
# 5) **nltk-** Used for Natural Language Processing.
# 
# ### R:
# 
# 1) **dplyr-** dplyr is the package for fast data manipulation.
# 
# 2) **ggplot2 and shiny-** R's famous package for making beautiful graphics. Python visuals are nowhere near to the quality of visuals created using this library.
# 
# 3) **caret and randomforest-** For Machine Learning purpose.
# 
# 4) **tidyr-** Tools for changing the layout of your data sets.
# 
# 5) **stringr-** Easy to learn tools for regular expressions and character strings.
# 
# ** Leaflet (folium in Python) and Plotly are common libraries in both languages, and are used to create interactive plots like geomaps, etc.**

# # Conclusions
# Some brief insights that we gathered from the notebook:
# 
# 1) Majority of the respondents are from USA followed by India. USA also had the maximum number of data scientists followed by India. Also the median Salary is highest in USA.
# 
# 2) Majority of the respondents are in the age bracket 20-35, which shows that data science is quite famous in the youngsters.
# 
# 3) The respondents are not just limited to Computer Science major, but also from majors like Statistics, health sciences,etc showing that Data Science is an interdisciplinary domain.
# 
# 4) Majority of the respondents are fully employed.
# 
# 5) Kaggle, Online Courses(Coursera,eDx,etc), Projects and Blogs(KDNuggets,AnalyticsVidya,etc) are the top resources/platforms for learning Data Science.
# 
# 6) Kaggle has the highest share for data acquisition whereas Github has the highest share for code sharing.
# 
# 7) Data Scientists have the highest Job Satisfaction level and the second highest median salary (after Operations Research Analyst). On the contrary, Programmers have the least Job Satisfaction level and one of the least median salary also.
# 
# 8) Data Scientists also get a hike of about 6-20% from their previous jobs.
# 
# #### Tips For Budding Data Scientists
# 
# 1) Learn **Python,R and SQL** as they are the most used languages by the Data Scientists. Python and R will help in analytics and predictive modeling while SQL is best for querying the databases.
# 
# 2) Learn Machine Learning Techniques like **Logistic Regression, Decision Trees, Support Vector Machines**, etc as they are most commonly used Machine Learning techniques/algorithms.
# 
# 3) **Deep Learning and Neural Nets** will be the most sought after techniques in the future, thus a good knowledge in them will be very helpful.
# 
# 4) Develop skills for **Gathering Data** and **Cleaning The Data** as they are the most time consuming processes in the workflow of a data scientist. 
# 
# 5) **Visualisations** are very important in Data Science projects and almost all projects require Visualisations for understanding the data better. So one should learn Data Visualisation as Data Scientists consider it to be a **necessary or nice to have skill.**
# 
# 6) **Maths and Stats** are very important in Data Science, so we should have good understanding of it for actually understanding how the algorithm works.
# 
# 7) **Projects** are the best way to learn Data Science according to Data Scientists.So working on projects will help you learn data science better.
# 
# 8) **Experience with ML Projects in company and Kaggle Competitions** are the best ways to show your working knowledge in Data Science. Working on ML projects in a company gives the experience of working with real world datasets, thereby enhancing the knowledge. Kaggle competitions are also a great medium, as you will be competing with Data Scientists over the world. Also a **Kaggle Rank** can be a good USP in the resume.
# 

# So I would like my conclude my analysis here. Thanks a lot for having a look at this notebook. I have linked another notebook which deals with survey data. Do have a look at it.
# 
# Link: [HackerRank Survey](https://www.kaggle.com/ash316/coders-not-hackers-hackerrank)
# 
# I Hope all you liked the notebook. Any suggestions and feedback are always welcome.
# 
# ### Please Upvote this notebook as it encourages me in doing better.
# 
# 
# ![](http://68.media.tumblr.com/e1aed171ded2bd78cc8dc0e73b594eaf/tumblr_o17frv0cdu1u9u459o1_500.gif)
