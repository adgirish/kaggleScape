
# coding: utf-8

# > **The Making of Data Scientist** <br> 
# We have people taking this survey from all across the world from different background and we need to understand their approach, route they took  to become Data scientist.There are many questions to be answered and Kaggle has done that through survey. Hopefully I would be able to visualize respondents response through graphs and guide the interested people.
# ![](https://www.everywishes.com/wp-content/uploads/2017/08/a39693e75542272dcb313f86805dcf6b.jpg)
# <br>

# In[ ]:


import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
plt.style.use('bmh')
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
from plotly.graph_objs import *
from mpl_toolkits.mplot3d import axes3d
import codecs
import matplotlib.image as mpimg
import plotly.tools as tls
from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imread
import base64
from os import path
from PIL import Image


# In[ ]:


conrate = pd.read_csv("../input/conversionRates.csv",encoding="ISO-8859-1", low_memory=False)
response = pd.read_csv("../input//freeformResponses.csv",encoding="ISO-8859-1", low_memory=False)
multch = pd.read_csv("../input//multipleChoiceResponses.csv", encoding="ISO-8859-1", low_memory=False)
schema = pd.read_csv("../input//schema.csv", encoding="ISO-8859-1", low_memory=False)
#RespondentTypeREADME.txt


# >  **Lets start exploration of the tools/libraries/learning platform used by aspiring data scientist.**
# 
#  <img src="https://www.amazing-animations.com/animations/computers5.gif" alt="Alt text that describes the graphic" title="superman" />
# 

# In[ ]:


f, ax = plt.subplots(figsize=(15, 7)) 
response['WorkLibrariesFreeForm'] = response['WorkLibrariesFreeForm'].replace(['Scikit-learn', 'scikit-learn','scikit learn'], 'sklearn')
response['WorkLibrariesFreeForm'] = response['WorkLibrariesFreeForm'].replace(['tensorflow'], 'Tensorflow')
response['WorkLibrariesFreeForm'] = response['WorkLibrariesFreeForm'].replace(['R libraries'], 'R')
response['WorkLibrariesFreeForm'] = response['WorkLibrariesFreeForm'].replace(['pandas'], 'Pandas')
response['WorkLibrariesFreeForm'] = response['WorkLibrariesFreeForm'].replace(['none'], 'None')
response['WorkLibrariesFreeForm'] = response['WorkLibrariesFreeForm'].replace(['keras'], 'Keras')
g = sns.barplot( y = response['WorkLibrariesFreeForm'].value_counts().head(10).index,
            x = response['WorkLibrariesFreeForm'].value_counts().head(10).values,
                palette="GnBu_d")
plt.title("Top 10 Work libraries used by data scientist")
plt.show()
#response.shape


# In[ ]:


''' 
use below code to check all the columns present in the dataset (we have more that 200 columns in this datsete)

columns = multch.columns
for x in columns : 
  print(x) 
''';


# 
# <center> Who are they  </center>
#                -----------------
# 
# ![](http://akns-images.eonline.com/eol_images/Entire_Site/201256/reg_600.justiceleague.mh.060612.jpg?downsize=600:*&crop=600:300;left,top) 
# <br>
# 
# 
# 

# **Around 82% respondents are male and 17 % are female. I am sure the number of female data scientist will increase in future**

# In[ ]:


df = pd.DataFrame(multch['GenderSelect'].value_counts().values,
                  index=multch['GenderSelect'].value_counts().index, 
                  columns=[' '])

df.plot(kind='pie', subplots=True, autopct='%1.0f%%', figsize=(8, 8))
#plt.subplots_adjust(wspace=0.5)
plt.show()


# **People who selected answers from multiple choice option with their current occupation and their college major**

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,11))
sns.barplot( y = multch['MajorSelect'].dropna().value_counts().head(20).index,
            x = multch['MajorSelect'].dropna().value_counts().head(20).values,
                palette="winter",ax=ax[0])
ax[0].set_title('Major')
ax[0].set_yticklabels(multch['MajorSelect'].dropna().value_counts().head(20).index, 
                      rotation='horizontal', fontsize='large')
ax[0].set_ylabel('')
sns.barplot( y = multch['CurrentJobTitleSelect'].dropna().value_counts().head(20).index,
            x = multch['CurrentJobTitleSelect'].dropna().value_counts().head(20).values,
                palette="summer",ax=ax[1])
ax[1].set_title('Current Job')
ax[1].set_yticklabels(multch['CurrentJobTitleSelect'].dropna().value_counts().head(20).index, 
                      rotation='horizontal', fontsize='large')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()


# 
# **People who didn't select answers from multiple choice option and had filled their responses with their current occupation and their college major**
# 

# In[ ]:


response['MajorFreeForm'] = response['MajorFreeForm'].replace(['Chemistry ', 'chemistry'], 'Chemistry')
response['MajorFreeForm'] = response['MajorFreeForm'].replace(['economics','Economics '], 'Economics')
response['MajorFreeForm'] = response['MajorFreeForm'].replace(['Electronics and communication engineering','Electronics and Communication','Electronics and communication'], 'Electronics and Communication Engineering')
f,ax=plt.subplots(1,2,figsize=(20,20))
sns.barplot( y = response['MajorFreeForm'].dropna().value_counts().head(20).index,
            x = response['MajorFreeForm'].dropna().value_counts().head(20).values,
                palette="winter",ax=ax[0])
ax[0].set_title('Major')
ax[0].set_yticklabels(response['MajorFreeForm'].dropna().value_counts().head(20).index, 
                      rotation='horizontal', fontsize='large')
ax[0].set_ylabel('')
sns.barplot( y = response['CurrentJobTitleFreeForm'].dropna().value_counts().head(20).index,
            x = response['CurrentJobTitleFreeForm'].dropna().value_counts().head(20).values,
                palette="summer",ax=ax[1])
ax[1].set_title('Current Job')
ax[1].set_yticklabels(response['CurrentJobTitleFreeForm'].dropna().value_counts().head(20).index, 
                      rotation='horizontal', fontsize='large')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.5)
plt.show()


# **Data science and machine learning is rapidly expanding and people from different educational background has shown willingness to enter into analytics field which is evident from above response.**

# In[ ]:


response['LearningPlatformFreeForm1'] = response['LearningPlatformFreeForm1'].replace(['coursera'], 'Coursera')
response['LearningPlatformFreeForm1'] = response['LearningPlatformFreeForm1'].replace(['Datacamp', 'Data Camp'], 'DataCamp')
response['LearningPlatformFreeForm3'] = response['LearningPlatformFreeForm3'].replace(['Datacamp', 'Data Camp','datacamp'], 'DataCamp')
response['LearningPlatformFreeForm2'] = response['LearningPlatformFreeForm2'].replace(['analytics vidhya', 'Analyticsvidhya'], 'Analytics Vidhya')


# In[ ]:


#f, ax = plt.subplots(figsize=(15, 7)) 
f,ax=plt.subplots(1,2,figsize=(15,7))
learningPlatform =  pd.concat(objs=[response['LearningPlatformFreeForm1'], 
                                    response['LearningPlatformFreeForm2'],
                                    response['LearningPlatformFreeForm3']],
                     axis=0).reset_index(drop=True)

learn=learningPlatform.str.split(',')
platform=[]
for i in learn.dropna():
    platform.extend(i)
pd.Series(platform).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.5,color=sns.color_palette('winter',15),ax=ax[0])
ax[0].set_title('Best Platforms to Learn(Free Form response)',size=15)

learn=multch['LearningPlatformSelect'].str.split(',')
platform=[]
for i in learn.dropna():
    platform.extend(i)
pd.Series(platform).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.5,color=sns.color_palette('winter',15),ax=ax[1])
ax[1].set_title('Best Platforms to Learn(Multiple choice response)',size=15)
plt.subplots_adjust(wspace=0.5)
plt.show()


# Check out the favourite Learning platforms of the respondents.   <br>
#       ---
# Interestingly we got list of online platforms from Free Response which hosts massive open online course(mooc). Courses offered by them  are well organized and they help create a customized plan for you to achieve based on your interests (Best example : COURSERA) <br><br>
# 
# **Kaggle:- **If you want to learn from the best then Kaggle is the best platform. The Key takeaways is Tips and Tricks shared by the experts through their kernels to solve real world problems. <br>
# **Online course :- **Refer first para on Free Response. <br>
# **Stack Overflow :-** If you are stuck somewhere in generating desired output or struggling with unseen errors. This is the place to get the help. <br>
# **Youtube videos :-** Life is easy when get uninterrupted playlists of a course. almost all mooc's are uploaded on youtube by people.
# 

# Next <br>
# Respondents were asked to answer below question with percentage.<br>
# What percentage of your current machine learning / data science training falls under each category? <br>
#      -----
# **(Total must equal 100%)**  <br><br>
# 
# 
# SelfTaught**(20%)** + Online Courses**(20%)** + Work**(10%)** + University**(10%) **+ Kaggle**(30%)** + Other**(10%)** = **100%** <br>
# for e.g. in above case if one of the respondent spends overall 10 hours learning then out of those 10 hours Kaggle contributes around 3 hours <br>
# 
# Similarly lets visualize percent spread for these categories for all Respondents. <br>
# 1. Boxplot <br>
# 2. Distplot (using Pairplot) <br>
# 

# In[ ]:


v_features = multch.iloc[:,59:65].columns
plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(1,6)
for i, cn in enumerate(multch[v_features]):
    ax = plt.subplot(gs[i])
    sns.boxplot(y = cn , data = multch)
    ax.set_title(str(cn)[16:])
    ax.set_ylabel(' ')


# In[ ]:


df = multch[['LearningCategorySelftTaught', 'LearningCategoryOnlineCourses',
       'LearningCategoryWork', 'LearningCategoryUniversity',
       'LearningCategoryKaggle', 'LearningCategoryOther']]

df = df.rename(columns=lambda x: x.replace('LearningCategory', ''))
sns.pairplot(df.dropna());


# **We can see that most of the respondents prefer to train themselves through self learning and Online courses.**

# Now lets plot answers of few important questions. 
#     --
# **On average, how many hours a week are being spent by the learners to study data science?** <br>
# (this question was asked to learners )<br><br>
# 
# **How long have you been writing code to analyze data?** <br>
# (this question was asked to everyone)

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,6))
sns.barplot( y = multch['TimeSpentStudying'].dropna().value_counts().index,
            x = multch['TimeSpentStudying'].dropna().value_counts().values,
                palette="winter",ax=ax[0])
ax[0].set_title('Time Spent Studying by Learners')
ax[0].set_yticklabels(multch['TimeSpentStudying'].dropna().value_counts().head(20).index, 
                      rotation='horizontal', fontsize='large')
ax[0].set_ylabel('')
sns.barplot( y = multch['Tenure'].dropna().value_counts().index,
            x = multch['Tenure'].dropna().value_counts().values,
                palette="summer",ax=ax[1])
ax[1].set_title('Experience as Data analyst')
ax[1].set_yticklabels(multch['Tenure'].dropna().value_counts().index, 
                      rotation='horizontal', fontsize='large')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.5)
plt.show()


# **Lets take a look at the employment status of the respondents.** 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(12,7))
g = sns.barplot( y = multch['EmploymentStatus'].value_counts().head(10).index,
            x = multch['EmploymentStatus'].value_counts().head(10).values,
                palette="GnBu_d",ax=ax[0])
ax[0].set_title("Employment status of the respondents")

labels = multch['EmploymentStatus'].value_counts().head(10).index
df = pd.DataFrame(multch['EmploymentStatus'].value_counts().head(10).values,
                  index=labels, 
                  columns=[' '])
ax[1].set_title("Employment status of the respondents")

pie = df.plot(kind='pie',subplots=True,autopct='%1.0f%%', ax=ax[1])
ax[1].legend(bbox_to_anchor=(0.8, 0.7))
plt.show()


# **Majority of the respondents are employed full time. ** <br>
# > Does that mean they all use data science and machine learning skills in their day to day job ? We need to answer this question by exploring other columns of this dataset.

# **Lets check which industry they belong to and the size of their Employer.** 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,8))
sns.barplot( y = multch['EmployerIndustry'].dropna().value_counts().index,
            x = multch['EmployerIndustry'].dropna().value_counts().values,
                palette="winter",ax=ax[0])
ax[0].set_title('Employer Industry')
ax[0].set_yticklabels(multch['EmployerIndustry'].dropna().value_counts().head(20).index, 
                      rotation='horizontal', fontsize='large')
ax[0].set_ylabel('')
sns.barplot( y = multch['EmployerSize'].dropna().value_counts().index,
            x = multch['EmployerSize'].dropna().value_counts().values,
                palette="summer",ax=ax[1])
ax[1].set_title('Employer Size')
ax[1].set_yticklabels(multch['EmployerSize'].dropna().value_counts().index, 
                      rotation='horizontal', fontsize='large')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.5)
plt.show()


# **No wonder majority of them come from Technology industry and from large Industries.** 

# How many years has your organization been utilizing advanced analytics/data science?
#       ---

# In[ ]:


multch['EmployerMLTime'].value_counts()

f, ax = plt.subplots(figsize=(15, 7)) 
g = sns.barplot( y = multch['EmployerMLTime'].value_counts().index,
            x = multch['EmployerMLTime'].value_counts().values,
                palette="GnBu_d")
plt.title("Advanced Analytics usage in the Respondents Organization");


# In[ ]:


groups_edu = multch.groupby(['EmployerIndustry', 'EmployerMLTime'])
cl = groups_edu.size().unstack()
f, ax = plt.subplots(figsize=(12, 20)) 
g = sns.heatmap(cl,cmap='coolwarm',linecolor='white',linewidths=3)
plt.show()


# **Many organizations lately started embracing the power of Advanced analytics to stay relevant **

# How has the size of your organization's ML/DS staff changed over the past year?
#       ---
#       
#  Check out the below Heatmap for the answer
#       

# In[ ]:


groups_edu = multch.groupby(['EmployerIndustry', 'EmployerSizeChange'])
cl = groups_edu.size().unstack()
f, ax = plt.subplots(figsize=(12, 20)) 
g = sns.heatmap(cl,cmap='coolwarm',linecolor='white',linewidths=3)
plt.show()


# Reiterating the conclusion from last graph again <br>
# **Many organizations lately started embracing the power of Advanced analytics to stay relevant **  <br>
#  > **As a result many Employers from industries like Technology, Academic and Financial have seen active hiring to build the team of Data scientists.**
# 

# In[ ]:


f, ax = plt.subplots(figsize=(15, 7)) 
g = sns.barplot( y = multch['Country'].value_counts().head(10).index,
            x = multch['Country'].value_counts().head(10).values,
                palette="GnBu_d")
plt.title("Top 10 countries with highest number of Respondents");


# **Indians and Americans have taken the survey in large number. ** <br>
# > India is the largest destination for the information technology in the world and IT industry is one of the fastest growing industries in the country. Leadership across IT giants are constantly pondering on this latest disruptive revolution in Big data Analytics might be the reason behind the large number from India.
# 

# In[ ]:


data =  dict(
        type = 'choropleth',
        locations = multch['Country'].value_counts().index,
        locationmode = 'country names',
        z = multch['Country'].value_counts().values,
        text = multch['Country'].value_counts().index,
        colorbar = {'title': 'Responses per Country '})

layout = dict( title = 'Respondents across the world',
         geo = dict(showframe = False,
         projection = {'type' : 'Mercator'}))

choromap3 = go.Figure(data = [data],layout=layout)
iplot(choromap3)


# In[ ]:


multch['CompensationAmount']=multch['CompensationAmount'].str.replace(',','')
multch['CompensationAmount']=multch['CompensationAmount'].str.replace('-','')
rates=pd.read_csv('../input/conversionRates.csv')
rates.drop('Unnamed: 0',axis=1,inplace=True)
salary=multch[['CompensationAmount','CompensationCurrency','GenderSelect','Country','CurrentJobTitleSelect']].dropna()
salary=salary.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')
salary['Salary']=pd.to_numeric(salary['CompensationAmount'])*salary['exchangeRate']
salary=salary[salary['Salary']<1000000]


# In[ ]:


Salmean = salary.groupby('Country').mean()
Top_salary = Salmean.sort_values('Salary', ascending=False).head(50)

f, ax = plt.subplots(figsize=(15, 20)) 
g = sns.barplot( y = Top_salary.index,
            x = Top_salary['Salary'],
                palette="GnBu_d")
plt.title("How data scientists are paid country-wise");


# In[ ]:


Salmean = salary.groupby('Country').mean()
Salcount = salary.groupby('Country').count()
Top_salary = Salmean.sort_values('Salary', ascending=False).head(50)
data =  dict(
        type = 'choropleth',
        locations = Top_salary.index,
        locationmode = 'country names',
        z = Top_salary['Salary'],
        text = Top_salary.index,
        colorbar = {'title': 'Mean Salaries per Country '})

layout = dict( title = 'How much data scientists are paid country-wise',
         geo = dict(showframe = False,
         projection = {'type' : 'Mercator'}))

choromap3 = go.Figure(data = [data],layout=layout)
iplot(choromap3)


# Clearly we can see major gap in salaries being  paid in West and East but does that mean Asians are underpaid **?** <br>
# > The answer is **'NO'**  <br>
# 
# There are multiple influencing factors like Cost of living , purchasing power , inflation which plays pivotal role in deciding your salary and  hence the survival in any country. <br>
# > I am not an expert economist , but you can draw your own conclusion by comparing the cost of living of any two countries by visiting below website. <br>
# > https://www.numbeo.com/cost-of-living/compare_countries_result.jsp?country1=India&country2=United+States
# 
# 
# 
# 

# In[ ]:


Salmean = salary.groupby('Country')['Salary'].mean().values
Salcount = salary.groupby('Country')['Country'].count()
Countries = Salcount.index
Salcount = salary.groupby('Country')['Country'].count().values
# Sort by number of respondents
idx = Salcount.argsort()
Countries, Salmean, Salcount = [np.take(x, idx) for x in [Countries, Salmean, Salcount]]


# In[ ]:


y = np.arange(Salmean.size)
fig, axes = plt.subplots(figsize=(15, 20),ncols=2, sharey=True)
axes[0].barh(y, Salcount, align='center', color=sns.color_palette('winter',15), zorder=10)
axes[0].set(title='Number of respondents per country')
axes[1].barh(y, Salmean, align='center', color=sns.color_palette('winter',15), zorder=10)
axes[1].set(title='Mean Salary')

axes[0].invert_xaxis()
axes[0].set(yticks=y, yticklabels=Countries)
axes[0].yaxis.tick_right()

for ax in axes.flat:
    ax.margins(0.03)
    ax.grid(True)

fig.tight_layout()
fig.subplots_adjust(wspace=0.30)
plt.show()


# **Above graph makes it easy to compare no of respondents from each country and the mean salary offered in that particular country**

# In[ ]:


#f, ax = plt.subplots(figsize=(15, 7)) 
f,ax=plt.subplots(1,2,figsize=(15,7))

sns.barplot( y = multch['LanguageRecommendationSelect'].value_counts().head(10).index,
            x = multch['LanguageRecommendationSelect'].value_counts().head(10).values,
                palette="GnBu_d",ax=ax[0])
ax[0].set_title("Preferred Language by Data scientist")


df = pd.DataFrame(multch['LanguageRecommendationSelect'].value_counts().head(10).values,
                  index=multch['LanguageRecommendationSelect'].value_counts().head(10).index, 
                  columns=[' '])
ax[1].set_title("Preferred Language share")
df.plot(kind='pie', subplots=True, autopct='%1.0f%%', ax=ax[1])
#plt.subplots_adjust(wspace=0.5)
plt.show()


# > **Python and R are the two most popular programming languages used by data analysts and data scientists. Both are free and and open source, and were developed in the early 1990s—R for statistical analysis and Python as a general-purpose programming language.** <br>
# 
# > 
# ![](http://2.bp.blogspot.com/-bEfj6xswjKs/U-5JVChWI6I/AAAAAAACp4Y/raTbid6fPkk/s1600/batman-v-superman-gif-poster.gif)

# In[ ]:


f, ax = plt.subplots(figsize=(15, 7)) 
g = sns.barplot( y = multch['MLToolNextYearSelect'].value_counts().head(10).index,
            x = multch['MLToolNextYearSelect'].value_counts().head(10).values,
                palette="GnBu_d")
plt.title("Machine Learning tool interested to learn next year")
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,7))
multch.Age[multch["LanguageRecommendationSelect"] == 'Python'].plot(kind='kde',ax=ax[0])    
multch.Age[multch["LanguageRecommendationSelect"] == 'R'].plot(kind='kde',ax=ax[0])    
ax[0].set_title("Age Distribution of Python and R lover")
ax[0].legend(('Pyhon', 'R'),loc='best')

multch.Age[multch["MLToolNextYearSelect"] == 'Python'].plot(kind='kde',ax=ax[1])    
multch.Age[multch["MLToolNextYearSelect"] == 'R'].plot(kind='kde',ax=ax[1])    
multch.Age[multch["MLToolNextYearSelect"] == 'TensorFlow'].plot(kind='kde',ax=ax[1])
ax[1].set_title("Age Distr of respondents who chose tools they would like to learn in future")
ax[1].legend(('Pyhon', 'R','TensorFlow'),loc='best')
plt.plot();


# **Preference of young respondents is more towards learning Python from above Age distribution . ** <br>
# **Lets draw box plots for clear comparison .**

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,7))
filtered_language_pref = multch[(multch['LanguageRecommendationSelect']=='Python') | 
                                (multch['LanguageRecommendationSelect']=='R')]

filtered_language_next = multch[(multch['MLToolNextYearSelect']=='Python') | 
                                (multch['MLToolNextYearSelect']=='R') |
                                (multch['MLToolNextYearSelect']=='TensorFlow')
                               ]
ax[0].set_title("Age Distribution of Python and R lover")
ax[0].set_xticklabels(filtered_language_pref['LanguageRecommendationSelect'], rotation='vertical', fontsize='large')
ax[1].set_xticklabels(filtered_language_next['MLToolNextYearSelect'], rotation='vertical', fontsize='large')
ax[1].set_title("Age Distr of respondents who chose tools they would like to learn in future")
ax[0].set_ylim(0,100)
ax[1].set_ylim(0,100)

sns.boxplot(x='LanguageRecommendationSelect',y='Age', data=filtered_language_pref,ax=ax[0])
sns.boxplot(x='MLToolNextYearSelect',y='Age', data=filtered_language_next,ax=ax[1]);


# **Which tools are used by the Data scientists most of the time ** ?

# In[ ]:


v_features = multch.iloc[:,81:129].columns
Feature = []
total = []
weightage = []
for i, cn in enumerate(multch[v_features]):
    Feature.append(str(cn)[18:]) 
    #print(str(cn)[20:])
    temp = multch[cn].value_counts().index
    val = 0
    count = 0
    tempval = multch[cn].value_counts().values
    for j, k in enumerate(temp):
        if k == 'Most of the time' :
           count = count + tempval[j]
           val = val + (tempval[j] * 1.5) 
        elif  k == 'Often' :
           count = count + tempval[j]
           val = val + tempval[j]
        elif  k == 'Sometimes' :
           count = count + tempval[j]
           val = val + (tempval[j] * 0.50)
        else :
            count = count + tempval[j]
            val = val + (tempval[j] * 0.1)
    total.append(count)
    weightage.append(val/count)   


# In[ ]:


weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
idx = weightage.argsort()
Feature, total, weightage = [np.take(x, idx) for x in [Feature, total, weightage]]
s = 10
size=[]
for i, cn in enumerate(weightage):
     s = s + 100        
     size.append(s)


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 30))
ax.scatter(size, total,marker="o", color="lightBlue", s=size, linewidths=10)
ax.set_xlabel('Tools most used by DS')
ax.set_ylabel('Total no of respondents')
for i, txt in enumerate(Feature):
    if (i % 2) == 0:
        ax.annotate(txt, (size[i],total[i]-(i*3)),fontsize=12,rotation=0,color='g')
        ax.annotate(str(weightage[i])[0:4], (size[i],total[i]+50),fontsize=(12+(i/3)),rotation=0,color='r')
    else :
         ax.annotate(txt, (size[i],total[i]-(i*3)),fontsize=12,rotation=0,color='g')
         ax.annotate(str(weightage[i])[0:4], (size[i],total[i]+50),fontsize=(12+(i/3)),rotation=0,color='r')


# Python is used by the Data scientists most of the time. Next tools we see are SQL , R , Jupyter if we keep both weightage number and number of respondents in mind.

# **Lets take a glance at necessity level of the skills required at the Job** <br>
# Check out below scatter plot and pie charts to compare the necessity level of the tool

# In[ ]:


v_features = multch.iloc[:,36:46].columns
Feature = []
total = []
weightage = []
for i, cn in enumerate(multch[v_features]):
    Feature.append(str(cn)[18:]) 
    #print(str(cn)[20:])
    temp = multch[cn].value_counts().index
    val = 0
    count = 0
    tempval = multch[cn].value_counts().values
    for j, k in enumerate(temp):
        if k == 'Necessary' :
           count = count + tempval[j]
           val = val + (tempval[j] * 1.5) 
        elif  k == 'Nice to have' :
           count = count + tempval[j]
           val = val + tempval[j]
        else :
            count = count + tempval[j]
            val = val + (tempval[j] * 0)
    total.append(count)
    weightage.append(val/count)   


# In[ ]:


weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
idx = weightage.argsort()
Feature, total, weightage = [np.take(x, idx) for x in [Feature, total, weightage]]
s = 10
size=[]
for i, cn in enumerate(weightage):
     s = s + 100        
     size.append(s)


# In[ ]:


fig, ax = plt.subplots(figsize=(18, 8))
ax.scatter(size, total,marker="o", color="lightBlue", s=size, linewidths=10)
ax.set_xlabel('Necessity level of Tools in job')
ax.set_ylabel('Total no of respondents')
for i, txt in enumerate(Feature):
    if (i % 2) == 0:
        ax.annotate(txt, (size[i],total[i]-(i*3)),fontsize=12+i,rotation=0,color='g')
        ax.annotate(str(weightage[i])[0:4], (size[i],total[i]+10),fontsize=(12+(i/3)),rotation=0,color='r')
    else :
         ax.annotate(txt, (size[i],total[i]-(i*3)),fontsize=12+i,rotation=0,color='g')
         ax.annotate(str(weightage[i])[0:4], (size[i],total[i]+10),fontsize=(12+(i/3)),rotation=0,color='r')


# In[ ]:


v_features = multch.iloc[:,36:46].columns
plt.figure(figsize=(12,28))
gs = gridspec.GridSpec(5,2)
for i, cn in enumerate(multch[v_features]):
    ax = plt.subplot(gs[i])
    temp = multch[cn].value_counts().index
    explode = []
    colors=[]
    for k in temp:
        if k == 'Necessary' :
           explode.append(0.1)
           colors.append('g')
        elif  k == 'Unnecessary' :
           colors.append('r')
           explode.append(0) 
        else :
            explode.append(0)
            colors.append('b')
    multch[cn].value_counts().plot.pie(autopct='%1.1f%%',explode=explode,colors=colors,shadow=True)
    ax.set_ylabel('')
    ax.set_title('Necessity Level of ' + str(cn)[18:] + ' in job')
    


# Above scatter plot and pie charts are self explantory and we can see respondents have voted for Python and stats. <br>
# As per the survey these two skills are enough to kick start your career in Data science field but the respondents(sample) here do not represent the data scientist population out there as we do not see respondents in great numbers from many countries.
# If we bring down the criteria of necessity level to 40% or further down to 35% we can see even having knowldge of SQL , R , Visualizations and Big data will help grab the attention of recruiters.
# 

# Preferred Language?  We have more people using Python than R<br>
# 
# > If you look at my first few graphs on University major and current job you will find most of the respondents are Computer science degree holder and working either as Data scientist or Software
#    engineer . <br> <br>
# A couple of years back C++ and Java was the main preferred languages to be tought in  almost all the universities across the   world to introduce young engineers to programming world. It becomes easy for engineers to relate and learn Python than R so that maybe the reason why Data scientists and Engineers are using Python more than R<br><br>
# 
# Respondents chose Python over R to be the next tool they are interested to learn.
# >  In machine learning applications, for data transformation and building applications, Python, as a general-purpose language,  seems much easier to use (however you can still build GUI and web services entirely in R). Also ML library is extraordinary, so people often choose it over R.
# 
# So we not only have more Python lover here but it will remain the popular tool among the aspiring scientist in near future.<br>
# <center> and the clear winner is Python  </center>
#                                                   ---------
# 
# ![](https://i.pinimg.com/originals/83/a1/71/83a1714544324afcd635f96ccd9c9932.gif)

# **Lets take a glace at usefulness of  various platforms contributing to your data science career**

# In[ ]:


v_features = multch.iloc[:,16:34].columns
plt.figure(figsize=(12,28))
gs = gridspec.GridSpec(6, 3)
for i, cn in enumerate(multch[v_features]):
    ax = plt.subplot(gs[i])
    temp = multch[cn].value_counts().index
    explode = []
    colors=[]
    for k in temp:
        if k == 'Very useful' :
           explode.append(0.1)
           colors.append('g')
        elif  k == 'Somewhat useful' :
           colors.append('blue')
           explode.append(0) 
        else :
            explode.append(0)
            colors.append('r')
    multch[cn].value_counts().plot.pie(autopct='%1.1f%%',explode=explode,colors=colors,shadow=True)
    ax.set_ylabel('')
    ax.set_title('Are/Is ' + str(cn)[26:] + ' useful ?')


# In[ ]:


v_features = multch.iloc[:,16:34].columns
Feature = []
total = []
weightage = []
for i, cn in enumerate(multch[v_features]):
    Feature.append(str(cn)[26:]) 
    #print(str(cn)[20:])
    temp = multch[cn].value_counts().index
    val = 0
    count = 0
    tempval = multch[cn].value_counts().values
    for j, k in enumerate(temp):
        if k == 'Very useful' :
           count = count + tempval[j]
           val = val + (tempval[j] * 1.5) 
        elif  k == 'Somewhat useful' :
           count = count + tempval[j]
           val = val + tempval[j]
        else :
            count = count + tempval[j]
            val = val + (tempval[j] * 0)
    total.append(count)
    weightage.append(val/count)   


# In[ ]:


weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
idx = weightage.argsort()
Feature, total, weightage = [np.take(x, idx) for x in [Feature, total, weightage]]
s = 10
size=[]
for i, cn in enumerate(weightage):
     s = s + 100        
     size.append(s)


# In[ ]:


fig, ax = plt.subplots(figsize=(18, 8))
ax.scatter(size, total,marker="o", color="lightBlue", s=size, linewidths=10)
ax.set_xlabel('Usefulness index of various Learning platform')
ax.set_ylabel('Total no of respondents')
for i, txt in enumerate(Feature):
    if (i % 2) == 0:
        ax.annotate(txt, (size[i],total[i]-300),fontsize=(11+i),rotation=0,color='g')
        ax.annotate(str(weightage[i])[0:4], (size[i]-50,total[i]+50),fontsize=(12+i),rotation=0,color='r')    
    else :
        ax.annotate(txt, (size[i],total[i]-300),fontsize=(11+i),rotation=-6,color='g')
        ax.annotate(str(weightage[i])[0:4], (size[i]-50,total[i]+50),fontsize=(12+i),rotation=0,color='r')    


# **Respondents have voted for Projects, Courses , Kaggle as the most Useful Learning platform.**

# **Lets analyse how often different machine learning algorithms are being used by the respondents.**

# In[ ]:


v_features = multch.iloc[:,133:163].columns
plt.figure(figsize=(20,55))
gs = gridspec.GridSpec(15, 2)
for i, cn in enumerate(multch[v_features]):
    ax = plt.subplot(gs[i])
    sns.countplot(y=str(cn), data=multch,order=multch[str(cn)].value_counts().index, palette="Set2")
    ax.set_title('how often do you use the "' + str(cn)[20:] + '" ?')
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')


# If you are interested for response on particular skill then refer above bar charts. <br>
#                        --
# Comparing all skills in one single graph ? <br>
#       --
# > We have to weigh the response from different categories like skill used 
# 'Most of the time' , skill used 'often' , skill used 'sometimes' to arrive at the number representing that particular skill . 
# Use number of respondents vs Skill most used by these respondents for clear distinction on graph for comparison. <br>
# 

# In[ ]:


v_features = multch.iloc[:,133:163].columns
Feature = []
total = []
weightage = []
for i, cn in enumerate(multch[v_features]):
    Feature.append(str(cn)[20:]) 
    #print(str(cn)[20:])
    temp = multch[cn].value_counts().index
    val = 0
    count = 0
    tempval = multch[cn].value_counts().values
    for j, k in enumerate(temp):
        if k == 'Most of the time' :
           count = count + tempval[j]
           val = val + (tempval[j] * 1.5) 
        elif  k == 'Often' :
           count = count + tempval[j]
           val = val + tempval[j]
        elif  k == 'Sometimes' :
           count = count + tempval[j]
           val = val + (tempval[j] * 0.50)
        else :
            count = count + tempval[j]
            val = val + (tempval[j] * 0.1)
    total.append(count)
    weightage.append(val/count) 


# In[ ]:


weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
idx = weightage.argsort()
Feature, total, weightage = [np.take(x, idx) for x in [Feature, total, weightage]]
s = 10
size=[]
for i, cn in enumerate(weightage):
     s = s + 100        
     size.append(s)


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 20))
ax.scatter(size, total,marker="o", color="lightBlue", s=size, linewidths=10)
ax.set_xlabel('Algorithms/skills most used by respondents')
ax.set_ylabel('Total no of respondents')
for i, txt in enumerate(Feature):
    if (i % 2) == 0:
        ax.annotate(txt, (size[i],total[i]-(i*3)),fontsize=12,rotation=-10,color='g')
        ax.annotate(str(weightage[i])[0:4], (size[i],total[i]+50),fontsize=(12+(i/3)),rotation=0,color='r')
    else :
         ax.annotate(txt, (size[i],total[i]-(i*3)),fontsize=12,rotation=0,color='g')
         ax.annotate(str(weightage[i])[0:4], (size[i],total[i]+50),fontsize=(12+(i/3)),rotation=0,color='r')


# **Data Visualization and Corss validation are used most of the time by respondents **
# > Data visualization to find interesting patterns which will guide you for model building <br>
#    Cross validation to see how your model will perform on unseen data or to declare fitness of the model <br>
# 
# **We have more respondents for Logistic regression than for GBM(Gradient boosting) but small sect of respondent of GBM use this skill most of the time compared to logistic regression**

# <center> How much time is needed for various activities DS performs during the job  </center>
#                                                   ---------
# ![](https://img.huffingtonpost.com/asset/56f97e341e0000b300705789.gif?ops=scalefit_600_noupscale)

# In[ ]:


v_features = multch.iloc[:,166:172].columns
plt.figure(figsize=(20,15))
gs = gridspec.GridSpec(3, 2)
for i, cn in enumerate(multch[v_features]):
    ax = plt.subplot(gs[i])
    sns.kdeplot(multch[str(cn)].dropna(), shade=True, color="r")
    ax.set_title('Time needed for ' + str(cn)[4:])
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')


# **Gathering data and building model using the data eats up lots of time compared to other activities** <br>
# Above graphs provide sufficient evidence to prove below lines are actually correct <br>
# 
# > Data scientists, according to interviews and expert estimates, spend 50 percent to 80 percent of their time mired in the mundane labor of collecting and preparing unruly digital data, before it can be explored for useful nuggets."  <br>
# ** -The New York Times **
#                                                

# **Lets explore Job satisfaction level of countries with highest number of respondents.**

# In[ ]:


v_countries = multch['Country'].value_counts().head(10).index
Feature = []
weightage = []
total = []

for i, cn in enumerate(v_countries):
    Feature.append(str(cn)) 
    filtered = multch[(multch['Country']==str(cn))]
    temp = filtered['JobSatisfaction'].value_counts().index
    tempval = filtered['JobSatisfaction'].value_counts().values
    val = 0
    count = 0
    for j, k in enumerate(temp):
        if k == '10 - Highly Satisfied' :
           count = count + tempval[j]
           val = val + (tempval[j] * 10) 
        elif  k == '1 - Highly Dissatisfied' :
           count = count + tempval[j]
           val = val + tempval[j]
        elif  k == 'I prefer not to share' :
           count = count + tempval[j]
           val = val + (tempval[j] * 0)
        else :
            count = count + tempval[j]
            val = val + (tempval[j] * int(k))
    total.append(count)
    weightage.append(val/count)
weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
idx = weightage.argsort()
Feature, total, weightage = [np.take(x, idx) for x in [Feature, total, weightage]]
s = 10
size=[]
for i, cn in enumerate(weightage):
     s = s + 100        
     size.append(s)


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))
ax.scatter(size, total,marker="o", color="lightBlue", s=size, linewidths=10)
ax.set_xlabel('Job Satisfaction index in Top 10 countries')
ax.set_ylabel('Total no of respondents')
for i, txt in enumerate(Feature):
    if (i % 2) == 0:
        ax.annotate(txt, (size[i],total[i]),fontsize=(12+i),rotation=0,color='g')
        ax.annotate(str(weightage[i])[0:4], (size[i],total[i]+50),fontsize=(12+i),rotation=0,color='r')
    else :
         ax.annotate(txt, (size[i],total[i]-150),fontsize=(12+i),rotation=0,color='g')
         ax.annotate(str(weightage[i])[0:4], (size[i],total[i]-70),fontsize=(12+i),rotation=0,color='r')


# The numbers above says it all . Working in USA , Canada and France can be a rewarding experince for aspiring Data scientist. <br>
# But wait , Are we too early to arrive at this conclusion or shall we wait for the one more comprehensive survey by Kaggle? 

# **What challenges are faced by Data scientist at workplace**

# In[ ]:


v_features = multch.iloc[:,174:196].columns
Feature = []
total = []
weightage = []
for i, cn in enumerate(multch[v_features]):
    Feature.append(str(cn)[22:]) 
    #print(str(cn)[20:])
    temp = multch[cn].value_counts().index
    val = 0
    count = 0
    tempval = multch[cn].value_counts().values
    for j, k in enumerate(temp):
        if k == 'Most of the time' :
           count = count + tempval[j]
           val = val + (tempval[j] * 1.5) 
        elif  k == 'Often' :
           count = count + tempval[j]
           val = val + tempval[j]
        elif  k == 'Sometimes' :
           count = count + tempval[j]
           val = val + (tempval[j] * 0.50)
        else :
            count = count + tempval[j]
            val = val + (tempval[j] * 0.1)
    total.append(count)
    weightage.append(val/count)   


# In[ ]:


weightage = np.array(weightage)
Feature = np.array(Feature)
total = np.array(total)
idx = weightage.argsort()
Feature, total, weightage = [np.take(x, idx) for x in [Feature, total, weightage]]
s = 10
size=[]
for i, cn in enumerate(weightage):
     s = s + 100        
     size.append(s)


# In[ ]:


fig, ax = plt.subplots(figsize=(17, 20))
ax.scatter(size, total,marker="o", color="lightBlue", s=size, linewidths=10)
ax.set_xlabel('Challenges DS faced most often @ workplace')
ax.set_ylabel('Total no of respondents')
for i, txt in enumerate(Feature):
    if (i % 2) == 0:
        ax.annotate(txt, (size[i]-50,total[i]-(i*3)),fontsize=12,rotation=-17,color='g')
        ax.annotate(str(weightage[i])[0:4], (size[i],total[i]+15),fontsize=(12+(i/3)),rotation=0,color='r')
    else :
         ax.annotate(txt, (size[i]-50,total[i]-(i*3)),fontsize=12,rotation=-17,color='g')
         ax.annotate(str(weightage[i])[0:4], (size[i],total[i]+15),fontsize=(12+(i/3)),rotation=0,color='r')


# **Dirty Data is the biggest challenge faced by the respondents. **  <br>
# You can easily figure out other challenges faced by Data Scientist by loking at the weightage numbers i showed on the graph.

# **Guys , Please let me know your thoughts on this kernel and whether it managed to unveil the secret powers of superheroes or not  :)**  <br>
# 
# Ending this kernel with below wordcloud highlighting important data points.

# In[ ]:


superman = b'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'


# In[ ]:


v_features = multch.iloc[:,36:46].columns
Feature = []
for i, cn in enumerate(multch[v_features]):
     Feature.append(str(cn)[18:]) 
v_features = multch['MajorSelect'].dropna().value_counts().head(20).index
for i, cn in enumerate(v_features):
     Feature.append(str(cn)) 
v_features =  multch['LanguageRecommendationSelect'].value_counts().head(10).index
for i, cn in enumerate(v_features):
     Feature.append(str(cn)) 
v_features = multch['MLToolNextYearSelect'].value_counts().head(10).index
for i, cn in enumerate(v_features):
     Feature.append(str(cn)) 


# In[ ]:


fig, ax = plt.subplots(figsize=(15,12), ncols=2, nrows=1)
# Generate the Mask for Supe
f1 = open("superman.png", "wb")
f1.write(codecs.decode(superman,'base64'))
f1.close()
#img1 = imread("superman.png")
#hcmask1 = img1
img1 = "superman.png"
# Read in the mask
hcmask1 = np.array(Image.open(path.join(img1)))

ax[0].imshow(mpimg.imread('superman.png'))
ax[0].axis('off')

#plt.figure(figsize=(28,25))
#plt.subplot(211)
wc = WordCloud(background_color="Black", max_words=10000, mask=hcmask1, 
               stopwords=STOPWORDS, max_font_size= 40)
wc.generate(" ".join(Feature))
ax[1].imshow(wc.recolor( colormap= 'nipy_spectral' , random_state=17), alpha=0.9)
ax[1].axis('off');

