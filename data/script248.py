
# coding: utf-8

# ## Coders Not Hackers (HackerRank Survey Analysis)
# 
# ![](https://amritchhetri.files.wordpress.com/2016/10/logo_wordmark-vertical-800x645.png?w=800)
# HackerRank is one of the most used platforms for practicing Competitive Coding. Not only students, but also businesses take advantage of it, by using it as a platform for hiring students. I remember giving so many hackerrank tests during my placements. It has many tracks like **Linux Shell Scripting, SQL, Programming Language Practise Problems, and also ML/AI Problems**(can be a competiton to Kaggle..:p). This year HackerRank conducted a survey for its users,collecting data like languages used by them, their source of study, their domain of study,etc. I too had participated in this survey, and it would be a fun analysing this dataset, and maybe I will be able to find my own response(seems impossible..xd).
# 
# A similar dataset was released by Kaggle last year, which had responses by Kaggle Users regarding their opinions on Data Science and Machine Learning.My notebook on the Kaggle ML Dataset is **[linked here](https://www.kaggle.com/ash316/novice-to-grandmaster)**. I will try to compare results based on some common parameters between both these datasets.So lets start our analysis.
# 
# #### Focus of the Notebook
# 
# As the dataset aims to find **Women's involvement in tech, opinions of students and professionals**, this notebook will also revolve around the same topic. We will try to find out how much are women involved in tech and some other insights with our exploration.
# 
# If u like this notebook, **Do Upvote**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls


# In[2]:


df=pd.read_csv('../input/HackerRank-Developer-Survey-2018-Values.csv')


# ## Checking Null Values

# In[3]:


import missingno as msno
msno.matrix(df)


# So the above diagram shows the amount of missing data in the dataset. If we use the traditional .isnull() method to check the number of null values, we will get a list of null values for all columns, which can be difficult to interpret. So we can see that the dataset has a lot of null values. It was completely expected, as it is a survey data, it will be filled with many blank and dummy values. I would like to thank [Aleksey Bilogur](https://www.kaggle.com/residentmario), for this great **missingno** library.

# ### Total Number of Respondents

# In[4]:


print('The Total Number of Respondents are:',df.shape[0])


# So we have data for more than 25k users. The Kaggle Dataset had responses for about 16k users, thus the Hackerrank dataset has about 9k more reponses.

# ## Gender Distribution

# In[5]:


df['q3Gender'].value_counts()[:3].plot.barh(width=0.9)
plt.show()


# The graph shows that only about **20%** of the respondents are women. This output is analogous to the Kaggle Survey Gender Distribution output, where the number of women were very less as compared to the male.

# ## Respondents Around the World

# In[6]:


countries=df['CountryNumeric2'].value_counts().to_frame()
data = [ dict(
        type = 'choropleth',
        locations = countries.index,
        locationmode = 'country names',
        z = countries['CountryNumeric2'],
        text = countries['CountryNumeric2'],
        colorscale ='Viridis',
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Survey Respondents'),
      ) ]

layout = dict(
    title = 'Survey Respondents by Nationality',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='survey-world-map')


# Okay so after the changes made by the HackerRank Team regarding the country encoding, we now see that **Ghana** is not the country with highest respondent but it is **India** followed by **United States**.

# ## Age Distribution

# In[7]:


f,ax=plt.subplots(1,2,figsize=(25,20))
ax1=df[df['q1AgeBeginCoding']!='#NULL!'].q1AgeBeginCoding.value_counts().sort_values(ascending=True).plot.barh(width=0.9,ax=ax[0],color='y')
for i, v in enumerate(df[df['q1AgeBeginCoding']!='#NULL!'].q1AgeBeginCoding.value_counts().sort_values(ascending=True)): 
    ax1.text(.8, i, v,fontsize=18,color='b',weight='bold')
ax[0].set_title('Age Began Coding',size=30)
ax2=df[df['q2Age']!='#NULL!'].q2Age.value_counts().sort_values(ascending=True).plot.barh(width=0.9,ax=ax[1],color='y')
for i, v in enumerate(df[df['q2Age']!='#NULL!'].q2Age.value_counts().sort_values(ascending=True)): 
    ax2.text(.8, i, v,fontsize=18,color='b',weight='bold')
ax[1].set_title('Present Age',size=30)
plt.show()


# Its evident that majority of the respondents lay in the age group of **(18-34)**. This observation is similar to that of Kaggle Survey. Also it seems that people are taking up coding at an early age of **(11-15)**, which is a very good trait. I remember memorising some boring textbooks at this age..not at all interesting. Lets dig in a bit and check how much are the Women's involved in coding, and at what age did they start coding.

# In[8]:


f,ax=plt.subplots(1,2,figsize=(25,12))
curr_age=df[df.q3Gender.isin(['Male','Female'])].groupby(['q2Age','q3Gender'])['StartDate'].count().reset_index()
curr_age=curr_age[curr_age['q2Age']!='#NULL!']
curr_age.pivot('q2Age','q3Gender','StartDate').plot.barh(ax=ax[0])
ax[0].set_title('Current Age')
code_age=df[df.q3Gender.isin(['Male','Female'])].groupby(['q1AgeBeginCoding','q3Gender'])['StartDate'].count().reset_index()
code_age=code_age[code_age['q1AgeBeginCoding']!='#NULL!']
plt.figure(figsize=(15,15))
code_age.pivot('q1AgeBeginCoding','q3Gender','StartDate').plot.barh(ax=ax[1])
ax[1].set_title('Age Started Coding')
plt.subplots_adjust(hspace=0.8)
plt.show()


# The above graph looks to be very promising. Older women i.e age range**(35+)** don't look to be  into tech. Also majority of women participating in the survey lay betweeen the age group **(18-34)**, and they also have started coding at an early age. Thus Younger Women are getting into tech and that is a **positive trait**. 

# ## Survey Response Time
# 
# There are about 35-40 questions asked in the survey, most of them being MCQ's. So it would take some 10-15  mins at max to complete the entire surevy. Let's check the time taken by the respondents.

# In[9]:


t1=pd.to_datetime(df['StartDate'])
t2=pd.to_datetime(df['EndDate'])
d = {'col1': t1, 'col2': t2}
trying=pd.DataFrame(d)
list1=[]
for i,j in zip(trying['col2'],trying['col1']):
    list1.append(pd.Timedelta(i-j).seconds / 60.0)


# In[10]:


print('Shortest Survey Time:',pd.Series(list1).min(),'minutes')
print('Longest Survey Time:',pd.Series(list1).max(),'minutes')
print('Mean Survey Time:',pd.Series(list1).mean(),'minutes')


# The results look funny. 35 questions in 2 mins, that means **(SKIP..SKIP..SKIP)**. And what was this person doing for 1436 minutes. Lets check if there are  more such values.

# In[11]:


(pd.Series(list1)>50).value_counts()


# Indeed there are many such values. I will remove them from histogram, to properly check the time taken by the users to respond.

# In[12]:


plt.figure(figsize=(10,6))
pd.Series(list1).hist(bins=500,edgecolor='black', linewidth=1.2)
plt.xlim([0,50])
plt.xticks(range(0,50,5))
plt.title('Response Time Distribution')
plt.show()


# As expected, majority of the response time lies between **(5-15)** mins.

# ## Degree Focus

# In[13]:


f,ax=plt.subplots(2,1,figsize=(15,12))
sns.countplot('q5DegreeFocus',hue='q3Gender',data=df[df['q5DegreeFocus']!='#NULL!'],ax=ax[0])
ax[0].set_title('STEM Courses')
sns.countplot('q0005_other',hue='q3Gender',data=df[df['q0005_other'].isin(df['q0005_other'].value_counts().index[:5])],ax=ax[1])
ax[1].set_title('Other Courses')
plt.show()


# A majority of the respondents are from the CS background. It is also good to see that about **50-60%** women are from CS background. The 2nd graph is from the q0005_other column, where the user can enter other courses other than the 1st graph.

# ## Working Industry

# In[14]:


plt.figure(figsize=(10,8))
df[(df['q10Industry']!='#NULL!')&(df['q8Student']!='Students')].q10Industry.value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r'))
plt.title('Working Industry')
plt.show()


# Around 6.5k respondents work in the Technology Industry., followed by Financial Services and Retail. I have **not considered students** as they don't really work in an industry.Lets dig in and check in which industries do Women work

# ### Women Working in Industry

# In[15]:


plt.figure(figsize=(8,8))
def absolute_value(val):
    a  = np.round(val/100.*sizes.sum(), 0)
    return a
sizes=df[(df['q3Gender']=='Female')&(df['q10Industry']!='#NULL!')&(df['q8Student']!='Students')].q10Industry.value_counts()[:10]
labels=df[(df['q3Gender']=='Female')&(df['q10Industry']!='#NULL!')&(df['q8Student']!='Students')].q10Industry.value_counts()[:10].index
plt.pie(sizes,autopct=absolute_value,labels=labels,colors=sns.color_palette('Set3',10))
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# The results are similar as the previous graph. About **50%** of the women work in tech industry, which is a good indication of women working in tech industry. But working in a tech company doesn't mean that a person is doing tech work. There are other jobs like business analyst, advisors, etc, which are not tech jobs. Lets see what are the different jobs done by them.

# ### Job Descriptions 

# In[16]:


plt.figure(figsize=(8,12))
women_indus=df[(df['q3Gender']=='Female')&(df['q10Industry']!='#NULL!')&(df['q8Student']!='Students')]
ax=women_indus.q9CurrentRole.value_counts().plot.barh(width=0.9,color=sns.color_palette('winter_r',15))
for i, v in enumerate(women_indus.q9CurrentRole.value_counts().values): 
    ax.text(.8, i, v,fontsize=12,color='b',weight='bold')
plt.gca().invert_yaxis()
plt.title('Current Job Description for Women')
ax.patches[0].set_facecolor('r')
plt.show()


# The output looks good. Almost every women in the Tech Industry are doing Tech jobs, with the highest being as Software Engineers. However the number of women in **Data Science** is very less, only 106. Maybe this is the domain where women need to concentrate more. Lets check poitions held by Women in their respective companies.

# ### Positions Held

# In[17]:


plt.figure(figsize=(8,10))
ax=women_indus.q8JobLevel.value_counts().plot.barh(width=0.9,color=sns.color_palette('winter_r',15))
for i, v in enumerate(women_indus.q8JobLevel.value_counts().values): 
    ax.text(.8, i, v,fontsize=12,color='blue',weight='bold')
plt.gca().invert_yaxis()
ax.patches[0].set_facecolor('r')
plt.title('Positions Held By Women')
plt.show()


# A very high number of women are still at the Junior Level, with a very few being CTO/VP etc.  This was expected, as we had seen earlier that majority of the women in the survey were young in the range **(18-24)** years, which means that they have just started their career.

# ### Emerging Tech Skill

# In[18]:


plt.figure(figsize=(12,10))
sns.countplot(y='q27EmergingTechSkill',hue='q3Gender',data=df[(df['q3Gender'].isin(['Male','Female']))&(df['q27EmergingTechSkill']!='#NULL!')])
plt.xticks(rotation=90)
plt.show()


# This graph looks promising, as a good number of women are looking forward to understand or learn **Machine Learning/AI**. Thus in the next year's survey data, I hope the count of women in Data Science/ML/AI increases by a good number. Lets check the other's column to explore the free response question

# In[19]:


from wordcloud import WordCloud, STOPWORDS
plt.figure(figsize=(16,8))
wc = WordCloud(background_color="white", max_words=1000, 
               stopwords=STOPWORDS,width=1000,height=1000)
wc.generate(" ".join(df['q0027_other'].dropna()))
plt.imshow(wc)
plt.axis('off')
plt.show()


# The results are very similar. **Data Science, AI, CLoud Computing and Cyber-Security** have a huge scope in the future. It is evident that with the increasiing data, we need to save the data, thus we need Cloud, then we need to analyse and build automative systems, for this we need Data Science/ML/AI and Security for storing the data and systems securely. Thus candidates developing skills in these domains have a bright future ahead.

# ### From The Recruiters Desk
# 
# Hiring Managers have a tough time recruiting the right talent, with the desired skills. Lets see how they evaluate candidates and the diffculties they face during recruitment.

# In[20]:


df1=df[df['q16HiringManager']=='Yes']
f,ax=plt.subplots(1,3,figsize=(25,10))
hire1=df1[df1.columns[df1.columns.str.contains('HirCha')]].apply(pd.Series.value_counts).melt().dropna().set_index('variable').sort_values(by='value',ascending=True)
hire1.index=hire1.index.str.replace('q17HirCha','')
hire1.plot.barh(width=0.9,ax=ax[0])
ax[0].set_title('Hiring Challenges')
hire2=df1[df1.columns[df1.columns.str.contains('TalTool')]].apply(pd.Series.value_counts).melt().dropna().set_index('variable').sort_values(by='value',ascending=True)
hire2.index=hire2.index.str.replace('q19TalTool','')
hire2.plot.barh(width=0.9,ax=ax[1])
ax[1].set_title('Talent Assessment Tools')
hire3=df1[df1.columns[df1.columns.str.contains('Cand')]].apply(pd.Series.value_counts).melt().dropna().set_index('variable').sort_values(by='value',ascending=True)
hire3.index=hire3.index.str.replace('q20Cand','')
hire3.plot.barh(width=0.9,ax=ax[2])
ax[2].set_title('Prefered Qualifications')
plt.show()


# #### Observations:
#  - **Assessing skills and Interviews** look be some main challenges Interviwers face while hiring candidate.Also it looks like there is a lack of supply of talented candidates in the market, as hiring managers find it difficult in finding so. But why do they find it difficult to assess skills? Maybe the candidates are not able to reflect their skills properly. The best way could be **a github profile, kaggle profile or similarly a hackerrank profile.**
#  - Resume Screening and Referral are the most common ways for assessing talent.
#  - **HackerRank lags behind when it comes to be used as a platform for hiring candidates**. Even though HackerRank is a great platform where students develop coding skills, not many managers are using it for talent assessment. As I had stated above that I have personally attempted HackerRank assessments, I feel it is better than traditional methods like Resume Screening, as it can really prove whether a candidate really knows something or not. Also being an automated test platform, results are delivered faster with in-depth analysis.
#  - Coming to preferred qualifications, **Work Experience and Projects** prove to be very beneficial for candidates. I had found same results when working with the Kaggle ML Survey. Thus Work-Ex and Projects matters a lot irrespective to the field of interest.
#  
#  Lets see what other skills/qualities hiring managers look at while hiring

# In[21]:


df1['q0020_other'].value_counts().to_frame()[:10].style.background_gradient(cmap='summer_r')


# The free response question is very less populated, thus we see very few numbers.  But we can say that managers look out for passionate and enthusiastic personalities.

# ### How Do I get Recruited??
# 
# Lets check what are the in-demand skills in the industry

# In[22]:


lang_prof=df1[df1.columns[df1.columns.str.contains('LangProf')]]
lang_prof=lang_prof.apply(pd.Series.value_counts)
lang_prof=lang_prof.melt()
lang_prof.dropna(inplace=True)
lang_prof['variable']=lang_prof['variable'].str.replace('q22LangProf','')
lang_prof.set_index('variable',inplace=True)
frame_prof=df1[df1.columns[df1.columns.str.contains('q23Frame')]]
frame_prof=frame_prof.apply(pd.Series.value_counts)
frame_prof=frame_prof.melt()
frame_prof.dropna(inplace=True)
frame_prof['variable']=frame_prof['variable'].str.replace('q23Frame','')
frame_prof.set_index('variable',inplace=True)
core_comp=df1[df1.columns[df1.columns.str.contains('CoreComp')]]
core_comp=core_comp.apply(pd.Series.value_counts)
core_comp=core_comp.melt()
core_comp.dropna(inplace=True)
core_comp['variable']=core_comp['variable'].str.replace('q21CoreComp','')
core_comp.set_index('variable',inplace=True)
f,ax=plt.subplots(1,3,figsize=(25,15))
lang_prof.sort_values(ascending=True,by='value').plot.barh(width=0.9,ax=ax[0],color=sns.color_palette('inferno_r',20))
ax[0].set_ylabel('Language')
ax[0].set_title('Programming Competancy in Developer Candidates')
frame_prof.sort_values(ascending=True,by='value').plot.barh(width=0.9,ax=ax[1],color=sns.color_palette('inferno_r',20))
ax[1].set_ylabel('Frameworks')
ax[1].set_title('FrameWorks Competancy in Developer Candidates')
core_comp.sort_values(ascending=True,by='value').plot.barh(width=0.9,ax=ax[2],color=sns.color_palette('inferno_r',20))
ax[2].set_ylabel('Skills')
ax[2].set_title('Other Skills in Developer Candidates')
plt.show()


# The above graphs are the in-demand skills/technologies that hiring managers look in candidates
# 
# #### Observations:
#  - **Java, Javascript and its frameworks like AngularJs and Node.js** are the most in demand skills that recruiters look in aspiring software developers.
#  - But **Problem Solving Skills** matters even more than any technology or languages. Problem solving skills include analytical thinking, mental ability, evaluation and many other abities and it is required in every industry. Think of data science, some people use R, some use Python and many other tools. But what makes a data scientist great, is his problem solving skills, not his tech stack. 
#  
#  Lets again check the free response column

# In[23]:


df['q0022_other'].value_counts().to_frame()[:10].style.background_gradient(cmap='summer_r')


# This is something good. We can see that **SQL** is also a highly demanded skill. This is analogous to the Kaggle Survey, where we saw that SQL was one of the most demanding skill in Data Science.

# ### How many recruitments for the next year??

# In[24]:


plt.figure(figsize=(6,6))
df1[df1['q18NumDevelopHireWithinNextYear']!='#NULL!'].q18NumDevelopHireWithinNextYear.value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r',15))
plt.title('Hires Within Next Year')
plt.show()


# Sadly managers are not looking to hire many candidates for the next year. But this situation may not be same in all countries. Lets check this by countries

# In[25]:


plt.figure(figsize=(8,8))
coun=df1.groupby(['CountryNumeric2','q18NumDevelopHireWithinNextYear'])['q3Gender'].count().reset_index()
coun=coun[coun['q18NumDevelopHireWithinNextYear']!='#NULL!']
coun=coun.pivot('CountryNumeric2','q18NumDevelopHireWithinNextYear','q3Gender').dropna(thresh=6)
sns.heatmap(coun,cmap='RdYlGn',fmt='2.0f',annot=True)
plt.show()


# **USA and India** are look to be the countries with many opportunities. Specially USA, as about 26 managers are looking to hire 1000+ candidates for next year. Surprisingly, there was no entry for Ghana. This shows that respondents from Ghana were only students.

# ### Languages Familarity

# In[26]:


import itertools
df2=df.copy()
df2['q8Student'].fillna('Not Student',inplace=True)
columns=['q25LangC','q25LangCPlusPlus','q25LangJava','q25LangPython','q25LangJavascript','q25LangCSharp','q25LangGo','q25Scala','q25LangPHP','q25LangR']
plt.subplots(figsize=(30,30))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2+1),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    sns.countplot(i,hue='q8Student',data=df2,order=df2[i].value_counts().index)
    plt.title(i,size=20)
    plt.ylabel('')
    plt.xlabel('')
plt.show()


# In[27]:


df['q25LangOther'].value_counts().to_frame()[:10].style.background_gradient(cmap='summer_r')


# #### Observations:
# 
#  - People are very much familiar with C/C++ and Java. An interesting finding is that more number of students are willing to learn C/C++ or Java as compared to working individuals, and that can be verified by the above graph. 
#  - **Python** has the best graph overall, where the number of people willing to learn and those who already know it are in good proportion. Thus having a good grasp over Python can be very beneficial.
#  - **Go and Scala** look to be the sleeping giants. Not many students currently know it, but a huge number of students as well as working professionals are willing to learn them. Thus learning these 2 languages also will be very beneficial.
#  - **Elixir** is another dynamic language that seems to be gaining popularity, as we can see many responses for elixir in the free response column for languages.

# ### Love or Hate??
# 
# Not everyone loves every language or framework. Many factors like the **syntax, scalability, ease of learning,** etc are some of the factors due to which programmers either hate or love a language.

# In[28]:


columns=['q28LoveC','q28LoveCPlusPlus','q28LoveJava','q28LovePython','q28LoveJavascript','q28LoveCSharp','q28LoveGo','q28LoveScala','q28LovePHP','q28LoveR']
plt.subplots(figsize=(30,30))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2+1),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    sns.countplot(i,hue='q8Student',data=df2,order=df2[i].value_counts().index)
    plt.title(i,size=20)
    plt.ylabel('')
    plt.xlabel('')
plt.show()


# #### Observations:
# 
#  - **Python** here again is a clear winner, with so much LOVE and almost no HATE
#  - The graphs for other languages look to be similar, with many people loving and many hating as well.

# ### Opinions(Students vs Professionals)

# In[29]:


working=df2[df2['q8Student']=='Not Student']
working=working[working.columns[working.columns.str.contains('q12JobCrit')]]
working=working.apply(pd.Series.value_counts).melt().dropna().set_index('variable')
students=df2[df2['q8Student']=='Students']
students=students[students.columns[students.columns.str.contains('q12JobCrit')]]
students=students.apply(pd.Series.value_counts).melt().dropna().set_index('variable')
working=working[working.index!='q12JobCritOther']
stu_work=working.merge(students,left_index=True,right_index=True,how='left')
stu_work['total']=stu_work['value_x']+stu_work['value_y']
stu_work['%work']=stu_work['value_x']/stu_work['total']
stu_work['%student']=stu_work['value_y']/stu_work['total']
stu_work.drop(['value_x','value_y','total'],axis=1,inplace=True)
stu_work.index=stu_work.index.str.replace('q12JobCrit','')
stu_work.plot.barh(stacked=True,width=0.9)
fig=plt.gcf()
fig.set_size_inches(8,8)
plt.title('Important Things for Job(Professionals vs Students)')
plt.show()


# Professionals are much more inclined towards **Compensations,stability, office proximity and perks**.This is implicit, as after certain time in one's career, people loose the dynamism, and enjoy perky lives. Case of students is bit different, as they are yet to start their careers. Thus they are enthusiastic,and are ready to learn anything that is thrown at them and thus **Compmission,valuation and findings** look valuable to them.

# ### How Did They Learn Coding(Students vs Professionals)

# In[30]:


working=df2[df2['q8Student']=='Not Student']
students=df2[df2['q8Student']=='Students']
working=working[working.columns[working.columns.str.contains('q6LearnCode')]]
working=working.apply(pd.Series.value_counts).melt().dropna().set_index('variable')
students=df2[df2['q8Student']=='Students']
students=students[students.columns[students.columns.str.contains('q6LearnCode')]]
students=students.apply(pd.Series.value_counts).melt().dropna().set_index('variable')
working=working[working.index!='q6LearnCode']
stu_work=working.merge(students,left_index=True,right_index=True,how='left')
stu_work['total']=stu_work['value_x']+stu_work['value_y']
stu_work['%work']=stu_work['value_x']/stu_work['total']
stu_work['%student']=stu_work['value_y']/stu_work['total']
stu_work.drop(['value_x','value_y','total'],axis=1,inplace=True)
stu_work.index=stu_work.index.str.replace('q6LearnCode','')
stu_work.plot.barh(stacked=True,width=0.9)
fig=plt.gcf()
fig.set_size_inches(8,4)
plt.title('Learnt Coding(Professionals vs Students)')
plt.show()


# **Learning at the university nd self-learning** are common ways in which people learncoding, irrespective of being student or working professional. Lets check the other options people use for learning coding,

# In[31]:


wc = WordCloud(background_color="white", max_words=1000, 
               stopwords=STOPWORDS,width=1000,height=1000)
wc.generate(" ".join(df['q0006_other'].dropna()))
plt.imshow(wc)
plt.axis('off')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()


# We see 2 prominent words **JOB and WORK**, where people learn coding. I have seen people from mechanical background with no experience with coding, who work for 2-3 years on their coding work and get well established in tech companies. It is indeed trues that during our work, we get exposed to many new technologies and in process of interpreting and working with it, we learn these stuff, and naturally get better at coding. Thats why **Work Experience** is so crucial and managers often search for experienced people.

# ### Simple Programming Questions Asked During Survey
# 
# The survey had some simple programming questions asked in-between the survey.Let's see how many of them got it right
# 
# **Level 1:What language does the following code snippet belong to:- int ptr = 1;**
# 
# **Ans) C**
# 
# **Level 2:What does this function do?**     
# 
# **def function(n):           for i in range(n):                  print "Hello, World!"**
# 
# **Ans) prints "Hello World!" n times**
# 
# **Level 3:Which of these mean that 'num' is even?**
# 
# **Ans) num%2 == 0**
# 
# **Level 4:Which of the following is useful in traversing a given graph by breadth-first search?**
# 
# **Ans)Queue**

# In[32]:


def correct_answer(ans,col):
    correct=ans
    for i,j in zip(df[col],df.index):
        if i==correct:
            df.loc[j,col]='Correct Answer'
        else:
            df.loc[j,col]='Wrong Answer'
correct_answer('C','q7Level1')
correct_answer('prints "Hello, World!" n times','q15Level2')
correct_answer('num%2 == 0','q31Level3')
correct_answer('Queue','q36Level4')


# In[33]:


ques=['q7Level1','q15Level2','q31Level3','q36Level4']
length=len(ques)
plt.figure(figsize=(10,10))
for i,j in itertools.zip_longest(ques,range(length)):
    plt.subplot((length/2),2,j+1)
    plt.subplots_adjust(wspace=0.8)
    df[i].value_counts().plot.pie(autopct='%1.1f%%',colors=['g','r'],wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })
    plt.title(i)
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.xlabel('')
    plt.ylabel('')
plt.show()



# The results of the first 3 questions look to be similar.However quite a few of them got the last one wrong, probably because it was based on data structures.

# ### HackerRank Reviews

# In[36]:


import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12,8))
gridspec.GridSpec(3,3)

plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.title('Recommend HackerRank')
df[df['q32RecommendHackerRank']!='#NULL!'].q32RecommendHackerRank.value_counts().plot.pie(autopct='%1.1f%%',shadow=True,colors=['g','r'])
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)

plt.subplot2grid((3,3), (0,2))
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.title('Positive Experience')
df['q34PositiveExp'].value_counts().sort_values().plot.barh(width=0.9,color=sns.color_palette('inferno_r'))

plt.subplot2grid((3,3), (1,2))
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.title('Ideal Test Length')
df[df['q34IdealLengHackerRankTest']!='#NULL!'].q34IdealLengHackerRankTest.value_counts().sort_values().plot.barh(width=0.9,color=sns.color_palette('cubehelix_r'))

plt.subplot2grid((3,3), (2,2))
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.title('HackerRank Challenge for Job?')
df[df['q33HackerRankChallforJob']!='#NULL!'].q33HackerRankChallforJob.value_counts().plot.barh(width=0.9,color=sns.color_palette('viridis'))
fig.tight_layout()
plt.subplots_adjust(wspace=0.8,hspace=0.4)
plt.show()


# #### Observations:
# 
#  - HackerRank looks to be recommended by everyone. Thanks to its **clean and beautiful UI** and wide range of problems belonging to different tracks.
#  - As we had seen earlier that not many managers are using HackerRank for their test assessments, a similar thing can be concluded from above. Even though it is so famous and recommended, it is not used much for assessments, and it can be easily seen as not many candidates have ever faced HackerRank assessments.If the HackerRank PR team is anyhow looking at this, I think they should start promoting their product on a larger scale, and maybe give me a small part of it...xd!

# ### Other than HackerRank??

# In[34]:


df[df.columns[df.columns.str.contains('q30LearnCode')]].apply(pd.Series.value_counts).melt().set_index('variable').dropna().sort_values(by='value').plot.barh(width=0.9,color=sns.color_palette('winter_r'))
fig=plt.gcf()
fig.set_size_inches(6,6)
plt.title('Other Learning Sources')
plt.show()


# **StackOverFlow and Youtube** are another medium where people learn to code. It is obvious because we learn by asking our doubts and learning from mistakes. Lets check the free response column..

# In[35]:


wc = WordCloud(background_color="white", max_words=1000, 
               stopwords=STOPWORDS,width=1000,height=1000)
wc.generate(" ".join(df['q0030_other'].dropna()))
plt.imshow(wc)
plt.axis('off')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()


# **Leetcode, Hackerearth, Codechef and geeksforgeeks** are some other online coding sites that look to be very famous among coders, and are the sources from where people learn coding and also compete.

# ## Conclusions :
# 
#  - Only about **20%** of the total respondents are **women**.
#  - **Ghana** has the highest number of respondents followed by India and USA.
#  - Majority of the respondents are in the age group**(18-34)**. It was also good to see that many people had started coding in an early stage i.e between **(11-20)** years.
#  - Majority of the respondents are from the CS background.
#  - **Specific to Women**:
#   - Women over the age of 35 are not much into tech, which means that about 10-15 years back not many women were getting into the tech industry. However it is good to see that many young laldies are now getting into tech world, and many of them have started coding at an early age. Thus the future will see a rise in female coders/programmers.
#   - About **50-60%** of the total women belong to the CS background, which again is a positive trait for women in tech.
#   - About **50%** of the total working women are working in the tech industry, with a majority of them being **Software Engineer and Full-Stack Developers**. However the number of women in **Data Science is very less.** However this condition may change in the near future as many women are looking to learn ML/AI in the future.
#   - Not many women are at a Senior Position, majority of them being **Level 1 developers.**
# 
# - **Recruiment Point of View:**
#  - **Java, JavaScript , AngularJs, Python and SQL** are some of the most common and important technological skills that recruiters search for.
#  - Apart from programming languages, **Problem Solving** skills is the most important skill that a candidate should have.
#  - **Work Experience and Projects** are also very important things that recruiters look for.
#  
# - **C,C++ and Java** are the most familiar languages to developers, among both **students and professionals**. However many students and professionals, who are not familiar with these languages, are still not willing to learn them as compared to other languages like **Python, Scala and Go**. The **popularity for dynamic and functional programming languages is on a rise**, and the demand for such talent will surely grow in the future. These languages are famous among both students and working professionals.
# - **Python is the most loved and least hated** language. This makes Python as a must for almost any person belonging to the technical domain.
# - **Security, AI/ML/Data Science and Cloud  Computing** are some very prominent domains  the future.
# 
# ## **Something For HackerRank People**:
#  - HackerRank has very positive reviews and ratings from its users. However not many recruiters use HackerRank for their assessment of new candidates. Thus HackerRank must do something so that their amazing platform is widely used by managers for recruitment purpose.

# Thanks a lot for having a look at this notebook. If u liked the notebook or learnt something from it, **PLEASE UPVOTE!!!**
# 
# ### Thank You
