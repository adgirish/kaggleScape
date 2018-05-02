
# coding: utf-8

# # H1B Applications Exploration
# 
# ![](http://d2r2ijn7njrktv.cloudfront.net/apnlive/uploads/2017/01/H1B.jpg)
# 
# ### What is H1B Visa?
# 
# The H1B visa is an employment-based, non-immigrant visa for temporary workers. For this visa, an employer must offer a job in the US and apply for your H1B visa petition with the US Immigration Department. This approved petition is a work permit which allows you to obtain a visa stamp and work in the U.S. for that employer. The H1B visa is issued for a specialty occupation, requires theoretical and practical application of a body of specialized knowledge and requires the visa holder to have at least a **Bachelors degree** or its equivalent. As per my knowledge, majority of the H1B holders are from Asia especially from **India and China**, not because we asian's are talented by birth..:D, but due to the availability of **cheap labour**.
# 
# But recently there has been many issues with the H1B visa and many new regulations have been enforced on it, so that only deserving candidates can acquire a H1B visa. Lets not go into that and check the given dataset. So this dataset has records from H1B visa applications for the years **2011-2016**. The only thing I was a bit sad about was the unavailabililty of the source country, i.e the country of origin of the applicant.Lets not crave about it and start with our analysis.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import folium
import folium.plugins
from folium import IFrame
from mpl_toolkits.basemap import Basemap
from IPython.display import HTML
import io
import base64
from matplotlib import animation,rc
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from IPython.display import HTML
import io
import base64
from matplotlib import animation,rc
import os
print(os.listdir("../input"))


# In[2]:


df=pd.read_csv('../input/h1b_kaggle.csv')


# In[3]:


df.head()


# In[4]:


print('Total H1B cases:',df.shape[0])


# So this dataset has more than **3 million** cases for H1B applications.

# In[5]:


df['SOC_NAME']=df['SOC_NAME'].str.upper()
def state(data):
    states = []
    data_worksite = df['WORKSITE']

    for worksite in data_worksite.iloc[:]:
        state = worksite.split(', ')[1]
        states.append(state)
    return pd.DataFrame(states, columns=['State'])

states=state(df)
df['State']=states['State']


# ## Some Basic Analysis

# In[6]:


plt.figure(figsize=(10,8))
ax=df['CASE_STATUS'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color='#ffd700')
for i, v in enumerate(df['CASE_STATUS'].value_counts().sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='r',weight='bold')
plt.title('Case Status for All Years')
plt.show()


# In[7]:


plt.figure(figsize=(12,6))
df['YEAR'].value_counts().sort_values().plot(marker='o')
plt.title('H1B Applicants by Year')
plt.xlim([2010,2017])
plt.show()


# The number of H1B visa applications are growing tremendously, with a spurt after 2013. But do all these appliactions get certified? Lets check..

# In[8]:


cer_den=df[df['CASE_STATUS'].isin(['CERTIFIED','DENIED'])]
cer_den=cer_den.groupby(['YEAR','CASE_STATUS'])['JOB_TITLE'].count().reset_index()
cer_den.pivot('YEAR','CASE_STATUS','JOB_TITLE').plot.bar(width=0.8)
fig=plt.gcf()
fig.set_size_inches(15,6)
plt.title('Case Status by Year')
plt.show()


# Everyone says that **USA** is the largest hotspot for jobs, and with the growing number of startups, the demand for talented people is also increasing. It is evident from the above graph that the number of certifies applications are growing, whereas the number of denied applications are diminshing.

# ### Regions with Highest Applications

# In[9]:


appli=df.groupby(['lat','lon'])['Unnamed: 0'].count().reset_index()
appli.columns=[['lat','lon','applications']]
locate=appli[['lat','lon']]
count=appli['applications']
def color_producer(elevation):
    if elevation < 1000:
        return 'red'
    elif 1000 <= elevation < 3000:
        return 'orange'
    else:
        return 'green'
map1 = folium.Map(location=[39.50, -98.35],tiles='CartoDB dark_matter',zoom_start=3.5)
for point in appli.index:
    folium.CircleMarker(list(locate.loc[point].values),popup='<b>No of Applications:</b>'+str(count.loc[point].values[0]),radius=count.loc[point].values[0]*0.0002,color=color_producer(count.loc[point].values[0]),fill_color=color_producer(count.loc[point].values[0]),fill=True).add_to(map1)
map1


# The eastern coast has the highest number of applications, which covers states like **NY,NJ**,etc.

# ## Inspiration 1: Which employers file the most petitions each year?

# In[10]:


plt.figure(figsize=(10,8))
ax=df['EMPLOYER_NAME'].value_counts().sort_values(ascending=False)[:10].plot.barh(width=0.9,color='#ffd700')
for i, v in enumerate(df['EMPLOYER_NAME'].value_counts().sort_values(ascending=False).values[:10]): 
    ax.text(.8, i, v,fontsize=12,color='r',weight='bold')
plt.title('Highest Employeer')
fig=plt.gca()
fig.invert_yaxis()
plt.show()


# I wasn't at all surprised to see the Top 3 employeers..:D. Anyone from India will agree that the Top 3 emoloyeers are the biggest recruiters in India, and thus they have the highest outsourcing powers. Lets check the number of certified and denied applications for these Top employeers.

# In[11]:


comp_den=df[df['CASE_STATUS'].isin(['CERTIFIED','DENIED'])]
comp_den=comp_den[comp_den['EMPLOYER_NAME'].isin(comp_den['EMPLOYER_NAME'].value_counts().sort_values(ascending=False)[:10].index)]
comp_den=comp_den.groupby(['EMPLOYER_NAME','CASE_STATUS'])['JOB_TITLE'].count().reset_index()
comp_den=comp_den.pivot('EMPLOYER_NAME','CASE_STATUS','JOB_TITLE')
plt.figure(figsize=(25,10))
plt.scatter('CERTIFIED','DENIED',data=comp_den,s=comp_den['CERTIFIED']*0.03)
for i in range(comp_den.shape[0]):
    plt.text(comp_den['CERTIFIED'].values[i],comp_den['DENIED'].values[i],s=comp_den.index[i],color='r',weight='bold')
plt.title('Status Certified vs Denied',size=30)
plt.xlabel('CERTIFIED')
plt.ylabel('DENIED')
plt.show()


# It is good to see that even with such high number of applications, Infosys has very few cases of **Denied** visa's. But the above graph has employeers with highest applications ,and it is possible that employeers with lesser applications have higher Certification rates.Lets check which all employeers have the highest **Certification rates.**

# In[12]:


emp_rate1=df[df['CASE_STATUS']=='CERTIFIED']
emp_rate1=emp_rate1.groupby(['EMPLOYER_NAME','CASE_STATUS'])['YEAR'].count().reset_index()
emp_rate2=df[df['CASE_STATUS']=='DENIED']
emp_rate2=emp_rate2.groupby(['EMPLOYER_NAME','CASE_STATUS'])['YEAR'].count().reset_index()
aa1=emp_rate2.sort_values('YEAR',ascending=False)[:100]
aa2=emp_rate1.sort_values('YEAR',ascending=False)[:100]
aa3=aa2.merge(aa1,left_on='EMPLOYER_NAME',right_on='EMPLOYER_NAME',how='left').dropna()
aa3['Acceptance_rate']=aa3['YEAR_x']/(aa3['YEAR_x']+aa3['YEAR_y'])
aa3.sort_values('Acceptance_rate',ascending=False)[['EMPLOYER_NAME','Acceptance_rate']][:10]


# **Microsoft** has the highest acceptance or certification rate.

# ### Employeer Applications by Year

# In[13]:


emp_year=df[df['EMPLOYER_NAME'].isin(df['EMPLOYER_NAME'].value_counts().sort_values(ascending=False)[:5].index)]
emp_year=emp_year.groupby(['EMPLOYER_NAME','YEAR'])['CASE_STATUS'].count().reset_index()
emp_year.pivot('YEAR','EMPLOYER_NAME','CASE_STATUS').plot.bar(width=0.7)
fig=plt.gcf()
fig.set_size_inches(15,8)
plt.show()


# ## Wage Distribution
# 

# In[14]:


plt.figure(figsize=(12,6))
df[df['PREVAILING_WAGE']<150000].PREVAILING_WAGE.hist(bins=40,color='khaki')
plt.axvline(df[df['PREVAILING_WAGE']<=150000].PREVAILING_WAGE.median(), color='green', linestyle='dashed', linewidth=4)
plt.title('Wage Distribution')
plt.show()


# There are many outliers in the prevailing wages, with values ranging up to billion dollars. The distribution is skewed towards the right. These are likely to bias the mean estimate. Thus we checked the **median** wage, which gives a better estimate as compared to the **mean** wage. Lets now split the wages by their CASE_STATUS.

# In[15]:


plt.figure(figsize=(12,6))
df[(df['PREVAILING_WAGE']<150000)&(df['CASE_STATUS']=='CERTIFIED')].PREVAILING_WAGE.hist(bins=50, color="lightgreen", alpha=0.7, label='Certified', normed=True)
plt.axvline(df[(df['PREVAILING_WAGE']<=150000)&(df['CASE_STATUS']=='CERTIFIED')].PREVAILING_WAGE.median(), color='green', linestyle='dashed', linewidth=4)
df[(df['PREVAILING_WAGE']<150000)&(df['CASE_STATUS']=='DENIED')].PREVAILING_WAGE.hist(bins=50,color="tomato", alpha=0.7, label='Denied', normed=True)
plt.axvline(df[(df['PREVAILING_WAGE']<=150000)&(df['CASE_STATUS']=='DENIED')].PREVAILING_WAGE.median(), color='red', linestyle='dashed', linewidth=4)
plt.legend()
plt.show()


# The median wage for Denied Cases is lower than that of Certified Cases. This might be a reason for Application denial, the applications might have a lower wage than required.

#  ### Lowest Median Salaries by Top Employeers
#  
#  As suggested by Pranav Pandya Sir, lets check which employeer with high applications offers the lowest median wages.

# In[16]:


high_emp=df['EMPLOYER_NAME'].value_counts().sort_values(ascending=False)[:20].to_frame()
df[df['EMPLOYER_NAME'].isin(high_emp.index)&(df['PREVAILING_WAGE']<=150000)].groupby(['EMPLOYER_NAME'])['PREVAILING_WAGE'].median().to_frame().sort_values(by='PREVAILING_WAGE')


# ## Inspiration 2: Is the number of petitions with Data Engineer job title increasing over time?

# In[17]:


data_peeps=df.dropna(subset=['JOB_TITLE'])
data_peeps=data_peeps[data_peeps['JOB_TITLE'].str.contains('DATA')]
data_scientists=data_peeps[data_peeps['JOB_TITLE'].str.contains('DATA SCIENTIST')]
data_analyst=data_peeps[data_peeps['JOB_TITLE'].str.contains('DATA ANALYST')]
data_eng=data_peeps[data_peeps['JOB_TITLE'].str.contains('DATA ENG')]


# In[18]:


f,ax=plt.subplots(1,3,figsize=(22,8))
data_scientists.groupby('YEAR')['CASE_STATUS'].count().plot(ax=ax[0],marker='o')
data_analyst.groupby('YEAR')['CASE_STATUS'].count().plot(ax=ax[1],marker='o')
data_eng.groupby('YEAR')['CASE_STATUS'].count().plot(ax=ax[2],marker='o')
data_scientists[data_scientists['CASE_STATUS']=='CERTIFIED'].YEAR.value_counts().plot(marker='o',ax=ax[0])
data_analyst[data_analyst['CASE_STATUS']=='CERTIFIED'].YEAR.value_counts().plot(marker='o',ax=ax[1])
data_eng[data_eng['CASE_STATUS']=='CERTIFIED'].YEAR.value_counts().plot(marker='o',ax=ax[2])
for i,j in zip([0,1,2],['Applications for Data Scientists','Applications for Data Analysts','Applications for Data Engineers']):
    ax[i].set_title(j)
for i in [0,1,2]:
    ax[i].set_xlim([2010,2017])
plt.show()


# It is indeed True. The number of applications related to **Data Jobs** are increasing tremendously. Also the red line denotes the **number of Certified Applications**. The number of applications are almost going up by almost **100%**. As mentioned earlier that USA is a hotspot for jobs, this graph will probably keep growing at a similar rate.

# ### Salary for Data Related Jobs

# In[19]:


f,ax=plt.subplots(1,3,figsize=(22,8))
data_scientists[data_scientists['PREVAILING_WAGE']<150000].groupby(['YEAR'])['PREVAILING_WAGE'].median().plot(ax=ax[0],marker='o')
data_analyst[data_analyst['PREVAILING_WAGE']<150000].groupby(['YEAR'])['PREVAILING_WAGE'].median().plot(ax=ax[1],marker='o')
data_eng[data_eng['PREVAILING_WAGE']<150000].groupby(['YEAR'])['PREVAILING_WAGE'].median().plot(ax=ax[2],marker='o')
for i,j in zip([0,1,2],['Salary for Data Scientists','Salary for Data Analysts','Salary for Data Engineers']):
    ax[i].set_title(j)
for i in [0,1,2]:
    ax[i].set_xlim([2010,2017])
plt.show()


# The median wage we saw was around **65k$ **. Thus **Data Scientists and Data Engineers** enjoy good salary benefits, however it is not true for **Data Analysts**. The salary for **Data Scientists** has dropped in the past few years, but still above median wage. The wage for **Data Engineers** has increased tremendously over the years.

# In[20]:


f,ax=plt.subplots(figsize=(18,8))
plt.boxplot([data_scientists[data_scientists['PREVAILING_WAGE']<200000].PREVAILING_WAGE,data_analyst[data_analyst['PREVAILING_WAGE']<200000].PREVAILING_WAGE,data_eng[data_eng['PREVAILING_WAGE']<200000].PREVAILING_WAGE])
ax.set_xticklabels(['Data Scientists','Data Analysts','Data Engineer'])
ax.set_title('Salary Distribution')
plt.show()


# ### Highest Paying Employeers

# In[21]:


plt.figure(figsize=(10,8))
ax=data_scientists.groupby('EMPLOYER_NAME')['PREVAILING_WAGE'].median().sort_values(ascending=False)[:10].plot.barh(width=0.9,color='#ffd700')
for i, v in enumerate(data_scientists.groupby('EMPLOYER_NAME')['PREVAILING_WAGE'].median().sort_values(ascending=False)[:10].values): 
    ax.text(.8, i, v,fontsize=12,color='r',weight='bold')
plt.title('Highest Paying Employeers for Data Scientists in $')
fig=plt.gca()
fig.invert_yaxis()
plt.show()
plt.show()


# In[23]:


sal_state=data_scientists.groupby(['lat','lon','State'])['PREVAILING_WAGE'].median().sort_values(ascending=False).reset_index()
appli=df.groupby(['lat','lon'])['Unnamed: 0'].count().reset_index()
locate=sal_state[['lat','lon']]
sal=sal_state['PREVAILING_WAGE']
state=sal_state['State']
def color_producer(elevation):
    if elevation < 75000:
        return 'red'
    elif 75000 <= elevation < 100000:
        return 'orange'
    else:
        return 'green'
map1 = folium.Map(location=[39.50, -98.35],tiles='CartoDB dark_matter',zoom_start=3.5)
for point in sal_state.index:
    folium.CircleMarker(list(locate.loc[point]),popup='<b>Average Salary in $: </b>'+str(sal.loc[point])+"<br><b> State: "+str(state.loc[point]),radius=sal.loc[point]*0.0001,color=color_producer(sal.loc[point]),fill_color=color_producer(sal.loc[point]),fill=True).add_to(map1)
map1


# **California** has the highest median salary for Data Scientists. The salary goes as low as 60k$. The salary range can be a result of many factors, like the experience of the candidate, organisational strength,etc. Also one thing to note is the Data Scientists positions are highly concentrated about the east and the west coast.

# ## Denied Cases By States

# In[24]:


def2=df[(df['CASE_STATUS']=='DENIED')&(df.State.isin(df[df['CASE_STATUS']=='DENIED'].State.value_counts()[:10].index))].groupby(['YEAR','State'])['Unnamed: 0'].count().reset_index()
fig=plt.figure(figsize=(20,8))
def animate(Year):
    ax = plt.axes()
    ax.clear()
    plt.scatter('State','Unnamed: 0',data=def2[def2['YEAR']==Year],s=def2['Unnamed: 0'])
    plt.title('Year: '+str(Year),size=30)
    plt.xlabel('CERTIFIED')
    plt.xlabel('DENIED')
    plt.ylim([0,6500])
ani = animation.FuncAnimation(fig,animate,list(def2.YEAR.unique()), interval = 1500)    
ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# The gif shows the states with highest number of Denied Cases over the years. The trend for California looks to be the best, as we see that the Denied cases hve dropped down from **5000+ cases in 2011**, to something near **2000 cases in the years (2013-2016)**. Similarly the denied cases have drastically gone down for New-York.
# 
# ##  Inspiration 3: Which industry has the most number of Data Scientist positions?

# In[22]:


plt.figure(figsize=(10,8))
data_coun=data_scientists['EMPLOYER_NAME'].value_counts()[:10]
ax=sns.barplot(y=data_coun.index,x=data_coun.values,palette=sns.color_palette('inferno',10))
for i, v in enumerate(data_coun.values): 
    ax.text(.5, i, v,fontsize=15,color='white',weight='bold')
plt.title('Companies Hiring Data Scientists')
plt.show()


# Since we don not have any column denoting the type of industry, we check the company hiring the highest number of data scientists. Seeing the Top 10 companies, we can say that majorly the **Tech Industry** has been hiring Data Scientists.

# ## Inspiration 4: Which part of the US has the most Hardware Engineer jobs?

# In[25]:


hardware=df.dropna(subset=['JOB_TITLE'])
hardware=hardware[hardware['JOB_TITLE'].str.contains('HARDWARE ENGINEER')]
hardware=hardware.groupby(['lat','lon','State'])['Unnamed: 0'].count().reset_index()
locate=hardware[['lat','lon']]
count=hardware['Unnamed: 0']
state=hardware['State']
def color_producer(count):
    if count < 10:
        return 'red'
    elif 10 <= count < 100:
        return 'orange'
    else:
        return 'green'
map1 = folium.Map(location=[39.50, -98.35],tiles='CartoDB dark_matter',zoom_start=3.5)
for point in hardware.index:
    folium.CircleMarker(list(locate.loc[point].values),popup='<b>Number of Applications: </b>'+str(count.loc[point])+"<br><b> State: "+str(state.loc[point]),radius=count.loc[point]*0.01,color=color_producer(count.loc[point]),fill_color=color_producer(count.loc[point]),fill=True).add_to(map1)
map1


# Unlike the Software jobs that are spread out in the eastern and the western coast, ** Hardware Engineering** jobs are mainly focused on the west coast, specially **California**.
# 
# I would stop my analysis here, but there is still lot many things that can be done with this dataset. Suggestions and Feedback are always appreciated.
# 
# ### Thank You
