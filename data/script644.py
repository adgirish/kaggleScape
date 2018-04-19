
# coding: utf-8

# 
# ![Hackerrank-code-like-a-girl](https://camo.githubusercontent.com/bcb153b5a4eaa2bf3f97776188c6d0d9f2ff6ce5/68747470733a2f2f64336b65757a6562326372686b6e2e636c6f756466726f6e742e6e65742f6861636b657272616e6b2f6173736574732f7374796c6567756964652f6c6f676f5f776f72646d61726b2d66356335656236316162306131353463336564396564613234643062396533312e737667)
# 
# Hackerrank is a programming community platform for coders ! It helds competition and programming challenges to brush up/hone coding skills in various languages  (including Java, C++, PHP, Python, SQL, JavaScript)  ! Not unlike Kaggle which is focused on Data Scientist/Machine Learning engineers, Hackerrank is a good way to practice and show your skills to potential employers.
# It is part of the growing gamification trend within competitive computer programming. We could ask ourselves what insights about women in tech the data provided by Hackerrank survey reveal !
# 
# As a young 2017 Graduate in Computer Science and Data Science and Woman in Tech myself, I am curious to see which trends we'll uncover :) Plus, I also wanted to gain more experience in data viz with Python. ^^
# 
# **RECAP **
# The data set we are releasing here is the full dataset of 25K responses from Hackerrank developer survey, which includes both students and professionals.
# 
# **Methodology for the survey **
# * A total of 25,090 professional and student developers completed our 10-minute online survey.
# * The survey was live from October 16 through November 1, 2017.
# * The survey was hosted by SurveyMonkey and we recruited respondents via email from our community of over 3.4 million members and through social media sites.
# * We removed responses that were incomplete as well as obvious spam submissions.
# * Not every question was shown to every respondent, as some questions were specifically for those involved in hiring. The codebook (HackerRank-Developer-Survey-2018-Codebook.csv) highlights under what conditions some questions were shown.
# * The Women In Tech 2018 report is based only on the 14K responses from professionals
# * Respondents who identified as students (q8Student=1; N=10351) were excluded from this report.
# * Respondents who identify as “non-binary” (q3Gender=3; N=76) were excluded from the male-female comparisons.
# 
# 
# 
# **Women in Tech**
# 
# 
# We know that Women in Tech are a minority, but what is the current situation in the past years ? More and more countries are putting effort into making women go into tech, has the situation improved from the past ? Let us get more in depth with this quick survey dataset !
# 
# ![](https://i1.wp.com/nmtechcouncil.org/wp-content/uploads/cover-graphic.jpeg?resize=700%2C367&ssl=1)
# 
# Summary
# 
# * [Q1 - Which languages are the most popular ?](#Q1)
# * [Q2 - Age distribution ?](#Q2)
# * [Q3 - At which age do they begin coding, differences between genders ?](#Q2)
# * [Q4 - Countries of Respondents ?](#Q3)
# * [Q5 - Top countries characteristics - age began coding ?](#Q5)
# 
# 
# 
# 

# In[3]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import plotly
import  plotly.offline as py
py.init_notebook_mode(connected=True)
#import plotly.plotly as py
import plotly.graph_objs as go


# In[4]:


df=pd.read_csv('../input/HackerRank-Developer-Survey-2018-Values.csv', parse_dates=['StartDate','EndDate'])
df_n = pd.read_csv('../input/HackerRank-Developer-Survey-2018-Numeric.csv', parse_dates=['StartDate','EndDate'])
df_women = df[df.q3Gender == 'Female']
df_men = df[df.q3Gender != 'Female']


# In[5]:


df.shape


# In[6]:


df = df.dropna(axis=0, how='all')
df.shape


# In[7]:


#c = 0 
#for i in df.columns : 
#    print(i + " "+ str(c))
#    c+= 1

df.head(1)


# <a id='Q1'></a>
# ## **Let's explore which languages are the most popular amongst the respondents classified by gender ! **
# 
# I will go back to think about this section later...

# In[8]:


prog = df[df.columns[139:163]]
prog['Gender'] = df['q3Gender']
prog = prog.dropna(axis=0, how='all')
prog.columns


# In[9]:


prog[0:5]


# In[59]:


for i in prog.columns[:-1] :
    print(i + ": "+str(prog[i].isnull().sum()))


# In[84]:


colors = ["blue", "orange", "greyish", "faded green", "dusty purple"]
fig, ax = plt.subplots(figsize=(20,20), ncols=5, nrows=5)
count = 0
times = 0
for i in prog.columns[:-1]:
    #sns.regplot(x='value', y='wage', data=df_melt, ax=axs[count])
    sns.countplot(x=str(i), hue="Gender", data=prog, palette = sns.xkcd_palette(colors), ax=ax[times][count])
    count += 1
    if count == 5 :
        times += 1
        count = 0

    


# **To be continued**

# <a id='Q2'></a>
# # Let's see how many women there are and the age distribution for both. The AgeBeginCoding value might also be interesting 

# In[43]:


trace1 = go.Bar(
    x=df_men['q2Age'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_men['q2Age'].value_counts().tolist(),np.sum(df_men['q2Age'].value_counts().tolist())).tolist(),100).tolist(),
    name='Men Respondents'
)
trace2 = go.Bar(
    x=df_women['q2Age'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_women['q2Age'].value_counts().tolist(),np.sum(df_women['q2Age'].value_counts().tolist())).tolist(),100).tolist(),
    name='Female Respondents'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# In[44]:


trace1 = go.Bar(
    x=df_men['q1AgeBeginCoding'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_men['q1AgeBeginCoding'].value_counts().tolist(),np.sum(df_men['q1AgeBeginCoding'].value_counts().tolist())).tolist(),100).tolist(),
    name='Men Respondents'
)
trace2 = go.Bar(
    x=df_women['q1AgeBeginCoding'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_women['q1AgeBeginCoding'].value_counts().tolist(),np.sum(df_women['q1AgeBeginCoding'].value_counts().tolist())).tolist(),100).tolist(),
    name='Female Respondents'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# ## We can see that women tend to learn later on compared to men, especially regarding the "11-15 years-old" (22% for men and 13.8% for women) begineers category. More than the half of women learn between 16-20 years old. 

# In[45]:


#df['time']=(df['EndDate']-df['StartDate']).astype('timedelta64[m]')


# <a id='Q3'></a>
# # Let's draw a global map to see from where are the majority of our respondents

# In[46]:


focus_country = df['CountryNumeric'].value_counts().to_frame()
print("our TOP 10 country respondents is :") 
print(focus_country.head(10).index)


# In[47]:


data = [ dict(
        type = 'choropleth',
        locations = focus_country.index,
        locationmode = 'country names',
        z = focus_country['CountryNumeric'],
        text = focus_country['CountryNumeric'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 1
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Respondents'),
      ) ]

layout = dict(
    title = 'Number of respondents by country',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )


# Source here : https://plot.ly/python/choropleth-maps/

# ### **It's surprising to see Ghana winning the race, a map of beginning of code per country would be useful to see if every country needs to put on efforts (?) I will also explore the career/ school degrees and specialty of the individuals #To follow**

# <a id='Q5'></a>
# ## **Let's see the age at which the top countries respondents learned to code **

# In[83]:


df_men_c = [0,0,0]
df_women_c = [0,0,0]
count = 0
for i in focus_country.head(3).index : 
    df_men_c[count] = df_men[df_men['CountryNumeric'] == i]
    df_women_c[count] = df_women[df_women['CountryNumeric'] == i]
    print('N° of Male respondents for '+ i + ' is : '+ str(df_men_c[count].shape[0]))
    print('N° of Female respondents for '+ i + ' is : '+ str(df_women_c[count].shape[0]))
    
    trace1 = go.Bar( 
    x=df_men_c[count]['q1AgeBeginCoding'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_men_c[count]['q1AgeBeginCoding'].value_counts().tolist(),np.sum(df_men_c[count]['q1AgeBeginCoding'].value_counts().tolist())).tolist(),100).tolist(),
    name='Men Respondents in '+i
    )
    trace2 = go.Bar(
    x=df_women_c[count]['q1AgeBeginCoding'].value_counts().index.tolist(),
    y=np.multiply(np.divide(df_women_c[count]['q1AgeBeginCoding'].value_counts().tolist(),np.sum(df_women_c[count]['q1AgeBeginCoding'].value_counts().tolist())).tolist(),100).tolist(),
    name='Female Respondents in '+i
    )

    data = [trace1, trace2]
    layout = go.Layout(
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='grouped-bar')
    count = count + 1


# We observe that most people learn to code between 16 and 20 years old. However, we also notice that in India the 2nd most represented group of beginners is 21-25 years old ! that is not the case in Ghana and USA where 2nd most seems to be 11-15 years. however girls are underrepresented in the USA for the 11-15 years old category. Maybe USA and India should put effort to make them learn to code earlier ?

# ## Let's see if people who started to code continued. to be continued :) Don't hesitate to comment and upvote if you liked this kernel !
