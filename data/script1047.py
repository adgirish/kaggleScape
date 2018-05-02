
# coding: utf-8

# 
# 
# <h1 style="font-size:30px;color:black;text-align:center; text-decoration:underline">                   An analysis of the most popular repos on Github[completed]</h1>
# <img src="https://kanbanize.com/blog/wp-content/uploads/2014/11/GitHub.jpg" alt="github" width=50% height=50%>
# <hr>
# _Areas of focus include: Type of repo,Metrics of popularity, Languages used_
# 
# _Data Source: https://www.kaggle.com/chasewillden/topstarredopensourceprojects _
# 
# #### Objective of this analysis:
# <br>
# <ol>
# <li>Learning how to read and analyse a dataset</li>
# <li>Understanding the dominant languages used for popular GitHub projects and mapping them</li>
# <li>Extracting the different domains of work done in these projects via the repositories tags</li>
# <li>Deriving conclusions over the popularity of respective domains and languages</li>
# </ol>
# <hr>
# **All text in blue represents a derived conclusion**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from ggplot import *
plt.style.use('default')


# In[ ]:


git_df = pd.read_csv("../input/TopStaredRepositories.csv", parse_dates=['Last Update Date'], dayfirst=True)
git_df.head()


# In[ ]:


git_df.info()


# <h2 style="text-decoration:underline">1. Popular Repositories</h2>
# <br>
# **Determining what constitutes a popular repository by extracting the range of maximum and minimum starred repositories**

# <em>Converting the alphanumeric format of "Number of Stars" to numeric

# In[ ]:


git_df_max = git_df['Number of Stars'].str.contains('k').all()
git_df_max


# In[ ]:


git_df['Number of Stars']=git_df['Number of Stars'].str.replace('k','').astype(float)


# ### 1.1 The top 5 repositories in the dataset [based on "Number of Stars"]

# In[ ]:


git_df.head()


# ### 1.2 The bottom 5 repositories in the dataset

# In[ ]:


git_df.tail()


# ### 1.3 Statistical analysis of the repositories

# In[ ]:


git_df['Number of Stars'].describe()


# <h4 style="color:blue">Most popular repo- 290,000 stars</h4>
# <h4 style="color:blue"> Least popular repo- 6400 stars</h4>
# <h4 style="color:blue">Average rating of repos- 13,000 stars</h4>

# ### 1.3 List of all repositories with stars > 13,000

# In[ ]:


popular_repos= git_df[git_df['Number of Stars'] > 13.0]
len(popular_repos)


# In[ ]:


popular_repos.head(8)


# In[ ]:


popular_repos.tail(8)


# 
# <h3 style="color:blue"> Here we see that freeCodeCamp tops the list with 290,000 stars to its repository.</h3>
# <hr>
# ### The above list lists all repos that have > 13,000 stars [above average repositories]
# #### A few more observations can be derived from this list:
# <ol>
# <li style="color:blue">5 of the most popular repos are frameworks</li>
# <li style="color:blue">The third, sixth and eighth most popular repos are educational, and instructive in nature.</li>
# </ol>

# In[ ]:


# classifying repositories according to the popularity
classified_repos=[]
for i in range(8,300,7):
    x = git_df[(git_df['Number of Stars'] >= i) & (git_df['Number of Stars'] <(i+7.0))]
    classified_repos.append(len(x))


# In[ ]:


indexes = []

for i in range (8000,300000, 7000):
    x = '[' + str(i) +','+ (str(i+7000)) + ')'
    indexes.append(x)


# In[ ]:


divided_repos = pd.Series(data=classified_repos, index=indexes)
divided_repos.plot(kind='bar', figsize=(15,10), color=['red'],legend=True, label='Number of repositories')


# <h2 style="text-decoration:underline">2. Popular Languages</h2>
# <br>
# **Determining the popularity of a language based on the number of repositories using it.**

# In[ ]:


x=git_df['Language'].value_counts()
x.head()
#p = ggplot(aes(x='index',y='count'), data =x) + geom_point(color='coral') + geom_line(color='red')
#print(p)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure()
x.plot(kind='barh',figsize=(15,10),grid=True, label='Number of repositories',legend='No of repos',title='No of repositories vs language used')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
x[:5].plot.pie(label="Division of the top 5 languages",fontsize=10,figsize=(10,10),legend=True)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
x[:20].plot.pie(label="Division of the top 20 languages",fontsize=10,figsize=(10,10),legend=True)


# <h2 style="text-decoration:underline">3. Popular Domains</h2>
# <br>
# **Determining the popular domains by analysing the repository tags**

# <em>Removing all the null-tags fields from the dataframe

# In[ ]:


#git_df['Number of Stars']=git_df['Number of Stars'].str.replace('k','').astype(float)
nonull_df = git_df[['Tags','Number of Stars']].dropna()
tags_list = nonull_df['Tags'].str.split(',')


# In[ ]:


tags_list.head()


# In[ ]:


initial = nonull_df['Tags'].str.split(',')
a = []
for item in initial:
       a = a+item
wc_text = ' '.join(a)

get_ipython().run_line_magic('matplotlib', 'inline')
wordcloud = WordCloud(background_color='black',width=800, height=400).generate(wc_text)
plt.figure(figsize=(25,10), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.tight_layout(pad=0)
plt.axis("off")


# In[ ]:


web_dev_count = 0
tags = ['javascript', 'css', 'html', 'nodejs', 'bootstrap','react', 'react-native', 'rest-api', 'rest', 'web-development','typescript','coffeescript']
for item in tags_list:
    if set(tags).intersection(item):
        web_dev_count+=1
web_dev_count


# In[ ]:


machine_data_count=0
mach=[]
tags=['machine-learning', 'jupyter','jupter-notebook', 'tensorflow','data-science','data-analytics']
for item in tags_list:
    if set(tags).intersection(item):
        machine_data_count+=1
        mach.append(item)
machine_data_count


# In[ ]:


mobile_dev_count=0
tags=['android','sdk','ios','swift','mobile','react','macos','windows']
for item in tags_list:
    if set(tags).intersection(item):
        mobile_dev_count+=1
mobile_dev_count


# In[ ]:


linux_dev_count=0
linux=[]
tags=['linux','unix','bash','shell','cli','bsd']
for item in tags_list:
    if set(tags).intersection(item):
        linux_dev_count+=1
        linux.append(item)
linux_dev_count


# In[ ]:


hardware_dev_count=0
hardware=[]
tags=['hardware','iot','smart','system','system-architecture','cloud']
for item in tags_list:
    if set(tags).intersection(item):
        hardware.append(item)
        hardware_dev_count+=1
hardware_dev_count


# In[ ]:


domain_series=pd.Series(index=['Web Development','Data Science and Machine Learning','Mobile Development','Linux and Shell Programming','System hardware and IOT'],
                        data=[web_dev_count,machine_data_count,mobile_dev_count,linux_dev_count,hardware_dev_count])


# In[ ]:


domain_series


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig_domain=domain_series.plot(lw=2,kind='barh',figsize=(20,10),color=['green'],grid=True,title='Domain-wise repository analysis',
                              )
fig_domain.set(xlabel="Number of repositories", ylabel="Domain Name")


# <h2 style="text-decoration:underline">3. Determing the correlation between Number of Tags and Number of Stars</h2>

# In[ ]:


nonull_df['CountTag']=0
for i in range(0,489,1):
    nonull_df['CountTag'].iloc[i] = len(list(nonull_df['Tags'].iloc[i].split(',')))


# In[ ]:


nonull_df['CountTag'].corr(nonull_df['Number of Stars'])


# <hr>
# <h2 style="text-decoration:underline">4. Conclusion</h2>
# <br>
# **Inferences from the analysis**
# <hr>
# <ol>
# <li>The most popular repository on GitHub is freeCodeCamp, with 290,000 stars</li>
# <li>In the top 8 repositories in the dataset, 3 are instructional and educational</li>
# <li>JavaScript is the most popularly used language, and constitutes <b>38.5 %</b> of the total languages in these repositories</li> 
# <li>Frameworks are the most popular type of projects across GitHub</li>
# <li>In domains, Web Development is the most popular domain of work, followed by Mobile (android, iOS, macOS, Windows) development</li>
# <li>There is no determinable correlation between the number of tags and the number of stars. The correlation coefficient is a weak 0.04646</li>

# ### [Extra] Popular python based projects 

# In[ ]:


python_tags = git_df[git_df['Language'] == 'Python'][['Username', 'Repository Name', 'Description', 'Tags']]


# In[ ]:


python_tags

