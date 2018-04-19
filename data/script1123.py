
# coding: utf-8

# # Introduction
# Alcohol is being consumed by students and the consequences of drinking alcohol on academic performance has not been fully undrestoon yet. Here, we compare and contrast alchol consumtion among male and female studenst and the effects of drinking alcholol on their academic performance.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

matplotlib.style.use('fivethirtyeight')
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['xtick.major.pad']='10'
matplotlib.rcParams['ytick.major.pad']='10'
#print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


df_list = [pd.read_csv('../input/student-%s.csv'%course) for course in ['mat', 'por']]
df_list[0]['class'] = 'mat'
df_list[1]['class'] = 'por'
df = df_list[0].append(df_list[1])
F_count = df.groupby(['sex'])['school'].count()[0]
M_count = df.groupby(['sex'])['school'].count()[1]
#df.info()


# In[ ]:


matplotlib.style.use('fivethirtyeight')
fig, axes = plt.subplots(3,1,figsize=(12,12))

df_temp = df.groupby(['class','sex']).size().to_frame(name='Count').reset_index()
df_temp2 = df_temp.pivot(index='class', columns='sex', values='Count')
ax = df_temp2.plot(ax=axes[0], kind='barh', stacked=False,legend=False)#, figsize=(12,4))#, color=('black','limegreen'))
ax.set_xlabel(" ", fontsize=20, labelpad = 20)
ax.set_ylabel("Course", fontsize=20, labelpad = 20)
ax.set_yticklabels(['Mathematics','Portuguese\nLanguage'])
ax.set_xlim([0,400])

i=0
for classname, fullclassname in zip(['por','mat'], ['Portuguese Language','Mathematics']):
    i+=1
    print(i)
    df_temp = df[df['class']==classname].groupby(['studytime','sex']).size().to_frame(name='Count').reset_index()
    df_temp2 = df_temp.pivot(index='studytime', columns='sex', values='Count')
    ax = df_temp2.plot(ax=axes[i],kind='barh', stacked=False, legend=False)#,sharex=axes[0])#, figsize=(12,4))
    ax.set_ylabel("Study Time [hrs]", fontsize=20, labelpad = 20)
    axes[i].set_title(fullclassname)
#studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
    ax.set_yticklabels(['0-2','2-5','5-10','+10'])
    ax.set_xlim([0,400])

ax.set_xlabel("Number of Students\n", fontsize=20, labelpad = 20)
leg = axes[0].legend(bbox_to_anchor=(1,1.2), loc='upper left', ncol=1)
leg.set_title('Sex')
axes[0].set_title('Registered Class and Study Time\n in Secondary School Students ')
fig.tight_layout()



#ax.set_title('Distribution',fontsize=20)


# In[ ]:


matplotlib.style.use('fivethirtyeight')
fig, axes = plt.subplots(2,1,figsize=(12,8))
i=0
for classname, fullclassname in zip(['por','mat'], ['Portuguese Language','Mathematics']):
    df_temp = df[df['class']==classname].groupby(['studytime','sex']).size().to_frame(name='Count').reset_index()
    df_temp.loc[(df_temp['sex']=='M'),'Count'] = 100*df_temp[(df_temp['sex']=='M')]['Count']/M_count
    df_temp.loc[(df_temp['sex']=='F'),'Count'] = 100*df_temp[(df_temp['sex']=='F')]['Count']/F_count
    df_temp2 = df_temp.pivot(index='studytime', columns='sex', values='Count')
    ax = df_temp2.plot(ax=axes[i],kind='barh', stacked=False, legend=False)#, figsize=(12,4))
    ax.set_ylabel("Study Time [hr]", fontsize=20, labelpad = 20)
    axes[i].set_title(fullclassname)
    ax.set_xlim([0,100])
    ax.set_yticklabels(['0-2','2-5','5-10','+10'])
    i+=1

axes[0].set_xlabel(" ", fontsize=20, labelpad = 20)
ax.set_xlabel("Percentage of Students [%]\n", fontsize=20, labelpad = 20)
leg = axes[0].legend(bbox_to_anchor=(1,1.2), loc='upper left', ncol=1)
leg.set_title('Sex')
fig.tight_layout()


# In[ ]:


matplotlib.style.use('fivethirtyeight')
fig, axes = plt.subplots(2,1,figsize=(12,8))

i=0
for alc, label in zip(['Walc','Dalc'],['Weekend','Daily']):
    df_temp = df.groupby([alc,'sex']).size().to_frame(name='Count').reset_index()
    df_temp.loc[(df_temp['sex']=='M'),'Count'] = 100*df_temp[(df_temp['sex']=='M')]['Count']/M_count
    df_temp.loc[(df_temp['sex']=='F'),'Count'] = 100*df_temp[(df_temp['sex']=='F')]['Count']/F_count
    df_temp2 = df_temp.pivot(index=alc, columns='sex', values='Count')
    ax = df_temp2.plot(ax=axes[i], kind='barh', stacked=False, legend=False)#, figsize=(12,4))
    ax.set_xlabel("Percentage of Students [%]\n", fontsize=20, labelpad = 20)
    ax.set_ylabel("%s Alcohol\n Consumption"%label, fontsize=20, labelpad = 20)
    #Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
    ax.set_yticklabels(['Very Low','Low','Moderate','High','Very High'])
    ax.set_xlim([0,100])
    i+=1
axes[0].set_xlabel(" ", fontsize=20, labelpad = 20)
ax.set_xlabel("Percentage of Students [%]\n", fontsize=20, labelpad = 20)
leg = axes[0].legend(bbox_to_anchor=(1,1.2), loc='upper left', ncol=1)
leg.set_title('Sex')
fig.tight_layout()


# In[ ]:


df[(df['sex']=='F') & (df['class']=='mat')]['G3'].plot(kind='kde')#alpha=0.75, bins=20)
ax = df[(df['sex']=='M') & (df['class']=='mat')]['G3'].plot(kind='kde',figsize=(12,4))#,alpha=0.75, bins=20)
ax.set_xlabel("Final Grade", fontsize=24, labelpad = 20)
ax.set_ylabel("P (Grade)", fontsize=24, labelpad = 20)
ax.set_xlim(0,20)
M_patch = mpatches.Patch(color='red',label='Male')
F_patch = mpatches.Patch(color='blue', label='Female')
ax.legend(handles=[F_patch, M_patch])


# In[ ]:


df[(df['sex']=='F') & (df['class']=='por')]['G3'].plot(kind='kde')#alpha=0.75, bins=20)
ax = df[(df['sex']=='M') & (df['class']=='por')]['G3'].plot(kind='kde',figsize=(12,4))#,alpha=0.75, bins=20)
ax.set_xlabel("Final Grade", fontsize=24, labelpad = 20)
ax.set_ylabel("P (Grade)", fontsize=24, labelpad = 20)
ax.set_xlim(0,20)
M_patch = mpatches.Patch(color='red',label='Male')
F_patch = mpatches.Patch(color='blue', label='Female')
ax.legend(handles=[F_patch, M_patch])


# In[ ]:


matplotlib.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(12,6))
ax = sns.swarmplot(x='Dalc',y='G3',hue='sex', data=df,dodge=True)
ax.set_xlabel("Workday Alcohol Consumption", fontsize=24, labelpad = 20)
ax.set_ylabel("Final Grade", fontsize=24, labelpad = 20)
ax.set_xticklabels(['Very Low','Low','Moderate','High','Very High'],rotation=0)
ax.set_title('Alcohol Consumption and School Performance\n')
ax.legend(ncol=2,loc='upper right')
fig.tight_layout()


# In[ ]:


fig, ax = plt.subplots(figsize=(12,6))
ax = sns.swarmplot(x='Walc',y='G3',hue='sex', data=df,dodge=True)
ax.set_xlabel("Weekend Alcohol Consumption", fontsize=24, labelpad = 20)
ax.set_ylabel("Final Grade", fontsize=24, labelpad = 20)
ax.set_xticklabels(['Very Low','Low','Moderate','High','Very High'],rotation=0)
ax.set_title('Alcohol Consumption and School Performance\n')
ax.legend(ncol=2,loc='upper right')
fig.tight_layout()


# In[ ]:


sns.set_palette("rainbow")
g = sns.factorplot(x="studytime", y="G3", hue='Dalc', data=df,
               col="sex", kind="swarm",size=5, aspect=1,legend=False,dodge=True); 
axes = g.axes.flatten()
axes[0].set_title("Female")
axes[1].set_title("Male")
axes[0].set_xlabel("Study Time (hrs)")
axes[1].set_xlabel("Study Time (hrs)")
axes[0].set_xticklabels(['0-2','2-5','5-10','+10'])
axes[1].set_xticklabels(['0-2','2-5','5-10','+10'])
axes[0].set_ylabel("Final Grade")
leg = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
leg.set_title('Daily\nAlcohol\nConsumption', prop={'size': 18, 'weight': 'normal'})
for i, text in zip(range(0,6),['Very Low','Low','Moderate','High','Very High']):
    leg.get_texts()[i].set_text(text)

plt.subplots_adjust(top=.925)
fig.tight_layout()

