
# coding: utf-8

# This dataset (as far as I understand) provides information retrieved through an informative system called HappyForce https://www.myhappyforce.com/en/  aimed at collecting data about employees satisfaction and happyness. 
# 
# By means of an app  employees have been asked to answer a question?
# **"How happy ar you today at work?**
# The following answers were possible:
# - 4 Great
# - 3 Good
# - 2 So So
# - 1 Pretty Bad
# 
# Whenever they want they can post this feeback about their happiness and leave a comment. The can also like or dislike comments. Plesae visit the related site for complete and correct information.
# ![happyforce](https://www.myhappyforce.com/wp-content/themes/happyforce/img/app.png)
# 
# Some important references are 
# * [1] J. Berengueres, G. Duran, D. Castro, Happiness,an inside job? Turnoverprediction using employee likeability, engagement and relative happiness, ASONAM 2017, Sidney.
# * [2] https://www.slideshare.net/harriken/ieee-happiness-an-inside-job-asoman-2017
# * [3] https://www.myhappyforce.com/en/ 
# 
# According to [1] the data spans 2.5 years and 4,356 employees of 34 companies  based  in  Barcelona.
# 
# This dataset is amazingly interesting because it provides different perspectives on an elusive phaenomenon like employees happines. It also provides different data challenges like:
# - time series like the trend of happiness self perception in time or the thrend of posted comments, etc.;
# - networks analysis of social interactions (i.e. likes, dislikes, etc.);
# - clustering;
# - prediction (i.e. predict whether an employee will evntually leave the company at the end of the observation time)

# ## Loading and Wrangling Data

# First of all I am going to prepare the data for the analysis I want to perform. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


churn = pd.read_csv('../input/churn.csv')
churn.head()


# In[ ]:


ci = pd.read_csv('../input/commentInteractions.csv')
ci.head()


# In[ ]:


ci.shape


# In[ ]:


cc = pd.read_csv('../input/comments_clean_anonimized.csv')
cc.head()


# In[ ]:


votes = pd.read_csv('../input/votes.csv')
votes.head()


# In[ ]:


len(votes['companyAlias'].unique())


# There are 37 companies ([1] claims data are gathered from 34 compnies). I am going to get the list of unique company ids and use it to chang their occurrences (i.e. in the votes dataframe) with the company ids index (a shorter and thus easier to manage integer).

# In[ ]:


companies = pd.Series(votes['companyAlias'].unique())
vc = [companies.values.tolist().index(company) for company in votes['companyAlias'].values]
churn_company = [companies.values.tolist().index(company) if company in companies.values else -1 for company in churn['companyAlias'].values ]
comment_company = [companies.values.tolist().index(company) if company in companies.values else -1 for company in ci['companyAlias'].values ]
comment_company2 = [companies.values.tolist().index(company) if company in companies.values else -1 for company in cc['companyAlias'].values ]

votes['companyAlias'] = vc
churn['companyAlias'] = churn_company
ci['companyAlias'] = comment_company
cc['companyAlias'] = comment_company2


# In[ ]:


dates = votes['voteDate'].str.replace('CET','')
dates = dates.str.replace('CEST','')
votes['voteDate']= dates


# In[ ]:


votes['voteDate'] = pd.to_datetime(votes['voteDate'],format="%a %b %d %H:%M:%S %Y")


# In[ ]:


votes['wday'] = votes['voteDate'].dt.dayofweek
votes['yday'] = votes['voteDate'].dt.dayofyear
votes['year'] = votes['voteDate'].dt.year


# In[ ]:


votes['year'].unique()


# We are dealing with four years of observations ranging from 2014 to 2017.

# In[ ]:


votes['year'] = votes['year']-2014


# In[ ]:


votes['employee'] = votes['companyAlias'].astype(str)+"_"+votes['employee'].astype(str)
churn['employee'] = churn['companyAlias'].astype(str)+"_"+churn['employee'].astype(str)
ci['employee'] = ci['companyAlias'].astype(str)+"_"+ci['employee'].astype(str)
cc['employee'] = cc['companyAlias'].astype(str)+"_"+cc['employee'].astype(str)

len(votes['employee'].unique())


# We are dealing with 4377 employees. Let's aggredate data on employee basis

# In[ ]:


employee = votes.groupby('employee',as_index=False).mean()
employee = employee.merge(churn,on=['employee','employee'],how='left').drop_duplicates(subset="employee")


# In[ ]:


employee['companyAlias'] = employee.companyAlias_x.astype(int)
employee = employee.drop(['companyAlias_x','companyAlias_y'],axis=1)
employee.head()


# ## Self-reported Happyness

# By means of an app [3] employees have been asked to answer a question?
# **"How happy ar you today at work?**
# The following answers were possible:
# - 4 Great
# - 3 Good
# - 2 So So
# - 1 Pretty Bad

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

f,axarr = fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))
data =votes.groupby('companyAlias').mean()
sns.barplot(x=data.index,y= data['vote'])


# In[ ]:


week_happ = votes.groupby('wday').mean()['vote']
sns.barplot(x = week_happ.index, y = week_happ.values)


# In[ ]:


churn_employee = employee[employee['stillExists']==True]
churn_employee = churn_employee.groupby('companyAlias').count()
tmp = employee.groupby('companyAlias',as_index=False).count()
churn_perc = 1- churn_employee['stillExists'].astype(float)/tmp['stillExists']
churn_perc = [0 if np.isnan(perc) else perc for perc in churn_perc]


# In[ ]:


data['churn_perc'] = churn_perc


# We can evaluate possible correlations betweean any pair of feature:

# In[ ]:


data.corr()


# as we can see  data shows a  negative correlation between the percentage of churn employees and the mean of happiness in the related company. This apparently shows that companies where employees feel unhappy tend to have an higher percentage of churn. The following diagram depicts this trend.

# In[ ]:


sns.regplot(data['vote'].values,data['churn_perc'].values)


# ## Being Social Matters

# In[ ]:


likes = ci[ci['liked']==True].groupby('employee',as_index=False).count()
likes = likes[['employee','liked']]
hates = ci[ci['disliked']==True].groupby('employee',as_index=False).count()
hates = hates[['employee','disliked']]
hated = cc[cc['dislikes']==True].groupby('employee',as_index=False).count()
hated = hated[['employee','dislikes']]
loved = cc[cc['likes']==True].groupby('employee',as_index=False).count()
loved = loved[['employee','likes']]
employee = employee.merge(likes,on='employee',how='left').drop_duplicates(subset="employee")
employee = employee.merge(hates,on='employee',how='left').drop_duplicates(subset="employee")
employee = employee.merge(hated,on='employee',how='left').drop_duplicates(subset="employee")
employee = employee.merge(loved,on='employee',how='left').drop_duplicates(subset="employee")
employee.shape


# I have counted the number of likes and dislikes for each employee. The following heatmap shows some possible correlations among the employee data. The employee happiness (Vote) seems to be directly related with the possibility that the employee remained in the company. The number of likes and dislikes the employee has given exhibits small correlation.
# An interesting thing to note is that the number of likes the employee got to his/her commnents are directly related to his/her perceived happiness and thus (possibly) to the possibility he/she will remain in the company.

# In[ ]:


f,ax = plt.subplots(1,1,figsize=(15,10))
red_emp = employee.drop(['companyAlias','lastParticipationDate','wday'],axis=1)
sns.heatmap(red_emp.corr())
plt.title('Features Correlation Heatmap',fontsize=24)
plt.show()


# In[ ]:


author_comments = cc[['employee','commentId']].drop_duplicates(subset='commentId')
author_dict = {commentId:author_comments['employee'].values[i] for i,commentId in enumerate(author_comments['commentId'].values)}
comments = [commentId for i,commentId in enumerate(author_comments['commentId'].values)]


# In[ ]:


#this is too computational intensive I will work on subsets
#authors = [author_dict[commentId] if commentId in comments else -1 for commentId in ci['commentId'].values]


# In[ ]:


from nxviz import CircosPlot


# ## PCA and Clustering

# In this section I am going to find a possible strategy to clusterize the employees. My idea is to create a dataframe of features where each row represents an employee and each column represent a day in the four years of observations. The value in that cell represents the vote (about its happiness) the employee posted, 0 otherwise.
# 
# This means 796 features  which are  lot. I therefore use PCA to reduce the number of features to 6. From these 6 components I have chosen 2 which, in my opinion, when plotted in 2D exhibits a distribution which is good for clustering.

# In[ ]:


votes_feature = votes[['employee','vote','wday','yday']]
votes_feature['yday'] = votes['yday']+ votes['year']*365
dummies = pd.get_dummies(votes_feature['yday'])
for i,row in enumerate(votes_feature.values):
    dummies.loc[i,row[3]]= row[1]
votes_feature=pd.concat([votes_feature,dummies],axis=1)


# In[ ]:


dummies = votes_feature.groupby('employee',as_index=False).sum().drop(['vote','wday','yday','employee'],axis=1)
dummies.head(2)


# In[ ]:


from sklearn.decomposition import PCA

#votes_feature = votes[['vote','wday','yday']]
pca = PCA(n_components=6)
pca.fit(dummies)
components = pca.transform(dummies)
components = pd.DataFrame(components,columns=['c1','c2','c3','c4','c5','c6'])


# In[ ]:


components.head(2)


# In[ ]:


tocluster = pd.DataFrame(components[['c6','c4']])


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import silhouette_score

clusterer = KMeans(n_clusters=4,random_state=42).fit(tocluster)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)


# I have plotted several pairs of the obtained components looking for one 2D distribution suitable for clustering. I hav decided to go with th pair (6,4) and 4 clusters.

# In[ ]:


import matplotlib
fig = plt.figure(figsize=(17,15))
colors = ['orange','blue','purple','green','brown','red','pink','white']
colored = [colors[k] for k in c_preds]
plt.scatter(tocluster['c6'],tocluster['c4'],  color = colored,s=10,alpha=0.5)
for ci,c in enumerate(centers):
    plt.plot(c[0], c[1], 'o', markersize=15, color='black', alpha=0.5, label=''+str(ci))
    plt.annotate(str(ci), (c[0],c[1]),fontsize=24,fontweight='bold')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()


# In[ ]:


employee['cluster']=c_preds


# here is how employees are distributed among clusters

# In[ ]:


tot_cluster = employee.groupby('cluster').count()['employee']
tot_churn = len(employee[employee['stillExists']==False])
tot_cluster


# In[ ]:


tmp = employee[employee['stillExists']==False].groupby('cluster',as_index='False').count()['employee']
churn_perc_totchurn = tmp/tot_churn
tmp=employee[employee['stillExists']==False].groupby('cluster',as_index='False').count()['employee']
churn_perc_totcluster = tmp/tot_cluster
fig1, axarr = plt.subplots(1,2,figsize=(17,10))
explode = (0, 0, 0.2, 0)
labels = 'Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'
axarr[0].pie(churn_perc_totchurn, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
axarr[0].axis('equal')  
axarr[0].set_title('Percentage of Churn in clusters with respect to the total of churn')
sns.barplot(x=[0,1,2,3],y=churn_perc_totcluster,ax=axarr[1])
axarr[1].set_title('Ratio of Churn in clusters with respect to the cluster population')
plt.show()


# This is interesting and I think it will deserve a more detailed analysis: although the third cluster is particulary small (597 individuals) it contains 20% of the employees that laved the Company. We might have found something intersting.

# In[ ]:


vote_cluster = employee.groupby('cluster',as_index='False').mean()['vote']
nvote_cluster = employee.groupby('cluster',as_index='False').mean()['numVotes']
likes_cluster = employee.groupby('cluster',as_index='False').mean()['likes']
dislikes_cluster = employee.groupby('cluster',as_index='False').mean()['dislikes']
liked_cluster = employee.groupby('cluster',as_index='False').mean()['liked']
disliked_cluster = employee.groupby('cluster',as_index='False').mean()['disliked']
fig,axarray = plt.subplots(3,2,figsize=(17,30))
sns.barplot(x=[0,1,2,3],y=vote_cluster,ax=axarray[0,1])
axarray[0,1].set_title('Happyness in Clusters')
sns.barplot(x=[0,1,2,3],y=vote_cluster,ax=axarray[0,0])
axarray[0,0].set_title('Happyness in Clusters')
sns.barplot(x=[0,1,2,3],y=nvote_cluster,ax=axarray[0,1])
axarray[0,1].set_title('#Votes in Clusters')
sns.barplot(x=[0,1,2,3],y=likes_cluster,ax=axarray[1,0])
axarray[1,0].set_title('#Likes Received in Clusters')
sns.barplot(x=[0,1,2,3],y=dislikes_cluster,ax=axarray[1,1])
axarray[1,1].set_title('#Dislikes Received in Clusters')
sns.barplot(x=[0,1,2,3],y=liked_cluster,ax=axarray[2,0])
axarray[2,0].set_title('#Liked Comments in Clusters')
sns.barplot(x=[0,1,2,3],y=disliked_cluster,ax=axarray[2,1])
axarray[2,1].set_title('#Disliked Comments in Clusters')
plt.show()


# ## Time Series

# In[ ]:


c1_sample = employee[employee['cluster']==2].head(5)


# In[ ]:


f,axarr = plt.subplots(3,2,figsize=(15,30))
sample = employee[employee['cluster']==2][0:6]
vtp = votes[votes['employee']==sample['employee'].iloc[0]]
axarr[0,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)
vtp = votes[votes['employee']==sample['employee'].iloc[1]]
axarr[0,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)
vtp = votes[votes['employee']==sample['employee'].iloc[2]]
axarr[1,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)
vtp = votes[votes['employee']==sample['employee'].iloc[3]]
axarr[1,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)
vtp = votes[votes['employee']==sample['employee'].iloc[4]]
axarr[2,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)
vtp = votes[votes['employee']==sample['employee'].iloc[5]]
axarr[2,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

plt.xlim(0,365)
plt.show()


# In[ ]:


f,axarr = plt.subplots(3,2,figsize=(15,30))
sample = employee[employee['cluster']==1][20:26]
vtp = votes[votes['employee']==sample['employee'].iloc[0]]
axarr[0,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)
vtp = votes[votes['employee']==sample['employee'].iloc[1]]
axarr[0,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)
vtp = votes[votes['employee']==sample['employee'].iloc[2]]
axarr[1,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)
vtp = votes[votes['employee']==sample['employee'].iloc[3]]
axarr[1,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)
vtp = votes[votes['employee']==sample['employee'].iloc[4]]
axarr[2,0].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)
vtp = votes[votes['employee']==sample['employee'].iloc[5]]
axarr[2,1].scatter(x=vtp['yday'].values,y=vtp['vote'].values,s=40,alpha=0.8)

plt.xlim(0,365)
plt.show()


# Work in progress - Stay Tuned
