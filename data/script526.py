
# coding: utf-8

# People can leave their workplace due to several reasons.
# 
# Some are happy with their current workplace, but happen to find a better opportunity. others are forced to leave, and some may simply suffer every minute until they finally have the chance to leave. 
# 
# Can these differences be detected in our data?

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('../input/HR_comma_sep.csv')


# # Evaluation and Satisfaction
# 
# These two features can teach us a lot about the psychological state of both the employer and the employee. let us plot a scatter plot for both the workers who left and those who stayed and see whether we can observe something: 

# In[ ]:


plt.figure(figsize = (8,8))
plt.subplot(1,2,1)
plt.plot(data.satisfaction_level[data.left == 1],data.last_evaluation[data.left == 1],'o', alpha = 0.1)
plt.ylabel('Last Evaluation')
plt.title('Employees who left')
plt.xlabel('Satisfaction level')

plt.subplot(1,2,2)
plt.title('Employees who stayed')
plt.plot(data.satisfaction_level[data.left == 0],data.last_evaluation[data.left == 0],'o', alpha = 0.1)
plt.xlim([0.4,1])
plt.ylabel('Last Evaluation')
plt.xlabel('Satisfaction level')


# # There are some suspiciously distinct patterns for employees who left
# 
# For both the employees who left and those who stayed, the data looks very unnatural. I cannot think of a reason why the distribution would look like a step function, somewhat homogeneous between some arbitrary evaluation and satisfaction values, and almost non-existent outside these limits.
# 
# However, analyzing the difference between the employees who left and those who stayed can still be valuable for 2 reasons:
# 
# 1. This is the data, so we might as well analyze it.
# 
# 2. More important - regardless of the source of the data - there are clear differences in the scatter plots between the 2 groups. so we can probably tell a story.
# 
# For the employees who didn't leave, the scatter is mostly homogeneous if we ignore the weird step function, and more dense at the higher values of both evaluation and satisfaction - implying that the normal situation for a worker is being both happy and appreciated.
# 
# But for those who left - there are very clear 3 clusters:
# 
# 1. The happy and appreciated. why would they leave? I called them, somewhat jokingly, **"Winners"** - those who leave because they were offered a better opportunity.
# 
# 
# 2. The appreciated but unhappy - Maybe they are over qualified for the job. I called them the **"Frustrated"**
# 
# 
# 3. The unappreciated and unhappy - It is not surprising that these would leave, possibly even fired. they are simply a "**Bad Match"**

# # If we have clusters, let's cluster them!

# In[ ]:


from sklearn.cluster import KMeans
kmeans_df =  data[data.left == 1].drop([ u'number_project',
       u'average_montly_hours', u'time_spend_company', u'Work_accident',
       u'left', u'promotion_last_5years', u'sales', u'salary'],axis = 1)
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(kmeans_df)
print(kmeans.cluster_centers_)

left = data[data.left == 1]
left['label'] = kmeans.labels_
plt.figure()
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.title('3 Clusters of employees who left')
plt.plot(left.satisfaction_level[left.label==0],left.last_evaluation[left.label==0],'o', alpha = 0.2, color = 'r')
plt.plot(left.satisfaction_level[left.label==1],left.last_evaluation[left.label==1],'o', alpha = 0.2, color = 'g')
plt.plot(left.satisfaction_level[left.label==2],left.last_evaluation[left.label==2],'o', alpha = 0.2, color = 'b')
plt.legend(['Winners','Frustrated','Bad Match'], loc = 3, fontsize = 15,frameon=True)


# # Now let's explore the differences between the groups

# In[ ]:


winners_hours_std = np.std(left.average_montly_hours[left.label == 0])
frustrated_hours_std = np.std(left.average_montly_hours[left.label == 1])
bad_match_hours_std = np.std(left.average_montly_hours[left.label == 2])
winners = left[left.label ==0]
frustrated = left[left.label == 1]
bad_match = left[left.label == 2]

def get_pct(df1,df2, value_list,feature):
    pct = []
    for value in value_list:
        pct.append(np.true_divide(len(df1[df1[feature] == value]),len(df2[df2[feature] == value])))
    return pct
columns = ['sales','winners','bad_match','frustrated']
winners_list = get_pct(winners,left,np.unique(left.sales),'sales')
frustrated_list = get_pct(frustrated,left,np.unique(left.sales),'sales')
bad_match_list = get_pct(bad_match,left,np.unique(left.sales),'sales')
plot_df = pd.DataFrame(columns = columns)
plot_df['sales'] = np.unique(left.sales)
plot_df['winners'] = winners_list
plot_df['bad_match'] = bad_match_list
plot_df['frustrated'] =frustrated_list
plot_df = plot_df.sort(columns = 'bad_match')



plt.figure()
values = np.unique(left.sales)
plt.bar(range(len(values)),plot_df.winners,width = 1, color = 'r',bottom=plot_df.bad_match + plot_df.frustrated)
plt.bar(range(len(values)),plot_df.frustrated, width = 1, color = 'g',bottom=plot_df.bad_match)
plt.bar(range(len(values)),plot_df.bad_match, width = 1, color = 'b')
plt.xticks(range(len(values))+ 0.5*np.ones(len(values)),plot_df.sales, rotation= 30)
plt.legend(['Winners','Frustrated','Bad Match'], loc = 3,frameon=True)

plt.title('Split of workers into the clusters for each category')


# We can see that the biggest share of "winners" is among R&D and product management people. whereas the biggest share of employees who are a "Bad Match" will be found among HR workers.
# 
# Let's see if the general distribution within the clusters is significantly different

# In[ ]:


def get_num(df,value_list,feature):
    out = []
    for val in value_list:
        out.append(np.true_divide(len(df[df[feature] == val]),len(df)))
    return out

winners_list = get_num(winners,np.unique(left.sales),'sales')
frustrated_list = get_num(frustrated,np.unique(left.sales),'sales')
bad_match_list = get_num(bad_match,np.unique(left.sales),'sales')
plot_df = pd.DataFrame(columns = columns)
plot_df['sales'] = np.unique(left.sales)
plot_df['winners'] = winners_list
plot_df['bad_match'] = bad_match_list
plot_df['frustrated'] = frustrated_list
plot_df = plot_df.sort(columns = 'bad_match')

plt.figure()
values = np.unique(left.sales)
plt.bar(range(len(values)),plot_df.winners,width = 0.25, color = 'r')
plt.bar(range(len(values))+0.25*np.ones(len(values)),plot_df.frustrated, width = 0.25, color = 'g')
plt.bar(range(len(values))+0.5*np.ones(len(values)),plot_df.bad_match, width = 0.25, color = 'b')
plt.xticks(range(len(values))+ 0.5*np.ones(len(values)),plot_df.sales, rotation= 30)
plt.legend(['Winners','Frustrated','Bad Match'], loc = 2)

plt.title('% of Workers in each cluster')


# While there are some differences, the distribution is still mostly controlled by the share of employees from the category among the total number of employees who left. This means that the cluster do not simply represent division to different categories occupation, but rather captures a real differences between employees in the same field.
# 
# Let's look the the average monthly hours distribution:

# In[ ]:


plt.figure()
import seaborn as sns
sns.kdeplot(winners.average_montly_hours, color = 'r')
sns.kdeplot(bad_match.average_montly_hours, color ='b')
sns.kdeplot(frustrated.average_montly_hours, color ='g')
plt.legend(['Winners','Bad Match','Frustrated'])
plt.title('Hours per month distribution')


# There's definitely some information here! 
# 
# It seems that the frustrated group works by far the most (possibly, and understandably, their reason to be frustrated). The winners also work a lot, and those who are a bad match word significantly less.
# 
# One can argue that the winners have the right balance between workaholism and rest, but 250 hours still sounds somewhat unbalanced for me :)
# 
# Let us again naively verify that these differences don't stem from one occupation being more dominant in one cluster:

# In[ ]:


print('HR - Average monthly hours')
print (' "Winners" ',np.mean(left.average_montly_hours[(left.sales == 'hr') & (left.label == 0)]))
print (' "Frustrated" ',np.mean(left.average_montly_hours[(left.sales == 'hr') & (left.label == 1)]))
print (' "Bad Match" ',np.mean(left.average_montly_hours[(left.sales == 'hr') & (left.label == 2)]))
print('R&D -  Average monthly hours')
print (' "Winners" ',np.mean(left.average_montly_hours[(left.sales == 'RandD') & (left.label == 0)]))
print (' "Frustrated" ',np.mean(left.average_montly_hours[(left.sales == 'RandD') & (left.label == 1)]))
print (' "Bad Match"',np.mean(left.average_montly_hours[(left.sales == 'RandD') & (left.label == 2)]))


# Again, the differences are significant within the categories 
# 
# Let's also look at the number of projects:

# In[ ]:


plt.figure()
plt.bar(0,np.mean(winners.number_project), color = 'r')
plt.bar(1,np.mean(frustrated.number_project), color = 'g')
plt.bar(2,np.mean(bad_match.number_project), color = 'b')
plt.title('Average Number of Projects')
plt.xticks([0.4,1.4,2.4],['Winners','Frustrated','Bad Match'])


# Again, the same story:
# 
# The bad match workers either get too little projects are cannot handle enough projects, which eventually makes them leave (either by choice or not).
# 
# The frustrated probably work too much, and it seems likely that they quit
# 
# The winners probably work as much as they would have wanted to, and yet choose to leave.
# 
# 
