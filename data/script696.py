
# coding: utf-8

# This is analysis of "Cheltenham's Facebook Groups" dataset. We'll discover if there is any difference in these local FB groups and what posts are bound to be most liked, most shared or most commented. We will use some statistical methods, such as [confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval) and [Mann–Whitney test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test).
# 
# *Note: for lazy ones, look at the plots in the middle and conclusions at the end.*

# ## Data obtaining

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.stats.weightstats import zconfint
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests 
get_ipython().run_line_magic('matplotlib', 'inline')


# First, let's read file "post.csv" to get information about posts (message, number of likes and shares).

# In[ ]:


posts = pd.read_csv('../input/post.csv', parse_dates=['timeStamp'])


# Now we are reading "comments.csv" to calculate number of comments for each post.

# In[ ]:


comments = pd.read_csv('../input/comment.csv')


# Let's make new dataframe with the folowing columns: msg (message text), likes (number of likes for this message), shares (number of shares), comments (number of comments), msg_len (message length), and gid (group id). 

# In[ ]:


com_count = comments.groupby('pid').count()['cid']
data = posts.join(com_count,on='pid', rsuffix='c')[['msg', 'likes', 'shares', 'cid', 'gid']]
data.columns = ['msg', 'likes', 'shares', 'comments', 'gid']
data['msg_len'] = data.msg.apply(len)


# Group IDs are too long, so we'll replace them with 1 (Elkins Park Happenings, EPH), 2 (Unofficial Cheltenham Township, UCT), 3 (Free Speech Zone, FSZ).

# In[ ]:


#117291968282998 Elkins Park Happenings
#25160801076 Unofficial Cheltenham Township
#1443890352589739 Free Speech Zone
data.gid = data.gid.map({117291968282998: 1, 25160801076: 2, 1443890352589739: 3})


# Finally, let's replace NaN values with zero and look at the resulting dataframe.

# In[ ]:


data.fillna(0,inplace=True)
data.head()


# ## Preliminary conclusions

# Now, let's visualize data obtained and see if we can say something interesting about our data.

# In[ ]:


sns.pairplot(data, hue='gid')


# Preliminary conclusions are the following:
#  - In groups 1 (EPH) and 2 (UCT) posts get more likes and repost than in group 3 (FSZ). 
#  - In group 2 (UCT) posts get more comments than in groups 1 (EPH) and  3 (FSZ)
#  - Message length is equal in all three groups.

# ## Hypotheses testing
# 
# Now let's test these hypotheses. We will calculate [95% confidence interval](https://en.wikipedia.org/wiki/Confidence_interval) for mean values of likes, shares, comments and message length in each group and compare them. 

# In[ ]:


park = data[data.gid == 1]
town = data[data.gid == 2]
free = data[data.gid == 3]

def conf_interval(field):
    """"
    Calculate confidence interval for given field
    """
    # I've rounded numbers to integers because estimated values (likes, shares, ...) are integers themselves.
    print("95% confidence interval for the EPH posts mean number of {:s}: ({z[0]:.0f}, {z[1]:.0f})".format(field, z=zconfint(park[field])))
    print("95% confidence interval for the UCT posts mean number of {:s}: ({z[0]:.0f}, {z[1]:.0f})".format(field, z=zconfint(town[field])))
    print("95% confidence interval for the FSZ posts mean number of {:s}: ({z[0]:.0f}, {z[1]:.0f})".format(field, z=zconfint(free[field])))


# In[ ]:


conf_interval('likes')


# Confidence intervals for mean number of likes in groups 1 and 2 intersect, but both lower bounds are greater than upper bound in group 3. But to be sure that mean number of likes (shares, comments, ...) is different for each group (or at least in groups 2 and 3)  let's apply [Mann–Whitney test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test) (significance level 0.05, two-sided alternative). We will compare 3 pairs of samples: EPH group vs UCT group, EPH vs FSZ, and UCT vs FSZ. This is multiple test and we will use [holm multiple test correction](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method). 
# 
# **Null hypothesis**: mean number of likes is equal in the particular pair of groups, **alternative**: null hypothesis is wrong (mean number of likes is not equal for the pair of groups). If p-value is less than 0.05 we reject null hypothesis.

# In[ ]:


def compare_means(field):
    """
    Mann–Whitney test to compare mean values level
    """
    mapping = {1: 'EPH', 2: 'UCT', 3: 'FSZ'}
        
    comparison = pd.DataFrame(columns=['group1', 'group2', 'p_value'])
    # compare number of <field> in each group 
    for i in range(1,4):
        for j in range(1,4):
            if i >= j:
                continue
            # obtaining p-value after Mann–Whitney U test
            p = mannwhitneyu(data[data.gid == i][field], data[data.gid == j][field])[1]
            comparison = comparison.append({'group1': mapping[i], 'group2': mapping[j], 'p_value': p},ignore_index=True)
    # holm correction
    rejected, p_corrected, a1, a2 = multipletests(comparison.p_value, 
                                            alpha = 0.05, 
                                            method = 'holm') 
    comparison['p_value_corrected'] = p_corrected
    comparison['rejected'] = rejected
    return comparison    


# Let's compare likes distribution in groups, if it equal or not. **p_value_corrected** here is new **p_value** after holm correction.

# In[ ]:


conf_interval('likes')
print(compare_means('likes'))
# compare number of likes in group1 with number of likes in group2, 
# and if rejected field is True make a conclusion that 
# mean number of likes in the first group is different from mean number of likes in the second one.  


# Great, we see that mean number of likes in each group is statistically different. But practically speaking there is no sense because the difference is only 1-2 likes.
# 
# **So, our conclusion #1: posts get about equal number of likes in all three groups .**

# Now, let's do the same test for number of shares.

# In[ ]:


conf_interval('shares')
print(compare_means('shares'))


# Confidence intervals for groups 1 and 3, 2 and 3 do not intersect, and Mann–Whitney test rejects all null hypotheses, so we can make another conclusion.
# 
# **Conclusion #2: posts from group 3 are almost never shared, and it has practical significance: if you want your message to be shared, you should choose one of the first two groups.**

# Let's do the same test for number of comments.

# In[ ]:


conf_interval('comments')
print(compare_means('comments'))


# In average, posts in each group have about equal number of comments. Yes, statistical test has shown significant difference in mean number of comments, but again there is no practical sense.
# 
# **Conclusion #3: hypothesis that posts in group 2 (UCT) get more comments is wrong. Posts in each group have about equal number of comments**

# Finally, let's do the same test for message length.

# In[ ]:


conf_interval('msg_len')
print(compare_means('msg_len'))


# Confidence intervals for groups 1 and 2, 1 and 3 do not intersect, and Mann–Whitney test rejects only first null hypothesis. It means that message length in group 2 is significantly longer than in group 1. Probably, messages in the first group have more photos instead.
# 
# **Conclusion #4: message length in group 2 (UCT) is longer than in group 1 (EPH).**
# 

# ## What do people like?
# 
# Some posts in this dataset have exceptional characteristics: some of them have enormous number of shares while having very little likes, and some posts got huge number of likes despite just few people decided to share them, and so on. Let's investigate why this happens, and what such posts contain.

# ### Many shares, few likes
# First of all, we'll look at the posts that have number of shares larger than other 98% posts, and that was shared much more often than liked. I think it's interesting to know why this happened.

# In[ ]:


shared = data[data.shares > data.shares.quantile(0.98)][data.shares > data.likes*10][['msg','shares']]

top = 10
print("top %d out of %d" % (top, shared.shape[0]))
sorted_data = shared.sort_values(by='shares', ascending=False)[:top]
for i in sorted_data.index.values:
    print('shares:',sorted_data.shares[i], '\n','message:', sorted_data.msg[i][:200], '\n')


# OK, this was obvious. People try to propagate on the internet information about lost and found pets. There is no need to like such posts, but reposts are very useful.

# ### Many likes, few shares
# Let's go further and consider opposite situation when post has a lot of likes and virtually none shares.

# In[ ]:


likes = data[data.likes > data.likes.quantile(0.98)][data.likes > data.shares*100][['msg', 'likes']]
print("top %d out of %d" % (top, likes.shape[0]))
sorted_data = likes.sort_values(by='likes', ascending=False)[:top]
for i in sorted_data.index.values:
    print('likes:',sorted_data.likes[i], '\n','message:', sorted_data.msg[i][:300], '\n')


# In this cluster we see messages of gratitude or messages from new neighbours wanting to get to know their new town. People try to show that they like such messages or to welcome the newcomers, and there is no need to share such posts outside the groups. This messages do not ask help, just post information about events. Posts have a positive tone mostly.

# ### Most commented
# 
# In this section let's discover what Cheltenham people discuss most willingly. Let's look at most commented posts.

# In[ ]:


discussed = data[data.comments > data.comments.quantile(0.98)][['msg', 'comments']]

print("top %d out of %d\n" % (top, discussed.shape[0]))
sorted_data = discussed.sort_values(by='comments', ascending=False)[:top]
for i in sorted_data.index.values:
    print('comments:',sorted_data.comments[i], '\n','message:', sorted_data.msg[i][:300], '\n')


# We can see that words such as "please, explain" or "do you know" appeare quite often in these messages. In this cluster we can find posts about local problems (school, sewer,  traffic) causing discussions, or direct questions about local events. And, of course, political posts. People here do not ask help, do not thank. Messages have neutral or negative tone.

# ## Conclusions
# 
# At the beginning we have made some preliminary conclusions and none of them was right. I'm reminding you:
# 
#  * In groups 1 (EPH) and 2 (UCT) posts get more likes and repost than in group 3 (FSZ). 
#  * In group 2 (UCT) posts get more comments than in groups 1 (EPH) and  3 (FSZ)
#  * Message length is equal in all three groups.
# 
# **What we see after testing:**
# 
# *Comparing Groups*
# 
#  * Posts get about equal number of likes in all three groups
#  * Posts get about equal number of comments in all three groups
#  * Posts from group 3 (FSZ) are almost never shared, and it has practical significance: if you want your message to be shared, you should choose one of the first two groups.
#  * Messages in group 2 (UCT) are longer (~300 symbols) than in group 1 (EPH) (~250 symbols). Probably, messages in the first group have more photos instead.
#  
# *Likes, shares, and coments*
# 
#  * Posts having large number of shares contain information about lost and found pets or ask for help.
#  * Posts having large number of likes are messages of gratitude or messages from new neighbours wanting to get to know their new town. Such posts have a positive tone mostly.
#  * Posts causing long discussions are posts about local problems such as school, sewer, traffic, messages with direct questions about local events, and political posts.
