
# coding: utf-8

# # **What Moneyball data can tell us about baseball?**
# 
# **Foreword:** *I am not an expert in baseball. I have only started watching baseball this year. I went through the whole season starting to look at several teams. But I have finally become infatuated by New York Yankees. The questions or conclusions below might seem stupid or naive to those who know baseball better. But honestly I do not really care. My intention is to understand this lovely game better with skills which I have in hand. And hopefully while doing this time till next season runs faster!*  
# *I also would be grateful for any comments and suggestions which you can leave in comments sections. I am learning so any advice/warning/suggestion will be extermely helpful!*

# **The mission of this notebook is to:**
# * practice data exploration skills
# * pracitce data visualization skills
# * play with some common statistics terms and learn how to apply them
# * complete "5 Day Data Challenge"
# * understand baseball as a game better
# * have fun

# **Why Moneball dataset?**
# 
# * it is renown for its impact on the game
# * it is relatively small and will not require a lot of memory or CPU 
# * it is realtively small and I can practically keep it in my head

# **Questions popping up in my head when I think about baseball:**
# 1. Are top teams always perform stable and how they manage to maintain their levels?
# 2. Is baseball a team sport or an individual one just being presented in a team format?
# 3. What is the role of a management staff in team's wins and losses?
# 4. Are all players going through ups and downs or there are some players who can maintain the level?
# 5. What can we say about teams ups and downs and how it translates to key performance indicators?
# 6. What is the correlation between Runs Scored, Runs Allowed and Wins? What do outliers tell us?
# 7. Does higher Batting Average mean more Runs Scored?
# 8. Does higher Batting Average mean higher On-Base Percentage?
# 9. Does high On-Base Percentage, Batting Average and other measurements guarantee you a place in Playoffs?
# 10. We could assume that the higher performance of a team, the more wins it had and the higher chances were to get to Post-Season. Is that true? Does a team need an outstanding performance to get to Playoffs?
# *To be continued*
# 
# Not all of them can be answered with this particular dataset.  
# Some of those do not perhaps have a definite answer at all.  
# But I believe questions are more important than answers.  

# **Content:**
# 1. Data Preparation
# 2. Data Exploration

# ______________________________________________________________________________________________________

# # 1. Data Preparation

# _______________________________________________________________________________________________________

# In data preparation stage I will do couple of simple things:
# * import necessary libraries
# * upload data
# * explore the basics about data, i.e. shape, columns etc.
# * remove 2 columns - RankSeason and RankPlayoffs. Those columns I will not use during my data exploration
# * convert one column - Year - to datetime object
# * Look at what teams are in the dataset
# * Look at some basic statistics of this dataset

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, iqr, chisquare
import squarify


# In[ ]:


df = pd.read_csv("../input/baseball.csv")


# In[ ]:


print(df.columns)


# In[ ]:


print(df.shape)


# In[ ]:


df = df[['Team', 'League', 'Year', 'RS', 'RA', 'W', 'OBP', 'SLG', 'BA',
       'Playoffs', 'G', 'OOBP', 'OSLG']]


# In[ ]:


print(df.shape)


# In[ ]:


df['Team'].value_counts()


# In[ ]:


df.describe()


# In[ ]:


print(df.dtypes)


# In[ ]:


df['Year'] = pd.to_datetime(df['Year'], format="%Y").dt.year


# In[ ]:


print(df.dtypes)


# _______________________________________________________________________________________________________

# # 2. Data Exploration

# _______________________________________________________________________________________________________

# First thing which would be interesting to do is to identify our so called "top" teams.  
# We can generally assume that "top" teams will be those who has high results in such parameters as:  
# **OBP** or On-Base % or On-Base Percentage  
# **SLG** or Slugging % or Slugging Percentage  
# **RS** or Runs Scored  
# **RA** or Runs Allowed  
# **W** or Wins  
# **BA** or Batting Average  
# **OOBP** or Opponent On-Base Percentage  
# **OSLG** or Opponent Slugging Percentage  
# 
# We can split those stats to 3 groups:  
# **Group 1.** Offensive stats: **OBP, SLG, BA**  
# **Group 2.** Defensive stats: **OOBP, OSLG**  
# **Group 3:** "Dependent" stats: **RS, RA, W**  
# 
# This is my division and maybe it is not correct. But let me try to explain my approach.  
# 
# If a team has high **OBP** or **On-Base %** it means that the team is good in taking bases. The more often a team takes bases the more chances it has to score runs **RS**. The more runs **RS** a team scores the more chances it has to win **W** a game. The more games a team wins during the season the more chances it has to get to **Playoffs**. And playoffs is basically what every team wants to achieve.  
# So you can see that **RS** (**Runs Scored**) and **W** (**Wins**) are in general - consequences of high **OBP**. The same can be said about **SLG** (**Slugging %**) and **BA** (**Batting Average**).
# 
# Defensive stats imply the same logic. The less **OOBP** a team has the less often an opponent takes bases, the less runs opponent scores and the less games it wins etc = more games you win!
# 
# That's why it would be interesting to explore the data from those 3 perspectives: *Offense*, *Defense* and *Outcome* of those two.
# 
# **NB**: notice, I am talking about chances above because in baseball noone can guarantee you anything. You can be trailing 9:1 in 5th inning and then win a game 14:11 like Yankees did in the game vs Orioles this season. So no guarantees, but your chances are higher if your performance is high. That's one of the reasons I love baseball!

# Let's check if my hypothesis is correct!

# **Offensive stats vs output**

# In[ ]:


fig = plt.figure(figsize=(12,12))
#fig.suptitle("Offensive stats and its impact on Runs Scored and Wins")
ax1 = fig.add_subplot(3,2,1)
ax2 = fig.add_subplot(3,2,2)
ax3 = fig.add_subplot(3,2,3)
ax4 = fig.add_subplot(3,2,4)
ax5 = fig.add_subplot(3,2,5)
ax6 = fig.add_subplot(3,2,6)
sns.regplot(x="OBP", y="RS", data=df, scatter=True, marker="+", ax=ax1)
sns.regplot(x="SLG", y="RS", data=df, scatter=True, marker="+", ax=ax2)
sns.regplot(x="OBP", y="W", data=df, scatter=True, marker="+", ax=ax3)
sns.regplot(x="SLG", y="W", data=df, scatter=True, marker="+", ax=ax4)
sns.regplot(x="BA", y="RS", data=df, scatter=True, marker="+", ax=ax5)
sns.regplot(x="BA", y="W", data=df, scatter=True, marker="+", ax=ax6)
ax1.set_xlabel("On-Base %")
ax1.set_ylabel("Runs Scored")
ax2.set_xlabel("Slugging %")
ax2.set_ylabel("Runs Scored")
ax3.set_xlabel("On-Base %")
ax3.set_ylabel("Wins")
ax4.set_xlabel("Slugging %")
ax4.set_ylabel("Wins")
ax5.set_xlabel("Batting Average")
ax5.set_ylabel("Runs Scored")
ax6.set_xlabel("Batting Average")
ax6.set_ylabel("Wins")
ax4.set_ylim([40,120])
ax6.set_ylim([40,120])
sns.despine()
plt.tight_layout()
plt.show()


# We see that there is a positive correlation between offensice stats and winning output.  
# What we can also observe is that clearly OPB, BA and SLG have a direct impact on Runs Scored. However BA/OBP/SLG correlations with Wins are slightly more gentle. Because as I said - scoring runs does not guarantee you a win.
# 
# Let's do the same for defensive stats.

# **Defensive stats vs output**

# In[ ]:


fig = plt.figure(figsize=(12,12))
fig.suptitle("Defensive stats and its impact on Runs Allowed and Wins")
ax1 = fig.add_subplot(2,2,1)
ax1.set_title("Opponent On-Base % vs Runs Allowed")
ax2 = fig.add_subplot(2,2,2)
ax2.set_title("Opponent Slugging % vs Runs Allowed")
ax3 = fig.add_subplot(2,2,3)
ax3.set_title("Opponent On-Base % vs Wins")
ax4 = fig.add_subplot(2,2,4)
ax4.set_title("Opponent Slugging % vs Wins")
sns.regplot(x="OOBP", y="RA", data=df, scatter=True, marker="+", ax=ax1)
sns.regplot(x="OSLG", y="RA", data=df, scatter=True, marker="+", ax=ax2)
sns.regplot(x="OOBP", y="W", data=df, scatter=True, marker="+", ax=ax3)
sns.regplot(x="OSLG", y="W", data=df, scatter=True, marker="+", ax=ax4)
ax1.set_xlabel("Opponent On-Base %")
ax1.set_ylabel("Runs Allowed")
ax2.set_xlabel("Opponent Slugging %")
ax2.set_ylabel("Runs Allowed")
ax3.set_xlabel("Opponent On-Base %")
ax3.set_ylabel("Wins")
ax4.set_xlabel("Opponent Slugging %")
ax4.set_ylabel("Wins")
ax4.set_ylim([40,120])
plt.show()


# Looks like our hypothesis was not that bad. We have the same picture with defensive stats. With the only difference that Opponent OBP and Opponent SLG means that our opponent scores runs which for team in offence are Runs Allowed. That's why top 2 plots look identical to the OBP and SLG above.

# We could as well look at it in a simpler way. For example, calculate r-value and plot it as a heatmap like this

# In[ ]:


corr_df = df.corr()
corr_df


# In[ ]:


fig = plt.figure(figsize=(16,12))
# Generate a mask for the upper triangle
mask = np.zeros_like(corr_df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.set_style("white")
sns.heatmap(corr_df, mask=mask, cmap=cmap, annot=True)


# This simple exercise takes us closer to next exploration task I would like to do. Define which teams are "top" teams.  
# Well even for someone who does not understand and does not even care about baseball the following will be obvious:  
# - team which scores more runs and has more wins is more likely to get to Playoffs and thus to be called a "top" team
# - team which allows less runs, has more chances to win games -> get to Playoffs -> be called a "top" team
# - finally, team which combines those 2 characteristics -> will most likely get to Playoffs -> most likely can be called a "top" team

# The first thing to do is to look at the distribution of every Offensive and Defensive stats for each team.  
# To do that I will use a box plot and as well a mean line.  
# This might be a simple approach but what I want to see is how stats of each team look like compared to mean

# In[ ]:


obp_mean = df['OBP'].mean()
slg_mean = df['SLG'].mean()
ba_mean = df['BA'].mean()
oobp_mean = df['OOBP'].mean()
oslg_mean = df['OSLG'].mean()


# **Offensive stats**

# In[ ]:


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1,1,1)
sns.boxplot(x=df['OBP'], y=df['Team'], ax=ax)
plt.title("On-Base % distribution by Team")
plt.axvline(obp_mean)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1,1,1)
ax = sns.boxplot(x=df['SLG'], y=df['Team'], ax=ax)
plt.title("Slugging % distribution by Team")
plt.axvline(slg_mean)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1,1,1)
sns.boxplot(x=df['BA'], y=df['Team'], ax=ax)
plt.title("Batting Average distribution by Team")
plt.axvline(ba_mean)
plt.show()


# **Defensive stats**

# In[ ]:


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1,1,1)
sns.boxplot(x=df['OOBP'], y=df['Team'], ax=ax)
plt.title("Opponent On-Base % distribution by Team")
plt.axvline(oobp_mean)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(1,1,1)
sns.boxplot(x=df['OSLG'], y=df['Team'], ax=ax)
plt.title("Opponent Slugging % distribution by Team")
plt.axvline(oslg_mean)
plt.show()


# Now those charts shows us a certain picture. 
# 
# Even visually we can observe a certain tendency without making any analysis: some teams are much better than average performance in the League and as well much better than other teams. Moreover some teams have more stable performance than others. And you can see that is true for all stats, offensive and defensive. To illustrate this, let's take 6 teams  based on our "visual" observation and make a "zoomed-in" box plot of On-Base % for those teams only.  
# For this exercise I have selected following teams (it was important that all teams present in every season and appear equal amount of times in the dataset):   
# BOS or Boston Red Sox  
# NYY or New York Yankees  
# HOU or Houston Astros  
# CHC or Chicago Cubs  
# NYM or New York Mets  
# LAD or Los Angeles Dodgers 
# 

# We need to create a subset of our main dataframe which will contain only teams listed above

# In[ ]:


teams = ['BOS', 'NYY', 'HOU', 'CHC', 'NYM', 'LAD']
exp_df = df.loc[df['Team'].isin(teams)]
exp_df.head(5)


# Now lets make not only box plot but as well a violin plot. Both types of plots will "add value" to each other and probably help us to get some more insights.

# In[ ]:


fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.boxplot(x=exp_df['OBP'], y=exp_df['Team'], ax=ax1)
sns.violinplot(x=exp_df['OBP'], y=exp_df['Team'], ax=ax2)
ax1.set_title("On-Base % distribution by for 6 selected teams")
ax2.set_title("On-Base % distribution by for 6 selected teams")
# I will use the "global" mean here
ax1.axvline(obp_mean)
ax2.axvline(obp_mean)
ax2.yaxis.set_visible(False)
sns.despine(left=True, bottom=True)
plt.show()


# I think this is a good illustration of what we were discussing above!  
# You can see how Red Sox and Yankees are dominating here although on that big plot above with all teams it was not that obvious.  
# But let us try to be a little bit more "scientific". What does this box plot tell us in statistical terms? Few observations we can make:   
# 
# * Median of Yankees and Red Sox is higher than global mean. In cases of BOS their 25-percentile does not even "touch" the mean line. Yanks look a little less impressive from that perspective. But still 2 "top" teams are much better in taking bases  
# * Next observation - stable performance of "top" teams. You can see that the BOS OBP range is rather narrow, Yanks have bigger range and both have no outliers. As an opposite situation - look at Mets who had really bad performance below 0.28 or Cubs who had a an OBP around 0.35.  
# * We can see that data distribution is mostly symmetric. In some cases there is some gentle right skeweness (i.e. Yankees or Cubs). Our violin plot as well supports that - 5 o 6 teams have an OBP distribution "close" to normal. Only Yankees are an exception and seem to be an outlier, although that is not that dramatic. 
# * In 5 of 6 cases data in IQR (Interquartile Range) is rather tightly grouped. Maybe Yankees here have a wider IQR. Which might mean that in general those teams keep their level of performance.  
# 
# Finally, I want to add one more small detail which will take us to the next step of our data exploration. This dataset is for 1962 - 2012 years (2 years are missing). During this period our teams have won World Series - the highest award you can get in baseball:  
# **Houston Astros - 0** World Series titles (they won their first WS in 2017)  
# **New York Mets - 2** World Series titles (1969, 1986)  
# **Chicago Cubs - 0** World Series titles (their last WS happened in 2016)  
# **Boston Red Sox - 2** World Series Titles (2004, 2007)  
# **New York Yankees - 8** World Series Titles (1962, 1977, 1978, 1996, 1998, 1999, 2000, 2009)  
# **Los Angeles Dodgers - 4** World Series Titles (1963, 1965, 1981, 1988)  
# 
# Of course we cannot make "big" conclusions out of this, but you can look at other stats box plots and see that same tendency: Red Sox and Yankees will look better in those stats too and that is reflected in 10 WS titles in total.
# 
# I will be working with same 6 teams further as it will be much easier to draw plots and make other observations.
# 
# Next interesting thing to see would be to count how often our 6 teams got to Playoffs (Not necessarily WS) and see if tendency described above will persist.
# 

# In[ ]:


playoff = {}
for team in teams:
    team_playoff = exp_df.loc[exp_df['Team'] == team, 'Playoffs'].sum()
    playoff[team] = team_playoff
print(playoff)


# In[ ]:


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(1,1,1)
ax = squarify.plot(sizes=playoff.values(), label=playoff.keys(),                     color=["red","green","blue", "grey", "orange"], alpha=.4 )
plt.axis('off')
plt.title('How many times teams appeared in playoffs from 1962 to 2012')
plt.show()


# Clearly we see who are our "top" teams and who - "outsiders".  
# 
# Finally, let's do the following. Let's plot our Offensive and Defensive stats, its correlation with number of Wins and color it depending on how many times our 6 teams went to Playoffs

# In[ ]:


g = sns.FacetGrid(exp_df, col='Team', hue='Playoffs')
g = g.map(plt.scatter, 'BA', 'W', edgecolor="w").add_legend()


# In[ ]:


g = sns.FacetGrid(exp_df, col='Team', hue='Playoffs')
g = g.map(plt.scatter, 'OBP', 'W', edgecolor="w").add_legend()


# In[ ]:


g = sns.FacetGrid(exp_df, col='Team', hue='Playoffs')
g = g.map(plt.scatter, 'SLG', 'W', edgecolor="w").add_legend()


# In[ ]:


g = sns.FacetGrid(exp_df, col='Team', hue='Playoffs')
g = g.map(plt.scatter, 'OOBP', 'W', edgecolor="w").add_legend()


# In[ ]:


g = sns.FacetGrid(exp_df, col='Team', hue='Playoffs')
g = g.map(plt.scatter, 'OSLG', 'W', edgecolor="w").add_legend()


# Without getting into much details we can see that good stats may bring you more wins and give you higher chances for Playoffs. However, as we have argued at the beginning of this notebook, in baseball there is no guarantee. And neither good stats nor wins can make sure that you will end up fighting for the crown not mentioning winning it!

# One last thing I would like to do in this notebook is to perform a simple statistical test called **T-test**.  
# As you might know there are 2 leagues in MLB - National League and American League. As in NBA people argue that West is stronger than East, in MLB some people say that National League performs differently compared to American League.  
# 
# So let's assume that our null-hypothesis is: National League and American League teams generally perform on the same level.
# 
# Let us prepare a separate dataframe for this exercise and perform a t-test to check our null hypothesis.

# In[ ]:


nl = df[df['League'] == 'NL']
al = df[df['League'] == 'AL']


# In[ ]:


nl.head()


# In[ ]:


t_test_runs_scored = ttest_ind(nl['RS'].values, al['RS'].values, equal_var=False)


# In[ ]:


print(t_test_runs_scored)


# In[ ]:


fig = plt.figure(figsize=(16,4))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.distplot(nl['RS'], ax=ax1)
sns.distplot(al['RS'], ax=ax2)


# In[ ]:


fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(1,1,1)
sns.kdeplot(nl['RS'], ax=ax1, legend=True, shade=True, label='NL RS')
sns.kdeplot(al['RS'], ax=ax1, legend=True, shade=True, label='AL RS')


# In[ ]:


t_test_ba = ttest_ind(nl['BA'].values, al['BA'].values)


# In[ ]:


print(t_test_ba)


# In[ ]:


fig = plt.figure(figsize=(16,4))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.distplot(nl['BA'], ax=ax1)
sns.distplot(al['BA'], ax=ax2)


# In[ ]:


fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(1,1,1)
sns.kdeplot(nl['BA'], ax=ax1, legend=True, shade=True, label='NL BA')
sns.kdeplot(al['BA'], ax=ax1, legend=True, shade=True, label='AL BA')


# We have taken couple of stats and it appears that NL or National League teams perform slightly higher than American League teams. So our null hypothesis was not correct. We could also say looking at calculated p-values that the differences between NL and AL hardly happened by chance. There must be some reason behind it. What reason? This question defintely is beyond the scope of this small exercise.

# # ** Final word **
# 
# This is the end of the notebook. There is a lot more can be done with this topic. Especially if we manage to add more data, for example, [lahman](http://www.seanlahman.com/baseball-archive/statistics/) dataset with a lot of players and teams stats. But nevertheless we have managed to answer some of questions posed at the beginning and we discovered some interesting facts about baseball.  I hope you liked it, for me it was a fun exercise.  Baseball is a great game!
# 
# Again if you have any feedback for me, I would be happy to hear that and learn something new!
# 
# # ** The End **

# ![Yankees Stadium](http://blog.parkwhiz.com/wp-content/uploads/2014/08/YAnkee-stadium.jpg)
