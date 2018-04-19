
# coding: utf-8

# #What kind of player will be admitted into the hall of fame in baseball?
# 
# ---  
# **Created by: Rocha**  
# **Date: April 2017**  
#   
# We can analyize the question from different perspectives. Such as:  
# 1. player's biographic data analysis.  
# 2. player's baseball skill analysis  
# 3. player's salary analysis  

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import os
from sklearn import preprocessing
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from matplotlib import pyplot as plt
pd.set_option("display.max_columns",50)


# ## hall of fame exploration
# From the data, we can see that the create of the hall of fame is from **1939** year.  

# In[ ]:


### read in data
hall_of_fame = pd.read_csv("../input/hall_of_fame.csv")
## to choose only the player 
hall_of_fame = hall_of_fame[hall_of_fame.category == "Player"]
print(hall_of_fame.info())
hall_of_fame.head()


# ### 1. Who voted the most player ? Whose vote has the power to HOF ?
# *BBWAA* vote the most player into the hall of fame, then is voted by *Veterans*, the third is voted by *Run Off*.  
# We see that the vote from the **Veterans** has the power to HOF. **Speical Election**,**Old Timers**,**Negro League** also have the great power. However, others vote don't see this phenomenen.

# In[ ]:


## calculate votedby data with inducted 
voted_by = hall_of_fame.groupby(["votedby","inducted"]).size().unstack()
norm_voted_by = pd.DataFrame(preprocessing.normalize(voted_by.fillna(0)),columns=voted_by.columns,index=voted_by.index)
norm_voted_by["delta"] = norm_voted_by["Y"] - norm_voted_by["N"]

##plot
fig,axes_voted = plt.subplots(2,1,figsize=(12,16))
colors = ["r","r","r","r","b","b","b","b"]
voted_by[["N","Y"]].plot.barh(title="Whose vote is most powerful 1 ?",ax=axes_voted[0])
norm_voted_by.delta.sort_values().plot.barh(title="Whose vote is most powerful 2 ?",                                            ax=axes_voted[1],color=colors)


# ### 2. How many people enter into the Hall Of Fame each year?
# Every year, the number of people chosen into the hall of fame is different. From the fig, we can see that:  
# 1. Some year we don't choose the popular player because of some reason.  
# 2. Some year we choose only a few people but some year we choose a lot. The **blue bar** shows the number of HOF, and the **red line** shows the HOF admit rate.

# In[ ]:


hall_of_fame[hall_of_fame.inducted == "Y"].groupby("yearid").size().describe()


# In[ ]:


## calculate the year with inducted
induct = hall_of_fame.groupby(["yearid","inducted"]).size().unstack().fillna(0)
induct["inducted rate"] = induct["Y"]/(induct["Y"]+induct["N"])

fig,axes_induct = plt.subplots(1,1,figsize=(16,10))
ax1 = induct.plot(y="inducted rate",kind="line",style="ro-",secondary_y=True,use_index=False,ax=axes_induct)
induct["Y"].plot(kind = "bar",title="number of people chosen into the HOF each year"                 ,ax=axes_induct,label="HOF",legend=True)


# ### 3. Participate the vote multi-times 
# Some people have many times take part in the elected to the hall of fame.
# A person even tried **20 times** in the ballot, that means 20 years. Most people only have **1** chance.

# In[ ]:


hall_of_fame.groupby("player_id").size().describe()


# In[ ]:


hall_of_fame.groupby("player_id").size().hist()


# ### 4. How many votes do we needed to be admitted to Hall Of Fame?
# The votes needed to enter in the hall of fame is ever increasing. In another words, with the more people have interest in baseball, the player need more vote to enter into the hall of fame

# In[ ]:


hall_of_fame[["yearid","needed"]].groupby("yearid").mean().fillna(0).plot.bar(figsize=(18,8),style = "o-",title="vote needed")


# ## Demographic Attribute

# In[ ]:


## read in player table
player = pd.read_csv("../input/player.csv",parse_dates=["debut","final_game"])
player["serviceYear"] = player["final_game"] - player["debut"]
# player["serviceYear"] = player.serviceYear.astype('timedelta64[D]')

## label the player whether they're enter into HOF
player = player.join(hall_of_fame[hall_of_fame.inducted == "Y"][["player_id","inducted"]].set_index("player_id"),                     on="player_id")
player.inducted.fillna("N",inplace=True)

print(player.info())
player.head()


# ### 5. Which name do people call most ? 
# #### Name WordCloud
# First, let's take it easy. Have a look at the first name players' most used. We see *John*, *Bill*, *Mike*, *Jim*,etc.. Do you feel interesting? Perhaps in each job, there may be a popular name.

# In[ ]:


wordcloud=WordCloud().generate_from_frequencies(player.name_first.value_counts().to_dict())
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# ### 6.  Player data Description
# We find something intersting. About **1010** people only appear once in league, there first appear is their last game. We also find a strange things, at **row 11509**, the player's service year is a minus. That's a little weird.  
# Average people, the **mean** of the service year is about **4.5** year and the person's **longest service year** is **35**.

# In[ ]:


player.describe()


# In[ ]:


player[player["serviceYear"] < pd.Timedelta("1 days")].head()


# ### 7. problem data finding 
# When we do some data exploration, we always have some snoop.  
# OK, you see we find some problem data.  
# A player's finalGame time is earlier than the debut time, that's surprsing.

# In[ ]:


player[player["serviceYear"] < pd.Timedelta("0 days")]


# ### 8. Who play the longest time ? 
# #### Do you have some interesting in who play the longest time in the baseball career?
# This one, Altrock Nicholas, serves about 30yrs, but he hasn't been admitted into the Hall Of Fame.

# In[ ]:


player.loc[player["serviceYear"].argmax()]


# ### 9. Does there any relationship bewteen the year they play and be admitted into the Hall Of Fame?
# We see that, the one enter into the hall of fame the youngest only just appear in one game, maybe there's some amazing reason behind it. We found that **1** people become the baseball hero used **less than 1 year**. Most of the player have their pass need average **17 years** fight. 

# In[ ]:


player[player.inducted == "Y"].serviceYear.astype('timedelta64[Y]').describe()


# In[ ]:


player[player.inducted == "Y"][player[player.inducted == "Y"].serviceYear.astype('timedelta64[Y]')<1]


# In[ ]:


sns.distplot(player.serviceYear.astype('timedelta64[Y]').dropna(),            kde_kws={"lw":3,"label":"Total People"})
g = sns.distplot(player[player.inducted == "Y"].serviceYear.astype('timedelta64[Y]').dropna(),             kde_kws={"lw": 3, "label": "Hall Of Fame"})
g.set_title("how many year takes player enter into the hall of fame ?")


# ### 10. The Height and The Weight
# From the fig, we can see that the **mean** of the height is about **70** inches and the **mean** of the weight is **186** pounds. They all follow pretty normal distribution.

# In[ ]:


fig,axes_w_h = plt.subplots(1,2,figsize=(18,6))
sns.distplot(player["height"].dropna(),ax=axes_w_h[0],hist_kws={"label":"Player's height histplot"})
axes_w_h[0].set_title("Player's height histplot")
axes_w_h[0].legend()
sns.distplot(player["weight"].dropna(),ax=axes_w_h[1],hist_kws={"label":"Player's weight histplot"})
axes_w_h[1].set_title("Player's weight histplot")
axes_w_h[1].legend()


# We can really see some pattern here. The ones chosen for the hall of fame. Their height and weight are in the center of the players. Maybe that's much more close to public aesthetics. Maybe they are not the best player. That need further validate.

# In[ ]:


g = sns.FacetGrid(player, hue="inducted", size=7,aspect=1.5,palette=sns.color_palette(sns.color_palette("Paired")))
g.map(plt.scatter, "height", "weight", s=50, edgecolor="white")
plt.title("What's the player admitted into the hall of fame?")
plt.legend()


# # Batting 

# In[ ]:


batting = pd.read_csv("../input/batting.csv")
print(batting.info())
batting.head()


# In[ ]:


player_batting = batting.groupby("player_id").sum().iloc[:,2:].fillna(0)
player_batting["ba"] = player_batting["h"].div(player_batting["ab"],fill_value=0)
player_batting = player_batting.join(player[["player_id","inducted"]].set_index("player_id"))


# ### 11. What about batting statics correlations?
# From the pair plot, we can see some interesting pattern:
# 1. the more **game** people play, the more **hit** they get.
# 2. the more **game** people play, the more **double hit** they get.
# 3. there's clearly positive relationship between **hit** and **double hit**.
# 4. we don't clearly see relations in **triple hit** and **home run**.
# 
# Beautiful heatmap talks all.

# In[ ]:


g = sns.pairplot(player_batting,vars=["g","h","double","triple","hr"],hue="inducted")
plt.title("batting statics pairplot")


# In[ ]:


plt.figure(figsize=(14,10))
g = sns.heatmap(player_batting.corr(),vmin=0,vmax=1,linewidths=.5,cmap="OrRd",annot=True)
g.set_title("batting statistics correlation heatmap",fontdict={"fontsize":15})


# ### 12. batting skill compare
# From the boxplot, we can see almost all the statistics feature in batting, the player admitted in hall of fame is above than the ones not admitted. 

# In[ ]:


sns.factorplot(x="variable",y="value",hue="inducted",               data = pd.melt(player_batting[["ab","r","h","inducted"]],id_vars="inducted"),               kind="box",size=10,aspect=1.5,showfliers=False)
plt.title("Compare boxplot 1")


# In[ ]:


sns.factorplot(x="variable",y="value",hue="inducted",data = pd.melt(player_batting.iloc[:,4:],id_vars="inducted")               ,kind="box",size=10,aspect=1.5,showfliers=False)
plt.title("Compare boxplot 2")


# ### batting average
# **batting average** is an important score to evalue a player, we can see clear different between the player in Hall Of Fame and the ones are not

# In[ ]:


### batting average
sns.distplot(player_batting.ba.dropna(),label= "Normal player")
g = sns.distplot(player_batting[player_batting["inducted"] == "Y"].ba.dropna(),label= "Hall Of Fame")
g.set_title("Batting Average distribution")
plt.legend()


# ## Salary  
# 
# ---  
# From the input data, we see only less than 1/4 that the salary is not null to the player.

# In[ ]:


salary = pd.read_csv("../input/salary.csv")
### join salary
player = player.join(salary.groupby(["player_id"])[["player_id","salary"]].mean(),on="player_id")
print(salary.info())
salary.head()


# In[ ]:


salary.player_id.describe()


# ### Salary time-series plot
# We can see the salary boom with the time. The barplot show the relationship between the median salary of the year and the time. From the in increasing, this could mean two things:
# - the bigger cake of the baseball game 
# - inflation

# In[ ]:


g = salary.groupby(["year"]).salary.median().plot.bar(title="salary boom")
g.set_ylabel("salary")


# ### 14. Salary difference between the normal player and Hall Of Fame  
# There's a clearly difference between the player admitted into the Hall Of Fame and the normal person.
# However,there're really enough outliers in the one not in HOF. In other words, they have very high salary, but that's not enough to let them enter into the HOF. At last, we could conclude that having high salary will give the player higher possibility admitted into HOF.

# In[ ]:


player.salary.describe()


# In[ ]:


fig,axes_salary = plt.subplots(1,2,figsize=(20,8))
sns.distplot(player[player["inducted"] == "N"].salary.dropna(),label= "Normal player",ax=axes_salary[0])
sns.distplot(player[player["inducted"] == "Y"].salary.dropna(),label= "Hall Of Fame",ax=axes_salary[0])
g = sns.boxplot(x="inducted",y="salary",data=player,ax=axes_salary[1])
axes_salary[0].legend()



# ## Awards
# 
# ---  
# 
# Only less than 1/10 player were awarded, let's check their relationship with the HOF.
# 

# In[ ]:


### read in award data
awards = pd.read_csv("../input/player_award.csv")
### label player with inducted data
awards = awards.join(player[["player_id","inducted"]].set_index("player_id"),on="player_id")
award_count = awards.groupby("player_id").size()
award_count.name = "award_count"
### label the number of awards to payer table
player = player.join(award_count,on="player_id")
print(awards.info())
awards.head()


# In[ ]:


awards.player_id.describe()


# We see most people just be awarded **1-2** times, but there really one person is awarded for **47** times. That's really a legend.

# In[ ]:


awards[awards["player_id"] == "bondsba01"]


# In[ ]:


awards.groupby("player_id").size().describe()


# In[ ]:


player.award_count.plot.hist()


# ### 15. Which Award is really the pass to HOF?
# The following fig shows that *Triple Crown* and *TSN Guild MVP* are really big passes to HOF. *Baseball Magazine All-Star* takes the most part of award and is also a good pass to HOF. There're also some irrelvent awarded such as *Golden Glove*, *Silver Slugger*. Awarded these prizes are not passes to HOF.

# In[ ]:


awarded = awards.groupby(["award_id","inducted"]).size().unstack()
awarded = awarded.fillna(0)
awarded["delta"] = awarded["Y"] - awarded["N"]
awarded["ratio"] = awarded["Y"]/awarded["N"]

fig,axes_award = plt.subplots(2,1,figsize=(14,20))
awarded.sort_values(by="delta",ascending=True)[["N","Y"]].plot(kind="barh",ax=axes_award[0]                                                               ,title="Pass of Awarded into HOF")
awarded.ratio.sort_values().plot.barh(ax=axes_award[1],title = "Admited into HOF ratio compare")


# ### 16. Have more awarded is a good signal to be admited into HOF?
# The logic seems to be true that have more awarded maybe a good signal to be admitted into HOF. The boxplot shows that: player in HOF really have more awards than the players haven't. But there always be some exception, for example, the one have 47 awarded.

# In[ ]:


sns.boxplot(x="inducted",y="award_count",data=player)


# # Conclusion and Future Work
# 
# ---
# 
# ## Conclusion
# Finally, we come to our journey end, there still have a lot of things to mining, but i will stop here.  
# 1. First we must learn what happening in Hall Of Fame, so we take an exploration in it. We found if we voted by somebody like **Veterans**, we will have a higher oppotunity to the HOF. Some year we really have a high rate of being admitted into HOF, the time is also a very important factor to HOF.
# 2. We have a explortion in player's biographic data and we found to be a member HOF really all have around 10 years experience and their body infomation shouldn't not be an outlier, to cater our fans passion.  
# 3. To be a HOF, we see the member of HOF have excellent skills to outperform other normal players. In the battlefield, skill talks.  
# 4. A member of HOF may not take the highest salary, however, because they should have magic sleeves, their salary are surely above most people. But high salary does not qualify them to HOF.
# 5. The one in the HOF may not have a lot awrards, but some award such as **Triple Crown** almost all admitted into HOF. In our imagnation, player hits many triple hit or homerun is truly a legend.  
# 
# ## Drawback declartion  
# There should be some mistakes in this project such as:  
# 1. dealing with the missing values, when there's NAN, i always use 0 to fill it.
# 2. The variables could have correlation, so not all the conclusion are causation inference but correlation.  
# 
# ## Future Work 
# Truly, the dataset is very huge, there're still a lot info we can mining, for example:  
# 1. player's performance is a huge goldmine.  
# 2. their team performance  
# 
# I will stop here. Thank for the people reading my work.
# 
