
# coding: utf-8

# Celebrity Death Analysis
# ========================
# The year 2016, left people with unfortunate sorrows as a lot of Celebrities died. Let's sneak why it happened?
# 
# If this EDA helps you, make sure to leave an upvote to motivate me to make more! :)
# 
# > **Part I : Confirming the hypothesis : Many Celebrities died in 2016**
# 
# > 1. Does the number of celebrities death is highest in 2016?
# 
# > **Part II : Finding the major causes for death in 2016**
# 
# > 1. What's the reason for celebrities death in 2016?
# 
# > **Part III : Going deep with the cause of death in 2016**
# 
# > 1. How the top causes of death in 2016 are different from previous years?
# > 2. Which country has the unusual increase in celebrities death due to Cancer in 2016? 
# > 3. Which type of Cancer was the villain to most of the Celebrities in 2016?
# 
#  > **Part IV : Going deep with month of death in 2016**
# 
#  >1. Is there something interesting in the number of deaths by month in 2016?
#  >2. How the main causes of deaths in March 2016 different from previous years? 
#  >3. Which countries has the different pattern of death in March 2016 when compared to previous years?
# 
#  >**Part V : Going deep with age_group in 2016**
#  
#  >1. Do the age group of celebrities died in 2016 is different from previous years?
#  
#  >**Part VI : Going deep with nationality in 2016**
#  
#  > 1. Is there any unusual increase in celebrities death of any particular country in 2016? 
# 
# >**Part VII : Going deep with Fame Score Parameter in 2016**
# 
# > Is there something interesting relation with the Fame Score and  Celebrities death in 2016?
# 
# >**Part VIII : General questions on deaths from  2006 - 2016**
#  
#  >1. Do most of the celebrity die during their young age or old age?
#  >2. What would be the main causes of death?
#  >3. What would be the main causes of death for each age category?
# 
# >**Conclusion : Why more Celebrities died in 2016?**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Looking into dataset

# In[ ]:


death = pd.read_csv("../input/celebrity_deaths_3.csv")
death.head(2)


# In cause_of_death column, many related terms are named differently. So grouped the related terms.

# In[ ]:


def group_deathcause(cause):
    mod_cause = ""
    cause = str(cause)
    if "cancer" in cause:
        mod_cause = "cancer"
    elif "heart" in cause or "cardiac" in cause:
        mod_cause = "heart disease"
    elif "stroke" in cause :
        mod_cause = "stroke"
    elif "diabetes" in cause:
        mod_cause = "diabetes"
    elif "gunshot" == cause:
        mod_cause = "shot"
    elif "suicide" in cause:
        mod_cause = "suicide"
    else:
        mod_cause = cause
    return mod_cause.strip()


# This function is to categorise the age of the celebrity as **Old**, **Adult**, **Young** and **Child**.

# In[ ]:


def age_categorizer(age):
    category = ""
    if (age<18):
        category = "child"
    elif (age<30):
        category = "young"
    elif (age<60) :
        category = "adult"
    else:
        category = "old"
    return category


# In[ ]:


def fam_filtering(row):
    op = ""
    if type(row) == str:
        if  ( "(" in row) and (")" in row):
            row = row[:row.index("(")] + row[row.index(")")+1 : ]
        op = row
    else:
        op = "unknown"
    return op
def fam_grp(row):
    op = ""
    row = str(row)
    if any( ext in row for ext in ["minister","politician", "political","parliament","secretary"]):
        op = "politician"
    elif "actress" in row:
        op = "actress"
    elif "actor" in row:
        op = "actor"
    elif "singer" in row:
        op = "singer"
    else:
        op = row
    return op.strip().lower()


# This all started with the question *"Why more celebrities died in 2016?"* . As we are going to investigate on it , we split the data into **deaths in 2016** and **deaths before 2016**

# In[ ]:


#data_cleaning
death_all = death.copy()
death_all = death_all.drop(death_all[death_all["age"] == 0 ].index)
death_all["cause_of_death"].fillna("unknown",inplace=True)
death_all["cause_of_death"] = death_all.apply (lambda row:group_deathcause(row["cause_of_death"]) , axis = 1)
death_all["age_category"] = death_all.apply (lambda row: age_categorizer (row["age"]),axis=1)
death_all["fame_tune"] = death_all.apply(lambda row: fam_grp(row["famous_for"]), axis=1)
death_all["fame_tune"] = death_all.apply(lambda row: fam_filtering(row["fame_tune"]), axis=1)
death_all["fame_tune"] = death_all.apply(lambda row: fam_grp(row["fame_tune"]), axis=1)
death_all["wiki_length"] = death_all.fame_score
death_all.wiki_length = death_all.wiki_length.fillna(0)
death_2016 = death_all[death_all.death_year == 2016]
death_rest_2016 = death_all[death_all.death_year != 2016]


# Part I : Confirming the hypothesis : Many Celebrities died in 2016
# ======================================================================
# 
# ----------

# ## Does the number of celebrities death is highest in 2016 ? ##
# Let's see !

# In[ ]:


sns.countplot(death_all.death_year)
plt.title("Number of Celebrities death every year")


# Yes, Its true ! 

# Part II : Finding the major causes for death in 2016
# ====================================================
# 
# ----------

# ## What's the reason for celebrities death in 2016? ##

# In[ ]:


def known_cod(df,grp_column):
    known_cod_df = df[df.cause_of_death != "unknown"].groupby(grp_column)["name"].count().reset_index()
    known_cod_df = known_cod_df.rename(columns ={"name":"count"})
    total = known_cod_df["count"].sum()
    known_cod_df["ratio"] = known_cod_df.apply(lambda row: row["count"]/total,axis=1)
    known_cod_df= known_cod_df.sort_values(by="ratio",ascending=False)
    return known_cod_df


# In[ ]:


dy = known_cod(death_2016,"cause_of_death").head(10)
plt.figure(figsize=(6,4))
ax = sns.barplot(x="count", y ="cause_of_death", data = dy)
ax.set(xlabel='number_of_deaths')
sns.despine(left=True, bottom=True)
plt.title("Top 10 (65%) reasons of cause of celebrities death in 2016")


# So the main reason is ***Cancer ! Cancer ! Cancer !*** 

# Part III : Going deep with the cause of death in 2016
# -------------
# ----------

# ##1. How the top causes of death in 2016 are different from previous years? ##

# In[ ]:


known2016rest = known_cod(death_rest_2016,"cause_of_death").rename(columns={"ratio":"before2016ratio"})[["cause_of_death","before2016ratio"]]
known2016 = known_cod(death_2016,"cause_of_death").head(10)
cod_ratio = pd.merge(known2016, known2016rest, on="cause_of_death",how="left")
ax = sns.pointplot(x="cause_of_death", y="ratio", data=cod_ratio,color="#bb3f3f", label="2016")
ax = sns.pointplot(x="cause_of_death", y="before2016ratio", data=cod_ratio,color="#4286f4", label="before_2016")
ax.set_xticklabels(rotation=90, labels=cod_ratio.cause_of_death)
ax.set(ylabel='ratio')
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color="#bb3f3f", label='2016')
blu_patch = mpatches.Patch(color="#4286f4", label='until 2016')
plt.legend(handles=[red_patch,blu_patch])
plt.title("Average number of Celebrities died by cause of death")
ax.text(4.0, 0.37, "Top 10 Causes for death of Celebrities died in 2016", ha ='left', fontsize = 6)
plt.show()


# This shows the reason behind more celebrities death in 2016 is due to more **Cancer** and **traffic collision** deaths when compared to previous years. The other reasons moreover follow the same pattern.

# ## 2. Which country has the unusual increase in celebrities death due to Cancer in 2016? ##

# In[ ]:


known2016rest = known_cod(death_rest_2016[death_rest_2016.cause_of_death == "cancer"],"nationality").rename(columns={"ratio":"before2016ratio"})[["nationality","before2016ratio"]]
known2016 = known_cod(death_2016[death_2016.cause_of_death == "cancer"],"nationality").head(10)
cod_ratio = pd.merge(known2016, known2016rest, on="nationality",how="left")
ax = sns.pointplot(x="nationality", y="ratio", data=cod_ratio,color="#bb3f3f", label="2016")
ax = sns.pointplot(x="nationality", y="before2016ratio", data=cod_ratio,color="#4286f4", label="before_2016")
ax.set_xticklabels(rotation=90, labels=cod_ratio.nationality)
ax.set(ylabel='ratio')
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color="#bb3f3f", label='2016')
blu_patch = mpatches.Patch(color="#4286f4", label='until 2016')
plt.legend(handles=[red_patch,blu_patch])
plt.title("Average number of Celebrities died because of Cancer by nationality")
ax.text(4.0, 0.56, "Top 10 Causes for death of Celebrities died in 2016", ha ='left', fontsize = 6)
plt.show()


# Interesting ! **English Celebrities** died at more unusual rate because of Cancer in 2016. 

# ## 3. Which type of Cancer was villain to most of Celebrities in 2016 ? ##

# The deaths due to cancer in the year 2016 is filtered. The general Cancer cause is removed from the list.

# In[ ]:


death_copy2 = death.copy()
death_copy2["cause_of_death"].fillna("unknown",inplace=True)

#selecting all rows re
x =death_copy2[ (death_copy2["cause_of_death"].str.contains("cancer") )&( death_copy2["death_year"] == 2016 )].groupby("cause_of_death")["name"].count().sort_values(ascending=False).reset_index().rename(columns={"name":"count"}).iloc[1:,:]
x_sum = x["count"].sum()
x["ratio"] = x.apply(lambda row : row["count"]/x_sum ,axis=1)
xtop_20 = x.head(20)

plt.figure(figsize=(4,6))
ax=sns.barplot(y="cause_of_death",x="ratio",data=xtop_20)
ax.set(xlabel='ratio_of_deaths')
plt.title("Top 20 different types of cancer which caused Celebrities Death in 2016  ")
plt.show()


# Part IV : Going deep with month of death in 2016
# ================================================
# 
# ----------

# ## 1. Is there something interesting in the number of deaths by month in 2016? ##
# Let's see !

# In[ ]:


month_lst = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
mnth_index = []
death_by_month = death_2016.groupby("death_month")["name"].count().reset_index().rename(columns={"name":"2016"})
for mnth in month_lst:
    mnth_index += death_by_month[death_by_month.death_month == mnth].index.tolist()
death_by_month_rest = death_rest_2016.groupby("death_month")["name"].count().reset_index().rename(columns={"name":"before2016"})
death_by_month = pd.merge(death_by_month,death_by_month_rest, on="death_month", how="left")
death_by_month = death_by_month.reindex(mnth_index)
tot_2016_deaths = death_by_month["2016"].sum()
tot_before2016_deaths = death_by_month["before2016"].sum()
death_by_month["2016_ratio"] = death_by_month.apply(lambda row: row["2016"]/tot_2016_deaths , axis=1)
death_by_month["before2016_ratio"] = death_by_month.apply(lambda row: row["before2016"]/tot_before2016_deaths , axis=1)
ax = sns.pointplot(x="death_month", y="2016_ratio", data=death_by_month,color="#bb3f3f", label="2016")
ax = sns.pointplot(x="death_month", y="before2016_ratio", data=death_by_month,color="#4286f4", label="before_2016")
ax.set_xticklabels(rotation=90, labels=death_by_month.death_month)
ax.set(ylabel='ratio')
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color="#bb3f3f", label='2016')
blu_patch = mpatches.Patch(color="#4286f4", label='until 2016')
plt.legend(handles=[red_patch,blu_patch])
plt.title("Average number of Celebrities death by month")
plt.show()


# In 2016, **March** and **December** months has an unusual increase in celebrities death when compared to average deaths until 2016

# ##2. How the main causes of deaths in March, 2016 different from previous years? ##

# In[ ]:


known2016rest = known_cod(death_rest_2016[death_rest_2016.death_month =="March"],"cause_of_death").rename(columns={"ratio":"before2016ratio"})[["cause_of_death","before2016ratio"]]
known2016 = known_cod(death_2016[death_2016.death_month =="March"],"cause_of_death").head(10)
cod_ratio = pd.merge(known2016, known2016rest, on="cause_of_death",how="left")
ax = sns.pointplot(x="cause_of_death", y="ratio", data=cod_ratio,color="#bb3f3f", label="2016")
ax = sns.pointplot(x="cause_of_death", y="before2016ratio", data=cod_ratio,color="#4286f4", label="before_2016")
ax.set_xticklabels(rotation=90, labels=cod_ratio.cause_of_death)
ax.set(ylabel='ratio')
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color="#bb3f3f", label='2016')
blu_patch = mpatches.Patch(color="#4286f4", label='until 2016')
plt.legend(handles=[red_patch,blu_patch])
plt.title("Average number of Celebrities died in March by cause of death")
ax.text(4.0, 0.33, "Top 10 Causes for death of Celebrities died in 2016", ha ='left', fontsize = 6)
plt.show()


# This shows that in March 2016, there is **major increase** in deaths by **Cancer** and also **minor increase** in deaths by **Plane crash, Leukemia, Traffic collision** .

# ## 3. Which countries has different pattern of death in March 2016 when compared to previous years ?##

# In[ ]:


known2016rest = known_cod(death_rest_2016[death_rest_2016.death_month =="March"],"nationality").rename(columns={"ratio":"before2016ratio"})[["nationality","before2016ratio"]]
known2016 = known_cod(death_2016[death_2016.death_month =="March"],"nationality").head(10)
cod_ratio = pd.merge(known2016, known2016rest, on="nationality",how="left")
ax = sns.pointplot(x="nationality", y="ratio", data=cod_ratio,color="#bb3f3f", label="2016")
ax = sns.pointplot(x="nationality", y="before2016ratio", data=cod_ratio,color="#4286f4", label="before_2016")
ax.set_xticklabels(rotation=90, labels=cod_ratio.nationality)
ax.set(ylabel='ratio')
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color="#bb3f3f", label='2016')
blu_patch = mpatches.Patch(color="#4286f4", label='until 2016')
plt.legend(handles=[red_patch,blu_patch])
plt.title("Average number of Celebrities died in March by Nationality")
ax.text(4.0, 0.55, "Top 10 Nationalities of Celebrities died in 2016", ha ='left', fontsize = 6)
plt.show()


# Surprising ! **America** has **fewer deaths in March** when compared to other countries in the top list!

# Part V : Going deep with age_group in 2016
# ==========================================
# 
# ----------

# ## Do the age group of celebrities died in 2016 is different from previous years ? ##

# In[ ]:


x = death_2016.groupby("age_category")["name"].count()
y = death_rest_2016.groupby("age_category")["name"].count()
#plt.title("")
colors = ["tomato","orange","aqua","hotpink"]
f = plt.figure(figsize=(8,6))
the_grid = GridSpec(2, 1)
plt.subplot(the_grid[0,0], aspect=1)
plt.pie(x, labels =x.index, autopct='%1.1f%%', startangle=340 , colors = colors)
plt.axis('equal')
plt.title("2016",fontweight="bold")
plt.tight_layout()
plt.subplot(the_grid[1,0], aspect=1)
plt.pie(y, labels =y.index, autopct='%1.1f%%', startangle=340 , colors = colors)
plt.axis('equal')
plt.title("before 2016",fontweight="bold")
plt.tight_layout()
f.suptitle("Composition of age group of Celebrities died in 2016 and before 2016",y=1.08)
f.text(0.9, 0.98, "old   : [61 +]", ha ='left', fontsize = 7)
f.text(0.9, 0.95, "adult : [30 - 60]", ha ='left', fontsize = 7)
f.text(0.9, 0.92, "young : [18 - 29]", ha ='left', fontsize = 7)
f.text(0.9, 0.89, "child : [0 - 17]", ha ='left', fontsize = 7)
plt.show()


# This show that in 2016 there is **decline** in the Celebrities death of **Adult age group** and **incline** in **Young age group** .

# Part VI : Going deep with nationality in 2016
# =============================================
# 
# ----------

# ## Is there any unusual increase in celebrities death of any particular country in 2016? ##

# In[ ]:


d_16 = death_2016.groupby("nationality")["name"].count().sort_values(ascending=False).reset_index().rename(columns={"name":"count"})
tot = d_16["count"].sum()
d_16["2016_ratio"] = d_16.apply(lambda row : row["count"]/tot,axis =1)
d_rest = death_rest_2016.groupby("nationality")["name"].count().sort_values(ascending=False).reset_index().rename(columns={"name":"count"})
tot = d_rest["count"].sum()
d_rest["before2016_ratio"] = d_rest.apply(lambda row : row["count"]/tot,axis =1)

march_nationality = pd.merge(d_16,d_rest[["nationality","before2016_ratio"]],on="nationality",how="left")
march_nationality = march_nationality.head(15)


# In[ ]:


ax = sns.pointplot(x="nationality", y="2016_ratio", data=march_nationality,color="#bb3f3f", label="2016")
ax = sns.pointplot(x="nationality", y="before2016_ratio", data=march_nationality,color="#4286f4", label="before_2016")
ax.set_xticklabels(rotation=90, labels=march_nationality.nationality)
ax.set(ylabel='ratio')
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color="#bb3f3f", label='2016')
blu_patch = mpatches.Patch(color="#4286f4", label='until 2016')
plt.legend(handles=[red_patch,blu_patch])
plt.title("Average number of Celebrities died by nationality")
ax.text(6.0, 0.46, "Top 15 Nationality of Celebrities who died in 2016", ha ='left', fontsize = 6)
plt.show()


# No ! The Celebrities death following the same pattern like previous years. 

# Part VII : Going deep with Fame Score Parameter in 2016
# =============================================
# 
# ----------

# ## Is there something interesting relation with the Fame Score and  Celebrities death in 2016?##

# In[ ]:


def known_cod_mean(df):
    known_cod_df = df[df.cause_of_death != "unknown"].groupby("death_month")["wiki_length"].mean().reset_index()
    #known_cod_df = known_cod_df.rename(columns ={"name":"count"})
    total = known_cod_df["wiki_length"].sum()
    known_cod_df["ratio"] = known_cod_df.apply(lambda row: row["wiki_length"]/total,axis=1)
    known_cod_df= known_cod_df.sort_values(by="ratio",ascending=False)
    return known_cod_df

known2016rest = known_cod_mean(death_rest_2016).rename(columns={"ratio":"before2016ratio"})[["death_month","before2016ratio"]]
known2016 = known_cod_mean(death_2016).head(10)
cod_ratio = pd.merge(known2016, known2016rest, on="death_month",how="left")
mnth_index = []
for mnth in month_lst:
    mnth_index += cod_ratio[cod_ratio.death_month == mnth].index.tolist()
cod_ratio =cod_ratio.reindex(mnth_index)
ax = sns.pointplot(x="death_month", y="before2016ratio", data=cod_ratio,color="#4286f4", label="before_2016")
ax = sns.pointplot(x="death_month", y="ratio", data=cod_ratio,color="#bb3f3f", label="2016")
ax.set_xticklabels(rotation=90, labels=cod_ratio.death_month)
ax.set(ylabel='ratio')
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color="#bb3f3f", label='2016')
blu_patch = mpatches.Patch(color="#4286f4", label='until 2016')
plt.legend(handles=[red_patch,blu_patch])
plt.title("Celebrities Wiki length Mean for every month")
#ax.text(4.0, 0.55, "Top 10 Nationalities of Celebrities died in 2016", ha ='left', fontsize = 6)
plt.show()


# This shows that highly famed Celebrities has died in the month of **March, April, June and November**. 

# Part VIII : General questions on deaths from 2006 - 2016
# =======================================================
# 
# ----------

# ##1. Does most of the celebrity die during their young age or old age ? ##
# Let's see

# In[ ]:


sns.boxplot(death.age)


# This shows that 50% of celebrities die in age interval of 69 to 87. So, most of the celebrities die in their old age.

# ## 2. What would be the main causes of death ?##
# Let's see

# In[ ]:


death_cause = death_all.groupby("cause_of_death")["name"].count().sort_values(ascending=False)
#death_cause.head(20)
comp = death_cause.ix[1:20]
y = death_cause.ix[21:].sum()
comp['others'] = y
plt.figure(figsize=(5,5))
plt.pie(comp,labels=comp.index, autopct='%1.1f%%', startangle=310 )
plt.tight_layout()
plt.axis('equal')
plt.title("Composition of known cause of death",y=1.08,fontweight="bold")


# The composition chart shows that **Cancer** and **Heart diseases** are the main reasons for celebrities death. Also, most of the causes in top20 are **Natural**

# ##3. What would be the main causes of death for each age category? ##
# Let's see !

# In[ ]:



age_category_rep =death_all.groupby(["age_category","cause_of_death"])["name"].count().sort_values(ascending=False)
f = plt.figure(figsize=(8,15))
the_grid = GridSpec(4, 1)
for cat in [("child",0,0),("young",1,0),("adult",2,0),("old",3,0)]:
    x = age_category_rep[cat[0]][1:10]
    y = age_category_rep[cat[0]][11:].sum()
    plt.subplot(the_grid[cat[1],cat[2] ], aspect=1)
    x["others"] = y    
    plt.pie(x, labels =x.index, autopct='%1.1f%%', startangle=10 )
    plt.axis('equal')
    plt.title(cat[0],y=1.08,fontweight="bold")
    plt.tight_layout()
f.suptitle("Composition of known cause of death for every category",y=1.03)
plt.show()


# The above composition shows an interesting result.
# 
#  1. **Child category** : 15% deaths are due Euthanized **[Mostly Natural deaths]**
#  2. **Young category** : the top causes of death are unnatural like traffic collision, suicide, shot etc.,**[ Mostly Unnatural deaths]**
#  3. **Adult category** : the top causes of death are like cancer, traffic collision etc., **[Mix of Natural and Unnatural deaths]**
#  4. **Old category** : as expected **[Mostly Natural deaths]**

# Conclusion
# =======================================================
# 
# ----------

# 
# Conclusion
# ==========
# 
# **Why more Celebrities died in 2016?**
# 
# The first thought came to my mind is that every year there will be an increase in the number of celebrities so there is nothing special in 2016. On contrary, I also had another thought if it would have been normal then why people talk more about it on twitter with hashtag [#celebritydeaths2016](https://twitter.com/hashtag/celebritydeaths2016?src=hash).
# 
# I began my exploration to clarify my thoughts. 
# 
# My first thought gained upper hand as my first visualization said that the number of celebrities death in 2016 was not considerably so high.
# 
# I went deep to know the reason behind the death of celebrities in 2016. It uncovered the fact that more celebrities died in 2016 due to Cancer and also the ratio of cancer deaths are much higher than previous years. Again this analysis didn't say much about why people had thought that 2016 took the life of more celebrities.
# 
# Then I went deep to know about patterns in celebrities death by month. This time my second thought gained upper hand because the analysis says the ratio of celebrities death in the last two month of the year where higher than the average ratio until previous years.
# 
# This pushed me to go deeper to check for some additional proofs for my second thought. This time I took age group. As I expected the year 2016 has slightly more young celebrities death than average. The popularity of any news is based on social media and social media is manipulated by youngsters. So this made my second thought stronger.
# 
# And then, the dataset provider added the column of fame score which was the length of Wikipedia page of celebrities. Another user made magnificent [EDA][1] on the same dataset and proved that the fame score does not have the correlation with age and years of celebrities. So I was good to go with fame score. This analysis made my second thought stronger as the mean fame score of celebrities died in 2016 was higher than the previous years. The important point I would feel is that the mean fame score of celebrities died in November is higher than the previous year and also as mentioned earlier November and December have more ratio of deaths in 2016 than previous years. 
# 
# I conclude by saying that even though the year 2016 doesn't have huge celebrities death as we thought, the death of highly famed celebrities and more deaths in the last two months of the year made people talk and feel about celebrities.
# 
# 
# > "*Great things happened in this world,
# > by simple motivations*".
# 
# > If this EDA
# > helps you, make sure to leave an
# > upvote to motivate me to make more! :)
# 
# 
# [1]: https://www.kaggle.com/drgilermo/d/hugodarwood/celebrity-deaths/dead-celebrities-a-deeper-look
