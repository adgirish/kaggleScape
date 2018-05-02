
# coding: utf-8

# As a serving U.S. Army infantry officer I want to briefly demonstrate the burden of casualties that infantry forces bore in Vietnam (as they do in every conflict).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv('../input/VietnamConflict.csv')
data1 = data[data['BRANCH'].isin(['ARMY', 'MARINE CORPS'])]


# Now we come to the part that requires specific domain knowledge. How many of these positions are infantry? Hint: not just the ones that specifically have 'infantry' in the title. Of course, I have to exercise some judgement here since job titles have changed over the years. In general I will not include Rangers and Special Forces in this list, even though the majority of those fine folks are infantry.

# In[ ]:


infantry_MOS = ['INFANTRY OPERATIONS AND INTELLIGENCE SPECIALIST', 'INDIRECT FIRE INFANTRYMAN', 'INFANTRY UNIT LEADER', 'PARACHUTIST, INFANTRY UNIT COMMANDER', 'INFANTRYMAN', 'RIFLEMAN', 'MACHINEGUNNER', 'INFANTRY OFFICER (I)', 'ASSAULTMAN', 'HEAVY ANTI-ARMOR WEAPONS INFANTRYMAN', 'MORTARMAN', 'INFANTRY UNIT COMMANDER', 'BASIC INFANTRY OFFICER', 'RANGER, OPERATIONS AND TRAINING STAFF OFFICER (G3,A3,S3)', 'INFANTRY SENIOR SERGEANT', 'BASIC INFANTRYMAN', 'RANGER, UNIT OFFICER, TRAINING CENTER', 'RANGER, INFANTRY UNIT COMMANDER', 'RANGER', 'INFANTRY UNIT COMMANDER, (MECHANIZED)', 'LAV ASSAULTMAN', 'SCOUT-SNIPER']
infantry = data1[data1['POSITION'].isin(infantry_MOS)]
# What proportion of Vietnam casualties were infantrymen?
infantry_cas = infantry.FATALITY_YEAR.count()
total_cas = data.FATALITY_YEAR.count()
infantry_portion = infantry_cas / total_cas * 100
print('Infantrymen sustained {}% of total casualties in the Vietnam war.'.format(infantry_portion))


# No surprises here. Actually it is probable that this number is abnormally low because of the nature of the conflict. Unconventional (i.e. guerilla) wars tend to cause greater casualties among non-infantry troops because guerilla fighters deliberately attempt to attack 'soft' (i.e. not combat) targets. This has certainly been true throughout the Global War on Terror.
# 
# Incidentally, this is not meant as a veiled criticism of guerilla tactics. It is good sense to attack non-combat units because they are easier to successfully hurt.

# Now let's take a look at infantry casualties by rank as compared to the remainder of the war's casualties. I expect to see that infantry casualties tend to be both lower ranking and younger than the remainder. I don't know that this will necessarily be true, however. I just know that infantrymen tend to be young and that infantry structure has proportionately more lower-ranking individuals than other unit types.

# In[ ]:


non_infantry = data1[-data1['POSITION'].isin(infantry_MOS)]
navy_af = data[data['BRANCH'].isin(['NAVY', 'AIR FORCE'])]
by_grade = ['E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'W01', 'W02', 'W03','W04', 'W05', 'O01', 'O02', 'O03', 'O04', 'O05', 'O06', 'O07']
plt.subplot(3,1,1)
_ = sns.countplot(x='PAY_GRADE', data=infantry, order=by_grade)
_ = plt.title('Infantry Casualties (Army & Marine Corps)')
_ = plt.xlabel('Pay Grade')
_ = plt.ylabel('Number of Casualties')
plt.subplot(3,1,2)
_ = sns.countplot(x='PAY_GRADE', data=non_infantry, order=by_grade)
_ = plt.title('Non-Infantry Casualties (Army & Marine Corps)')
_ = plt.xlabel('Pay Grade')
_ = plt.ylabel('Number of Casualties')
plt.subplot(3,1,3)
_ = sns.countplot(x='PAY_GRADE', data=navy_af, order=by_grade)
_ = plt.title('Navy & Air Force Casualties')
_ = plt.xlabel('Pay Grade')
_ = plt.ylabel('Number of Casualties')
plt.tight_layout()
plt.show()    


# As expected, infantry casualties tend to occur at lower pay grades than non-infantry casualties. There is an immediately noticeable increase in the proportional number of officer casualties among non-infantry service-members (SMs) within the Army and Marine Corps. I expect this would become considerably more pronounced if I had seperated out all combat arms troops (i.e. cavalry, armor, special forces, rangers) instead of just infantry.
# 
# Within the Navy and Air Force the casualty structure is significantly different, with nearly as many deaths in the officer as in the enlisted ranks. This is likely because fixed-wing aircraft pilots are officers and the Navy and Air Force both lost good numbers of aircraft during the war. It is possible with this dataset to look into this more closely by seperating out the pilots in these two services, but for now I'm going to move on. If I get distracted I'll never finish, after all.

# In[ ]:


# I have to disable a chained assignment warning here because it keeps popping up but has not value
# to what I am actually doing here (as far as I can tell)
pd.options.mode.chained_assignment = None
# Infantry ages first...
birth = pd.Series(infantry.loc[:,'BIRTH_YEAR'].floordiv(10000), index=infantry.index)
infantry.loc[:,'BIRTH_YR'] = birth
for row in infantry:
    age_at_death = []
    birth = infantry.BIRTH_YR
    death = infantry.FATALITY_YEAR
    age = death - birth
    age_at_death.extend(age)
age = pd.Series(age_at_death, index=infantry.index)
infantry.loc[:,'AGE'] = age
# then non-infantry...
birth = pd.Series(non_infantry.loc[:,'BIRTH_YEAR'].floordiv(10000), index=non_infantry.index)
non_infantry.loc[:,'BIRTH_YR'] = birth
for row in non_infantry:    
    age_at_death2 = []
    birth = non_infantry.BIRTH_YR
    death = non_infantry.FATALITY_YEAR
    age = death - birth
    age_at_death2.extend(age)
age = pd.Series(age_at_death2, index=non_infantry.index)
non_infantry.loc[:,'AGE'] = age
# and lastly my sister services...
birth = pd.Series(navy_af.loc[:,'BIRTH_YEAR'].floordiv(10000), index=navy_af.index)
navy_af.loc[:,'BIRTH_YR'] = birth
for row in navy_af:    
    age_at_death3 = []
    birth = navy_af.BIRTH_YR
    death = navy_af.FATALITY_YEAR
    age = death - birth
    age_at_death3.extend(age)
age = pd.Series(age_at_death3, index=navy_af.index)
navy_af.loc[:,'AGE'] = age

plt.subplot(3,1,1)
_ = sns.countplot(x='AGE', data=infantry)
_ = plt.title('Infantry Age at Death')
_ = plt.xlabel('Age')
_ = plt.ylabel('Number of Fatalities')
plt.subplot(3,1,2)
_ = sns.countplot(x='AGE', data=non_infantry)
_ = plt.title('Non-Infantry Age at Death (Army & Marine Corps)')
_ = plt.xlabel('Age')
_ = plt.ylabel('Number of Fatalities')
plt.subplot(3,1,3)
_ = sns.countplot(x='AGE', data=navy_af)
_ = plt.title('Navy & Air Force Ages at Death')
_ = plt.xlabel('Age')
_ = plt.ylabel('Number of Fatalities')
plt.tight_layout()
plt.show()   

inf_mean = infantry.AGE.mean()
inf_median = infantry.AGE.median()
non_inf_mean = non_infantry.AGE.mean()
non_inf_median = non_infantry.AGE.median()
oth_svc_mean = navy_af.AGE.mean()
oth_svc_median = navy_af.AGE.median()
print('Infantry mean and median age at death are ' + str(inf_mean) + ' and ' + str(inf_median) + ' , respectively.')
print('Non-Infantry mean and median age at death are ' + str(non_inf_mean) + ' and ' + str(non_inf_median) + ' , respectively.')
print('AF/Navy mean and median age at death are ' + str(oth_svc_mean) + ' and ' + str(oth_svc_median) + ' , respectively.')


# Looks like there is little difference between the infantry and non-infantry ages at death for the Army and Marine Corps. There is a quite significant difference between those two services and the Navy and Air Force, however. It is quite clear from this analysis that the burden of bleeding for one's country falls quite disproportionately on the young in the Army and Marine Corps, but this hardly comes as a surprise.
# 
# Now I'll see how hostile and non-hostile deaths stack up between my categories of service-members.

# In[ ]:


infantry['HOSTILITY_CONDITIONS'] = infantry['HOSTILITY_CONDITIONS'].replace(['H', 'NH'], ['Hostile', 'Non-hostile'])
_ = sns.countplot(x='HOSTILITY_CONDITIONS', data=infantry)
_ = plt.title('Casualty Breakdown, Infantry')
_ = plt.xlabel('Hostility Conditions')
_ = plt.ylabel('Number of Fatalities')
plt.show()

total_deaths = infantry['HOSTILITY_CONDITIONS'].count()
hostile = infantry[infantry['HOSTILITY_CONDITIONS'] == 'Hostile']
hostile_death = hostile['HOSTILITY_CONDITIONS'].count()
non_hostile = infantry[infantry['HOSTILITY_CONDITIONS'] == 'Non-hostile']
non_hostile_death = non_hostile['HOSTILITY_CONDITIONS'].count()
non_hostile_ratio = non_hostile_death / total_deaths * 100
hostile_ratio = hostile_death / total_deaths * 100
print('Infantry SMs sustained {}% hostile casualties and {}% non-hostile casualties.'.format(hostile_ratio, non_hostile_ratio))


# In[ ]:


non_infantry['HOSTILITY_CONDITIONS'] = non_infantry['HOSTILITY_CONDITIONS'].replace(['H', 'NH'], ['Hostile', 'Non-hostile'])
_ = sns.countplot(x='HOSTILITY_CONDITIONS', data=non_infantry)
_ = plt.title('Casualty Breakdown, Non-Infantry')
_ = plt.xlabel('Hostility Conditions')
_ = plt.ylabel('Number of Fatalities')
plt.show()

total_deaths = non_infantry['HOSTILITY_CONDITIONS'].count()
ninf_hostile = non_infantry[non_infantry['HOSTILITY_CONDITIONS'] == 'Hostile']
hostile_death = ninf_hostile['HOSTILITY_CONDITIONS'].count()
ninf_non_hostile = non_infantry[non_infantry['HOSTILITY_CONDITIONS'] == 'Non-hostile']
ninf_non_hostile_death = ninf_non_hostile['HOSTILITY_CONDITIONS'].count()
non_hostile_ratio = ninf_non_hostile_death / total_deaths * 100
hostile_ratio = hostile_death / total_deaths * 100
print('Non-infantry SMs sustained {}% hostile casualties and {}% non-hostile casualties.'.format(hostile_ratio, non_hostile_ratio))


# In[ ]:


navy_af['HOSTILITY_CONDITIONS'] = navy_af['HOSTILITY_CONDITIONS'].replace(['H', 'NH'], ['Hostile', 'Non-hostile'])
_ = sns.countplot(x='HOSTILITY_CONDITIONS', data=navy_af, order=['Hostile', 'Non-hostile'])
_ = plt.title('Casualty Breakdown, Navy & Air Force')
_ = plt.xlabel('Hostility Conditions')
_ = plt.ylabel('Number of Fatalities')
plt.show()

total_deaths = navy_af['HOSTILITY_CONDITIONS'].count()
naf_hostile = navy_af[navy_af['HOSTILITY_CONDITIONS'] == 'Hostile']
hostile_death = naf_hostile['HOSTILITY_CONDITIONS'].count()
naf_non_hostile = navy_af[navy_af['HOSTILITY_CONDITIONS'] == 'Non-hostile']
naf_non_hostile_death = naf_non_hostile['HOSTILITY_CONDITIONS'].count()
naf_non_hostile_ratio = naf_non_hostile_death / total_deaths * 100
hostile_ratio = hostile_death / total_deaths * 100
print('Navy & Air Force SMs sustained {}% hostile casualties and {}% non-hostile casualties.'.format(hostile_ratio, non_hostile_ratio))


# It is interesting to note that non-infantry casualties are roughly similiar proportionately to Navy/Air Force casualties over the course of the conflict. This might indicate that there are more similarities between the experience of war between Navy/Air Force personnel and non-infantry Soldiers and Marines than there are between non-infantry Soldiers/Marines and their infantry brethren. One way to shed a little additional light on this thought is to see if there are similarities in the way in which these folks perished.

# In[ ]:


order = ['ACCIDENT', 'ILLNESS', 'HOMICIDE', 'SELF-INFLICTED', 'PRESUMED DEAD']
plt.subplot(2,1,1)
_ = sns.countplot(x='FATALITY', data=ninf_non_hostile, order=order)
_ = plt.title('Army/Marine Corps Non-infantry Non-hostile Fatalities by Category')
_ = plt.xlabel('Fatality Category')
_ = plt.ylabel('Number of Fatalities')

plt.subplot(2,1,2)
_ = sns.countplot(x='FATALITY', data=naf_non_hostile, order=order)
_ = plt.title('Navy/Air Force Non-hostile Fatalities by Category')
_ = plt.xlabel('Fatality Category')
_ = plt.ylabel('Number of Fatalities')
plt.tight_layout()
plt.show()


# Overall quite similiar profiles for non-hostile fatalities. So what about hostile ones? For this we'll have to drill a little deeper and look at actual cause of death.

# In[ ]:


plt.subplot(2,1,1)
_ = sns.countplot(x='FATALITY_2', data=ninf_hostile)
_ = plt.title('Army/Marine Corps Non-infantry Non-hostile Fatalities by Category')
_ = plt.xlabel('Fatality Category')
_ = plt.xticks(rotation=15)
_ = plt.ylabel('Number of Fatalities')

plt.subplot(2,1,2)
_ = sns.countplot(x='FATALITY_2', data=naf_hostile)
_ = plt.title('Navy/Air Force Non-hostile Fatalities by Category')
_ = plt.xlabel('Fatality Category')
_ = plt.xticks(rotation=15)
_ = plt.ylabel('Number of Fatalities')
plt.tight_layout()
plt.show()


# In[ ]:


words = ninf_hostile['FATALITY_2'].tolist()
words = str(words)

from wordcloud import WordCloud

wordcloud = WordCloud(width=1200, height=1000).generate(text=words)
plt.title('Non-infantry Hostile Fatality WordCloud')
plt.imshow(wordcloud)
plt.show()


# In[ ]:



words2 = naf_hostile['FATALITY_2'].tolist()
words2 = str(words2)
wordcloud2 = WordCloud(width=1200, height=1000).generate(text=words2)
plt.title('Navy/Air Force Hostile Fatality WordCloud')
plt.imshow(wordcloud2)
plt.show()


# I think this does a much better job of illustrating the difference in the way hostile casualties were inflicted on these forces. Clearly Navy and Air Force casualties mostly involved aircraft crashing or getting shot down while non-infantry Marine and Army casualties were more likely to be inflicted by small arms fire.
# 
# Another interesting thing to look at in this dataset might be *when* these SMs died. If we can divide deaths between the rainy and dry seasons in Vietnam we might be able to illustrate whether or not there was a 'fighting season' during this conflict. As an example, in Afghanistan today most fighting occurs during the spring and summer months. Warmer temperatures are more conducive to ill-equipped troops maneuvering and attacking U.S. and coalition forces. I am curious to know if the same was true in Vietnam. If it was it ought to be indicated pretty clearly by the distribution of hostile infantry casualties by month. So let's take a look and see what we can discover.
# 
# First I'll just look at how casualties are dispersed across the duration of the conflict. I have seen in some of the other kernels done on this dataset that there are casualties in the 2000s, which are obviously incorrect. I assume that these represent casualties that were discovered during the recent recovery efforts in Vietnam. If this is true I will have to remove these to make my analysis meaningful (since clearly I won't know which year/month these fine Americans died).

# In[ ]:


hostile['FATALITY_DATE'] = hostile['FATALITY_DATE'].floordiv(100)

_ = sns.distplot(hostile['FATALITY_DATE'])
_ = plt.title('Infantry Hostile Fire Casualties by Date')
plt.show()


# As expected.

# In[ ]:


hostile = hostile[hostile['FATALITY_DATE'] < 197505]

hostile['FATALITY_MONTH'] = hostile['FATALITY_DATE'] - 190000


# In[ ]:


hostile['FATALITY_MONTH'] = 0
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196201, 196301, 196401, 196501, 196601, 196701, 196801, 196901, 197001, 197101, 197201, 197301, 197401, 197501])] = 'January'
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196202, 196302, 196402, 196502, 196602, 196702, 196802, 196902, 197002, 197102, 197202, 197302, 197402, 197502])] = 'February'
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196203, 196303, 196403, 196503, 196603, 196703, 196803, 196903, 197003, 197103, 197203, 197303, 197403, 197503])] = 'March'
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196204, 196304, 196404, 196504, 196604, 196704, 196804, 196904, 197004, 197104, 197204, 197304, 197404, 197504])] = 'April'
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196205, 196305, 196405, 196505, 196605, 196705, 196805, 196905, 197005, 197105, 197205, 197305, 197405, 197505])] = 'May'
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196206, 196306, 196406, 196506, 196606, 196706, 196806, 196906, 197006, 197106, 197206, 197306, 197406, 197506])] = 'June'
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196207, 196307, 196407, 196507, 196607, 196707, 196807, 196907, 197007, 197107, 197207, 197307, 197407, 197507])] = 'July'
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196208, 196308, 196408, 196508, 196608, 196708, 196808, 196908, 197008, 197108, 197208, 197308, 197408, 197508])] = 'August'
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196209, 196309, 196409, 196509, 196609, 196709, 196809, 196909, 197009, 197109, 197209, 197309, 197409, 197509])] = 'September'
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196210, 196310, 196410, 196510, 196610, 196710, 196810, 196910, 197010, 197110, 197210, 197310, 197410, 197510])] = 'October'
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196211, 196311, 196411, 196511, 196611, 196711, 196811, 196911, 197011, 197111, 197211, 197311, 197411, 197511])] = 'November'
hostile['FATALITY_MONTH'][hostile['FATALITY_DATE'].isin([196212, 196312, 196412, 196512, 196612, 196712, 196812, 196912, 197012, 197112, 197212, 197312, 197412, 197512])] = 'December'


# In[ ]:


hostile['season'] = 0

hostile['season'][hostile['FATALITY_MONTH'].isin(['October', 'November', 'December'])] = 'rainy'
hostile['season'][-hostile['FATALITY_MONTH'].isin(['October', 'November', 'December'])] = 'dry'


# In[ ]:


order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
_ = sns.countplot(x='FATALITY_MONTH', hue='season', data=hostile, order=order, hue_order=['rainy', 'dry'])
_ = plt.title('Infantry Fatalities by Hostile Fire (by month)')
_ = plt.xlabel('Month')
_ = plt.ylabel('Number of Casualties')
plt.xticks(rotation=40)
plt.show()


# In[ ]:


plt.subplot(2,1,1)
_ = sns.countplot(x='FATALITY_YEAR', data=hostile)
_ = plt.title('Infantry Hostile Casualties by Year')
_ = plt.xlabel('Fatality Year')
_ = plt.ylabel('Number of Casualties')
plt.xticks(rotation=45)
plt.subplot(2,1,2)
_ = sns.countplot(x='FATALITY_YEAR', data=data)
_ = plt.title('Total Fatalities by Year')
_ = plt.xlabel('Fatality Year')
_ = plt.ylabel('Number of Casualties')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# All right, looks like my hypothesis was correct. Casualties were lower during the rainy season months than during the dry season. The highest casualties over the course of the conflict occurred during the dry season and gradually trailed off as the rainy season approached. Apparently the 'fighting season' during the Vietnam conflict occurred during the dry season, which is not particularly surprising.
# 
# I also plotted casualties by year for Infantry and the total force just for fun. Evidently 1968 was a bad year.
# 
# Well, I think that answers all of my questions for this dataset. It's been fun exploring it.  Please feel free to upvote or comment as you see fit. 
# 
# Upvotes are always appreciated!
# ===============================
