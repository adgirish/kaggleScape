
# coding: utf-8

# In this kernel I have done data visualisation of the IPL datasets using pandas,Matplotlib and seaborn library and plotted some interesting observations.

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


#Loading all the necessary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualisation
import seaborn as sns #for visualisation
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Reading data from Comma Separated Values files
deliveries=pd.read_csv("../input/deliveries.csv")
matches=pd.read_csv("../input/matches.csv")


# In[ ]:


col_names=matches.columns.tolist()
print("column names:")
print(col_names)


# In[ ]:


print("Sample data:")
deliveries.head(6)


# In[ ]:


matches.head(6)


# In[ ]:


#matches.info()
matches.describe()


# In[ ]:


#matches['player_of_match'].head()


# In[ ]:


player_of_the_match= pd.pivot_table(matches,values=['player_of_match'],index=['season'],columns=['city'],aggfunc='count',margins=False)

plt.figure(figsize=(10,10))
sns.heatmap(player_of_the_match['player_of_match'],linewidths=.5,annot=True,vmin=0.01,cmap='YlGnBu')
plt.title('Number of player of the match in cities for particular year')


# In[ ]:


big_margin=matches[(matches['win_by_runs']>=50) | (matches['win_by_wickets']>=8)]
big_margin
big_margin.winner.unique()


# In[ ]:


KKR=big_margin[big_margin['winner']=='Kolkata Knight Riders'].count()
MI=big_margin[big_margin['winner']=='Mumbai Indians'].count()
KXP=big_margin[big_margin['winner']=='Kings XI Punjab'].count()
CSK=big_margin[big_margin['winner']=='Chennai Super Kings'].count()
DC=big_margin[big_margin['winner']=='Deccan Chargers'].count()
DD=big_margin[big_margin['winner']=='Delhi Daredevils'].count()
RC=big_margin[big_margin['winner']=='Royal Challengers Bangalore'].count()
KT=big_margin[big_margin['winner']=='Kochi Tuskers Kerala'].count()
SH=big_margin[big_margin['winner']=='Sunrisers Hyderabad'].count()
RPS=big_margin[big_margin['winner']=='Rising Pune Supergiants'].count()
RR=big_margin[big_margin['winner']=='Rajasthan Royals'].count()


# In[ ]:


KKR_winner=KKR['winner']
MI_winner=MI['winner']
KXP_winner=KXP['winner']
CSK_winner=CSK['winner']
DC_winner=DC['winner']
DD_winner=DD['winner']
RC_winner=RC['winner']
KT_winner=KT['winner']
SH_winner=SH['winner']
RPS_winner=RPS['winner']
RR_winner=RR['winner']


# In[ ]:


winners=pd.Series([KKR_winner,SH_winner,KXP_winner,RC_winner,DC_winner,DD_winner,CSK_winner,KT_winner,MI_winner,RPS_winner,RR_winner],index=['KKR','SH','KXP','RC','DC','DD','CSK','KT','MI','RPS','RR'])
winners_df=pd.DataFrame(winners,columns=['No. of wins by big margin'])
winners_df


# In[ ]:


labels = ['KKR','SH','KXP','RC','DC','DD','CSK','KT','MI','RPS','RR']
sizes = [14, 4, 9, 21, 5, 14 , 20 ,2 ,18 ,1 , 17]
colors = ['#EEE5DE','#FFC1C1','#FF8247','#FF3030','#FFEC8B','#9AC9CB','#FFFFE0','#C0FF3E','#00FFFF','#FFFF00','#C4C4C4']
explode = (0, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0)  # only "explode" the 7th slice (i.e. 'RC')
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90,radius=1.5)
plt.title("Winning percentage of teams by large margin(more than 50 runs or by more than 8 wickets)",x=0,y=-0.3)

#Percentage of wins by team which is greater than or equal to 50 runs or by grater than or equal to 8 wickets


# In[ ]:


matches['player_of_match'].unique()
player_of_match=[]
for p in matches['player_of_match'].unique():
    player_of_match.append({'Player': p})
    
deliveries['batsman'].unique()
No_of_batsman=[]
for b in deliveries['batsman'].unique():
    No_of_batsman.append({'Batsman':b})


# In[ ]:


pd.DataFrame(player_of_match)


# In[ ]:


pd.DataFrame(No_of_batsman)


# In[ ]:


matches_played_KKR=matches[(matches['team1']=='Kolkata Knight Riders') | (matches['team2']=='Kolkata Knight Riders')]
matches_played_MI=matches[(matches['team1']=='Mumbai Indians') | (matches['team2']=='Mumbai Indians')]
matches_played_KXP=matches[(matches['team1']=='Kings XI Punjab') | (matches['team2']=='Kings XI Punjab')]
matches_played_CSK=matches[(matches['team1']=='Chennai Super Kings') | (matches['team2']=='Chennai Super Kings')]
matches_played_DC=matches[(matches['team1']=='Deccan Chargers') | (matches['team2']=='Deccan Chargers')]
matches_played_DD=matches[(matches['team1']=='Delhi Daredevils') | (matches['team2']=='Delhi Daredevils')]
matches_played_RCB=matches[(matches['team1']=='Royal Challengers Bangalore') | (matches['team2']=='Royal Challengers Bangalore')]
matches_played_KT=matches[(matches['team1']=='Kochi Tuskers Kerala') | (matches['team2']=='Kochi Tuskers Kerala')]
matches_played_SH=matches[(matches['team1']=='Sunrisers Hyderabad') | (matches['team2']=='Sunrisers Hyderabad')]
matches_played_RPS=matches[(matches['team1']=='Rising Pune Supergiants') | (matches['team2']=='Rising Pune Supergiants')]
matches_played_RR=matches[(matches['team1']=='Rajasthan Royals') | (matches['team2']=='Rajasthan Royals')]

A=matches_played_KKR['id'].count()
B=matches_played_MI['id'].count()
C=matches_played_KXP['id'].count()
D=matches_played_CSK['id'].count()
E=matches_played_DC['id'].count()
F=matches_played_DD['id'].count()
G=matches_played_RCB['id'].count()
H=matches_played_KT['id'].count()
I=matches_played_SH['id'].count()
J=matches_played_RPS['id'].count()
K=matches_played_RR['id'].count()

A


# In[ ]:


matches_won_KKR=matches[matches['winner']=='Kolkata Knight Riders']
matches_won_MI=matches[matches['winner']=='Mumbai Indians']
matches_won_KXP=matches[matches['winner']=='Kings XI Punjab']
matches_won_CSK=matches[matches['winner']=='Chennai Super Kings']
matches_won_DC=matches[matches['winner']=='Deccan Chargers']
matches_won_DD=matches[matches['winner']=='Delhi Daredevils']
matches_won_RCB=matches[matches['winner']=='Royal Challengers Bangalore']
matches_won_KT=matches[matches['winner']=='Kochi Tuskers Kerala']
matches_won_SH=matches[matches['winner']=='Sunrisers Hyderabad']
matches_won_RPS=matches[matches['winner']=='Rising Pune Supergiants']
matches_won_RR=matches[matches['winner']=='Rajasthan Royals']


O=matches_won_KKR['id'].count()
P=matches_won_MI['id'].count()
Q=matches_won_KXP['id'].count()
R=matches_won_CSK['id'].count()
S=matches_won_DC['id'].count()
T=matches_won_DD['id'].count()
U=matches_won_RCB['id'].count()
V=matches_won_KT['id'].count()
W=matches_won_SH['id'].count()
X=matches_won_RPS['id'].count()
Y=matches_won_RR['id'].count()

O


# In[ ]:


n_bins = 11
ind = np.arange(n_bins)
width = 0.50

plt.figure(figsize=(10,10))

matches_played=[A,B,C,D,E,F,G,H,I,J,K]
matches_won=[O,P,Q,R,S,T,U,V,W,X,Y]

p1 = plt.bar(ind, matches_played, width, color='LightSkyBlue')
p2 = plt.bar(ind, matches_won, width, color='Lime')

plt.ylabel('Number of Matches')
plt.xlabel('Teams')
plt.title('Overall performance of the team')
plt.xticks(ind + width/2., ('KKR', 'MI', 'KXP', 'CSK', 'DC', 'DD', 'RCB', 'KT', 'SH', 'RPS', 'RR'))
plt.yticks(np.arange(0, 200, 5))
plt.legend((p1[0], p2[0]), ('matches_played', 'matches_won'))


# In[ ]:


KKR_toss_won=matches[matches['toss_winner']=='Kolkata Knight Riders'].id.count()
MI_toss_won=matches[matches['toss_winner']=='Mumbai Indians'].id.count()
KXP_toss_won=matches[matches['toss_winner']=='Kings XI Punjab'].id.count()
CSK_toss_won=matches[matches['toss_winner']=='Chennai Super Kings'].id.count()
DC_toss_won=matches[matches['toss_winner']=='Deccan Chargers'].id.count()
DD_toss_won=matches[matches['toss_winner']=='Delhi Daredevils'].id.count()
RCB_toss_won=matches[matches['toss_winner']=='Royal Challengers Bangalore'].id.count()
KT_toss_won=matches[matches['toss_winner']=='Kochi Tuskers Kerala'].id.count()
SH_toss_won=matches[matches['toss_winner']=='Sunrisers Hyderabad'].id.count()
RSP_toss_won=matches[matches['toss_winner']=='Rising Pune Supergiants'].id.count()
RR_toss_won=matches[matches['toss_winner']=='Rajasthan Royals'].id.count()


KKR_match_won=matches[(matches['toss_winner']=='Kolkata Knight Riders') & (matches['winner']=='Kolkata Knight Riders')].id.count()
MI_match_won=matches[(matches['toss_winner']=='Mumbai Indians') & (matches['winner']=='Mumbai Indians')].id.count()
KXP_match_won=matches[(matches['toss_winner']=='Kings XI Punjab') & (matches['winner']=='Kings XI Punjab')].id.count()
CSK_match_won=matches[(matches['toss_winner']=='Chennai Super Kings') & (matches['winner']=='Chennai Super Kings')].id.count()
DC_match_won=matches[(matches['toss_winner']=='Deccan Chargers') & (matches['winner']=='Deccan Chargers')].id.count()
DD_match_won=matches[(matches['toss_winner']=='Delhi Daredevils') & (matches['winner']=='Delhi Daredevils')].id.count()
RCB_match_won=matches[(matches['toss_winner']=='Royal Challengers Bangalore') & (matches['winner']=='Royal Challengers Bangalore')].id.count()
KT_match_won=matches[(matches['toss_winner']=='Kochi Tuskers Kerala') & (matches['winner']=='Kochi Tuskers Kerala')].id.count()
SH_match_won=matches[(matches['toss_winner']=='Sunrisers Hyderabad') & (matches['winner']=='Sunrisers Hyderabad')].id.count()
RSP_match_won=matches[(matches['toss_winner']=='Rising Pune Supergiants') & (matches['winner']=='Rising Pune Supergiants')].id.count()
RR_match_won=matches[(matches['toss_winner']=='Rajasthan Royals') & (matches['winner']=='Rajasthan Royals')].id.count()


# In[ ]:


n_bins = 11
ind = np.arange(n_bins)
width = 0.50

plt.figure(figsize=(10,10))

toss_won=[KKR_toss_won,MI_toss_won,KXP_toss_won,CSK_toss_won,DC_toss_won,DD_toss_won,RCB_toss_won,KT_toss_won,SH_toss_won,RSP_toss_won,RR_toss_won]
match_won=[KKR_match_won,MI_match_won,KXP_match_won,CSK_match_won,DC_match_won,DD_match_won,RCB_match_won,KT_match_won,SH_match_won,RSP_match_won,RR_match_won]

p1 = plt.bar(ind, toss_won, width, color='firebrick')
p2 = plt.bar(ind, match_won, width, color='aqua')

plt.ylabel('Toss')
plt.xlabel('Teams')
plt.title('Toss factor')
plt.xticks(ind + width/2., ('KKR', 'MI', 'KXP', 'CSK', 'DC', 'DD', 'RCB', 'KT', 'SH', 'RSP', 'RR'))
plt.yticks(np.arange(0, 100, 5))
plt.legend((p1[0], p2[0]), ('toss_won', 'match_won'))


# In[ ]:


plt.figure(figsize=(8,10))
maximum_runs = deliveries.groupby(['batsman'])['batsman_runs'].sum()
maximum_runs
maximum_runs.sort_values(ascending = False,inplace='True')
maximum_runs[:10].plot(x= 'bowler', y = 'runs', kind = 'bar', colormap = 'Pastel2')
plt.xlabel('Batsmen')
plt.ylabel('Most Runs in IPL')

