
# coding: utf-8

# ## Introduction ##
# I previously wrote a "naive betting analysis" kernel (see [here][1]), wherein I explored if we'd make a profit by just betting on the favourite or the longshot outcome in each match. Turns out those strategies would only yield losses in the long run, so other smarter strategies were required to beat the bookies. In this notebook I'm exploring accumulator bets.
# 
# An accumulator is when you place a single bet on the outcome of multiple matches together. Say there are 4 matches M1, M2, M3 & M4, with odds (for the favourite in each of these matches) being O1, O2, O3 & O4. Say I feel confident that the favourites are going to win these 4 matches. So instead of betting 1 dollar separately on each match, I'll place a single bet of 1 dollar on all 4 of these teams to win. (referred to as a 4-fold accumulator). The new odds O_accum=O1 X O2 X O3 X O4, & I'll receive a payout *only if* all 4 of my teams win. There is a risk involved as it takes only 1 wrong outcome to screw up your bet, but it can be used to substantially increase your payouts when you're fairly confident.
# 
# Whats more, Bet365 also offers a bonus when you win your accumulator bet, based on the number of folds involved (check details [here][2]). So in my example above, if I bet 1 dollar and my 4-fold accumulator bet came true, I'd receive an additional 10% bonus, so my payout would be:
# 
# P=O1 X O2 X O3 X O4 X 1.10
# 
# Now that we understand how the accumulator bet works, we can get started. I'm using data only of the past 5 seasons for my analysis (i.e from 2011/12 season onwards). Based on my own knowledge of the football leagues, I picked 12 teams that were dominant over that period. I'll analyse the payouts I'd receive if I placed accumulator bets on matches of these teams. The teams I've chosen are:
# 
#  - Barcelona (3X winner, 2X runner-up in Spanish league)
#  - Real Madrid (1X winner, 3X runner-up in Spanish league)
#  - Atletico Madrid (1X winner in Spanish league)
#  - Paris Saint Germain (5X winner in French league)
#  - Juventus (5X winner in Italian league)
#  - Bayern Munich (4X winner, 1X runner-up in German league)
#  - Borussia Dortmund (1X winner, 3X runner-up in German league)
#  - Celtic (5X winner in Scottish league)
#  - Benfica (3X winner, 2X runner-up in Portuguese league)
#  - Porto (2X winner, 1X runner-up in Portuguese league)
#  - Manchester City (2X winner, 2X runner-up in EPL)
#  - Arsenal (consistently finish in top 4 of the EPL every year)
# 
# I'll first compute the net-payout for each team individually over the 5 seasons. I'll sort them in descending order of payout, & build a 2-fold accumulator of the best 2 teams. I'll progressively keep increasing the folds & compute the net-payout, to find the "optimum accumulator" that maximizes payout.
# 
# 
#   [1]: https://www.kaggle.com/sadz2201/d/hugomathien/soccer/naive-betting-analysis/notebook
#   [2]: http://extra.bet365.com/promotions/soccer/soccer-accumulator-bonus

# **Import Libraries, load the data**

# In[ ]:


import pandas as pd
import sqlite3
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

#load data (make sure you have downloaded database.sqlite)
with sqlite3.connect('../input/database.sqlite') as con:
    countries = pd.read_sql_query("SELECT * from Country", con)
    matches = pd.read_sql_query("SELECT * from Match", con)
    leagues = pd.read_sql_query("SELECT * from League", con)
    teams = pd.read_sql_query("SELECT * from Team", con)


# Merge data, select most recent 5 seasons, retain only relevant columns

# In[ ]:


selected_countries = ['Scotland','France','Germany','Italy','Spain','Portugal','England']
countries = countries[countries.name.isin(selected_countries)]
leagues = countries.merge(leagues,on='id',suffixes=('', '_y'))

#There's a special character in the long name "Atlético Madrid".
#This can be a pain in the ass, so I'm gonna change it for simplicity.
teams.loc[teams.team_api_id==9906,"team_long_name"] = "Atletico Madrid"

#retain only data from 2011-12 season
matches=matches[matches.date>='2011-08-01']
matches = matches[matches.league_id.isin(leagues.id)]
matches = matches[['id', 'country_id' ,'league_id', 'season', 'stage', 'date','match_api_id', 'home_team_api_id', 'away_team_api_id','home_team_goal','away_team_goal','B365H', 'B365D' ,'B365A']]
matches.dropna(inplace=True)
matches.head()


# Team API id's don't tell us anything. Lets merge team names

# In[ ]:


matches=matches.merge(teams,left_on='home_team_api_id',right_on='team_api_id',suffixes=('','_h'))
matches=matches.merge(teams,left_on='away_team_api_id',right_on='team_api_id',suffixes=('','_a'))
matches=matches[['id', 'season', 'date','home_team_goal','away_team_goal','B365H', 'B365D' ,'B365A',
                 'team_long_name','team_long_name_a']]
matches.head()


# Retain data of only our chosen 12 teams for further analysis. Also since we have multiple teams from the same league (Barca, Real & Atletico from Spain; Bayern & Dortmund from Germany; Benfica & Porto from Portugal; Mancity & Arsenal from EPL), I'll remove the matches where they face off against each other. It's hard to pick a winner in these, so it's best to leave them out of our accumulators

# In[ ]:


accumulator_teams=['FC Barcelona','Real Madrid CF','Celtic','FC Porto','SL Benfica','Juventus','FC Bayern Munich','Paris Saint-Germain','Manchester City','Atletico Madrid','Borussia Dortmund','Arsenal']

#matches where any of our 12 teams is playing at home
matches_h=matches[matches.team_long_name.isin(accumulator_teams)]
#matches where any of our 12 teams is playing away
matches_a=matches[matches.team_long_name_a.isin(accumulator_teams)]
#concat & drop duplicates
matches=pd.concat([matches_h,matches_a],axis=0)
matches.drop_duplicates(inplace=True)

matches=matches.sort_values(by='date')
#remove matches where our teams are facing off against each other
matches=matches[~((matches.team_long_name.isin(accumulator_teams)) & (matches.team_long_name_a.isin(accumulator_teams)))]

matches.head()


# For our accumulators, we'll have to group the matches based on date. It is highly unlikely that all our teams play a match on the same date, but it is likely that they play within the same gameweek. Thus we need to do some post-processing on dates.
# 
# There can be weekend matches (Sat/Sun), or midweek matches (Tue/Wed). In extreme cases, a weekend match can be preponed to Friday or postponed to Monday, or a midweek match postponed to Thursday. 
# I'll change the date of all weekend matches to the corresponding Saturday, & all midweek matches to the corresponding Tuesday.

# In[ ]:


matches.date=pd.to_datetime(matches.date)
#monday matches. subtract 2 to make it saturday
m0=matches[matches.date.dt.weekday==0]
m0.date=m0.date-timedelta(days=2)

#tuesday matches
m1=matches[matches.date.dt.weekday==1]
#wednesday matches. subtract 1 to make it tuesday
m2=matches[matches.date.dt.weekday==2]
m2.date=m2.date-timedelta(days=1)
#thursday matches. subtract 2 to make it tuesday
m3=matches[matches.date.dt.weekday==3]
m3.date=m3.date-timedelta(days=2)

#friday matches. add 1 to make it saturday
m4=matches[matches.date.dt.weekday==4]
m4.date=m4.date+timedelta(days=1)
#saturday matches
m5=matches[matches.date.dt.weekday==5]
#sunday matches. subtract 1 to make it saturday
m6=matches[matches.date.dt.weekday==6]
m6.date=m6.date-timedelta(days=1)

#merge all, sort by date
matches=pd.concat([m0,m1,m2,m3,m4,m5,m6],axis=0)
matches=matches.sort_values(by='date')
del m0,m1,m2,m3,m4,m5,m6

#checking if we have only saturday & tuesday now
matches.date.dt.weekday.value_counts()


# Find out which of our 12 chosen teams is playing in each match. Also find the venue & odds for the team. 

# In[ ]:


matches['our_team']='abc'
matches['our_venue']='H'
matches['our_odds']=matches.B365H

is_home=matches.team_long_name.isin(accumulator_teams)
#our team is playing at home
matches.our_team[is_home==True]=matches.team_long_name[is_home==True]

#our team is playing away.
matches.our_team[is_home==False]=matches.team_long_name_a[is_home==False]
matches.our_venue[is_home==False]='A'
matches.our_odds[is_home==False]=matches.B365A[is_home==False]


# Compute the result of each match based on the goals. Also compute our payout for each match assuming we bet 1$ on our chosen team. 

# In[ ]:


matches['result']='H'
matches.loc[matches.home_team_goal==matches.away_team_goal,"result"]='D'
matches.loc[matches.home_team_goal<matches.away_team_goal,"result"]='A'

matches['payout']=matches.our_odds
#our team either lost or drew. reset payout to 0
matches.loc[~(matches.result==matches.our_venue),"payout"]=0
matches.head()


# Sanity check: Lets see if we're profitable so far. (Note, this is just individual bets, we haven't gone into accumulators yet).  

# In[ ]:


print(sum(matches.payout)/matches.shape[0])


# Turns out we're just about even. Let's analyze the matches won & net payout by each team.

# In[ ]:


team_n=matches.our_team.value_counts()
print ("win percentage by team:")
print(matches[matches.payout!=0].our_team.value_counts()/team_n)
print("_"*50)
print ("net payout by team:")
indiv_payout=matches.groupby('our_team')['payout'].sum()
indiv_payout=indiv_payout/team_n
print(indiv_payout)


# Juventus & Benfica seem to be our star performers, with 14 & 10.68% profit themselves. I'm quite surprised that Atletico, PSG & Manchester City are profitable or even despite a low wins percentage, while Barcelona have losses. Also, the net payout for Real Madrid & Arsenal is nearly identical, despite Real winning 79.4% of their matches and Arsenal only winning 58.33%. Although a few teams have losses, I do believe they can help improve the overall profits of our accumulators because of their good wins percentage. 
# 
# **Accumulator**

# In[ ]:


#our teams list in sorted order of individual profits
accumulator_teams=['Juventus','SL Benfica','Atletico Madrid','FC Bayern Munich','Paris Saint-Germain','Manchester City','Real Madrid CF','Arsenal','Celtic','FC Porto','FC Barcelona','Borussia Dortmund']
#list of bet365 bonus payouts
#bonus[k]= bet365 bonus for k-fold accumulator
bonus=[1,1,1,1.05,1.1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]

#blank dict
accum_payouts={}
for k in range(2,len(accumulator_teams)+1):
    #choose first k teams from the team list
    accum_subset=accumulator_teams[:k]
    
    #choose only matches involving these teams
    matches_kfold=matches[matches.our_team.isin(accum_subset)]
    #count of matches per date.
    date_counts=matches_kfold.date.value_counts().reset_index()
    date_counts.columns=['date','counts']
    
    #select only the dates where all k teams are in action
    dates_kfold=date_counts[date_counts.counts==k].date
    #retain only the matches happening on these dates
    matches_kfold=matches_kfold[matches_kfold.date.isin(dates_kfold)]
    #k-fold accumulator payout (product of payouts of all k teams on that date)
    payout_kfold=matches_kfold.groupby('date')['payout'].prod()
    #multiply bonus
    bonus_payout_kfold=payout_kfold* bonus[k]
    print(str(k) + " fold:")
    print(accum_subset)
    print("#bets: " + str(len(payout_kfold)))
    print("#correct predictions: " + str(len(payout_kfold[payout_kfold!=0])))
    print("Net outcome (without bonus): " + str(sum(payout_kfold)/len(payout_kfold)))
    print("Net outcome (after bonus): " + str(sum(bonus_payout_kfold)/len(payout_kfold)))
    print("_"*50)
    accum_payouts[k]=sum(bonus_payout_kfold)/len(payout_kfold)
    
#print the best choice of k, the corresponding teams & net payout.
best_k=max(accum_payouts,key=accum_payouts.get)
print("best k= " +str(best_k) )
print(accumulator_teams[:best_k])
print("best payout= " +str(accum_payouts[best_k]))


# **Success!**
# 
# We've just found that accumulators are a winning strategy. The best results are obtained by placing an 8-fold accumulator in every week when Juventus, Benfica, Atletico, PSG, Bayern, ManCity, Real Madrid & Arsenal are all in action. A 7-fold accumulator ignoring Arsenal from the list is not too far off either. This strategy would've given us a net profit of 219.6% (i.e more than tripling our bank balance) over 5 years, which I think is a pretty good return on investment. Note that the success rate of our bets is quite low (20.3% for 7-fold, 11.3% for 8-fold). Even so, our winnings due to the high odds & bonuses far outweigh our losses.
# 
# I'm fairly convinced, & am going to try out this strategy at bet365.com for the remainder of the 2016-17 season. If you are heading over to create your account & start betting too, please upvote this notebook before you do so. :-)
# [Also note - the 8 fold accumulator has already been successful 3 times so far this season. Twice for league fixtures, & once when I included cup fixtures of some of these teams. Also, Celtic have won 23 of their 24 league games so far this season, so it might make sense to make it into 9 fold accumulator to milk a little more profits :-) ]
# 
# There's still quite some room for experimentation. We could add in other teams to the mix. We can also try out better optimization & search strategies. (I've just restricted myself to greedy optimization by adding in the next-best team in each iteration)
# 
# **Ending notes**
# 
#  - 219% profit might be an optimistic estimate, because I've used hindsight knowledge of teams over the past 5 seasons, & I'm assuming that they'll continue to be dominant in the near future as well. Also as teams continue to dominate, the bookmakers will realize this & progressively keep decreasing the odds on their matches.
#  - Despite it's success in the past, there is no guarantee it'll work in the future.
#  - This a proposed strategy for success over time, & not to become a millionaire overnight.
#  - Please choose your betting stakes wisely, & bear in mind that I take no responsibility for any losses incurred.
# 
#  
