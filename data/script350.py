
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# # Scripty McScriptface the Lazy Kaggler
# 
# Kaggle Scripts were launched in April 2015 and have received both positive and negative feedback.
# 
# One of the common critiques of the Scripts is that they allow people to achieve high leaderboard positions with very little effort - literally by clicking a button.
# 
# I want to investigate what kind of results could be achieved by using only the public scripts. Enter Scripty McScriptface, a Kaggler who does nothing but submit public scripts for every competition where they are enabled.
# 
# Where would this person be today in the Kaggle global ranking?
# 
# ## Load the data
# 
# Read in competitions data, use only competitions that award points.

# In[ ]:


competitions = (pd.read_csv('../input/Competitions.csv')
                .rename(columns={'Id':'CompetitionId'}))
competitions = competitions[(competitions.UserRankMultiplier > 0)]


# Figure out if competition's evaluation metric should be maximized or minimized:

# In[ ]:


evals = (pd.read_csv('../input/EvaluationAlgorithms.csv')
           .rename(columns={'Id':'EvaluationAlgorithmId'}))
competitions = competitions.merge(evals[['EvaluationAlgorithmId','IsMax']], 
                                  how='left',on='EvaluationAlgorithmId')
# Fill missing values for two competitions
competitions.loc[competitions.CompetitionId==4488,'IsMax'] = True # Flavours of physics
competitions.loc[competitions.CompetitionId==4704,'IsMax'] = False # Santa's Stolen Sleigh


# Which competitions have scripts?

# In[ ]:


scriptprojects = pd.read_csv('../input/ScriptProjects.csv')
competitions = competitions[competitions.CompetitionId.isin(scriptprojects.CompetitionId)]
print("Found {} competitions with scripts enabled.".format(competitions.shape[0]))
if competitions.IsMax.isnull().any():
    # in case this is rerun after more competitions are added
    print("Please fill IsMax value for:")
    print(competitions.loc[competitions.IsMax.isnull(),['CompetitionId','Title']])


# Read in teams data and filter by competition.

# In[ ]:


teams = (pd.read_csv('../input/Teams.csv')
         .rename(columns={'Id':'TeamId'}))
teams = teams[teams.CompetitionId.isin(competitions.CompetitionId)]
teams['Score'] = teams.Score.astype(float)


# Read in submissions data and filter by competition (through teams selected previously). Filter out submissions with errors and submissions made after the competition deadline.

# In[ ]:


submissions = pd.read_csv('../input/Submissions.csv')
submissions = submissions[(submissions.TeamId.isin(teams.TeamId))
                         &(submissions.IsAfterDeadline==False)
                         &(~(submissions.PublicScore.isnull()))]
submissions = submissions.merge(teams[['TeamId','CompetitionId']],
                                how='left',on='TeamId')
submissions = submissions.merge(competitions[['CompetitionId','IsMax']],
                                how='left',on='CompetitionId')


# ## How many teams use script submissions?
# Calculate how many teams participated in each competition and how many submitted at least one script result. Submissions have a `SourceScriptVersionId` field that lets us determine if a submission was from a script. We don't catch submissions that come from downloading the script's results and submitting a .csv file. 

# In[ ]:


competitions.set_index("CompetitionId",inplace=True)
# How many teams participated in a competition?
competitions['Nteams'] = (submissions.groupby('CompetitionId')
                          ['TeamId'].nunique())
# How many teams used at least one script submission?
competitions['TeamsSubmittedScripts'] = (submissions
                                         [~(submissions.SourceScriptVersionId.isnull())]
                                         .groupby('CompetitionId')['TeamId'].nunique())


# Another interesting quantity is how many teams actually selected their script submission for scoring. We have an IsSelected field in the Submissions table, so we can use that. Usually any team may select up to two submissions for scoring. If a team selects less than two then their best public submissions are used instead.
# 
# How many submissions do teams usually select?

# In[ ]:


submissions.groupby('TeamId')['IsSelected'].sum().value_counts()


# So a lot of teams did not select any submissions at all. We need to figure out which two submissions were used for scoring those teams.

# In[ ]:


def isscored(group):
    # if two or less submissions select all
    if group.shape[0] <= 2:
        pd.Series(np.ones(group.shape[0],dtype=np.bool),index=group.index)
    nsel = group.IsSelected.sum()
    # if two selected return them
    if nsel == 2:
        return group.IsSelected
    # if need to select more - choose by highest public score
    toselect = list(group.IsSelected.values.nonzero()[0])
    ismax = group['IsMax'].iloc[0]
    ind = np.argsort(group['PublicScore'].values)
    scored = group.IsSelected.copy()
    if ismax:
        ind = ind[::-1]
    for i in ind:
        if i not in toselect:
            toselect.append(i)
        if len(toselect)==2:
            break
    scored.iloc[toselect] = True
    return scored


# In[ ]:


submissions['PublicScore'] = submissions['PublicScore'].astype(float)
submissions['PrivateScore'] = submissions['PrivateScore'].astype(float)
scored = submissions.groupby('TeamId',sort=False).apply(isscored)
scored.index = scored.index.droplevel()
submissions['IsScored'] = scored


# In[ ]:


# How many teams selected a script submission for private LB scoring?
competitions['TeamsSelectedScripts'] = (submissions
                                        [~(submissions.SourceScriptVersionId.isnull())&
                                          (submissions.IsScored)]
                                        .groupby('CompetitionId')['TeamId'].nunique())


# Finally, let's look at the plot of our results.

# In[ ]:


competitions.sort_values(by='Nteams',inplace=True)
fig, ax = plt.subplots(figsize=(10,8))
h = np.arange(len(competitions))
colors = cm.Blues(np.linspace(0.5, 1, 3))
ax.barh(h, competitions.Nteams,color=colors[0])
ax.barh(h, competitions.TeamsSubmittedScripts,color=colors[1])
ax.barh(h, competitions.TeamsSelectedScripts,color=colors[2])
ax.set_yticks(h+0.4)
ax.set_yticklabels(competitions.Title.values);
ax.set_ylabel('');
ax.legend(['Total teams',
           'Submitted from a script',
           'Selected a script submission'],loc=4,fontsize='large');
ax.set_title('Usage of script submissions by teams');
ax.set_ylim(0,h.max()+1);


# Things to note here:
# 
# * Some competitions like Otto Group and Restaurant Revenue Prediction have scripts enabled but don't have any script submissions. This is because "submit to competition" button was not yet available at the time.
# * Around 25% of teams use script submissions.
# * More than half of the teams that submit script results actually select these submissions for scoring.

# ## Scripty's results
# 
# ### Submit two best scripts from the public LB
# For estimating what kind of results we might expect from using public scripts I'll assume that Scripty looks for best performing scripts on the public LB and selects the best two for scoring on the private LB.

# In[ ]:


competitions["NScriptSubs"] = (submissions
                               [~(submissions.SourceScriptVersionId.isnull())]
                               .groupby('CompetitionId')['Id'].count())
scriptycomps = competitions[competitions.NScriptSubs > 0].copy()
scriptycomps.shape


# In[ ]:


def find_private_score(df):
    if df.SourceScriptVersionId.isnull().all():
        # no scripts
        return
    ismax = df.IsMax.iloc[0]
    submit = (df.loc[~(df.SourceScriptVersionId.isnull())]
                .groupby('SourceScriptVersionId')
                [['PublicScore','PrivateScore']]
                .agg('first')
                .sort_values(by='PublicScore',ascending = not ismax)
                .iloc[:2])
    score = submit.PrivateScore.max() if ismax else submit.PrivateScore.min()
    # Find scores from all teams
    results = (df.loc[df.IsScored]
                 .groupby('TeamId')
                 ['PrivateScore']
                 .agg('max' if ismax else 'min')
                 .sort_values(ascending = not ismax)
                 .values)
    if ismax:
        ranktail = (results <  score).nonzero()[0][0] + 1
        rankhead = (results <= score).nonzero()[0][0] + 1
    else:
        ranktail = (results >  score).nonzero()[0][0] + 1
        rankhead = (results >= score).nonzero()[0][0] + 1
    rank = int(0.5*(ranktail+rankhead))
    return pd.Series({'Rank':rank,'Score':score})

scriptycomps[['Rank','Score']] = (submissions.groupby('CompetitionId')
                                             .apply(find_private_score))
scriptycomps['TopPerc'] = np.ceil(100*scriptycomps['Rank']
                                  /scriptycomps['Nteams'])
scriptycomps['Points'] = (1.0e5*((scriptycomps.Rank)**(-0.75))
                          *np.log10(1+np.log10(scriptycomps.Nteams))
                          *scriptycomps.UserRankMultiplier)
scriptycomps[['Title','Score','Nteams',
              'Rank','TopPerc','Points']].sort_values(by='Rank')


# The result in Liberty Mutual Group competition seems unbelievably good. [Here](https://www.kaggle.com/chriscc/liberty-mutual-group-property-inspection-prediction/blah-xgb/run/45838) is the responsible script. Apparently even the script's author did not guess how well it was likely to perform on the private Leaderboard as it was not chosen. The best actual result that comes from a script submission has rank 150 in this competition.
# 
# How many badges did Scripty earn?

# In[ ]:


top10p = (scriptycomps.TopPerc <= 10).sum()
top25p = ((scriptycomps.TopPerc > 10)&(scriptycomps.TopPerc <= 25)).sum()
print("{} Top10% badges and {} Top25% badges".format(top10p, top25p))


# Not bad... Well, at least he didn't earn a master status this way.
# 
# How many points does he have now?

# In[ ]:


lastdeadline = pd.to_datetime(competitions.Deadline.max())
decay = np.exp((pd.to_datetime(scriptycomps.Deadline) - lastdeadline).dt.days/500)
totalpoints = (decay*scriptycomps.Points).sum()
totalpoints


# Well, shame... it's more than I have. What's his global ranking now?

# In[ ]:


users = pd.read_csv('../input/Users.csv').sort_values(by='Points',ascending=False)
rank = (users.Points < totalpoints).nonzero()[0][0] + 1
print("Number {} in the global ranking".format(rank))


# We could argue that the single excellent result in Liberty Mutual competition affects our calculations too strongly. If we change the 15th place to 150th (best script submission result on private LB) then the results become:

# In[ ]:


scriptycomps.loc[4471,'Rank'] = 150
scriptycomps['TopPerc'] = np.ceil(100*scriptycomps['Rank']/scriptycomps['Nteams'])
scriptycomps['Points'] = 1.0e5*((scriptycomps.Rank)**(-0.75))*np.log10(1+np.log10(scriptycomps.Nteams))*scriptycomps.UserRankMultiplier
totalpoints1 = (decay*scriptycomps.Points).sum()
totalpoints1


# In[ ]:


rank1 = (users.Points < totalpoints1).nonzero()[0][0] + 1
rank1


# The rank is predictably worse but still in the top 500.
# 
# ### Submit what everyone else submits
# 
# A valid point was raised in the comments that it's not always possible to find the best scoring script (it may be hidden), and also it's not possible for Scripty to be the first to submit anything. So I came up with another way of calculating Scripty's rank.
# 
# I look at script submissions that were actually used for scoring the participants. 
# Among these I find two most popular script versions, and that's what Scripty selects. 
# Then his score is the best private score of these two submissions. 
# Since there's a lot of people with the same score I take the rank of the middle of this group.
# I think this deals with both stated problems.

# In[ ]:


def find_private_score(df):
    if df.SourceScriptVersionId.isnull().all():
        # no scripts
        return
    ismax = df.IsMax.iloc[0]
    competition = df.name
    submit = (df.loc[~(df.SourceScriptVersionId.isnull())
                     &(df.IsScored)]
                .groupby('SourceScriptVersionId')
                .agg({'PublicScore':'first','PrivateScore':'first','Id':'size'})
                .rename(columns={'Id':'Nteams'})
                .sort_values(by='Nteams',ascending = False)
                .iloc[:2])
    score = submit.PrivateScore.max() if ismax else submit.PrivateScore.min()
    # Find scores from all teams
    results = (df.loc[df.IsScored]
                 .groupby('TeamId')
                 ['PrivateScore']
                 .agg('max' if ismax else 'min')
                 .sort_values(ascending = not ismax)
                 .values)
    rank = int(np.median((results==score).nonzero()[0])) + 1
    return pd.Series({'Rank':rank,'Score':score})

scriptycomps[['Rank','Score']] = (submissions.groupby('CompetitionId')
                                             .apply(find_private_score))
scriptycomps['TopPerc'] = np.ceil(100*scriptycomps['Rank']
                                  /scriptycomps['Nteams'])
scriptycomps['Points'] = (1.0e5*((scriptycomps.Rank)**(-0.75))
                          *np.log10(1+np.log10(scriptycomps.Nteams))
                          *scriptycomps.UserRankMultiplier)
scriptycomps[['Title','Score','Nteams',
              'Rank','TopPerc','Points']].sort_values(by='Rank')


# In[ ]:


top10p = (scriptycomps.TopPerc <= 10).sum()
top25p = ((scriptycomps.TopPerc > 10)&(scriptycomps.TopPerc <= 25)).sum()
print("{} Top10% badges and {} Top25% badges".format(top10p, top25p))
totalpoints2 = (decay*scriptycomps.Points).sum()
rank2 = (users.Points < totalpoints2).nonzero()[0][0] + 1
print("Ranked {} with {:.1f} points.".format(rank2,totalpoints2))


# Scripty's performance became less stellar with this approach, but he's still in 
# the global top 1000.
# 
# 
# ===
# 
# I guess this does prove the point that prowling the leaderboards and looking for 
# best performing scripts could bring you well into the top 1000 Kagglers. 
# Even into the top 500 with some luck. 
# 
# I can see why people who don't have time for participating in every competition would be frustrated about being pushed down in the ratings by the script chasers.
# 
# This could potentially devalue Kaggle global rankings. Still for me the main benefit of every competition was the things I learned along the way and not the final placement. I value being able to discuss the competition's dataset properties and quirks, which kinds of models worked and why. And this is something a script chaser would lack. Also "beating that damn script" can be a nice motivator.
# 
# 
# I think the scripts do provide great opportunities for code sharing and learning. I wish there was some parallel ranking system that would reward posting useful and interesting scripts. Then Kaggle could be a step closer to the Home for Data Science =).
