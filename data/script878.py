
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# # Scripty Medals
# 
# This notebook continues the analysis done in "[Scripty McScriptface the Lazy Kaggler](https://www.kaggle.com/dvasyukova/d/kaggle/meta-kaggle/scripty-mcscriptface-the-lazy-kaggler)".
# 
# The purpose is to calculate how many competitions medals can be earned by using public script submissions. 
# 
# Criteria for medals: [Kaggle Progression System](https://www.kaggle.com/progression).
# 
# ## Load data

# In[ ]:


# Competitions - use only those that award points
competitions = (pd.read_csv('../input/Competitions.csv')
                .rename(columns={'Id':'CompetitionId'}))
competitions = competitions[(competitions.UserRankMultiplier > 0)]
# Scriptprojects to link scripts to competitions
scriptprojects = (pd.read_csv('../input/ScriptProjects.csv')
                    .rename(columns={'Id':'ScriptProjectId'}))
# Evaluation algorithms
evaluationalgorithms = (pd.read_csv('../input/EvaluationAlgorithms.csv')
                          .rename(columns={'Id':'EvaluationAlgorithmId'}))
competitions = (competitions.merge(scriptprojects[['ScriptProjectId','CompetitionId']],
                                   on='CompetitionId',how='left')
                            .merge(evaluationalgorithms[['IsMax','EvaluationAlgorithmId']],
                                   on='EvaluationAlgorithmId',how='left')
                            .dropna(subset = ['ScriptProjectId'])
                            .set_index('CompetitionId'))
competitions['ScriptProjectId'] = competitions['ScriptProjectId'].astype(int)
# Fill missing values for two competitions
competitions.loc[4488,'IsMax'] = True # Flavours of physics
competitions.loc[4704,'IsMax'] = False # Santa's Stolen Sleigh
# Teams
teams = (pd.read_csv('../input/Teams.csv')
         .rename(columns={'Id':'TeamId'}))
teams = teams[teams.CompetitionId.isin(competitions.index)]
teams['Score'] = teams.Score.astype(float)
# Submissions
submissions = pd.read_csv('../input/Submissions.csv')
submissions = submissions.dropna(subset=['Id','DateSubmitted','PublicScore'])
submissions.DateSubmitted = pd.to_datetime(submissions.DateSubmitted)
submissions = submissions[(submissions.TeamId.isin(teams.TeamId))
                         &(submissions.IsAfterDeadline==False)
                         &(~(submissions.PublicScore.isnull()))]
submissions = submissions.merge(teams[['TeamId','CompetitionId']],
                                how='left',on='TeamId')
submissions = submissions.merge(competitions[['IsMax']],
                                how='left',left_on='CompetitionId', right_index=True)


# Find ranks sufficient for medals.

# In[ ]:


competitions['Nteams'] = teams.groupby('CompetitionId').size()
#competitions[['Bronze','Silver','Gold']] = np.zeros((competitions.shape[0],3),dtype=int)
t = competitions.Nteams
competitions.loc[t <= 99, 'Bronze'] = np.floor(0.4*t)
competitions.loc[t <= 99, 'Silver'] = np.floor(0.2*t)
competitions.loc[t <= 99, 'Gold'] = np.floor(0.1*t)

competitions.loc[(100<=t)&(t<=249),'Bronze'] = np.floor(0.4*t)
competitions.loc[(100<=t)&(t<=249),'Silver'] = np.floor(0.2*t)
competitions.loc[(100<=t)&(t<=249), 'Gold'] = 10

competitions.loc[(250<=t)&(t<=999),'Bronze'] = 100
competitions.loc[(250<=t)&(t<=999),'Silver'] = 50
competitions.loc[(250<=t)&(t<=999), 'Gold'] = 10 + np.floor(0.002*t)

competitions.loc[t >= 1000, 'Bronze'] = np.floor(0.1*t)
competitions.loc[t >= 1000, 'Silver'] = np.floor(0.05*t)
competitions.loc[t >= 1000, 'Gold'] = 10 + np.floor(0.002*t)

#competitions[['Nteams','Gold','Silver','Bronze','Title']]


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
submissions['PublicScore'] = submissions['PublicScore'].astype(float)
submissions['PrivateScore'] = submissions['PrivateScore'].astype(float)
scored = submissions.groupby('TeamId',sort=False).apply(isscored)
scored.index = scored.index.droplevel()
submissions['IsScored'] = scored


# ## Submit two best public scripts

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
# Fix Liberty Mutual result
scriptycomps.loc[4471,'Rank'] = 150


# In[ ]:


scriptycomps = scriptycomps.sort_values(by='Nteams')
fig, ax = plt.subplots(figsize=(10,8))
h = np.arange(len(scriptycomps))
ax.barh(h, scriptycomps.Nteams,color='white')
ax.barh(h, scriptycomps.Bronze,color='#F0BA7C')
ax.barh(h, scriptycomps.Silver,color='#E9E9E9')
ax.barh(h, scriptycomps.Gold,color='#FFD448')
ax.set_yticks(h+0.4)
ax.set_yticklabels(scriptycomps.Title.values);
ax.set_ylabel('');
ax.scatter(scriptycomps.Rank,h + 0.4,s=40,c='b',zorder=100,marker='d',alpha=0.6)
ax.set_xlim(0,1000)
ax.legend(['Scripty\'s rank','None',
           'Bronze','Silver','Gold'],loc=4,fontsize='large');
ax.set_title('Medals from submitting the best public script');
ax.set_xlabel('Rank')
ax.set_ylim(0,h.max()+1);


# ## Submit most popular script versions

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


# In[ ]:


#scriptycomps = scriptycomps.sort_values(by='Nteams')
fig, ax = plt.subplots(figsize=(10,8))
h = np.arange(len(scriptycomps))
ax.barh(h, scriptycomps.Nteams,color='white')
ax.barh(h, scriptycomps.Bronze,color='#F0BA7C')
ax.barh(h, scriptycomps.Silver,color='#E9E9E9')
ax.barh(h, scriptycomps.Gold,color='#FFD448')
ax.set_yticks(h+0.4)
ax.set_yticklabels(scriptycomps.Title.values);
ax.set_ylabel('');
ax.scatter(scriptycomps.Rank,h + 0.4,s=40,c='b',zorder=100,marker='d',alpha=0.6)
ax.set_xlim(0,1000)
ax.legend(['Scripty\'s rank','None',
           'Bronze','Silver','Gold'],loc=4,fontsize='large');
ax.set_title('Medals from submitting most popular script versions');
ax.set_xlabel('Rank')
ax.set_ylim(0,h.max()+1);


# It looks like submitting most popular script versions gets you no medals at all, while chasing the best scoring scripts on public LB could bring some bronze medals. 
