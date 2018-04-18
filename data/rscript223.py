import pandas as pd
import numpy as np
from sklearn import *
import glob

datafiles = sorted(glob.glob('../input/**.csv'))
datafiles = {file.split('/')[-1].split('.')[0]: pd.read_csv(file, encoding='latin-1') for file in datafiles}
print([k for k in datafiles])
datafiles['WNCAATourneyCompactResults_PrelimData2018']['SecondaryTourney'] = 'NCAA'
datafiles['WRegularSeasonCompactResults_PrelimData2018']['SecondaryTourney'] = 'Regular'

#Presets
WLoc = {'A': 1, 'H': 2, 'N': 3}
SecondaryTourney = {'NIT': 1, 'CBI': 2, 'CIT': 3, 'V16': 4, 'Regular': 5 ,'NCAA': 6}

games = pd.concat((datafiles['WNCAATourneyCompactResults_PrelimData2018'],datafiles['WRegularSeasonCompactResults_PrelimData2018']), axis=0, ignore_index=True)
games.reset_index(drop=True, inplace=True)
games['WLoc'] = games['WLoc'].map(WLoc)
games['SecondaryTourney'] = games['SecondaryTourney'].map(SecondaryTourney)
games.head()

#Add Ids
games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

#Add Seeds
seeds = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in datafiles['WNCAATourneySeeds_SampleTourney2018'].values}
#Add 2018
if 2018 not in datafiles['WNCAATourneySeeds']['Season'].unique():
    seeds = {**seeds, **{k.replace('2017_','2018_'):seeds[k] for k in seeds if '2017_' in k}}

games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)

#Additional Features & Clean Up
games['ScoreDiff'] = games['WScore'] - games['LScore'] 
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed'] 
games = games.fillna(-1)

#Test Set
sub = datafiles['WSampleSubmissionStage1']
#sub = datafiles['WSampleSubmissionStage2_SampleTourney2018']
sub['WLoc'] = 3 #N
sub['SecondaryTourney'] = 6 #NCAA
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])
sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)
sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed'] 

#Leaky (No Validation)
sdn = games.groupby(['IDTeams'], as_index=False)[['ScoreDiffNorm']].mean()
sub = pd.merge(sub, sdn, how='left', on=['IDTeams'])
sub['ScoreDiffNorm'] = sub['ScoreDiffNorm'].fillna(0.)

#Interactions
inter = games[['IDTeam2','IDTeam1','Season','Pred']].rename(columns={'IDTeam2':'Target','IDTeam1':'Common'})
inter['Pred'] = inter['Pred'] * -1
inter = pd.concat((inter,games[['IDTeam1','IDTeam2','Season','Pred']].rename(columns={'IDTeam1':'Target','IDTeam2':'Common'})), axis=0, ignore_index=True).reset_index(drop=True)
inter = inter[inter['Season']>2013] #Limit
inter = pd.merge(inter, inter, how='inner', on=['Common','Season'])
inter = inter[inter['Target_x'] != inter['Target_y']]
inter['IDTeams'] = inter.apply(lambda r: '_'.join(map(str, [r['Target_x'].split('_')[1],r['Target_y'].split('_')[1]])), axis=1)
inter = inter[['IDTeams','Pred_x']]
inter = inter.groupby(['IDTeams'], as_index=False)[['Pred_x']].sum()
inter = {k:int(v) for k, v in inter.values}

games['Inter'] = games['IDTeams'].map(inter).fillna(0)
sub['Inter'] = sub['IDTeams'].map(inter).fillna(0)
col = [c for c in games.columns if c not in ['ID', 'Team1','Team2', 'IDTeams','IDTeam1','IDTeam2','Pred','DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'ScoreDiff']]

reg = linear_model.HuberRegressor()
reg.fit(games[col], games['Pred'])
sub['Pred'] = reg.predict(sub[col]).clip(0.05, 0.95)
sub[['ID','Pred']].to_csv('rh3p_submission.csv', index=False)

reg = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=18, n_estimators=100)
reg.fit(games[col], games['Pred'])
sub['Pred'] = reg.predict_proba(sub[col])[:,1]
sub['Pred'] = sub['Pred'].clip(0.05, 0.95)
sub[['ID','Pred']].to_csv('rh3p_etr_submission.csv', index=False)