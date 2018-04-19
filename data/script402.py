
# coding: utf-8

# ![](https://cdn-images-1.medium.com/max/1600/1*Q1rw8zw3pL7vCgTt3JU9RA.jpeg)

# Will update notebook regularly...
# 
# Current coverage:
# * Clean data
# * Data visualization (Heatmap, Violinplot)
# * Win rate by champion
# * Most imbalanced match-up
# * Best counter to certain champ + role
# * Predict win / loss given specific team comp
# * SKT vs SSG analysis (2017 World Championship BO5)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None
plt.style.use('ggplot')


# In[ ]:


champs = pd.read_csv('../input/champs.csv')
champs.shape


# In[ ]:


matches = pd.read_csv('../input/matches.csv')
matches.shape


# In[ ]:


participants = pd.read_csv('../input/participants.csv')
participants.shape


# In[ ]:


stats1 = pd.read_csv('../input/stats1.csv')
stats1.shape


# In[ ]:


stats2 = pd.read_csv('../input/stats2.csv')
stats2.shape


# In[ ]:


stats = stats1.append(stats2)
stats.shape


# In[ ]:


df = pd.merge(participants, stats, how = 'left', on = ['id'], suffixes=('', '_y'))
df = pd.merge(df, champs, how = 'left', left_on = 'championid', right_on = 'id', suffixes=('', '_y'))
df = pd.merge(df, matches, how = 'left', left_on = 'matchid', right_on = 'id', suffixes=('', '_y'))

def final_position(row):
    if row['role'] in ('DUO_SUPPORT', 'DUO_CARRY'):
        return row['role']
    else:
        return row['position']

df['adjposition'] = df.apply(final_position, axis = 1) 

df['team'] = df['player'].apply(lambda x: '1' if x <= 5 else '2')
df['team_role'] = df['team'] + ' - ' + df['adjposition']

# remove matchid with duplicate roles, e.g. 3 MID in same team, etc
remove_index = []
for i in ('1 - MID', '1 - TOP', '1 - DUO_SUPPORT', '1 - DUO_CARRY', '1 - JUNGLE', '2 - MID', '2 - TOP', '2 - DUO_SUPPORT', '2 - DUO_CARRY', '2 - JUNGLE'):
    df_remove = df[df['team_role'] == i].groupby('matchid').agg({'team_role':'count'})
    remove_index.extend(df_remove[df_remove['team_role']!=1].index.values)
    
# remove unclassified BOT, correct ones should be DUO_SUPPORT OR DUO_CARRY
remove_index.extend(df[df['adjposition'] == 'BOT']['matchid'].unique())
remove_index = list(set(remove_index))

print('# matches in dataset before cleaning: {}'.format(df['matchid'].nunique()))
df = df[~df['matchid'].isin(remove_index)]
print('# matches in dataset after cleaning: {}'.format(df['matchid'].nunique()))


# In[ ]:


df = df[['id', 'matchid', 'player', 'name', 'adjposition', 'team_role', 'win', 'kills', 'deaths', 'assists', 'turretkills','totdmgtochamp', 'totheal', 'totminionskilled', 'goldspent', 'totdmgtaken', 'inhibkills', 'pinksbought', 'wardsplaced', 'duration', 'platformid', 'seasonid', 'version']]
df.head(10)


# In[ ]:


df_v = df.copy()
# put upper and lower limit
df_v['wardsplaced'] = df_v['wardsplaced'].apply(lambda x: x if x<30 else 30)
df_v['wardsplaced'] = df_v['wardsplaced'].apply(lambda x: x if x>0 else 0)

plt.figure(figsize = (15,10))
sns.violinplot(x="seasonid", y="wardsplaced", hue="win", data=df_v, split=True, inner = 'quartile')
plt.title('Wardsplaced by season: win vs loss')


# We can see that players placed more wards in recent seasons, no matter win or loss. Wards play a more important part nowadays

# In[ ]:


plt.figure(figsize = (15,10))

# put upper and lower limit
df_v['kills'] = df_v['kills'].apply(lambda x: x if x<20 else 20)
df_v['kills'] = df_v['kills'].apply(lambda x: x if x>0 else 0)

sns.violinplot(x="seasonid", y="kills", hue="win", data=df_v, split=True, inner = 'quartile')
plt.title('Kills by season: win vs loss')


# In[ ]:


df_corr = df._get_numeric_data()
df_corr = df_corr.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

mask = np.zeros_like(df_corr.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 150, as_cmap=True)

plt.figure(figsize = (15,10))
sns.heatmap(df_corr.corr(), cmap = cmap, annot = True, fmt = '.2f', mask = mask, square=True, linewidths=.5, center = 0)
plt.title('Correlations - win vs factors (all games)')


# In[ ]:


df_corr_2 = df._get_numeric_data()
# for games less than 25mins
df_corr_2 = df_corr_2[df_corr_2['duration'] <= 1500]
df_corr_2 = df_corr_2.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

mask = np.zeros_like(df_corr_2.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 150, as_cmap=True)

plt.figure(figsize = (15,10))
sns.heatmap(df_corr_2.corr(), cmap = cmap, annot = True, fmt = '.2f', mask = mask, square=True, linewidths=.5, center = 0)
plt.title('Correlations - win vs factors (for games last less than 25 mins)')


# In[ ]:


df_corr_3 = df._get_numeric_data()
# for games more than 40mins
df_corr_3 = df_corr_3[df_corr_3['duration'] > 2400]
df_corr_3 = df_corr_3.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

mask = np.zeros_like(df_corr_3.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(10, 150, as_cmap=True)

plt.figure(figsize = (15,10))
sns.heatmap(df_corr_3.corr(), cmap = cmap, annot = True, fmt = '.2f', mask = mask, square=True, linewidths=.5, center = 0)
plt.title('Correlations - win vs factors (for games last more than 40 mins)')


# In[ ]:


pd.options.display.float_format = '{:,.1f}'.format

df_win_rate = df.groupby('name').agg({'win': 'sum', 'name': 'count', 'kills': 'mean', 'deaths': 'mean', 'assists': 'mean'})
df_win_rate.columns = ['win matches', 'total matches', 'K', 'D', 'A']
df_win_rate['win rate'] = df_win_rate['win matches'] /  df_win_rate['total matches'] * 100
df_win_rate['KDA'] = (df_win_rate['K'] + df_win_rate['A']) / df_win_rate['D']
df_win_rate = df_win_rate.sort_values('win rate', ascending = False)
df_win_rate = df_win_rate[['total matches', 'win rate', 'K', 'D', 'A', 'KDA']]
print('Top 10 win rate')
print(df_win_rate.head(10))
print('Bottom 10 win rate')
print(df_win_rate.tail(10))


# In[ ]:


df_win_rate.reset_index(inplace = True)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

df_win_rate['color map'] = df_win_rate['win rate'].apply(lambda x: 'green' if x > 50 else 'red')

ax = df_win_rate.plot(kind = 'scatter', x = 'total matches', y = 'win rate', color = df_win_rate['color map'].tolist(), figsize = (15,10), title = 'win rate vs # matches by champions')

label_point(df_win_rate['total matches'], df_win_rate['win rate'], df_win_rate['name'], ax)


# In[ ]:


pd.options.display.float_format = '{:,.1f}'.format

df_win_rate_role = df.groupby(['name','adjposition']).agg({'win': 'sum', 'name': 'count', 'kills': 'mean', 'deaths': 'mean', 'assists': 'mean'})
df_win_rate_role.columns = ['win matches', 'total matches', 'K', 'D', 'A']
df_win_rate_role['win rate'] = df_win_rate_role['win matches'] /  df_win_rate_role['total matches'] * 100
df_win_rate_role['KDA'] = (df_win_rate_role['K'] + df_win_rate_role['A']) / df_win_rate_role['D']
df_win_rate_role = df_win_rate_role.sort_values('win rate', ascending = False)
df_win_rate_role = df_win_rate_role[['total matches', 'win rate', 'K', 'D', 'A', 'KDA']]

# occur > 0.01% of matches
df_win_rate_role = df_win_rate_role[df_win_rate_role['total matches'] > df_win_rate_role['total matches'].sum()*0.0001]
print('Top 10 win rate with role (occur > 0.01% of total # matches)')
print(df_win_rate_role.head(10))
print('Bottom 10 win rate with role (occur > 0.01% of total # matches)')
print(df_win_rate_role.tail(10))


# In[ ]:


df_2 = df.sort_values(['matchid','adjposition'], ascending = [1,1])

df_2['shift 1'] = df_2['name'].shift()
df_2['shift -1'] = df_2['name'].shift(-1)

def get_matchup(x):
    if x['player'] <= 5:
        if x['name'] < x['shift -1']:
            name_return = x['name'] + ' vs ' + x['shift -1']
        else:
            name_return = x['shift -1'] + ' vs ' + x['name']
    else:
        if x['name'] < x['shift 1']:
            name_return = x['name'] + ' vs ' + x['shift 1']
        else:
            name_return = x['shift 1'] + ' vs ' + x['name']
    return name_return

df_2['match up'] = df_2.apply(get_matchup, axis = 1)
df_2['win_adj'] = df_2.apply(lambda x: x['win'] if x['name'] == x['match up'].split(' vs')[0] else 0, axis = 1)

df_2.head(10)
    


# In[ ]:


df_matchup = df_2.groupby(['adjposition', 'match up']).agg({'win_adj': 'sum', 'match up': 'count'})
df_matchup.columns = ['win matches', 'total matches']
df_matchup['total matches'] = df_matchup['total matches'] / 2
df_matchup['win rate'] = df_matchup['win matches'] /  df_matchup['total matches']  * 100
df_matchup['dominant score'] = df_matchup['win rate'] - 50
df_matchup['dominant score (ND)'] = abs(df_matchup['dominant score'])
df_matchup = df_matchup[df_matchup['total matches'] > df_matchup['total matches'].sum()*0.0001]

df_matchup = df_matchup.sort_values('dominant score (ND)', ascending = False)
df_matchup = df_matchup[['total matches', 'dominant score']]                   
df_matchup = df_matchup.reset_index()

print('Dominant score +/- means first/second champion dominant:')

for i in df_matchup['adjposition'].unique(): 
        print('\n{}:'.format(i))
        print(df_matchup[df_matchup['adjposition'] == i].iloc[:,1:].head(5))


# In[ ]:


def get_best_counter(champion, role):
    df_matchup_temp = df_matchup[(df_matchup['match up'].str.contains(champion)) & (df_matchup['adjposition'] == role)]
    df_matchup_temp['champion'] = df_matchup_temp['match up'].apply(lambda x: x.split(' vs ')[0] if x.split(' vs ')[1] == champion else x.split(' vs ')[1])
    df_matchup_temp['advantage'] = df_matchup_temp.apply(lambda x: x['dominant score']*-1 if x['match up'].split(' vs ')[0] == champion else x['dominant score'], axis = 1)
    df_matchup_temp = df_matchup_temp[df_matchup_temp['advantage']>0].sort_values('advantage', ascending = False)
    print('Best counter for {} - {}:'.format(role, champion))
    print(df_matchup_temp[['champion', 'total matches', 'advantage']])
    return

# you know why
champion = 'Teemo'
role = 'TOP'
get_best_counter(champion, role)


# In[ ]:


# Ahri seems OP from previous result, let's see who counters her
champion = 'Ahri'
role = 'MID'
get_best_counter(champion, role)


# In[ ]:


# how about cocky Riven
champion = 'Riven'
role = 'TOP'
get_best_counter(champion, role)

# Conventional counter = Renekton, Unconventional counter = Xin Zhao (more effective as well)


# In[ ]:


# pivot data to what we want in format

df_3 = df[['matchid', 'player', 'name', 'team_role', 'win']]

df_3 = df_3.pivot(index = 'matchid', columns = 'team_role', values = 'name')
df_3 = df_3.reset_index()
df_3 = df_3.merge(df[df['player'] == 1][['matchid', 'win']], left_on = 'matchid', right_on = 'matchid', how = 'left')
df_3 = df_3[df_3.columns.difference(['matchid'])]
df_3 = df_3.rename(columns = {'win': 'T1 win'})

df_3.head(10)


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.cross_validation import train_test_split

# remove missing data
print('Before drop missing data: {}'.format(len(df_3)))
df_3 = df_3.dropna()
print('After drop missing data: {}'.format(len(df_3)))

y = df_3['T1 win']
X = df_3[df_3.columns.difference(['T1 win'])]

# label string to numeric
le_t = X.apply(le.fit)
X_t_1 = X.apply(le.fit_transform)

enc = preprocessing.OneHotEncoder()
enc_t = enc.fit(X_t_1)
X_t_2 = enc_t.transform(X_t_1)

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_t_1, y, random_state=0)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_t_2, y, random_state=0)



# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
import xgboost
import seaborn as sns

print('Accuracy on dataset converted from label to integer category:')

clf_lr = LogisticRegression(random_state = 0).fit(X_train_1, y_train_1)
acc_lr = clf_lr.score(X_test_1, y_test_1)
print('logistic regression : {}'.format(acc_lr))

clf_bnb = BernoulliNB().fit(X_train_1, y_train_1)
acc_bnb = clf_bnb.score(X_test_1, y_test_1)
print('naive bayes : {}'.format(acc_bnb))

clf_xb = xgboost.XGBClassifier(random_state = 0).fit(X_train_1, y_train_1)
acc_xb = clf_xb.score(X_test_1, y_test_1)
print('xgboost : {}'.format(acc_xb))


print('\n')

# category with just 0 / 1, no magnitude meaning in category like above approach
print('Accuracy on dataset converted from label to binary category:')

clf_lr = LogisticRegression(random_state = 0).fit(X_train_2, y_train_2)
acc_lr = clf_lr.score(X_test_2, y_test_2)
print('logistic regression : {}'.format(acc_lr))

clf_bnb = BernoulliNB().fit(X_train_2, y_train_2)
acc_bnb = clf_bnb.score(X_test_2, y_test_2)
print('naive bayes : {}'.format(acc_bnb))

clf_xb = xgboost.XGBClassifier(random_state = 0).fit(X_train_2, y_train_2)
acc_xb = clf_xb.score(X_test_2, y_test_2)
print('xgboost : {}'.format(acc_xb))


# In[ ]:


# adding platform id (player base style) and season id (time) as features

df_4 = df[['matchid', 'player', 'name', 'team_role', 'win']]

df_4 = df_4.pivot(index = 'matchid', columns = 'team_role', values = 'name')
df_4 = df_4.reset_index()
df_4 = df_4.merge(df[df['player'] == 1][['matchid', 'win', 'platformid', 'seasonid', 'version']], left_on = 'matchid', right_on = 'matchid', how = 'left')
df_4 = df_4[df_4.columns.difference(['matchid', 'version'])]
df_4 = df_4.rename(columns = {'win': 'T1 win'})

df_4.head(10)


# In[ ]:


df['name'].unique()


# In[ ]:


# remove missing data

df_4 = df_4.dropna()

y = df_4['T1 win']
X = df_4[df_4.columns.difference(['T1 win'])]

# prepare for prediction, below is a team comp I am curious about
s = pd.Series(['KogMaw', 'Leona', 'Lee Sin', 'Ahri', 'Riven', 'Kalista', 'Blitzcrank', 'Xin Zhao', 'Ryze', 'Shen', 'NA1', 8], index = df_4[df_4.columns.difference(['T1 win'])].columns)

# prepare for SKT vs SSG, no KR server available so I use NA as reference
s_g1 = pd.Series(['Varus', 'Lulu', 'Gragas', 'Cassiopeia', 'Gnar', 'Xayah', 'Janna', 'Zac', 'Malzahar', 'Kennen', 'NA1', 8], index = df_4[df_4.columns.difference(['T1 win'])].columns)
s_g2 = pd.Series(['Varus', 'Lulu', 'Gragas', 'Ryze', 'Yasuo', 'Xayah', 'Janna', 'Jarvan IV', 'Malzahar', 'Gnar', 'NA1', 8], index = df_4[df_4.columns.difference(['T1 win'])].columns)
s_g3 = pd.Series(['Tristana', 'Leona', 'Gragas', 'Karma', 'Trundle', 'Varus', 'Lulu', 'Sejuani', 'Malzahar', 'ChoGath', 'NA1', 8], index = df_4[df_4.columns.difference(['T1 win'])].columns)


# add for transformation
X = X.append(s, ignore_index=True)
X = X.append(s_g1, ignore_index=True)
X = X.append(s_g2, ignore_index=True)
X = X.append(s_g3, ignore_index=True)

# label string to numeric
le_t = X.apply(le.fit)
X_t_1 = X.apply(le.fit_transform)

enc = preprocessing.OneHotEncoder()
enc_t = enc.fit(X_t_1)
X_t_2 = enc_t.transform(X_t_1)

# split train & test, exclude last row (our curious comp)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_t_2[:-4], y, random_state=0)

# category with just 0 / 1, no magnitude meaning in category like above approach
print('Accuracy on dataset converted from label to binary category (with platform id, seasonid):')

clf_lr = LogisticRegression(random_state = 0).fit(X_train_2, y_train_2)
acc_lr = clf_lr.score(X_test_2, y_test_2)
print('logistic regression : {}'.format(acc_lr))

clf_bnb = BernoulliNB().fit(X_train_2, y_train_2)
acc_bnb = clf_bnb.score(X_test_2, y_test_2)
print('naive bayes : {}'.format(acc_bnb))

clf_xb = xgboost.XGBClassifier(random_state = 0).fit(X_train_2, y_train_2)
acc_xb = clf_xb.score(X_test_2, y_test_2)
print('xgboost : {}'.format(acc_xb))


# low accuracy might mean Riot is doing a pretty good job in balancing the champion, or people tends to play OP champs more

# In[ ]:


# predict certain team comp win rate
print('Team 1 win rate for below team comp: {} \n{}'.format(clf_lr.predict_proba(X_t_2[-4])[0][1], pd.DataFrame(s, columns = ['Champ'])))


# Reasonable if you are familiar with the scene, right?

# In[ ]:


# predict SKT vs SSG 2017 World Final win rate
print('Team 1 = SKT / Team 2 = SSG')
print('Game 1 SKT win rate: {} \n{}\n'.format(clf_lr.predict_proba(X_t_2[-3])[0][1], pd.DataFrame(s_g1, columns = ['Champ'])))
print('Game 2 SKT win rate: {} \n{}\n'.format(clf_lr.predict_proba(X_t_2[-2])[0][1], pd.DataFrame(s_g2, columns = ['Champ'])))
print('Game 3 SKT win rate: {} \n{}\n'.format(clf_lr.predict_proba(X_t_2[-1])[0][1], pd.DataFrame(s_g3, columns = ['Champ'])))


# Looks like relatively bad draft for SKT for all 3 games...especially game 2 which they try for a more creative pick - yasuo

# ![](https://realsport101.com/wp-content/uploads/2017/11/ssg-crown-winning-1.jpg)
