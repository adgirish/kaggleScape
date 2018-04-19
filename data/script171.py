
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/data.csv')
df.info()


# In[ ]:


# Court visualization of misses and shots
court_scale, alpha = 7, 0.05
plt.figure(figsize=(2 * court_scale, court_scale*(84.0/50.0)))
# hit
plt.subplot(121)
h = df.loc[df.shot_made_flag == 1]
plt.scatter(h.loc_x, h.loc_y, color='green', alpha=alpha)
plt.title('Shots Made')
ax = plt.gca()
ax.set_ylim([-50, 900])
# miss
plt.subplot(122)
h = df.loc[df.shot_made_flag == 0]
plt.scatter(h.loc_x, h.loc_y, color='red', alpha=alpha)
plt.title('Shots missed')
ax = plt.gca()
ax.set_ylim([-50, 900])
plt.savefig('shots_made_and_missed.png')


# Notice the red dot at where the basket should be? He misses a lot of shots from under the basket. Must be those common ball scenarios.

# In[ ]:


# combined shot types
groups = df.groupby('combined_shot_type')


fig, ax = plt.subplots(figsize=(court_scale, court_scale*(84.0/50.0)))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
alpha = 0.2
alphas, n = [], float(len(df.combined_shot_type))
for u in [i[0] for i in groups]:
    d = len(df.loc[df.combined_shot_type == u, 'combined_shot_type'])
    alphas.append(np.log1p(d))
for (name, group), alp in zip(groups, alphas):
    ax.plot(group.loc_x, group.loc_y,
            marker='.', linestyle='', ms=12,
            label=name, alpha=alp)
ax.legend()
plt.savefig('combined_shot_type_layout.png')


# In[ ]:


court_scale, alpha = 5, 0.5
df['unique_first_words'] = df.action_type.str.split(' ').str[0]
uq_count = len(df['unique_first_words'].unique())
a = int(uq_count / 2) + 1

groups = df.groupby('unique_first_words')
fig, ax = plt.subplots(figsize=(2 * court_scale, a * 1.1 * court_scale*(84.0/50.0)))
X, Y = np.array([(i, 0) for i in np.arange(-400, 400, 0.1)]), np.array([(0, i) for i in np.arange(-60, 1000, 0.1)])
for index, (name, group) in enumerate(groups):
    plt.subplot(a, 2, index + 1)
    h = group.loc[group.shot_made_flag == 1, ['loc_y', 'loc_x']]
    m = group.loc[group.shot_made_flag == 0, ['loc_y', 'loc_x']]
    
    plt.plot(h.loc_x, h.loc_y,
            marker='.', linestyle='', ms=12,
            label=name, alpha=alpha, color='green')
    
    plt.plot(m.loc_x, m.loc_y,
            marker='.', linestyle='', ms=12,
            label=name, alpha=alpha, color='red')
    x_lim = group.loc_x.mean() + 3* group.loc_x.std()
    y_lim = group.loc_y.mean() + 3* group.loc_y.std()
    plt.plot(X[:, 0], X[:, 1], 'black')
    plt.plot(Y[:, 0], Y[:, 1], 'black')
    plt.xlim([-x_lim, x_lim])
    plt.ylim([-y_lim, y_lim])
    
    plt.title(name)
    plt.savefig('action_type_first_words.png')


# In[ ]:


# Shot location by seconds remaining.
# farther shots are made only when next to zero seconds are left
# The greener the dot, the less the time left
court_scale, alpha = 7, 0.1

fig = plt.figure(figsize=(2 * court_scale, court_scale*(84.0/50.0)))
plt.subplot(121)
plt.scatter(df.loc_x, df.loc_y, alpha=alpha, c=df.seconds_remaining, cmap='Greens_r')
plt.title('Seconds Remaining')
plt.subplot(122)
plt.scatter(df.loc_x, df.loc_y, alpha=alpha, c=df.minutes_remaining, cmap='Greens_r')
plt.title('Minutes Remaining')
plt.savefig('time_remaining_shot_layout.png')


# In[ ]:


# Shooting accuracy with shot distance
def get_acc(df, against):
    ct = pd.crosstab(df.shot_made_flag, df[against]).apply(lambda x:x/x.sum(), axis=0)
    x, y = ct.columns, ct.values[1, :]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y)
    plt.xlabel(against)
    plt.ylabel('% shots made')
    plt.savefig(against + '_vs_accuracy.png')
get_acc(df, 'shot_distance')


# In[ ]:


data = df[['loc_x', 'loc_y', 'shot_made_flag']]
data = data.dropna()
def test_it(data):
    clf = RandomForestClassifier(n_jobs=-1)  # A super simple classifier
    return cross_val_score(clf, data.drop('shot_made_flag', 1), data.shot_made_flag,
                           scoring='roc_auc', cv=10
                          )
test_it(data).mean()


# In[ ]:


# A joint plot should give us some density measures
# Looks like a classifier will do a terrible job at this.
sns.jointplot(x="loc_x", y="loc_y", data=data, kind='kde')


# In[ ]:


# If we take only the y location into consideration, we should see some improvement
data = df[['loc_y', 'shot_made_flag']]
data = data.dropna()
test_it(data).mean()


# In[ ]:


# That does make improvements, though a more accurate measure of what we are
# trying to do here would be shot_distance
data = df[['shot_distance', 'shot_made_flag']]
data = data.dropna()
test_it(data).mean()


# In[ ]:


# What can we learn from time?
get_acc(df, 'seconds_remaining')


# In[ ]:


# Not much there, except perhapsin the < 5 second zone
# Let's test it
data = df[['seconds_remaining', 'shot_distance', 'shot_made_flag']].dropna()
test_it(data).mean()


# In[ ]:


# Nope! as expected, this is not a good estimator feature.
# Kobe would not change performance with pressure! He's better than that.
# Let's see minutes remaining, simply because we can
get_acc(df, 'minutes_remaining')


# In[ ]:


# Not much there either. Let's see which period is better for our man
get_acc(df, 'period')


# In[ ]:


# Not much variation there either. Kobe is really consistent.
# let's see season
print(df.season.unique())
df['season_start_year'] = df.season.str.split('-').str[0]
df['season_start_year'] = df['season_start_year'].astype(int)
get_acc(df, 'season_start_year')


# In[ ]:


# Although we are seeing some seasonality here, let's not forget the scale.
# To be sure, we add this and test our classifier

data = df[['season_start_year', 'shot_distance', 'shot_made_flag']].dropna()
test_it(data).mean()


# In[ ]:


# As expected, our predictions decrease in effectiveness.
# What about the action type field?
action_map = {action: i for i, action in enumerate(df.action_type.unique())}
df['action_type_enumerated'] = df.action_type.map(action_map)
get_acc(df, 'action_type_enumerated')


# In[ ]:


# There is a lot of variation here. Probably if we could exploit the inherent
# ordering of numbers and not depend on the enumeration values...(I degress)
# Instead of one-hot encoding it let's create a map which sorts them in order of
# increasing accuracy
def sort_encode(df, field):
    ct = pd.crosstab(df.shot_made_flag, df[field]).apply(lambda x:x/x.sum(), axis=0)
    temp = list(zip(ct.values[1, :], ct.columns))
    temp.sort()
    new_map = {}
    for index, (acc, old_number) in enumerate(temp):
        new_map[old_number] = index
    new_field = field + '_sort_enumerated'
    df[new_field] = df[field].map(new_map)
    get_acc(df, new_field)
sort_encode(df, 'action_type_enumerated')


# In[ ]:


# Pretty neat now huh? Let's see if all this monkeying around has been useful or not.
data = df[['action_type_enumerated', 'shot_distance', 'shot_made_flag']].dropna()
x = test_it(data)
data = df[['action_type_enumerated_sort_enumerated', 'shot_distance', 'shot_made_flag']].dropna()
y = test_it(data)
print(x.mean(), y.mean())


# In[ ]:


# Seems like action type gave a good kick to our prediction capability. The difference in
# scores is negligible enough to have come from random seeds in the RNG
# Ordering it only made our graph look nicer. It would impact another classifier though,
# something like an SVM which works on kernel methods would be affected by the mapping.

# Let's look at opponents. We do the same things as with action_types.
opponent_map = {opp: i for i, opp in enumerate(df.opponent.unique())}
df['opponent_enumerated'] = df.opponent.map(opponent_map)

sort_encode(df, 'opponent_enumerated')


# In[ ]:


# this might have an effect. Let's see by throwing it in the mix.
data = df[['action_type_enumerated', 'shot_distance',
           'shot_made_flag', 'opponent_enumerated_sort_enumerated']].dropna()
test_it(data).mean()


# In[ ]:


# And we lose improvement again. Let's see what matchup has to offer. We notice that there are two
# things in this field. A vs. B and A @ B. Assuming that @ means away, let's see if it offers any
# improvements
df['away'] = df.matchup.str.contains('@')
data = df[['action_type_enumerated', 'shot_distance',
           'shot_made_flag', 'away']].dropna()
test_it(data).mean()


# In[ ]:


# Some minor improvements have occured. Try forking this notebook and looking at other fields
# See what combinations offer advantages. We now move to the classifier.
# We focus our attention on tuning the classifier. To keep things simple we will use the same
# one we have been using up till now.
data = df[['action_type_enumerated', 'shot_distance',
           'shot_made_flag', 'away']].dropna()

# We see how score improves with estimators.
estimators, scores = list(range(1, 100, 5)), []
for i in estimators:
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=i, random_state=2016)
    x = cross_val_score(clf, data.drop('shot_made_flag', 1), data.shot_made_flag,
                              scoring='roc_auc', cv=10)
    scores.append(x)
x = [i for i in estimators for j in range(10)]
sns.boxplot(x, np.array(scores).flatten())


# In[ ]:


# As seen, we probably need about 70 estimators to have a good enough estimator.
# After that it's diminished results.
# Let's look at tree depth.
depth, scores = list(range(1, 20, 1)), []
for i in depth:
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=i, random_state=2016)
    x = cross_val_score(clf, data.drop('shot_made_flag', 1), data.shot_made_flag,
                              scoring='roc_auc', cv=10)
    scores.append(x)
x = [i for i in depth for j in range(10)]
sns.boxplot(x, np.array(scores).flatten())


# In[ ]:


# So based on the peak of the boxplots we select 7 to be our maxdepth
# Let's predict based on all this work for now. We avoid looking at Leakage for the time
# being and move on..

clf = RandomForestClassifier(n_jobs=-1, n_estimators=70, max_depth=7, random_state=2016) # a more powerful classifier

train = df.loc[~df.shot_made_flag.isnull(), ['action_type_enumerated_sort_enumerated',
                                             'shot_distance', 'shot_made_flag', 'away']]
test = df.loc[df.shot_made_flag.isnull(), ['action_type_enumerated_sort_enumerated',
                                           'shot_distance', 'shot_id', 'away']]
# Impute
mode = test.action_type_enumerated_sort_enumerated.mode()[0]
test.action_type_enumerated_sort_enumerated.fillna(mode, inplace=True)

# Train and predict
clf.fit(train.drop('shot_made_flag', 1), train.shot_made_flag)
predictions = clf.predict_proba(test.drop('shot_id', 1))


# In[ ]:


#print(check_output(["cat", "../input/sample_submission.csv"]).decode("utf8"))
#shot_id,shot_made_flag
#1,0

submission = pd.DataFrame({'shot_id': test.shot_id,
                           'shot_made_flag': predictions[:, 1]})
submission[['shot_id', 'shot_made_flag']].to_csv('submission.csv', index=False)


# # Something important though:
# 
# - a lot of our graphs were showing Kobe's accuracy below 0.5
# - It reminds me of the illusion of skill as discussed in "Thinking, Fast and Slow" 
# - It does go up(accuracy) to 1 in the action type graph but I suspect it is due to small sample size in that type of action.
