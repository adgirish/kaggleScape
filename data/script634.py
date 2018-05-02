
# coding: utf-8

# *Last edit by David Lao - 2018/04/02*
# <br>
# <br>
# 
# ![](https://s.yimg.com/ny/api/res/1.2/sybAk0uKKJIlkVhfz5Jtuw--/YXBwaWQ9aGlnaGxhbmRlcjtzbT0xO3c9ODAw/http://media.zenfs.com/en-GB/homerun/uk.goal.com/42bf6d3fd31612f17e7b59c6260595db)
# 
# 
# # FIFA Analytics - Predict Player Position through Machine Learning
# 
# 
# 
# After part 1 analysis on [finding the best squad](https://www.kaggle.com/laowingkin/fifa-18-find-the-best-squad), I will try to train the model and predict the player's preferred position based on different features. The dataset contains all FIFA 18 players **(~18K players)** with **70+** attributes.
# 
# 
# ## Contents
# 
# * Data manipulation
# * Data normalization (and why it is needed)
# * Predict binary targets (attack vs defend positions) with logistic regression
# * Feature selections
# * Study PCA impact on model accuracy
# * Study LDA impact on model accuracy (*Edit on 2017/10/13*)
# * Predict all positions (14 targets) with logistic regression (*Edit on 2017/10/06*)
# * Predict all positions (14 targets) with random forest (*Edit on 2017/10/06*)
# * Predict all positions (14 targets) with neural networks (*Edit on 2017/10/09*)
# 
# <br>
# Appreciate if you can **Upvote** if this notebook is helpful to you in some ways!
# 

# ## Data manipulation

# First let's import some packages:

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

get_ipython().run_line_magic('matplotlib', 'inline')


# Load the data:

# In[ ]:


df = pd.read_csv('../input/CompleteDataset.csv')
df.head()


# Check out columns:

# In[ ]:


df.columns


# Gather only columns that we need for this analysis purpose:

# In[ ]:


# GK attributes are not our interest
columns_needed = ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control',
       'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing',
       'Free kick accuracy', 'Heading accuracy', 'Interceptions',
       'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties',
       'Positioning', 'Reactions', 'Short passing', 'Shot power',
       'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle',
       'Strength', 'Vision', 'Volleys', 'Preferred Positions']

# attack attribute first, then defence, then mixed
columns_needed_rearranged = ['Aggression','Crossing', 'Curve', 'Dribbling', 'Finishing',
       'Free kick accuracy', 'Heading accuracy', 'Long shots','Penalties', 'Shot power', 'Volleys', 
       'Short passing', 'Long passing',
       'Interceptions', 'Marking', 'Sliding tackle', 'Standing tackle',
       'Strength', 'Vision', 'Acceleration', 'Agility', 
       'Reactions', 'Stamina', 'Balance', 'Ball control','Composure','Jumping', 
       'Sprint speed', 'Positioning','Preferred Positions']

df = df[columns_needed_rearranged]
df.head()


# We don't want to classify GK because it will be too obvious:

# In[ ]:


df['Preferred Positions'] = df['Preferred Positions'].str.strip()
df = df[df['Preferred Positions'] != 'GK']
df.head()



# Check any missing data:

# In[ ]:


df.isnull().values.any()


# All possible outcome for preferred position:

# In[ ]:


p = df['Preferred Positions'].str.split().apply(lambda x: x[0]).unique()
p


# Handle players with multiple preferred positions: duplicate a set of data for each

# In[ ]:


# copy a structure
df_new = df.copy()
df_new.drop(df_new.index, inplace=True)

for i in p:
    df_temp = df[df['Preferred Positions'].str.contains(i)]
    df_temp['Preferred Positions'] = i
    df_new = df_new.append(df_temp, ignore_index=True)
    
df_new.iloc[::500, :]
            



# Some of the attributes have '+/-' sign, let's perform the calculation rather than keeping them as string:

# In[ ]:


cols = [col for col in df_new.columns if col not in ['Preferred Positions']]

for i in cols:
    df_new[i] = df_new[i].apply(lambda x: eval(x) if isinstance(x,str) else x)

df_new.iloc[::500, :]


# ## Data normalization (and why it is needed)

# How does the attributes pattern look like for each position? i.e. ST with resampling of every 200 players:

# In[ ]:


fig, ax = plt.subplots()
df_new_ST = df_new[df_new['Preferred Positions'] == 'ST'].iloc[::200,:-1]
df_new_ST.T.plot.line(color = 'black', figsize = (15,10), legend = False, ylim = (0, 110), title = "ST's attributes distribution", ax=ax)

ax.set_xlabel('Attributes')
ax.set_ylabel('Rating')

ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(labels = cols, rotation=90)

for ln in ax.lines:
    ln.set_linewidth(1)

ax.axvline(0, color='red', linestyle='--')   
ax.axvline(12.9, color='red', linestyle='--')

ax.axvline(13, color='blue', linestyle='--')
ax.axvline(17, color='blue', linestyle='--')

ax.axvline(17.1, color='green', linestyle='--')
ax.axvline(28, color='green', linestyle='--')

ax.text(5, 100, 'Attack Attributes', color = 'red', weight = 'bold')
ax.text(13.5, 100, 'Defend Attributes', color = 'blue', weight = 'bold')
ax.text(22, 100, 'Mixed Attributes', color = 'green', weight = 'bold')


# The pattern is quite obvious, but some outliers are found. It is because some of the players are extremely good at their positions and some of them aren't. But I believe most of them should follow the similar pattern if we have normalized them, for the same position. Let's try it out:

# In[ ]:


df_new_ST_normalized = df_new_ST.div(df_new_ST.sum(axis=1), axis=0)

fig, ax = plt.subplots()
df_new_ST_normalized.T.plot.line(color = 'black', figsize = (15,10), ylim = (0, 0.08), legend = False, title = "ST's attributes distribution (normalized)", ax=ax)

ax.set_xlabel('Attributes')
ax.set_ylabel('Normalized Rating')

ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(labels = cols, rotation=90)

for ln in ax.lines:
    ln.set_linewidth(1)

ax.axvline(0, color='red', linestyle='--')   
ax.axvline(12.9, color='red', linestyle='--')

ax.axvline(13, color='blue', linestyle='--')
ax.axvline(17, color='blue', linestyle='--')

ax.axvline(17.1, color='green', linestyle='--')
ax.axvline(28, color='green', linestyle='--')

ax.text(5, 0.07, 'Attack Attributes', color = 'red', weight = 'bold')
ax.text(13.5, 0.07, 'Defend Attributes', color = 'blue', weight = 'bold')
ax.text(22, 0.07, 'Mixed Attributes', color = 'green', weight = 'bold') 


# ## Predict binary targets (attack vs defend positions) with logistic regression

# The pattern after normalization looks much more obvious. Lets do below:
# 
# * Normalize the whole dataset
# 
# 
# * Reclassify the target value (preferred positions) to binary groups as below:
# - 1 =  attack positions = ST, RW, LW, RM, CM, LM, CAM, CF 
# - 0 = defened positions = CDM, CB, LB, RB, RWB, LWB
# 
# 

# In[ ]:


df_new_normalized = df_new.iloc[:,:-1].div(df_new.iloc[:,:-1].sum(axis=1), axis=0)
mapping = {'ST': 1, 'RW': 1, 'LW': 1, 'RM': 1, 'CM': 1, 'LM': 1, 'CAM': 1, 'CF': 1, 'CDM': 0, 'CB': 0, 'LB': 0, 'RB': 0, 'RWB': 0, 'LWB': 0}
df_new_normalized['Preferred Positions'] = df_new['Preferred Positions']
df_new_normalized = df_new_normalized.replace({'Preferred Positions': mapping})

df_new_normalized.iloc[::1000,]


# Split train test dataset:

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_new_normalized.iloc[:,:-1], df_new_normalized.iloc[:,-1], random_state=0)

print('X train shape: {}'.format(X_train.shape))
print('X test shape: {}'.format(X_test.shape))
print('y train shape: {}'.format(y_train.shape))
print('y test shape: {}'.format(y_test.shape))


# Apply logistic regression to training set:

# In[ ]:


clf_d = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
acc_d = clf_d.score(X_test, y_test)
print ('Dummy Classifier (most frequent class): {}'.format(acc_d))

clf = LogisticRegression().fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print ('Logistic Regression Accuracy: {}'.format(acc))


# ## Feature selections

# Can we further improve the model accuracy by looking at feature importance?

# In[ ]:


Coef_list = list(sorted(zip(X_train.columns, abs(clf.coef_[0])),key=lambda x: -x[1]))
Coef_table = pd.DataFrame(np.array(Coef_list).reshape(-1,2), columns = ['Attributes', 'Coef'])

print (Coef_table)


# Let's try limit the features to top 10 only:

# In[ ]:


target_cols = Coef_table[:10]['Attributes'].tolist()

clf_2 = LogisticRegression().fit(X_train[target_cols], y_train)
acc_2 = clf_2.score(X_test[target_cols], y_test)
print ('Logistic Regression Accuracy (10 features): {}'.format(acc_2))


# The accuracy is slightly improved. Another interesting observation is **positioning** attribute is quite significant in the model and probably should not be classified under *mix* attributes, while some *attack* attributes like **crossing** is not that significant (probably because LB / RB have high crossing ratings too)
# 
# Another interesting observation is that the most significant attributes are mostly defensive ones, meaning that they are most helpful to help us identify the role of a player. In other words, it is extremely rare for a player in an attacking role like ST to be good at those defensive skills, while it is relatively easier for a player in a defensive role to be good at some attacking skills like crossing and free kick accuracy, etc.

# ## Study PCA impact on model accuracy

# Reviewing the whole project again, I would think some of the attributes are quite related to others and therefore I can make use of some dimension reduction technique (namely PCA), I reference this [article](https://www.kaggle.com/arthurtok/principal-component-analysis-with-kmeans-visuals) from [Anisotropic](https://www.kaggle.com/arthurtok) for helping me to implement the model. Let's look at heatmap first:

# In[ ]:


f, ax = plt.subplots(figsize=(20, 20))

plt.title('Pearson Correlation of Player attributes')

sns.heatmap(df_new.corr(),linewidths=0.25,vmax=1.0, square=True, cmap = 'PuBu', linecolor='black', annot=True)


# Let's find the reasonable number of component of PCA *(I re-used most codes from above article and only modify to suit my needs)*:

# In[ ]:


cov_mat = np.cov(df_new.iloc[:,:-1].T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

print(list(zip(range(29),cum_var_exp)))

# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 
plt.figure(figsize=(10, 10))
plt.bar(range(len(var_exp)), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')
plt.step(range(len(var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()



# 17 components are needed for 95%+ of variance level:

# In[ ]:


pca = PCA(n_components=17)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(df_new.iloc[:,:-1], df_new.iloc[:,-1], random_state=0)

X_train_2_pca = pca.fit_transform(X_train_2)
X_train_2_pca = pd.DataFrame(X_train_2_pca)

x_test_2_pca = pca.transform(X_test_2)

clf_17d = LogisticRegression().fit(X_train_2_pca, y_train_2)
acc_17d = clf_17d.score(x_test_2_pca, y_test_2)
print ('Logistic Regression Accuracy with PCA (17 components): {}'.format(acc_17d))


# PCA reduced the performance of our logistic regression model performance. Why is that? I did some research online and found some good explanation [here](https://stackoverflow.com/questions/36668768/why-did-pca-reduced-the-performance-of-logistic-regression)

# ## Study LDA impact on model accuracy

# While PCA ignores class label, LDA takes into consideration of class label to computes the directions (“linear discriminants”) that will represent the axes that that maximize the separation between multiple classes. More detailed explanation can be find [here](http://sebastianraschka.com/Articles/2014_python_lda.html#principal-component-analysis-vs-linear-discriminant-analysis).
# 
# Let's see how it is implemented:

# In[ ]:


lda = LDA(n_components=None)

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(df_new.iloc[:,:-1], df_new.iloc[:,-1], random_state=0)

X_lda = lda.fit(X_train_3, y_train_3)

lda_var_ratios = lda.explained_variance_ratio_

# get number of components needed to explain 95% variance
def select_n_components(var_ratio, goal_var: float) -> int:
    
    total_variance = 0.0
    n_components = 0
    
    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break

    return n_components

print('Number of components needed to explain 95% variance: {}'.format(select_n_components(lda_var_ratios, 0.95)))


# In[ ]:


lda_n = LDA(n_components=3)
X_train_3_lda = lda_n.fit_transform(X_train_3, y_train_3)
X_train_3_lda = pd.DataFrame(X_train_3_lda)

X_test_3_lda = lda_n.transform(X_test_3)

clf_3d = LogisticRegression().fit(X_train_3_lda, y_train_3)
acc_3d = clf_3d.score(X_test_3_lda, y_test_3)
print ('Logistic Regression Accuracy with LDA (3 components): {}'.format(acc_3d))


# ## Predict all positions (14 targets) with logistic regression

# Now let's try to predict all positions (14 targets, e.g. ST, LW, etc) with logistic regression. First we need to prepare to data:

# In[ ]:


df_new_normalized_all = df_new.copy()
mapping_all = {'ST': 0, 'RW': 1, 'LW': 2, 'RM': 3, 'CM': 4, 'LM': 5, 'CAM': 6, 'CF': 7, 'CDM': 8, 'CB': 9, 'LB': 10, 'RB': 11, 'RWB': 12, 'LWB': 13}

df_new_normalized_all = df_new_normalized_all.replace({'Preferred Positions': mapping_all})
df_new_normalized_all.iloc[::1000,]


# Fit the model:

# In[ ]:


X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(df_new_normalized_all.iloc[:,:-1], df_new_normalized_all.iloc[:,-1], random_state=0)

print('X train shape: {}'.format(X_train_all.shape))
print('X test shape: {}'.format(X_test_all.shape))
print('y train shape: {}'.format(y_train_all.shape))
print('y test shape: {}'.format(y_test_all.shape))


# In[ ]:


clf_d_all = DummyClassifier(strategy = 'most_frequent').fit(X_train_all, y_train_all)
acc_d_all = clf_d_all.score(X_test_all, y_test_all)
print ('Dummy Classifier (most frequent class): {}'.format(acc_d_all))

clf_all = LogisticRegression().fit(X_train_all, y_train_all)
acc_all = clf_all.score(X_test_all, y_test_all)
print ('Logistic Regression Accuracy: {}'.format(acc_all))


# ## Predict all positions (14 targets) with random forest

# The accuracy significantly drops when we tried to classify 14 targets instead of 2. Let's try out other model, Random forest:

# In[ ]:


clf_all_for = RandomForestClassifier(random_state=0).fit(X_train_all, y_train_all)
acc_all_for = clf_all_for.score(X_test_all, y_test_all)
print ('Random Forest Accuracy (Default parameters): {}'.format(acc_all_for))


# Accuracy isn't satisfying, let's try use gridsearchcv to find better parameters to the model:

# In[ ]:


parameters_f = [{'max_depth': range(2,10), 'n_estimators': range(2,8,2), 'max_features': range(10,20)}]
clf_all_for_g = GridSearchCV(RandomForestClassifier(random_state=0), parameters_f)
clf_all_for_g.fit(X_train_all, y_train_all)

print('Best score for train data:', clf_all_for_g.best_score_)
print('Best depth:',clf_all_for_g.best_estimator_.max_depth)
print('Best n trees:',clf_all_for_g.best_estimator_.n_estimators)
print('Best n features:',clf_all_for_g.best_estimator_.max_features)
print('Score for test data:',clf_all_for_g.score(X_test_all, y_test_all))



# ## Predict all positions (14 targets) with neural networks

# Let's look at neural networks and parameters:

# In[ ]:


clf_all_nn = MLPClassifier(random_state=0).fit(X_train_all, y_train_all)
acc_all_nn = clf_all_nn.score(X_test_all, y_test_all)
print ('Neural Networks Accuracy (Default parameters): {}'.format(acc_all_nn))

parameters_n = [{'alpha': [0.0001, 0.001, 0.01, 0.1], 'hidden_layer_sizes':[(10,),(20,),(100,)]}]
clf_all_nn_g = GridSearchCV(MLPClassifier(random_state=0), parameters_n)
clf_all_nn_g.fit(X_train_all, y_train_all)

print('Best score for train data:', clf_all_nn_g.best_score_)
print('Best alpha:',clf_all_nn_g.best_estimator_.alpha)
print('Best hidden_layer_sizes:',clf_all_nn_g.best_estimator_.hidden_layer_sizes)
print('Score for test data:',clf_all_nn_g.score(X_test_all, y_test_all))


# Accuracy improves but is not very high. This may be due to the fact that many players have mixed attributes, like ST / LW / RW may process very similar pattern in attributes. We might need more other signficant features in this case.

# Hope it is a good read, welcome any discussion :)
