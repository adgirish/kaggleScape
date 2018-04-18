
# coding: utf-8

# ### Here is a simple MachineLearning notebook for beginners. The notebook includes:
# 1. #### Feature Engineering
# 2. #### Visualization
# 3. #### Voting Classifier

# In[ ]:


# Import the required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


# Below we will import the data for this project. There are 2 data files:
# - train.csv -> Contains the training data.
# - test.csv -> Contains the testing data for which we have to predict the category.

# In[ ]:


train_data_orig = pd.read_csv('../input/train.csv')
test_data_orig = pd.read_csv('../input/test.csv')


# We will print the shape of train and test dataframes to get a view of the number of columns and rows.

# In[ ]:


print("Shape of Training Data")
print(train_data_orig.shape)
print("\n")

print("Shape of Testing Data")
print(test_data_orig.shape)


# Below we will print the columns present in Training and Testing data.

# In[ ]:


print("Columns in Training Data")
print(train_data_orig.columns)
print("\n")

print("Columns in Testing Data")
print(test_data_orig.columns)


# From the above details we could clearly see that based on the independet features 'bone_length', 'rotting_flesh', 'hair_length', 'has_soul' and 'color' we have to predict the dependent variable - 'type'.

# In[ ]:


train_data_orig.info()


# In[ ]:


train_data_orig.head()


# Drop the 'id' column from train and test data as it is not required.

# In[ ]:


train_data = train_data_orig.drop(['id'], axis = 1)
test_data = test_data_orig.drop(['id'], axis = 1)


# From the above information we colud see that 'bone_length', 'rotting_flesh', 'hair_length' and 'has_soul' are float values. 'color' and 'type' have categorical values. Below, we will try to get more about each of the columns.

# In[ ]:


train_data.describe()


# In[ ]:


test_data.describe()


# In[ ]:


print(np.sort(train_data['color'].unique()))
print(np.sort(test_data['color'].unique()))


# In[ ]:


print(np.sort(train_data['type'].unique()))


# The data looks good without any missing values. As we have got some details about our data we will do a bit more of analysis using some visualizations.

# In[ ]:


# Use LabelEncoder for the 'color' feature
color_le = preprocessing.LabelEncoder()
color_le.fit(train_data['color'])
train_data['color_int'] = color_le.transform(train_data['color'])

_ = sns.pairplot(train_data.drop('color', axis = 1), hue = 'type', palette = 'muted', diag_kind='kde')

train_data.drop('color_int', axis = 1, inplace = True)


# Show the correlation between each of the features via a heatmap.

# In[ ]:


_ = sns.heatmap(train_data.corr(), annot = True, fmt = ".2f", cmap = 'YlGnBu')


# In[ ]:


g = sns.FacetGrid(pd.melt(train_data, id_vars='type', value_vars = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']), col = 'type')
g = g.map(sns.boxplot, 'value', 'variable', palette = 'muted')


# We will create a DecisionTreeClassifier which will serve as a base classifier for comapring the accuracy with our future models.

# In[ ]:


df = pd.get_dummies(train_data.drop('type', axis = 1))
X_train, X_test, y_train, y_test = train_test_split(df, train_data['type'], test_size = 0.25, random_state = 0)

dt_clf = DecisionTreeClassifier(random_state = 0)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
print("\nAccuracy Score is: " + str(metrics.accuracy_score(y_test, y_pred)))


# In[ ]:


accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)


# In[ ]:


X_train = pd.get_dummies(train_data.drop('type', axis = 1))
y_train = train_data['type']
X_test = pd.get_dummies(test_data)


# In[ ]:


params = {'n_estimators':[10, 20, 50, 100], 'criterion':['gini', 'entropy'], 'max_depth':[None, 5, 10, 25, 50]}
rf = RandomForestClassifier(random_state = 0)
clf = GridSearchCV(rf, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)
clf.fit(X_train, y_train)
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))

rf_best = RandomForestClassifier(n_estimators = 10, random_state = 0)


# In[ ]:


params = {'n_estimators':[10, 25, 50, 100], 'max_samples':[1, 3, 5, 10]}
bag = BaggingClassifier(random_state = 0)
clf = GridSearchCV(bag, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)
clf.fit(X_train, y_train)
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))

bag_best = BaggingClassifier(max_samples = 5, n_estimators = 25, random_state = 0)


# In[ ]:


params = {'learning_rate':[0.05, 0.1, 0.5], 'n_estimators':[100, 200, 500], 'max_depth':[2, 3, 5, 10]}
gbc = GradientBoostingClassifier(random_state = 0)
clf = GridSearchCV(gbc, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)
clf.fit(X_train, y_train)
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))

gbc_best = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 5, n_estimators = 100, random_state = 0)


# In[ ]:


params = {'n_neighbors':[3, 5, 10, 20], 'leaf_size':[20, 30, 50], 'p':[1, 2, 5], 'weights':['uniform', 'distance']}
knc = KNeighborsClassifier()
clf = GridSearchCV(knc, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)
clf.fit(X_train, y_train)
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))

knc_best = KNeighborsClassifier(n_neighbors = 10)


# In[ ]:


params = {'penalty':['l1', 'l2'], 'C':[1, 2, 3, 5, 10]}
lr = LogisticRegression(random_state = 0)
clf = GridSearchCV(lr, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)
clf.fit(X_train, y_train)
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))

lr_best = LogisticRegression(penalty = 'l1', C = 1, random_state = 0)


# In[ ]:


params = {'kernel':['linear', 'rbf'], 'C':[1, 3, 5, 10], 'degree':[3, 5, 10]}
svc = SVC(probability = True, random_state = 0)
clf = GridSearchCV(svc, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)
clf.fit(X_train, y_train)
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))

svc_best = SVC(C = 10, degree = 3, kernel = 'linear', probability = True, random_state = 0)


# In[ ]:


voting_clf = VotingClassifier(estimators=[('rf', rf_best), ('bag', bag_best), ('gbc', gbc_best), ('lr', lr_best), ('svc', svc_best)]
                              , voting='hard')
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print("\nAccuracy Score for VotingClassifier is: " + str(voting_clf.score(X_train, y_train)))


# In[ ]:


submission = pd.DataFrame({'id':test_data_orig['id'], 'type':y_pred})
submission.to_csv('../working/submission.csv', index=False)


# The above code gets a LB score of 0.73345.
