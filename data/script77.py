
# coding: utf-8

# Hugues Fontenelle
# 7 October 2016
# 
# # Pima Indians Diabetes Database
# ## Predict the onset of diabetes based on diagnostic measures
# 
# Hi folks. I'm new to this, so let me try out what I've learned so far. Your comments are welcome!

# First, let's load the data, and split it in four. It is the fold used the authors of the original paper.

# In[ ]:


import numpy as np

f = open("../input/diabetes.csv")
f.readline()  # skip the header
data = np.loadtxt(f, delimiter = ',')
X = data[:, :-1]
y = data[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Let's try out a bunch of classifiers, all with default parameters.

# In[ ]:


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"
        ]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]


# Now run all the classifiers, using 5-fold cross validation.

# In[ ]:


from sklearn.model_selection import cross_val_score

# iterate over classifiers
results = {}
for name, clf in zip(names, classifiers):
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    results[name] = scores


# Here are the results:

# In[ ]:


for name, scores in results.items():
    print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100*scores.mean(), 100*scores.std() * 2))


# Seems like a Linear SVM performs best.
# Let's try some parameter optimization.

# In[ ]:


from sklearn.grid_search import GridSearchCV

clf = SVC(kernel="linear")

# prepare a range of values to test
param_grid = [
  {'C': [.01, .1, 1, 10], 'kernel': ['linear']},
 ]

grid = GridSearchCV(estimator=clf, param_grid=param_grid)
grid.fit(X_train, y_train)
print(grid)


# In[ ]:


# summarize the results of the grid search
print("Best score: %0.2f%%" % (100*grid.best_score_))
print("Best estimator for parameter C: %f" % (grid.best_estimator_.C))


# Finaly, train the Linear SVM (with param `C=0.1`) on the whole train set, and evaluate on the test set

# In[ ]:


clf = SVC(kernel="linear", C=0.1)
clf.fit(X_train, y_train)
y_eval = clf.predict(X_test)


# In[ ]:


acc = sum(y_eval == y_test) / float(len(y_test))
print("Accuracy: %.2f%%" % (100*acc))


# We did it :-)

# **edit**
# 
# I was _probably_ a bit lucky for this particular fold (`random_state=0`). Why would the accuracy on the test be higher than on the optimized trained set? Let's re-run a 5-fold cv on the whole data:

# In[ ]:


clf = SVC(kernel="linear", C=0.1)
scores_final = cross_val_score(clf, X, y, cv=5)


# In[ ]:


scores_final.mean(), scores_final.std()
print("Final model | Accuracy: %0.2f%% (+/- %0.2f%%)" % (100*scores_final.mean(), 100*scores_final.std() * 2))


# ..which is more realistic!
# 
# I am wondering, at which stage do I then use this test set?
