
# coding: utf-8

# ## This Notebook is gonna show you, how do linear, non-linear, and tree-based machine learning classifier divide the Boundary.
# ---
# `Updated: Boosting, 11, Sept, 2017`
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.get_backend()
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# - Load the dataset

# In[ ]:


iris = pd.read_csv('../input/Iris.csv')
iris.head()


# ## Label Encoder

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
iris.Species = le.fit_transform(iris.Species)


# ### We may check the mapped values by using
# `transform`
# 
# - Iris-setosa' correspond to __0__
# - Iris-versicolor' correspond to __1__
# - Iris-virginica'  correspond to __2__

# In[ ]:


le.transform(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])


# - Split the data into train/test

# In[ ]:


feature_names_iris = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X_iris = iris[feature_names_iris]
y_iris = iris['Species']
target_names_iris = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
X_iris_2d = iris[['PetalLengthCm','PetalWidthCm']]
y_iris_2d = iris['Species']


# In[ ]:


X_iris_2d.head(2)


# ## Helper function 

# In[ ]:


def plot_class_regions_for_classifier_subplot(clf, X, y, X_test, y_test, title, subplot, 
                                              target_names = None, plot_decision_regions = True):

    numClasses = np.amax(y) + 1
    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    cmap_light = ListedColormap(color_list_light[0:numClasses])
    cmap_bold  = ListedColormap(color_list_bold[0:numClasses])

    h = 0.03
    k = 0.5
    x_plot_adjust = 0.1
    y_plot_adjust = 0.1
    plot_symbol_size = 50

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    x2, y2 = np.meshgrid(np.arange(x_min-k, x_max+k, h), np.arange(y_min-k, y_max+k, h))

    P = clf.predict(np.c_[x2.ravel(), y2.ravel()])
    P = P.reshape(x2.shape)

    if plot_decision_regions:
        subplot.contourf(x2, y2, P, cmap=cmap_light, alpha = 0.8)

    subplot.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
    subplot.set_xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
    subplot.set_ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)

    if (X_test is not None):
        subplot.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, s=plot_symbol_size, 
                        marker='^', edgecolor = 'black')
        train_score = clf.score(X, y)
        test_score  = clf.score(X_test, y_test)
        title = title + "\nTrain score = {:.2f}, Test score = {:.2f}".format(train_score, test_score)

    subplot.set_title(title)

    if (target_names is not None):
        legend_handles = []
        for i in range(0, len(target_names)):
            patch = mpatches.Patch(color=color_list_bold[i], label=target_names[i])
            legend_handles.append(patch)
        subplot.legend(loc=0, handles=legend_handles)

def plot_class_regions_for_classifier(clf, X, y, X_test=None, y_test=None, title=None, target_names = None, plot_decision_regions = True):

    numClasses = np.amax(y) + 1
    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    cmap_light = ListedColormap(color_list_light[0:numClasses])
    cmap_bold  = ListedColormap(color_list_bold[0:numClasses])

    h = 0.03
    k = 0.5
    x_plot_adjust = 0.1
    y_plot_adjust = 0.1
    plot_symbol_size = 50

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    x2, y2 = np.meshgrid(np.arange(x_min-k, x_max+k, h), np.arange(y_min-k, y_max+k, h))

    P = clf.predict(np.c_[x2.ravel(), y2.ravel()])
    P = P.reshape(x2.shape)
    plt.figure()
    if plot_decision_regions:
        plt.contourf(x2, y2, P, cmap=cmap_light, alpha = 0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
    plt.xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
    plt.ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)

    if (X_test is not None):
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, s=plot_symbol_size, marker='^', edgecolor = 'black')
        train_score = clf.score(X, y)
        test_score  = clf.score(X_test, y_test)
        title = title + "\nTrain score = {:.2f}, Test score = {:.2f}".format(train_score, test_score)

    if (target_names is not None):
        legend_handles = []
        for i in range(0, len(target_names)):
            patch = mpatches.Patch(color=color_list_bold[i], label=target_names[i])
            legend_handles.append(patch)
        plt.legend(loc=0, handles=legend_handles)

    if (title is not None):
        plt.title(title)
    plt.show()

def plot_class_regions_for_classifier_subplot(clf, X, y, X_test, y_test, title, subplot, target_names = None, plot_decision_regions = True):

    numClasses = np.amax(y) + 1
    color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    cmap_light = ListedColormap(color_list_light[0:numClasses])
    cmap_bold  = ListedColormap(color_list_bold[0:numClasses])

    h = 0.03
    k = 0.5
    x_plot_adjust = 0.1
    y_plot_adjust = 0.1
    plot_symbol_size = 50

    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    x2, y2 = np.meshgrid(np.arange(x_min-k, x_max+k, h), np.arange(y_min-k, y_max+k, h))

    P = clf.predict(np.c_[x2.ravel(), y2.ravel()])
    P = P.reshape(x2.shape)

    if plot_decision_regions:
        subplot.contourf(x2, y2, P, cmap=cmap_light, alpha = 0.8)

    subplot.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=plot_symbol_size, edgecolor = 'black')
    subplot.set_xlim(x_min - x_plot_adjust, x_max + x_plot_adjust)
    subplot.set_ylim(y_min - y_plot_adjust, y_max + y_plot_adjust)

    if (X_test is not None):
        subplot.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, s=plot_symbol_size, marker='^', edgecolor = 'black')
        train_score = clf.score(X, y)
        test_score  = clf.score(X_test, y_test)
        title = title + "\nTrain score = {:.2f}, Test score = {:.2f}".format(train_score, test_score)

    subplot.set_title(title)

    if (target_names is not None):
        legend_handles = []
        for i in range(0, len(target_names)):
            patch = mpatches.Patch(color=color_list_bold[i], label=target_names[i])
            legend_handles.append(patch)
        subplot.legend(loc=0, handles=legend_handles)


def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap, linewidth=10)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


# # Voting Classifier

# In[ ]:


from sklearn import datasets
Iris = datasets.load_iris()
X = Iris.data[:, [2, 3]]
y = Iris.target
color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA']#, '#AAAAFF']
color_list_bold = ['#EEEE00', '#000000', '#00CC00']#, '#0000CC']
custom_cmap2 = ListedColormap(color_list_light)
custom_cmap1 = ListedColormap(color_list_bold)
# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=5)
clf2 = KNeighborsClassifier(n_neighbors=6)
clf3 = SVC(kernel='rbf', probability=True,gamma=5,C=1)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['Decision Tree (depth=5)', 'KNN (k=6)',
                         'Kernel SVM', 'Soft Voting']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.8,cmap=custom_cmap2)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,cmap=custom_cmap1,
                                  s=20, edgecolor='black')
    axarr[idx[0], idx[1]].set_title(tt)
plt.show()


# ## Multi-class classification with Non-Linear Classifier
# 

# ### You will see how does machine draw the _boundary_ when change the hyper parameters or kernel
# 
# 
# The Support Vector Machine's kernel function can be any of the following:
# 
# - polynomial: d is specified by keyword degree, r by coef0.![](http://scikit-learn.org/stable/_images/math/caed8545ad94ab355e204242314fb76bb96b2b09.png) 
# 
# - rbf: is specified by keyword gamma, must be greater than 0.![](http://scikit-learn.org/stable/_images/math/d571609cf042d44f541e8c11efbc305354206096.png) 
#     - Bigger gamma smaller distance, vice versa.
# 
# ## RBF kernel: 
# - using both C and gamma parameter 
# 
# 
# 

# In[ ]:


y_iris_versicolor = y_iris_2d# == 1

X_train, X_test, y_train, y_test = (
train_test_split(X_iris_2d.as_matrix(),
                y_iris_versicolor.as_matrix(),
                random_state=0))
fig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=200)

for this_gamma, this_axis in zip([0.01, 1, 5], subaxes):
    
    for this_C, subplot in zip([0.1, 1, 15, 250], this_axis):
        title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
        clf = SVC(kernel = 'rbf', gamma = this_gamma,
                 C = this_C,random_state=0).fit(X_train, y_train)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                                 X_test, y_test, title,
                                                 subplot)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# ## Poly kernel: 
# - using both **C** and **gamma** parameter 

# In[ ]:


y_iris_versicolor = y_iris_2d# == 1

X_train, X_test, y_train, y_test = (
train_test_split(X_iris_2d.as_matrix(),
                y_iris_versicolor.as_matrix(),
                random_state=0))
fig, subaxes = plt.subplots(3, 4, figsize=(15, 10), dpi=200)

for this_gamma, this_axis in zip([0.01, 1, 5], subaxes):
    
    for this_C, subplot in zip([0.1, 1, 15, 250], this_axis):
        title = 'gamma = {:.2f}, C = {:.2f}'.format(this_gamma, this_C)
        clf = SVC(kernel = 'poly', gamma = this_gamma,
                 C = this_C,random_state=0).fit(X_train, y_train)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                                 X_test, y_test, title,
                                                 subplot)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# ## Multi-class classification with Linear Classifier

# ### Linear SVC draw mutiple lines to divide the boundary.

# In[ ]:


from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(X_iris_2d, y_iris_2d, random_state = 0)

clf = LinearSVC(C=5, random_state = 987).fit(X_train, y_train)
print('Coefficients:\n', clf.coef_)
print('Intercepts:\n', clf.intercept_)


# In[ ]:


plt.figure(figsize=(6,6))
colors = ['r', 'g', 'b']
cmap_iris = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.scatter(X_iris_2d[['PetalLengthCm']], X_iris_2d[['PetalWidthCm']],
           c=y_iris_2d, cmap=cmap_iris, edgecolor = 'black', alpha=.7)

x_0_range = np.linspace(-10, 15)

for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b']):

    plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)
    
plt.legend(target_names_iris,loc='best')
plt.xlabel('Length')
plt.ylabel('Width')
plt.xlim(-2, 8)
plt.ylim(-2, 4)
plt.show()


# ### Tree based classifier.
# DecisionTreeClassifier with different depth.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = (
train_test_split(X_iris_2d.as_matrix(),
                y_iris_versicolor.as_matrix(),
                random_state=0))

fig, subaxes = plt.subplots(3, 1, figsize=(4, 11), dpi=100)
for this_depth, subplot in zip([2,4,6], subaxes):
    clf = DecisionTreeClassifier(max_depth=this_depth,).fit(X_train, y_train)
    title = 'Decision Tree Classifier: \nDepth = {:d}'.format(this_depth)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                             None, None, title, subplot)
    plt.tight_layout()


# ###  Logistic regression regularization: using C parameter and penalty.

# In[ ]:


fig, subaxes = plt.subplots(2, 4, figsize=(15, 8), dpi=200)
for this_penalty, this_axis in zip(['l1','l2'], subaxes): 
    for this_C, subplot in zip([0.1, 1, 10, 200], this_axis):
        title = 'penalty = {:s}, C = {:.2f}'.format(this_penalty, this_C)
        clf = LogisticRegression(C=this_C,penalty=this_penalty,random_state=0).fit(X_train, y_train)
        plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                                 X_test, y_test, title,
                                                 subplot)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# ### Boosting
# **Boosting: **The boosting is to create complementary base-leanrers by training the new learner using the examples that the previous leaners do not agree. A common implementation is AdaBoost (Adaptive Boosting). 

# In[ ]:


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    #colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    colors = ('#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    colors2 = ('#EEEE00', '#000000', '#00CC00', '#0000CC', 'cyan')
    cmap2 = ListedColormap(colors2[:len(np.unique(y))])
    #color_list_light = ['#FFFFAA', '#EFEFEF', '#AAFFAA', '#AAAAFF']
    #color_list_bold = ['#EEEE00', '#000000', '#00CC00', '#0000CC']
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap2(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

for depth in [1,2,3,4,5]:
    tree = DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train)
    ada = AdaBoostClassifier(base_estimator=tree, n_estimators=100).fit(X_train, y_train)

    fig, subaxes = plt.subplots(figsize=(6, 6))
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined, y=y_combined, 
                      classifier=tree,
                      test_idx=range(len(y_train), 
                                     len(y_train) + len(y_test)))
    plt.title('Decision tree \n max_depth = {:d}'.format(depth))
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

    fig, subaxes = plt.subplots(figsize=(6, 6))
    plot_decision_regions(X=X_combined, y=y_combined, 
                      classifier=ada,
                      test_idx=range(len(y_train), 
                                     len(y_train) + len(y_test)))
    plt.title('AdaBoost base=Decision tree\n max_depth = {:d}'.format(depth))
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

