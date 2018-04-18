
# coding: utf-8

# I am  going to apply 6 Supervised machine learning models on the given dataset.The strategy is to apply default model first with no tuning of the hyperparameter and then tuned them with different hyperparameter values and then I'll plot ROC curve to select the best machine learning model.The models used are as follows:
# **1) Principal Component Analysis
# **2)Logistic Regression**
#  **3)Gaussian Naive Bayes**
#   **4)Support Vector Machine**
#   **5)Random Forest Classifier** 
#   **6)Decision trees**
#    **7)Simple neural network**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ###  Importing all the libraries

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# ### Reading the file 

# In[ ]:


data = pd.read_csv("../input/mushrooms.csv")
data.head(6)


# ### Let us check if there is any null values

# In[ ]:


data.isnull().sum()


# In[ ]:


data['class'].unique()


# **Thus we have two claasification. Either the mushroom is poisonous or edible**

# In[ ]:


data.shape


# **Thus we have 22 features(1st one is label)  and 8124 instances.Now let us check which features constitutes maximum information.** 

# **We can see that the dataset has values in strings.We need to convert all the unique values to integers. Thus we perform label encoding on the data.**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])
 
data.head()


# ### Checking the encoded values

# In[ ]:


data['stalk-color-above-ring'].unique()


# In[ ]:


print(data.groupby('class').size())


# ### Plotting boxplot to see the distribution of the data

# In[ ]:


'''
# Create a figure instance
fig, axes = plt.subplots(nrows=2 ,ncols=2 ,figsize=(9, 9))

# Create an axes instance and the boxplot
bp1 = axes[0,0].boxplot(data['stalk-color-above-ring'],patch_artist=True)

bp2 = axes[0,1].boxplot(data['stalk-color-below-ring'],patch_artist=True)

bp3 = axes[1,0].boxplot(data['stalk-surface-below-ring'],patch_artist=True)

bp4 = axes[1,1].boxplot(data['stalk-surface-above-ring'],patch_artist=True)
'''
ax = sns.boxplot(x='class', y='stalk-color-above-ring', 
                data=data)
ax = sns.stripplot(x="class", y='stalk-color-above-ring',
                   data=data, jitter=True,
                   edgecolor="gray")
sns.plt.title("Class w.r.t stalkcolor above ring",fontsize=12)


# **Separating features and label**

# In[ ]:


X = data.iloc[:,1:23]  # all rows, all the features and no labels
y = data.iloc[:, 0]  # all rows, label only
X.head()
y.head()


# In[ ]:


X.describe()


# In[ ]:


y.head()


# In[ ]:


data.corr()


# # Standardising the data

# In[ ]:


# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)
X


# **Note**: We can avoid PCA here since the dataset is very small.

# # Principal Component Analysis

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)


# In[ ]:


covariance=pca.get_covariance()
#covariance


# In[ ]:


explained_variance=pca.explained_variance_
explained_variance


# In[ ]:


with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))
    
    plt.bar(range(22), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()


# **We can see that the last 4 components has less amount of variance  of the data.The 1st 17 components retains more than 90% of the data.**

# ### Let us take only first two principal components and visualise it using K-means clustering

# In[ ]:


N=data.values
pca = PCA(n_components=2)
x = pca.fit_transform(N)
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1])
plt.show()


# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=5)
X_clustered = kmeans.fit_predict(N)

LABEL_COLOR_MAP = {0 : 'g',
                   1 : 'y'
                  }

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
plt.figure(figsize = (5,5))
plt.scatter(x[:,0],x[:,1], c= label_color)
plt.show()


# ### Thus using K-means we are able segregate 2 classes well using the first two components with maximum variance.

# # Performing PCA by taking 17 components with maximum Variance

# In[ ]:


pca_modified=PCA(n_components=17)
pca_modified.fit_transform(X)


# ### Splitting the data into training and testing dataset

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)


# # Default Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

model_LR= LogisticRegression()


# In[ ]:


model_LR.fit(X_train,y_train)


# In[ ]:


y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
model_LR.score(X_test, y_pred)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# # Logistic Regression(Tuned model)

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

LR_model= LogisticRegression()

tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,
              'penalty':['l1','l2']
                   }


# **L1 and L2 are regularization parameters.They're used to avoid overfiting.Both L1 and L2 regularization prevents overfitting by shrinking (imposing a penalty) on the coefficients.**  
#     **L1 is the first moment norm |x1-x2| (|w| for regularization case) that is simply the absolute dıstance between two points where L2 is second moment norm corresponding to Eucledian Distance that is  |x1-x2|^2 (|w|^2 for regularization case).**     
#        **In simple words,L2 (Ridge) shrinks all the coefficient by the same proportions but eliminates none, while L1 (Lasso) can shrink some coefficients to zero, performing variable selection.
# If all the features are correlated with the label, ridge outperforms lasso, as the coefficients are never zero in ridge. If only a subset of features are correlated with the label, lasso outperforms ridge as in lasso model some coefficient can be shrunken to zero.**

# ### Taking a look at the correlation 

# In[ ]:


data.corr()


# **The grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the tuned_parameter.The GridSearchCV instance implements the usual estimator API: when “fitting” it on a dataset all the possible combinations of parameter values are evaluated and the best combination is retained.**

# In[ ]:


from sklearn.model_selection import GridSearchCV

LR= GridSearchCV(LR_model, tuned_parameters,cv=10)


# In[ ]:


LR.fit(X_train,y_train)


# In[ ]:


print(LR.best_params_)


# In[ ]:


y_prob = LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
LR.score(X_test, y_pred)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:



import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[ ]:


LR_ridge= LogisticRegression(penalty='l2')
LR_ridge.fit(X_train,y_train)


# In[ ]:


y_prob = LR_ridge.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
LR_ridge.score(X_test, y_pred)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:



from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:



import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# # Gaussian Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
model_naive = GaussianNB()
model_naive.fit(X_train, y_train)


# In[ ]:


y_prob = model_naive.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
model_naive.score(X_test, y_pred)


# In[ ]:


print("Number of mislabeled points from %d points : %d"
      % (X_test.shape[0],(y_test!= y_pred).sum()))


# In[ ]:


scores = cross_val_score(model_naive, X, y, cv=10, scoring='accuracy')
print(scores)


# In[ ]:


scores.mean()


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# # Support Vector Machine

# In[ ]:


from sklearn.svm import SVC
svm_model= SVC()


# The **gamma** parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’. The **gamma** parameters can be seen as the inverse of the radius of influence of samples selected by the model as support vectors.
# 
# The **C** parameter trades off misclassification of training examples against simplicity of the decision surface. A low **C** makes the decision surface smooth, while a high **C** aims at classifying all training examples correctly by giving the model freedom to select more samples as support vectors.

# # Support Vector Machine without polynomial kernel

# In[ ]:


tuned_parameters = {
 'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],
 'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],
 #'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']
    }


# The grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the tuned_parameter**.The GridSearchCV instance implements the usual estimator API: when “fitting” it on a dataset all the possible combinations of parameter values are evaluated and the best combination is retained.
# But it is proving computationally expensive here.So I am opting for RandomizedSearchCV.
# 
# RandomizedSearchCV implements a randomized search over parameters, where each setting is sampled from a distribution over possible parameter values. This has two main benefits over an exhaustive search:
# 1)A budget can be chosen independent of the number of parameters and possible values.
# 2)Adding parameters that do not influence the performance does not decrease efficiency.

# In[ ]:


from sklearn.grid_search import RandomizedSearchCV

model_svm = RandomizedSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy',n_iter=20)


# In[ ]:


model_svm.fit(X_train, y_train)
print(model_svm.best_score_)


# In[ ]:


print(model_svm.grid_scores_)


# In[ ]:


print(model_svm.best_params_)


# In[ ]:



y_pred= model_svm.predict(X_test)
print(metrics.accuracy_score(y_pred,y_test))


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# # Support Vector machine with polynomial Kernel

# In[ ]:


tuned_parameters = {
 'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],
 'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],
 'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']
    }


# In[ ]:


from sklearn.grid_search import RandomizedSearchCV

model_svm = RandomizedSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy',n_iter=20)


# In[ ]:


model_svm.fit(X_train, y_train)
print(model_svm.best_score_)


# In[ ]:


print(model_svm.grid_scores_)


# In[ ]:


print(model_svm.best_params_)


# In[ ]:


y_pred= model_svm.predict(X_test)
print(metrics.accuracy_score(y_pred,y_test))


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# ### Trying default model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_RR=RandomForestClassifier()

#tuned_parameters = {'min_samples_leaf': range(5,10,5), 'n_estimators' : range(50,200,50),
                    #'max_depth': range(5,15,5), 'max_features':range(5,20,5)
                    #}
               


# In[ ]:


model_RR.fit(X_train,y_train)


# In[ ]:


y_prob = model_RR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
model_RR.score(X_test, y_pred)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# ### Thus default Random forest model is giving us best accuracy.

# ### Let us tuned the parameters of Random Forest just for the purpose of knowledge

# **There are 3 features which can be tuned to improve the performance of Random Forest**    
# 
# **1) max_features 2) n_estimators  3) min_sample_leaf**

#  **A)max_features**: These are the maximum number of features Random Forest is allowed to try in individual tree.
# **1)Auto** : This will simply take all the features which make sense in every tree.Here we simply do not put any restrictions on the individual tree.
# **2)sqrt** : This option will take square root of the total number of features in individual run. For instance, if the total number of variables are 100, we can only take 10 of them in individual tree.
# **3)log2**:It  is another option which takes log to the base 2 of the features input.
# 
# **Increasing max_features generally improves the performance of the model as at each node now we have a higher number of options to be considered.But, for sure, you decrease the speed of algorithm by increasing the max_features. Hence, you need to strike the right balance and choose the optimal max_features.**

# **B) n_estimators** : This is the number of trees you want to build before taking the maximum voting or averages of predictions. Higher number of trees give you better performance but makes your code slower. You should choose as high value as your processor can handle because this makes your predictions stronger and more stable.

# **C)min_sample_leaf**:  Leaf is the end node of a decision tree. A smaller leaf makes the model more prone to capturing noise in train data. Hence it is important to try different values to get good estimate.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_RR=RandomForestClassifier()

tuned_parameters = {'min_samples_leaf': range(10,100,10), 'n_estimators' : range(10,100,10),
                    'max_features':['auto','sqrt','log2']
                    }
    


# ### n_jobs
# **This parameter tells the engine how many processors is it allowed to use. A value of “-1” means there is no restriction whereas a value of “1” means it can only use one processor.**

# In[ ]:


from sklearn.grid_search import RandomizedSearchCV

RR_model= RandomizedSearchCV(model_RR, tuned_parameters,cv=10,scoring='accuracy',n_iter=20,n_jobs= -1)


# In[ ]:


RR_model.fit(X_train,y_train)


# In[ ]:


print(RR_model.grid_scores_)


# In[ ]:


print(RR_model.best_score_)


# In[ ]:


print(RR_model.best_params_)


# In[ ]:


y_prob = RR_model.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
RR_model.score(X_test, y_pred)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:



from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# ### Default Decision Tree model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier()


# In[ ]:


model_tree.fit(X_train, y_train)


# In[ ]:


y_prob = model_tree.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
model_tree.score(X_test, y_pred)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# ### Thus default decision tree model is giving us best accuracy score

# ### Let us tune the hyperparameters of the Decision tree model

# **1)Criterion:** Decision trees use multiple algorithms to decide to split a node in two or more sub-nodes.Decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes. The details of Gini and entropy needs detail explanation.
# 
# 2)**max_depth(Maximum depth of tree (vertical depth)):**
# Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
# 
# **max_features** and **min_samples_leaf** is same as Random Forest classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

model_DD = DecisionTreeClassifier()


tuned_parameters= {'criterion': ['gini','entropy'], 'max_features': ["auto","sqrt","log2"],
                   'min_samples_leaf': range(1,100,1) , 'max_depth': range(1,50,1)
                  }
           


# In[ ]:


from sklearn.grid_search import RandomizedSearchCV
DD_model= RandomizedSearchCV(model_DD, tuned_parameters,cv=10,scoring='accuracy',n_iter=20,n_jobs= -1,random_state=5)


# In[ ]:


DD_model.fit(X_train, y_train)


# In[ ]:


print(DD_model.grid_scores_)


# In[ ]:


print(DD_model.best_score_)


# In[ ]:


print(DD_model.best_params_)


# In[ ]:


y_prob = DD_model.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
DD_model.score(X_test, y_pred)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# ## Neural Network

# ### Applying default Neural Network model

# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


mlp = MLPClassifier()
mlp.fit(X_train,y_train)


# In[ ]:


y_prob = mlp.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
mlp.score(X_test, y_pred)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix


# In[ ]:


auc_roc=metrics.classification_report(y_test,y_pred)
auc_roc


# In[ ]:


auc_roc=metrics.roc_auc_score(y_test,y_pred)
auc_roc


# In[ ]:


from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# ### Tuning the hyperparameters of the neural network

# It is turning out to be computationally expensive for me with tuned model. Hence I am not running this. Also any suggestion to improvise it is welcome. :)

# **1) hidden_layer_sizes**  : Number of hidden layers in the network.(default is 100).Large number may overfit the data.
# 
# **2)activation**: Activation function for the hidden layer.
# A)‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
# B)‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
# C)‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
# 
# **3)alpha:** L2 penalty (regularization term) parameter.(default 0.0001)
# 
# **4)max_iter:** Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations.(default 200)

# In[ ]:


'''
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()

tuned_parameters={'hidden_layer_sizes': range(1,200,10) , 'activation': ['tanh','logistic','relu'],
                  'alpha':[0.0001,0.001,0.01,0.1,1,10], 'max_iter': range(50,200,50)
    
}
'''


# In[ ]:


#from sklearn.grid_search import RandomizedSearchCV
#model_mlp= RandomizedSearchCV(mlp_model, tuned_parameters,cv=10,scoring='accuracy',n_iter=5,n_jobs= -1,random_state=5)


# In[ ]:


#model_mlp.fit(X_train, y_train)


# In[ ]:


#print(model_mlp.grid_scores_)


# In[ ]:


#print(model_mlp.best_score_)


# In[ ]:


#print(model_mlp.best_params_)


# In[ ]:


'''
y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
model_LR.score(X_test, y_pred)
'''


# In[ ]:


#confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
#confusion_matrix


# In[ ]:


#auc_roc=metrics.classification_report(y_test,y_pred)
#auc_roc


# In[ ]:


#auc_roc=metrics.roc_auc_score(y_test,y_pred)
#auc_roc


# In[ ]:


'''
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
'''


# In[ ]:


'''
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
'''

