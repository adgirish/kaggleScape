
# coding: utf-8

# # Introduction
# --- 
# 
# I have a background in bioinformatics and this dataset was one of my favorites in grad school. For one of my projects, I analyzed the data in R but I have since adopted Python as my language of chocie. The aim of this kernel was to refresh myself and implement machine learning algorithms in Python. I accomplished what I wanted but I keep coming back to try new things.  The current state of this kernel is a comparison of GridSearchCV and RandomizedSearchCV for tuning hyperparameters with 10-fold cross-validation and eventually I would like to implement BayesianSearch for parameters.
# 
# A heafty chunk of this kernel is just processing the data, so feel free to fork the kernel and use that part. The last half of the kernel is the comparison of the two parameter search methods. All in all, I found that both of these methods led to overfitting of the training data -- none of the models generalized to test data at all. That's the gist of what I learned but there is a little more discussion about it at the end of the kernel.
# 
# Enjoy!
# <br><br>
# 
# # About the original study
# I need to re-read the study, but from what I recall, these gene expression values come from cancer  patients with either acute lymphocytic leukemia (ALL) or acute myeloid leukemia (AML). In the original study, this datset was used to classify which type of cancer each patient had based on measurements of their gene expressions. The study was published in 1999 and was the first(?) to show that cancer types can be determined based on gene expressions alone. 
# 
# # About the data
# - Each row represents a different gene
# - Columns 1 and 2 are descriptions about that gene 
# - Each numbered column is a patient
# - Each patient has 7129 gene expression values - i.e each patient has one value for each gene 
# - The **training** data contain gene expression values for patients 1 through 38
# - The **test** data contain gene expression values for patients 39 through 72
# <br><br>
# 
# 
# # To do list:
# ---
# ### General cleanup
# - Write more functions for redundant testing (maybe, long term wish list)
# 
# ### Dimentionality reduction
# - Kanavanand has a great PCA analysis [here](https://www.kaggle.com/kanav0183/pca-analysis-for-geneclassification)
# 
# ### Tuning Hyperparamters:
# - Compare GridSearchCV to Bayesian Optimzation (I heard it's more efficient)
#     
# <br><br>

# # Data processing steps
# 
# 1. Remove columns that contain "Call" data, I'm not sure what they are but they doesn't seem useful
# 2. Transpose the dataframe so that each row is a patient and each column is a gene
# 3. Remove gene description and set the gene accession numbers as the column headers
# 4. Merge the data (expression values) with the class labels (patient numbers)

# In[ ]:


import itertools
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import scipy


# In[ ]:


testfile='../input/data_set_ALL_AML_independent.csv'
trainfile='../input/data_set_ALL_AML_train.csv'
patient_cancer='../input/actual.csv'

train = pd.read_csv(trainfile)
test = pd.read_csv(testfile)
patient_cancer = pd.read_csv(patient_cancer)


# In[ ]:


train.head()


# In[ ]:


# Remove "call" columns from training a test dataframes
train_keepers = [col for col in train.columns if "call" not in col]
test_keepers = [col for col in test.columns if "call" not in col]

train = train[train_keepers]
test = test[test_keepers]


# In[ ]:


train.head()


# In[ ]:


# Transpose the columns and rows so that genes become features and rows become observations
train = train.T
test = test.T
train.head()


# In[ ]:


# Clean up the column names for training data
train.columns = train.iloc[1]
train = train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

# Clean up the column names for training data
test.columns = test.iloc[1]
test = test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

train.head()


# ### Combine the data (gene expression) with class labels (patient numbers)

# In[ ]:


# Reset the index. The indexes of two dataframes need to be the same before you combine them
train = train.reset_index(drop=True)

# Subset the first 38 patient's cancer types
pc_train = patient_cancer[patient_cancer.patient <= 38].reset_index(drop=True)

# Combine dataframes for first 38 patients: Patient number + cancer type + gene expression values
train = pd.concat([pc_train,train], axis=1)


# Handle the test data for patients 38 through 72
# Clean up the index
test = test.reset_index(drop=True)

# Subset the last patient's cancer types to test
pc_test = patient_cancer[patient_cancer.patient > 38].reset_index(drop=True)

# Combine dataframes for last patients: Patient number + cancer type + gene expression values
test = pd.concat([pc_test,test], axis=1)


# # EDA
# ---
# 
# There's a bunch of data, so to speed things up I'm only using a small sample of the training data for the EDA.
# 

# In[ ]:


sample = train.iloc[:,2:].sample(n=100, axis=1)
sample["cancer"] = train.cancer
sample.describe().round()


# # To standardize or not to standardize
# ---
# 
# This is for a visual reference on how data changes after scaling. And for the record, I use the words standardize and scale interchangably. I think it's technically "standardizing", but scikit-learn calls it scaling. 
# 
# Standardize  = For each value, subtract the mean and scale to unit variance
# 
# 
# 

# In[ ]:


from sklearn import preprocessing


# ### Distribution of the random sample before standardizing
# ---

# In[ ]:


sample = sample.drop("cancer", axis=1)
sample.plot(kind="hist", legend=None, bins=20, color='k')
sample.plot(kind="kde", legend=None);


# Depending on the random sample and the histogram-bin-kung-fu, the data usually has a long skinny tail to the right. This KDE plot shows the distribution of indivudal features, but it's not very helpful here. This will change after standardizing the data. 

# ### Distribution of the random sample after standardizing
# ---
# 

# In[ ]:


sample_scaled = pd.DataFrame(preprocessing.scale(sample))
sample_scaled.plot(kind="hist", normed=True, legend=None, bins=10, color='k')
sample_scaled.plot(kind="kde", legend=None);


# There's quite a difference after standardizing the features. The KDE plot is much more useful in showing the individual distributions too. This is the result of subtracting the mean. Subtracting the mean from each feature centers them on zero. Neat!

# # Process the full set
# ---
# 
# 
# 

# In[ ]:


# StandardScaler to remove mean and scale to unit variance
from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler().fit(train.iloc[:,2:])
scaled_train = scaler.transform(train.iloc[:,2:])
scaled_test = scaler.transform(test.iloc[:,2:])

x_train = train.iloc[:,2:]
y_train = train.iloc[:,1]
x_test = test.iloc[:,2:]
y_test = test.iloc[:,1]


# 
# # Classifiers
# ---

# In[ ]:


# Grid Search for tuning parameters
from sklearn.model_selection import GridSearchCV
# RandomizedSearch for tuning (possibly faster than GridSearch)
from sklearn.model_selection import RandomizedSearchCV
# Bayessian optimization supposedly faster than GridSearch
from bayes_opt import BayesianOptimization

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# # Helper functions

# In[ ]:


# CHERCHEZ FOR PARAMETERS
def cherchez(estimator, param_grid, search):
    """
    This is a helper function for tuning hyperparameters using teh two search methods.
    Methods must be GridSearchCV or RandomizedSearchCV.
    Inputs:
        estimator: Logistic regression, SVM, KNN, etc
        param_grid: Range of parameters to search
        search: Grid search or Randomized search
    Output:
        Returns the estimator instance, clf
    
    """   
    try:
        if search == "grid":
            clf = GridSearchCV(
                estimator=estimator, 
                param_grid=param_grid, 
                scoring=None,
                n_jobs=-1, 
                cv=10, 
                verbose=0,
                return_train_score=True
            )
        elif search == "random":           
            clf = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=10,
                n_jobs=-1,
                cv=10,
                verbose=0,
                random_state=1,
                return_train_score=True
            )
    except:
        print('Search argument has to be "grid" or "random"')
        sys.exit(0)
        
    # Fit the model
    clf.fit(X=scaled_train, y=y_train)
    
    return clf   


# In[ ]:


# Function for plotting the confusion matrices
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Plots the confusion matrix. Modified verison from 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    Inputs: 
        cm: confusion matrix
        title: Title of plot
    """
    classes=["AML", "ALL"]    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.bone)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    thresh = cm.mean()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]), 
                 horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black")    


# # Models being tested
# 1. Logisitc Regresison
#   - Using Grid search and Randomized search for tuning hyperparameters
# 2. C-Support Vector Classification (SVM)
#   - Using Grid search and Randomized search for tuning hyperparameters
# 3. K-Nearest Neighbors Classifier
#   - Using Grid search and Randomized search for tuning hyperparameters
# 4. Decision Tree Classifier
#   - Using only Grid search

# In[ ]:


# Logistic Regression
# Paramaters
logreg_params = {} 
logreg_params["C"] =  [0.01, 0.1, 10, 100]
logreg_params["fit_intercept"] =  [True, False]
logreg_params["warm_start"] = [True,False]
logreg_params["random_state"] = [1]

lr_dist = {}
lr_dist["C"] = scipy.stats.expon(scale=.01)
lr_dist["fit_intercept"] =  [True, False]
lr_dist["warm_start"] = [True,False]
lr_dist["random_state"] = [1]

logregression_grid = cherchez(LogisticRegression(), logreg_params, search="grid")
acc = accuracy_score(y_true=y_test, y_pred=logregression_grid.predict(scaled_test))
cfmatrix_grid = confusion_matrix(y_true=y_test, y_pred=logregression_grid.predict(scaled_test))
print("**Grid search results**")
print("Best training accuracy:\t", logregression_grid.best_score_)
print("Test accuracy:\t", acc)

logregression_random = cherchez(LogisticRegression(), lr_dist, search="random")
acc = accuracy_score(y_true=y_test, y_pred=logregression_random.predict(scaled_test))
cfmatrix_rand = confusion_matrix(y_true=y_test, y_pred=logregression_random.predict(scaled_test))
print("**Random search results**")
print("Best training accuracy:\t", logregression_random.best_score_)
print("Test accuracy:\t", acc)

plt.subplots(1,2)
plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plot_confusion_matrix(cfmatrix_rand, title="Random Search Confusion Matrix")
plt.subplot(121)
plot_confusion_matrix(cfmatrix_grid, title="Grid Search Confusion Matrix")


# Discussion:<br>
# In this case, Logistic Regression wasn't very good at classifying between AML and ALL. In both Grid and Random searches, the differences between the training and test scores show that the model is overfitting to the training data. In fact, the random search results actually overfit more than the grid search which is not what I expected.  Neither search type improved actual prediction results. Comparing the accuracy in confusion matrices the true positives (top left and bottm right corners) are almost the same for each type of search.
# 
# Conclusion: <br>
# A randomized parameter search resulted in higher variance. Possible caveats are that the dataset is rather small and I really haven't tried to choose proper parameters to tune so there may be room for a *little* improvement. 
# <br><br>

# In[ ]:


# SVM
svm_param = {
    "C": [.01, .1, 1, 5, 10, 100],
    "gamma": [0, .01, .1, 1, 5, 10, 100],
    "kernel": ["rbf"],
    "random_state": [1]
}

svm_dist = {
    "C": scipy.stats.expon(scale=.01),
    "gamma": scipy.stats.expon(scale=.01),
    "kernel": ["rbf"],
    "random_state": [1]
}

svm_grid = cherchez(SVC(), svm_param, "grid")
acc = accuracy_score(y_true=y_test, y_pred=svm_grid.predict(scaled_test))
cfmatrix_grid = confusion_matrix(y_true=y_test, y_pred=svm_grid.predict(scaled_test))
print("**Grid search results**")
print("Best training accuracy:\t", svm_grid.best_score_)
print("Test accuracy:\t", acc)

svm_random = cherchez(SVC(), svm_dist, "random")
acc = accuracy_score(y_true=y_test, y_pred=svm_random.predict(scaled_test))
cfmatrix_rand = confusion_matrix(y_true=y_test, y_pred=svm_random.predict(scaled_test))
print("**Random search results**")
print("Best training accuracy:\t", svm_random.best_score_)
print("Test accuracy:\t", acc)

plt.subplots(1,2)
plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plot_confusion_matrix(cfmatrix_rand, title="Random Search Confusion Matrix")
plt.subplot(121)
plot_confusion_matrix(cfmatrix_grid, title="Grid Search Confusion Matrix")


# Discussion:<br>
# Well, this example goes to show that if you just predict that every patient has AML, you'll be correct more often than wrong. The model predicted that all 34 patients in the test data had AML and was absolutely right for 20 of them but absolutely wrong for the other 14 patients. 
# 
# Conclusion<br>
# I wouldn't use this particular model to classify cancer patients.
# <br><br>

# In[ ]:


# KNN
knn_param = {
    "n_neighbors": [i for i in range(1,30,5)],
    "weights": ["uniform", "distance"],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "leaf_size": [1, 10, 30],
    "p": [1,2]
}

knn_dist = {
    "n_neighbors": scipy.stats.randint(1,33),
    "weights": ["uniform", "distance"],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "leaf_size": scipy.stats.randint(1,1000),
    "p": [1,2]
}

knn_grid = cherchez(KNeighborsClassifier(), knn_param, "grid")
acc = accuracy_score(y_true=y_test, y_pred=knn_grid.predict(scaled_test))
cfmatrix_grid = confusion_matrix(y_true=y_test, y_pred=svm_grid.predict(scaled_test))
print("**Grid search results**")
print("Best training accuracy:\t", knn_grid.best_score_)
print("Test accuracy:\t", acc)

knn_random = cherchez(KNeighborsClassifier(), knn_dist, "random")
acc = accuracy_score(y_true=y_test, y_pred=knn_random.predict(scaled_test))
cfmatrix_rand = confusion_matrix(y_true=y_test, y_pred=knn_random.predict(scaled_test))
print("**Random search results**")
print("Best training accuracy:\t", knn_random.best_score_)
print("Test accuracy:\t", acc)

plt.subplots(1,2)
plt.subplots_adjust(left=-0.5, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plot_confusion_matrix(cfmatrix_rand, title="Random Search Confusion Matrix")
plt.subplot(121)
plot_confusion_matrix(cfmatrix_grid, title="Grid Search Confusion Matrix")


# Discusison:<br>
# As with the other models, an automated search for hyperparameters with KNN just overfits the training data and doesn't generalize to the test data. The one difference here is that a randomized search produced a model which predicted that at least some of the patients had ALL rather than predicting that all patients have AML.
# 
# Conclusion:<br>
# My main problem is that the GridSearch and RandomSearch use the model that fits the training data the best without taking into account how well it will generalizes to the test data. 

# In[ ]:


# Decision tree classifier
dtc_param = {
    "max_depth": [None],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "min_weight_fraction_leaf": [0.],
    "max_features": [None],
    "random_state": [4],
    "max_leaf_nodes": [None], # None = infinity or int
    "presort": [True, False]
}


dtc_grid = cherchez(DecisionTreeClassifier(), dtc_param, "grid")
acc = accuracy_score(y_true=y_test, y_pred=dtc_grid.predict(scaled_test))
cfmatrix_grid = confusion_matrix(y_true=y_test, y_pred=dtc_grid.predict(scaled_test))
print("**Grid search results**")
print("Best training accuracy:\t", dtc_grid.best_score_)
print("Test accuracy:\t", acc)

plot_confusion_matrix(cfmatrix_grid, title="Decision Tree Confusion Matrix")


# Discussion:<br>
# This model accurately predicted more ALL patients than any of the other models and had the lowest false positive rate, but frankly that's just painting it in a positive light and I'm not even bothering to calculate it. The model still has poor accuracy overall, doesn't generalize to the test data, and has a huge variance problem
# 
# Conclusion:<br>
# This experiment makes a good case for splitting your data into train, validation, and test sets. It would be nice if GridSearch and RandomSearch could refit the best predictors that minimize the variance between the accuracy of train and validation sets. As it is, overfitting the training data has been a problem for each of these models. It's possible that I could investigate which parameters would be better to tune and it's possible that there isn't enough data, and it's also possible that this isn't a good classification problem. Maybe I need to approach this as a clustering problem. What do you think?

# # Sources:
# 
# Golub et al: https://www.ncbi.nlm.nih.gov/pubmed/10521349
# 
# Bayesian optimization: https://arxiv.org/pdf/1012.2599v1.pdf
# 
