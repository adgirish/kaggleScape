
# coding: utf-8

# Intro
# -----
# 
# Hello! This is a short guide on how to set up basic classifiers for the "Leaf Classification" dataset. I'm still a novice in the field, I've been poking around Kaggle for a while now, so this is more of a notebook for myself, to keep track of what I am learning, than an actual guide. Suggestions and comments are more than welcome!  
# I will use three of the simplest classification methods (Naïve Bayes, Random Forest and Logistic Regression) and, given the high number of features in this dataset, I will briefly comment on how to reduce their correlation with Principal Component Analysis.

# Here are some of the kernels on Kaggle that I used to learn and to build my code:  
# [10 Classifier Showdown][1],
# [Random Forest][2],
# [Logistic Regression][3].
# 
# 
# 
# [1]: https://www.kaggle.com/jeffd23/leaf-classification/10-classifier-showdown-in-scikit-learn
# [2]: https://www.kaggle.com/sunnyrain/leaf-classification/random-forests-with-0-68-score/code
# [3]: https://www.kaggle.com/bmetka/leaf-classification/logistic-regression/code

# Import Libraries and Define Functions
# -------------------------------------

# In[ ]:


#Let's start importing the libraries that we will use
import csv as csv 
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from random import randint
from scipy import stats  

#Here are the sklearn libaries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.decomposition import PCA

#Here we define a function to calculate the Pearson's correlation coefficient
#which we will use in a later part of the notebook
def pearson(x,y):
	if len(x)!=len(y):
		print("I can't calculate Pearson's Coefficient, sets are not of the same length!")
		return
	else:
		sumxy = 0
		for i in range(len(x)):
			sumxy = sumxy + x[i]*y[i]
		return (sumxy - len(x)*np.mean(x)*np.mean(y))			/((len(x)-1)*np.std(x, ddof=1)*np.std(y, ddof=1))


# Import and Prepare the Data
# ---------------------------

# In[ ]:


#Import the dataset and do some basic manipulation
traindf = pd.read_csv('../input/train.csv', header=0) 
#testdf = pd.read_csv('../input/test.csv', header=0) I won't use the test set in this notebook

#We can have a look at the data, shape and types, but I'll skip this step here
#traindf.dtypes
#traindf.info()
#traindf.describe
#The dataset is complete, so there's no need here to clean it from empty entries.
#traindf = traindf.dropna() 

#We separate the features from the classes, 
#we can either put them in ndarrays or leave them as pandas dataframes, since sklearn can handle both. 
#x_train = traindf.values[:, 2:] 
#y_train = traindf.values[:, 1]
x_train = traindf.drop(['id', 'species'], axis=1)
y_train = traindf.pop('species')
#x_test = traindf.drop(['id'], axis=1)

#Sometimes it may be useful to encode labels with numeric values, but is unnecessary in this case 
#le = LabelEncoder().fit(traindf['species']) 
#y_train = le.transform(train['species'])
#classes = list(le.classes_)

#However, it's a good idea to standardize the data (namely to rescale it around zero 
#and with unit variance) to avoid that certain unscaled features 
#may weight more on the classifier decision 
scaler = StandardScaler().fit(x_train) #find mean and std for the standardization
x_train = scaler.transform(x_train) #standardize the training values
#x_test = scaler.transform(x_test)


# First Classifiers
# ---------------
# 
# We can now start setting up our classifiers and naively apply them to the dataset as it is. Ideally, one should first look at the data in depth and think how to reduce it or manipulate it to get the most out of the classifiers. However, we will need these results later, to compare how effective the Features Reduction is in each case. 
# To compare performance and efficacy of each technique we will use a  **K-fold cross validation**. This technique randomly splits the training dataset in k subsets. While one subset is kept to test the model, the remaining k-1 sets are used to train the data.

# In[ ]:


#Initialise the K-fold with k=5
kfold = KFold(n_splits=5, shuffle=True, random_state=4)


# We start with **Naïve Bayes**, one of the most basic classifiers. Here the Bayesian probability theorem is used to predict the classes, with the "naïve" assumption that the features are independent. In the *sklearn* library implementation **Gaussian Naïve Bayes**, the likelihood of the features is Gaussian-shaped and its parameters are calculated with the maximum likelihood method.   

# In[ ]:


#Initialise Naive Bayes
nb = GaussianNB()
#We can now run the K-fold validation on the dataset with Naive Bayes
#this will output an array of scores, so we can check the mean and standard deviation
nb_validation=[nb.fit(x_train[train], y_train[train]).score(x_train[test], y_train[test]).mean()            for train, test in kfold.split(x_train)]


# We repeat the process, this time with **Random Forest**, one of the most popular classifiers here on Kaggle. An high number of decision trees (or *forest*) are built and trained on the dataset and the mode of the results is then given as output. 
# To improve statistics and avoid over-fitting, the **Extremely Randomized Trees** (or *Extra-Trees*) found in *sklearn* is used here. This implementation generates randomised decision trees which are then fitted on random sub-sets of the data and finally averaged. 

# In[ ]:


#Initialise Extra-Trees Random Forest
rf = ExtraTreesClassifier(n_estimators=500, random_state=0)
#Run K-fold validation with RF
#Again the classifier is trained on the k-1 sub-sets and then tested on the remaining k-th subset
#and scores are calcualted
rf_validation=[rf.fit(x_train[train], y_train[train]).score(x_train[test], y_train[test]).mean()                for train, test in kfold.split(x_train)]


# The *sklearn* implementation of **Random Forest** allows to get information on which features are the most important in the classification; although not that useful here, it can usually be beneficial in the features reduction phase, to identify which descriptor are more likely to lead to loss of valuable information if dropped.

# In[ ]:


#We extract the importances, their indices and standard deviations
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
imp_std = np.std([est.feature_importances_ for est in rf.estimators_], axis=0)

#And we plot the first and last 10 features out of curiosity
fig = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(1, 2)#, height_ratios=[1, 1]) 
ax1, ax2 = fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])
ax1.margins(0.05), ax2.margins(0.05) 
ax1.bar(range(10), importances[indices][:10],        color="#6480e5", yerr=imp_std[indices][:10], ecolor='#31427e', align="center")
ax2.bar(range(10), importances[indices][-10:],        color="#e56464", yerr=imp_std[indices][-10:], ecolor='#7e3131', align="center")
ax1.set_xticks(range(10)), ax2.set_xticks(range(10))
ax1.set_xticklabels(indices[:10]), ax2.set_xticklabels(indices[-10:])
ax1.set_xlim([-1, 10]), ax2.set_xlim([-1, 10])
ax1.set_ylim([0, 0.035]), ax2.set_ylim([0, 0.035])
ax1.set_xlabel('Feature #'), ax2.set_xlabel('Feature #')
ax1.set_ylabel('Random Forest Normalized Importance') 
ax2.set_ylabel('Random Forest Normalized Importance')
ax1.set_title('First 10 Important Features'), ax2.set_title('Last 10 Important Features')
gs1.tight_layout(fig)
#plt.show()


# Finally we try **Logistic Regression**, another very popular classifier. It's a **Generalized Linear Model** (the target values is expected to be a linear combination of the input variables) for classification, where a logistic (or *sigmoid*) function is fitted on the data to describe the probability of an outcome at each trial. This model requires in input set of *hyper-parameters*, that can't be learned by the model.  **Exhaustive Grid Search** comes to the rescue: given a range for each parameter,  it explores the hyper-space of the parameters within the boundaries set by these ranges and finds the values that maximise specific scoring functions.

# In[ ]:


#We first define the ranges for each parameter we are interested in searching 
#(while the others are left as default):
#C is the inverse of the regularization strength
#tol is the tolerance for stopping the criteria
params = {'C':[100, 1000], 'tol': [0.001, 0.0001]}
#We initialise the Logistic Regression
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
#We initialise the Exhaustive Grid Search, we leave the scoring as the default function of 
#the classifier singe log loss gives an error when running with K-fold cross validation
#add n_jobs=-1 in a parallel computing calculation to use all CPUs available
#cv=3 increasing this parameter makes it too difficult for kaggle to run the script
gs = GridSearchCV(lr, params, scoring=None, refit='True', cv=3) 
gs_validation=[gs.fit(x_train[train], y_train[train]).score(x_train[test], y_train[test]).mean()                for train, test in kfold.split(x_train)]


# Now that we set up our three classifiers we can check the validation results and compare their performances looking at the averages of their cross validation results.

# In[ ]:


print("Validation Results\n==========================================")
print("Naive Bayes: " + '{:1.3f}'.format(np.mean(nb_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(nb_validation)))
print("Random Forest: " + '{:1.3f}'.format(np.mean(rf_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(rf_validation)))
print("Logistic Regression: " + '{:1.3f}'.format(np.mean(gs_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(gs_validation)))


# The results are very interesting, while both Random Forest and Logistic Regression perform fairly well (with the latter slightly better than the former), Naïve Bayes validation result is extremely low. 
# This could be due to the fact that the initial assumption of independent features doesn't hold of this dataset. It would be worth at this point to polish a bit the features, and see if by reducing their number we can improve the results of our classifiers.

# Features Reduction
# ------------------

# A straightforward way to simplify the features is to remove possible correlations. We can intuitively assume that Naïve Bayes results will improve significantly, but it would be good to see if this choice will also improve the validation results for the other two classifiers. 
# 
# The dataset contains  total of 192 descriptors, subdivided in three categories: margin, shape and texture. We can first check if within each one of these categories the features are correlated by calculating correlation scores between couples of features (in this case we will use the *Pearson's correlation coefficient*), and then if one of the categories has highly correlated results, we can build a correlation matrix among all the descriptors within such category.

# In[ ]:


#First we find the sets of margin, shape and texture columns 
margin_cols = [col for col in traindf.columns if 'margin' in col]
shape_cols = [col for col in traindf.columns if 'shape' in col] 
texture_cols = [col for col in traindf.columns if 'texture' in col] 
margin_pear, shape_pear, texture_pear = [],[],[]

#Then we calculate the correlation coefficients for each couple of columns: we can either do this
#between random columns of between consecutive columns, the difference won't matter much since we are
#just exploring the data
for i in range(len(margin_cols)-1):
    margin_pear.append(pearson(traindf[margin_cols[i]],traindf[margin_cols[i+1]]))
	#margin_pear.append(pearson(traindf[margin_cols[randint(0,len(margin_cols)-1)]],\
        #traindf[margin_cols[randint(0,len(margin_cols)-1)]]))
for i in range(len(shape_cols)-1):
	shape_pear.append(pearson(traindf[shape_cols[i]],traindf[shape_cols[i+1]]))
	#shape_pear.append(pearson(traindf[shape_cols[randint(0,len(shape_cols)-1)]],\
        #traindf[shape_cols[randint(0,len(shape_cols)-1)]]))
for i in range(len(texture_cols)-1):
	texture_pear.append(pearson(traindf[texture_cols[i]],traindf[texture_cols[i+1]]))
	#texture_pear.append(pearson(traindf[texture_cols[randint(0,len(texture_cols)-1)]],\
        #traindf[texture_cols[randint(0,len(texture_cols)-1)]]))

#We calculate average and standard deviation for each cathergory 
#and we give it a position on the X axis of the graph
margin_mean, margin_std = np.mean(margin_pear), np.std(margin_pear, ddof=1)
margin_x=[0]*len(margin_pear)
shape_mean, shape_std =	np.mean(shape_pear), np.std(shape_pear, ddof=1)
shape_x=[1]*len(shape_pear)	
texture_mean, texture_std =	np.mean(texture_pear), np.std(texture_pear, ddof=1)	
texture_x=[2]*len(texture_pear)

#We set up the graph
fig = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(1, 2)#, height_ratios=[1, 1]) 
ax1, ax2 = fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])
ax1.margins(0.05), ax2.margins(0.05) 

#We fill the first graph with a scatter plot on a single axis for each category and we add
#mean and standard deviation, which we can also print to screen as a reference
ax1.scatter(margin_x, margin_pear, color='blue', alpha=.3, s=100)
ax1.errorbar([0],margin_mean, yerr=margin_std, color='white', alpha=1, fmt='o', mec='white', lw=2)
ax1.scatter(shape_x, shape_pear, color='red', alpha=.3, s=100)
ax1.errorbar([1],shape_mean, yerr=shape_std, color='white', alpha=1, fmt='o', mec='white', lw=2)
ax1.scatter(texture_x, texture_pear, color='green', alpha=.3, s=100)
ax1.errorbar([2],texture_mean, yerr=texture_std, color='white', alpha=1, fmt='o', mec='white', lw=2)
ax1.set_ylim(-1.25, 1.25), ax1.set_xlim(-0.25, 2.25)
ax1.set_xticks([0,1,2]), ax1.set_xticklabels(['margin','shape','texture'], rotation='vertical')
ax1.set_xlabel('Category'), ax1.set_ylabel('Pearson\'s Correlation')
ax1.set_title('Neighbours Correlation')
ax1.set_aspect(2.5)

print("Pearson's Correlation between neighbours\n==========================================")
print("Margin: " + '{:1.3f}'.format(margin_mean) + u' \u00B1 '        + '{:1.3f}'.format(margin_std))
print("Shape: " + '{:1.3f}'.format(shape_mean) + u' \u00B1 '        + '{:1.3f}'.format(shape_std))
print("Texture: " + '{:1.3f}'.format(texture_mean) + u' \u00B1 '        + '{:1.3f}'.format(texture_std))

#And now, we build a more detailed (and expensive!) correlation matrix, 
#but only for the shape category, which, as we will see, is highly correlated
shape_mat=[]

for i in range(traindf[shape_cols].shape[1]):
    shape_mat.append([])
    for j in range(traindf[shape_cols].shape[1]):
        shape_mat[i].append(pearson(traindf[shape_cols[i]],traindf[shape_cols[j]]))

cmap = cm.RdBu_r
MS= ax2.imshow(shape_mat, interpolation='none', cmap=cmap, vmin=-1, vmax=1)
ax2.set_xlabel('Shape Feature'), ax2.set_ylabel('Shape Feature')
cbar = plt.colorbar(MS, ticks=np.arange(-1.0,1.1,0.2))
cbar.set_label('Pearson\'s Correlation')
ax2.set_title('Shape Category Correlation Matrix')

#And we have a look at the resulting graphs
gs1.tight_layout(fig)
#plt.show()


# The *Pearson's Coefficient* goes from 1 for perfectly correlated arrays, to -1 for perfectly anti-correlated arrays. The midpoint 0 represents uncorrelated data. The results shown in the graph are very interesting. From the plot on the left it's possible to observe that couples of features categorised under "texture" (in green) are fairly uncorrelated, the mean is close to zero and the variance is low, with only one case with correlation higher then 0.5. Similarly,  the "margin" features are centred in zero, but their variance is higher and we can observe a few points with about 0.8 of correlation.
# 
# The most striking result is however observed with the "shape" category, where close couples of features are highly correlated. Taking random couples of descriptors doesn't improve the situation much, and this can be easily understood by looking in detail at the second graph, where the correlation matrix of all the features is shown. The descriptors are highly correlated with their closest neighbours, but also with other features periodically. This data forms a nice pattern, yes, but can lead to bad results if not properly taken care of. This also confirms our theory that the starting assumption of Naïve Bayes was wrong. 
# 
# Obviously correlation is not the only parameter one should look at when cleaning the data, but for this notebook we are going to focus only on this problem.

# Principal Component Analysis
# ----------------------------
# 
# **Principal Component Analysis**  (or *PCA* in short) allows to transform sets of correlated variables, like our leaves features, into linearly uncorrelated, orthogonal, vectors (the principal components). The output sets are ordered so that the first principal component accounts for as much variability (and thus information) as possible from the input data, and it has the largest variance, while the following vectors have the highest possible variance allowed by their orthogonality.  
# 
# The number of sets in output can be lower than the number of input features, sometimes all information from our descriptors can be contained in a lower number of vectors, and for this reason **PCA** is often used as a dimensionality reduction method. It implementation in *sklearn* is based on the **Singular Value Decomposition** (or *SVD*),  a method to extract the eigenvectors, the orthogonal vectors, by means of a factorisation of the matrix of the input features. 
# 
# Although in principle we could apply **PCA** it only to a subset of the training data, given that some of the categories don't have much correlation among their members, we will apply here to the whole data set, to make sure that all features are orthogonal.

# In[ ]:


#We initialise pca choosing Minka’s MLE to guess the minimum number of output components necessary
#to maintain the same information coming from the input descriptors and we ask to solve SVD in full
pca = PCA(n_components = 'mle', svd_solver = 'full')
#Then we fit pca on our training set and we apply to the same entire set
x_train_pca=pca.fit_transform(x_train)

#Now we can compare the dimensions of the training set before and after applying PCA and see if we 
#managed to reduce the number of features. 
print("Number of descriptors before PCA: " + '{:1.0f}'.format(x_train.shape[1]))
print("Number of descriptors after PCA: " + '{:1.0f}'.format(x_train_pca.shape[1]))


# As we can see, **PCA** only reduced the features by one element. This doesn't mean, however, that the results won't be improved. We can now apply again our classifiers to this new set and see if anything has changed.

# In[ ]:


#Naive Bayes
nb_validation=[nb.fit(x_train_pca[train], y_train[train]).score(x_train_pca[test], y_train[test]).mean()            for train, test in kfold.split(x_train)]
#Random Forest
rf_validation=[rf.fit(x_train_pca[train], y_train[train]).score(x_train_pca[test], y_train[test]).mean()                for train, test in kfold.split(x_train)]
#Logistic Regression
gs_validation=[gs.fit(x_train_pca[train], y_train[train]).score(x_train_pca[test], y_train[test]).mean()                for train, test in kfold.split(x_train)]

#And we print the results
print("Validation Results After PCA\n==========================================")
print("Naive Bayes: " + '{:1.3f}'.format(np.mean(nb_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(nb_validation)))
print("Random Forest: " + '{:1.3f}'.format(np.mean(rf_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(rf_validation)))
print("Logistic Regression: " + '{:1.3f}'.format(np.mean(gs_validation)) + u' \u00B1 '        + '{:1.3f}'.format(np.std(gs_validation)))


# As we expected, the validation results improve significantly for the Naïve Bayes classifier, since now the features independence assumption is correct. No difference is observed instead for the other two classifiers, where the results are consistent with those before **PCA** within a standard deviation. 

# In[ ]:


#Again, we can check if anything changed in the features importance of our Random Forest classifier
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
imp_std = np.std([est.feature_importances_ for est in rf.estimators_], axis=0)

fig = plt.figure(figsize=(8, 6))
gs1 = gridspec.GridSpec(1, 2)#, height_ratios=[1, 1]) 
ax1, ax2 = fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])
ax1.margins(0.05), ax2.margins(0.05) 
ax1.bar(range(10), importances[indices][:10],        color="#6480e5", yerr=imp_std[indices][:10], ecolor='#31427e', align="center")
ax2.bar(range(10), importances[indices][-10:],        color="#e56464", yerr=imp_std[indices][-10:], ecolor='#7e3131', align="center")
ax1.set_xticks(range(10)), ax2.set_xticks(range(10))
ax1.set_xticklabels(indices[:10]) ,ax2.set_xticklabels(indices[-10:])
ax1.set_xlim([-1, 10]), ax2.set_xlim([-1, 10])
ax1.set_ylim([0, 0.035]), ax2.set_ylim([0, 0.035])
ax1.set_xlabel('Feature #'), ax2.set_xlabel('Feature #')
ax1.set_ylabel('Random Forest Normalized Importance'), ax2.set_ylabel('Random Forest Normalized Importance')
ax1.set_title('First 10 Important Features (after PCA)'), ax2.set_title('Last 10 Important Features (after PCA)')
gs1.tight_layout(fig)
#plt.show()


# However, looking at the feature importances of the **Random Forest** classifier we can see how the information has been restructured and divided among the features. The 10 first important features (on the left) are taken among the first in the first vectors of the **PCA**, the ones with highest variance, while the least important features for the classifier are mostly among the last vectors of our modified dataset, where the remaining information has been stored. Dropping them would lead to a modest loss of information. Moreover, the actual value of the importance is increased for the first ten vectors and equally low for the last ten.  
# It is interesting to note how this rearrangement of the information through **PCA** and the elimination of just one descriptor doesn't significantly affect the results for the **Random Forest** classifier. 

# These results gives us a nice example of how the results of different classifiers are highly dependent on the structure of our input data. One should consider carefully which classifier to use depending on the kind problem and data provided and only after an in depth analysis and proper cleaning of the dataset.
