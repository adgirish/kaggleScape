
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score as auc
import time

plt.rcParams['figure.figsize'] = (12, 6)

#%% load data and remove constant and duplicate columns  (taken from a kaggle script)

trainDataFrame = pd.read_csv('../input/train.csv')

# remove constant columns
colsToRemove = []
for col in trainDataFrame.columns:
    if trainDataFrame[col].std() == 0:
        colsToRemove.append(col)

trainDataFrame.drop(colsToRemove, axis=1, inplace=True)

# remove duplicate columns
colsToRemove = []
columns = trainDataFrame.columns
for i in range(len(columns)-1):
    v = trainDataFrame[columns[i]].values
    for j in range(i+1,len(columns)):
        if np.array_equal(v,trainDataFrame[columns[j]].values):
            colsToRemove.append(columns[j])

trainDataFrame.drop(colsToRemove, axis=1, inplace=True)

trainLabels = trainDataFrame['TARGET']
trainFeatures = trainDataFrame.drop(['ID','TARGET'], axis=1)


# ### Build an estimator trying to predict the target for each feature individually
# 

# In[ ]:


#%% look at single feature performance

verySimpleLearner = ensemble.GradientBoostingClassifier(n_estimators=10, max_features=1, max_depth=3,
                                                        min_samples_leaf=100,learning_rate=0.3, subsample=0.65,
                                                        loss='deviance', random_state=1)

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainFeatures, trainLabels, test_size=0.5, random_state=1)
        
startTime = time.time()
singleFeatureAUC_list = []
singleFeatureAUC_dict = {}
for feature in X_train.columns:
    trainInputFeature = X_train[feature].values.reshape(-1,1)
    validInputFeature = X_valid[feature].values.reshape(-1,1)
    verySimpleLearner.fit(trainInputFeature, y_train)
    
    trainAUC = auc(y_train, verySimpleLearner.predict_proba(trainInputFeature)[:,1])
    validAUC = auc(y_valid, verySimpleLearner.predict_proba(validInputFeature)[:,1])
        
    singleFeatureAUC_list.append(validAUC)
    singleFeatureAUC_dict[feature] = validAUC
        
validAUC = np.array(singleFeatureAUC_list)
timeToTrain = (time.time()-startTime)/60
print("(min,mean,max) AUC = (%.3f,%.3f,%.3f). took %.2f minutes" %(validAUC.min(),validAUC.mean(),validAUC.max(), timeToTrain))

# show the scatter plot of the individual feature performance 
plt.figure(); plt.hist(validAUC, 50, normed=1, facecolor='blue', alpha=0.75)
plt.xlabel('AUC'); plt.ylabel('frequency'); plt.title('single feature AUC histogram'); plt.show()


# ### Show single feature AUC performace

# In[ ]:


# create a table with features sorted according to AUC
singleFeatureTable = pd.DataFrame(index=range(len(singleFeatureAUC_dict.keys())), columns=['feature','AUC'])
for k,key in enumerate(singleFeatureAUC_dict):
    singleFeatureTable.ix[k,'feature'] = key
    singleFeatureTable.ix[k,'AUC'] = singleFeatureAUC_dict[key]
singleFeatureTable = singleFeatureTable.sort_values(by='AUC', axis=0, ascending=False).reset_index(drop=True)

singleFeatureTable.ix[:15,:]


# ### Show scatter pltos of (feature, target) for the top performing single features

# In[ ]:


numSubPlotRows = 1
numSubPlotCols = 2
for plotInd in range(8):
    plt.figure()
    for k in range(numSubPlotRows*numSubPlotCols):
        tableRow = numSubPlotRows*numSubPlotCols*plotInd+k
        x = X_train[singleFeatureTable.ix[tableRow,'feature']].values.reshape(-1,1)[:,0]
        
        # use a huristic to find out if the variable is categorical, and if so add some random noise to it
        if len(np.unique(x)) < 20:
            diffVec = abs(x[1:]-x[:-1])
            minDistBetweenCategories = min(diffVec[diffVec > 0])
            x = x + 0.12*minDistBetweenCategories*np.random.randn(np.shape(x)[0])
            
        y = y_train + 0.12*np.random.randn(np.shape(y_train)[0])
        # take only 3000 samples to be presented due to plotting issues
        randPermutation = np.random.choice(len(x), 3000, replace=False)
        plt.subplot(numSubPlotRows,numSubPlotCols,k+1)
        plt.scatter(x[randPermutation], y[randPermutation], c=y_train[randPermutation], cmap='jet', alpha=0.25)
        plt.xlabel(singleFeatureTable.ix[tableRow,'feature']); plt.ylabel('y GT')
        plt.title('AUC = %.4f' %(singleFeatureTable.ix[tableRow,'AUC']))            
        plt.ylim(-0.5,1.5); plt.tight_layout()


# ### Build an estimator trying to predict the target with pairs of features

# In[ ]:


#%% look at performance of pairs of features

# limit run time (on all feature combinations should take a few hours)
numFeaturesToUse = 20
featuresToUse = singleFeatureTable.ix[0:numFeaturesToUse-1,'feature']

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainFeatures, trainLabels, test_size=0.5, random_state=1)
    
startTime = time.time()
featurePairAUC_list = []
featurePairAUC_dict = {}

for feature1Ind in range(len(featuresToUse)-1):
    featureName1 = featuresToUse[feature1Ind]
    trainInputFeature1 = X_train[featureName1].values.reshape(-1,1)
    validInputFeature1 = X_valid[featureName1].values.reshape(-1,1)

    for feature2Ind in range(feature1Ind+1,len(featuresToUse)-1):
        featureName2 = featuresToUse[feature2Ind]
        trainInputFeature2 = X_train[featureName2].values.reshape(-1,1)
        validInputFeature2 = X_valid[featureName2].values.reshape(-1,1)

        trainInputFeatures = np.hstack((trainInputFeature1,trainInputFeature2))
        validInputFeatures = np.hstack((validInputFeature1,validInputFeature2))
        
        verySimpleLearner.fit(trainInputFeatures, y_train)
        
        trainAUC = auc(y_train, verySimpleLearner.predict_proba(trainInputFeatures)[:,1])
        validAUC = auc(y_valid, verySimpleLearner.predict_proba(validInputFeatures)[:,1])
            
        featurePairAUC_list.append(validAUC)
        featurePairAUC_dict[(featureName1,featureName2)] = validAUC
        
validAUC = np.array(featurePairAUC_list)
timeToTrain = (time.time()-startTime)/60
print("(min,mean,max) AUC = (%.3f,%.3f,%.3f). took %.1f minutes" % (validAUC.min(),validAUC.mean(),validAUC.max(), timeToTrain))

# show the histogram of the feature combinations performance 
plt.figure(); plt.hist(validAUC, 50, normed=1, facecolor='blue', alpha=0.75)
plt.xlabel('AUC'); plt.ylabel('frequency'); plt.title('feature pair AUC histogram'); plt.show()


# ### Show AUC performace of best pairs of features

# In[ ]:


# create a table with features sorted according to AUC
featureCombinationsTable = pd.DataFrame(index=range(len(featurePairAUC_list)), columns=['feature1','feature2','AUC'])
for k,key in enumerate(featurePairAUC_dict):
    featureCombinationsTable.ix[k,'feature1'] = key[0]
    featureCombinationsTable.ix[k,'feature2'] = key[1]
    featureCombinationsTable.ix[k,'AUC'] = featurePairAUC_dict[key]
featureCombinationsTable = featureCombinationsTable.sort_values(by='AUC', axis=0, ascending=False).reset_index(drop=True)

featureCombinationsTable.ix[:20,:]


# ### Show the top performing feature pairs
# scatter pltos of (feature1, feature2) with colored labels 
# 

# In[ ]:


# show the scatter plot of best feature pair combinations
numPlotRows = 1
numPlotCols = 2
for plotInd in range(8):
    plt.figure()
    for k in range(numPlotRows*numPlotCols):
        tableRow = numPlotRows*numPlotCols*plotInd+k
        x = X_train[featureCombinationsTable.ix[tableRow,'feature1']].values.reshape(-1,1)[:,0]
        y = X_train[featureCombinationsTable.ix[tableRow,'feature2']].values.reshape(-1,1)[:,0]

        # use a huristic to find out if the variables are categorical, and if so add some random noise to them
        if len(np.unique(x)) < 20:
            diffVec = abs(x[1:]-x[:-1])
            minDistBetweenCategories = min(diffVec[diffVec > 0])
            x = x + 0.12*minDistBetweenCategories*np.random.randn(np.shape(x)[0])

        if len(np.unique(y)) < 20:
            diffVec = abs(y[1:]-y[:-1])
            minDistBetweenCategories = min(diffVec[diffVec > 0])
            y = y + 0.12*minDistBetweenCategories*np.random.randn(np.shape(y)[0])

        colors = y_train
        # take only 3000 samples to be presented due to plotting issues
        randPermutation = np.random.choice(len(x), 3000, replace=False)
        plt.subplot(numPlotRows,numPlotCols,k+1)
        plt.scatter(x[randPermutation], y[randPermutation], s=(3+1.6*colors[randPermutation])**2, c=-colors[randPermutation], cmap='spring', alpha=0.75)
        plt.xlabel(featureCombinationsTable.ix[tableRow,'feature1']); plt.ylabel(featureCombinationsTable.ix[tableRow,'feature2'])
        plt.title('AUC = %.4f' %(featureCombinationsTable.ix[tableRow,'AUC'])); plt.tight_layout()

