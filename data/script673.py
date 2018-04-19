
# coding: utf-8

# # The 12 Different Types of Kagglers 
# 
# They say people are not averages. They also say you can drown in a pool who's average depth is 12 cm.   
# In this script we try to move a little past "the average kaggler" and present "the 12 typical kagglers". 
# These are 12 imagined people that represent the kaggle community hopefully better than that of a single average kaggler.  
# This will be analogous to the following: instead of talking about "the pool's average depth is 12cm" move on to talk about "95% of the pool's area is 2 cm deep on average, and 5% of the pool's area is 2 meters deep on average".  
# 
# This dataset holds the 2017 kaggle data science survay results. It is presented to us here in a very raw and unprocessed form (to say the least), so we'll first need to move quite a bit of data around before we can continue.  
# 
# After a little bit of pre-processing we will look at correlations between several key data fields, and then continue to cluster analyize the ~16,000 kagglers that have answered this survey.
# 
# **In the final and main** part of the script, we present a short summery paragraph of each kaggler type one by one. 

# # The TL;DR Version:
# To see where these descriptions come from, please continue to the analysis
# 
# ### Paragraph Summery of Kaggler Type no. 1:
# The most frequent kaggler (accounts for slightly less than 15% of kagglers) is a 28 year old male from India, he is employed full-time as a Software Developer or a Data Scientist, he has some industry experience, he majored in Computer Science (CS) during University and holds a Master's degree. His annual salary is about 14K dollars a year. He is looking forward to learn about deep learning in the upcoming year and specifically using tensorflow.
# 
# ### Paragraph Summery of Kaggler Type no. 2:
# The 2nd most frequent kaggler (accounts for slightly less than 13% of kagglers) is a 27 year old male from India, he is employed full-time as a Software Developer, has little industry experience, he has majored in CS during university and holds a Bachelor's degree.
# 
# ### Paragraph Summery of Kaggler Type no. 3:
# The 3rd most frequent kaggler (accounts for about 11% of kagglers) is a 34 year old male from the US, he is employed full-time, and doesn't really have the time to fill out an internet survey.
# 
# ### Paragraph Summery of Kaggler Type no. 4:
# The 4th most frequent kaggler (accounts for about 9% of kagglers) is a 29 year old female from the US, she is employed full-time as a Data Scientist, has 3-5 years of industry experience, her background is CS and she holds a Master's degree. She didn't share her salary but we can infer from her background that she's not starving.
# 
# ### Paragraph Summery of Kaggler Type no. 5:
# The 5th most frequent kaggler (accounts for about 8.5% of kagglers) is a 30 year old male from the US, he is employed full-time as a Data Scientist, has 3-5 years of industry experience and holds a Master's degree in Mathematics/Statistics. He mainly works with python. His anual salary is 92K dollars. Next year, he wants to learn more about deep learning and specifically using tensorflow.
# 
# ### Paragraph Summery of Kaggler Type no. 6:
# The 6th most frequent kaggler (accounts for about 8.5% of kagglers) is a 22 year old male from the India, he is not employed, has no experience, he holds a Bachelor's degree majoring in CS. He mainly works with python. He spends 2-10 hours each week learning Data Science. His first exposure to data science was at online courses and he values kaggle competitions very highly as a potential credential.
# 
# ### Paragraph Summery of Kaggler Type no. 7:
# The 7th most frequent kaggler (accounts for about 8% of kagglers) is a 44 year old male from the US, he is employed full time at various different professions, has more than 10 years of experience, holds a Master's degree and majored in CS. His first training is University (they didn't have online courses 20 years ago...). He mainly works with python. He too is looking forward to gaining experience with deep learning and tensorflow in the upcoming year.
# 
# ### Paragraph Summery of Kaggler Type no. 8:
# The 8th most frequent kaggler (accounts for about 7.4% of kagglers) is a 34 year old male from around the world, he is employed full time as a Software Developer, but has no Data Science experience. he holds a Master's degree and majored in CS. His first Data Science training is coming from online courses and he spends 2-10 hours each week learning data science. He working with python and like everyone else he's also looking forward to gaining experience with deep learning in the upcoming year.
# 
# ### Paragraph Summery of Kaggler Type no. 9:
# The 9th most frequent kaggler (accounts for about 5.8% of kagglers) is a 30 year old male from around the world. He is not employed. He holds a Master's degree and comes from CS and Electrical Engineering (EE). His first Data Science training is coming from online courses and he spends most of his time learning Data Science. He working with a basic laptop and using python. Like everyone else he's also looking forward to learning about deep learning in the upcoming year.
# 
# ### Paragraph Summery of Kaggler Type no. 10:
# The 10th most frequent kaggler (accounts for about 5.6% of kagglers) is a 36 year old male from the US, he is employed full time and has more than 10 years of experience. He holds a Master's degree and comes from an eclectic background. He was first trained at University and is also self taught.
# 
# ### Paragraph Summery of Kaggler Type no. 11:
# The 11th most frequent kaggler (accounts for about 4.6% of kagglers) is a 40 year old male from the US, he is employed full-time as a Data Scientist, has more than 10 years of industry experience, his background is diverse (CS, EE, Math) and he holds a Master's degree. He self taught himself Data Science, and mainly works with python. His anual salary is 181K dollars. Like everyone else, this experienced kaggler wants to learn more about deep learning and specifically using tensorflow.
# 
# ### Paragraph Summery of Kaggler Type no. 12:
# The 12th most frequent and final kaggler (accounts for about 4% of kagglers) is a 26 year old female from the US. She is unemployed, has little industry experience, her background is diverse (CS, EE) and she holds either a Bachelor's or a Master's degree. She learns data science around 2-10 hours a week, and mainly works with python running her code on a basic laptop. Like everyone else, she is interested to learn more about deep learning in the upcoming year.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn import cluster, decomposition, preprocessing


# # Load Data

# In[ ]:


#%% load data
multipleChoiceResponsesDF = pd.read_csv('../input/multipleChoiceResponses.csv', encoding='ISO-8859-1', low_memory=False)
processedDF = multipleChoiceResponsesDF.copy()
processedDF = processedDF.fillna(value='NaN')
allQuestions = processedDF.columns.tolist()


# # Move data around and create a basic subset of numeric features
# 

# In[ ]:


#%% create annual salary in US dollars feature
processedDF['CompensationAmount'] = processedDF['CompensationAmount'].str.replace(',','')
processedDF['CompensationAmount'] = processedDF['CompensationAmount'].str.replace('-','')
processedDF.loc[processedDF['CompensationAmount'] == 'NaN','CompensationAmount'] = '0'
processedDF.loc[processedDF['CompensationAmount'] == '','CompensationAmount'] = '0'
processedDF['CompensationAmount'] = processedDF['CompensationAmount'].astype(float)

conversionRates = pd.read_csv('../input/conversionRates.csv', encoding='ISO-8859-1').set_index('originCountry')
conversionRates = pd.read_csv('../input/conversionRates.csv').set_index('originCountry')
conversionRates.loc['USD']['exchangeRate']

exchangeRate = []
for row in range(processedDF.shape[0]):
    if processedDF.loc[row,'CompensationCurrency'] not in conversionRates.index.tolist():
        exchangeRate.append(1.0)
    else:
        exchangeRate.append(conversionRates.loc[processedDF.loc[row,'CompensationCurrency']]['exchangeRate'])
        
processedDF['exchangeRate'] = exchangeRate
processedDF['annualSalary_USD'] = processedDF['CompensationAmount']*processedDF['exchangeRate']

processedDF.loc[processedDF['annualSalary_USD'] > 300000, 'annualSalary_USD'] = 300000
processedDF['annualSalary_USD'] = processedDF['annualSalary_USD']/1000.0


# In[ ]:


#%% collect all basic features ('age','education level','seniority', 'salary', ...)
def GetDictValueForKey(x):
    return answerToNumericalDict[x]

basicFeatures = ['Country','GenderSelect','Age','FormalEducation','Tenure','annualSalary_USD','MajorSelect','EmploymentStatus','CurrentJobTitleSelect','LanguageRecommendationSelect','TimeSpentStudying']
basicSubsetDF = processedDF[basicFeatures]

additionalFeatures = ['FirstTrainingSelect','ProveKnowledgeSelect','AlgorithmUnderstandingLevel','MLMethodNextYearSelect','MLToolNextYearSelect','HardwarePersonalProjectsSelect','JobSearchResource','EmployerSearchMethod']
additionalSubsetDF = processedDF[additionalFeatures]

# add impatience variables that counts the number of NaNs in a given row for the basic and additional subsets
def CountNaNs(row):
    return (row == 'NaN').sum()

basicSubsetDF['impatience_basic'] = basicSubsetDF.apply(CountNaNs,axis=1)
basicSubsetDF['impatience_additional'] = additionalSubsetDF.apply(CountNaNs,axis=1)
basicSubsetDF['impatience'] = basicSubsetDF['impatience_basic'] + basicSubsetDF['impatience_additional']

# cap age to be in [15,85] range
basicSubsetDF.loc[basicSubsetDF['Age'] == 'NaN','Age'] = basicSubsetDF.loc[basicSubsetDF['Age'] != 'NaN','Age'].mean()
basicSubsetDF.loc[basicSubsetDF['Age'] <= 15,'Age'] = 15
basicSubsetDF.loc[basicSubsetDF['Age'] >= 85,'Age'] = 85

basicSubsetNumericDF = pd.DataFrame()
basicSubsetNumericDF['Age'] = basicSubsetDF['Age']

# transform formal education into an ordinal variable
answerToNumericalDict = {'I prefer not to answer': 10.0,
                         'NaN': 11.0,
                         'I did not complete any formal education past high school': 12.0,
                         'Professional degree': 14.0,
                         "Some college/university study without earning a bachelor's degree": 14.5,
                         "Bachelor's degree": 15.5,
                         "Master's degree": 18.0,
                         "Doctoral degree": 22.0}

basicSubsetNumericDF['Education_Years'] = basicSubsetDF['FormalEducation'].apply(GetDictValueForKey)

# transform tenure into an ordinal variable
answerToNumericalDict = {"I don't write code to analyze data": -0.5,
                         'NaN': 0.0,
                         'Less than a year': 0.5,
                         '1 to 2 years': 1.5,
                         '3 to 5 years': 4.0,
                         '6 to 10 years': 8.0,
                         'More than 10 years': 12.0}

basicSubsetNumericDF['Experience_Years'] = basicSubsetDF['Tenure'].apply(GetDictValueForKey)

# anual salary
basicSubsetNumericDF['annualSalary_USD'] = basicSubsetDF['annualSalary_USD']

# gender to numerical 
answerToNumericalDict = {'Male': -1.0,
                         'NaN': 0.0,
                         'A different identity': 0.0,
                         'Non-binary, genderqueer, or gender non-conforming': 0.0,
                         'Female': 1.0}

basicSubsetNumericDF['Gender'] = basicSubsetDF['GenderSelect'].apply(GetDictValueForKey)

# transform time spent studying to ordinal
answerToNumericalDict = {'NaN': 0.0,
                         '0 - 1 hour': 0.5,
                         '2 - 10 hours': 6.0,
                         '11 - 39 hours': 25.0,
                         '40+': 45.0}

basicSubsetNumericDF['Study_Hours'] = basicSubsetDF['TimeSpentStudying'].apply(GetDictValueForKey)

# add impatience field
basicSubsetNumericDF['impatience'] = basicSubsetDF['impatience']


# In[ ]:


basicSubsetNumericDF.head(15)


# You can unhide the above code to see the mapping of the different features. In short, I've replaced 'NaN' ages with average age, I've tranformed the educations and expreience from string format to approximate number of years. The annual salary is converted based on the provided country exchange rate and presented in units of thousands of dollars per year. For gender, +1 is female, -1 is male, 0 is other. The Impaitiance column is the number of NaN fields in the row for the most basic questions (we will see exactly what they are in the future).   
# I would like to take this opportunity to thank [I Coder](https://www.kaggle.com/ash316), I stole a few pieces of code from his script regarding the salary calculation.

# # Show the correlation matrix of these features

# In[ ]:


#%% show correlations between the most basic feature
basicSubsetNoisyNumericDF = basicSubsetNumericDF.copy()
basicSubsetNoisyNumericDF['Age'] = basicSubsetNoisyNumericDF['Age'].astype(float)
plt.figure(figsize=(12,10)); plt.title('Basic Features - Correlation Matrix', fontsize=22)
sns.heatmap(basicSubsetNoisyNumericDF.corr(), vmin=-1, vmax=1, fmt='.2f', annot=True, cmap='jet'); 
plt.yticks(rotation=0); plt.xticks(rotation=15);


# We can see that 'education', 'expreience', 'salary' and 'age' are positivley correlated, as one would expect. We can see that the gender is not correlated with anything, indicating no clear gender bias for the women that do enter the field of data science, which is a good sign.  
# 
# And we can see an interesting negative correlation between 'impatiance' and 'education' (and also between  'impatiance' and 'expreience' and 'salary'.  
# Even though I would love to write something like "impatiant people are less educated and earn less", but the truth is that this is mainly the result of my descision to fill 'NaN's in the 'education', 'experience', and 'salary' fields as low numeric values.
# 
# # Show the pairwise scatter plots of the basic features

# In[ ]:


for col in ['impatience','Gender','Education_Years','Experience_Years','Study_Hours']:
    basicSubsetNoisyNumericDF[col] *= (1.0 + 0.05*np.random.randn(basicSubsetNoisyNumericDF.shape[0]))
    basicSubsetNoisyNumericDF[col] += 0.3*np.random.randn(basicSubsetNoisyNumericDF.shape[0])

g = sns.pairplot(basicSubsetNoisyNumericDF, diag_kind="kde", plot_kws=dict(s=2, edgecolor="r", alpha=0.1), diag_kws=dict(shade=True));
g.fig.subplots_adjust(top=0.95);
g.fig.suptitle('Basic Features - Pair Scatter Plots', fontsize=30);


# I've added a little bit of noise to the variables that have a small amount of unique values to get a better feel for the relative amounts.  
# 
# # Continue with Data Cleaning - One Hot encode Categoricals

# In[ ]:


#%% apply whitening on each of the basic numeric features we've seen so far
scaledBasicSubset = preprocessing.StandardScaler().fit_transform(basicSubsetNumericDF.values);
numericDF = pd.DataFrame(scaledBasicSubset,columns=basicSubsetNumericDF.columns);

#%% apply one hot encoding to all other features and add it to our numeric dataframe
listOfColsToOneHotEncode = ['Country','MajorSelect','EmploymentStatus','CurrentJobTitleSelect',
                            'LanguageRecommendationSelect','FirstTrainingSelect','ProveKnowledgeSelect',
                            'AlgorithmUnderstandingLevel','MLMethodNextYearSelect','MLToolNextYearSelect',
                            'HardwarePersonalProjectsSelect','JobSearchResource','EmployerSearchMethod']

for col in listOfColsToOneHotEncode:
    labelEncoder = preprocessing.LabelEncoder()
    labelTfromed = labelEncoder.fit_transform(processedDF[col])
    oneHotEncoder = preprocessing.OneHotEncoder()
    oneHotTformed = oneHotEncoder.fit_transform(labelTfromed.reshape(-1,1))
    currOneHotDF = pd.DataFrame(oneHotTformed.todense(), columns = [col+'_OneHot_'+str(x) for x in range(len(labelEncoder.classes_))])
    numericDF = pd.concat((numericDF,currOneHotDF),axis=1)


# # Move on to the "Learning Platform Usefulness" Questions
# We first convert the string answers to numeric ordinal values and then apply dimentinality reduction using PCA

# In[ ]:


#%% add learning platform usefulness features to our numeric dataframe
def GetDictValueForKey(x):
    return answerToNumericalDict[x]

allLearningPlatformColumns = [q for q in allQuestions if q.find('LearningPlatformUsefulness') >= 0]
answerToNumericalDict = {'Not Useful':-1.0,'NaN':0.0,'Somewhat useful':1.0,'Very useful':2.0}

learningUsefulnessOrigDF = processedDF.loc[:,allLearningPlatformColumns]
learningUsefulnessOrigDF = learningUsefulnessOrigDF.applymap(GetDictValueForKey)

# compress cols to eliminate outliers and apply whitening using PCA
numComponents = 12
learningUsefulnessPCAModel = decomposition.PCA(n_components=numComponents,whiten=True)
learningUsefulnessFeatures = learningUsefulnessPCAModel.fit_transform(learningUsefulnessOrigDF)

explainedVarVec = learningUsefulnessPCAModel.explained_variance_ratio_
print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))

newColNames = ['learning_PCA_%d'%(x+1) for x in range(numComponents)]
learningUsefulnessDF = pd.DataFrame(data=learningUsefulnessFeatures, columns=newColNames)

importanceWeight = 0.5
numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*learningUsefulnessDF),axis=1)


# # Apply the same procedure on "Job Skill Importance" Questions

# In[ ]:


#%% add job skill imporance features to our numeric dataframe
allJobSkillColumns = [q for q in allQuestions if q.find('JobSkillImportance') >= 0] 
answerToNumericalDict = {'Unnecessary':-1.0,'NaN':0.0,'Nice to have':1.0,'Necessary':2.0}

jobSkillOrigDF = processedDF.loc[:,allJobSkillColumns]
jobSkillOrigDF = jobSkillOrigDF.applymap(GetDictValueForKey)

# compress cols to eliminate outliers and apply whitening using PCA
numComponents = 7
jobSkillPCAModel = decomposition.PCA(n_components=numComponents,whiten=True)
jobSkillFeatures = jobSkillPCAModel.fit_transform(jobSkillOrigDF)

explainedVarVec = jobSkillPCAModel.explained_variance_ratio_
print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))

newColNames = ['jobSkill_PCA_%d'%(x+1) for x in range(numComponents)]
jobSkillDF = pd.DataFrame(data=jobSkillFeatures, columns=newColNames)

importanceWeight = 0.5
numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*jobSkillDF),axis=1)


# # Apply the same procedure on "Work Tools and Methods" Questions

# In[ ]:


#%% add work tools and methods frequency features to our dataframe
allWorkToolsColumns = [q for q in allQuestions if q.find('WorkToolsFrequency') >= 0] 
allWorkMethodsColumns = [q for q in allQuestions if q.find('WorkMethodsFrequency') >= 0] 
answerToNumericalDict = {'NaN':0.0,'Rarely':1.0,'Sometimes':2.0,'Often':3.0,'Most of the time':4.0}

workToolsOrigDF = processedDF.loc[:,allWorkToolsColumns+allWorkMethodsColumns]
workToolsOrigDF = workToolsOrigDF.applymap(GetDictValueForKey)

# compress cols to eliminate outliers and apply whitening using PCA
numComponents = 38
workToolsPCAModel = decomposition.PCA(n_components=numComponents,whiten=True)
workToolsFeatures = workToolsPCAModel.fit_transform(workToolsOrigDF)

explainedVarVec = workToolsPCAModel.explained_variance_ratio_
print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))

newColNames = ['workTools_PCA_%d'%(x+1) for x in range(numComponents)]
workToolsDF = pd.DataFrame(data=workToolsFeatures, columns=newColNames)

importanceWeight = 0.5
numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*workToolsDF),axis=1)


# # Apply the same procedure on "Work Challanges" Questions

# In[ ]:


#%% add work challanges features to our dataframe
allWorkChallengesColumns = [q for q in allQuestions if q.find('WorkChallengeFrequency') >= 0]
answerToNumericalDict = {'NaN':0.0,'Rarely':1.0,'Sometimes':2.0,'Often':3.0,'Most of the time':4.0}

workChallangesOrigDF = processedDF.loc[:,allWorkChallengesColumns]
workChallangesOrigDF = workChallangesOrigDF.applymap(GetDictValueForKey)

# compress cols to eliminate outliers and apply whitening using PCA
numComponents = 16
workChallengesPCAModel = decomposition.PCA(n_components=numComponents,whiten=True)
workChallengesFeatures = workChallengesPCAModel.fit_transform(workChallangesOrigDF)

explainedVarVec = workChallengesPCAModel.explained_variance_ratio_
print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))

newColNames = ['workChallenges_PCA_%d'%(x+1) for x in range(numComponents)]
workChallengesDF = pd.DataFrame(data=workChallengesFeatures, columns=newColNames)

importanceWeight = 0.5
numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*workChallengesDF),axis=1)


# # Apply the same procedure on "Job Selection Factors" Questions

# In[ ]:


#%% add job selection factors features to our dataframe
allJobFactorsColumns = [q for q in allQuestions if q.find('JobFactor') >= 0] 
answerToNumericalDict = {'Not important':-1.0,'NaN':0.0,'Somewhat important':1.0,'Very Important':2.0}

jobPreferenceOrigDF = processedDF.loc[:,allJobFactorsColumns]
jobPreferenceOrigDF = jobPreferenceOrigDF.applymap(GetDictValueForKey)

# compress cols to eliminate outliers and apply whitening using PCA
numComponents = 10
jobPreferencePCAModel = decomposition.PCA(n_components=numComponents,whiten=True)
jobPreferenceFeatures = jobPreferencePCAModel.fit_transform(jobPreferenceOrigDF)

explainedVarVec = jobPreferencePCAModel.explained_variance_ratio_
print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))

newColNames = ['jobPreference_PCA_%d'%(x+1) for x in range(numComponents)]
jobPreferenceDF = pd.DataFrame(data=jobPreferenceFeatures, columns=newColNames)

importanceWeight = 0.5
numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*jobPreferenceDF),axis=1)


# # Apply the same procedure on "Time Allocation" Questions

# In[ ]:


#%% add time allocation distribution features to our dataframe
def ReplaceOnlyNaNs(x):
    if x == 'NaN':
        return 0.0
    else:
        return x

allTimeAllocationColumns = ['TimeGatheringData', 'TimeModelBuilding', 'TimeProduction', 'TimeVisualizing', 'TimeFindingInsights', 'TimeOtherSelect']
timeAllocationOrigDF = processedDF.loc[:,allTimeAllocationColumns]
timeAllocationOrigDF = timeAllocationOrigDF.applymap(ReplaceOnlyNaNs)

numComponents = 4
timeAllocationPCAModel = decomposition.PCA(n_components=numComponents,whiten=True)
timeAllocationFeatures = timeAllocationPCAModel.fit_transform(timeAllocationOrigDF)

explainedVarVec = timeAllocationPCAModel.explained_variance_ratio_
print('Total explained percent by PCA model with %d components is %.1f%s' %(numComponents, 100*explainedVarVec.sum(),'%'))

newColNames = ['timeAllocation_PCA_%d'%(x+1) for x in range(numComponents)]
timeAllocationeDF = pd.DataFrame(data=timeAllocationFeatures, columns=newColNames)

importanceWeight = 0.5
numericDF = pd.concat((numericDF, (importanceWeight/numComponents)*jobPreferenceDF),axis=1)


# # We now Finally have a Numeric Reperesentation of the Dataset
# On this representation we can apply kmeans to cluster the data.

# In[ ]:


numericDF.shape


# 
# # First, let's figure out how many clusters we should use?

# In[ ]:


#%% we now finally have a numeric representation of the dataset and we are ready to cluster the users
listOfNumClusters = [1,2,4,6,9,12,16,32,64,128,256]
listOfInertia = []
for numClusters in listOfNumClusters:
    KMeansModel = cluster.MiniBatchKMeans(n_clusters=numClusters, batch_size=2100, n_init=5, random_state=1)
    KMeansModel.fit(numericDF)
    listOfInertia.append(KMeansModel.inertia_)
explainedPercent = 100*(1-(np.array(listOfInertia)/listOfInertia[0]))

# plot the explained percent as a function of number of clusters
percentExplainedTarget = 40

numDesiredClusterInd = np.nonzero(explainedPercent > percentExplainedTarget)[0][0]
numDesiredClusters = listOfNumClusters[numDesiredClusterInd]

explainedPercentReached = explainedPercent[numDesiredClusterInd]
plt.figure(figsize=(14,6)); plt.plot(listOfNumClusters,explainedPercent,c='b')
plt.scatter(numDesiredClusters,explainedPercentReached,s=150,c='r')
plt.xlabel('Number of Clusters', fontsize=20); plt.ylabel('Explained Percent', fontsize=20)
plt.title('Desired Number of Clusters = %d, Explained Percent = %.2f%s' %(numDesiredClusters,explainedPercentReached,'%'),fontsize=22);
plt.xlim(-1,listOfNumClusters[-1]+1); plt.ylim(0,60);


# We can see here that we can't explain too much of the variance in the dataset, even with 256 clusters we only get to ~60% of variance explained.  
# This is really because each and everyone of us is a fairly "unique snowflake" that gives different answers to the survey questions, but we'll be practical and make due with what we have and take the smallest amount of clusters that capture the largest amount of variance.

# # Cluster the dataset with the selected number of clusters (12)

# In[ ]:


#%% for the selected number of clusters, redo the Kmeans and sort the clusters by frequency
KMeansModel = cluster.KMeans(n_clusters=numDesiredClusters, n_init=15, random_state=10)
KMeansModel.fit(numericDF)

clusterInds = KMeansModel.predict(numericDF)

clusterFrequency = []
for clusterInd in range(numDesiredClusters):
    clusterFrequency.append((clusterInds == clusterInd).sum()/float(len(clusterInds)))
clusterFrequency = np.array(clusterFrequency)
sortedClusterFrequency = np.flipud(np.sort(np.array(clusterFrequency)))
sortedClustersByFrequency = np.flipud(np.argsort(clusterFrequency))


# # Show a Subset of Question Responses by the 15 Nearest Neighbors of Kaggler Type no. 1:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 0
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 1:
# The most frequent kaggler (accounts for slightly less than 15% of kagglers) is a ~28 year old male from India, 
# he is employed full-time as a Software Developer or a Data Scientist, he has some industry experience, 
# he majored in Computer Science during University and holds a Master's degree. His annual salary is about 14K dollars a year. He is looking forward to learn about deep learning in the upcoming year and specifically using tensorflow.

# # Nearest Neighbors of Kaggler Type no. 2:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 1
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 2:
# The second most frequent kaggler (accounts for slightly less than 13% of kagglers) is a ~27 year old male from India, 
# he is employed full-time as a Software Developer, has little industry experience, he has majored in Computer Science during university, holds a Bachelor's degree.

# # Nearest Neighbors of Kaggler Type no. 3:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 2
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 3:
# The third most frequent kaggler (accounts for about 11% of kagglers) is a 34 year old male from the US, he is employed full-time, and doesn't really have the time to fill out an internet survey.

# # Nearest Neighbors of Kaggler Type no. 4:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 3
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 4:
# The forth most frequent kaggler (accounts for about 9% of kagglers) is a ~29 year old female from the Unites States, she is employed full-time as a data scientist, has 3-5 years of industry experience, her background is Computer Science and she holds a Master's degree. She didn't share her salary but we can infer that it's most likely an OK salary.

# # Nearest Neighbors of Kaggler Type no. 5:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 4
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 5:
# The fifth most frequent kaggler (accounts for about 8.5% of kagglers) is a ~30 year old male from the Unites States, he is employed full-time as a data scientist, has 3-5 years of industry experience, his background is Mathematics and statistics and he holds a Master's degree. He mainly works with python. His anual salary is 92K dollars. Next year, he wants to learn more about deep learning and specifically using tensorflow.

# # Nearest Neighbors of Kaggler Type no. 6:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 5
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 6:
# The 6th most frequent kaggler (accounts for about 8.5% of kagglers) is a ~22 year old male from the India, he is not employed, has no experience, he holds a Bachelor's degree majoring in Computer Science. He mainly works with python. He spends 2-10 hours each week learning Data Science. He's first exposure with data science is using online courses and he values kaggle competitions very high as a potential credential.

# # Nearest Neighbors of Kaggler Type no. 7:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 6
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 7:
# The 7th most frequent kaggler (accounts for about 8% of kagglers) is a ~44 year old male from the United States, he is employed full time at various different professions, has more than 10 years of experience, holds a Master's degree and majored in Computer Science. His first training is University (they didn't have online courses 20 years ago...). He mainly works with python. He's looking forward to gaining experience also with deep learning and tensorflow in the upcoming year.

# # Nearest Neighbors of Kaggler Type no. 8:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 7
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 8:
# The 8th most frequent kaggler (accounts for about 7.4% of kagglers) is a ~34 year old male from around the world, he is employed full time as a software developer, but has no datascience experience. he holds a Master's degree and majored in Computer Science. His first Data Science training is coming from online courses and he spends 2-10 hours each week learning data science. He working with python and like everyone else he's also looking forward to gaining experience also with deep learning in the upcoming year.

# # Nearest Neighbors of Kaggler Type no. 9:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 8
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 9:
# The 9th most frequent kaggler (accounts for about 5.8% of kagglers) is a ~30 year old male from around the world, he is not employed. He holds a Master's degree and comes from computer sceince and electrical engineering. His first Data Science training is coming from online courses and he spends most of his time learning Data Science. He working with a basic laptop and python. Like everyone else he's also looking forward to learning about deep learning in the upcoming year.

# # Nearest Neighbors of Kaggler Type no. 10:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 9
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 10:
# The 10th most frequent kaggler (accounts for about 5.6% of kagglers) is a ~36 year old male from the United States, he is employed full time and has more than 10 years of experience. He holds a Master's degree and comes from an eclectic background. His first training is coming from university courses and self teaching.

# # Nearest Neighbors of Kaggler Type no. 11:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 10
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 11:
# The 11th kaggler (accounts for about 4.6% of kagglers) is a ~40 year old male from the Unites States, he is employed full-time as a data scientist, has more than 10 years of industry experience, his background is diverse (CS, EE, Math) and he holds a Master's degree. He self taught himself data science, and mainly works with python. His anual salary is 181K dollars. Like everyone else, this experienced kaggler wants to learn more about deep learning and specifically using tensorflow.

# # Nearest Neighbors of Kaggler Type no. 12:

# In[ ]:


#%% show the attribures of most frequent kaggler
def GetMstCommonElement(a_list):
    return max(set(a_list), key=a_list.count)

# select cluster
k = 11
selectedCluster = sortedClustersByFrequency[k]

# find nearest neighbors
numNeighbors = 15
distFromCluster = KMeansModel.transform(numericDF)[:,selectedCluster]
distFromCluster[clusterInds != selectedCluster] = np.inf
nearestNeighborInds = np.argsort(distFromCluster)[:numNeighbors]


# In[ ]:


basicSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


additionalSubsetDF.loc[nearestNeighborInds,:]


# In[ ]:


#show original data for neighbors
print('-'*40)
print('Represents %.1f%s of kagglers' %(100.0*sortedClusterFrequency[k],'%'))
print('Average Age = %.1f' %(basicSubsetDF.loc[nearestNeighborInds,'Age'].astype(float).mean()))
print('Average Salary in USD = %.2fK' %(basicSubsetDF.loc[nearestNeighborInds,'annualSalary_USD'].astype(float).mean()))
print('Most Common Gender is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'GenderSelect'].tolist()))
print('Most Common Country is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Country'].tolist()))
print('Most Common Formal Education is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'FormalEducation'].tolist()))
print('Most Common Major is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'MajorSelect'].tolist()))
print('Most Common Employment Status is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'EmploymentStatus'].tolist()))
print('Most Common Tenure is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'Tenure'].tolist()))
print('Most Common Job title is "%s"' %GetMstCommonElement(basicSubsetDF.loc[nearestNeighborInds,'CurrentJobTitleSelect'].tolist()))
print('Most Common First training is "%s"' %GetMstCommonElement(additionalSubsetDF.loc[nearestNeighborInds,'FirstTrainingSelect'].tolist()))
print('-'*40)


# # Paragraph Summery of Kaggler Type no. 12:
# The 12th and final kaggler (accounts for about 4% of kagglers) is a ~26 year old female from the Unites States. She is unemployed, has little industry experience, her background is diverse (CS, EE) and she holds either a Bachelor's or a Master's degree. She learns data science around 2-10 hours a week, and mainly works with python and using a basic laptop. Like everyone else, she is interested to learn more about deep learning.
