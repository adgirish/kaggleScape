
# coding: utf-8

# # DATA SCIENTIST
# **In this tutorial, I only explain you what you need to be a data scientist neither more nor less.**
# 
# Data scientist need to have these skills:
# 1. Basic Tools: Like python, R or SQL. You do not need to know everything. What you only need is to learn how to use **python**
# 1. Basic Statistics: Like mean, median or standart deviation. If you know basic statistics, you can use **python** easily. 
# 1. Data Munging: Working with messy and difficult data. Like a inconsistent date and string formatting. As you guess, **python** helps us.
# 1. Data Visualization: Title is actually explanatory. We will visualize the data with **python** like matplot and seaborn libraries.
# 1. Machine Learning: You do not need to understand math behind the machine learning technique. You only need is understanding basics of machine learning and learning how to implement it while using **python**.
# 
# ### As a summary we will learn python to be data scientist !!!
# ## For parts 1, 2, 3, 4, 5 and 6, look at DATA SCIENCE TUTORIAL for BEGINNERS
# https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners/
# ## In this tutorial, I am not going to learn machine learning to you, I am going to explain how to learn something by yourself.
# # *Confucius: Give a man a fish, and you feed him for a day. Teach a man to fish, and you feed him for a lifetime*
# **Content:**
# 1. Introduction to Python:
#     1. Matplotlib
#     1. Dictionaries 
#     1. Pandas
#     1. Logic, control flow and filtering
#     1. Loop data structures
# 1. Python Data Science Toolbox:
#     1. User defined function 
#     1. Scope
#     1. Nested function
#     1. Default and flexible arguments
#     1. Lambda function
#     1. Anonymous function
#     1. Iterators
#     1. List comprehension
# 1. Cleaning Data
#     1. Diagnose data for cleaning
#     1. Explotary data analysis
#     1. Visual exploratory data analysis
#     1. Tidy data
#     1. Pivoting data
#     1. Concatenating data
#     1. Data types
#     1. Missing data and testing with assert
# 1. Pandas Foundation
#     1. Review of pandas
#     1. Building data frames from scratch
#     1. Visual exploratory data analysis
#     1. Statistical explatory data analysis
#     1. Indexing pandas time series
#     1. Resampling pandas time series
# 1. Manipulating Data Frames with Pandas
#     1. Indexing data frames
#     1. Slicing data frames
#     1. Filtering data frames
#     1. Transforming data frames
#     1. Index objects and labeled data
#     1. Hierarchical indexing
#     1. Pivoting data frames
#     1. Stacking and unstacking data frames
#     1. Melting data frames
#     1. Categoricals and groupby
# 1. Data Visualization
#     1. Seaborn: https://www.kaggle.com/kanncaa1/seaborn-for-beginners
#     1. Bokeh: https://www.kaggle.com/kanncaa1/interactive-bokeh-tutorial-part-1
#     1. Bokeh: https://www.kaggle.com/kanncaa1/interactive-bokeh-tutorial-part-2
# 1. Statistical Thinking
#     1. coming soon
# 1. Machine Learning
#     1. Supervised Learning
#         1. EDA(Exploratory Data Analysis)
#         1. K-Nearest Neighbors (KNN)
#         1. Regression
#         1. Cross Validation (CV)
#         1. ROC Curve
#         1. Hyperparameter Tuning
#         1. Pre-procesing Data
#     1. Unsupervised Learning
#         1. Kmeans Clustering
#         1. Evaluation of Clustering
#         1. Standardization
#         1. Hierachy
#         1. T - Distributed Stochastic Neighbor Embedding (T - SNE)
#         1. Principle Component Analysis (PCA)
# 1. Deep Learning
#     1. https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners
# 1. Time Series Prediction
#     1. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read csv (comma separated value) into data
data = pd.read_csv('../input/column_2C_weka.csv')
print(plt.style.available) # look at available plot styles
plt.style.use('ggplot')


# # 8. MACHINE LEARNING (ML)
# In python there are some ML libraries like sklearn, keras or tensorflow. We will use sklearn.
# 

# ## A. SUPERVISED LEARNING
# * Supervised learning: It uses data that has labels. Example, there are orthopedic patients data that have labels *normal* and *abnormal*.
#     * There are features(predictor variable) and target variable. Features are like *pelvic radius* or *sacral slope*(If you have no idea what these are like me, you can look images in google like what I did :) )Target variables are labels *normal* and *abnormal*
#     * Aim is that as given features(input) predict whether target variable(output) is *normal* or *abnormal*
#     * Classification: target variable consists of categories like normal or abnormal
#     * Regression: target variable is continious like stock market
#     * If these explanations are not enough for you, just google them. However, be careful about terminology: features = predictor variable = independent variable = columns = inputs. target variable = responce variable = class = dependent variable = output = result

# ### EXPLORATORY DATA ANALYSIS (EDA)
# * In order to make something in data, as you know you need to explore data. Detailed exploratory data analysis is in my Data Science Tutorial for Beginners
# * I always start with *head()* to see features that are *pelvic_incidence,	pelvic_tilt numeric,	lumbar_lordosis_angle,	sacral_slope,	pelvic_radius* and 	*degree_spondylolisthesis* and target variable that is *class*
# * head(): default value of it shows first 5 rows(samples). If you want to see for example 100 rows just write head(100)
# 

# In[ ]:


# to see features and target variable
data.head()


# In[ ]:


# Well know question is is there any NaN value and length of this data so lets look at info
data.info()


# As you can see:
# * length: 310 (range index)
# * Features are float
# * Target variables are object that is like string
# * Okey we have some ideas about data but lets look go inside data deeper
#     * describe(): I explain it in previous tutorial so there is a Quiz :)
#         * Why we need to see statistics like mean, std, max or min? I hate from quizzes :) so answer: In order to visualize data, values should be closer each other. As you can see values looks like closer. At least there is no incompatible values like mean of one feature is 0.1 and other is 1000. Also there are another reasons that I will mention next parts.

# In[ ]:


data.describe()


# pd.plotting.scatter_matrix:
# * green: *normal* and red: *abnormal*
# * c:  color
# * figsize: figure size
# * diagonal: histohram of each features
# * alpha: opacity
# * s: size of marker
# * marker: marker type 

# In[ ]:


color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()


# Okay, as you understand in scatter matrix there are relations between each feature but how many *normal(green)* and *abnormal(red)* classes are there. 
# * Searborn library has *countplot()* that counts number of classes
# * Also you can print it with *value_counts()* method
# 
# <br> This data looks like balanced. Actually there is no definiton or numeric value of balanced data but this data is balanced enough for us.
# <br> Now lets learn first classification method KNN

# In[ ]:


sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()


# ###  K-NEAREST NEIGHBORS (KNN)
# * KNN: Look at the K closest labeled data points
# * Classification method.
# * First we need to train our data. Train = fit
# * fit(): fits the data, train the data.
# * predict(): predicts the data
# <br> If you do not understand what is KNN, look at youtube there are videos like 4-5 minutes. You can understand better with it.
# <br> Lets learn how to implement it with sklearn
# * x: features
# * y: target variables(normal, abnormal)
# * n_neighbors: K. In this example it is 3. it means that Look at the 3 closest labeled data points
# 

# In[ ]:


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
knn.fit(x,y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))


# * Well, we fit the data and predict it with KNN. 
# * So, do we predict correct or what is our accuracy or the accuracy is best metric to evaluate our result? Lets give answer of this questions
# <br> Measuring model performance:
# * Accuracy which is fraction of correct predictions is commonly used metric. We will use it know but there is another problem
# 
# <br>As you see I train data with x (features) and again predict the x(features). Yes you are reading right but yes you are right again it is absurd :)
# 
# <br>Therefore we need to split our data train and test sets.
# * train: use train set by fitting
# * test: make prediction on test set.
# * With train and test sets, fitted data and tested data are completely different
# * train_test_split(x,y,test_size = 0.3,random_state = 1)
#     * x: features
#     * y: target variables (normal,abnormal)
#     * test_size: percentage of test size. Example test_size = 0.3, test size = 30% and train size = 70%
#     * random_state: sets a seed. If this seed is same number, train_test_split() produce exact same split at each time
# * fit(x_train,y_train): fit on train sets
# * score(x_test,y_test)): predict and give accuracy on test sets

# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy


# Accuracy is 86% so is it good ? I do not know actually, we will see at the end of tutorial.
# <br> Now the question is why we choose K = 3 or what value we need to choose K. The answer is in model complexity
# 
# <br> Model complexity:
# * K has general name. It is called a hyperparameter. For now just know K is hyperparameter and we need to choose it that gives best performace. 
# * Literature says if k is small, model is complex model can lead to overfit. It means that model memorizes the train sets and cannot predict test set with good accuracy.
# * If k is big, model that is less complex model can lead to underfit. 
# * At below, I range K value from 1 to 25(exclude) and find accuracy for each K value. As you can see in plot, when K is 1 it memozize train sets and cannot give good accuracy on test set (overfit). Also if K is 18, model is lead to underfit. Again accuracy is not enough. However look at when K is 18(best performance), accuracy has highest value almost 88%. 
# 
# 

# In[ ]:


# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('k value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# ### Up to this point what you learn:
# * Supervised learning
# * Exploratory data analysis
# * KNN
#     * How to split data
#     * How to fit, predict data
#     * How to measure medel performance (accuracy)
#     * How to choose hyperparameter (K)
#     
# **<br> What happens if I chance the title KNN and make it some other classification technique like Random Forest?**
# * The answer is **nothing**. What you need to is just watch a video about what is random forest in youtube and implement what you learn in KNN. Because the idea and even most of the codes (only KNeighborsClassifier need to be RandomForestClassifier ) are same. You need to split, fit, predict your data and measue performance and choose hyperparameter of random forest(like max_depth). 

# ### REGRESSION
# * Supervised learning
# * We will learn linear and logistic regressions
# * This orthopedic patients data is not proper for regression so I only use two features that are *sacral_slope* and *pelvic_incidence* of abnormal 
#     * I  consider feature is pelvic_incidence and target is sacral_slope 
#     * Lets look at scatter plot so as to understand it better
#     * reshape(-1,1): If you do not use it shape of x or y becaomes (210,) and we cannot use it in sklearn, so we use shape(-1,1) and shape of x or y be (210, 1). 

# In[ ]:


# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable
data1 = data[data['class'] =='Abnormal']
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)
# Scatter
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()


# Now we have our data to make regression. In regression problems target value is continuously varying variable such as price of house or sacral_slope. Lets fit line into this points.
# 
# <br> Linear regression
# * y = ax + b       where  y = target, x = feature and a = parameter of model
# * We choose parameter of model(a) according to minimum error function that is lost function
# * In linear regression we use Ordinary Least Square (OLS) as lost function.
# * OLS: sum all residuals but some positive and negative residuals can cancel each other so we sum of square of residuals. It is called OLS
# * Score: Score uses R^2 method that is ((y_pred - y_mean)^2 )/(y_actual - y_mean)^2

# In[ ]:


# LinearRegression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# Predict space
predict_space = np.linspace(min(x), max(x)).reshape(-1,1)
# Fit
reg.fit(x,y)
# Predict
predicted = reg.predict(predict_space)
# R^2 
print('R^2 score: ',reg.score(x, y))
# Plot regression line and scatter
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()


# ### CROSS VALIDATION
# As you know in KNN method we use train test split with random_state that split exactly same at each time. However, if we do not use random_state, data is split differently at each time and according to split accuracy will be different. Therefore, we can conclude that model performance is dependent on train_test_split. For example you split, fit and predict data 5 times and accuracies are 0.89, 0.9, 0.91, 0.92 and 0.93, respectively. Which accuracy do you use? Do you know what accuracy will be at 6th times split, train and predict. The answer is I do not know but if I use cross validation I can find acceptable accuracy.
# <br> Cross Validation (CV)
# * K folds = K fold CV.
# * Look at this image it defines better than me :)
# * When K is increase, computationally cost is increase
# * cross_val_score(reg,x,y,cv=5): use reg(linear regression) with x and y that we define at above and K is 5. It means 5 times(split, train,predict)
# 

# In[ ]:


# CV
from sklearn.model_selection import cross_val_score
reg = LinearRegression()
k = 5
cv_result = cross_val_score(reg,x,y,cv=k) # uses R^2 as score 
print('CV Scores: ',cv_result)
print('CV scores average: ',np.sum(cv_result)/k)


# ### Regularized Regression
# As we learn linear regression choose parameters (coefficients) while minimizing lost function. If linear regression thinks that one of the feature is important, it gives high coefficient to this feature. However, this can cause overfitting that is like memorizing in KNN. In order to avoid overfitting, we use regularization that penalize large coefficients.
# 
# * Ridge regression: First regularization technique. Also it is called L2 regularization. 
#     * Ridge regression lost fuction = OLS + alpha * sum(parameter^2)
#     * alpha is parameter we need to choose to fit and predict. Picking alpha is similar to picking K in KNN. As you understand alpha is hyperparameter that we need to choose for best accuracy and model complexity. This process is called hyperparameter tuning.
#     * What if alpha is zero? lost function = OLS so that is linear rigression :)
#     * If alpha is small that can cause overfitting
#     * If alpha is big that can cause underfitting. But do not ask what is small and big. These can be change from problem to problem.
# * Lasso regression: Second regularization technique. Also it is called L1 regularization. 
#     * Lasso regression lost fuction = OLS + alpha * sum(absolute_value(parameter))
#     * It can be used to select important features od the data. Because features whose values are not shrinked to zero, is chosen by lasso regression
#     * In order to choose feature, I add new features in our regression data
#     
# <br> Linear vs Ridge vs Lasso
# First impression: Linear
# Feature Selection: 1.Lasso 2.Ridge
# Regression model: 1.Ridge 2.Lasso 3.Linear 
# 
# 

# In[ ]:


# Ridge
from sklearn.linear_model import Ridge
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2, test_size = 0.3)
ridge = Ridge(alpha = 0.1, normalize = True)
ridge.fit(x_train,y_train)
ridge_predict = ridge.predict(x_test)
print('Ridge score: ',ridge.score(x_test,y_test))


# In[ ]:


# Lasso
from sklearn.linear_model import Lasso
x = np.array(data1.loc[:,['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','pelvic_radius']])
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 3, test_size = 0.3)
lasso = Lasso(alpha = 0.1, normalize = True)
lasso.fit(x_train,y_train)
ridge_predict = lasso.predict(x_test)
print('Lasso score: ',lasso.score(x_test,y_test))
print('Lasso coefficients: ',lasso.coef_)


# As you can see *pelvic_incidence* and *pelvic_tilt numeric* are important features but others are not important
# 
# <br> Now lets discuss accuracy. Is it enough for measurement of model selection. For example, there is a data that includes 95% normal and 5% abnormal samples and our model uses accuracy for measurement metric. Then our model predict 100% normal for all samples and accuracy is 95% but it classify all abnormal samples wrong. Therefore we need to use confusion matrix as a model measurement matris in imbalance data.
# <br> While using confusion matrix lets use Random forest classifier to diversify classification methods.
# * tp = true positive(20), fp = false positive(7), fn = false negative(8), tn = true negative(58) 
# * tp = Prediction is positive(normal) and actual is positive(normal). 
# * fp = Prediction is positive(normal) and actual is negative(abnormal).
# * fn = Prediction is negative(abnormal) and actual is positive(normal).
# * tn = Prediction is negative(abnormal) and actual is negative(abnormal)
# * precision = tp / (tp+fp)
# * recall = tp / (tp+fn)
# * f1 = 2 * precision * recall / ( precision + recall)
# 
# 

# In[ ]:


# Confusion matrix with random forest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
rf = RandomForestClassifier(random_state = 4)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
print('Classification report: \n',classification_report(y_test,y_pred))


# In[ ]:


# visualize with seaborn library
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()


# ### ROC Curve with Logistic Regression 
# * logistic regression output is probabilities
# * If probability is higher than 0.5 data is labeled 1(abnormal) else 0(normal)
# * By default logistic regression threshold is 0.5
# * ROC is receiver operationg characteristic. In this curve x axis is false positive rate and y axis is true positive rate
# * If the curve in plot is closer to left-top corner, test is more accurate.
# * Roc curve score is auc that is computation area under the curve from prediction scores
# * We want auc to closer 1
# * fpr = False Positive Rate
# * tpr = True Positive Rate
# * If you want, I made ROC, Random forest and K fold CV in this tutorial. https://www.kaggle.com/kanncaa1/roc-curve-with-k-fold-cv/

# In[ ]:


# ROC Curve with logistic regression
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
# abnormal = 1 and normal = 0
data['class_binary'] = [1 if i == 'Abnormal' else 0 for i in data.loc[:,'class']]
x,y = data.loc[:,(data.columns != 'class') & (data.columns != 'class_binary')], data.loc[:,'class_binary']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred_prob = logreg.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()


# ### HYPERPARAMETER TUNING
# As I mention at KNN there are hyperparameters that are need to be tuned
# * For example: 
#     * k at KNN
#     * alpha at Ridge and Lasso
#     * Random forest parameters like max_depth
#     * linear regression parameters(coefficients)
# * Hyperparameter tuning: 
#     * try all of combinations of different parameters
#     * fit all of them
#     * measure prediction performance
#     * see how well each performs
#     * finally choose best hyperparameters
# * This process is most difficult part of this tutorial. Because we will write a lot of for loops to iterate all combinations. Just I am kidding sorry for this :) (We actually did it at KNN part)
# * We only need is one line code that is GridSearchCV
#     * grid: K is from 1 to 50(exclude)
#     * GridSearchCV takes knn and grid and makes grid search. It means combination of all hyperparameters. Here it is k.
# 

# In[ ]:


# grid search cross validation with 1 hyperparameter
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
knn_cv.fit(x,y)# Fit

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))


# Other grid search example with 2 hyperparameter
# * First hyperparameter is C:logistic regression regularization parameter
#     * If C is high: overfit
#     * If C is low: underfit
# * Second hyperparameter is penalty(lost function): l1 (Lasso) or l2(Ridge) as we learnt at linear regression part.

# In[ ]:


# grid search cross validation with 2 hyperparameter
# 1. hyperparameter is C:logistic regression regularization parameter
# 2. penalty l1 or l2
# Hyperparameter grid
param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 12)
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,param_grid,cv=3)
logreg_cv.fit(x_train,y_train)

# Print the optimal parameters and best score
print("Tuned hyperparameters : {}".format(logreg_cv.best_params_))
print("Best Accuracy: {}".format(logreg_cv.best_score_))


# ### PRE-PROCESSING DATA
# * In real life data can include objects or categorical data in order to use them in sklearn we need to encode them into numerical data
# * In data, class is *abnormal* and *normal*. Lets convert them into numeric value (actually I did it in logistic regression part with different method)
# * 2 different feature is created with the name *class_Abnormal* and *class_Normal*
# * However we need to drop one of the column because they are duplicated

# In[ ]:


# Load data
data = pd.read_csv('../input/column_2C_weka.csv')
# get_dummies
df = pd.get_dummies(data)
df.head(10)


# In[ ]:


# drop one of the feature
df.drop("class_Normal",axis = 1, inplace = True) 
df.head(10)
# instead of two steps we can make it with one step pd.get_dummies(data,drop_first = True)


# Other preprocessing step is centering, scaling or normalizing 
# * If you listen my advice and watch KNN in youtube, you have noticed that KNN uses form of distance for classificaiton like some oher methods. Therefore, we need to scale data. For this reason, we use 
#     * standardization: ( x - x.mean) / x.variance   or   x - x.min / x.range
# * pipeline: The purpose of the pipeline is to assemble several steps like svm(classifier) and  standardization(pre-processing)
# * How we create parameters name: for example SVM_ _C :  stepName__parameterName
# * Then grid search to find best parameters

# In[ ]:


# SVM, pre-process and pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 1)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x_train,y_train)

y_pred = cv.predict(x_test)

print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))


# ## UNSUPERVISED LEARNING
# * Unsupervised learning: It uses data that has unlabeled and uncover hidden patterns from unlabeled data. Example, there are orthopedic patients data that do not have labels. You do not know which orthopedic patient is normal or abnormal.
# * As you know orthopedic patients data is labeled (supervised) data. It has target variables. In order to work on unsupervised learning, lets drop target variables and to visualize just consider *pelvic_radius* and *degree_spondylolisthesis*
# 

# ### KMEANS
# * Lets try our first unsupervised method that is KMeans Cluster
# * KMeans Cluster: The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity
# * KMeans(n_clusters = 2): n_clusters = 2 means that create 2 cluster

# In[ ]:


# As you can see there is no labels in data
data = pd.read_csv('../input/column_2C_weka.csv')
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()


# In[ ]:


# KMeans Clustering
data2 = data.loc[:,['degree_spondylolisthesis','pelvic_radius']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data2)
labels = kmeans.predict(data2)
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = labels)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()


# ### EVALUATING OF CLUSTERING
# We cluster data in two groups. Okey well is that correct clustering? In order to evaluate clustering we will use cross tabulation table.
# * There are two clusters that are *0* and *1* 
# * First class *0* includes 138 abnormal and 100 normal patients
# * Second class *1* includes 72 abnormal and 0 normal patiens
# *The majority of two clusters are abnormal patients.

# In[ ]:


# cross tabulation table
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# The new question is that we know how many class data includes, but what if number of class is unknow in data. This is kind of like hyperparameter in KNN or regressions. 
# * inertia: how spread out the clusters are distance from each sample
# * lower inertia means more clusters
# * What is the best number of clusters ?
#     *There are low inertia and not too many cluster trade off so we can choose elbow

# In[ ]:


# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()


# ### STANDARDIZATION
# * Standardizaton is important for both supervised and unsupervised learning
# * Do not forget standardization as pre-processing
# * As we already have visualized data so you got the idea. Now we can use all features for clustering.
# * We can use pipeline like supervised learning.

# In[ ]:


data = pd.read_csv('../input/column_2C_weka.csv')
data3 = data.drop('class',axis = 1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kmeans)
pipe.fit(data3)
labels = pipe.predict(data3)
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# ### HIERARCHY
# * vertical lines are clusters
# * height on dendogram: distance between merging cluster
# * method= 'single' : closest points of clusters

# In[ ]:


from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(data3.iloc[200:220,:],method = 'single')
dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)
plt.show()


#  ### T - Distributed Stochastic Neighbor Embedding (T - SNE)
#  * learning rate: 50-200 in normal
#  * fit_transform: it is both fit and transform. t-sne has only have fit_transform
#  * Varieties have same position relative to one another

# In[ ]:


from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(data2)
x = transformed[:,0]
y = transformed[:,1]
plt.scatter(x,y,c = color_list )
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()


# ### PRINCIPLE COMPONENT ANALYSIS (PCA)
# * Fundemental dimension reduction technique
# * first step is decorrelation:
#     * rotates data samples to be aligned with axes
#     * shifts data asmples so they have mean zero
#     * no information lost
#     * fit() : learn how to shift samples
#     * transform(): apply the learned transformation. It can also be applies test data
# * Resulting PCA features are not linearly correlated
# * Principle components: directions of variance

# In[ ]:


# PCA
from sklearn.decomposition import PCA
model = PCA()
model.fit(data3)
transformed = model.transform(data3)
print('Principle components: ',model.components_)


# In[ ]:


# PCA variance
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler,pca)
pipeline.fit(data3)

plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()


# * Second step: intrinsic dimension: number of feature needed to approximate the data essential idea behind dimension reduction
# * PCA identifies intrinsic dimension when samples have any number of features
# * intrinsic dimension = number of PCA feature with significant variance
# * In order to choose intrinsic dimension try all of them and find best accuracy

# In[ ]:


# apply PCA
pca = PCA(n_components = 2)
pca.fit(data3)
transformed = pca.transform(data3)
x = transformed[:,0]
y = transformed[:,1]
plt.scatter(x,y,c = color_list)
plt.show()


# # CONCLUSION
# This is the end of DATA SCIENCE tutorial. The first part is here:
# <br>  https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners/
# <br>**If you have any question or suggest, I will be happy to hear it.**
