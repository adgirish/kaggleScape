
# coding: utf-8

# # House price prediction using multiple regression analysis
# 
# # Part 2: Regression Models
# 
# The following notebook presents a thought process of predicting a continuous variable through Machine Learning methods. More specifically, we want to predict house prices based on multiple features using regression analysis. 
# 
# As an example, we will use a dataset of house sales in King County, where Seattle is located.
# 
# In the [first][1] part of the analysis, we set up the context using map visualization, and highlighted the association between the variables in our dataset. 
# 
# This is, for example, a map of King County showing the average house price per zipcode. We can see the disparities between the different zipcodes. The location of the houses should play an important role in our regression model. 
# 
# ![price per zipcode][2]
# 
# In this second notebook we will apply multiple regression models. We will talk about model complexity and how we can select the best predictive model using a validation set or cross-validation techniques.
# 
# ## 1. Preparation
# 
# As in Part 1, Let's first load the libraries and the dataset
# 
# 
#   [1]: https://www.kaggle.com/harlfoxem/d/harlfoxem/housesalesprediction/house-price-prediction-part-1/notebook
#   [2]: https://harlfoxem.github.io/img/King_County_House_Prediction_files/price.png

# In[ ]:


import numpy as np # NumPy is the fundamental package for scientific computing

import pandas as pd # Pandas is an easy-to-use data structures and data analysis tools
pd.set_option('display.max_columns', None) # To display all columns

import matplotlib.pyplot as plt # Matplotlib is a python 2D plotting library
get_ipython().run_line_magic('matplotlib', 'inline')
# A magic command that tells matplotlib to render figures as static images in the Notebook.

import seaborn as sns # Seaborn is a visualization library based on matplotlib (attractive statistical graphics).
sns.set_style('whitegrid') # One of the five seaborn themes
import warnings
warnings.filterwarnings('ignore') # To ignore some of seaborn warning msg

from scipy import stats

from sklearn import linear_model # Scikit learn library that implements generalized linear models
from sklearn import neighbors # provides functionality for unsupervised and supervised neighbors-based learning methods
from sklearn.metrics import mean_squared_error # Mean squared error regression loss
from sklearn import preprocessing # provides functions and classes to change raw feature vectors

from math import log


# In[ ]:


data = pd.read_csv("../input/kc_house_data.csv", parse_dates = ['date']) # load the data into a pandas dataframe
data.head(2) # Show the first 2 lines


# ### Data Cleaning
# 
# Let's reduce the dataset by dropping columns that won't be used during the analysis.

# In[ ]:


data.drop(['id', 'date'], axis = 1, inplace = True)


# ### Data Transformation
# 
# Following the correlation analysis in Part 1, let's create some new variables in our dataset. 

# In[ ]:


data['basement_present'] = data['sqft_basement'].apply(lambda x: 1 if x > 0 else 0) # Indicate whether there is a basement or not
data['renovated'] = data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) # 1 if the house has been renovated


# ### Encode categorical variables using dummies
# 
# A Dummy variable is an artificial variable created to represent an attribute with two or more distinct categories/levels. In this example, we will analyse *bedrooms* and *bathrooms* as continuous and therefore will encode the following:
# 
# * floors
# * view
# * condition and
# * grade

# In[ ]:


categorial_cols = ['floors', 'view', 'condition', 'grade']

for cc in categorial_cols:
    dummies = pd.get_dummies(data[cc], drop_first=False)
    dummies = dummies.add_prefix("{}#".format(cc))
    data.drop(cc, axis=1, inplace=True)
    data = data.join(dummies)


# We saw that zipcodes are also related to price. However, encoded all zipcodes will add 70 dummies variables. Instead, we will only encode the 6 most expensive zipcodes as shown in the map.

# In[ ]:


dummies_zipcodes = pd.get_dummies(data['zipcode'], drop_first=False)
dummies_zipcodes.reset_index(inplace=True)
dummies_zipcodes = dummies_zipcodes.add_prefix("{}#".format('zipcode'))
dummies_zipcodes = dummies_zipcodes[['zipcode#98004','zipcode#98102','zipcode#98109','zipcode#98112','zipcode#98039','zipcode#98040']]
data.drop('zipcode', axis=1, inplace=True)
data = data.join(dummies_zipcodes)

data.dtypes


# ### Split the data
# 
# We will split the dataframe into training and testing data using a 80%/20% ratio

# In[ ]:


from sklearn.cross_validation import train_test_split
train_data, test_data = train_test_split(data, train_size = 0.8, random_state = 10)


# ## 2. Regression Models
# 
# In this section, we will train numerous regression models on the train data (e.g., simple linear regression, lasso, nearest neighbor) and evaluate their performance using Root Mean Squared Error (RMSE) on the test data.
# 
# ### 2.1 Simple Linear Regression
# 
# Let's first predict house prices using simple (one input) linear regression.

# In[ ]:


# A function that take one input of the dataset and return the RMSE (of the test data), and the intercept and coefficient
def simple_linear_model(train, test, input_feature):
    regr = linear_model.LinearRegression() # Create a linear regression object
    regr.fit(train.as_matrix(columns = [input_feature]), train.as_matrix(columns = ['price'])) # Train the model
    RMSE = mean_squared_error(test.as_matrix(columns = ['price']), 
                              regr.predict(test.as_matrix(columns = [input_feature])))**0.5 # Calculate the RMSE on test data
    return RMSE, regr.intercept_[0], regr.coef_[0][0]


# Let's create a simple linear regression model using sqft_living as input and calculate the RMSE on the test data.

# In[ ]:


RMSE, w0, w1 = simple_linear_model(train_data, test_data, 'sqft_living')
print ('RMSE for sqft_living is: %s ' %RMSE)
print ('intercept is: %s' %w0)
print ('coefficient is: %s' %w1)


# Similarly, we can run the same test on all the features in the dataset and assess which one would be the best estimator of house price using just a single linear regression model.

# In[ ]:


input_list = data.columns.values.tolist() # list of column name
input_list.remove('price')
simple_linear_result = pd.DataFrame(columns = ['feature', 'RMSE', 'intercept', 'coefficient'])

# loop that calculate the RMSE of the test data for each input 
for p in input_list:
    RMSE, w1, w0 = simple_linear_model(train_data, test_data, p)
    simple_linear_result = simple_linear_result.append({'feature':p, 'RMSE':RMSE, 'intercept':w0, 'coefficient': w1}
                                                       ,ignore_index=True)
simple_linear_result.sort_values('RMSE').head(10) # display the 10 best estimators


# When using simple linear regression, sqft_living provides the smallest test error estimate of house price for the dataset considered.
# 
# ### 2.2 Multiple Regression
# 
# Now let's try to predict *price* using multiple features. We can modify the simple linear regression function above to take multiple features as input.

# In[ ]:


# A function that take multiple features as input and return the RMSE (of the test data), and the  intercept and coefficients
def multiple_regression_model(train, test, input_features):
    regr = linear_model.LinearRegression() # Create a linear regression object
    regr.fit(train.as_matrix(columns = input_features), train.as_matrix(columns = ['price'])) # Train the model
    RMSE = mean_squared_error(test.as_matrix(columns = ['price']), 
                              regr.predict(test.as_matrix(columns = input_features)))**0.5 # Calculate the RMSE on test data
    return RMSE, regr.intercept_[0], regr.coef_ 


# Let's try with a few examples:

# In[ ]:


print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 
                                                                             test_data, ['sqft_living','bathrooms','bedrooms']))
print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 
                                                                             test_data, ['sqft_above','view#0','bathrooms']))
print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 
                                                                             test_data, ['bathrooms','bedrooms']))
print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 
                                                                             test_data, ['view#0','grade#12','bedrooms','sqft_basement']))
print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 
                                                                             test_data, ['sqft_living','bathrooms','view#0']))


# We can also try to fit a higher-order polynomial on the input. For example, we can try to fit a qudratic function on sqft_living

# In[ ]:


train_data['sqft_living_squared'] = train_data['sqft_living'].apply(lambda x: x**2) # create a new column in train_data
test_data['sqft_living_squared'] = test_data['sqft_living'].apply(lambda x: x**2) # create a new column in test_data
print ('RMSE: %s, intercept: %s, coefficients: %s' %multiple_regression_model(train_data, 
                                                                             test_data, ['sqft_living','sqft_living_squared']))


# While we can get better performance than simple linear models, a few problems remain. 
# 
# * First, we don't know which feature to select. Obviously some combinations of features will yield smaller RMSE on the test set.
# * Second, we don't know how many features to select. This is because the more features we incorporate in the train model, the more overfit we get on the train data, resulting in higher error on the test data.
# 
# One solution would be to test multiple features combinations (all?) and keep the solution with the smallest error value calculated on the test data. However, this is an overly optimistic approach, since the model complexity is selected to minimize the test error (error is biased). A more sophisticated approach is to use two sets for testing our models, a.k.a: a validation set and a test set. We select model complexity to minimize error on the validation set and approximate the generalization error based on the test set.
# 
# Going through all subsets of features combinations is most often computationally infeasible. For example, having 30 features yield more than 1 billion combinations. Another approach is to use a greedy technique like a forward stepwise algorithm where the best estimator feature is added to the set of already selected features at each iteration. For example, let's pretend that the best single estimator is sqft_living. In the 2nd step of the greedy algorithm, we test all the remaining features one by one in combinations with sqft_living (e.g., sqft_living and bedrooms, sqft_living and waterfront, etc) and select the best combination using training error. At the end, we select the model complexity (number of features) using the validation error and estimate the generalization error using the test set.
# 
# Let's try this method.

# In[ ]:


# we're first going to add more features into the dataset.

# sqft_living cubed
train_data['sqft_living_cubed'] = train_data['sqft_living'].apply(lambda x: x**3) 
test_data['sqft_living_cubed'] = test_data['sqft_living'].apply(lambda x: x**3) 

# bedrooms_squared: this feature will mostly affect houses with many bedrooms.
train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2) 
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)

# bedrooms times bathrooms gives what's called an "interaction" feature. It is large when both of them are large.
train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']

# Taking the log of squarefeet has the effect of bringing large values closer together and spreading out small values.
train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))

train_data.shape


# In[ ]:


# split the train_data to include a validation set (train_data2 = 60%, validation_data = 20%, test_data = 20%)
train_data_2, validation_data = train_test_split(train_data, train_size = 0.75, random_state = 50)


# In[ ]:


# A function that take multiple features as input and return the RMSE (of the train and validation data)
def RMSE(train, validation, features, new_input):
    features_list = list(features)
    features_list.append(new_input)
    regr = linear_model.LinearRegression() # Create a linear regression object
    regr.fit(train.as_matrix(columns = features_list), train.as_matrix(columns = ['price'])) # Train the model
    RMSE_train = mean_squared_error(train.as_matrix(columns = ['price']), 
                              regr.predict(train.as_matrix(columns = features_list)))**0.5 # Calculate the RMSE on train data
    RMSE_validation = mean_squared_error(validation.as_matrix(columns = ['price']), 
                              regr.predict(validation.as_matrix(columns = features_list)))**0.5 # Calculate the RMSE on train data
    return RMSE_train, RMSE_validation 


# In[ ]:


input_list = train_data_2.columns.values.tolist() # list of column name
input_list.remove('price')

# list of features included in the regression model and the calculated train and validation errors (RMSE)
regression_greedy_algorithm = pd.DataFrame(columns = ['feature', 'train_error', 'validation_error'])  
i = 0
temp_list = []

# a while loop going through all the features in the dataframe
while i < len(train_data_2.columns)-1:
    
    # a temporary dataframe to select the best feature at each iteration
    temp = pd.DataFrame(columns = ['feature', 'train_error', 'validation_error'])
    
    # a for loop to test all the remaining features
    for p in input_list:
        RMSE_train, RMSE_validation = RMSE(train_data_2, validation_data, temp_list, p)
        temp = temp.append({'feature':p, 'train_error':RMSE_train, 'validation_error':RMSE_validation}, ignore_index=True)
        
    temp = temp.sort_values('train_error') # select the best feature using train error
    best = temp.iloc[0,0]
    temp_list.append(best)
    regression_greedy_algorithm = regression_greedy_algorithm.append({'feature': best, 
                                                  'train_error': temp.iloc[0,1], 'validation_error': temp.iloc[0,2]}, 
                                                 ignore_index=True) # add the feature to the dataframe
    input_list.remove(best) # remove the best feature from the list of available features
    i += 1
regression_greedy_algorithm


# We can see that the validation error is minimum when we reach 25 features in the model (condition#4). We stop the selection here even if the training error keeps getting smaller (overfitting).
# 
# Let's now calculate an estimation of the generalization error using test_data

# In[ ]:


greedy_algo_features_list = regression_greedy_algorithm['feature'].tolist()[:24] # select the first 30 features
test_error, _, _ = multiple_regression_model(train_data_2, test_data, greedy_algo_features_list)
print ('test error (RMSE) is: %s' %test_error)


# The test error is getting smaller.
# 
# * Note 1: We could have used k-fold cross validation instead of a validation set.
# * Note 2: Other greedy algorithms exist (e.g., backward stepwise, combining forward and backward steps)
# 
# Now, instead of searching over a discrete set of solutions using greedy algorithms, we can use another technique called regularization. We start with all possible features in the model and shrink the coefficients (weights). Two main regularization techniques exist, Ridge regression (a.k.a L2 regularization) and Lasso regression (a.k.a L1 regularization).
# 
# ### 2.3 Ridge Regression
# 
# Ridge regression aims to avoid overfitting by adding a cost to the Residual Sum of Squares (RSS) term of standard least squares that depends on the 2-norm of the coefficients.  The result is penalizing fits with large coefficients.  The strength of this penalty, and thus the fit vs. model complexity, is controlled by a parameter alpha (here called "L2_penalty").
# 
# Let's test two models using alpha equal 1 and 10.

# In[ ]:


input_feature = train_data.columns.values.tolist() # list of column name
input_feature.remove('price')

for i in [1,10]:
    ridge = linear_model.Ridge(alpha = i, normalize = True) # initialize the model
    ridge.fit(train_data.as_matrix(columns = input_feature), train_data.as_matrix(columns = ['price'])) # fit the train data
    print ('test error (RMSE) is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 
                              ridge.predict(test_data.as_matrix(columns = [input_feature])))**0.5) # predict price and test error


# Now the question is, how do we pick alpha to minimize the error?
# 
# Alpha is a measure of model complexity. So, as before, we could use a validation set to select it. However, that approach has a major disadvantage: it leaves fewer observations available for training. A better approach to overcome this issue is to use a cross-validation technique. It uses all of the training set in a smart way. k-fold cross-validation for example involves dividing the training set into k segments of roughly equal size. Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set. The major difference is that we repeat the process k times. We then compute the average of the k validation errors, and use it as an estimate of the generalization error. We then select the alpha value that generate the smallest validation error. The best approximation occurs for validation sets of size 1, where k is equal to the number of observations. It is called leave-one-out cross validation. It is however computationally intensive.
# 
# In this example we'll use a ridge regression with an implemented cross-validation method from the scikit learn library. By default, it performs Generalized Cross-Validation, which is a form of efficient Leave-One-Out cross-validation.

# In[ ]:


ridgeCV = linear_model.RidgeCV(alphas = np.linspace(1.0e-10,1,num = 100), normalize = True, store_cv_values = True) # initialize the model
ridgeCV.fit(train_data.as_matrix(columns = input_feature), train_data.as_matrix(columns = ['price'])) # fit the train data
print ('best alpha is: %s' %ridgeCV.alpha_) # get the best alpha
print ('test error (RMSE) is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 
                              ridgeCV.predict(test_data.as_matrix(columns = [input_feature])))**0.5) # predict price and test error


# Using every features in the dataset and a ridge regression model with an efficient  LOO cross-validation method yield a test error of 171567.
# 
# ### 2.4 Lasso Regression
# 
# Lasso regression jointly shrinks coefficients to avoid overfitting, and implicitly performs feature selection by setting some coefficients exactly to 0 for sufficiently large penalty strength alpha (here called "L1_penalty"). In particular, lasso takes the RSS term of standard least squares and adds a 1-norm cost of the coefficients.
# 
# Let 's train multiple models using different alpha values and asses the test error.

# In[ ]:


for i in [0.01,0.1,1,250,500,1000]:
    lasso = linear_model.Lasso(alpha = i, normalize = True) # initialize the model
    lasso.fit(train_data.as_matrix(columns = input_feature), train_data.as_matrix(columns = ['price'])) # fit the train data
    print (lasso.sparse_coef_.getnnz) # number of non zero weights
    print ('test error (RMSE) is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 
                              lasso.predict(test_data.as_matrix(columns = [input_feature])))**0.5) # predict price and test error


# As alpha increases, the number of features included in the model decrease. As in Ridge regression, we also can use cross-validation methods that select the best alpha to provide the best predictive accuracy. However, this technique tends to favor less sparse solutions and smaller alpha than optimal choice for feature selection.

# In[ ]:


lassoCV = linear_model.LassoCV(normalize = True) # initialize the model (alphas are set automatically)
lassoCV.fit(train_data.as_matrix(columns = input_feature), np.ravel(train_data.as_matrix(columns = ['price']))) # fit the train data
print ('best alpha is: %s' %lassoCV.alpha_) # get the best alpha
print ('number of non zero weigths is: %s' %np.count_nonzero(lassoCV.coef_)) # number of non zero weights
print ('test error (RMSE) is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 
                              lassoCV.predict(test_data.as_matrix(columns = [input_feature])))**0.5) # predict price and test error


# 39 features remain in the model, yielding a test error of 171369.
# 
# ### 2.5 k-Nearest Neighbors (NN) Regression
# 
# To finish, let's talk about another regression model, k-NN regression. The k-NN algorithm is used for estimating continuous variables. One such algorithm uses a weighted average of the k nearest neighbors, weighted by the inverse of their distance. It is the nonparamtric equivalent of ordinary least square regression.

# In[ ]:


# normalize the data
train_X = train_data.as_matrix(columns = input_feature)
scaler = preprocessing.StandardScaler().fit(train_X)
train_X_scaled = scaler.transform(train_X)
test_X = test_data.as_matrix(columns = [input_feature])
test_X_scaled = scaler.transform(test_X)

knn = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance') # initialize the model
knn.fit(train_X_scaled, train_data.as_matrix(columns = ['price'])) # fit the train data
print ('test error (RMSE) is: %s' %mean_squared_error(test_data.as_matrix(columns = ['price']), 
                              knn.predict(test_X_scaled))**0.5) # predict price and test error


# Similarly, the number of neighbor k is related to the complexity of the regression model. We can optimize the model accuracy by running cross-validation on the number of k-neighbors to include. 
# 
# ## Conclusion
# 
# In this notebook I tried to give on overview of regression methods using a dataset of house sales. Obviously, many other methods exit for regression (e.g., Elastic Net, kernel regression, Bayesian regression). Model selection should be based on your data and application.
