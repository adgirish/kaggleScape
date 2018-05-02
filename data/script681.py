
# coding: utf-8

# This notebook is going to explore different convenient ways to train a model and make a submission. 
# 
# This is an easy method of testing different models. This model codifies the methodology presented in the following link into something that is easy to work with. 
# 
# https://www.kaggle.com/sudalairajkumar/two-sigma-financial-modeling/univariate-analysis-regression-lb-0-006

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNetCV, LinearRegression

import kagglegym


# In[ ]:


class fitModel():
    '''
        This class is going to take train values
        and a particular type of model and take care of
        the prediction step and wil contain a fit
        step. 
        
        Remember to send in a copy of train because
        there is a high chance that it will be modified
        
        the model is a sklearn model like ElasticNetCV
        
        all other parameters are passed onto the model
    '''
    
    def __init__(self, model, train, columns):

        # first save the model ...
        self.model   = model
        self.columns = columns
        
        # Get the X, and y values, 
        y = np.array(train.y)
        
        X = train[columns]
        self.xMeans = X.mean(axis=0) # Remember to save this value
        self.xStd   = X.std(axis=0)  # Remember to save this value

        X = np.array(X.fillna( self.xMeans ))
        X = (X - np.array(self.xMeans))/np.array(self.xStd)
        
        # fit the model
        self.model.fit(X, y)
        
        return
    
    def predict(self, features):
        '''
            This function is going to return the predicted
            value of the function that we are trying to 
            predict, given the observations. 
        '''
        X = features[self.columns]
        X = np.array(X.fillna( self.xMeans ))
        X = (X - np.array(self.xMeans))/np.array(self.xStd)

        return self.model.predict(X)


# In[ ]:


def checkModel(modelToUse, columns):
    '''
        This  function checks and makes sure that the 
        model provided is doing what it is supposed to
        do. This is a sanity check ...
    '''
    
    rewards = []
    env = kagglegym.make()
    observation = env.reset()
    
    train = observation.train
    
    # Just to make things easier to visualize
    # and also to speed things up ...
    # -----------------------------------------
    train   = train[['timestamp', 'y'] + columns]
    train   = train.groupby('timestamp').aggregate(np.mean)
    train.y = np.cumsum(train.y) # easier to visualize
    
    print('fitting a model')
    model = fitModel(modelToUse, train, columns)
    
    print('predict the same data')
    yHat = model.predict(train) # We already select required columns
    
    plt.figure()
    plt.plot(yHat, color='black', lw=2, label='predicted')
    plt.plot(train.y, '.', mec='None', mfc='orange', label='original')
    plt.legend(loc='lower right')
    
    return
    
columns = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']
checkModel(LinearRegression(), columns)
plt.title('four columns')
    
# Get all columns here 
env     = kagglegym.make()
allCols = env.reset().train.columns

checkModel(LinearRegression(), [c for c in allCols if 'fundamental' in c])
plt.title('fundamentals')

checkModel(LinearRegression(), [c for c in allCols if 'technical' in c])
plt.title('technicals')




# Now, we are going to start the model, and fit the model with an elastic net. 

# In[ ]:


def getScore(modelToUse, columns):
    
    print('Starting a new calculation for score')
    rewards = []
    env = kagglegym.make()
    observation = env.reset()
    
    print('fitting a model')
    model = fitModel(modelToUse, observation.train.copy(), columns)

    print('Starting to fit a model')
    while True:
        
        prediction  = model.predict(observation.features.copy())
        target      = observation.target
        target['y'] = prediction
        
        timestamp = observation.features["timestamp"][0]
        if timestamp % 100 == 0:
            print(timestamp)

        observation, reward, done, info = env.step(target)
        rewards.append(reward)
        if done: break
            
    return info['public_score'], rewards


# In[ ]:


columns = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']
getScore(ElasticNetCV(), columns)[0]

