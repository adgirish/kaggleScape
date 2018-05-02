
# coding: utf-8

# I found kagglegym_emulation to be very helpfull (https://www.kaggle.com/slothouber/two-sigma-financial-modeling/kagglegym-emulation). What this script does is validating it against the actual kagglegym. I used some snippets from this script https://www.kaggle.com/sankhamukherjee/two-sigma-financial-modeling/prediction-model-elastic-net. 
# 
# Vote up if you find it meaningful :)

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNetCV
import kagglegym
import math


# In[ ]:


# kagglegym_emulation code
def r_score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    r = np.sign(r2) * np.sqrt(np.abs(r2))
    return max(-1, r)


class Observation(object):
    def __init__(self, train, target, features):
        self.train = train
        self.target = target
        self.features = features


class Environment(object):
    def __init__(self):
        with pd.HDFStore("../input/train.h5", "r") as hfdata:
            self.timestamp = 0
            fullset = hfdata.get("train")
            self.unique_timestamp = fullset["timestamp"].unique()
            # Get a list of unique timestamps
            # use the first half for training and
            # the second half for the test set
            n = len(self.unique_timestamp)
            i = int(n/2)
            timesplit = self.unique_timestamp[i]
            self.n = n
            self.unique_idx = i
            self.train = fullset[fullset.timestamp < timesplit]
            self.test = fullset[fullset.timestamp >= timesplit]

            # Needed to compute final score
            self.full = self.test.loc[:, ['timestamp', 'y']]
            self.full['y_hat'] = 0.0
            self.temp_test_y = None

    def reset(self):
        timesplit = self.unique_timestamp[self.unique_idx]

        self.unique_idx = int(self.n / 2)
        self.unique_idx += 1
        subset = self.test[self.test.timestamp == timesplit]

        # reset index to conform to how kagglegym works
        target = subset.loc[:, ['id', 'y']].reset_index(drop=True)
        self.temp_test_y = target['y']

        target.loc[:, 'y'] = 0.0  # set the prediction column to zero

        # changed bounds to 0:110 from 1:111 to mimic the behavior
        # of api for feature
        features = subset.iloc[:, :110].reset_index(drop=True)

        observation = Observation(self.train, target, features)
        return observation

    def step(self, target):
        timesplit = self.unique_timestamp[self.unique_idx-1]
        # Since full and target have a different index we need
        # to do a _values trick here to get the assignment working
        y_hat = target.loc[:, ['y']]
        self.full.loc[self.full.timestamp == timesplit, ['y_hat']] = y_hat._values

        if self.unique_idx == self.n:
            done = True
            observation = None
            reward = r_score(self.temp_test_y, target.loc[:, 'y'])
            score = r_score(self.full['y'], self.full['y_hat'])
            info = {'public_score': score}
        else:
            reward = r_score(self.temp_test_y, target.loc[:, 'y'])
            done = False
            info = {}
            timesplit = self.unique_timestamp[self.unique_idx]
            self.unique_idx += 1
            subset = self.test[self.test.timestamp == timesplit]

            # reset index to conform to how kagglegym works
            target = subset.loc[:, ['id', 'y']].reset_index(drop=True)
            self.temp_test_y = target['y']

            # set the prediction column to zero
            target.loc[:, 'y'] = 0

            # column bound change on the subset
            # reset index to conform to how kagglegym works
            features = subset.iloc[:, 0:110].reset_index(drop=True)

            observation = Observation(self.train, target, features)

        return observation, reward, done, info

    def __str__(self):
        return "Environment()"


def make():
    return Environment()


# In[ ]:


# predictive model wrapper, also see https://www.kaggle.com/sankhamukherjee/two-sigma-financial-modeling/prediction-model-elastic-net
class fitModel():
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
        X = features[self.columns]
        X = np.array(X.fillna( self.xMeans ))
        X = (X - np.array(self.xMeans))/np.array(self.xStd)

        return self.model.predict(X)


# In[ ]:


def list_match(list_a, list_b):
    for i, j in zip(list_a, list_b):
        if i != j:
            return False
    return True


# In[ ]:


# Validaiton of kagglegym_emulation
env = kagglegym.make()
env_test = make()

# Check observations
observation = env.reset()
observation_test = env_test.reset()
assert list_match(observation.train.id.values, observation_test.train.id.values)    
    
elastic_net = ElasticNetCV()
columns = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']
model = fitModel(elastic_net, observation.train.copy(), columns)
model_test = fitModel(elastic_net, observation_test.train.copy(), columns)

while True:
        
    prediction       = model.predict(observation.features.copy())
    prediction_test  = model_test.predict(observation_test.features.copy())
    
    assert list_match(prediction, prediction_test)
  
    
    target           = observation.target
    target_test      = observation_test.target
    target['y'] = prediction
    target_test['y'] = prediction_test
        
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print(timestamp)
    
    observation, reward, done, info = env.step(target)
    observation_test, reward_test, done_test, info_test = env_test.step(target)
    

    assert done == done_test
    assert math.isclose(reward, reward_test, abs_tol=5e-05)
    

    if done: 
        assert math.isclose(info['public_score'],info_test['public_score'],  abs_tol=1e-07)
        print('Info:',info['public_score'],'Info-test:',info_test['public_score'])
        break


# **VALIDATED SUCCESSFULLY !!!**
