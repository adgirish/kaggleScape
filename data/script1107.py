
# coding: utf-8

# I suspect that the pups will be difficult to count on the images because they are dark grey just like the rocks, they are usually very close to their mother, and they are pretty similar in shape, size, and color to the seals (see 5.jpg for example). However, all other sea lion types are light brown so they should be easier to locate and count. I explore in this notebook whether it is possible to predict the number of pups based on the number of other sea lions. That is, the pups are not directly counted but their number is inferred/estimated based on the number of males, females, juveniles, and subadult males using a regression model. This alternative approach could turn out to be more accurate than directly counting the pups and I hope some of you will find it useful to improve the overall accuracy of the counts.

# In[ ]:


# load packages and read in the data

from subprocess import check_output
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib

data = np.genfromtxt('../input/Train/train.csv', delimiter=',',skip_header=1,usecols=(1,2,3,4,5))
X = data[:,:4]
Y = data[:,4]


# **Step 1)** It is probably relatively easy to count the light brown blobs on the image (the sum of non-pup sea lions). So the first step is to check if the number of pups correlate with the number of non-pup sea lions. 

# In[ ]:


blobs = np.sum(X,axis=1)

plt.scatter(blobs,Y)
plt.xlabel('#non-pup sea lions')
plt.ylabel('#pups')
plt.show()


# The correlation looks pretty weak because often the sea lion colony has zero or very few pups. 
# 
# **Step 2)** Develop a regression model and use the number of adult_males, subadult_males, adult_females, and juveniles as features. The idea is that the family demography of sea lion colonies might help to improve the prediction.  
# 
# The data is split into train (80%) and test (20%),  GridSearchCV is used on the training set to tune some xgboost parameters (learning rate and the number of trees only), and the test set is used to report RMSE and to save the feature importances. These steps (random splitting, grid search, RMSE, feature importances) are performed 10 times to assess the impact of random splitting on the RMSE. 
# 
# The prediction of the last model is visualized as well as the feature importances with their standard deviations.
# 

# In[ ]:


# a function to do the training and prediction
def train_pred(n_sims,X,Y,f_names,test_size):
    RMSE = np.zeros(n_sims)
    f_imp = np.zeros([n_sims,np.shape(X)[1]])

    for i in range(n_sims):

        # split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        # initialize XGBRegressor
        GB = xgb.XGBRegressor()

        # the parameter grid below was too much on the kaggle kernel
        #param_grid = {"learning_rate": [0.01,0.03,0.1],
        #              "objective": ['reg:linear'],
        #              "n_estimators": [300,1000,3000]}
        # do GridSearch
        #search_GB = GridSearchCV(GB,param_grid,cv=4,n_jobs=-1).fit(X_train,Y_train)
        # the best parameters should not be on the edges of the parameter grid
        #print('   ',search_GB.best_params_)
        # train the best model
        #xgb_pups = xgb.XGBRegressor(**search_GB.best_params_).fit(X_train, Y_train)

        # preselected parameters
        param_grid = {"learning_rate": 0.03,
                      "objective": 'reg:linear',
                      "n_estimators": 300}
        xgb_pups = xgb.XGBRegressor(**param_grid).fit(X_train, Y_train)

        # predict on the test set
        preds = xgb_pups.predict(X_test)

        # feature importance
        b = xgb_pups.booster()
        f_imp[i,:] = list(b.get_fscore().values())

        # rmse of prediction
        RMSE[i] = np.sqrt(mean_squared_error(Y_test, preds))
    
    # visualize the prediction of the last model
    plt.scatter(Y_test,preds,label = 'regression model')
    plt.plot(np.arange(np.max(Y_test)),np.arange(np.max(Y_test)),color='k',label='perfect prediction')
    plt.title('predictions of the last model')
    plt.legend(loc='best')
    plt.xlabel('true #pups')
    plt.ylabel('predicted #pups')
    plt.show()
    
    return RMSE, f_imp


# In[ ]:


f_names = ['adult males','subadult males','adult females','juveniles']
RMSE, f_imp = train_pred(10,X,Y,f_names,test_size=0.2)

print('RMSE = ',np.around(np.mean(RMSE),1),'+/-',np.around(np.std(RMSE),1))

plt.bar(range(len(f_names)),np.mean(f_imp,axis=0),width=0.8,yerr = np.std(f_imp,axis=0))
plt.ylabel('f score')
plt.xticks(range(len(f_names)), f_names)
plt.show()


# **Step 3)** Let's do some feature engineering! The sea lion counts are normalized by the total count, and the total count is added as an additional feature. 
# 
# As before, the prediction of the last model is visualized as well as the feature importances with their standard deviations.

# In[ ]:


X_sum = np.sum(X,axis=1)
X_new = np.hstack((X/X_sum[:,None],X_sum[:,None]))
f_names_new = np.append(f_names,'sum')

RMSE_new, f_imp_new = train_pred(10,X_new,Y,f_names_new,test_size=0.2)

print('RMSE = ',np.around(np.mean(RMSE_new),1),'+/-',np.around(np.std(RMSE_new),1))

plt.bar(range(len(f_names_new)),np.mean(f_imp_new,axis=0),width=0.8,yerr = np.std(f_imp_new,axis=0))
plt.ylabel('f score')
plt.xticks(range(len(f_names_new)), f_names_new,rotation='vertical')
plt.show()


# The RMSE did not improve significantly.
