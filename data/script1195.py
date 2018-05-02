
# coding: utf-8

# **This Ipython notebook is focused on analyzing missing values detection and multicollinearity in dataset**
# 
#  - Visualizing Datatype Of Variable 
#  - Missing Value Analysis
#  - Basic Feature Engineering From Timestamp
#  - Outlier Analysis  
#  - Univariate Analysis
#  - Top Contributing Features Using XGB
#  - Correlation Analysis
#  - Multicollinearity Analysis
#  - Bivariate Analysis (Price Doc vs Top    Contributing features)
#  - Simple Submission (0.313)

# **Library Imports**

# In[ ]:


import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# **Lets Read In Data**

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# **Shape Of Dataset**

# In[ ]:


train.shape


# **First Few Rows Of Dataset**

# In[ ]:


train.head(3).transpose()


# ##  Visualizing Datatypes of Variable

# In[ ]:


dataTypeDf = pd.DataFrame(train.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})
fig,ax = plt.subplots()
fig.set_size_inches(20,5)
sn.barplot(data=dataTypeDf,x="variableType",y="count",ax=ax)
ax.set(xlabel='Variable Type', ylabel='Count',title="Variables Count Across Datatype")


# # Missing Value Analysis

# In[ ]:


missingValueColumns = train.columns[train.isnull().any()].tolist()
msno.bar(train[missingValueColumns],            figsize=(20,8),color=(0.5, 0.5, 1),fontsize=12,labels=True,)


# In[ ]:


msno.matrix(train[missingValueColumns],width_ratios=(10,1),            figsize=(20,8),color=(0.5, 0.5, 1),fontsize=12,sparkline=True,labels=True)


# In[ ]:


msno.heatmap(train[missingValueColumns],figsize=(20,20))


# # Simple Feature Engineering From Timestamp

# In[ ]:


train['yearmonth'] = train['timestamp'].apply(lambda x: x[:4] + "-" + x[5:7])
train['year'] = train['timestamp'].apply(lambda x: x[:4])
train['month'] = train['timestamp'].apply(lambda x: x[5:7])

test['yearmonth'] = test['timestamp'].apply(lambda x: x[:4] + "-" + x[5:7])
test['year'] = test['timestamp'].apply(lambda x: x[:4])
test['month'] = test['timestamp'].apply(lambda x: x[5:7])


# # Outlier Analysis

# In[ ]:


fig,(ax1,ax2) = plt.subplots(ncols=2)
fig.set_size_inches(20,5)
train['yearmonth'] = train['timestamp'].apply(lambda x: x[:4] + "-" + x[5:7])
train['year'] = train['timestamp'].apply(lambda x: x[:4])
train['month'] = train['timestamp'].apply(lambda x: x[5:7])
sn.boxplot(data=train,y="price_doc",orient="v",ax=ax1)
sn.boxplot(data=train,x="price_doc",y="year",orient="h",ax=ax2)

fig1,ax3 = plt.subplots()
fig1.set_size_inches(20,5)
sn.boxplot(data=train,x="month",y="price_doc",orient="v",ax=ax3)
ax1.set(ylabel='Price Doc',title="Box Plot On Price Doc")
ax2.set(xlabel='Price Doc', ylabel='Year',title="Box Plot On Price Doc Across Year")
ax3.set(xlabel='Month', ylabel='Count',title="Box Plot On Price Doc Across Month")


# # Univariate Analysis #
# 
#  - Price Doc
#  - Build Year

# ## Price Doc Distribution##

# In[ ]:


fig,axes = plt.subplots(ncols=2)
fig.set_size_inches(20, 10)
stats.probplot(train["price_doc"], dist='norm', fit=True, plot=axes[0])
stats.probplot(np.log1p(train["price_doc"]), dist='norm', fit=True, plot=axes[1])


# ## Build Year ##

# In[ ]:


fig,ax= plt.subplots()
fig.set_size_inches(20,8)
trainBuild = train.dropna()
trainBuild["yearbuilt"] = trainBuild["build_year"].map(lambda x:str(x).split(".")[0])
sn.countplot(data=trainBuild,x="yearbuilt",ax=ax)
ax.set(xlabel='Build Year', ylabel='Count',title="No of Buildings Across Year",label='big')
plt.xticks(rotation=90)


# # Finding Top Contributing features 

# In[ ]:


from sklearn import model_selection, preprocessing
import xgboost as xgb

y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        
for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=100, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()


# In[ ]:


num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)


# In[ ]:


featureImportance = model.get_fscore()
features = pd.DataFrame()
features['features'] = featureImportance.keys()
features['importance'] = featureImportance.values()
features.sort_values(by=['importance'],ascending=False,inplace=True)
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
plt.xticks(rotation=60)
sn.barplot(data=features.head(30),x="features",y="importance",ax=ax,orient="v")


# # Correlation Analysis

# In[ ]:


topFeatures = features["features"].tolist()[:15]
topFeatures.append("price_doc")
corrMatt = train[topFeatures].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)


# ## Multicollinearity Analysis
# Most of the variables are highly collinear in nature. It is highly possbile to have multicollinearity problem in the dataset. For example variable such 0_13_all, 0_13_male, 0_13_female exhibit very collinearity. It is advisable to retain any one of them during model building. Since the following takes more than stipulated time. I have commented it out.

# Following code may take more time to run depends on the configuation of the system. So I have included the final image output of the analysis for reference

# ["Image For Multicollinearity Analysis"](https://github.com/viveksrinivasanss/sources/blob/master/multicollinearity.png)

# In[ ]:


#from statsmodels.stats.outliers_influence import variance_inflation_factor  

#def calculate_vif_(X):
#    variables = list(X.columns)
#    vif = {variable:variance_inflation_factor(exog=X.values, exog_idx=ix) for ix,variable in enumerate(list(X.columns))}
#    return vif

#numericalCol = []
#for f in train.columns:
#    if train[f].dtype!='object':
#        numericalCol.append(f)
#trainNA = train[numericalCol].dropna() 
#vifDict = calculate_vif_(trainNA)


# In[ ]:


# vifDf = pd.DataFrame()
# vifDf['variables'] = vifDict.keys()
# vifDf['vifScore'] = vifDict.values()
# vifDf.sort_values(by=['vifScore'],ascending=True,inplace=True)
# validVariables = vifDf[vifDf["vifScore"]<=5]
# variablesWithMC  = vifDf[vifDf["vifScore"]>5]

# fig,(ax1,ax2) = plt.subplots(ncols=2)
# fig.set_size_inches(20,15)
# sn.barplot(data=validVariables,x="vifScore",y="variables",ax=ax1,orient="h")
# sn.barplot(data=variablesWithMC.head(100),x="vifScore",y="variables",ax=ax2,orient="h")
# ax2.set(xlabel='Features', ylabel='VIF Score',title="Valid Variables Without Multicollinearity")
# ax2.set(xlabel='Features', ylabel='VIF Score',title="Variables Which Exhibit Multicollinearity")


# # Bivariate Analysis

# **Lets Us Understand Relationship Between Top Contributing Features And Price Doc**
# 
#  - full_sq
#  - life_sq
#  - floor
#  - max_floor
#  - build_year

# In[ ]:


for col in ["price_doc","full_sq","life_sq"]:
    ulimit = np.percentile(train[col].values, 99.5)
    llimit = np.percentile(train[col].values, 0.5)
    train[col].ix[train[col]>ulimit] = ulimit
    train[col].ix[train[col]<llimit] = llimit


# ## Full Square Vs Price Doc ##

# In[ ]:


plt.figure(figsize=(12,12))
sn.jointplot(x=np.log1p(train.full_sq.values), y=np.log1p(train.price_doc.values), size=10,kind="hex")
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of Total area in square metre', fontsize=12)
plt.show()


# ## Life Square Vs Price Doc ##

# In[ ]:


plt.figure(figsize=(12,12))
sn.jointplot(x=np.log1p(train.life_sq.values), y=np.log1p(train.price_doc.values), size=10,kind="hex")
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of living area in square metre', fontsize=12)
plt.show()


# ## Floor Vs Price Doc ##

# In[ ]:


fig,ax= plt.subplots()
fig.set_size_inches(20,8)
sn.boxplot(x="floor", y="price_doc", data=train,ax=ax)
ax.set(ylabel='Price Doc',xlabel="Floor",title="Floor Vs Price Doc")


# ## Max Floor Vs Price Doc ##

# In[ ]:


fig,ax= plt.subplots()
fig.set_size_inches(20,8)
sn.boxplot(x="max_floor", y="price_doc", data=train,ax=ax)
ax.set(ylabel='Price Doc',xlabel="Max Floor",title="Max Floor Vs Price Doc")
plt.xticks(rotation=90) 


# ## Build Year Vs Price Doc ##

# In[ ]:


fig,ax= plt.subplots()
fig.set_size_inches(20,8)
trainBuild = train.dropna()
trainBuild["yearbuilt"] = trainBuild["build_year"].map(lambda x:str(x).split(".")[0])
trainBuildGrouped = trainBuild.groupby(["yearbuilt"])["price_doc"].mean().to_frame().reset_index()
sn.pointplot(x=trainBuildGrouped["yearbuilt"], y=trainBuildGrouped["price_doc"], data=trainBuildGrouped, join=True,ax=ax,color="#34495e")
ax.set(xlabel='Build Year', ylabel='Price Doc',title="Average Price Doc Across Year",label='big')
plt.xticks(rotation=90)
plt.ylim([0,70000000])


# **Lets Do A Simple Submission**

# In[ ]:


y_predict = model.predict(dtest)
id_test = test.id
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()


# In[ ]:


output.to_csv('./naive_xgb_without_fe.csv', index=False)


# **More Analysis To Come. Kindly Refer Back**

# **KIndly Upvote If You Find It Useful**
