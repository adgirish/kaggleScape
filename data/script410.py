
# coding: utf-8

# This projects highcardinality parameters onto the median first then uses Genetic Programming to predict the outputs.  I suggest you use XGB or any other favourite predictor - there is nothing stopping you adding some of the GP features to your models though.

# In[ ]:


import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[ ]:


def ProjectOnMedian(data1, data2, columnName):
    grpOutcomes = data1.groupby(list([columnName]))['y'].median().reset_index()
    grpCount = data1.groupby(list([columnName]))['y'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.y
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['y'].values
    x = pd.merge(data2[[columnName, 'y']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=list([columnName]),
                 left_index=True)['y']

    
    return x.values


# In[ ]:


directory = '../input/'
train = pd.read_csv(directory+'train_2016_v2.csv')
sample = pd.read_csv(directory+'sample_submission.csv')
properties = pd.read_csv(directory+'properties_2016.csv')


# In[ ]:


properties.hashottuborspa = properties.hashottuborspa.astype(str)
properties.fireplaceflag = properties.fireplaceflag.astype(str)
for c in properties.columns:
    if properties[c].dtype == 'object':
        print(c)
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))


# In[ ]:


highcardinality = ['airconditioningtypeid',
                   'architecturalstyletypeid',
                   'buildingclasstypeid',
                   'buildingqualitytypeid',
                   'decktypeid',
                   'fips',
                   'hashottuborspa',
                   'heatingorsystemtypeid',
                   'pooltypeid10',
                   'pooltypeid2',
                   'pooltypeid7',
                   'propertycountylandusecode',
                   'propertylandusetypeid',
                   'regionidcity',
                   'regionidcounty',
                   'regionidneighborhood',
                   'regionidzip',
                   'storytypeid',
                   'typeconstructiontypeid',
                   'fireplaceflag',
                   'taxdelinquencyflag']


# In[ ]:


sample = sample.rename(columns={'ParcelId':'parcelid'})
sample.head()


# In[ ]:


train = train.merge(properties, how='left', on='parcelid')
test = sample.merge(properties, how='left', on='parcelid')
train['month'] = pd.DatetimeIndex(train['transactiondate']).month
test['month'] = -1
logerrors = train.logerror.ravel()
train.drop(['logerror','transactiondate'],inplace=True,axis=1)


test = test[train.columns]
train.insert(1,'nans',train.isnull().sum(axis=1))
test.insert(1,'nans',train.isnull().sum(axis=1))
train['y'] = logerrors
test['y'] = np.nan


# Now to project highcardinality parameters to medians

# In[ ]:


from sklearn.model_selection import KFold
blindloodata = None
folds = 20
kf = KFold(n_splits=folds,shuffle=True,random_state=42)
for i, (train_index, test_index) in enumerate(kf.split(range(train.shape[0]))):
    print('Fold:',i)
    blindtrain = train.loc[test_index].copy() 
    vistrain = train.loc[train_index].copy()



    for c in highcardinality:
        blindtrain.insert(1,'loo'+c, ProjectOnMedian(vistrain,
                                                     blindtrain,c))
    if(blindloodata is None):
        blindloodata = blindtrain.copy()
    else:
        blindloodata = pd.concat([blindloodata,blindtrain])

for c in highcardinality:
    test.insert(1,'loo'+c, ProjectOnMedian(train,
                                           test,c))
test.drop(highcardinality,inplace=True,axis=1)

train = blindloodata
train.drop(highcardinality,inplace=True,axis=1)


# Fill in Nans with Median Values

# In[ ]:


feats = train.columns[1:-1]
for c in feats:
    train[c] = train[c].fillna(train[c].median())
    test[c] = test[c].fillna(train[c].median())


# Remove outliers from training and convert to a simple logistic regression problem

# In[ ]:


xtrain = train.loc[(train.y>-0.418)&(train.y< 0.418)].copy()
xtrain = xtrain.reset_index(drop=True)
features = xtrain.columns[1:-1]
xtrain.y = xtrain.y+.418
xtrain.y /= (2*.418)


# I scale the parameters to stop genetic programming from having to find a good scaling as well as a good prediction

# In[ ]:


ss = StandardScaler()
ss.fit(pd.concat([xtrain[features],test[features]]))
xtrain[features] = ss.transform(xtrain[features])
test[features] = ss.transform(test[features])


# The following models were found using Genetic Programming - just train your models on xtrain and you will be good to go

# In[ ]:


def Outputs(p):
    return 1./(1.+np.exp(-p))


def GP1(data):
    v = pd.DataFrame()
    v["i0"] = 0.050000*np.tanh((data["calculatedfinishedsquarefeet"] + (data["loopropertycountylandusecode"] + (data["loopropertycountylandusecode"] + (((1.324320 + data["looregionidneighborhood"]) + data["looregionidcity"])/2.0)))))
    v["i1"] = 0.020220*np.tanh((data["loopropertycountylandusecode"] + ((((data["looregionidzip"] + (data["finishedsquarefeet12"] * data["finishedsquarefeet12"]))/2.0) + ((data["lootaxdelinquencyflag"] + data["loohashottuborspa"])/2.0))/2.0)))
    v["i2"] = 0.046480*np.tanh(((data["loopropertycountylandusecode"] + ((data["taxvaluedollarcnt"] - data["taxamount"]) + np.tanh((data["taxamount"] * data["taxamount"])))) * 2.0))
    v["i3"] = 0.050000*np.tanh(((((((data["looregionidzip"] + (-(data["censustractandblock"])))/2.0) - data["loopooltypeid2"]) / 2.0) + (data["calculatedfinishedsquarefeet"] - data["taxamount"]))/2.0))
    v["i4"] = 0.050000*np.tanh(np.tanh(np.tanh(((((-(data["loohashottuborspa"])) * ((data["loopooltypeid7"] + data["finishedsquarefeet50"])/2.0)) * 2.0) - data["loopooltypeid10"]))))
    v["i5"] = 0.050000*np.tanh(((((data["unitcnt"] * data["latitude"]) * data["unitcnt"]) - data["loodecktypeid"]) + (data["latitude"] * data["loopropertycountylandusecode"])))
    v["i6"] = 0.043490*np.tanh(np.tanh(((data["lootaxdelinquencyflag"] + (data["lootypeconstructiontypeid"] * (data["loofips"] - (data["month"] * data["month"])))) * 2.0)))
    v["i7"] = 0.049950*np.tanh(((((data["calculatedfinishedsquarefeet"] + (-(np.tanh(data["bedroomcnt"]))))/2.0) + (data["looregionidcity"] * (data["calculatedfinishedsquarefeet"] / 2.0)))/2.0))
    v["i8"] = 0.050000*np.tanh(((-((((data["taxamount"] + data["unitcnt"]) + (((-(data["taxamount"])) * data["taxamount"]) / 2.0))/2.0))) / 2.0))
    v["i9"] = 0.050000*np.tanh(((data["loopropertycountylandusecode"] + ((((data["loopropertycountylandusecode"] + np.tanh(np.tanh(data["nans"])))/2.0) + np.tanh((-(data["looairconditioningtypeid"]))))/2.0))/2.0))
    v["i10"] = 0.050000*np.tanh(((data["looregionidzip"] + ((data["bathroomcnt"] - data["structuretaxvaluedollarcnt"]) + (-(data["propertyzoningdesc"])))) * 0.051282))
    v["i11"] = 0.041230*np.tanh((np.tanh(((data["taxamount"] * data["lotsizesquarefeet"]) * 2.0)) + ((data["taxamount"] * data["lotsizesquarefeet"]) * data["finishedsquarefeet12"])))
    v["i12"] = 0.047560*np.tanh((data["looregionidcity"] * (-((data["loopropertycountylandusecode"] - ((data["looregionidneighborhood"] + (-((data["unitcnt"] * data["looregionidcity"]))))/2.0))))))
    v["i13"] = 0.046070*np.tanh((0.907692 - np.tanh(((((-(0.907692)) + ((0.907692 + data["month"])/2.0))/2.0) * data["month"]))))
    v["i14"] = 0.042910*np.tanh(((data["loopropertycountylandusecode"] * ((data["loobuildingqualitytypeid"] - data["calculatedfinishedsquarefeet"]) - np.tanh((data["landtaxvaluedollarcnt"] + data["censustractandblock"])))) * 2.0))
    v["i15"] = 0.042240*np.tanh((0.051282 * ((-(3.0)) + (data["finishedsquarefeet12"] + ((-(3.0)) + data["month"])))))
    v["i16"] = 0.045380*np.tanh((((data["roomcnt"] * (data["finishedsquarefeet6"] - (data["poolsizesum"] * (data["poolsizesum"] * data["garagetotalsqft"])))) + 0.051282)/2.0))
    v["i17"] = 0.050000*np.tanh(((data["looregionidcity"] + ((((-(data["calculatedbathnbr"])) + data["lootaxdelinquencyflag"])/2.0) + np.tanh(data["taxvaluedollarcnt"]))) * data["unitcnt"]))
    v["i18"] = 0.050000*np.tanh((data["finishedsquarefeet15"] * (((data["finishedsquarefeet15"] + np.tanh(data["looregionidneighborhood"])) + data["garagetotalsqft"]) - data["censustractandblock"])))
    v["i19"] = 0.047800*np.tanh((0.051282 * (((data["finishedsquarefeet12"] - data["looarchitecturalstyletypeid"]) + (data["looarchitecturalstyletypeid"] * data["looarchitecturalstyletypeid"])) - data["looheatingorsystemtypeid"])))
    v["i20"] = 0.050000*np.tanh((((np.tanh(data["longitude"]) + np.tanh(np.tanh(data["yearbuilt"])))/2.0) * ((data["loostorytypeid"] + (data["loobuildingqualitytypeid"] / 2.0))/2.0)))
    v["i21"] = 0.050000*np.tanh(np.tanh((data["unitcnt"] * np.tanh((data["looregionidcity"] - (data["yearbuilt"] + data["loopropertycountylandusecode"]))))))
    v["i22"] = 0.037700*np.tanh((data["numberofstories"] * (data["loopropertycountylandusecode"] - (((data["numberofstories"] + data["looairconditioningtypeid"]) * data["finishedsquarefeet6"]) * data["numberofstories"]))))
    v["i23"] = 0.050000*np.tanh((-((data["finishedsquarefeet15"] * (((data["looregionidzip"] - data["finishedsquarefeet15"]) + (data["finishedsquarefeet50"] + data["unitcnt"]))/2.0)))))
    v["i24"] = 0.038820*np.tanh(((np.tanh(((data["taxvaluedollarcnt"] / 2.0) - data["structuretaxvaluedollarcnt"])) - (np.tanh(data["looairconditioningtypeid"]) * data["garagetotalsqft"])) / 2.0))
    v["i25"] = 0.050000*np.tanh((-((data["unitcnt"] * (((data["fullbathcnt"] + (-(data["finishedsquarefeet15"])))/2.0) + ((data["loopropertycountylandusecode"] * 2.0) * 2.0))))))
    v["i26"] = 0.050000*np.tanh((data["finishedsquarefeet6"] * (-((((data["numberofstories"] + data["yearbuilt"]) + data["looregionidneighborhood"]) * data["structuretaxvaluedollarcnt"])))))
    v["i27"] = 0.050000*np.tanh((np.tanh(((data["yardbuildingsqft17"] + ((data["yearbuilt"] / 2.0) / 2.0)) + data["unitcnt"])) * np.tanh(data["taxamount"])))
    v["i28"] = 0.039390*np.tanh(((data["loopooltypeid2"] * data["lootypeconstructiontypeid"]) + ((data["yearbuilt"] * ((-(0.051282)) * data["looregionidcity"])) * 2.0)))
    v["i29"] = 0.050000*np.tanh((data["fireplacecnt"] * ((((data["yardbuildingsqft17"] + (data["finishedsquarefeet15"] + data["loohashottuborspa"]))/2.0) + data["taxamount"]) * data["finishedsquarefeet15"])))
    v["i30"] = 0.048560*np.tanh((((((data["loobuildingqualitytypeid"] - data["structuretaxvaluedollarcnt"]) * data["landtaxvaluedollarcnt"]) + data["landtaxvaluedollarcnt"]) * data["landtaxvaluedollarcnt"]) * data["threequarterbathnbr"]))
    v["i31"] = 0.039110*np.tanh((np.tanh(np.tanh((0.907692 - np.tanh(((data["month"] + (data["looregionidneighborhood"] * data["month"]))/2.0))))) / 2.0))
    v["i32"] = 0.045330*np.tanh((((data["fireplacecnt"] + ((data["finishedfloor1squarefeet"] + data["lotsizesquarefeet"]) + data["landtaxvaluedollarcnt"]))/2.0) * (data["lotsizesquarefeet"] * data["garagetotalsqft"])))
    v["i33"] = 0.043900*np.tanh((data["loopropertylandusetypeid"] * (data["garagetotalsqft"] - (np.tanh(np.tanh((((data["finishedsquarefeet12"] * 2.0) * 2.0) * 2.0))) * 2.0))))
    v["i34"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] * (((data["looregionidcounty"] * data["taxvaluedollarcnt"]) + ((data["loofips"] * data["looregionidneighborhood"]) - data["garagetotalsqft"]))/2.0)))
    v["i35"] = 0.050000*np.tanh(((-((data["finishedsquarefeet6"] * (data["finishedsquarefeet6"] + ((data["looregionidcity"] * data["looregionidcity"]) * 2.0))))) * 2.0))
    v["i36"] = 0.044330*np.tanh(((((data["basementsqft"] + data["threequarterbathnbr"]) + data["lotsizesquarefeet"]) + data["finishedsquarefeet15"]) * (data["structuretaxvaluedollarcnt"] * data["numberofstories"])))
    v["i37"] = 0.049980*np.tanh((data["loopropertycountylandusecode"] * (data["threequarterbathnbr"] + np.tanh(((-((data["censustractandblock"] / 2.0))) + (-(data["landtaxvaluedollarcnt"])))))))
    v["i38"] = 0.049930*np.tanh((data["finishedsquarefeet13"] * (data["fireplacecnt"] + (data["looregionidneighborhood"] - ((np.tanh((data["structuretaxvaluedollarcnt"] * 2.0)) * 2.0) * 2.0)))))
    v["i39"] = 0.049980*np.tanh(((data["taxamount"] * ((data["longitude"] * (data["longitude"] * data["longitude"])) - data["finishedsquarefeet6"])) * data["finishedsquarefeet6"]))
    v["i40"] = 0.050000*np.tanh((data["finishedsquarefeet15"] * ((data["longitude"] + ((data["taxdelinquencyyear"] + (data["bedroomcnt"] * ((data["loopooltypeid7"] + data["latitude"])/2.0)))/2.0))/2.0)))
    v["i41"] = 0.050000*np.tanh(((0.051282 * data["looregionidneighborhood"]) * (data["calculatedbathnbr"] * (data["looregionidneighborhood"] - (data["loopooltypeid10"] * data["looregionidneighborhood"])))))
    v["i42"] = 0.047730*np.tanh((((data["finishedsquarefeet6"] - data["unitcnt"]) * data["numberofstories"]) * (data["finishedsquarefeet6"] - data["looregionidcity"])))
    v["i43"] = 0.050000*np.tanh((data["threequarterbathnbr"] * (((data["calculatedfinishedsquarefeet"] + data["latitude"])/2.0) * (data["threequarterbathnbr"] * data["loopropertycountylandusecode"]))))
    v["i44"] = 0.049990*np.tanh((data["finishedsquarefeet15"] * ((data["yardbuildingsqft17"] * data["loodecktypeid"]) + (data["taxamount"] * data["lootaxdelinquencyflag"]))))
    v["i45"] = 0.025700*np.tanh((((data["bedroomcnt"] * np.tanh(((data["calculatedfinishedsquarefeet"] / 2.0) - data["loopooltypeid7"]))) / 2.0) / 2.0))
    v["i46"] = 0.050000*np.tanh((-((0.051282 * np.tanh((((data["loofireplaceflag"] - data["looregionidzip"]) + data["bedroomcnt"]) * 2.0))))))
    v["i47"] = 0.050000*np.tanh((data["structuretaxvaluedollarcnt"] * (data["taxdelinquencyyear"] * (data["taxdelinquencyyear"] * ((((-(data["landtaxvaluedollarcnt"])) / 2.0) + data["taxamount"])/2.0)))))
    v["i48"] = 0.046110*np.tanh((((data["loopropertycountylandusecode"] * data["loopropertycountylandusecode"]) + (data["taxamount"] * np.tanh((data["loopropertycountylandusecode"] - data["looairconditioningtypeid"]))))/2.0))
    v["i49"] = 0.049980*np.tanh((data["loopropertycountylandusecode"] * ((((data["threequarterbathnbr"] + data["loohashottuborspa"]) - data["unitcnt"]) + (data["taxdelinquencyyear"] - data["landtaxvaluedollarcnt"]))/2.0)))
    v["i50"] = 0.032180*np.tanh((data["unitcnt"] * ((data["bedroomcnt"] * ((data["looregionidneighborhood"] - data["loobuildingclasstypeid"]) + data["lotsizesquarefeet"])) + data["lotsizesquarefeet"])))
    v["i51"] = 0.050000*np.tanh((data["assessmentyear"] * (((data["looregionidzip"] * (data["looregionidzip"] * data["finishedsquarefeet12"])) - data["loobuildingqualitytypeid"]) * 2.0)))
    v["i52"] = 0.040370*np.tanh((np.tanh((data["taxamount"] - data["taxvaluedollarcnt"])) * data["looregionidzip"]))
    v["i53"] = 0.050000*np.tanh((data["looregionidneighborhood"] * (data["finishedsquarefeet15"] * (((data["lootaxdelinquencyflag"] * data["bedroomcnt"]) * data["lootaxdelinquencyflag"]) - data["bedroomcnt"]))))
    v["i54"] = 0.043110*np.tanh((((((data["bedroomcnt"] + ((data["censustractandblock"] * data["bedroomcnt"]) * data["censustractandblock"]))/2.0) + data["lootaxdelinquencyflag"])/2.0) * data["loopropertycountylandusecode"]))
    v["i55"] = 0.042420*np.tanh((((((np.tanh(data["taxamount"]) * ((-(data["looregionidneighborhood"])) * data["looregionidneighborhood"])) / 2.0) / 2.0) + 0.051282)/2.0))
    v["i56"] = 0.047870*np.tanh(np.tanh(np.tanh(np.tanh((((data["taxvaluedollarcnt"] + (data["landtaxvaluedollarcnt"] * data["lootypeconstructiontypeid"])) + (-(data["taxamount"])))/2.0)))))
    v["i57"] = 0.050000*np.tanh((data["loopropertylandusetypeid"] * ((data["loofips"] + ((data["loofips"] + data["loopropertylandusetypeid"]) - data["censustractandblock"])) - data["censustractandblock"])))
    v["i58"] = 0.044510*np.tanh(((data["censustractandblock"] * 0.051282) * (data["garagecarcnt"] * (((data["numberofstories"] + data["numberofstories"])/2.0) + data["looregionidneighborhood"]))))
    v["i59"] = 0.049920*np.tanh(((data["unitcnt"] * (data["taxdelinquencyyear"] + (data["bedroomcnt"] + (data["structuretaxvaluedollarcnt"] * data["fireplacecnt"])))) * data["structuretaxvaluedollarcnt"]))
    v["i60"] = 0.050000*np.tanh((data["finishedsquarefeet12"] * (data["taxdelinquencyyear"] * ((data["landtaxvaluedollarcnt"] + ((-(data["yearbuilt"])) + data["finishedsquarefeet12"]))/2.0))))
    v["i61"] = 0.050000*np.tanh((data["structuretaxvaluedollarcnt"] * (-(((data["looregionidzip"] * 2.0) * (data["finishedsquarefeet15"] - data["finishedsquarefeet6"]))))))
    v["i62"] = 0.050000*np.tanh(((data["censustractandblock"] + ((data["censustractandblock"] + data["bedroomcnt"])/2.0)) * (data["bedroomcnt"] * (data["nans"] * data["yardbuildingsqft26"]))))
    v["i63"] = 0.050000*np.tanh((-((data["loopropertylandusetypeid"] * np.tanh((data["bathroomcnt"] + (data["propertyzoningdesc"] + (data["propertyzoningdesc"] + data["nans"]))))))))
    v["i64"] = 0.050000*np.tanh((data["loopropertylandusetypeid"] * np.tanh((((((data["looregionidcity"] * 2.0) + data["taxamount"]) * 2.0) + data["longitude"]) * 2.0))))
    v["i65"] = 0.050000*np.tanh((data["yardbuildingsqft26"] * (data["yardbuildingsqft17"] + ((np.tanh(((data["fireplacecnt"] + np.tanh(data["yardbuildingsqft26"]))/2.0)) + data["yardbuildingsqft17"])/2.0))))
    v["i66"] = 0.050000*np.tanh(((data["nans"] * (((data["loobuildingclasstypeid"] * data["nans"]) - ((data["unitcnt"] / 2.0) / 2.0)) / 2.0)) / 2.0))
    v["i67"] = 0.050000*np.tanh((data["censustractandblock"] * (((data["taxdelinquencyyear"] * data["garagetotalsqft"]) + np.tanh(((data["loohashottuborspa"] / 2.0) / 2.0)))/2.0)))
    v["i68"] = 0.049960*np.tanh((data["loopropertylandusetypeid"] * ((-((data["looregionidzip"] * data["garagetotalsqft"]))) - (data["loopropertylandusetypeid"] * data["looregionidzip"]))))
    v["i69"] = 0.050000*np.tanh((-(((data["looregionidcity"] + data["looregionidzip"]) * (((data["looregionidzip"] + np.tanh(data["lootypeconstructiontypeid"]))/2.0) * data["lootypeconstructiontypeid"])))))
    v["i70"] = 0.049990*np.tanh((data["finishedsquarefeet15"] * (((data["latitude"] * data["finishedsquarefeet15"]) + (data["garagetotalsqft"] + data["loohashottuborspa"]))/2.0)))
    v["i71"] = 0.040550*np.tanh((np.tanh((((data["taxvaluedollarcnt"] - data["taxamount"]) * (-((data["taxvaluedollarcnt"] + data["yearbuilt"])))) * 2.0)) / 2.0))
    v["i72"] = 0.050000*np.tanh((data["loopropertylandusetypeid"] * (-(((((data["longitude"] * data["numberofstories"]) + data["finishedsquarefeet12"])/2.0) + data["censustractandblock"])))))
    v["i73"] = 0.046530*np.tanh(((((((-(data["finishedsquarefeet15"])) * data["unitcnt"]) * data["unitcnt"]) + np.tanh(np.tanh(data["taxdelinquencyyear"])))/2.0) / 2.0))
    v["i74"] = 0.038200*np.tanh((((data["loopropertycountylandusecode"] - (data["censustractandblock"] * data["looregionidneighborhood"])) * (-(data["unitcnt"]))) / 2.0))
    v["i75"] = 0.047960*np.tanh((np.tanh(data["lotsizesquarefeet"]) * (((data["propertyzoningdesc"] / 2.0) + (((data["latitude"] * data["propertyzoningdesc"]) + data["yearbuilt"])/2.0))/2.0)))
    v["i76"] = 0.050000*np.tanh((((data["lotsizesquarefeet"] + data["assessmentyear"])/2.0) * ((((data["looregionidcity"] - data["finishedsquarefeet50"]) * data["looregionidcity"]) + data["censustractandblock"])/2.0)))
    v["i77"] = 0.048250*np.tanh((data["finishedsquarefeet13"] * (((data["fireplacecnt"] - data["poolsizesum"]) + ((data["fireplacecnt"] + ((data["landtaxvaluedollarcnt"] * 2.0) * 2.0))/2.0))/2.0)))
    v["i78"] = 0.047320*np.tanh(((np.tanh((-(((data["looairconditioningtypeid"] * data["loopooltypeid7"]) * data["loopooltypeid7"])))) + (data["finishedsquarefeet6"] * data["fireplacecnt"]))/2.0))
    v["i79"] = 0.049970*np.tanh((data["finishedsquarefeet6"] * ((-((data["structuretaxvaluedollarcnt"] * (data["bedroomcnt"] + (data["numberofstories"] - data["taxvaluedollarcnt"]))))) * 2.0)))
    return Outputs(v.sum(axis=1))*(2*.418)-.418


def GP2(data):
    v = pd.DataFrame()
    v["i0"] = 0.050000*np.tanh(((data["loopropertycountylandusecode"] + (data["loopropertycountylandusecode"] + ((1.370370 + data["looregionidcity"])/2.0))) + ((data["looregionidzip"] + data["finishedsquarefeet12"])/2.0)))
    v["i1"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] + (((data["calculatedfinishedsquarefeet"] * data["calculatedfinishedsquarefeet"]) + ((data["calculatedfinishedsquarefeet"] + (data["looregionidneighborhood"] - data["loopooltypeid2"]))/2.0))/2.0)))
    v["i2"] = 0.050000*np.tanh(((data["loopropertycountylandusecode"] * 2.0) - ((data["month"] * data["loopooltypeid7"]) - ((data["finishedsquarefeet15"] + np.tanh(data["loohashottuborspa"]))/2.0))))
    v["i3"] = 0.050000*np.tanh((((((data["taxvaluedollarcnt"] - data["structuretaxvaluedollarcnt"]) + (data["lootaxdelinquencyflag"] / 2.0)) + data["calculatedfinishedsquarefeet"])/2.0) - np.tanh(data["taxamount"])))
    v["i4"] = 0.049990*np.tanh(((((((data["looregionidcity"] - data["roomcnt"]) + (-(data["loodecktypeid"])))/2.0) + (data["looregionidzip"] * data["calculatedfinishedsquarefeet"]))/2.0) / 2.0))
    v["i5"] = 0.050000*np.tanh((data["unitcnt"] * ((data["latitude"] + ((data["yardbuildingsqft17"] + data["looregionidcity"])/2.0)) + (data["looairconditioningtypeid"] + data["finishedsquarefeet50"]))))
    v["i6"] = 0.050000*np.tanh((data["taxvaluedollarcnt"] - (data["taxamount"] - (np.tanh((data["taxamount"] * (data["yearbuilt"] + data["unitcnt"]))) / 2.0))))
    v["i7"] = 0.050000*np.tanh((-(((((data["taxvaluedollarcnt"] * data["loofips"]) + ((data["taxamount"] + ((data["lootypeconstructiontypeid"] + data["looregionidcounty"])/2.0))/2.0))/2.0) / 2.0))))
    v["i8"] = 0.050000*np.tanh(((data["loopropertycountylandusecode"] + ((((data["numberofstories"] + ((data["looregionidcity"] * data["looregionidcity"]) + data["looregionidneighborhood"]))/2.0) / 2.0) / 2.0))/2.0))
    v["i9"] = 0.050000*np.tanh(((data["loopropertylandusetypeid"] + (data["censustractandblock"] * ((data["looairconditioningtypeid"] + (data["looarchitecturalstyletypeid"] * (data["loopooltypeid10"] + data["looarchitecturalstyletypeid"])))/2.0)))/2.0))
    v["i10"] = 0.050000*np.tanh(((-(np.tanh(data["unitcnt"]))) * (data["calculatedbathnbr"] - np.tanh((data["looregionidcity"] * 2.0)))))
    v["i11"] = 0.048190*np.tanh(((-((data["loopropertycountylandusecode"] * (np.tanh((data["landtaxvaluedollarcnt"] * 2.0)) - data["loobuildingqualitytypeid"])))) * 2.0))
    v["i12"] = 0.042680*np.tanh(np.tanh((np.tanh((data["finishedsquarefeet15"] * (((data["bedroomcnt"] * 2.0) - data["loohashottuborspa"]) - data["bathroomcnt"]))) * 2.0)))
    v["i13"] = 0.050000*np.tanh((((data["calculatedfinishedsquarefeet"] + np.tanh(((np.tanh((-(data["garagetotalsqft"]))) + (-((data["taxamount"] * 2.0))))/2.0)))/2.0) / 2.0))
    v["i14"] = 0.040180*np.tanh((-((((data["lootypeconstructiontypeid"] * 2.0) * data["month"]) * (data["month"] + ((-1.0 * 2.0) * 2.0))))))
    v["i15"] = 0.036760*np.tanh(np.tanh(np.tanh(np.tanh(((-(data["finishedsquarefeet13"])) * (((data["nans"] * data["nans"]) + data["loopooltypeid10"])/2.0))))))
    v["i16"] = 0.048920*np.tanh(((((data["loopropertylandusetypeid"] * 2.0) * 2.0) + np.tanh(np.tanh((((data["taxamount"] * data["taxamount"]) * 2.0) * 2.0))))/2.0))
    v["i17"] = 0.050000*np.tanh(((data["loobuildingqualitytypeid"] * ((((np.tanh(data["longitude"]) + (data["landtaxvaluedollarcnt"] * data["longitude"]))/2.0) + 0.109589)/2.0)) / 2.0))
    v["i18"] = 0.050000*np.tanh((data["finishedsquarefeet6"] * (data["looregionidneighborhood"] + (((data["landtaxvaluedollarcnt"] * data["fullbathcnt"]) + data["roomcnt"]) + data["roomcnt"]))))
    v["i19"] = 0.050000*np.tanh(((data["loopropertycountylandusecode"] * ((-(data["censustractandblock"])) + ((data["numberofstories"] + data["bathroomcnt"]) / 2.0))) * data["censustractandblock"]))
    v["i20"] = 0.050000*np.tanh(((-((data["unitcnt"] * data["looregionidcity"]))) * (data["looheatingorsystemtypeid"] * data["looregionidzip"])))
    v["i21"] = 0.050000*np.tanh((np.tanh(np.tanh((((data["loopropertycountylandusecode"] + data["loopooltypeid7"]) * 2.0) * (-(np.tanh(data["nans"])))))) / 2.0))
    v["i22"] = 0.035970*np.tanh(((((-(((np.tanh((data["looregionidcity"] * 2.0)) + (data["looheatingorsystemtypeid"] / 2.0))/2.0))) + np.tanh(data["landtaxvaluedollarcnt"]))/2.0) / 2.0))
    v["i23"] = 0.049990*np.tanh(((data["looheatingorsystemtypeid"] - (((1.718750 + data["yardbuildingsqft17"]) + data["looregionidzip"])/2.0)) * (data["looregionidzip"] * data["loopropertycountylandusecode"])))
    v["i24"] = 0.050000*np.tanh((-(((data["finishedsquarefeet15"] + data["taxdelinquencyyear"]) * (data["looregionidneighborhood"] * (data["finishedsquarefeet12"] + data["finishedsquarefeet15"]))))))
    v["i25"] = 0.050000*np.tanh((-((data["unitcnt"] * (((data["unitcnt"] + data["loopropertycountylandusecode"])/2.0) + (np.tanh(data["yearbuilt"]) - data["taxamount"]))))))
    v["i26"] = 0.050000*np.tanh((data["garagetotalsqft"] * (data["lotsizesquarefeet"] * (((data["finishedsquarefeet50"] + data["lotsizesquarefeet"])/2.0) + ((data["finishedsquarefeet12"] + data["loopooltypeid7"])/2.0)))))
    v["i27"] = 0.047640*np.tanh(((data["taxamount"] + data["lootypeconstructiontypeid"]) * np.tanh((data["yardbuildingsqft17"] + data["lotsizesquarefeet"]))))
    v["i28"] = 0.050000*np.tanh((data["looairconditioningtypeid"] * (np.tanh(data["landtaxvaluedollarcnt"]) - data["landtaxvaluedollarcnt"])))
    v["i29"] = 0.047290*np.tanh((data["month"] * (((-((-1.0 + np.tanh(data["month"])))) * 2.0) * 2.0)))
    v["i30"] = 0.050000*np.tanh((0.109589 * ((-1.0 + (data["censustractandblock"] - (((data["poolsizesum"] + 0.535354) + data["taxdelinquencyyear"])/2.0)))/2.0)))
    v["i31"] = 0.049960*np.tanh(((data["assessmentyear"] * ((data["month"] - 5.0) / 2.0)) * (data["month"] - data["bedroomcnt"])))
    v["i32"] = 0.044620*np.tanh((-(((data["loopropertycountylandusecode"] * data["taxamount"]) + ((0.297619 / 2.0) + (data["loopropertylandusetypeid"] * 2.0))))))
    v["i33"] = 0.050000*np.tanh((data["assessmentyear"] * ((data["latitude"] - (data["fireplacecnt"] * data["yardbuildingsqft17"])) + (data["loohashottuborspa"] * data["nans"]))))
    v["i34"] = 0.050000*np.tanh(((data["longitude"] - (data["calculatedfinishedsquarefeet"] * (data["looregionidcity"] + (data["looregionidcity"] * data["looregionidzip"])))) * data["finishedsquarefeet15"]))
    v["i35"] = 0.042890*np.tanh((((data["finishedsquarefeet15"] * np.tanh(np.tanh((data["garagetotalsqft"] + (data["finishedsquarefeet15"] * data["latitude"]))))) * 2.0) * 2.0))
    v["i36"] = 0.049970*np.tanh(((data["lotsizesquarefeet"] + ((data["lotsizesquarefeet"] + data["loopropertycountylandusecode"])/2.0)) * (data["threequarterbathnbr"] + (data["bedroomcnt"] * data["unitcnt"]))))
    v["i37"] = 0.049910*np.tanh(((data["finishedsquarefeet15"] * ((data["lootaxdelinquencyflag"] * data["finishedsquarefeet15"]) - ((data["looregionidzip"] / 2.0) + data["looregionidcity"]))) / 2.0))
    v["i38"] = 0.042620*np.tanh((data["loopropertycountylandusecode"] * ((((data["structuretaxvaluedollarcnt"] * data["structuretaxvaluedollarcnt"]) + data["loohashottuborspa"]) + (data["yardbuildingsqft17"] * data["fullbathcnt"]))/2.0)))
    v["i39"] = 0.050000*np.tanh((np.tanh(np.tanh(data["loopooltypeid2"])) * (data["garagetotalsqft"] * (data["landtaxvaluedollarcnt"] - np.tanh(np.tanh(data["loopooltypeid2"]))))))
    v["i40"] = 0.050000*np.tanh(((data["finishedsquarefeet6"] * 2.0) * ((-((data["structuretaxvaluedollarcnt"] * data["finishedsquarefeet6"]))) + np.tanh((-(data["taxvaluedollarcnt"]))))))
    v["i41"] = 0.050000*np.tanh(((((data["loostorytypeid"] + (data["loobuildingclasstypeid"] * data["lootaxdelinquencyflag"]))/2.0) + (data["lootaxdelinquencyflag"] * data["taxamount"])) * data["finishedsquarefeet15"]))
    v["i42"] = 0.049990*np.tanh((-((data["loopropertylandusetypeid"] * (np.tanh(((data["finishedsquarefeet12"] + (data["bedroomcnt"] * data["bedroomcnt"])) * 2.0)) * 2.0)))))
    v["i43"] = 0.050000*np.tanh((-((data["finishedsquarefeet13"] * ((data["bedroomcnt"] + (data["bedroomcnt"] * data["yearbuilt"])) * data["yearbuilt"])))))
    v["i44"] = 0.039600*np.tanh((data["finishedsquarefeet13"] * ((data["basementsqft"] * (data["basementsqft"] * 2.0)) + ((data["nans"] / 2.0) - data["looheatingorsystemtypeid"]))))
    v["i45"] = 0.048900*np.tanh((((-((data["loofireplaceflag"] - data["garagetotalsqft"]))) + data["looregionidzip"]) * (data["looregionidzip"] * data["assessmentyear"])))
    v["i46"] = 0.050000*np.tanh(((data["loopropertycountylandusecode"] * data["threequarterbathnbr"]) * (((data["loopropertycountylandusecode"] * data["loofireplaceflag"]) - data["loofireplaceflag"]) + data["garagecarcnt"])))
    v["i47"] = 0.050000*np.tanh((-(((data["structuretaxvaluedollarcnt"] * data["finishedsquarefeet6"]) * (data["unitcnt"] - (data["roomcnt"] - (data["fullbathcnt"] * 2.0)))))))
    v["i48"] = 0.042040*np.tanh(((data["looregionidcounty"] * data["loopropertylandusetypeid"]) * (data["yearbuilt"] + (((-1.0 + data["bedroomcnt"])/2.0) - data["lootaxdelinquencyflag"]))))
    v["i49"] = 0.049990*np.tanh((data["landtaxvaluedollarcnt"] * ((data["loobuildingqualitytypeid"] - (data["taxamount"] - np.tanh(data["finishedsquarefeet12"]))) * data["loopropertylandusetypeid"])))
    v["i50"] = 0.050000*np.tanh(((data["finishedsquarefeet6"] * ((-((np.tanh((data["calculatedfinishedsquarefeet"] * 2.0)) - (-(data["looregionidzip"]))))) * 2.0)) * 2.0))
    v["i51"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] * (data["looregionidneighborhood"] * ((data["yearbuilt"] - ((data["loopropertycountylandusecode"] * 2.0) * 2.0)) - data["loopropertycountylandusecode"]))))
    v["i52"] = 0.048510*np.tanh((data["loopropertycountylandusecode"] * ((((-(data["garagetotalsqft"])) * data["looregionidcity"]) - np.tanh(data["looregionidcity"])) - data["unitcnt"])))
    v["i53"] = 0.043830*np.tanh((np.tanh((data["finishedsquarefeet13"] + (np.tanh(data["finishedsquarefeet13"]) + (data["lootaxdelinquencyflag"] * data["taxamount"])))) * data["fireplacecnt"]))
    v["i54"] = 0.050000*np.tanh(((data["bedroomcnt"] * ((((np.tanh((data["latitude"] * 2.0)) + data["finishedfloor1squarefeet"])/2.0) + data["rawcensustractandblock"])/2.0)) * 0.109589))
    v["i55"] = 0.049980*np.tanh((data["lootaxdelinquencyflag"] * ((np.tanh((data["looregionidneighborhood"] * data["garagecarcnt"])) + (data["taxamount"] * data["looairconditioningtypeid"]))/2.0)))
    v["i56"] = 0.045680*np.tanh((data["garagecarcnt"] * (data["assessmentyear"] * ((data["numberofstories"] - data["loodecktypeid"]) + (data["numberofstories"] - data["looregionidzip"])))))
    v["i57"] = 0.049990*np.tanh(((((data["fullbathcnt"] + (data["looregionidcity"] * (5.0 * data["looregionidcity"])))/2.0) * data["finishedsquarefeet6"]) * data["finishedsquarefeet6"]))
    v["i58"] = 0.049990*np.tanh(((data["lootaxdelinquencyflag"] * ((data["finishedsquarefeet15"] + (np.tanh(data["taxvaluedollarcnt"]) / 2.0))/2.0)) * np.tanh(np.tanh(data["taxamount"]))))
    v["i59"] = 0.044650*np.tanh(((data["taxamount"] / 2.0) * (data["taxamount"] - (np.tanh((3.833330 * data["taxamount"])) * data["taxamount"]))))
    v["i60"] = 0.044540*np.tanh(np.tanh(np.tanh(((-1.0 + np.tanh(((1.718750 + (-(data["looairconditioningtypeid"]))) + data["poolsizesum"])))/2.0))))
    v["i61"] = 0.050000*np.tanh((data["loofireplaceflag"] * np.tanh((data["loofireplaceflag"] * ((data["calculatedbathnbr"] + data["bedroomcnt"]) * (data["loodecktypeid"] * 2.0))))))
    v["i62"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] * (np.tanh(data["landtaxvaluedollarcnt"]) * (data["loopropertycountylandusecode"] * (data["loopropertycountylandusecode"] * data["propertyzoningdesc"])))))
    v["i63"] = 0.050000*np.tanh((data["yardbuildingsqft26"] * ((data["structuretaxvaluedollarcnt"] + data["lootaxdelinquencyflag"]) - (data["bedroomcnt"] + (data["bedroomcnt"] * data["lootaxdelinquencyflag"])))))
    v["i64"] = 0.049960*np.tanh((((data["censustractandblock"] * data["loopropertycountylandusecode"]) * data["censustractandblock"]) * (((data["loopropertycountylandusecode"] * data["censustractandblock"]) + data["bedroomcnt"])/2.0)))
    v["i65"] = 0.049810*np.tanh((data["basementsqft"] * (((data["calculatedbathnbr"] - (data["loofireplaceflag"] * data["loopooltypeid2"])) - data["looregionidneighborhood"]) - data["taxamount"])))
    v["i66"] = 0.050000*np.tanh((((-((((data["yardbuildingsqft26"] / 2.0) * data["structuretaxvaluedollarcnt"]) * data["yardbuildingsqft26"]))) + (data["finishedsquarefeet15"] * data["lotsizesquarefeet"]))/2.0))
    v["i67"] = 0.050000*np.tanh(((data["calculatedfinishedsquarefeet"] * (((data["looairconditioningtypeid"] - data["unitcnt"]) - data["loobuildingclasstypeid"]) * data["loopropertycountylandusecode"])) * 2.0))
    v["i68"] = 0.036530*np.tanh((data["taxamount"] * (((np.tanh((np.tanh(data["loofireplaceflag"]) - data["longitude"])) + data["loobuildingclasstypeid"])/2.0) * data["lootaxdelinquencyflag"])))
    v["i69"] = 0.050000*np.tanh(((data["structuretaxvaluedollarcnt"] * ((data["numberofstories"] + (data["numberofstories"] * data["lotsizesquarefeet"]))/2.0)) * (data["lotsizesquarefeet"] - data["finishedsquarefeet6"])))
    v["i70"] = 0.048130*np.tanh((data["unitcnt"] * (((data["nans"] * (-(((data["unitcnt"] + data["loopropertycountylandusecode"])/2.0)))) + (data["unitcnt"] / 2.0))/2.0)))
    v["i71"] = 0.042190*np.tanh((data["loopropertylandusetypeid"] * np.tanh(((((-(data["numberofstories"])) * 2.0) + data["longitude"]) * 2.0))))
    v["i72"] = 0.049990*np.tanh(((-(((data["loopropertycountylandusecode"] + (-(data["loobuildingclasstypeid"])))/2.0))) + (data["loopropertycountylandusecode"] * (data["taxamount"] * data["loopropertycountylandusecode"]))))
    v["i73"] = 0.049990*np.tanh((data["loobuildingclasstypeid"] * (data["unitcnt"] + (data["loofireplaceflag"] - (data["unitcnt"] * (data["unitcnt"] * data["unitcnt"]))))))
    v["i74"] = 0.028420*np.tanh(((((data["structuretaxvaluedollarcnt"] * data["looheatingorsystemtypeid"]) + (np.tanh((data["looheatingorsystemtypeid"] * (-(data["bedroomcnt"])))) / 2.0))/2.0) / 2.0))
    v["i75"] = 0.042200*np.tanh((((-(data["lotsizesquarefeet"])) * ((-(data["taxdelinquencyyear"])) * data["taxamount"])) * (data["taxdelinquencyyear"] + data["taxamount"])))
    v["i76"] = 0.049990*np.tanh((((data["lotsizesquarefeet"] * (data["landtaxvaluedollarcnt"] - data["looregionidcity"])) - data["lootaxdelinquencyflag"]) * (data["looregionidcity"] * data["unitcnt"])))
    v["i77"] = 0.050000*np.tanh((0.989011 + np.tanh((-((data["month"] + ((data["month"] * data["taxamount"]) * data["calculatedbathnbr"])))))))
    v["i78"] = 0.041330*np.tanh(((data["landtaxvaluedollarcnt"] * data["unitcnt"]) * (((data["taxvaluedollarcnt"] * 2.0) + (data["nans"] - np.tanh(2.800000)))/2.0)))
    v["i79"] = 0.049930*np.tanh(((data["taxamount"] * (data["finishedsquarefeet15"] * 2.0)) * ((data["calculatedfinishedsquarefeet"] * data["lootaxdelinquencyflag"]) + data["lootaxdelinquencyflag"])))
    return Outputs(v.sum(axis=1))*(2*.418)-.418


def GP3(data):
    v = pd.DataFrame()
    v["i0"] = 0.050000*np.tanh(((data["finishedsquarefeet12"] + ((((data["finishedsquarefeet12"] * data["finishedsquarefeet12"]) + (2.0))/2.0) + (data["looregionidcity"] + data["looregionidneighborhood"])))/2.0))
    v["i1"] = 0.044940*np.tanh(((data["landtaxvaluedollarcnt"] * data["taxamount"]) + (((data["looregionidzip"] + (data["calculatedfinishedsquarefeet"] - data["loopooltypeid2"]))/2.0) - data["taxamount"])))
    v["i2"] = 0.050000*np.tanh(((((np.tanh(data["loohashottuborspa"]) + ((data["calculatedfinishedsquarefeet"] / 2.0) - data["loopooltypeid7"]))/2.0) + (data["loopropertycountylandusecode"] * 2.0)) * 2.0))
    v["i3"] = 0.049810*np.tanh((((data["loopropertycountylandusecode"] * 2.0) + ((data["lootaxdelinquencyflag"] - ((data["rawcensustractandblock"] + (data["loodecktypeid"] + data["finishedsquarefeet50"]))/2.0)) / 2.0))/2.0))
    v["i4"] = 0.046150*np.tanh((((data["finishedsquarefeet15"] + data["loopropertycountylandusecode"]) + (((data["finishedsquarefeet12"] * data["looregionidzip"]) + (-(np.tanh(data["bedroomcnt"]))))/2.0))/2.0))
    v["i5"] = 0.041700*np.tanh(((data["month"] * ((data["lootypeconstructiontypeid"] * 2.0) * (-((-2.0 + data["month"]))))) - 0.044444))
    v["i6"] = 0.041970*np.tanh(((np.tanh(np.tanh((0.452632 + (-((data["month"] - 2.0)))))) / 2.0) / 2.0))
    v["i7"] = 0.050000*np.tanh((data["unitcnt"] * (((data["looregionidcity"] * 2.0) + ((data["latitude"] - np.tanh(data["yearbuilt"])) * 2.0))/2.0)))
    v["i8"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] * (-((data["loofips"] + (data["looregionidcity"] + np.tanh((data["landtaxvaluedollarcnt"] + data["landtaxvaluedollarcnt"]))))))))
    v["i9"] = 0.042640*np.tanh(((0.044444 + np.tanh((data["looheatingorsystemtypeid"] * (data["finishedsquarefeet15"] - (data["looairconditioningtypeid"] + 0.117647)))))/2.0))
    v["i10"] = 0.050000*np.tanh(np.tanh((data["taxvaluedollarcnt"] * (data["taxvaluedollarcnt"] * np.tanh((data["taxvaluedollarcnt"] * ((data["lotsizesquarefeet"] * 2.0) * 2.0)))))))
    v["i11"] = 0.012260*np.tanh((data["calculatedbathnbr"] * (((-(data["unitcnt"])) + ((data["month"] * 0.044444) * data["calculatedfinishedsquarefeet"]))/2.0)))
    v["i12"] = 0.050000*np.tanh((0.117647 * (data["taxvaluedollarcnt"] * (data["yearbuilt"] + ((data["unitcnt"] * 2.0) * 2.0)))))
    v["i13"] = 0.050000*np.tanh((-(((data["taxamount"] - data["taxvaluedollarcnt"]) - (-((np.tanh(np.tanh((data["taxamount"] / 2.0))) / 2.0)))))))
    v["i14"] = 0.049990*np.tanh((((data["finishedsquarefeet13"] + ((data["finishedsquarefeet12"] + (-(((((data["looheatingorsystemtypeid"] + data["yardbuildingsqft17"])/2.0) + data["garagetotalsqft"])/2.0))))/2.0))/2.0) / 2.0))
    v["i15"] = 0.047320*np.tanh((-(((data["loopropertycountylandusecode"] * (np.tanh((data["landtaxvaluedollarcnt"] * 2.0)) * 2.0)) + (data["assessmentyear"] * data["looregionidcounty"])))))
    v["i16"] = 0.050000*np.tanh((data["loopropertylandusetypeid"] * ((data["loobuildingqualitytypeid"] + ((data["numberofstories"] + data["loohashottuborspa"])/2.0)) * (data["landtaxvaluedollarcnt"] - data["longitude"]))))
    v["i17"] = 0.049840*np.tanh((data["month"] * (((((data["month"] * 0.044444) / 2.0) / 2.0) - 0.044444) / 2.0)))
    v["i18"] = 0.044060*np.tanh(np.tanh(((0.896552 + (data["unitcnt"] * (-(data["fullbathcnt"])))) - np.tanh((data["month"] / 2.0)))))
    v["i19"] = 0.050000*np.tanh(((data["finishedsquarefeet15"] + data["loostorytypeid"]) * (data["longitude"] - (data["propertyzoningdesc"] * (data["finishedsquarefeet15"] + data["finishedsquarefeet15"])))))
    v["i20"] = 0.049900*np.tanh((((((data["looregionidcity"] * data["unitcnt"]) * (-(data["looregionidcity"]))) + (data["roomcnt"] * data["finishedsquarefeet6"]))/2.0) * 2.0))
    v["i21"] = 0.050000*np.tanh((data["assessmentyear"] * ((data["bedroomcnt"] * data["nans"]) + ((data["nans"] * data["looregionidneighborhood"]) + data["nans"]))))
    v["i22"] = 0.044870*np.tanh(((np.tanh((-(np.tanh(((np.tanh((data["looregionidcity"] * (8.0))) + data["structuretaxvaluedollarcnt"])/2.0))))) / 2.0) / 2.0))
    v["i23"] = 0.050000*np.tanh((data["finishedsquarefeet15"] * (data["loohashottuborspa"] + (data["longitude"] + (data["finishedsquarefeet15"] * (data["lootaxdelinquencyflag"] + data["latitude"]))))))
    v["i24"] = 0.048800*np.tanh((data["looarchitecturalstyletypeid"] * (((data["lootaxdelinquencyflag"] - data["looregionidcity"]) * 2.0) + ((data["loopooltypeid10"] + data["looarchitecturalstyletypeid"])/2.0))))
    v["i25"] = 0.050000*np.tanh((data["finishedsquarefeet15"] * (data["taxdelinquencyyear"] - ((data["finishedsquarefeet15"] + (data["looregionidcity"] + (data["looregionidneighborhood"] * data["finishedsquarefeet15"])))/2.0))))
    v["i26"] = 0.050000*np.tanh(((((np.tanh((data["landtaxvaluedollarcnt"] + data["finishedsquarefeet15"])) / 2.0) / 2.0) / 2.0) * (data["yearbuilt"] * data["latitude"])))
    v["i27"] = 0.050000*np.tanh((((data["unitcnt"] * (((data["unitcnt"] * data["finishedsquarefeet12"]) / 2.0) - (data["loopropertycountylandusecode"] * 2.0))) * 2.0) * 2.0))
    v["i28"] = 0.050000*np.tanh(((data["loopropertycountylandusecode"] * ((data["loobuildingqualitytypeid"] + data["numberofstories"])/2.0)) - (data["loopropertylandusetypeid"] * ((1.621620 + data["bedroomcnt"])/2.0))))
    v["i29"] = 0.046000*np.tanh((data["taxamount"] * ((data["yearbuilt"] + (data["yearbuilt"] * (-(np.tanh(data["taxvaluedollarcnt"])))))/2.0)))
    v["i30"] = 0.037600*np.tanh(((data["bedroomcnt"] * (data["loopropertylandusetypeid"] * 2.0)) * (-((np.tanh(data["bedroomcnt"]) - (data["looheatingorsystemtypeid"] * 2.0))))))
    v["i31"] = 0.046600*np.tanh(((data["loopropertylandusetypeid"] + (data["finishedsquarefeet6"] * ((data["finishedsquarefeet6"] * ((-(data["taxvaluedollarcnt"])) * 2.0)) * 2.0)))/2.0))
    v["i32"] = 0.050000*np.tanh((data["unitcnt"] * np.tanh(((data["taxamount"] + ((data["taxvaluedollarcnt"] + data["lootaxdelinquencyflag"]) * data["nans"])) * 2.0))))
    v["i33"] = 0.039870*np.tanh(np.tanh(((((data["looregionidzip"] * (-((data["looheatingorsystemtypeid"] / 2.0)))) / 2.0) + (data["loopooltypeid10"] * data["loopooltypeid10"]))/2.0)))
    v["i34"] = 0.050000*np.tanh((data["loobuildingqualitytypeid"] * (0.044444 * (data["longitude"] + np.tanh((data["longitude"] + data["looregionidneighborhood"]))))))
    v["i35"] = 0.050000*np.tanh((np.tanh(((np.tanh((data["looregionidneighborhood"] * data["looregionidneighborhood"])) + (data["looregionidneighborhood"] * data["looregionidcity"]))/2.0)) / 2.0))
    v["i36"] = 0.041090*np.tanh((data["threequarterbathnbr"] * (data["taxamount"] * ((data["fireplacecnt"] + ((data["fireplacecnt"] + data["finishedsquarefeet50"])/2.0)) * data["month"]))))
    v["i37"] = 0.050000*np.tanh((-(np.tanh((data["loopropertylandusetypeid"] * ((data["fireplacecnt"] + ((data["latitude"] + data["bathroomcnt"]) + data["numberofstories"]))/2.0))))))
    v["i38"] = 0.049980*np.tanh((np.tanh(data["bathroomcnt"]) * np.tanh((data["taxamount"] - data["taxvaluedollarcnt"]))))
    v["i39"] = 0.050000*np.tanh((0.044444 * (-((np.tanh(((data["looregionidneighborhood"] + data["looregionidzip"]) - data["nans"])) - data["landtaxvaluedollarcnt"])))))
    v["i40"] = 0.050000*np.tanh((data["finishedsquarefeet6"] * (-((data["loopooltypeid7"] + ((data["finishedsquarefeet6"] - (data["taxvaluedollarcnt"] * 2.0)) * data["structuretaxvaluedollarcnt"]))))))
    v["i41"] = 0.050000*np.tanh((data["loopropertylandusetypeid"] * (((data["taxamount"] - data["landtaxvaluedollarcnt"]) + data["taxamount"]) - (data["looregionidneighborhood"] * data["taxamount"]))))
    v["i42"] = 0.043730*np.tanh((((((((data["looregionidcity"] * data["numberofstories"]) * 2.0) * 2.0) + data["fireplacecnt"])/2.0) - data["censustractandblock"]) * data["loopropertycountylandusecode"]))
    v["i43"] = 0.050000*np.tanh((data["taxamount"] * (((data["looairconditioningtypeid"] + data["loopropertycountylandusecode"])/2.0) * ((data["bedroomcnt"] * data["bedroomcnt"]) * data["loopropertycountylandusecode"]))))
    v["i44"] = 0.050000*np.tanh(((data["finishedsquarefeet6"] * ((data["looregionidneighborhood"] - np.tanh((data["finishedsquarefeet6"] - data["fireplacecnt"]))) - data["looregionidcity"])) * 2.0))
    v["i45"] = 0.050000*np.tanh((data["finishedsquarefeet15"] * ((data["taxamount"] * (data["looairconditioningtypeid"] * data["finishedsquarefeet12"])) + (data["lootaxdelinquencyflag"] * data["taxamount"]))))
    v["i46"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] * ((data["loopropertycountylandusecode"] * data["censustractandblock"]) + ((((data["yardbuildingsqft26"] + data["bedroomcnt"])/2.0) + data["loohashottuborspa"])/2.0))))
    v["i47"] = 0.049670*np.tanh((data["loopropertylandusetypeid"] * (((data["unitcnt"] * data["bathroomcnt"]) + data["looregionidcity"]) - (data["calculatedfinishedsquarefeet"] * data["calculatedfinishedsquarefeet"]))))
    v["i48"] = 0.049370*np.tanh((data["assessmentyear"] * (data["nans"] - (data["yardbuildingsqft17"] + (data["loofireplaceflag"] * (data["looregionidneighborhood"] * data["fullbathcnt"]))))))
    v["i49"] = 0.050000*np.tanh(((data["taxamount"] - data["taxvaluedollarcnt"]) * (-(((0.452632 + (-(data["looheatingorsystemtypeid"])))/2.0)))))
    v["i50"] = 0.049130*np.tanh((data["loopropertycountylandusecode"] * (data["looregionidcity"] * ((data["finishedsquarefeet12"] * (-(data["finishedsquarefeet12"]))) - data["loopropertycountylandusecode"]))))
    v["i51"] = 0.047800*np.tanh(((data["latitude"] * np.tanh(data["looairconditioningtypeid"])) * (((np.tanh(data["looairconditioningtypeid"]) * data["taxvaluedollarcnt"]) + data["looregionidneighborhood"])/2.0)))
    v["i52"] = 0.050000*np.tanh(((data["assessmentyear"] * data["month"]) * np.tanh((data["month"] - (5.0 + np.tanh(5.0))))))
    v["i53"] = 0.050000*np.tanh((0.044444 * np.tanh(((((data["latitude"] * 2.0) - (data["finishedsquarefeet12"] * data["latitude"])) * 2.0) * 2.0))))
    v["i54"] = 0.050000*np.tanh((data["finishedsquarefeet15"] * (((data["latitude"] - (data["looregionidzip"] * data["latitude"])) * data["finishedsquarefeet15"]) - data["fireplacecnt"])))
    v["i55"] = 0.049980*np.tanh((data["loopropertycountylandusecode"] * ((data["latitude"] + (data["garagecarcnt"] * (data["threequarterbathnbr"] + (data["latitude"] * data["structuretaxvaluedollarcnt"]))))/2.0)))
    v["i56"] = 0.048200*np.tanh((-(((data["loopropertycountylandusecode"] + (((-((data["loopropertycountylandusecode"] * data["threequarterbathnbr"]))) * data["threequarterbathnbr"]) * data["taxamount"]))/2.0))))
    v["i57"] = 0.050000*np.tanh((data["basementsqft"] * (data["loofireplaceflag"] + ((5.0 * np.tanh(data["looregionidzip"])) * (data["loofireplaceflag"] * 2.0)))))
    v["i58"] = 0.050000*np.tanh((data["looairconditioningtypeid"] * (((data["landtaxvaluedollarcnt"] + data["lootypeconstructiontypeid"])/2.0) * (-(((data["looregionidzip"] + (data["bedroomcnt"] / 2.0))/2.0))))))
    v["i59"] = 0.050000*np.tanh(np.tanh(((data["structuretaxvaluedollarcnt"] - (data["basementsqft"] * data["basementsqft"])) * (data["longitude"] * (-(data["lootaxdelinquencyflag"]))))))
    v["i60"] = 0.046730*np.tanh(((data["nans"] / 2.0) * ((data["threequarterbathnbr"] + ((((data["garagecarcnt"] + data["nans"])/2.0) / 2.0) * data["threequarterbathnbr"]))/2.0)))
    v["i61"] = 0.031290*np.tanh(((((-(data["structuretaxvaluedollarcnt"])) + data["threequarterbathnbr"])/2.0) * (((data["bedroomcnt"] * data["structuretaxvaluedollarcnt"]) + (-(data["structuretaxvaluedollarcnt"])))/2.0)))
    v["i62"] = 0.043710*np.tanh((((data["yardbuildingsqft26"] * (data["bedroomcnt"] * ((data["longitude"] - data["yardbuildingsqft17"]) - data["bedroomcnt"]))) * 2.0) * 2.0))
    v["i63"] = 0.050000*np.tanh((((-((data["looregionidzip"] * (data["looregionidzip"] * (((data["loopropertylandusetypeid"] / 2.0) + data["lootypeconstructiontypeid"])/2.0))))) + data["loopropertylandusetypeid"])/2.0))
    v["i64"] = 0.039110*np.tanh(((-((data["unitcnt"] + np.tanh(((data["yearbuilt"] * 2.0) * 2.0))))) * ((data["unitcnt"] + data["loopropertycountylandusecode"])/2.0)))
    v["i65"] = 0.050000*np.tanh((-(((data["yearbuilt"] * (data["looregionidzip"] * data["loopropertylandusetypeid"])) * (data["looregionidneighborhood"] - np.tanh(data["taxamount"]))))))
    v["i66"] = 0.049940*np.tanh((np.tanh(np.tanh(np.tanh(np.tanh(np.tanh(((data["calculatedfinishedsquarefeet"] / 2.0) - np.tanh(data["structuretaxvaluedollarcnt"]))))))) / 2.0))
    v["i67"] = 0.050000*np.tanh(np.tanh(np.tanh((data["lotsizesquarefeet"] * (data["fireplacecnt"] * ((data["landtaxvaluedollarcnt"] * data["landtaxvaluedollarcnt"]) + data["finishedsquarefeet50"]))))))
    v["i68"] = 0.049970*np.tanh(((-((data["loopropertylandusetypeid"] * 2.0))) * (np.tanh((data["landtaxvaluedollarcnt"] - data["threequarterbathnbr"])) + data["lotsizesquarefeet"])))
    v["i69"] = 0.049990*np.tanh((data["loopropertycountylandusecode"] * np.tanh((-(((data["garagetotalsqft"] + (data["loopropertycountylandusecode"] * data["looregionidneighborhood"])) + data["unitcnt"]))))))
    v["i70"] = 0.050000*np.tanh(((data["garagetotalsqft"] + ((data["taxdelinquencyyear"] / 2.0) * data["finishedsquarefeet13"])) * (data["taxdelinquencyyear"] / 2.0)))
    v["i71"] = 0.045670*np.tanh((-((data["loodecktypeid"] * ((data["loostorytypeid"] - data["structuretaxvaluedollarcnt"]) - (-(data["garagecarcnt"])))))))
    v["i72"] = 0.049980*np.tanh((-((((data["censustractandblock"] * (data["basementsqft"] * data["looregionidzip"])) * data["looregionidzip"]) * data["looregionidzip"]))))
    v["i73"] = 0.050000*np.tanh((data["lotsizesquarefeet"] * np.tanh((((data["loopooltypeid7"] + (-(data["looregionidneighborhood"])))/2.0) - (data["looregionidneighborhood"] * data["looregionidneighborhood"])))))
    v["i74"] = 0.050000*np.tanh((-(((data["calculatedbathnbr"] * 2.0) * (((data["basementsqft"] + data["loobuildingclasstypeid"]) * data["loofireplaceflag"]) + data["loostorytypeid"])))))
    v["i75"] = 0.050000*np.tanh((data["structuretaxvaluedollarcnt"] * (data["finishedsquarefeet6"] * ((data["looregionidcity"] - data["looregionidneighborhood"]) - data["finishedsquarefeet6"]))))
    v["i76"] = 0.050000*np.tanh(((-((data["loopropertylandusetypeid"] * data["lotsizesquarefeet"]))) * (((data["loopropertylandusetypeid"] + data["loofireplaceflag"])/2.0) + (data["garagetotalsqft"] * 2.0))))
    v["i77"] = 0.041490*np.tanh(((data["unitcnt"] * (data["unitcnt"] * ((data["loopropertycountylandusecode"] + data["landtaxvaluedollarcnt"])/2.0))) * (data["bedroomcnt"] - data["unitcnt"])))
    v["i78"] = 0.049940*np.tanh((data["taxdelinquencyyear"] * ((data["taxdelinquencyyear"] * (data["structuretaxvaluedollarcnt"] * ((data["propertyzoningdesc"] + data["loofips"])/2.0))) * data["unitcnt"])))
    v["i79"] = 0.050000*np.tanh((((((data["looregionidneighborhood"] + data["looregionidcity"])/2.0) * data["fullbathcnt"]) * data["lootypeconstructiontypeid"]) * (data["lootypeconstructiontypeid"] + data["looregionidneighborhood"])))
    return Outputs(v.sum(axis=1))*(2*.418)-.418


def GP4(data):
    v = pd.DataFrame()
    v["i0"] = 0.050000*np.tanh((((data["looregionidzip"] + data["looregionidcity"])/2.0) + ((data["loopropertycountylandusecode"] + (data["loopropertycountylandusecode"] + 0.690141)) + data["calculatedfinishedsquarefeet"])))
    v["i1"] = 0.000690*np.tanh(((data["calculatedfinishedsquarefeet"] * data["calculatedfinishedsquarefeet"]) + (((data["lootaxdelinquencyflag"] + ((data["calculatedfinishedsquarefeet"] + data["looregionidneighborhood"])/2.0))/2.0) - data["loopooltypeid2"])))
    v["i2"] = 0.050000*np.tanh((((((data["calculatedfinishedsquarefeet"] * data["bathroomcnt"]) + ((data["loohashottuborspa"] + data["looregionidneighborhood"])/2.0))/2.0) + data["loopropertycountylandusecode"]) + data["loopropertycountylandusecode"]))
    v["i3"] = 0.050000*np.tanh((((-((data["loopooltypeid2"] + (np.tanh(data["taxamount"]) * 2.0)))) + ((data["taxvaluedollarcnt"] + data["finishedsquarefeet12"])/2.0))/2.0))
    v["i4"] = 0.043340*np.tanh((data["lootaxdelinquencyflag"] + np.tanh((((data["taxamount"] * data["taxamount"]) + (data["taxvaluedollarcnt"] - data["taxamount"])) * 2.0))))
    v["i5"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] - ((((data["loodecktypeid"] + ((data["rawcensustractandblock"] + data["finishedsquarefeet50"])/2.0))/2.0) + np.tanh((data["loopooltypeid7"] * 2.0)))/2.0)))
    v["i6"] = 0.049960*np.tanh((data["loopropertycountylandusecode"] * (-((data["loopropertycountylandusecode"] + (((data["looregionidcity"] * 2.0) - data["loobuildingqualitytypeid"]) - data["latitude"]))))))
    v["i7"] = 0.050000*np.tanh(((data["month"] * (data["month"] * data["lootypeconstructiontypeid"])) * (((3.863640 - data["month"]) + 2.307690)/2.0)))
    v["i8"] = 0.046240*np.tanh(((((data["calculatedfinishedsquarefeet"] - data["taxamount"]) + np.tanh((data["calculatedbathnbr"] - data["bedroomcnt"])))/2.0) / 2.0))
    v["i9"] = 0.050000*np.tanh(((-(data["lootypeconstructiontypeid"])) - (np.tanh(data["garagetotalsqft"]) * np.tanh(((data["structuretaxvaluedollarcnt"] / 2.0) + data["looairconditioningtypeid"])))))
    v["i10"] = 0.050000*np.tanh(((-((1.227270 * (data["unitcnt"] * data["looregionidcity"])))) * (((data["looregionidcity"] + data["finishedsquarefeet12"])/2.0) * 2.0)))
    v["i11"] = 0.042240*np.tanh((np.tanh((1.0 * 2.0)) - np.tanh((data["month"] + (np.tanh(data["looregionidcity"]) - 1.0)))))
    v["i12"] = 0.050000*np.tanh(((data["finishedsquarefeet15"] + (data["month"] * (data["finishedsquarefeet15"] * ((data["finishedsquarefeet15"] * 2.0) * data["latitude"]))))/2.0))
    v["i13"] = 0.035990*np.tanh((((data["landtaxvaluedollarcnt"] + (-(data["taxamount"])))/2.0) + (data["lotsizesquarefeet"] * (data["taxamount"] + data["taxamount"]))))
    v["i14"] = 0.049080*np.tanh(((np.tanh(np.tanh((data["taxamount"] * ((-(np.tanh(data["bathroomcnt"]))) + data["yearbuilt"])))) + data["loopropertylandusetypeid"])/2.0))
    v["i15"] = 0.049990*np.tanh(np.tanh(np.tanh(np.tanh((data["poolsizesum"] * (((-((data["poolsizesum"] - data["garagetotalsqft"]))) * 2.0) * 2.0))))))
    v["i16"] = 0.050000*np.tanh((data["unitcnt"] * (((data["looregionidcity"] + (data["taxvaluedollarcnt"] - (data["yearbuilt"] + data["unitcnt"])))/2.0) - data["loopropertycountylandusecode"])))
    v["i17"] = 0.050000*np.tanh((data["fullbathcnt"] * (data["loobuildingqualitytypeid"] * ((((data["loobuildingqualitytypeid"] + data["looarchitecturalstyletypeid"])/2.0) + data["looairconditioningtypeid"]) * data["looarchitecturalstyletypeid"]))))
    v["i18"] = 0.050000*np.tanh(((-(data["loopropertylandusetypeid"])) * ((data["calculatedfinishedsquarefeet"] * data["calculatedfinishedsquarefeet"]) + np.tanh((data["latitude"] + data["finishedsquarefeet12"])))))
    v["i19"] = 0.050000*np.tanh((((data["loopropertylandusetypeid"] - data["loopropertycountylandusecode"]) * data["landtaxvaluedollarcnt"]) - (((data["loopropertycountylandusecode"] * data["propertyzoningdesc"]) + data["loostorytypeid"])/2.0)))
    v["i20"] = 0.050000*np.tanh((data["lotsizesquarefeet"] * (data["garagetotalsqft"] * ((data["fireplacecnt"] + (((data["garagetotalsqft"] + (2.17611956596374512)) + data["loopooltypeid7"])/2.0))/2.0))))
    v["i21"] = 0.050000*np.tanh((np.tanh(((((data["taxvaluedollarcnt"] * (data["taxvaluedollarcnt"] + data["fullbathcnt"])) * 2.0) * 2.0) * 2.0)) * data["unitcnt"]))
    v["i22"] = 0.050000*np.tanh((((data["finishedsquarefeet13"] * np.tanh(np.tanh((1.322580 + (data["nans"] - data["longitude"]))))) * 2.0) * 2.0))
    v["i23"] = 0.050000*np.tanh(((data["unitcnt"] * (data["loohashottuborspa"] - (((data["unitcnt"] + data["structuretaxvaluedollarcnt"])/2.0) - data["latitude"]))) / 2.0))
    v["i24"] = 0.049980*np.tanh((data["loopropertylandusetypeid"] * (2.261900 + ((data["longitude"] + ((data["yardbuildingsqft17"] - data["numberofstories"]) - data["month"]))/2.0))))
    v["i25"] = 0.049970*np.tanh(((np.tanh(data["looregionidcity"]) / 2.0) * (((-1.0 + ((data["looregionidcity"] / 2.0) * data["looregionidcity"]))/2.0) / 2.0)))
    v["i26"] = 0.050000*np.tanh((((data["assessmentyear"] - data["finishedsquarefeet13"]) * ((data["loofireplaceflag"] + (data["looheatingorsystemtypeid"] * 2.0)) + data["finishedsquarefeet13"])) / 2.0))
    v["i27"] = 0.049990*np.tanh((((data["finishedsquarefeet12"] * (data["looregionidzip"] * ((((data["looregionidzip"] + data["longitude"])/2.0) + 0.533333)/2.0))) / 2.0) / 2.0))
    v["i28"] = 0.050000*np.tanh(((data["longitude"] * (((data["loopropertylandusetypeid"] * data["finishedsquarefeet50"]) + (data["taxvaluedollarcnt"] * (data["loobuildingqualitytypeid"] / 2.0)))/2.0)) / 2.0))
    v["i29"] = 0.021980*np.tanh((data["structuretaxvaluedollarcnt"] * (((((data["finishedsquarefeet50"] * data["fullbathcnt"]) * 2.0) - data["looregionidneighborhood"]) * 2.0) * data["finishedsquarefeet15"])))
    v["i30"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] * (np.tanh(((np.tanh(data["bathroomcnt"]) - data["garagetotalsqft"]) - data["landtaxvaluedollarcnt"])) - data["unitcnt"])))
    v["i31"] = 0.050000*np.tanh(((data["looregionidneighborhood"] * ((((data["looregionidcity"] + data["lootaxdelinquencyflag"])/2.0) + (data["looregionidneighborhood"] * (-(data["looairconditioningtypeid"]))))/2.0)) / 2.0))
    v["i32"] = 0.050000*np.tanh((data["bedroomcnt"] * (data["loopropertylandusetypeid"] * (data["propertyzoningdesc"] - (((data["finishedfloor1squarefeet"] + data["calculatedfinishedsquarefeet"])/2.0) - data["looregionidzip"])))))
    v["i33"] = 0.050000*np.tanh((data["unitcnt"] * (data["loofips"] * ((((-(data["loopropertycountylandusecode"])) * data["rawcensustractandblock"]) + data["calculatedbathnbr"])/2.0))))
    v["i34"] = 0.050000*np.tanh((data["loopropertylandusetypeid"] * (-((((data["structuretaxvaluedollarcnt"] * data["looregionidneighborhood"]) - data["looregionidcity"]) - (-(data["looregionidzip"])))))))
    v["i35"] = 0.050000*np.tanh((-(((data["finishedsquarefeet6"] * (data["finishedsquarefeet6"] * np.tanh((data["yearbuilt"] + data["looregionidzip"])))) * data["finishedsquarefeet6"]))))
    v["i36"] = 0.050000*np.tanh((data["finishedsquarefeet15"] * ((data["looheatingorsystemtypeid"] + (data["taxdelinquencyyear"] + (data["calculatedfinishedsquarefeet"] * (data["looheatingorsystemtypeid"] - data["looregionidzip"]))))/2.0)))
    v["i37"] = 0.050000*np.tanh((data["lootaxdelinquencyflag"] * (data["taxamount"] * (data["finishedsquarefeet15"] + (data["loobuildingclasstypeid"] + (data["finishedsquarefeet15"] - data["loopropertycountylandusecode"]))))))
    v["i38"] = 0.050000*np.tanh((data["loobuildingqualitytypeid"] * (data["yardbuildingsqft26"] * ((data["bedroomcnt"] * 2.0) - np.tanh(data["yearbuilt"])))))
    v["i39"] = 0.050000*np.tanh(((data["fireplacecnt"] / 2.0) * (data["yardbuildingsqft26"] * (data["finishedsquarefeet50"] - (((data["yardbuildingsqft26"] / 2.0) + data["fireplacecnt"])/2.0)))))
    v["i40"] = 0.049990*np.tanh((-((data["unitcnt"] * ((data["loopropertycountylandusecode"] + ((data["taxdelinquencyyear"] + ((data["basementsqft"] * data["basementsqft"]) / 2.0))/2.0))/2.0)))))
    v["i41"] = 0.050000*np.tanh((((data["roomcnt"] + data["looregionidneighborhood"])/2.0) * (data["finishedsquarefeet6"] - (data["looregionidneighborhood"] * (data["looregionidcity"] * data["loopooltypeid2"])))))
    v["i42"] = 0.050000*np.tanh(((-((data["unitcnt"] * ((data["looregionidcity"] + ((data["unitcnt"] + data["propertyzoningdesc"])/2.0))/2.0)))) * data["lootaxdelinquencyflag"]))
    v["i43"] = 0.046890*np.tanh((np.tanh((data["lotsizesquarefeet"] + (data["nans"] * (data["taxvaluedollarcnt"] + data["lootaxdelinquencyflag"])))) * data["unitcnt"]))
    v["i44"] = 0.050000*np.tanh((data["looregionidneighborhood"] * (data["loopropertylandusetypeid"] * (data["yearbuilt"] + ((data["roomcnt"] + data["loohashottuborspa"]) + data["latitude"])))))
    v["i45"] = 0.045380*np.tanh((-(((data["unitcnt"] + ((-((data["numberofstories"] * data["looregionidzip"]))) + data["unitcnt"])) * data["loopropertycountylandusecode"]))))
    v["i46"] = 0.050000*np.tanh(((data["loopropertylandusetypeid"] - data["finishedsquarefeet6"]) * (data["taxamount"] + ((data["taxamount"] * data["looairconditioningtypeid"]) * data["taxamount"]))))
    v["i47"] = 0.049890*np.tanh((data["assessmentyear"] * (data["loofireplaceflag"] * np.tanh((data["loofireplaceflag"] * (-((data["looregionidzip"] + data["looregionidcity"]))))))))
    v["i48"] = 0.050000*np.tanh((data["finishedsquarefeet15"] * (((data["finishedsquarefeet15"] + data["structuretaxvaluedollarcnt"])/2.0) * ((data["longitude"] * data["longitude"]) - data["structuretaxvaluedollarcnt"]))))
    v["i49"] = 0.050000*np.tanh(((data["finishedsquarefeet13"] * np.tanh(((((data["latitude"] * data["bedroomcnt"]) + data["finishedsquarefeet12"]) * 2.0) * 2.0))) * 2.0))
    v["i50"] = 0.050000*np.tanh(((data["structuretaxvaluedollarcnt"] * ((data["structuretaxvaluedollarcnt"] - 0.671233) * data["lotsizesquarefeet"])) * data["lotsizesquarefeet"]))
    v["i51"] = 0.049220*np.tanh((((data["loopropertylandusetypeid"] * (data["loopropertylandusetypeid"] * 2.0)) * ((data["numberofstories"] * data["numberofstories"]) - data["loopropertycountylandusecode"])) * 2.0))
    v["i52"] = 0.050000*np.tanh((data["assessmentyear"] * ((data["looregionidcity"] - data["loobuildingqualitytypeid"]) - ((data["propertyzoningdesc"] + data["taxdelinquencyyear"]) + data["looheatingorsystemtypeid"]))))
    v["i53"] = 0.043620*np.tanh(((data["finishedsquarefeet6"] - ((((data["loopooltypeid2"] * data["garagecarcnt"]) + data["loopooltypeid10"])/2.0) * data["latitude"])) * data["loopooltypeid7"]))
    v["i54"] = 0.050000*np.tanh(((data["structuretaxvaluedollarcnt"] * (data["longitude"] * (data["structuretaxvaluedollarcnt"] + (-(3.863640))))) * data["threequarterbathnbr"]))
    v["i55"] = 0.046670*np.tanh(((((data["fireplacecnt"] + 0.596774) * data["loohashottuborspa"]) * data["threequarterbathnbr"]) + (data["loopropertycountylandusecode"] * data["loohashottuborspa"])))
    v["i56"] = 0.050000*np.tanh((data["censustractandblock"] * ((data["loopropertycountylandusecode"] / 2.0) * ((data["loopropertycountylandusecode"] * data["rawcensustractandblock"]) - data["censustractandblock"]))))
    v["i57"] = 0.050000*np.tanh((((data["loopooltypeid10"] + (data["lootaxdelinquencyflag"] + -2.0)) * data["looregionidcity"]) * (data["lootypeconstructiontypeid"] * data["looregionidcity"])))
    v["i58"] = 0.049990*np.tanh((-(((((np.tanh(data["bedroomcnt"]) * data["bedroomcnt"]) + data["garagecarcnt"]) * data["censustractandblock"]) * data["loopropertylandusetypeid"]))))
    v["i59"] = 0.050000*np.tanh((data["finishedsquarefeet13"] * ((np.tanh(data["garagetotalsqft"]) * ((-(data["loofips"])) - data["loofips"])) - data["loofips"])))
    v["i60"] = 0.049980*np.tanh(((((data["structuretaxvaluedollarcnt"] * data["structuretaxvaluedollarcnt"]) + data["looairconditioningtypeid"]) * (data["structuretaxvaluedollarcnt"] * data["looairconditioningtypeid"])) * data["loopropertycountylandusecode"]))
    v["i61"] = 0.050000*np.tanh(((data["loopropertycountylandusecode"] - data["looregionidneighborhood"]) * (data["taxdelinquencyyear"] * (((data["garagetotalsqft"] * data["looregionidneighborhood"]) + data["propertyzoningdesc"])/2.0))))
    v["i62"] = 0.050000*np.tanh((data["bedroomcnt"] * (((data["threequarterbathnbr"] + (data["censustractandblock"] + (data["bedroomcnt"] * data["looairconditioningtypeid"])))/2.0) * data["loopropertycountylandusecode"])))
    v["i63"] = 0.038650*np.tanh(((((data["finishedsquarefeet13"] * 2.0) * 2.0) * 2.0) * (data["censustractandblock"] - ((data["finishedsquarefeet13"] + data["loofips"])/2.0))))
    v["i64"] = 0.050000*np.tanh((data["finishedsquarefeet15"] * ((data["latitude"] + ((((data["yardbuildingsqft17"] + data["bedroomcnt"])/2.0) * data["latitude"]) * data["loopooltypeid7"]))/2.0)))
    v["i65"] = 0.047770*np.tanh((data["garagetotalsqft"] * ((((np.tanh((data["lootypeconstructiontypeid"] * 2.0)) * np.tanh(data["bathroomcnt"])) * 2.0) * 2.0) * 2.0)))
    v["i66"] = 0.046240*np.tanh((data["yearbuilt"] * (data["numberofstories"] * (data["finishedsquarefeet13"] * ((data["longitude"] - data["yardbuildingsqft26"]) + data["garagecarcnt"])))))
    v["i67"] = 0.050000*np.tanh((data["looregionidcity"] * (data["loopropertylandusetypeid"] * np.tanh((data["bathroomcnt"] + (data["fullbathcnt"] + 0.419355))))))
    v["i68"] = 0.046150*np.tanh((((((data["assessmentyear"] * data["finishedsquarefeet12"]) * 2.0) * np.tanh(data["looregionidzip"])) * 2.0) * 2.0))
    v["i69"] = 0.050000*np.tanh((data["calculatedbathnbr"] * (((data["unitcnt"] + data["looregionidneighborhood"])/2.0) * (data["lootaxdelinquencyflag"] * data["finishedsquarefeet15"]))))
    v["i70"] = 0.050000*np.tanh(((-(np.tanh(np.tanh(data["structuretaxvaluedollarcnt"])))) * (data["finishedsquarefeet6"] * ((data["finishedsquarefeet6"] + (data["looregionidneighborhood"] * 2.0))/2.0))))
    v["i71"] = 0.049960*np.tanh(((data["yardbuildingsqft17"] * data["loopropertycountylandusecode"]) * ((data["loopropertycountylandusecode"] + data["garagecarcnt"]) - (data["looregionidneighborhood"] + data["looregionidcity"]))))
    v["i72"] = 0.029720*np.tanh(((data["looregionidneighborhood"] * (((((data["bathroomcnt"] + data["poolsizesum"])/2.0) + np.tanh(data["bathroomcnt"]))/2.0) * data["numberofstories"])) / 2.0))
    v["i73"] = 0.050000*np.tanh(((data["finishedsquarefeet13"] + (((data["looregionidneighborhood"] * data["looregionidneighborhood"]) * data["loopropertycountylandusecode"]) * data["garagetotalsqft"]))/2.0))
    v["i74"] = 0.050000*np.tanh(((data["rawcensustractandblock"] * ((((data["looheatingorsystemtypeid"] + data["loobuildingclasstypeid"])/2.0) + data["loohashottuborspa"])/2.0)) * (-(data["unitcnt"]))))
    v["i75"] = 0.049980*np.tanh((data["loopropertycountylandusecode"] * ((data["looregionidzip"] * 2.0) * ((data["looheatingorsystemtypeid"] + (data["loopropertycountylandusecode"] * data["finishedsquarefeet12"]))/2.0))))
    v["i76"] = 0.050000*np.tanh((data["finishedsquarefeet15"] * ((((np.tanh(data["looregionidneighborhood"]) + data["yardbuildingsqft17"])/2.0) + (data["fireplacecnt"] * data["fullbathcnt"]))/2.0)))
    v["i77"] = 0.039310*np.tanh((((np.tanh(data["loopooltypeid2"]) + (((data["landtaxvaluedollarcnt"] * data["finishedsquarefeet13"]) + (data["yardbuildingsqft26"] / 2.0))/2.0))/2.0) * data["landtaxvaluedollarcnt"]))
    v["i78"] = 0.050000*np.tanh((((data["loobuildingclasstypeid"] * ((((5.0) - data["month"]) * data["month"]) + data["month"])) * 2.0) * 2.0))
    v["i79"] = 0.050000*np.tanh((((data["finishedsquarefeet6"] * ((data["looregionidcity"] * data["numberofstories"]) * data["finishedsquarefeet6"])) + (data["loopooltypeid2"] * data["finishedsquarefeet6"]))/2.0))
    return Outputs(v.sum(axis=1))*(2*.418)-.418


def GP5(data):
    v = pd.DataFrame()
    v["i0"] = 0.050000*np.tanh(((data["looregionidzip"] + (1.166670 + (data["finishedsquarefeet12"] + ((data["looregionidneighborhood"] + ((data["loohashottuborspa"] + data["looregionidcity"])/2.0))/2.0))))/2.0))
    v["i1"] = 0.049220*np.tanh(np.tanh((((data["loopropertycountylandusecode"] + (data["taxvaluedollarcnt"] * (data["taxamount"] - data["loopooltypeid2"]))) * 2.0) * 2.0)))
    v["i2"] = 0.050000*np.tanh((((data["calculatedfinishedsquarefeet"] + (((((data["landtaxvaluedollarcnt"] + data["lootaxdelinquencyflag"])/2.0) + data["looregionidcity"])/2.0) - data["taxamount"]))/2.0) + data["loopropertycountylandusecode"]))
    v["i3"] = 0.049980*np.tanh((((((-(((data["finishedsquarefeet50"] + data["looregionidcounty"])/2.0))) / 2.0) + data["loopropertycountylandusecode"])/2.0) + (data["loopropertycountylandusecode"] * data["latitude"])))
    v["i4"] = 0.050000*np.tanh(((data["calculatedfinishedsquarefeet"] * ((np.tanh(data["loohashottuborspa"]) + ((data["garagecarcnt"] + data["calculatedfinishedsquarefeet"])/2.0))/2.0)) - data["loodecktypeid"]))
    v["i5"] = 0.050000*np.tanh(((data["loopropertycountylandusecode"] * (data["loopooltypeid7"] - data["censustractandblock"])) - ((-(data["taxvaluedollarcnt"])) + data["taxamount"])))
    v["i6"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] + (data["unitcnt"] * (data["looregionidcity"] + (((data["unitcnt"] * data["latitude"]) * 2.0) * 2.0)))))
    v["i7"] = 0.050000*np.tanh(np.tanh(((data["poolsizesum"] * (data["loopooltypeid2"] - data["poolsizesum"])) - (np.tanh(data["looairconditioningtypeid"]) * data["garagetotalsqft"]))))
    v["i8"] = 0.050000*np.tanh(((data["month"] * (data["lootypeconstructiontypeid"] * 2.0)) + (data["month"] * (data["month"] * (-(data["lootypeconstructiontypeid"]))))))
    v["i9"] = 0.044080*np.tanh((np.tanh((((data["looarchitecturalstyletypeid"] * 2.0) * 2.0) - (data["structuretaxvaluedollarcnt"] - np.tanh(np.tanh(data["calculatedfinishedsquarefeet"]))))) / 2.0))
    v["i10"] = 0.048090*np.tanh(((((data["looregionidneighborhood"] / 2.0) + np.tanh(np.tanh((data["garagetotalsqft"] * (-(data["looheatingorsystemtypeid"]))))))/2.0) / 2.0))
    v["i11"] = 0.050000*np.tanh(((data["looregionidzip"] * (((data["calculatedfinishedsquarefeet"] / 2.0) - data["unitcnt"]) - (data["looregionidcity"] * data["unitcnt"]))) / 2.0))
    v["i12"] = 0.045390*np.tanh(((np.tanh(((data["latitude"] + np.tanh(data["longitude"])) * data["taxamount"])) + (data["lotsizesquarefeet"] * data["taxamount"]))/2.0))
    v["i13"] = 0.049990*np.tanh(((((data["unitcnt"] * np.tanh(((data["landtaxvaluedollarcnt"] * data["taxvaluedollarcnt"]) * 2.0))) * 2.0) * 2.0) - data["unitcnt"]))
    v["i14"] = 0.050000*np.tanh((((((((data["garagecarcnt"] + (data["bathroomcnt"] - data["loostorytypeid"]))/2.0) - data["bedroomcnt"]) + data["calculatedfinishedsquarefeet"])/2.0) / 2.0) / 2.0))
    v["i15"] = 0.043850*np.tanh((data["looarchitecturalstyletypeid"] * ((((data["looheatingorsystemtypeid"] * 2.0) * 2.0) + data["looarchitecturalstyletypeid"]) + ((data["loopooltypeid10"] + data["loofireplaceflag"])/2.0))))
    v["i16"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] * (-(((((data["landtaxvaluedollarcnt"] + data["censustractandblock"])/2.0) + (data["unitcnt"] * 2.0)) + data["looregionidcity"])))))
    v["i17"] = 0.050000*np.tanh(((data["loobuildingclasstypeid"] + np.tanh(((data["yearbuilt"] * (data["taxamount"] / 2.0)) - np.tanh((data["taxamount"] / 2.0)))))/2.0))
    v["i18"] = 0.050000*np.tanh((data["finishedsquarefeet13"] * ((np.tanh((-(data["looregionidcity"]))) + (data["landtaxvaluedollarcnt"] - data["looregionidcounty"])) - data["yardbuildingsqft17"])))
    v["i19"] = 0.050000*np.tanh((data["loopropertylandusetypeid"] * (-(((data["finishedfloor1squarefeet"] + (np.tanh(data["nans"]) + (data["basementsqft"] * data["basementsqft"])))/2.0)))))
    v["i20"] = 0.050000*np.tanh((data["finishedsquarefeet6"] * (data["roomcnt"] - ((((data["looregionidzip"] + data["roomcnt"])/2.0) + data["yearbuilt"]) * data["numberofstories"]))))
    v["i21"] = 0.047810*np.tanh((data["loopropertylandusetypeid"] * ((data["garagetotalsqft"] * ((3.0 + data["garagetotalsqft"])/2.0)) - ((data["yearbuilt"] + data["numberofstories"])/2.0))))
    v["i22"] = 0.050000*np.tanh((((((data["structuretaxvaluedollarcnt"] * ((data["taxvaluedollarcnt"] + (-(data["loobuildingqualitytypeid"])))/2.0)) / 2.0) / 2.0) * data["taxvaluedollarcnt"]) / 2.0))
    v["i23"] = 0.049880*np.tanh((((((data["lotsizesquarefeet"] * data["loopropertycountylandusecode"]) * (data["censustractandblock"] - data["loobuildingqualitytypeid"])) * 2.0) * 2.0) * 2.0))
    v["i24"] = 0.050000*np.tanh((-(((data["month"] * (data["month"] * data["loostorytypeid"])) * ((data["month"] / 2.0) - 1.904760)))))
    v["i25"] = 0.050000*np.tanh(((data["loopropertylandusetypeid"] + np.tanh(((data["fireplacecnt"] + (data["taxamount"] * (data["fireplacecnt"] * (-(data["fireplacecnt"])))))/2.0)))/2.0))
    v["i26"] = 0.039400*np.tanh(((data["rawcensustractandblock"] * (data["loopooltypeid2"] * (data["landtaxvaluedollarcnt"] - 0.264706))) - (data["finishedfloor1squarefeet"] * data["loopropertylandusetypeid"])))
    v["i27"] = 0.050000*np.tanh((data["yearbuilt"] * (data["loopropertylandusetypeid"] * (np.tanh(((np.tanh(data["looregionidzip"]) + data["calculatedfinishedsquarefeet"]) * 2.0)) * 2.0))))
    v["i28"] = 0.044440*np.tanh((-((data["unitcnt"] * ((data["fullbathcnt"] + ((data["looregionidneighborhood"] * data["finishedsquarefeet12"]) + data["yearbuilt"]))/2.0)))))
    v["i29"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] * ((((data["loohashottuborspa"] + (np.tanh(data["numberofstories"]) + data["looarchitecturalstyletypeid"]))/2.0) - data["unitcnt"]) * 2.0)))
    v["i30"] = 0.050000*np.tanh((data["looregionidzip"] * np.tanh((-((((data["finishedsquarefeet15"] + (data["loopropertylandusetypeid"] * data["looregionidzip"]))/2.0) * data["bathroomcnt"]))))))
    v["i31"] = 0.037880*np.tanh((data["unitcnt"] * np.tanh((((data["unitcnt"] / 2.0) + (-(data["nans"])))/2.0))))
    v["i32"] = 0.047850*np.tanh(np.tanh((((np.tanh((data["month"] - 3.0)) * 2.0) - np.tanh(3.0)) * data["unitcnt"])))
    v["i33"] = 0.044520*np.tanh((data["loopropertycountylandusecode"] * ((data["lotsizesquarefeet"] + ((np.tanh(((data["looairconditioningtypeid"] + data["threequarterbathnbr"])/2.0)) * 2.0) * 2.0)) * 2.0)))
    v["i34"] = 0.008860*np.tanh(np.tanh((data["bedroomcnt"] * ((np.tanh((((data["taxvaluedollarcnt"] * data["unitcnt"]) * 2.0) * 2.0)) + data["finishedsquarefeet15"])/2.0))))
    v["i35"] = 0.044660*np.tanh(((((data["garagetotalsqft"] + data["unitcnt"]) * 2.0) + data["looairconditioningtypeid"]) * (data["unitcnt"] * data["structuretaxvaluedollarcnt"])))
    v["i36"] = 0.050000*np.tanh(((data["taxamount"] * (data["calculatedfinishedsquarefeet"] * ((data["finishedsquarefeet6"] * (data["loobuildingqualitytypeid"] - data["calculatedbathnbr"])) * 2.0))) * 2.0))
    v["i37"] = 0.049950*np.tanh((-((data["loopropertylandusetypeid"] * (((data["numberofstories"] + data["looregionidzip"]) + (np.tanh(data["loodecktypeid"]) * data["loodecktypeid"]))/2.0)))))
    v["i38"] = 0.048210*np.tanh(((((data["taxdelinquencyyear"] * data["finishedsquarefeet15"]) * (data["finishedsquarefeet15"] * data["taxdelinquencyyear"])) + (data["loobuildingqualitytypeid"] * data["finishedsquarefeet15"]))/2.0))
    v["i39"] = 0.050000*np.tanh((((data["looarchitecturalstyletypeid"] * np.tanh((((data["looarchitecturalstyletypeid"] + (data["looregionidcity"] * 2.0)) * 2.0) * 2.0))) * 2.0) * 2.0))
    v["i40"] = 0.050000*np.tanh((-((((((data["loopropertycountylandusecode"] + data["looregionidzip"])/2.0) + data["basementsqft"])/2.0) * (data["looregionidzip"] * data["loopropertycountylandusecode"])))))
    v["i41"] = 0.049970*np.tanh(((((data["finishedsquarefeet12"] + data["finishedsquarefeet15"]) + data["looregionidneighborhood"]) * data["latitude"]) * (data["finishedsquarefeet15"] * 2.0)))
    v["i42"] = 0.049970*np.tanh(((data["loopropertycountylandusecode"] + data["finishedsquarefeet15"]) * np.tanh(((data["numberofstories"] - np.tanh(data["landtaxvaluedollarcnt"])) - data["censustractandblock"]))))
    v["i43"] = 0.042920*np.tanh((data["loopropertycountylandusecode"] * (data["looheatingorsystemtypeid"] + ((data["censustractandblock"] * data["loopropertycountylandusecode"]) - 0.563830))))
    v["i44"] = 0.050000*np.tanh((data["loopropertylandusetypeid"] * ((np.tanh((data["finishedsquarefeet12"] * ((data["taxvaluedollarcnt"] - data["calculatedfinishedsquarefeet"]) * 2.0))) * 2.0) * 2.0)))
    v["i45"] = 0.044000*np.tanh((data["finishedsquarefeet13"] * ((((-(data["propertyzoningdesc"])) + (-((data["propertyzoningdesc"] + 3.071430))))/2.0) - data["finishedsquarefeet13"])))
    v["i46"] = 0.049970*np.tanh((np.tanh((((-(data["threequarterbathnbr"])) + ((data["finishedsquarefeet50"] * (data["yardbuildingsqft17"] * data["landtaxvaluedollarcnt"])) / 2.0))/2.0)) / 2.0))
    v["i47"] = 0.049940*np.tanh(np.tanh(np.tanh(((((1.0 + np.tanh((-(data["month"])))) * 2.0) - data["assessmentyear"]) * 2.0))))
    v["i48"] = 0.050000*np.tanh(((data["looregionidcity"] * (data["looregionidcity"] * (((data["lotsizesquarefeet"] + ((data["calculatedfinishedsquarefeet"] + data["loohashottuborspa"])/2.0))/2.0) / 2.0))) / 2.0))
    v["i49"] = 0.050000*np.tanh(((data["looregionidzip"] * data["unitcnt"]) * ((data["taxdelinquencyyear"] + (data["looregionidcity"] * (data["looregionidzip"] * data["unitcnt"])))/2.0)))
    v["i50"] = 0.050000*np.tanh((data["lootypeconstructiontypeid"] * ((0.541667 - (data["looregionidzip"] * data["looregionidzip"])) - (data["looregionidzip"] * data["looregionidzip"]))))
    v["i51"] = 0.050000*np.tanh(np.tanh((data["finishedsquarefeet13"] * ((data["yearbuilt"] * data["taxamount"]) + (data["month"] - ((3.0) * 2.0))))))
    v["i52"] = 0.049980*np.tanh(((data["assessmentyear"] * ((((data["bedroomcnt"] - data["bathroomcnt"]) - data["looregionidneighborhood"]) * data["bedroomcnt"]) * 2.0)) * 2.0))
    v["i53"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] * (((data["taxdelinquencyyear"] - (data["loopropertycountylandusecode"] * data["fullbathcnt"])) + (data["poolsizesum"] * data["fullbathcnt"]))/2.0)))
    v["i54"] = 0.050000*np.tanh((data["loopropertycountylandusecode"] * (np.tanh((np.tanh(data["loobuildingqualitytypeid"]) - data["landtaxvaluedollarcnt"])) + (data["loopooltypeid2"] * data["looregionidzip"]))))
    v["i55"] = 0.046900*np.tanh(((np.tanh(data["finishedsquarefeet12"]) * ((data["taxvaluedollarcnt"] - data["loobuildingqualitytypeid"]) - data["looregionidneighborhood"])) * data["taxdelinquencyyear"]))
    v["i56"] = 0.033740*np.tanh(((((data["looheatingorsystemtypeid"] * np.tanh(((10.0) * data["taxdelinquencyyear"]))) + (data["longitude"] * data["taxdelinquencyyear"]))/2.0) / 2.0))
    v["i57"] = 0.050000*np.tanh(((data["unitcnt"] * ((data["looregionidcity"] * data["bathroomcnt"]) + data["lotsizesquarefeet"])) * (data["looregionidcity"] * data["lotsizesquarefeet"])))
    v["i58"] = 0.049990*np.tanh((data["lootypeconstructiontypeid"] * ((((data["yardbuildingsqft17"] * data["loodecktypeid"]) + data["landtaxvaluedollarcnt"])/2.0) + (data["looregionidneighborhood"] * data["looregionidneighborhood"]))))
    v["i59"] = 0.045460*np.tanh(((np.tanh(np.tanh((((data["looregionidneighborhood"] * data["looregionidneighborhood"]) - np.tanh(data["looregionidneighborhood"])) * 2.0))) / 2.0) / 2.0))
    v["i60"] = 0.040140*np.tanh((data["loopropertylandusetypeid"] * (data["calculatedbathnbr"] * (data["looregionidcity"] + ((data["looregionidcity"] + (data["looregionidcity"] - data["finishedsquarefeet12"]))/2.0)))))
    v["i61"] = 0.039520*np.tanh(((np.tanh(np.tanh((data["censustractandblock"] * (data["censustractandblock"] * data["finishedsquarefeet13"])))) - data["finishedsquarefeet13"]) - data["finishedsquarefeet13"]))
    v["i62"] = 0.050000*np.tanh(((data["loopropertycountylandusecode"] * data["looregionidzip"]) * (((data["loopooltypeid10"] + (data["looheatingorsystemtypeid"] - data["yardbuildingsqft17"]))/2.0) - data["poolsizesum"])))
    v["i63"] = 0.049910*np.tanh(((((np.tanh(data["loopooltypeid10"]) - data["threequarterbathnbr"]) + data["yearbuilt"])/2.0) * np.tanh((data["loopooltypeid10"] - data["basementsqft"]))))
    v["i64"] = 0.047310*np.tanh((((data["taxdelinquencyyear"] * (data["taxvaluedollarcnt"] / 2.0)) * (data["structuretaxvaluedollarcnt"] + data["propertyzoningdesc"])) / 2.0))
    v["i65"] = 0.050000*np.tanh((data["yardbuildingsqft26"] * ((((((data["loobuildingclasstypeid"] + data["looregionidcity"])/2.0) + data["loopooltypeid7"])/2.0) + (data["yardbuildingsqft17"] * data["loofips"]))/2.0)))
    v["i66"] = 0.050000*np.tanh((data["loopooltypeid10"] * (data["lootypeconstructiontypeid"] + ((data["garagecarcnt"] * (data["threequarterbathnbr"] * data["loopooltypeid2"])) * data["landtaxvaluedollarcnt"]))))
    v["i67"] = 0.049990*np.tanh(((((((data["fireplacecnt"] * data["lotsizesquarefeet"]) + data["loopooltypeid2"])/2.0) + data["finishedfloor1squarefeet"])/2.0) * (data["fireplacecnt"] * data["lotsizesquarefeet"])))
    v["i68"] = 0.050000*np.tanh((((data["yardbuildingsqft26"] * data["yardbuildingsqft26"]) + 1.561400) * (data["yardbuildingsqft26"] * (data["loobuildingqualitytypeid"] * data["bedroomcnt"]))))
    v["i69"] = 0.047480*np.tanh(np.tanh((np.tanh(data["loopropertycountylandusecode"]) * ((data["threequarterbathnbr"] + (16.600000 * (data["loopropertylandusetypeid"] * data["loofips"])))/2.0))))
    v["i70"] = 0.049980*np.tanh(((data["looairconditioningtypeid"] - (data["yearbuilt"] - data["loobuildingqualitytypeid"])) * (data["yearbuilt"] * (data["finishedsquarefeet12"] * data["finishedsquarefeet13"]))))
    v["i71"] = 0.050000*np.tanh((data["finishedsquarefeet13"] * (data["bathroomcnt"] + (data["looregionidneighborhood"] * (data["lootaxdelinquencyflag"] * (data["looregionidneighborhood"] * data["looregionidneighborhood"]))))))
    v["i72"] = 0.043200*np.tanh(((((data["taxvaluedollarcnt"] - data["taxamount"]) + data["looheatingorsystemtypeid"])/2.0) * (data["looheatingorsystemtypeid"] * (data["taxvaluedollarcnt"] - data["taxamount"]))))
    v["i73"] = 0.050000*np.tanh((np.tanh(((data["bathroomcnt"] + (1.788460 + data["nans"])) - data["loopropertylandusetypeid"])) * (-(data["loopropertylandusetypeid"]))))
    v["i74"] = 0.049990*np.tanh((data["lotsizesquarefeet"] * (data["calculatedbathnbr"] * (data["unitcnt"] * ((data["looregionidneighborhood"] * data["looregionidzip"]) + data["calculatedbathnbr"])))))
    v["i75"] = 0.050000*np.tanh((((data["loobuildingqualitytypeid"] + (-((data["yardbuildingsqft26"] * data["structuretaxvaluedollarcnt"])))) * 2.0) * data["yardbuildingsqft26"]))
    v["i76"] = 0.038990*np.tanh((data["loopropertycountylandusecode"] * (((data["loopooltypeid7"] + data["yardbuildingsqft26"])/2.0) * (data["fireplacecnt"] + (data["yardbuildingsqft26"] / 2.0)))))
    v["i77"] = 0.049990*np.tanh((data["structuretaxvaluedollarcnt"] * (data["finishedsquarefeet6"] * ((data["finishedsquarefeet6"] * data["fireplacecnt"]) - (data["yearbuilt"] + data["fullbathcnt"])))))
    v["i78"] = 0.050000*np.tanh((data["finishedsquarefeet6"] * ((((data["fireplacecnt"] + data["looregionidneighborhood"])/2.0) + (data["yardbuildingsqft17"] * (data["loopooltypeid7"] + data["loopooltypeid7"])))/2.0)))
    v["i79"] = 0.050000*np.tanh((np.tanh((-(data["finishedsquarefeet15"]))) * (((data["finishedsquarefeet15"] * data["propertyzoningdesc"]) + data["fireplacecnt"])/2.0)))
    return Outputs(v.sum(axis=1))*(2*.418)-.418


def GP(data):
    return (GP1(data) +
            GP2(data) +
            GP3(data) +
            GP4(data) +
            GP5(data))/5.


# Now to create the submission

# In[ ]:


sub = pd.read_csv(directory+'sample_submission.csv')
testpreds = GP(test)
for i, c in enumerate(sub.columns[sub.columns != 'ParcelId']):
    sub[c] = testpreds
sub.to_csv('xxx.csv.gz',index=False,compression='gzip')

