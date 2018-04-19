
# coding: utf-8

# Updates:
# 
# The stacking model:
# https://www.kaggle.com/schoolpal/nn-stacking-magic-no-magic-30409-private-31063
# 
# ----------------------------------
# 
# This is the DNN model which also used as one basis model for the stacking. The LB score at 0.308, and LB 0.306 when linearly combined with the XGB and LGB results produced using
# https://www.kaggle.com/schoolpal/lgbm-lb-0-3093-0-3094 and https://www.kaggle.com/schoolpal/modifications-to-reynaldo-s-script. 
# 
# The model uses feature from to other models (XGB, LGB), so it is also a stacking model.  If the XGB and LGB scores were not used, the DNN's performance is at LB 0.312-0.313,.
# 
# The macro data and a few FE were used for the DNN model. If you are interested, read the comments in prepare_data() and nn().
# 
# I set the epochs to 1 in order to bypass the limit on running time for kernel. The epochs should be 40. 
# 
# If you want to try it on your own machine, use Tensorflow backend. You will need GPU to run this script (took 10 minutes on GTX 1080). Btw, I am migrating to Pytorch.

# In[ ]:



import os
import sys
from keras.layers.advanced_activations import *
from keras.callbacks import LearningRateScheduler
import pickle

from keras.layers.merge import *
from keras.layers.noise import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,Imputer,RobustScaler
from keras.optimizers import SGD,RMSprop,Adam,Adadelta
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Input, Embedding,Flatten
from keras.layers.normalization import BatchNormalization
import pandas as pd
import numpy as np
np.random.seed(1)
import pdb
import keras
NORM=False
LOG=True
def last_days(train,day_range=30,cols=['full_sq'],changes=False):
    days=train.set_index('timestamp').groupby(pd.TimeGrouper('D'))
    for col in cols:
        means=days[col].mean()
        if changes:
            means=means.pct_change()
        means_avg=dict()
        for i in range(len(means)):
            key=means.index[i].year*10000+means.index[i].month*100+means.index[i].day
            if i>0:
                start=max(i-day_range,0)
                means_avg[key]=means.iloc[start:i].mean()
            else:
                means_avg[key]=0
        val_list=[]
        for i in range(len(train)):
            t=train.iloc[i].timestamp
            key=t.year*10000+t.month*100+t.day
            val_list.append(means_avg[key])
        train[col+'_avg'+str(day_range)]=val_list
    return train
def fill_maxfloor(df_all):
    # Bhavesh Ghodasara's idea
    apartments=df_all.sub_area + df_all['ID_bus_terminal'].astype(str)+df_all['bus_terminal_avto_km'].astype(str)
    df_all['an']=apartments
    df_all.loc[df_all['max_floor']==0,'max_floor']=np.NaN
    grouped_years=df_all.groupby(['an']).max_floor
    an_years=grouped_years.median()
    an_years_max=grouped_years.max()
    an_years_min=grouped_years.min()
    an_years[an_years_max-an_years_min>1]=np.NaN
    df_all=df_all.join(an_years,on='an',rsuffix='_an')
    df_all.loc[df_all.max_floor.isnull(),'max_floor']=df_all['max_floor_an'][df_all.max_floor.isnull()]
    df_all.drop(['an','max_floor_an'],inplace=True,axis=1)
    return df_all

def fill_years(df_all,threshold=3,preprocess=True):
    # Bhavesh Ghodasara's idea
    apartments=df_all.sub_area + df_all['ID_bus_terminal'].astype(str)+df_all['bus_terminal_avto_km'].astype(str)
    df_all['an']=apartments
    build_years=df_all.build_year.copy()
    if preprocess:
        df_all.loc[df_all['build_year']<1,'build_year']=np.NaN
    grouped_years=df_all.groupby(['an']).build_year
    an_years=grouped_years.median()
    an_years_max=grouped_years.max()
    an_years_min=grouped_years.min()
    an_years[an_years_max-an_years_min>threshold]=np.NaN
    df_all=df_all.join(an_years,on='an',rsuffix='_an')
    df_all.loc[df_all.build_year.isnull(),'build_year']=df_all['build_year_an'][df_all.build_year.isnull()]
    if not preprocess:
        df_all.loc[df_all.build_year.isnull(),'build_year']=build_years[df_all.build_year.isnull()]
    df_all.drop(['an','build_year_an'],inplace=True,axis=1)

    return df_all
def step_decay(epoch):
    lr=0.01
    start=15
    step=5
    if epoch<start:
        return lr
    else:
        lr=lr/np.power(2.0,(1+(epoch-start)/step))
        return lr
def bad_weights(train,prices,price_sq):
    weights=np.ones(len(price_sq))
    weights[(train.product_type=='Investment') & (price_sq<40000)]=0.1
    return weights

def get_excluded():
    # Taken from wti200's kernel
    excluded={
        "young_male", "school_education_centers_top_20_raion", "0_17_female", "railroad_1line", "7_14_female", "0_17_all", "children_school","ecology", "16_29_male", "mosque_count_3000", "female_f", "church_count_1000", "railroad_terminal_raion","mosque_count_5000", "big_road1_1line", "mosque_count_1000", "7_14_male", "0_6_female", "oil_chemistry_raion","young_all", "0_17_male", "ID_bus_terminal", "university_top_20_raion", "mosque_count_500","ID_big_road1","ID_railroad_terminal", "ID_railroad_station_walk", "ID_big_road2", "ID_metro", "ID_railroad_station_avto","0_13_all", "mosque_count_2000", "work_male", "16_29_all", "young_female", "work_female", "0_13_female","ekder_female", "7_14_all", "big_church_count_500","leisure_count_500", "cafe_sum_1500_max_price_avg", "leisure_count_2000","office_count_500", "male_f", "nuclear_reactor_raion", "0_6_male", "church_count_500", "build_count_before_1920","thermal_power_plant_raion", "cafe_count_2000_na_price", "cafe_count_500_price_high","market_count_2000", "museum_visitis_per_100_cap", "trc_count_500", "market_count_1000", "work_all", "additional_education_raion","build_count_slag", "leisure_count_1000", "0_13_male", "office_raion","raion_build_count_with_builddate_info", "market_count_3000", "ekder_all", "trc_count_1000", "build_count_1946-1970","office_count_1500", "cafe_count_1500_na_price", "big_church_count_5000", "big_church_count_1000", "build_count_foam","church_count_1500", "church_count_3000", "leisure_count_1500","16_29_female", "build_count_after_1995", "cafe_avg_price_1500", "office_sqm_1000", "cafe_avg_price_5000", "cafe_avg_price_2000","big_church_count_1500", "full_all", "cafe_sum_5000_min_price_avg","office_sqm_2000", "church_count_5000","0_6_all", "detention_facility_raion", "cafe_avg_price_3000""young_male", "school_education_centers_top_20_raion", "0_17_female", "railroad_1line", "7_14_female", "0_17_all", "children_school","ecology", "16_29_male", "mosque_count_3000", "female_f", "church_count_1000", "railroad_terminal_raion","mosque_count_5000", "big_road1_1line", "mosque_count_1000", "7_14_male", "0_6_female", "oil_chemistry_raion","young_all", "0_17_male", "ID_bus_terminal", "university_top_20_raion", "mosque_count_500","ID_big_road1","ID_railroad_terminal", "ID_railroad_station_walk", "ID_big_road2", "ID_metro", "ID_railroad_station_avto","0_13_all", "mosque_count_2000", "work_male", "16_29_all", "young_female", "work_female", "0_13_female","ekder_female", "7_14_all", "big_church_count_500","leisure_count_500", "cafe_sum_1500_max_price_avg", "leisure_count_2000","office_count_500", "male_f", "nuclear_reactor_raion", "0_6_male", "church_count_500", "build_count_before_1920","thermal_power_plant_raion", "cafe_count_2000_na_price", "cafe_count_500_price_high","market_count_2000", "museum_visitis_per_100_cap", "trc_count_500", "market_count_1000", "work_all", "additional_education_raion","build_count_slag", "leisure_count_1000", "0_13_male", "office_raion","raion_build_count_with_builddate_info", "market_count_3000", "ekder_all", "trc_count_1000", "build_count_1946-1970","office_count_1500", "cafe_count_1500_na_price", "big_church_count_5000", "big_church_count_1000", "build_count_foam","church_count_1500", "church_count_3000", "leisure_count_1500","16_29_female", "build_count_after_1995", "cafe_avg_price_1500", "office_sqm_1000", "cafe_avg_price_5000", "cafe_avg_price_2000","big_church_count_1500", "full_all", "cafe_sum_5000_min_price_avg","office_sqm_2000", "church_count_5000","0_6_all", "detention_facility_raion", "cafe_avg_price_3000"
    }
    return excluded




# In[ ]:


def prepare_data():
    # Copied wti200's kernel: from https://www.kaggle.com/wti200/deep-neural-network-for-starters-r
    excluded=get_excluded()
    df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
    df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

    #-------------------------------------
    # Note that the following is essential to get good performance:
    # You can produce these pkl by using the two kernels with 5 fold non-shuffle traning, as we usually did in stacking
    #-------------------------------------
    # https://www.kaggle.com/schoolpal/lgbm-lb-0-3093-0-3094
    # https://www.kaggle.com/schoolpal/modifications-to-reynaldo-s-script
    
    # (xgb_train,xgb_test)=pickle.load(open('xgb_predicted.pkl'))
    # (xgb_train_log,xgb_test_log)=pickle.load(open('xgb_predicted_log.pkl'))
    # (lgb_train,lgb_test)=pickle.load(open('lgb_predicted.pkl'))
    
    # df_train['xgb_score']=xgb_train
    # df_train['log_xgb_score']=np.log(xgb_train)
    # df_train['lgb_score']=lgb_train
    # df_train['lgb_score_log']=np.log(lgb_train)
    ## df_train['log_xgb_score']=xgb_train_log
    ## df_train['log_xgb_score_log']=np.log(xgb_train_log)
    # df_test['xgb_score']=xgb_test
    # df_test['log_xgb_score']=np.log(xgb_test)
    # df_test['lgb_score']=lgb_test
    # df_test['lgb_score_log']=np.log(lgb_test)
    ## df_test['log_xgb_score']=xgb_test_log
    ## df_test['log_xgb_score_log']=np.log(xgb_test_log)

    # Magic number from Andy's script (Louis?)
    df_train['price_doc']*=0.969

    full_sq=df_train.full_sq.copy()
    full_sq[full_sq<5]=np.NaN
    
    price_sq=df_train.price_doc/full_sq
    #Remove the extreme prices, took from someone's kernel (sry)
    df_train=df_train[(price_sq<600000) & (price_sq>10000)]
    price_sq=price_sq[(price_sq<600000) & (price_sq>10000)]

    y_train=df_train.price_doc
    df_train.drop(['price_doc'],inplace=True,axis=1)
    num_train=df_train.shape[0]
    da=pd.concat([df_train,df_test])
    da=da.reset_index(drop=True)
    '''
    The feature enginering part, most of the FE were took from other peole's kernel.
    last_days method adds the mean of full_sq for all the house sold in last 30 days.
    This feature was motivated from my autoregression model for monthly prices. What does this feature capture? I tried daily sum of full_sq which clearly indicates the supply and demand. However, the local CV results of monthly price prediction actually prefer mean! I think  this feature somehow captured the supply and demand for luxury or economic properties.
'''
    da=last_days(da)
    # These two features are only necessary as I removed the outlier feature values (> 4 SD) for all features, but these two are important to keep.
    da['build_year1']=((da['build_year']==1) & (da.product_type=='OwnerOccupier')).astype(int)
    da['build_year0']=((da['build_year']==0) & (da.product_type=='OwnerOccupier')).astype(int)

    # Fill some missing values based on location (Bhavesh Ghodasara's idea for
    # identify location)
    da=fill_years(da)
    da=fill_maxfloor(da)

    # Not necessary, I just fix it in order to calculate price per square meter for the sample weights
    da.loc[da['life_sq']<5,'life_sq']=np.NaN
    da.loc[da['full_sq']<5,'full_sq']=np.NaN

    # 0.7 come from the mean ratio (0.65?) between full_sq and life_sq,0.65 also works
    da['life_sq']=np.where(da.life_sq.isnull(),da.full_sq*0.7,da.life_sq)
    da['build_year']=np.where((da.build_year>1690) & (da.build_year<2020),da.build_year,np.NaN)
    da['max_floor']=np.where(da.max_floor<da.floor,da.floor+1,da.max_floor)
    da['material']=da['material'].astype(str)
    da.loc[da.state==33,'state']=3

    to_remove=[]
    product_types=pd.factorize(da.product_type)[0]
    product_types_string=da.product_type.copy()

    da['month']=da.timestamp.dt.year.astype(str)

    # The year_month feature was added to nullify  the effect of
    # "year_month" as I set the year_month of the test data to be NaN
    # I hope to nullify any effect of time. This is equivalent to say that we don't know the time for test data.
    # Any time effect must be learned from macro feature

    da['year_month']=da.timestamp.dt.year
    da['year_month']=(da['year_month']*100+da.timestamp.dt.month)
    da.loc[da['year_month']>201506,'year_month']=np.NaN
    da['year_month']=da['year_month'].astype(str)
    
    df_cat=None
    for c in da.columns:
        if da[c].dtype=='object':
            oh=pd.get_dummies(da[c],prefix=c)
            
            if df_cat is None:
                df_cat=oh
            else:
                df_cat=pd.concat([df_cat,oh],axis=1)
            to_remove.append(c)
    da.drop(to_remove,inplace=True,axis=1)
    # Remove rare one hot encoded features
    to_remove=[]
    if df_cat is not None:
        sums=df_cat.sum(axis=0)
        to_remove=sums[sums<200].index.values
        df_cat=df_cat.loc[:,df_cat.columns.difference(to_remove)]
        da = pd.concat([da, df_cat], axis=1)
    if excluded is not None:
        for c in excluded:
            if c in da.columns:
                da.drop([c],inplace=True,axis=1)
    # These additional features are taken from
    # https://www.kaggle.com/wti200/deep-neural-network-for-starters-r
    da['na_count']=da.isnull().sum(axis=1)
    da['rel_floor']=da.floor/da.max_floor
    da['diff_floor']=da.max_floor-da.floor
    da['rel_kitchen_sq']=da.kitch_sq-da.full_sq
    da['rel_life_sq']=da.life_sq/da.full_sq
    da['rel_kitch_life']=da.kitch_sq/da.life_sq
    da['rel_sq_per_floor']=da.full_sq/da.floor
    da['diff_life_sq']=da.full_sq-da.life_sq
    da['building_age']=da.timestamp.dt.year-da.build_year
    
    da['new_house_own']=((da['building_age']<=0) & (product_types_string=='OwnerOccupier')).astype(int)
    da['old_house_own']=((da['building_age']>0) & (product_types_string=='OwnerOccupier')).astype(int)
    # Macro features, finally!!!
    # The unemployment info for 2016 was missing. So the unemployment rate were taken from OCED website
    # The original unemployment data is useful, but OCED's data is better (LB score)
    # These macro features are selected from my autoregresion time series model
    # for the monthly mean prices based on the local CV results. "eurrub" and "brent" for Investment properties, and "unemployment" for OwerOccupier. 
    macro_cols=['timestamp','brent','eurrub','unemployment']
    macro=pd.read_csv('../input/macro.csv',parse_dates=['timestamp'])
    # Load the OCED unemployment
    # macro=macro_lib.fix(macro)
    macro=macro.loc[:,macro_cols]
    da=da.join(macro.set_index('timestamp'),on='timestamp')
    da[da==np.inf]=np.NaN
    if 'index' in da.columns:
        da.drop(['index'],inplace=True,axis=1)
    # Give tax-purpose properties a very low sample weights
    sample_weights=bad_weights(df_train,y_train,price_sq)
    train=da[:num_train].drop(['timestamp','id'],axis=1)
    test=da[num_train:].drop(['timestamp','id'],axis=1)
    # identify the binary features for excluding them from scaling
    bin_inds=[]
    for c in train.columns:
        if train.loc[:,c].unique().shape[0]==2 and train.loc[:,c].unique().sum()==1:
            bin_inds.append(train.columns.get_loc(c))
    return train,test,y_train,da[num_train:].id,bin_inds,sample_weights


def norm(train,test,feature_names):
    all_data=np.vstack((train,test))
    original=all_data.copy()
    if len(bin_inds)>0:
        bin_data=original[:,bin_inds].astype(int)
        bin_data[np.isnan(bin_data)]=0
        all_data=np.delete(all_data,bin_inds,axis=1)
        feature_names=[feature_names[i] for i in range(len(feature_names)) if i not in bin_inds]
    skip_inds=['xgb_score','xgb_score_log','log_xgb_score','log_xgb_score_log','rf_score','rf_score_log']
    skip_inds=[feature_names.index(ind) for ind in skip_inds if ind in feature_names]

    # Simple mean imputer, with standard scaler we can make all
    # NaN value zero, essentially cancel out its effect.
    imputer=Imputer(strategy='mean',copy=True,axis=0)
    all_data=imputer.fit_transform(all_data)
    # Remove all the feature values have SD greater than 4, we will skip
    # for the price feature as they are critical for our performance
    STD_LIMIT=4
    to_remove=[]
    for ci in range(all_data.shape[1]):
        if ci in skip_inds:
            continue
        dc=all_data[:,ci].copy()
        mean=np.mean(all_data[:,ci])
        std=np.std(all_data[:,ci])
        if std==0:
            to_remove.append(ci)
        else:
            all_data[(dc-mean)/float(std)>STD_LIMIT,ci]=mean
            all_data[(dc-mean)/float(std)<-STD_LIMIT,ci]=mean
    # Remove empty feature
    all_data=np.delete(all_data,to_remove,axis=1)

    train=all_data[0:train.shape[0],:]
    test=all_data[train.shape[0]:,:]

    scaler=StandardScaler()
    scaler.fit(np.vstack((train,test)))
    train=scaler.transform(train)
    test=scaler.transform(test)

    if len(bin_inds)>0:
        all_data=np.vstack((train,test))
        all_data=np.hstack((all_data,bin_data))
        train=all_data[0:train.shape[0],:]
        test=all_data[train.shape[0]:,:]

    # I don't want to do instance normalization as it will change
    # the logic of these noisy features for no reason.  However, as we can't use
    # BatchNormalization, DNN model converge faster and work better
    # with normalized data. I did the outlier value removal, so this should has
    # little impact on the real performance?
    norm=Normalizer(norm='l2')
    norm.fit(np.vstack((train,test)))
    train=norm.transform(train)
    test=norm.transform(test)
    return train,test
def nn(train,test,y_train,bin_inds,sample_weights,feature_names):
    train,test=norm(train,test,feature_names)
    if LOG:
        y_train=np.log1p(y_train)
    graph_in=Input(shape=(train.shape[1],),name='feature_data')
    
    # I used to have an embedding for product types !
    # product_type_in=Input(shape=(1,),name='product_type')

    model=Sequential()
    # Data augumentation using GassianDropout, basicly simulate the
    # random effect in this dataset :->
    out=GaussianDropout(0.1)(graph_in)
    out=Dense(2048)(out)
    out=Activation('relu')(out)
    if NORM:
        # It's a pitty that I cannot use batchnorm, which make things harder to learn
        out=BatchNormalization()(out)
    out=Dropout(0.3)(out) 
    out=Dense(1024)(out)
    if NORM:
        out=BatchNormalization()(out)
    out=Activation('relu')(out)
    out=Dropout(0.3)(out)
    out=Dense(512)(out)
    if NORM:
        out=BatchNormalization()(out)
    out=Activation('relu')(out)
    out=Dropout(0.3)(out)
    out=Dense(1)(out)
    graph=Model(inputs=[graph_in],outputs=out)
    print(graph.summary())
    model.add(graph)
    # Decaying learning rate
    lrate=LearningRateScheduler(step_decay)
    # Use clipnorm to prevent gradient explosion
    optimizer=SGD(lr=0.0, momentum=0.5,nesterov=True,clipnorm=100)
    model.compile(loss = 'mse', optimizer = optimizer)
    # You must set the shuffle to False! as the instances are very dependent!
    # -------------------------------------------
    # Note: I used epochs=40 for the real script
    # ------------------------------------------
    model.fit({'feature_data':train},y_train,batch_size=16,verbose=1,epochs=1,callbacks=[lrate],shuffle=False,sample_weight=sample_weights)
    if LOG:
        predicted=np.expm1(model.predict({'feature_data':test})[:,0])
    else:
        predicted=model.predict({'feature_data':test})[:,0]

    return predicted



# In[ ]:



if __name__=='__main__':
    train,test,y_train,id_test,bin_inds,sample_weights=prepare_data()
    predicted = nn(train.values,test.values,y_train.values,bin_inds,sample_weights,train.columns.tolist())
    output = pd.DataFrame({'id': id_test, 'price_doc': predicted})
    output.to_csv('dnn.csv', index=False)

