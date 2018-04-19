
# coding: utf-8

# This is the stacking model using NN. This gives the public LB score at 0.30409, and private LB score 0.31063. The score will be ranked at 4 on public LB, and 5 on private LB (generalized well?) This and the other kernels I posted are the best part of my model, the other parts are not very informative.
# 
# The core idea of this script is to replace linear combination and eliminate magic numbers. From the script, it contains no magic number anymore, however, the actual base model still got magic number. If we remove all the magic numbers from the base model, an earlier version of this script got LB score 0.3065. With this version, I think the LB score would be 0.305. This is not tested on public LB, but this is the relative improvement for the magic number version (from 0.305 to 0.304).
# 
# This script is far from optimal, for example fixing the investment=null in test data would bring some improvements and so on. If BoxCox is used, the performance can be substantially improved. If you combine it with your own model, top 3 
#  on both private/public board would definitely be possible.
# 
# You can read the method nn() and prepare_data() for more details.
# 
# I cannot run this script in kernel as it needs the base model's output which can be generated using 
# 
# 1.  https://www.kaggle.com/schoolpal/nn-model-lb-0-306-to-0-308
# 2. https://www.kaggle.com/schoolpal/lgbm-lb-0-3093-0-3094
# 3. https://www.kaggle.com/schoolpal/modifications-to-reynaldo-s-script
# 
# with 5 fold non-shuffle training.
# 

# In[ ]:


import os,sys
from keras.layers.advanced_activations import *
from keras.callbacks import LearningRateScheduler
import pickle
from keras.layers.merge import *
from keras.layers.noise import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,Imputer,RobustScaler
from keras.optimizers import SGD,RMSprop,Adam,Adadelta
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Input, Embedding,Flatten,Lambda
from keras.layers.normalization import BatchNormalization
import pandas as pd
import numpy as np
np.random.seed(1)
import pdb
import keras
NORM=False
LOG=True

def get_excluded():
    # Taken from wti200's kernel
    excluded={
        "young_male", "school_education_centers_top_20_raion", "0_17_female", "railroad_1line", "7_14_female", "0_17_all", "children_school","ecology", "16_29_male", "mosque_count_3000", "female_f", "church_count_1000", "railroad_terminal_raion","mosque_count_5000", "big_road1_1line", "mosque_count_1000", "7_14_male", "0_6_female", "oil_chemistry_raion","young_all", "0_17_male", "ID_bus_terminal", "university_top_20_raion", "mosque_count_500","ID_big_road1","ID_railroad_terminal", "ID_railroad_station_walk", "ID_big_road2", "ID_metro", "ID_railroad_station_avto","0_13_all", "mosque_count_2000", "work_male", "16_29_all", "young_female", "work_female", "0_13_female","ekder_female", "7_14_all", "big_church_count_500","leisure_count_500", "cafe_sum_1500_max_price_avg", "leisure_count_2000","office_count_500", "male_f", "nuclear_reactor_raion", "0_6_male", "church_count_500", "build_count_before_1920","thermal_power_plant_raion", "cafe_count_2000_na_price", "cafe_count_500_price_high","market_count_2000", "museum_visitis_per_100_cap", "trc_count_500", "market_count_1000", "work_all", "additional_education_raion","build_count_slag", "leisure_count_1000", "0_13_male", "office_raion","raion_build_count_with_builddate_info", "market_count_3000", "ekder_all", "trc_count_1000", "build_count_1946-1970","office_count_1500", "cafe_count_1500_na_price", "big_church_count_5000", "big_church_count_1000", "build_count_foam","church_count_1500", "church_count_3000", "leisure_count_1500","16_29_female", "build_count_after_1995", "cafe_avg_price_1500", "office_sqm_1000", "cafe_avg_price_5000", "cafe_avg_price_2000","big_church_count_1500", "full_all", "cafe_sum_5000_min_price_avg","office_sqm_2000", "church_count_5000","0_6_all", "detention_facility_raion", "cafe_avg_price_3000""young_male", "school_education_centers_top_20_raion", "0_17_female", "railroad_1line", "7_14_female", "0_17_all", "children_school","ecology", "16_29_male", "mosque_count_3000", "female_f", "church_count_1000", "railroad_terminal_raion","mosque_count_5000", "big_road1_1line", "mosque_count_1000", "7_14_male", "0_6_female", "oil_chemistry_raion","young_all", "0_17_male", "ID_bus_terminal", "university_top_20_raion", "mosque_count_500","ID_big_road1","ID_railroad_terminal", "ID_railroad_station_walk", "ID_big_road2", "ID_metro", "ID_railroad_station_avto","0_13_all", "mosque_count_2000", "work_male", "16_29_all", "young_female", "work_female", "0_13_female","ekder_female", "7_14_all", "big_church_count_500","leisure_count_500", "cafe_sum_1500_max_price_avg", "leisure_count_2000","office_count_500", "male_f", "nuclear_reactor_raion", "0_6_male", "church_count_500", "build_count_before_1920","thermal_power_plant_raion", "cafe_count_2000_na_price", "cafe_count_500_price_high","market_count_2000", "museum_visitis_per_100_cap", "trc_count_500", "market_count_1000", "work_all", "additional_education_raion","build_count_slag", "leisure_count_1000", "0_13_male", "office_raion","raion_build_count_with_builddate_info", "market_count_3000", "ekder_all", "trc_count_1000", "build_count_1946-1970","office_count_1500", "cafe_count_1500_na_price", "big_church_count_5000", "big_church_count_1000", "build_count_foam","church_count_1500", "church_count_3000", "leisure_count_1500","16_29_female", "build_count_after_1995", "cafe_avg_price_1500", "office_sqm_1000", "cafe_avg_price_5000", "cafe_avg_price_2000","big_church_count_1500", "full_all", "cafe_sum_5000_min_price_avg","office_sqm_2000", "church_count_5000","0_6_all", "detention_facility_raion", "cafe_avg_price_3000"
    }
    return excluded

def step_decay(epoch):
    lr=0.01
    start=5
    step=5
    if epoch<start:
        return lr
    else:
        lr=lr/np.power(2.0,(1+(epoch-start)/step))
        return lr



    return weights
# Time based sample weights, this improves the LB score only
# a little bit(0.0002)
def time_weights(train,prices,price_sq):
    weights=np.ones(len(price_sq))
    weights[(train.timestamp.dt.year==2011)]*=0.5
    weights[(train.timestamp.dt.year==2012)]*=0.8
    weights[(train.timestamp.dt.year==2013)]*=1.4
    weights[(train.timestamp.dt.year==2014)]*=0.9
    weights[(train.timestamp.dt.year==2015)]*=2
    return weights


def prepare_data():
    excluded=get_excluded()
    df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
    df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
    #These scores are generated using the following kernel with 5 fold non-shuffle traning:
    # https://www.kaggle.com/schoolpal/nn-model-lb-0-306-to-0-308
    # https://www.kaggle.com/schoolpal/lgbm-lb-0-3093-0-3094
    # https://www.kaggle.com/schoolpal/modifications-to-reynaldo-s-script
    
    (xgb_train,xgb_test)=pickle.load(open('xgb_predicted.pkl'))
    (xgb_train_log,xgb_test_log)=pickle.load(open('xgb_predicted_log.pkl'))
    (lgb_train,lgb_test)=pickle.load(open('lgb_predicted.pkl'))
    (nn_train,nn_test)=pickle.load(open('nn_predicted_log.pkl'))
    df_train['xgb_score']=xgb_train
    df_train['xgb_score_log']=xgb_train_log
    df_train['log_xgb_score']=np.log(xgb_train)
    df_train['log_xgb_score_log']=np.log(xgb_train_log)
    df_train['nn_score']=nn_train
    df_train['nn_score_log']=np.log(nn_train)
    df_train['lgb_score']=lgb_train
    df_train['lgb_score_log']=np.log(lgb_train)
    df_test['xgb_score']=xgb_test
    df_test['xgb_score_log']=xgb_test_log
    df_test['log_xgb_score']=np.log(xgb_test)
    df_test['log_xgb_score_log']=np.log(xgb_test_log)
    df_test['nn_score']=nn_test
    df_test['nn_score_log']=np.log(nn_test)
    df_test['lgb_score']=lgb_test
    df_test['lgb_score_log']=np.log(lgb_test)
 
    full_sq=df_train.full_sq.copy()
    full_sq[full_sq<5]=np.NaN
    
    price_sq=df_train.price_doc/full_sq
    df_train=df_train[(price_sq<600000) & (price_sq>10000)]
    price_sq=price_sq[(price_sq<600000) & (price_sq>10000)]


    y_train=df_train.price_doc
    df_train.drop(['price_doc'],inplace=True,axis=1)
    num_train=df_train.shape[0]
    
    da=pd.concat([df_train,df_test])
    da=da.reset_index(drop=True)


    da['build_year_0']=(da.build_year==0).astype(int)
    da['build_year_1']=(da.build_year==1).astype(int)
    da['build_year_null']=(da.build_year.isnull()).astype(int)
    da['build_year_not_1']=(da.build_year!=1).astype(int)
    da['olds']=da.timestamp.dt.year-da.build_year

    da['investment_very_old']=((da.product_type=='Investment') & (da.olds>54)).astype(int)
    da['not_investment_very_old']=((da.product_type!='Investment') | (da.olds<=54)).astype(int)
    da['investment_old']=((da.product_type=='Investment') & (da.olds>10)).astype(int)
    da['not_investment_old']=((da.product_type!='Investment') | (da.olds<=10)).astype(int)
    da['investment_new']=((da.product_type=='Investment') & (da.olds<=10)).astype(int)
    da['not_investment_new']=((da.product_type!='Investment') | (da.olds>10)).astype(int)

    da['own_new']=((da.product_type=='OwnerOccupier') & (da.olds==0)).astype(int)
    da['not_own_new']=((da.product_type!='OwnerOccupier') | (da.olds==0)).astype(int)
    da['year_month']=da.timestamp.dt.year
    da['year_month']=(da['year_month']*100+da.timestamp.dt.month)
    da['year_month']=da['year_month'].astype(str)
    da['month']=da.timestamp.dt.month.astype(str)
    da['close_to_green']=(da.green_zone_km<0.05).astype(int)
    da['close_to_rail']=(da.railroad_km<0.5).astype(int)
    da['close_to_school']=(da.school_km<0.1).astype(int)

    cols=[]
    # The actual feature used, only use the feature related to the building of the property but not the property itself (e.g. full_sq,life_sq)
    
    cols1=['timestamp','id','product_type','olds','max_floor','year_month','close_to_green','close_to_rail','close_to_school','build_year_1','build_year_not_1','investment_olds6','not_investment_very_old','investment_old','not_investment_old','month']
    
    price_cols=['log_xgb_score','lgb_score_log','nn_score_log']
    cols=set(cols1)
    cols.update(price_cols)
    da=da.loc[:,cols]
    df_cat=None
    to_remove=[]
    for c in da.columns:
        if da[c].dtype=='object':
            oh=pd.get_dummies(da[c],prefix=c)
            
            if df_cat is None:
                df_cat=oh
            else:
                df_cat=pd.concat([df_cat,oh],axis=1)
            to_remove.append(c)
    da.drop(to_remove,inplace=True,axis=1)
    if df_cat is not None:
        sums=df_cat.sum(axis=0)
        to_remove=sums[sums<200].index.values
        df_cat=df_cat.loc[:,df_cat.columns.difference(to_remove)]
        da = pd.concat([da, df_cat], axis=1)
    # The macro features
    macro_cols=['timestamp','eurrub','unemployment','brent','emigrant','mortgage_growth']
    macro=pd.read_csv('input/macro.csv',parse_dates=['timestamp'])
    # Add the unemployment data from OCED and emigrant data from Fedora state static website
    # macro=macro_lib.fix(macro)
    macro=macro.loc[:,macro_cols]
    da=da.join(macro.set_index('timestamp'),on='timestamp')
    da[da==np.inf]=np.NaN
    if 'index' in da.columns:
        da.drop(['index'],inplace=True,axis=1)
    sample_weights=time_weights(df_train,y_train,price_sq)
    train=da[:num_train].drop(['timestamp','id'],axis=1)
    test=da[num_train:].drop(['timestamp','id'],axis=1)
    aux_cols=[c for c in train.columns if c not in price_cols and (c in cols1 or c not in cols)]

    train_prices=train.loc[:,price_cols]
    train_aux1=train.loc[:,aux_cols]
    test_prices=test.loc[:,price_cols]
    test_aux1=test.loc[:,aux_cols]
    bin_inds=[]
    for c in train.columns:
        if train.loc[:,c].unique().shape[0]==2 and train.loc[:,c].unique().sum()==1:
            bin_inds.append(train.columns.get_loc(c))
    return train_prices,test_prices,train_aux1,test_aux1,y_train,da[num_train:].id,sample_weights,bin_inds

def norm(train,test):
    all_data=np.vstack((train,test))
    original=all_data.copy()
    bin_inds=[]
    for ci in range(all_data.shape[1]):
        if np.unique(all_data[:,ci]).shape[0]==2 and all_data[:,ci].max()==1:
            bin_inds.append(ci)
    if len(bin_inds)>0:
        bin_data=original[:,bin_inds].astype(int)
        bin_data[np.isnan(bin_data)]=0
        all_data=np.delete(all_data,bin_inds,axis=1)

    
    imputer=Imputer(strategy='mean',copy=True,axis=0)
    all_data=imputer.fit_transform(all_data)
    STD_LIMIT=4
    to_remove=[]
    for ci in range(all_data.shape[1]):
        dc=all_data[:,ci].copy()
        mean=np.mean(all_data[:,ci])
        std=np.std(all_data[:,ci])
        if std==0:
            to_remove.append(ci)
        else:
            all_data[(dc-mean)/float(std)>STD_LIMIT,ci]=mean
            all_data[(dc-mean)/float(std)<-STD_LIMIT,ci]=mean
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

    norm=Normalizer(norm='l2')
    norm.fit(np.vstack((train,test)))
    train=norm.transform(train)
    test=norm.transform(test)
    return train,test
def nn(train_price,test_price,train_aux1,test_aux1,y_train,sample_weights,bin_inds):
    ymin=None
    train_aux1,test_aux1=norm(train_aux1,test_aux1)

    if LOG:
        y_train=np.log1p(y_train)
    price_in=Input(shape=(train_price.shape[1],),name='price_data')
    aux_in1=Input(shape=(train_aux1.shape[1],),name='aux_data1')
    model=Sequential()
    # The following part learn the linear combination weights
    aux_out=Dense(256,activation='relu')(aux_in1)
    aux_out=Dropout(0.3)(aux_out)
    aux_out=Dense(128,activation='relu')(aux_out)
    aux_out=Dropout(0.3)(aux_out)
    aux_out=Dense(64,activation='relu')(aux_out)
    aux_out=Dense(3,activation='softmax')(aux_out)
    # Do the linear combination
    out=dot([price_in,aux_out],-1)
    
    scale_out=Dense(1,activation='hard_sigmoid',kernel_regularizer=regularizers.l1())(aux_in1)
    # Scale the instace in the range 0.85-1.02, the weights are learned.
    # This corrects some errors in the base model
    scale_out=Lambda(lambda x: (0.17*x+0.85) )(scale_out)
    out=multiply([scale_out,out])
    graph=Model(inputs=[price_in,aux_in1],outputs=out)
    print(graph.summary())
    model.add(graph)
    lrate=LearningRateScheduler(step_decay)
    optimizer=SGD(lr=0.01, momentum=0.5,nesterov=True)
    model.compile(loss = 'mse', optimizer = optimizer)
    model.fit({'price_data':train_price,'aux_data1':train_aux1},y_train,batch_size=64,verbose=1,epochs=1,callbacks=[lrate],shuffle=False,sample_weight=sample_weights)#,validation_split=0.1)
    predicted=np.expm1(model.predict({'price_data':test_price,'aux_data1':test_aux1})[:,0])

    return predicted
if __name__=='__main__':
    pass
#    train_prices,test_prices,train_aux1,test_aux1,y_train,id_test,sample_weights,bin_inds=prepare_data()
#    predicted = nn(train_prices.values,test_prices.values,train_aux1.values,test_aux1.values,y_train.values,sample_weights,bin_inds)
#    output = pd.DataFrame({'id': id_test, 'price_doc': predicted})
#    output=output.reset_index(drop=True)
#    test=pd.read_csv('input/test.csv',parse_dates=['timestamp'])
#    output.to_csv('dnn2.csv', index=False)

