
# coding: utf-8

# Motivation: auc/gini with keras

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


print('Reading files')
train  =pd.read_csv("../input/train.csv")
test   =pd.read_csv("../input/test.csv")
col_x= train.columns.drop(['target'])
col  = train.columns.drop(['id','target'])
print('OK')


# Blablah keras toys you want to use

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import keras.models 


# Kaggle discussion/kernels metrics

# In[ ]:


import tensorflow as tf
import keras.backend as K

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41108
def jacek_auc(y_true, y_pred):
   score, up_opt = tf.metrics.auc(y_true, y_pred)
   #score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)    
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score

# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41015
# AUC for a binary classifier
def discussion41015_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

#---------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

#----------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P


# Any model, just an example:

# In[ ]:


def model_relu1():
    model = Sequential()
    model.add(Dense(1024, input_dim=57, activation='relu', name='in'))
    model.add(Dense(   1, activation='sigmoid', name='out'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[jacek_auc,discussion41015_auc])
    return model


# # Option 1 - magic , create an callback and handle everything

# In[ ]:


#go here, it's easier to understand callbacks reading keras source code:
#   https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L838
#   https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L1040

from sklearn.metrics import roc_auc_score
class GiniWithEarlyStopping(keras.callbacks.Callback):
    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size=1024):
        #print("self vars: ",vars(self))  #uncomment and discover some things =)
        
        # FROM EARLY STOP
        super(GiniWithEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size=predict_batch_size
    
    def on_batch_begin(self, batch, logs={}):
        if(self.verbose > 1):
            if(batch!=0):
                print("")
            print("Hi! on_batch_begin() , batch=",batch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)
    
    def on_batch_end(self, batch, logs={}):
        if(self.verbose > 1):
            print("Hi! on_batch_end() , batch=",batch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)
    
    def on_train_begin(self, logs={}):
        if(self.verbose > 1):
            print("Hi! on_train_begin() ,logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
    
    def on_train_end(self, logs={}):
        if(self.verbose > 1):
            print("Hi! on_train_end() ,logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch ',self.stopped_epoch,': GiniEarlyStopping')
    
    def on_epoch_begin(self, epoch, logs={}):
        if(self.verbose > 1):
            print("Hi! on_epoch_begin() , epoch=",epoch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

    def on_epoch_end(self, epoch, logs={}):
        if(self.validation_data):
            y_hat_val=self.model.predict(self.validation_data[0],batch_size=self.predict_batch_size)
            
        if(self.verbose > 1):
            print("Hi! on_epoch_end() , epoch=",epoch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)
        
        #i didn't found train data to check gini on train set (@TODO HERE)
        # from source code of Keras: https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L1127
        # for cbk in callbacks:
        #     cbk.validation_data = val_ins
        # Probably we will need to change keras... 
        # 
        
            print("    GINI Callback:")
            if(self.validation_data):
                print('        validation_data.inputs       : ',np.shape(self.validation_data[0]))
                print('        validation_data.targets      : ',np.shape(self.validation_data[1]))
                print("        roc_auc_score(y_real,y_hat)  : ",roc_auc_score(self.validation_data[1], y_hat_val ))
                print("        gini_normalized(y_real,y_hat): ",gini_normalized(self.validation_data[1], y_hat_val))
                print("        roc_auc_scores*2-1           : ",roc_auc_score(self.validation_data[1], y_hat_val)*2-1)
        
            print('    Logs (others metrics):',logs)
        # FROM EARLY STOP
        if(self.validation_data):
            if (self.verbose == 1):
                print("\n GINI Callback:",gini_normalized(self.validation_data[1], y_hat_val))
            current = gini_normalized(self.validation_data[1], y_hat_val)
            
            # we can include an "gambiarra" (very usefull brazilian portuguese word)
            # to logs (scores) and use others callbacks too....
            # logs['gini_val']=current
            
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True


# In[ ]:


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# reduce train size, just to this kernel example
t=train[0:1000]
# batch_size=500 ~= 2 batchs
estimator = KerasClassifier(build_fn=model_relu1, nb_epoch=3, batch_size=500, verbose=1)



cb = [
    # verbose =2 make many prints (nice to learn keras callback)
    GiniWithEarlyStopping(patience=1, verbose=2) 
]

estimator.fit(t[col].values,t['target'],epochs=100,validation_split=.2,callbacks=cb)




# I don't know why the last line " < keras.callbacks.History at 0x..... > " anyone please check it and comment to fix

# In[ ]:


cb = [
    # verbose =1 print gini per epoch
    GiniWithEarlyStopping(patience=1, verbose=1) 
]

estimator.fit(t[col].values,t['target'],epochs=100,validation_split=.2,callbacks=cb)



# I don't know why the last line " < keras.callbacks.History at 0x..... > " anyone please check it and comment to fix

# In[ ]:


cb = [
    # verbose =0 don't print
    GiniWithEarlyStopping(patience=1, verbose=0) 
]

estimator.fit(t[col].values,t['target'],epochs=100,validation_split=.2,callbacks=cb)



# # Option 2 - magic, Include metric in logs dictionary
# 
# example with Roc

# In[ ]:


from sklearn.metrics import roc_auc_score
class RocAucMetricCallback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size=predict_batch_size
        self.include_on_batch=include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if(self.include_on_batch):
            logs['roc_auc_val']=float('-inf')
            if(self.validation_data):
                logs['roc_auc_val']=roc_auc_score(self.validation_data[1], 
                                                  self.model.predict(self.validation_data[0],
                                                                     batch_size=self.predict_batch_size))

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc_val']=float('-inf')
        if(self.validation_data):
            logs['roc_auc_val']=roc_auc_score(self.validation_data[1], 
                                              self.model.predict(self.validation_data[0],
                                                                 batch_size=self.predict_batch_size))



# In[ ]:


from keras.callbacks import EarlyStopping
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# reduce train size, just to this kernel example
t=train[0:1000]
# batch_size=500 ~= 2 batchs
estimator = KerasClassifier(build_fn=model_relu1, nb_epoch=3, batch_size=500, verbose=1)

cb = [
    RocAucMetricCallback(), # include it before EarlyStopping!
    EarlyStopping(monitor='roc_auc_val',patience=1, verbose=2) 
]

estimator.fit(t[col].values,t['target'],epochs=100,validation_split=.2,callbacks=cb)




# "Epoch 00002: early stopping" - Nice =D
# 

# In[ ]:


cb = [
    EarlyStopping(monitor='roc_auc_val',patience=1, verbose=2), 
    RocAucMetricCallback(), # include it before EarlyStopping! i told you...
]
estimator.fit(t[col].values,t['target'],epochs=100,validation_split=.2,callbacks=cb)

