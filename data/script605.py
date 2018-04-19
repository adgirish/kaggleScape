
# coding: utf-8

# This notebook demonstrates a custom loss function for neural nets, that provides a differentiable approximation to AUC. AUC, in turn, has a linear relationship with Gini, hence this is very useful when we want to train a network to maximise AUC.
# 
# We set up 2 identical NNs and run them for a few epochs, to show how this approach improves convergence on AUC compared to binary crossentropy.
# 
# I've used this to get a network that has a local CV AUC around 0.642, which corresponds to Gini of 0.284. The performance on the LB test set is considerably worse (around 0.276)
# 
# This is hacked together from various bits of my local code, and hasn't been thoroughly tested, so let please me know of any bugs etc.
# 
# I would have coded as a script, but I need to use the Theano backend as the AUC function uses Theano specific code. If anyone knows how to make Kaggle Kernels use the Theano backend for script, let me know.
# 
# First of all, imports and constants

# In[1]:


import numpy as np
import pandas as pd

get_ipython().run_line_magic('env', 'KERAS_BACKEND=theano')

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import custom_object_scope
from keras import callbacks

from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

import theano

# train and test data path
#DATA_TRAIN_PATH = '../input/train.csv'
#DATA_TEST_PATH = '../input/test.csv'

DATA_TRAIN_PATH = 'c:\\projects\\psdriver\\data\\train.csv'
DATA_TEST_PATH = 'c:\\projects\\psdriver\\data\\test.csv'


featuresToDrop = [
    'ps_calc_10',
    'ps_calc_01',
    'ps_calc_02',
    'ps_calc_03',
    'ps_calc_13',
    'ps_calc_08',
    'ps_calc_07',
    'ps_calc_12',
    'ps_calc_04',
    'ps_calc_17_bin',
    'ps_car_10_cat',
    'ps_car_11_cat',
    'ps_calc_14',
    'ps_calc_11',
    'ps_calc_06',
    'ps_calc_16_bin',
    'ps_calc_19_bin',
    'ps_calc_20_bin',
    'ps_calc_15_bin',
    'ps_ind_11_bin',
    'ps_ind_10_bin'
]



# Now, the secret sauce

# In[ ]:



# An analogue to AUC which takes the differences between each pair of true/false predictions
# and takes the average sigmoid of the differences to get a differentiable loss function.
# Based on code and ideas from https://github.com/Lasagne/Lasagne/issues/767
def soft_AUC_theano(y_true, y_pred):
    # Extract 1s
    pos_pred_vr = y_pred[y_true.nonzero()]
    # Extract zeroes
    neg_pred_vr = y_pred[theano.tensor.eq(y_true, 0).nonzero()]
    # Broadcast the subtraction to give a matrix of differences  between pairs of observations.
    pred_diffs_vr = pos_pred_vr.dimshuffle(0, 'x') - neg_pred_vr.dimshuffle('x', 0)
    # Get signmoid of each pair.
    stats = theano.tensor.nnet.sigmoid(pred_diffs_vr * 2)
    # Take average and reverse sign
    return 1-theano.tensor.mean(stats) # as we want to minimise, and get this to zero


# This callback records the SKLearn calculated AUC each round, for use by early stopping
# It also has slots where you can save down metadata or the model at useful points -
# for Kaggle kernel purposes I've commented these out
class AUC_SKlearn_callback(callbacks.Callback):
    def __init__(self, X_train, y_train, useCv = True):
        super(AUC_SKlearn_callback, self).__init__()
        self.bestAucCv = 0
        self.bestAucTrain = 0
        self.cvLosses = []
        self.bestCvLoss = 1,
        self.X_train = X_train
        self.y_train = y_train
        self.useCv = useCv

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_pred = self.model.predict(np.array(self.X_train))
        aucTrain = roc_auc_score(self.y_train, train_pred)
        print("SKLearn Train AUC score: " + str(aucTrain))

        if (self.bestAucTrain < aucTrain):
            self.bestAucTrain = aucTrain
            print ("Best SKlearn AUC training score so far")
            #**TODO: Add your own logging/saving/record keeping code here

        if (self.useCv) :
            cv_pred = self.model.predict(self.validation_data[0])
            aucCv = roc_auc_score(self.validation_data[1], cv_pred)
            print ("SKLearn CV AUC score: " +  str(aucCv))

            if (self.bestAucCv < aucCv) :
                # Great! New best *actual* CV AUC found (as opposed to the proxy AUC surface we are descending)
                print("Best SKLearn genuine AUC so far so saving model")
                self.bestAucCv = aucCv

                # **TODO: Add your own logging/model saving/record keeping code here.
                self.model.save("best_auc_model.h5", overwrite=True)

            vl = logs.get('val_loss')
            if (self.bestCvLoss < vl) :
                print("Best val loss on SoftAUC so far")
                #**TODO -  Add your own logging/saving/record keeping code here.
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        # logs include loss, and optionally acc( if accuracy monitoring is enabled).
        return


# Create the model.
def create_model_AUC(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout):
    return create_model(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout, "AUC")

def create_model_bce(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout):
    return create_model(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout, "crossentropy")


def create_model(input_dim, first_layer_size, second_layer_size, third_layer_size, lr, l2reg, dropout, mode="AUC") :
    print("Creating model with input dim ", input_dim)
    # likely to need tuning!
    reg = regularizers.l2(l2reg)

    model = Sequential()

    model.add(Dense(units=first_layer_size, kernel_initializer='lecun_normal', kernel_regularizer=reg, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(units=second_layer_size, kernel_initializer='lecun_normal', activation='relu', kernel_regularizer=reg))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(dropout))

    model.add(Dense(units=third_layer_size, kernel_initializer='lecun_normal', activation='relu', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(Dense(1, kernel_initializer='lecun_normal', activation='sigmoid'))

    # classifier.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mae', 'accuracy'])
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if (mode == "AUC"):
        model.compile(loss=soft_AUC_theano, metrics=[soft_AUC_theano], optimizer=opt)  # not sure whether to use metrics here?
    else:
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)  # not sure whether to use metrics here?
    return model


def train_model( X_train, y_train, model, valSplit=0.15, epochs = 5, batch_size = 4096):

    callbacksList = [AUC_SKlearn_callback(X_train, y_train, useCv = (valSplit > 0))]
    if (valSplit > 0) :
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5,
                                                       verbose=0, mode='min')
        callbacksList.append( early_stopping )
    return model.fit(x=np.array(X_train), y=np.array(y_train),
                        callbacks=callbacksList, validation_split=valSplit,
                        verbose=2, batch_size=batch_size, epochs=epochs)



def scale_features(df_for_range, df_to_scale, columnsToScale) :
    # Scale columnsToScale in df_to_scale
    columnsOut = list(map( (lambda x: x + "_scaled"), columnsToScale))
    for c, co in zip(columnsToScale, columnsOut) :
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        print("scaling ", c ," to ",co)
        vals = df_for_range[c].values.reshape(-1, 1)
        scaler.fit(vals )
        df_to_scale[co]=scaler.transform(df_to_scale[c].values.reshape(-1,1))

    df_to_scale.drop (columnsToScale, axis=1, inplace = True)

    return df_to_scale


def one_hot (df, cols):
    # One hot cols requested, drop original cols, return df
    df = pd.concat([df, pd.get_dummies(df[cols], columns=cols)], axis=1)
    df.drop(cols, axis=1, inplace = True)
    return df

def get_data() :
    X_train = pd.read_csv(DATA_TRAIN_PATH, index_col = "id")
    X_test = pd.read_csv(DATA_TEST_PATH, index_col = "id")

    y_train = pd.DataFrame(index = X_train.index)
    y_train['target'] = X_train.loc[:,'target']
    X_train.drop ('target', axis=1, inplace = True)
    X_train.drop (featuresToDrop, axis=1, inplace = True)
    X_test.drop (featuresToDrop,axis=1, inplace = True)

    # car_11 is really a cat col
    X_train.rename(columns={'ps_car_11': 'ps_car_11a_cat'}, inplace=True)
    X_test.rename(columns={'ps_car_11': 'ps_car_11a_cat'}, inplace=True)

    cat_cols = [elem for elem in list(X_train.columns) if "cat" in elem]
    bin_cols = [elem for elem in list(X_train.columns) if "bin" in elem]
    other_cols = [elem for elem in list(X_train.columns) if elem not in bin_cols and elem not in cat_cols]

    # Scale numeric features in region of -1,1 using training set as the scaling range
    X_test = scale_features(X_train, X_test, columnsToScale=other_cols)
    X_train = scale_features(X_train, X_train, columnsToScale=other_cols)

    X_train = one_hot(X_train, cat_cols)
    X_test = one_hot(X_test, cat_cols)


    return X_train, X_test, y_train


def makeOutputFile(pred_fun, test, subsFile) :
    df_out = pd.DataFrame(index=test.index)
    y_pred = pred_fun( test )
    df_out['target'] = y_pred
    df_out.to_csv(subsFile, index_label="id")

def main() :
    X_train, X_test, y_train = get_data()
    model = create_model( input_dim=X_train.shape[1],
                          first_layer_size=300,
                          second_layer_size=200,
                          third_layer_size=200,
                          lr=0.0001,
                          l2reg = 0.1,
                          dropout = 0.2,
                          mode="AUC")

    train_model(X_train, y_train, model)

    with custom_object_scope({'soft_AUC_theano': soft_AUC_theano}):
        pred_fun = lambda x: model.predict(np.array(x))
        makeOutputFile(pred_fun, X_test, "auc.csv")

    model = create_model_bce( input_dim=X_train.shape[1],
                          first_layer_size=300,
                          second_layer_size=200,
                          third_layer_size=200,
                          lr=0.0001,
                          l2reg = 0.1,
                          dropout = 0.2)

    train_model(X_train, y_train, model)

    pred_fun = lambda x: model.predict(np.array(x))
    makeOutputFile(pred_fun, X_test, "no_auc.csv")

main()

