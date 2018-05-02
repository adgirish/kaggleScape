
# coding: utf-8

# The purpose of this notebook is to check the effect of both scale_pos_weight and duplication on a LightGBM classifier.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import gc
from numba import jit
from sklearn.preprocessing import LabelEncoder
import time 
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import itertools
import math
np.set_printoptions(precision=3)


# In[ ]:


@jit  # for more info please visit https://numba.pydata.org/
def eval_gini(y_true, y_prob):
    """
    Original author CMPM 
    https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

# This is taken from sklearn examples and is available at
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### Get the data and reduce the features 
# Feature selection is done according to kernel 
# https://www.kaggle.com/ogrellier/noise-analysis-of-porto-seguro-s-features 
# 

# In[ ]:


trn_df = pd.read_csv("../input/train.csv", index_col=0)
target = trn_df["target"]
del trn_df["target"]

train_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
	"ps_reg_03",  #            : 1408.42 / shadow  511.15
	"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
	"ps_ind_03",  #            : 1219.47 / shadow  230.55
	"ps_ind_15",  #            :  922.18 / shadow  242.00
	"ps_reg_02",  #            :  920.65 / shadow  267.50
	"ps_car_14",  #            :  798.48 / shadow  549.58
	"ps_car_12",  #            :  731.93 / shadow  293.62
	"ps_car_01_cat",  #        :  698.07 / shadow  178.72
	"ps_car_07_cat",  #        :  694.53 / shadow   36.35
	"ps_ind_17_bin",  #        :  620.77 / shadow   23.15
	"ps_car_03_cat",  #        :  611.73 / shadow   50.67
	"ps_reg_01",  #            :  598.60 / shadow  178.57
	"ps_car_15",  #            :  593.35 / shadow  226.43
	"ps_ind_01",  #            :  547.32 / shadow  154.58
	"ps_ind_16_bin",  #        :  475.37 / shadow   34.17
	"ps_ind_07_bin",  #        :  435.28 / shadow   28.92
	"ps_car_06_cat",  #        :  398.02 / shadow  212.43
	"ps_car_04_cat",  #        :  376.87 / shadow   76.98
	"ps_ind_06_bin",  #        :  370.97 / shadow   36.13
	"ps_car_09_cat",  #        :  214.12 / shadow   81.38
	"ps_car_02_cat",  #        :  203.03 / shadow   26.67
	"ps_ind_02_cat",  #        :  189.47 / shadow   65.68
	"ps_car_11",  #            :  173.28 / shadow   76.45
	"ps_car_05_cat",  #        :  172.75 / shadow   62.92
	"ps_calc_09",  #           :  169.13 / shadow  129.72
	"ps_calc_05",  #           :  148.83 / shadow  120.68
	"ps_ind_08_bin",  #        :  140.73 / shadow   27.63
	"ps_car_08_cat",  #        :  120.87 / shadow   28.82
	"ps_ind_09_bin",  #        :  113.92 / shadow   27.05
	"ps_ind_04_cat",  #        :  107.27 / shadow   37.43
	"ps_ind_18_bin",  #        :   77.42 / shadow   25.97
	"ps_ind_12_bin",  #        :   39.67 / shadow   15.52
	"ps_ind_14",  #            :   37.37 / shadow   16.65
]
    
trn_df = trn_df[train_features]


# ### Let's have a look at the evolution of predictions with scale_pose_weight

# In[ ]:


scale_pos_weights = range(1, 20, 2)
oof_proba = np.empty((len(trn_df), len(scale_pos_weights)))
oof_label = np.empty((len(trn_df), len(scale_pos_weights)))
for i_w, scale_pos_weight in enumerate(scale_pos_weights):
    n_splits = 2 # That's enough to get a feel on what's going on
    n_estimators = 100
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=14) 

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
        trn_dat, trn_tgt = trn_df.iloc[trn_idx], target.iloc[trn_idx]
        val_dat, val_tgt = trn_df.iloc[val_idx], target.iloc[val_idx]

        clf = LGBMClassifier(n_estimators=n_estimators,
                             max_depth=-1,
                             num_leaves=25,
                             learning_rate=.1, 
                             subsample=.8, 
                             colsample_bytree=.8,
                             min_split_gain=1,
                             reg_alpha=0,
                             reg_lambda=0,
                             scale_pos_weight=scale_pos_weight, # <= We do not overweight positive samples
                             n_jobs=2)

        clf.fit(trn_dat, trn_tgt, 
                eval_set=[(trn_dat, trn_tgt), (val_dat, val_tgt)],
                eval_metric="auc",
                early_stopping_rounds=None,
                verbose=False)

        oof_proba[val_idx, i_w] = clf.predict_proba(val_dat)[:, 1]
        oof_label[val_idx, i_w] = clf.predict(val_dat)
        
    print("Full OOF score : %.6f for scale_pos_weight = %2d" 
          % (eval_gini(target, oof_proba[:, i_w]), scale_pos_weight))


# ### Now let's look at the evolution of f1_scores against scale_pos_weight
# We can see that probabilities are more spread into [0, 1] space when scale_pos_weight increase

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
plt.rc('legend', fontsize=18) 
plt.rc('axes', labelsize=18)
plt.rc('axes', titlesize=24)
for i_w, scale_pos_weight in enumerate(scale_pos_weights):
    # Get False positives, true positives and the list of thresholds used to compute them
    fpr, tpr, thresholds = roc_curve(target, oof_proba[:, i_w])
    # Compute recall, precision and f1_score
    recall = tpr
    precision = tpr / (fpr + tpr + 1e-5)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-5)
    # Finally plot the f1_scores against thresholds
    plt.plot(thresholds[-30000:], f1_scores[-30000:], 
             label="scale_pos_weight=%2d" % scale_pos_weight)
plt.title("F1 scores against threshold for different scale_pos_weight")
plt.ylabel("F1 score")
plt.xlabel("Probability thresholds")
plt.legend(loc="lower left")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))



# ### Confusion matrices
# To see the way probabilities spread more when scale_pos_weight increase let's display the evolution of the confusion matrices 

# In[ ]:


fig = plt.figure(figsize=(20, 15))
plt.rc('legend', fontsize=10) 
plt.rc('axes', labelsize=10)
plt.rc('axes', titlesize=10)
gs = gridspec.GridSpec(int(len(scale_pos_weights) / 2), 2)
for i_w, weight in enumerate(scale_pos_weights):
    ax = plt.subplot(gs[int(i_w / 2), i_w % 2])
    class_names = ["safe", "unsafe"]
    cnf_matrix = confusion_matrix(target, oof_label[:, i_w])
    plot_confusion_matrix(cnf_matrix, classes=class_names, 
                          normalize=True,
                          title='Matrix for scale_pos_weight = %2d, Gini %.6f' 
                          % (weight, eval_gini(target, oof_proba[:, i_w])))
plt.tight_layout()


# As you can see we have more and more true positives while false negative increase as well. 
# 

# ### What about duplication ?

# In[ ]:


dupes = np.arange(0.5, 3.1, .5)
oof_proba = np.empty((len(trn_df), len(dupes)))
oof_label = np.empty((len(trn_df), len(dupes)))
for i_w, dupe in enumerate(dupes):
    n_splits = 2 # That's enough to get a feel on what's going on
    n_estimators = 100
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=14) 

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
        trn_dat, trn_tgt = trn_df.iloc[trn_idx], target.iloc[trn_idx]
        val_dat, val_tgt = trn_df.iloc[val_idx], target.iloc[val_idx]

        clf = LGBMClassifier(n_estimators=n_estimators,
                             max_depth=-1,
                             num_leaves=25,
                             learning_rate=.1, 
                             subsample=.8, 
                             colsample_bytree=.8,
                             min_split_gain=1,
                             reg_alpha=0,
                             reg_lambda=0,
                             scale_pos_weight=1,
                             min_child_weight=1,
                             n_jobs=2)
        # Duplicate positives on the training part
        pos = pd.Series(trn_tgt == 1)
        pos_dat = trn_dat.loc[pos]
        pos_tgt = trn_tgt.loc[pos]
        pos_idx = np.arange(len(pos_tgt))
        if dupe <= 1.0: 
            # Add positive examples
            np.random.shuffle(pos_idx)
            trn_dat = pd.concat([trn_dat, pos_dat.iloc[pos_idx[:int(len(pos) * dupe)]]], axis=0)
            trn_tgt = pd.concat([trn_tgt, pos_tgt.iloc[pos_idx[:int(len(pos) * dupe)]]], axis=0)
        else:
            dupint = math.floor(dupe)
            remain = dupe - dupint
            for i in range(dupint):
                trn_dat = pd.concat([trn_dat, pos_dat], axis=0)
                trn_tgt = pd.concat([trn_tgt, pos_tgt], axis=0)
            np.random.shuffle(pos_idx)
            trn_dat = pd.concat([trn_dat, pos_dat.iloc[pos_idx[:int(len(pos) * dupe)]]], axis=0)
            trn_tgt = pd.concat([trn_tgt, pos_tgt.iloc[pos_idx[:int(len(pos) * dupe)]]], axis=0)
        print(len(trn_dat), len(pos_dat))
        # Shuffle data
        idx = np.arange(len(trn_dat))
        np.random.shuffle(idx)
        trn_dat = trn_dat.iloc[idx]
        trn_tgt = trn_tgt.iloc[idx]
        
        clf.fit(trn_dat, trn_tgt, 
                eval_set=[(trn_dat, trn_tgt), (val_dat, val_tgt)],
                eval_metric="auc",
                early_stopping_rounds=None,
                verbose=False)

        oof_proba[val_idx, i_w] = clf.predict_proba(val_dat)[:, 1]
        oof_label[val_idx, i_w] = clf.predict(val_dat)

    print("Full OOF score : %.6f for duplication %.1f" 
          % (eval_gini(target, oof_proba[:, i_w]), dupe))


# ### Let's check how confusion matrices are affected
# Please remember labels are computed using a .5 threshold

# In[ ]:


fig = plt.figure(figsize=(20, 15))
plt.rc('legend', fontsize=10) 
plt.rc('axes', labelsize=10)
plt.rc('axes', titlesize=10)
gs = gridspec.GridSpec(math.ceil(len(dupes) / 2), 2)
for i_w, weight in enumerate(dupes):
    ax = plt.subplot(gs[int(i_w / 2), i_w % 2])
    class_names = ["safe", "unsafe"]
    cnf_matrix = confusion_matrix(target, oof_label[:, i_w])
    plot_confusion_matrix(cnf_matrix, classes=class_names, 
                          normalize=False,
                          title='Matrix for duplication = %.1f, Gini %.6f' 
                          % (weight, eval_gini(target, oof_proba[:, i_w])))
plt.tight_layout()


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
plt.rc('legend', fontsize=18) 
plt.rc('axes', labelsize=18)
plt.rc('axes', titlesize=24)
for i_w, dupe in enumerate(dupes):
    # Get False positives, true positives and the list of thresholds used to compute them
    fpr, tpr, thresholds = roc_curve(target, oof_proba[:, i_w])
    # Compute recall, precision and f1_score
    recall = tpr
    precision = tpr / (fpr + tpr + 1e-5)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-5)
    # Finally plot the f1_scores against thresholds
    plt.plot(thresholds[-30000:], f1_scores[-30000:], 
             label="duplication rate x %.1f" % dupe)
plt.title("F1 scores against threshold for different duplication rate")
plt.ylabel("F1 score")
plt.xlabel("Probability thresholds")
plt.legend(loc="lower left")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# I must say this is a strange plot with lots of overlap. I'm really puzzled by this...
# 
# At least we can see that you really need a strong duplication rate to spread probabilities the way scale_pos_weight does. As far as I'm concerned I will probably go the scale_pos_weight way. If you plan to merge predictions you will have to make sure they span approximately the same way in the [0, 1] space, all the more if you use geometric or harmonic mean...  
