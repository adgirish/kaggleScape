from multiprocessing import *
import numpy as np
import pandas as pd
from sklearn import *
import tensorflow as tf
import lightgbm as lgb
import xgboost as xgb
from scipy.io.wavfile import read as scipy_read
from scipy.io.wavfile import write as scipy_write
from scipy import signal
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import glob, random, os

def fscipy_read(path):
    try:
        return scipy_read(path)[1]
    except:
        print(path)
        return

def transform_df(df):
    df = pd.DataFrame(df)
    df['wav'] = df['path'].map(lambda x: fscipy_read(x))
    df['rate'] = df['path'].map(lambda x: fscipy_read(x).shape)
    df['min'] = df['path'].map(lambda x: np.min(fscipy_read(x)))
    df['max'] = df['path'].map(lambda x: np.max(fscipy_read(x)))
    df['med'] = df['path'].map(lambda x: np.median(fscipy_read(x)))
    df['mea'] = df['path'].map(lambda x: np.mean(fscipy_read(x)))
    df['a_min'] = df['path'].map(lambda x: np.min(np.abs(fscipy_read(x))))
    df['a_max'] = df['path'].map(lambda x: np.max(np.abs(fscipy_read(x))))
    df['a_med'] = df['path'].map(lambda x: np.median(np.abs(fscipy_read(x))))
    df['a_mea'] = df['path'].map(lambda x: np.mean(np.abs(fscipy_read(x))))
    df['quiet1'] = df['path'].map(lambda x: (np.abs(fscipy_read(x))<100).sum())
    df['quiet2'] = df['path'].map(lambda x: (np.abs(fscipy_read(x))<200).sum())
    df['quiet3'] = df['path'].map(lambda x: (np.abs(fscipy_read(x))<1000).sum())
    df['b_sum2'] = df['path'].map(lambda x: np.sum(fscipy_read(x)[::2]))
    df['b_sum4'] = df['path'].map(lambda x: np.sum(fscipy_read(x)[::4]))
    df['b_sum8'] = df['path'].map(lambda x: np.sum(fscipy_read(x)[::8]))
    df['b_sum16'] = df['path'].map(lambda x: np.sum(fscipy_read(x)[::16]))
    df['b_sum32'] = df['path'].map(lambda x: np.sum(fscipy_read(x)[::32]))
    #df['freq_sum'] = df['path'].map(lambda x: np.sum(np.abs(np.fft.fft(fscipy_read(x)))))
    #df['freq_med'] = df['path'].map(lambda x: np.median(np.abs(np.fft.fft(fscipy_read(x)))))
    #df['freq_mean'] = df['path'].map(lambda x: np.mean(np.abs(np.fft.fft(fscipy_read(x)))))
    #df['peaks_100'] = df['path'].map(lambda x: len(signal.find_peaks_cwt(fscipy_read(x), np.arange(1, 100))))
    #df['peaks_200'] = df['path'].map(lambda x: len(signal.find_peaks_cwt(fscipy_read(x), np.arange(1, 200))))
    #df['peaks_300'] = df['path'].map(lambda x: len(signal.find_peaks_cwt(fscipy_read(x), np.arange(1, 300))))
    #df['peaks_400'] = df['path'].map(lambda x: len(signal.find_peaks_cwt(fscipy_read(x), np.arange(1, 400))))
    #df['peaks_500'] = df['path'].map(lambda x: len(signal.find_peaks_cwt(fscipy_read(x), np.arange(1, 500))))
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

#if not os.path.exists('../input/train/audio/silence'): #use on your own device to add files and use for noise in class files
#        os.makedirs('../input/train/audio/silence')
        
samples = pd.DataFrame(glob.glob('../input/train/audio/_background_noise_/**.wav'), columns=['path'])
n = 10000
for w in samples.path:
    rate, wav = scipy_read(w)
    print(w, rate, wav.shape)
    for i in range(0,len(wav)-16000,16000):
        #scipy_write('../input/train/audio/silence/' + str(n) + '.wav', rate, wav[i:i+16000])
        n +=1

audio_classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence','unknown']
print('train...')
train = pd.DataFrame(glob.glob('../input/train/audio/**/**.wav'), columns=['path'])
train['label'] = train['path'].map(lambda x: x.split('/')[4])
train['label'] = train['label'].map(lambda x: x if x in audio_classes else 'unknown')
train['fname'] = train['path'].map(lambda x: x.split('/')[5])
train = multi_transform(train)
train = train[train['rate']==(16000,)]
unknown = train[train['label']=='unknown'].sample(396)
train = train[train['label']!='unknown']
train = pd.concat((train, unknown), axis=0, ignore_index=True).reset_index(drop=True)
print('After Shape: ', train.values.shape)
print(train['label'].value_counts())
print('test...')
test = pd.DataFrame(glob.glob('../input/test/audio/**'), columns=['path'])
test = test[test['path']!='../input/test/audio/clip_a2fda9e27.wav']
test['fname'] = test['path'].map(lambda x: x.split('/')[4])
test = multi_transform(test)

i = 5
print(train['path'].iloc[i], train['rate'].iloc[i])
plt.plot(train['wav'].iloc[i])
plt.xlabel("Time (samples)")
plt.title(train['label'].iloc[i])
plt.show()
plt.savefig('img.png')

#the rest requires a test set
"""
freq, time, spec = signal.spectrogram(scipy_read(train['path'][0])[1], scipy_read(train['path'][0])[0])
plt.plot(spec)
plt.imshow(spec)

col = ['min','max','med', 'mea','a_min','a_max','a_med', 'a_mea', 'quiet1', 'quiet2', 'quiet3','b_sum2','b_sum4','b_sum8','b_sum16','b_sum32'] #, 'peaks_100', 'peaks_200', 'peaks_300', 'peaks_400', 'peaks_500']

features1 = np.array([np.array(row[1000:15000:40]).astype(int) for row in train['wav'].values])
areas1 = np.array([np.mean(np.split(np.abs(row),100), axis=1) for row in train['wav'].values]) * 2
areas2 = np.array([np.median(np.split(np.abs(row),100), axis=1) for row in train['wav'].values])
features1 = np.concatenate([features1, train[col].values, areas1, areas2], axis=1)
lbl = preprocessing.LabelEncoder()
y = lbl.fit_transform(train['label'].values)
print(features1.shape)

features2 = np.array([np.array(row[1000:15000:40]).astype(int) for row in test['wav'].values])
areas1 = np.array([np.mean(np.split(np.abs(row),100), axis=1) for row in test['wav'].values]) * 2
areas2 = np.array([np.median(np.split(np.abs(row),100), axis=1) for row in test['wav'].values])
features2 = np.concatenate([features2, test[col].values, areas1, areas2], axis=1)
fname = test['fname'].values
print(features2.shape)

#XGBoost
fold = 5
pred = []
for i in range(fold):
    np.random.seed(i)
    random.seed(i)
    x1, x2, y1, y2 = model_selection.train_test_split(features1, y, test_size=0.2, random_state=i)
    params = {'eta': 0.2, 'max_depth': 4, 'objective': 'multi:softprob', 'eval_metric': ['merror'], 'num_class': len(lbl.classes_), 'seed': i, 'silent': True}
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 500,  watchlist, verbose_eval=20, early_stopping_rounds=30)
    pred.append(model.predict(xgb.DMatrix(features2), ntree_limit=model.best_ntree_limit+10))
    if i>0:
        pred[0] = np.array(pred[0]) + np.array(pred[i])
pred = pd.DataFrame(pred[0] / fold, columns=lbl.classes_)
pred['fname'] = fname

plt.rcParams['figure.figsize'] = (8.0, 12.0)
xgb.plot_importance(booster=model); plt.show(); #plt.savefig('xgb_fi.png')

#LightGBM
#params = {'learning_rate': 0.2, 'max_depth': 4, 'boosting_type': 'gbdt', 'objective': 'multiclass', 'metric' : ['multi_logloss','multi_error'], 'is_training_metric': True, 'num_class': len(lbl.classes_), 'seed': 2}
#model = lgb.train(params, lgb.Dataset(x1, label=y1), 2000, lgb.Dataset(x2, label=y2), verbose_eval=10, early_stopping_rounds=20)
#pred = pd.DataFrame(model.predict(features2, num_iteration=model.best_iteration), columns=lbl.classes_)
#pred['fname'] = fname

#TensorFlow Classifier
#x1, x2, y1, y2 = model_selection.train_test_split(features1, y, test_size=0.3, random_state=8)
#x1 = x1.astype(np.float32)
#x2 = x2.astype(np.float32)
#y1 = y1.astype(np.int)
#y2 = y2.astype(np.int)
#col = [tf.feature_column.numeric_column('x', shape=x1.shape[1:])]
#opt = tf.train.AdamOptimizer(learning_rate=0.0003, beta1=0.7, beta2=0.999, epsilon=1e-07, name='Adam')
#clf = tf.estimator.DNNClassifier(optimizer=opt, feature_columns=col, hidden_units=[150, 250, 200], n_classes=len(lbl.classes_))
#clf.train(input_fn=tf.estimator.inputs.numpy_input_fn(x={'x': x1}, y=y1, shuffle=True), max_steps=1000)
#score = clf.evaluate(input_fn=tf.estimator.inputs.numpy_input_fn(x={'x': x2}, y=y2, num_epochs=1, shuffle=True))["accuracy"]
#print('accuracy: ', score)
#pred = clf.predict(input_fn=tf.estimator.inputs.numpy_input_fn(x={'x': features2.astype(np.float32)}, num_epochs=1, shuffle=False))
#pred = [p["probabilities"].astype(np.float32) for p in pred]
#pred  = pd.DataFrame(pred, columns=lbl.classes_)
#pred['fname'] = fname

label = []
for r in pred[lbl.classes_].values:
    r = list(r)
    label.append([j[1] for j in sorted([[r[i],lbl.classes_[i]] for i in range(len(lbl.classes_))], reverse=True)][0])
label = [l if l in audio_classes else 'silence' for l in label]
pred['label'] = label

sub = pd.read_csv('../input/sample_submission.csv')
sub = pd.merge(sub, pred[['fname','label']], how='left', on='fname')
sub = sub[['fname','label_y']].fillna('unknown')
sub.columns = ['fname','label']
sub.to_csv("submission.csv", index=False)
print(sub.label.value_counts())
"""
#:)

pd.read_csv('../input/sample_submission.csv').to_csv('sample.csv', index=False)

#from subprocess import check_output
#print(check_output(["find","/", "-name", "tensorflow"]).decode("utf8"))
#print(check_output(["ls", "/opt/conda/lib/python3.6/site-packages/tensorflow/examples"]).decode("utf8"))
#No Speech Commands