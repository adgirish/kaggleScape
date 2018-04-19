import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv').fillna(' ')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
tr_ids = train[['id']]
train[class_names] = train[class_names].astype(np.int8)
target = train[class_names]

print('Tfidf word vector')
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

print('Tfidf char vector')
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

print('stack both')
#train_features = hstack([train_char_features, train_word_features])
#test_features = hstack([test_char_features, test_word_features])

#train_features = train_word_features
#test_features = test_word_features

train_features = hstack([train_char_features, train_word_features]).tocsr()
test_features = hstack([test_char_features, test_word_features]).tocsr()

scores = []
scores_classes = np.zeros((len(class_names), 5))

submission = pd.DataFrame.from_dict({'id': test['id']})
submission_oof = train[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

idpred = tr_ids
number_of_folds = 5

#kfolder=StratifiedKFold(train_text, n_folds=number_of_folds,shuffle=True, random_state=15)
kfolder = KFold(n_splits=number_of_folds, shuffle=True, random_state=239)


for j, (class_name) in enumerate(class_names):
    
    print('class_name is: ' + class_name)
    avreal = target[class_name]
    lr_cv_sum = 0
    lr_test_pred = np.zeros(test.shape[0])
    lr_avpred = np.zeros(train.shape[0])
    
    for i, (train_index, val_index) in enumerate(kfolder.split(train_features)):
        print(train_index)
        print(val_index)
        X_train, X_val = train_features[train_index], train_features[val_index]
        y_train, y_val = target.loc[train_index], target.loc[val_index]

        classifier = Ridge(alpha=20, copy_X=True, fit_intercept=True, solver='auto',max_iter=100,normalize=False, random_state=0,  tol=0.0025)
        
        #classifier = Lasso(alpha=0.1,normalize=True, max_iter=1e5)
    #    classifier = ElasticNet(alpha=1.0, l1_ratio =0.5)
        classifier.fit(X_train, y_train[class_name])
        scores_val = classifier.predict(X_val)
        lr_avpred[val_index] = scores_val
        lr_test_pred += classifier.predict(test_features)
        scores_classes[j][i] = roc_auc_score(y_val[class_name], scores_val)
        print('\n Fold %02d class %s AUC: %.6f' % ((i+1), class_name, scores_classes[j][i]))

    lr_cv_score = (lr_cv_sum / number_of_folds)
    lr_oof_auc = roc_auc_score(avreal, lr_avpred)
    print('\n Average class %s AUC:\t%.6f' % (class_name, np.mean(scores_classes[j])))
    print(' Out-of-fold class %s AUC:\t%.6f' % (class_name, lr_oof_auc))

    submission[class_name] = lr_test_pred / number_of_folds
    submission_oof[class_name] = lr_avpred

#print('\n Overall AUC:\t%.6f' % (np.mean(scores_classes)))
submission.to_csv('5-fold_elast_test.csv', index=False)
submission_oof.to_csv('5-fold_ridge_train.csv', index=False)