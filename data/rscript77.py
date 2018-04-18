import gc
import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import lightgbm as lgb


# Load Data
dtype = {
    'id': str,
    'teacher_id': str,
    'teacher_prefix': str,
    'school_state': str,
    'project_submitted_datetime': str,
    'project_grade_category': str,
    'project_subject_categories': str,
    'project_subject_subcategories': str,
    'project_title': str,
    'project_essay_1': str,
    'project_essay_2': str,
    'project_essay_3': str,
    'project_essay_4': str,
    'project_resource_summary': str,
    'teacher_number_of_previously_posted_projects': int,
    'project_is_approved': np.uint8,
}
data_path = os.path.join('..', 'input')
train = pd.read_csv(os.path.join(data_path, 'train.csv'), dtype=dtype, low_memory=True)
test = pd.read_csv(os.path.join(data_path, 'test.csv'), dtype=dtype, low_memory=True)
res = pd.read_csv(os.path.join(data_path, 'resources.csv'))

print(train.head())
# print(test.head())
print(train.shape, test.shape)


# Preprocess data
train['project_essay'] = train.apply(lambda row: ' '.join([
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    ]), axis=1)
test['project_essay'] = test.apply(lambda row: ' '.join([
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    ]), axis=1)

# Extract features
def extract_features(df):
    df['project_title_len'] = df['project_title'].apply(lambda x: len(str(x)))
    df['project_essay_1_len'] = df['project_essay_1'].apply(lambda x: len(str(x)))
    df['project_essay_2_len'] = df['project_essay_2'].apply(lambda x: len(str(x)))
    df['project_essay_3_len'] = df['project_essay_3'].apply(lambda x: len(str(x)))
    df['project_essay_4_len'] = df['project_essay_4'].apply(lambda x: len(str(x)))
    df['project_resource_summary_len'] = df['project_resource_summary'].apply(lambda x: len(str(x)))
    
    df['project_title_wc'] = df['project_title'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_1_wc'] = df['project_essay_1'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_2_wc'] = df['project_essay_2'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_3_wc'] = df['project_essay_3'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_4_wc'] = df['project_essay_4'].apply(lambda x: len(str(x).split(' ')))
    df['project_resource_summary_wc'] = df['project_resource_summary'].apply(lambda x: len(str(x).split(' ')))
  
extract_features(train)
extract_features(test)

train.drop([
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4'], axis=1, inplace=True)
test.drop([
    'project_essay_1', 
    'project_essay_2', 
    'project_essay_3', 
    'project_essay_4'], axis=1, inplace=True)

df_all = pd.concat([train, test], axis=0)
gc.collect()

# Accepted projects counter (gave imrovement on CV but worse on LB, need to be implemented withing a CV loop with splitting data by time)
# df_all['project_is_approved'].fillna(0, inplace=True)
# cumsums = df_all[
#             ['id', 
#             'teacher_id', 
#             'project_submitted_datetime', 
#             'project_is_approved']].\
#         sort_values('project_submitted_datetime').\
#         groupby(['teacher_id']).agg({'project_is_approved': lambda x: x.shift().sum(), 'id': 'first'}).fillna(0).\
#         groupby(level=0).agg({'project_is_approved': 'cumsum', 'id': 'first'}).reset_index()
# cumsums = pd.DataFrame(cumsums)
# cumsums.rename(columns={'project_is_approved': 'teacher_number_of_previously_accepted_projects'}, inplace=True)
# print(cumsums.head())
# train = train.merge(cumsums, on=['id', 'teacher_id'], how='left')
# test = test.merge(cumsums, on=['id', 'teacher_id'], how='left')

# train['approve_rate'] = (train['teacher_number_of_previously_accepted_projects'] + 5)/\
#     (train['teacher_number_of_previously_posted_projects'] + 10)
# test['approve_rate'] = (test['teacher_number_of_previously_accepted_projects'] + 5)/\
#     (test['teacher_number_of_previously_posted_projects'] + 10)

# print(train.head())
# print(test.head())

# Merge with resources
res = pd.DataFrame(res[['id', 'quantity', 'price']].groupby('id').agg(\
    {
        'quantity': [
            'sum',
            'min', 
            'max', 
            'mean', 
            'std', 
            # lambda x: len(np.unique(x)),
        ],
        'price': [
            'count', 
            'sum', 
            'min', 
            'max', 
            'mean', 
            'std', 
            lambda x: len(np.unique(x)),
        ]}
    )).reset_index()
res.columns = ['_'.join(col) for col in res.columns]
res.rename(columns={'id_': 'id'}, inplace=True)
res['mean_price'] = res['price_sum']/res['quantity_sum']
# res['price_max_to_price_min'] = res['price_max']/res['price_min']
# res['quantity_max_to_quantity_min'] = res['quantity_max']/res['quantity_min']

print(res.head())
train = train.merge(res, on='id', how='left')
test = test.merge(res, on='id', how='left')
del res
gc.collect()

# Preprocess columns with label encoder
print('Label Encoder...')
cols = [
    'teacher_id', 
    'teacher_prefix', 
    'school_state', 
    'project_grade_category', 
    'project_subject_categories', 
    'project_subject_subcategories'
]

for c in tqdm(cols):
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
del le
gc.collect()
print('Done.')


# Preprocess timestamp
print('Preprocessing timestamp...')
def process_timestamp(df):
    df['year'] = df['project_submitted_datetime'].apply(lambda x: int(x.split('-')[0]))
    df['month'] = df['project_submitted_datetime'].apply(lambda x: int(x.split('-')[1]))
    df['date'] = df['project_submitted_datetime'].apply(lambda x: int(x.split(' ')[0].split('-')[2]))
    df['day_of_week'] = pd.to_datetime(df['project_submitted_datetime']).dt.weekday
    df['hour'] = df['project_submitted_datetime'].apply(lambda x: int(x.split(' ')[-1].split(':')[0]))
    df['minute'] = df['project_submitted_datetime'].apply(lambda x: int(x.split(' ')[-1].split(':')[1]))
    df['project_submitted_datetime'] = pd.to_datetime(df['project_submitted_datetime']).values.astype(np.int64)

process_timestamp(train)
process_timestamp(test)
print('Done.')

# Preprocess text
print('Preprocessing text...')
cols = [
    'project_title', 
    'project_essay', 
    'project_resource_summary'
]
n_features = [
    400, 
    4040, 
    400,
]

for c_i, c in tqdm(enumerate(cols)):
    tfidf = TfidfVectorizer(
        max_features=n_features[c_i],
        norm='l2',
        )
    tfidf.fit(df_all[c])
    tfidf_train = np.array(tfidf.transform(train[c]).toarray(), dtype=np.float16)
    tfidf_test = np.array(tfidf.transform(test[c]).toarray(), dtype=np.float16)

    for i in range(n_features[c_i]):
        train[c + '_tfidf_' + str(i)] = tfidf_train[:, i]
        test[c + '_tfidf_' + str(i)] = tfidf_test[:, i]
        
    del tfidf, tfidf_train, tfidf_test
    gc.collect()
    
print('Done.')
del df_all
gc.collect()

# Prepare data
cols_to_drop = [
    'id',
    'teacher_id',
    'project_title', 
    'project_essay', 
    'project_resource_summary',
    'project_is_approved',
]
X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train['project_is_approved']
X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test['id'].values
feature_names = list(X.columns)
print(X.shape, X_test.shape)

del train, test
gc.collect()

# Build the model
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
auc_buf = []   

for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 14,
        'num_leaves': 31,
        'learning_rate': 0.025,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 1,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0,
    }  

    lgb_train = lgb.Dataset(
        X.loc[train_index], 
        y.loc[train_index], 
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X.loc[valid_index], 
        y.loc[valid_index],
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(60):
            if i < len(tuples):
                print(tuples[i])
            else:
                break
            
        del importance, model_fnames, tuples

    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
    auc = roc_auc_score(y.loc[valid_index], p)

    print('{} AUC: {}'.format(cnt, auc))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    auc_buf.append(auc)

    cnt += 1
    if cnt > 0: # Comment this to run several folds
        break
    
    del model, lgb_train, lgb_valid, p
    gc.collect

auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))

preds = p_buf/cnt

# Prepare submission
subm = pd.DataFrame()
subm['id'] = id_test
subm['project_is_approved'] = preds
subm.to_csv('submission.csv', index=False)
