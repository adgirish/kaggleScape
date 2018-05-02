
# coding: utf-8

# **Update**
# 
# *Included Neural Networks in the learning ensemble*

# Hi everybody,
# 
# in this notebook, I'm going to present some text and numeric feature extraction techniques. Some of them are already presented in the other kernels and some are new. We will focus mostly on the text and we try to place ourselves in the shoes of grant administrators to see what they might focus on when processing the application, *consciously* or *unconsciously*.
# 
# Let's start with importing modules and loading the files:

# In[1]:


import pylab as pl # linear algebra + plots
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import gc
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict, Counter
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from scipy.stats import pearsonr
from scipy.sparse import hstack
from multiprocessing import Pool

Folder = "../input/"
Ttr = pd.read_csv(Folder + 'train.csv')
Tts = pd.read_csv(Folder + 'test.csv', low_memory=False)
R = pd.read_csv(Folder + 'resources.csv')


# **Data Cleaning**
# 
# We know from the data description page that the essay column formats had changed on 2016-05-17, and thereafter, there are only 2 essays; essay 1 matches to the combination of essays 1&2 and new essay 2 is somehow equal to old essays 3&4.
# 
# So, I first move the contents of 'project_essay_2' to 'project_essay_4' when essay 4 is nan, then we simply combine 1&2 and 3&4 to make a uniform dataset.

# In[2]:


# combine the tables into one
target = 'project_is_approved'
Ttr['tr'] = 1; Tts['tr'] = 0
Ttr['ts'] = 0; Tts['ts'] = 1

T = pd.concat((Ttr,Tts))

T.loc[T.project_essay_4.isnull(), ['project_essay_4','project_essay_2']] =     T.loc[T.project_essay_4.isnull(), ['project_essay_2','project_essay_4']].values

T[['project_essay_2','project_essay_3']] = T[['project_essay_2','project_essay_3']].fillna('')

T['project_essay_1'] = T.apply(lambda row: ' '.join([str(row['project_essay_1']), 
                                                     str(row['project_essay_2'])]), axis=1)
T['project_essay_2'] = T.apply(lambda row: ' '.join([str(row['project_essay_3']),
                                                     str(row['project_essay_4'])]), axis=1)

T = T.drop(['project_essay_3', 'project_essay_4'], axis=1)


# **Resource Features**
# 
# Here we extract some features from the resource file. For each application, there are some resources listed in this file. We can extract how many items and at what prices are requested. minimum, maximum and average price and quantity of each item and for all requested items per application can be important in the decision-making process.
# 
# Also, I combine the resource description columns and make a new text column in table T. Later, we will do text analysis on this column as well.

# In[7]:


R['priceAll'] = R['quantity']*R['price']
newR = R.groupby('id').agg({'description':'count',
                            'quantity':'sum',
                            'price':'sum',
                            'priceAll':'sum'}).rename(columns={'description':'items'})
newR['avgPrice'] = newR.priceAll / newR.quantity
numFeatures = ['items', 'quantity', 'price', 'priceAll', 'avgPrice']

for func in ['min', 'max', 'mean']:
    newR = newR.join(R.groupby('id').agg({'quantity':func,
                                          'price':func,
                                          'priceAll':func}).rename(
                                columns={'quantity':func+'Quantity',
                                         'price':func+'Price',
                                         'priceAll':func+'PriceAll'}).fillna(0))
    numFeatures += [func+'Quantity', func+'Price', func+'PriceAll']

newR = newR.join(R.groupby('id').agg({'description':lambda x:' '.join(x.values.astype(str))}).rename(
    columns={'description':'resource_description'}))

T = T.join(newR, on='id')

del Ttr, Tts, R, newR
gc.collect();


# **Statistical Features**
# 
# We know some teachers have applied many times, and knowing the history of their applications, can be helpful to predict approval. So, I convert the teacher_id to numeric values and include it in my numeric features.
# 
# Often times, knowing the statistics of categorical features, i.e. knowing how many times a certain value has repeated in the dataset can help. So let's extract this information:

# In[8]:


le = LabelEncoder()
T['teacher_id'] = le.fit_transform(T['teacher_id'])
numFeatures += ['teacher_number_of_previously_posted_projects','teacher_id']

statFeatures = []
for col in ['school_state', 'teacher_id', 'teacher_prefix', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories', 'teacher_number_of_previously_posted_projects']:
    Stat = T[['id', col]].groupby(col).agg('count').rename(columns={'id':col+'_stat'})
    Stat /= Stat.sum()
    T = T.join(Stat, on=col)
    statFeatures.append(col+'_stat')


# **Sentimental Analysis**
# 
# With the help of textblob module, we can find polarity and subjectivity of texts to some extent. It is, unfortunately, a little time-consuming. There might be other modules that work faster like [VADER-Sentiment](https://github.com/cjhutto/vaderSentiment). Though, I haven't checked other modules. Their quality of analysis can also be different. Have you ever tried other modules? Do you know any better one?
# 
# Another way of doing (sort of) sentimental analysis is to check for certain words and characters in the texts. I, personally, for example, feel uncomfortable if a text has so many exclamation marks :D. But, seriously, some of these may have an unconscious effect on the examiner. For example, if any words are bolded by ", or the number of sentences (number of "."), number of paragraphs (\r), talking about money ($) or percentages (%), having a URL (http), etc. can influence the decision. What other words or characters do you think can be important?
# 
# Talking about **I** or **WE** and having positive or negative words and phrases like that can also be influential. In one of the following sections (**Text Features**), by extracting n-grams, I hope to catch such phrases if they appear as repeated patterns.
# 
# The number of words or the length of the texts can be another factor that can influence the decision unconsciously (or even consciously!). Number of transitional words, verbs, adjectives, adverbs, etc. in an essay can also indicate some aspects of the quality of the text.
# 
# But, certainly, the quality of the essays is the most effective factor in my opinion. Things like the grammar errors, spelling errors, quality of the texts, word choices etc. are very important. Another important factor, if I was a grant examiner, would have been to check if the application writer could relate their needs to the resources they want through essays and project title. One primitive way to do this is to check for common words in different texts. Let me know if you know any better way to do these type of analysis.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'textColumns = [\'project_essay_1\', \'project_essay_2\', \'project_resource_summary\', \'resource_description\', \'project_title\']\n\ndef getSentFeat(s):\n    sent = TextBlob(s).sentiment\n    return (sent.polarity, sent.subjectivity)\n\nprint(\'sentimental analysis\')\nwith Pool(4) as p:\n    for col in textColumns:\n        temp = pl.array(list(p.map(getSentFeat, T[col])))\n        T[col+\'_pol\'] = temp[:,0]\n        T[col+\'_sub\'] = temp[:,1]\n        numFeatures += [col+\'_pol\', col+\'_sub\']\n\nprint(\'key words\')\nKeyChars = [\'!\', \'\\?\', \'@\', \'#\', \'\\$\', \'%\', \'&\', \'\\*\', \'\\(\', \'\\[\', \'\\{\', \'\\|\', \'-\', \'_\', \'=\', \'\\+\',\n            \'\\.\', \':\', \';\', \',\', \'/\', \'\\\\\\\\r\', \'\\\\\\\\t\', \'\\\\"\', \'\\.\\.\\.\', \'etc\', \'http\']\nfor col in textColumns:\n    for c in KeyChars:\n        T[col+\'_\'+c] = T[col].apply(lambda x: len(re.findall(c, x.lower())))\n        numFeatures.append(col+\'_\'+c)\n\n#####\nprint(\'num words\')\nfor col in textColumns:\n    T[\'n_\'+col] = T[col].apply(lambda x: len(x.split()))\n    numFeatures.append(\'n_\'+col)\n\n#####\nprint(\'word tags\')\nTags = [\'CC\', \'CD\', \'DT\', \'IN\', \'JJ\', \'LS\', \'MD\', \'NN\', \'NNS\', \'NNP\', \'NNPS\', \n        \'PDT\', \'POS\', \'PRP\', \'PRP$\', \'RB\', \'RBR\', \'RBS\', \'RP\', \'SYM\', \'TO\', \n        \'UH\', \'VB\', \'VBD\', \'VBG\', \'VBN\', \'VBP\', \'VBZ\', \'WDT\', \'WP\', \'WP$\', \'WRB\']\ndef getTagFeat(s):\n    d = Counter([t[1] for t in pos_tag(s.split())])\n    return [d[t] for t in Tags]\n\nwith Pool(4) as p:\n    for col in textColumns:\n        temp = pl.array(list(p.map(getTagFeat, T[col])))\n        for i, t in enumerate(Tags):\n            if temp[:,i].sum() == 0:\n                continue\n            T[col+\'_\'+t] = temp[:, i]\n            numFeatures += [col+\'_\'+t]\n\n#####\nprint(\'common words\')\nfor i, col1 in enumerate(textColumns[:-1]):\n    for col2 in textColumns[i+1:]:\n        T[\'%s_%s_common\' % (col1, col2)] = T.apply(lambda row:len(set(re.split(\'\\W\', row[col1])).intersection(re.split(\'\\W\', row[col2]))), axis=1)\n        numFeatures.append(\'%s_%s_common\' % (col1, col2))\n')


# Guess what! someone didn't like **!**s in essays. 

# In[ ]:


pl.figure(figsize=(15,5))
sns.violinplot(data=T,x=target,y='project_essay_2_!');
pl.figure(figsize=(15,5))
sns.violinplot(data=T,x=target,y='project_essay_1_!');


# **Time Features**
# 
# The time at which the proposal was submitted can be important. Most importantly, we know thanks to [Heads or Tails](https://www.kaggle.com/headsortails/an-educated-guess-update-feature-engineering) that there is a slight approval rate modulation over time. So we need to extract date info. Day of the week it has been posted can also play a role. I doubt if the hour it was submitted has any significance, but let's let the decision trees take care of that. Next, let's extract some statistics from time features as well.

# In[ ]:


dateCol = 'project_submitted_datetime'
def getTimeFeatures(T):
    T['year'] = T[dateCol].apply(lambda x: x.year)
    T['month'] = T[dateCol].apply(lambda x: x.month)
    T['day'] = T[dateCol].apply(lambda x: x.day)
    T['dow'] = T[dateCol].apply(lambda x: x.dayofweek)
    T['hour'] = T[dateCol].apply(lambda x: x.hour)
    T['days'] = (T[dateCol]-T[dateCol].min()).apply(lambda x: x.days)
    return T

T[dateCol] = pd.to_datetime(T[dateCol])
T = getTimeFeatures(T)

P_tar = T[T.tr==1][target].mean()
timeFeatures = ['year', 'month', 'day', 'dow', 'hour', 'days']
for col in timeFeatures:
    Stat = T[['id', col]].groupby(col).agg('count').rename(columns={'id':col+'_stat'})
    Stat /= Stat.sum()
    T = T.join(Stat, on=col)
    statFeatures.append(col+'_stat')

numFeatures += timeFeatures
numFeatures += statFeatures


# **Polynomial Features**
# 
# So far, I have extracted some numerical features. Often it helps the decision trees to provide some polynomial features to them. Here I include first-order interaction polynomials, and I check for the significance of the new variable before adding it to the columns. I add it only if it really helps to predict the approval better. A trick that I'm using here is that, maybe, the division of two variables is more significantly predicting the target! That would be the case if 1/V is a more significant predictor than V. So, I check for the significance of 1/(V+1) and V+1 (+1 is to avoid production or division by 0), and replace the most significant one to the original variable V. What do you think about this? It certainly helped though!
# 
# By checking the significance and correlation in training set, there will be an over-training chance, which I'm trying to decrease by computing the average of correlations and p-values over randomly selected subsets.

# In[ ]:


get_ipython().run_cell_magic('time', '', "T2 = T[numFeatures+['id','tr','ts',target]].copy()\nTtr = T2[T.tr==1]\nTar_tr = Ttr[target].values\nn = 10\ninx = [pl.randint(0, Ttr.shape[0], int(Ttr.shape[0]/n)) for k in range(n)]\n# inx is used for crossvalidation of calculating the correlation and p-value\nCorr = {}\nfor c in numFeatures:\n    # since some values might be 0s, I use x+1 to avoid missing some important relations\n    C1,P1=pl.nanmean([pearsonr(Tar_tr[inx[k]],   (1+Ttr[c].iloc[inx[k]])) for k in range(n)], 0)\n    C2,P2=pl.nanmean([pearsonr(Tar_tr[inx[k]], 1/(1+Ttr[c].iloc[inx[k]])) for k in range(n)], 0)\n    if P2<P1:\n        T2[c] = 1/(1+T2[c])\n        Corr[c] = [C2,P2]\n    else:\n        T2[c] = 1+T2[c]\n        Corr[c] = [C1,P1]\n\npolyCol = []\nthrP = 0.01\nthrC = 0.02\nprint('columns \\t\\t\\t Corr1 \\t\\t Corr2 \\t\\t Corr Combined')\nfor i, c1 in enumerate(numFeatures[:-1]):\n    C1, P1 = Corr[c1]\n    for c2 in numFeatures[i+1:]:\n        C2, P2 = Corr[c2]\n        V = T2[c1] * T2[c2]\n        Vtr = V[T2.tr==1].values\n        C, P = pl.nanmean([pearsonr(Tar_tr[inx[k]], Vtr[inx[k]]) for k in range(n)], 0)\n        if P<thrP and abs(C) - max(abs(C1),abs(C2)) > thrC:\n            T[c1+'_'+c2+'_poly'] = V\n            polyCol.append(c1+'_'+c2+'_poly')\n            print(c1+'_'+c2, '\\t\\t(%g, %g)\\t(%g, %g)\\t(%g, %g)'%(C1,P1, C2,P2, C,P))\n\nnumFeatures += polyCol\nprint(len(numFeatures))\ndel T2, Ttr\ngc.collect();")


# For example, the variable created out of *maxPrice* and *meanPrice* is much more informative:

# In[ ]:


pl.figure(figsize=(15,5));sns.violinplot(data=T,x=target,y='maxPrice')
pl.figure(figsize=(15,5));sns.violinplot(data=T,x=target,y='meanPrice')
pl.figure(figsize=(15,5));sns.violinplot(data=T,x=target,y='maxPrice_meanPrice_poly');


# **Categorical Features**
# 
# Next, we include categorical features. Two well-known ways are to use one-hot encoding (one column per value with 0s and 1s) or label encoding (assigning a number to each value). I tried both; sometimes one-hot works better and sometimes label encoder. Currently, I have not activated the one-hot encoder. Categorical features are teacher prefix, state, grade, and subject categories.

# In[9]:


def getCatFeatures(T, Col, Encoder='OneHot'):
    ohe = OneHotEncoder()
    le = LabelEncoder()
    if Encoder=='OneHot':
        X = ohe.fit_transform(le.fit_transform(T[Col].fillna('')).reshape((-1,1)))
    else:
        X = le.fit_transform(T[Col].fillna(''))
    return X

Encoder = 'OneHot'
X_tp = getCatFeatures(T, 'teacher_prefix', Encoder)
X_sc = getCatFeatures(T, 'school_state', Encoder)
X_pgc = getCatFeatures(T, 'project_grade_category', Encoder)
X_psc = getCatFeatures(T, 'project_subject_categories', Encoder)
X_pssc = getCatFeatures(T, 'project_subject_subcategories', Encoder)


if Encoder=='OneHot':
    X_cat = hstack((X_tp, X_sc, X_pgc, X_psc, X_pssc))
else:
    X_cat = pl.array((X_tp, X_sc, X_pgc, X_psc, X_pssc)).T

del X_tp, X_sc, X_pgc, X_psc, X_pssc


# **Text Features**
# 
# Finally, we do text analysis. For this section, I used both Tf-IDF and count vectorizer and interestingly, count vectorizer with binary features, showing only if a word is in the text, has the best performance in my experience. Other than that, since there are mis-spellings in the texts, it would have helped to check for spelling errors first. I found "TextBlob" and "autocorrect" modules for this purpose but, unfortunately, it was so slow and I didn't use it at last. Do you know any better way to do that?Also, I decided not using any stop words because some of them can actually be useful in this case and after all they are only a few words.
# 
# I tried using dimensionality reduction techniques to reduce the dimensions following the idea of Latent Semantic Analysis, but it didn't help the prediction as well.

# In[12]:


get_ipython().run_cell_magic('time', '', "# from nltk.stem.wordnet import WordNetLemmatizer\n# from autocorrect import spell  # as spell checker and corrector\n# L = WordNetLemmatizer()\np = PorterStemmer()\ndef wordPreProcess(sentence):\n    return ' '.join([p.stem(x.lower()) for x in re.split('\\W', sentence) if len(x) >= 1])\n# return ' '.join([p.stem(L.lemmatize(spell(x.lower()))) for x in re.split('\\W', sentence) if len(x) > 1])\n\n\ndef getTextFeatures(T, Col, max_features=10000, verbose=True):\n    if verbose:\n        print('processing: ', Col)\n    vectorizer = CountVectorizer(stop_words=None,\n                                 preprocessor=wordPreProcess,\n                                 max_features=max_features,\n                                 binary=True,\n                                 ngram_range=(1,2))\n#     vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'),\n#                                  preprocessor=wordPreProcess,\n#                                  max_features=max_features)\n    X = vectorizer.fit_transform(T[Col])\n    return X, vectorizer.get_feature_names()\n\nn_es1, n_es2, n_prs, n_rd, n_pt = 3000, 8000, 2000, 3000, 1000\nX_es1, feat_es1 = getTextFeatures(T, 'project_essay_1', max_features=n_es1)\nX_es2, feat_es2 = getTextFeatures(T, 'project_essay_2', max_features=n_es2)\nX_prs, feat_prs = getTextFeatures(T, 'project_resource_summary', max_features=n_prs)\nX_rd, feat_rd = getTextFeatures(T, 'resource_description', max_features=n_rd)\nX_pt, feat_pt = getTextFeatures(T, 'project_title', max_features=n_pt)\n\nX_txt = hstack((X_es1, X_es2, X_prs, X_rd, X_pt))\ndel X_es1, X_es2, X_prs, X_rd, X_pt\n\n# \n# from sklearn.decomposition import TruncatedSVD\n# svd = TruncatedSVD(1000)\n# X_txt = svd.fit_transform(X_txt)")


# Finally, let's make up the train and test matrices:
# 
# we should normalize the values if we want to use neural networks. Since my sparse features are 0s and 1s, I only apply it to numerical values.

# In[16]:


from sklearn.preprocessing import StandardScaler
X = hstack((X_txt, X_cat, StandardScaler().fit_transform(T[numFeatures].fillna(0)))).tocsr()

Xtr = X[pl.find(T.tr==1), :]
Xts = X[pl.find(T.ts==1), :]
Ttr_tar = T[T.tr==1][target].values
Tts = T[T.ts==1][['id',target]]

Yts = []
del T
del X
gc.collect();


# **Training Models**
# 
# Here I train XGB, LGB and NN models and stack them. I have two levels of stacking. First level on the results of learners by LinearRegression. The second level is by simple average over the results taken by different validation sets. In this version of the kernel I do it only for one validation set because of the time limit, but the result will be of course better if we stack more models. My NN model is inspired by [huiqin's kernel](https://www.kaggle.com/qinhui1999/deep-learning-is-all-you-need-lb-0-80x). Concisely, the structure of my network is like this:
# 
# ![NN structure](https://i.imgur.com/bkvRySn.jpg)

# In[25]:


from keras.layers import Input, Dense, Flatten, concatenate, Dropout, Embedding, SpatialDropout1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def breakInput(X1):
    X2 = []
    i = 0
    for n in [n_es1, n_es2, n_prs, n_rd, n_pt, X_cat.shape[1], len(numFeatures)]:
        X2.append(X1[:,i:i+n])
        i += n
    return X2

def getModel(HLs, Drop=0.25, OP=optimizers.Adam()):
    temp = []
    inputs_txt = []
    for n in [n_es1, n_es2, n_prs, n_rd, n_pt]:
        input_txt = Input((n, ))
        X_feat = Dropout(Drop)(input_txt)
        X_feat = Dense(int(n/100), activation="linear")(X_feat)
        X_feat = Dropout(Drop)(X_feat)
        temp.append(X_feat)
        inputs_txt.append(input_txt)

    x_1 = concatenate(temp)
    x_1 = Dense(20, activation="relu")(x_1)
    x_1 = Dropout(Drop)(x_1)

    input_cat = Input((X_cat.shape[1], ))
#     x_2 = Dropout(Drop)(input_cat)
    x_2 = Embedding(2, 10, input_length=X_cat.shape[1])(input_cat)
    x_2 = SpatialDropout1D(Drop)(x_2)
    x_2 = Flatten()(x_2)

    input_num = Input((len(numFeatures), ))
    x_3 = Dropout(Drop)(input_num)
    
    x = concatenate([x_1, x_2, x_3])

    for HL in HLs:
        x = Dense(HL, activation="relu")(x)
        x = Dropout(Drop)(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs_txt+[input_cat, input_num], outputs=output)
    model.compile(
            optimizer=OP,
            loss='binary_crossentropy',
            metrics=['binary_accuracy'])
    return model

def trainNN(X_train, X_val, Tar_train, Tar_val, HL=[50], Drop=0.5, OP=optimizers.Adam()):
    file_path='NN.h5'
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=6)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.5,
                                   patience=2,
                                   verbose=1,
                                   epsilon=3e-4,
                                   mode='min')

    model = getModel(HL, Drop, OP)
    model.fit(breakInput(X_train), Tar_train, validation_data=(breakInput(X_val), Tar_val),
                        verbose=2, epochs=50, batch_size=1000, callbacks=[early, lr_reduced, checkpoint])
    model.load_weights(file_path)
    return model

params_xgb = {
        'eta': 0.05,
        'max_depth': 4,
        'subsample': 0.85,
        'colsample_bytree': 0.25,
        'min_child_weight': 3,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 0,
        'silent': 1,
    }
params_lgb = {
        'boosting_type': 'dart',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 10,
        'learning_rate': 0.05,
        'feature_fraction': 0.25,
        'bagging_fraction': 0.85,
        'seed': 0,
        'verbose': 0,
    }
nCV = 1 # should be ideally larger
for i in range(nCV):
    gc.collect()
    X_train, X_val, Tar_train, Tar_val = train_test_split(Xtr, Ttr_tar, test_size=0.15, random_state=i, stratify=Ttr_tar)
    # XGB
    dtrain = xgb.DMatrix(X_train, label=Tar_train)
    dval   = xgb.DMatrix(X_val, label=Tar_val)
    watchlist = [(dtrain, 'train'), (dval, 'valid')]
    model = xgb.train(params_xgb, dtrain, 5000,  watchlist, maximize=True, verbose_eval=200, early_stopping_rounds=200)
    Yvl1 = model.predict(dval)
    Yts1 = model.predict(xgb.DMatrix(Xts))
    # LGB
    dtrain = lgb.Dataset(X_train, Tar_train)
    dval   = lgb.Dataset(X_val, Tar_val)
    model = lgb.train(params_lgb, dtrain, num_boost_round=10000, valid_sets=[dtrain, dval], early_stopping_rounds=200, verbose_eval=200)
    Yvl2 = model.predict(X_val)
    Yts2 = model.predict(Xts)
    # NN
    model = trainNN(X_train, X_val, Tar_train, Tar_val, HL=[50], Drop=0.5, OP=optimizers.Adam())
    Yvl3 = model.predict(breakInput(X_val)).squeeze()
    Yts3 = model.predict(breakInput(Xts)).squeeze()
    # stack
    M = LinearRegression()
    M.fit(pl.array([Yvl1, Yvl2, Yvl3]).T, Tar_val)
    Yts.append(M.predict(pl.array([Yts1, Yts2, Yts3]).T))


# **Output for Test Set**
# 
# At last, we make the stack of test set outputs by simple averaging, maybe rank average or median work better, I didn't try.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
Tts[target] = MinMaxScaler().fit_transform(pl.array(Yts).mean(0).reshape(-1,1))
Tts[['id', target]].to_csv('text_cat_num_xgb_lgb_NN.csv', index=False)


# **Further Possible Improvements**
# 
# * One obvious way to improve it is to play with the decision tree parameters, stacking more models, and perhaps stacking different kinds of learners
# * Fluency and articulation of the texts can be an important factor if we could somehow measure it
# * Checking for existence of special keywords that might attract or repel the reader -- if they are not already captured by the extracted n-grams
# * Checking and correcting for the spell of the words, before stemming them can help. Also, maybe the existence of spell or grammatical errors influences the decision
# * Checking for concurrency of the texts or required resources to special events that have occurred at that time might be useful. Maybe because of some events, some proposals are being accepted more easily, due to public awareness or hotness of some topics
# 
# Thanks for staying so far! Hope it helps.
# 
# Let me know if you have any comments, or suggestions for improvement, or if you think I can do some parts more efficiently.
