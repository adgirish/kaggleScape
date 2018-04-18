
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import regex as re
import regex


# In[ ]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')


# In[ ]:


train_links = train["comment_text"].apply(lambda x: len(re.findall("(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?",str(x)))).values.reshape(len(train), 1)
test_links = test["comment_text"].apply(lambda x: len(re.findall("(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?",str(x)))).values.reshape(len(test), 1)


links_n = np.append(train_links, test_links)
linksmean = train_links.mean()
linksstd = test_links.std()

train_links_n = (train_links - linksmean) / linksstd
test_links_n = (test_links - linksmean) / linksstd


# In[ ]:


repl = {
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " frown ",
    ":(": " frown ",
    ":s": " frown ",
    ":-s": " frown ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

keys = [i for i in repl.keys()]

new_train_data = []
new_test_data = []
ltr = train["comment_text"].tolist()
lte = test["comment_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_train_data.append(xx)
for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_test_data.append(xx)
train["new_comment_text"] = new_train_data
test["new_comment_text"] = new_test_data

trate = train["new_comment_text"].tolist()
tete = test["new_comment_text"].tolist()
for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
train["comment_text"] = trate
test["comment_text"] = tete
del trate, tete
train.drop(["new_comment_text"], axis=1, inplace=True)
test.drop(["new_comment_text"], axis=1, inplace=True)


# In[ ]:


# repl = {
#     "yay!": " good ",
#     "yay": " good ",
#     "yaay": " good ",
#     "yaaay": " good ",
#     "yaaaay": " good ",
#     "yaaaaay": " good ",
#     ":/": " bad ",
#     ":&gt;": " sad ",
#     ":')": " sad ",
#     ":-(": " frown ",
#     ":(": " frown ",
#     ":s": " frown ",
#     ":-s": " frown ",
#     "&lt;3": " heart ",
#     ":d": " smile ",
#     ":p": " smile ",
#     ":dd": " smile ",
#     "8)": " smile ",
#     ":-)": " smile ",
#     ":)": " smile ",
#     ";)": " smile ",
#     "(-:": " smile ",
#     "(:": " smile ",
#     ":/": " worry ",
#     ":&gt;": " angry ",
#     ":')": " sad ",
#     ":-(": " sad ",
#     ":(": " sad ",
#     ":s": " sad ",
#     ":-s": " sad ",
#     r"\br\b": "are",
#     r"\bu\b": "you",
#     r"\bhaha\b": "ha",
#     r"\bhahaha\b": "ha",
#     r"\bdon't\b": "do not",
#     r"\bdoesn't\b": "does not",
#     r"\bdidn't\b": "did not",
#     r"\bhasn't\b": "has not",
#     r"\bhaven't\b": "have not",
#     r"\bhadn't\b": "had not",
#     r"\bwon't\b": "will not",
#     r"\bwouldn't\b": "would not",
#     r"\bcan't\b": "can not",
#     r"\bcannot\b": "can not",
#     r"\bi'm\b": "i am",
#     "m": "am",
#     "r": "are",
#     "u": "you",
#     "haha": "ha",
#     "hahaha": "ha",
#     "don't": "do not",
#     "doesn't": "does not",
#     "didn't": "did not",
#     "hasn't": "has not",
#     "haven't": "have not",
#     "hadn't": "had not",
#     "won't": "will not",
#     "wouldn't": "would not",
#     "can't": "can not",
#     "cannot": "can not",
#     "i'm": "i am",
#     "m": "am",
#     "i'll" : "i will",
#     "its" : "it is",
#     "it's" : "it is",
#     "'s" : " is",
#     "that's" : "that is",
#     "weren't" : "were not",
# }

# keys = [i for i in repl.keys()]

# new_train_data = []
# new_test_data = []
# ltr = train["comment_text"].tolist()
# lte = test["comment_text"].tolist()
# for i in ltr:
#     arr = str(i).split()
#     xx = ""
#     for j in arr:
#         j = str(j).lower()
#         if j[:4] == 'http' or j[:3] == 'www':
#             continue
#         if j in keys:
#             # print("inn")
#             j = repl[j]
#         xx += j + " "
#     new_train_data.append(xx)
# for i in lte:
#     arr = str(i).split()
#     xx = ""
#     for j in arr:
#         j = str(j).lower()
#         if j[:4] == 'http' or j[:3] == 'www':
#             continue
#         if j in keys:
#             # print("inn")
#             j = repl[j]
#         xx += j + " "
#     new_test_data.append(xx)
# train["new_comment_text"] = new_train_data
# test["new_comment_text"] = new_test_data

# trate = train["new_comment_text"].tolist()
# tete = test["new_comment_text"].tolist()
# for i, c in enumerate(trate):
#     trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
# for i, c in enumerate(tete):
#     tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
# train["comment_text"] = trate
# test["comment_text"] = tete
# del trate, tete
# train.drop(["new_comment_text"], axis=1, inplace=True)
# test.drop(["new_comment_text"], axis=1, inplace=True)

# train_text = train['comment_text']
# test_text = test['comment_text']


# In[ ]:


train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])


# In[ ]:


import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# In[ ]:


# train_word_n = train['comment_text'].apply(lambda x: len(x.split(' '))).values.reshape(len(train), 1)
# test_word_n = test['comment_text'].apply(lambda x: len(x.split(' '))).values.reshape(len(test), 1)

# word_n = np.append(train_word_n, test_word_n)
# wnmean = word_n.mean()
# wnstd = word_n.std()

# train_word_nn = (train_word_n - wnmean) / wnstd
# test_word_nn = (test_word_n - wnmean) / wnstd


# In[ ]:


cont_patterns = [
        (b'US', b'United States'),
        (b'IT', b'Information Technology'),
        (b'(W|w)on\'t', b'will not'),
        (b'(C|c)an\'t', b'can not'),
        (b'(I|i)\'m', b'i am'),
        (b'(A|a)in\'t', b'is not'),
        (b'(\w+)\'ll', b'\g<1> will'),
        (b'(\w+)n\'t', b'\g<1> not'),
        (b'(\w+)\'ve', b'\g<1> have'),
        (b'(\w+)\'s', b'\g<1> is'),
        (b'(\w+)\'re', b'\g<1> are'),
        (b'(\w+)\'d', b'\g<1> would'),
    ]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]


def prepare_for_char_n_gram(text):
    """ Simple text clean up process"""
    # 1. Go to lower case (only good for english)
    # Go to bytes_strings as I had issues removing all \n in r""
    clean = bytes(text.lower(), encoding="utf-8")
    # 2. Drop \n and  \t
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\b", b" ")
    clean = clean.replace(b"\r", b" ")
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    # 4. Drop puntuation
    # I could have used regex package with regex.sub(b"\p{P}", " ")
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
    # 5. Drop numbers - as a scientist I don't think numbers are toxic ;-)
    clean = re.sub(b"\d+", b" ", clean)
    # 6. Remove extra spaces - At the end of previous operations we multiplied space accurences
    clean = re.sub(b'\s+', b' ', clean)
    # Remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)
    # 7. Now replace words by words surrounded by # signs
    # e.g. my name is bond would become #my# #name# #is# #bond#
    # clean = re.sub(b"([a-z]+)", b"#\g<1>#", clean)
    clean = re.sub(b" ", b"# #", clean)  # Replace space
    clean = b"#" + clean + b"#"  # add leading and trailing #

    return str(clean, 'utf-8')

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))


# In[ ]:


def get_indicators_and_clean_comments(df):
    """
    Check all sorts of content as it may help find toxic comment
    Though I'm not sure all of them improve scores
    """
    # Count number of \n
#     df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
    # Get length in words and characters
    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))
    # TODO chars per row
    # Check number of upper case, if you're angry you may write in upper case
    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    # Number of F words - f..k contains folk, fork,
    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
    # Number of S word
    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    # Number of D words
    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    # Number of occurence of You, insulting someone usually needs someone called : you
    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    # Just to check you really refered to my mother ;-)
    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    # Just checking for toxic 19th century vocabulary
    df["nb_ng"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    # Some Sentences start with a <:> so it may help
    df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))
    # Check for time stamp
    df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
    # Check for dates 18:44, 8 December 2010
    df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    # Check for date short 8 December 2010
    df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
    # Check for http links
#     df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    # check for mail
    df["has_mail"] = df["comment_text"].apply(
        lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
    )
    # Looking for words surrounded by == word == or """" word """"
    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))

    # Now clean comments
    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))

    # Get the new length in words and characters
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))
    # Number of different characters used in a comment
    # Using the f word only will reduce the number of letters required in the comment
    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(
        lambda x: 1 + min(99, len(x)))


# In[ ]:


for df in [train, test]:
   get_indicators_and_clean_comments(df)


# In[ ]:


num_features = [f_ for f_ in train.columns
                if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars", 'has_ip_address'] + class_names]

# TODO: normalize
for f in num_features:
    all_cut = pd.cut(pd.concat([train[f], test[f]], axis=0), bins=20, labels=False, retbins=False)
    train[f] = all_cut.values[:train.shape[0]]
    test[f] = all_cut.values[train.shape[0]:]

train_num_features = train[num_features].values
test_num_features = test[num_features].values


# In[ ]:


train_text = train['clean_comment'].fillna("")
test_text = test['clean_comment'].fillna("")
all_text = pd.concat([train_text, test_text])


# In[ ]:


word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
        analyzer='word',
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=20000) # TODO: maybe more

# word_vectorizer = TfidfVectorizer(sublinear_tf=True,
#                                   strip_accents='unicode',
#                                   analyzer='word',
#                                   token_pattern=r'\w{1,}',
#                                   ngram_range=(1,2),
#                                   max_features=30000)

# word_vectorizer = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
#                min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
#                smooth_idf=1, sublinear_tf=1 )


# In[ ]:


word_vectorizer.fit(all_text)


# In[ ]:


train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


# In[ ]:


def char_analyzer(text):
    """
    This is used to split strings in small lots
    I saw this in an article (I can't find the link anymore)
    so <talk> and <talking> would have <Tal> <alk> in common
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]


# In[ ]:


char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        tokenizer=char_analyzer,
        analyzer='word',
        ngram_range=(1, 3),
        max_df=0.9,
        max_features=60000) #50k
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)


# In[ ]:


train_features = hstack([train_word_features, train_links_n, train_char_features, train_num_features]).tocsr()
test_features = hstack([test_word_features, test_links_n, test_char_features, test_num_features]).tocsr()


# In[ ]:


print(train_features.shape)
print(test_features.shape)


# In[ ]:


all_parameters = {
                  'C'             : [1.048113, 0.1930, 0.596362, 0.25595, 0.449843, 0.25595],
                  'tol'           : [0.1, 0.1, 0.046416, 0.0215443, 0.1, 0.01],
                  'solver'        : ['lbfgs', 'newton-cg', 'lbfgs', 'newton-cg', 'newton-cg', 'lbfgs'],
                  'fit_intercept' : [True, True, True, True, True, True],
                  'penalty'       : ['l2', 'l2', 'l2', 'l2', 'l2', 'l2'],
                  'class_weight'  : [None, 'balanced', 'balanced', 'balanced', 'balanced', 'balanced'],
                 }


# In[ ]:


# scores= []

# for j, class_name in enumerate(class_names):
#     classifier = LogisticRegression(
#         C=all_parameters['C'][j],
#         max_iter=200,
#         tol=all_parameters['tol'][j],
#         solver=all_parameters['solver'][j],
#         fit_intercept=all_parameters['fit_intercept'][j],
#         penalty=all_parameters['penalty'][j],
#         dual=False,
#         class_weight=all_parameters['class_weight'][j],
#         verbose=0)

#     train_target = train[class_name]

#     cv_score = np.mean(cross_val_score(classifier, train_features, train_target, scoring='roc_auc'))
    
#     print('CV score for class {} is {}'.format(class_name, cv_score))
#     scores.append(cv_score)

# print('Total score is {}'.format(np.mean(scores)))


# In[ ]:


submission = pd.DataFrame.from_dict({'id': test['id']})


# In[ ]:


for j, class_name in enumerate(class_names):
    classifier = LogisticRegression(
        C=all_parameters['C'][j],
        max_iter=200,
        tol=all_parameters['tol'][j],
        solver=all_parameters['solver'][j],
        fit_intercept=all_parameters['fit_intercept'][j],
        penalty=all_parameters['penalty'][j],
        dual=False,
        class_weight=all_parameters['class_weight'][j],
        verbose=0)

    train_target = train[class_name]
    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]
    print(class_name)


# In[ ]:


submission.to_csv('submission.csv', index=False)

