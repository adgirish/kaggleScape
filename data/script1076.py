
# coding: utf-8

# Hey dear Kagglers, I'm excited to share with you my very first notebook and I'll be very happy to get some advice on the many things I can improve in my investigation into the Quora dataset. Here goes...
# 
# I decided to take a hybrid approach (including naive as well as tf-idf features).
# 
# We start by first deriving the naive features:
# 
#  - Similarity: basic similarity ratio between the two question strings
#  - Pruned similarity: similarity of the two question strings excluding the stopwords

# In[ ]:


import pandas as pd
pd.set_option('max_colwidth', 250) #so that the full column of tagged sentences can be displayed
import numpy as np
import nltk
from nltk.corpus import stopwords
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning) #to stop the annoying deprecation warnings from sklearn

#Some simple functions
def remove_stopwords(tokenized_sent):
    unique_stopwords = set(stopwords.words('english'))
    return [word for word in tokenized_sent if word.lower() not in unique_stopwords]

def concatenate_tokens(token_list):
    return str(' '.join(token_list))

def find_similarity(sent1, sent2):
	return SequenceMatcher(lambda x: x in (' ', '?', '.', '""', '!'), sent1, sent2).ratio()

def return_common_tokens(sent1, sent2):
    return " ".join([word.lower() for word in sent1 if word in sent2])

def convert_tokens_lower(tokens):
    return [token.lower() for token in tokens]

#Reading the train file
train_sample = pd.read_csv('../input/train.csv', encoding = 'utf-8', index_col = 0, header = 0, iterator = True).get_chunk(100000)

transformed_sentences_train = pd.DataFrame(index = train_sample.index)
naive_similarity = pd.DataFrame()
temp_features = pd.DataFrame()
dictionary = pd.DataFrame()

#Deriving the naive features
for i in (1, 2):
        transformed_sentences_train['question%s_tokens' % i] = train_sample['question%s' % i].apply(nltk.word_tokenize)
        transformed_sentences_train['question%s_lowercase_tokens' % i] = transformed_sentences_train['question%s_tokens' % i].apply(convert_tokens_lower)
        transformed_sentences_train['question%s_lowercase' % i] = transformed_sentences_train['question%s_lowercase_tokens' % i].apply(concatenate_tokens)
        transformed_sentences_train['question%s_words' % i] = transformed_sentences_train['question%s_tokens' % i].apply(remove_stopwords)
        transformed_sentences_train['question%s_pruned' % i] = transformed_sentences_train['question%s_words' % i].apply(concatenate_tokens)
naive_similarity['similarity'] = np.vectorize(find_similarity)(train_sample['question1'], train_sample['question2'])
naive_similarity['pruned_similarity'] = np.vectorize(find_similarity)(transformed_sentences_train['question1_pruned'], transformed_sentences_train['question2_pruned'])
temp_features['common_tokens'] = np.vectorize(return_common_tokens)(transformed_sentences_train['question1_tokens'], transformed_sentences_train['question2_tokens'])

print (naive_similarity[:20])


# This is supposed to catch the most elementary non-duplicates (where the questions are obviously different), e.g. question id 3:
# 
#  - Why am I mentally very lonely? How can I solve it?
#  - Find the remainder when [math]23^{24}[/math] is divided by 24,23?
# 
# As we can see from the output, the similarity there is 14% and the pruned similarity is 11%
# 
# Next, we can enrich the feature set by adding the term frequency inverse dictionary frequency measure (tf-idf). The term frequency is the count of a term in a specific question, the inverse document frequency is the log of the total number of questions divided by the number of questions containing the term. Here is the derivation using scikit-learn's library:

# In[ ]:


dictionary = pd.DataFrame()

#Deriving the TF-IDF
dictionary['concatenated_questions'] = transformed_sentences_train['question1_lowercase'] + transformed_sentences_train['question2_lowercase']

vectorizer = CountVectorizer()
terms_matrix = vectorizer.fit_transform(dictionary['concatenated_questions'])
terms_matrix_1 = vectorizer.transform(transformed_sentences_train['question1_lowercase'])
terms_matrix_2 = vectorizer.transform(transformed_sentences_train['question2_lowercase'])
common_terms_matrx = vectorizer.transform(temp_features['common_tokens'])

transformer = TfidfTransformer(smooth_idf = False)
weights_matrix = transformer.fit_transform(terms_matrix)
weights_matrix_1 = transformer.transform(terms_matrix_1)
weights_matrix_2 = transformer.transform(terms_matrix_2)
common_weights_matrix = transformer.transform(common_terms_matrx)

#Converting the sparse matrices into dataframes
transformed_matrix_1 = weights_matrix_1.tocoo(copy = False)
transformed_matrix_2 = weights_matrix_2.tocoo(copy = False)
transformed_common_weights_matrix = common_weights_matrix.tocoo(copy = False)

weights_dataframe_1 = pd.DataFrame({'index': transformed_matrix_1.row, 'term_id': transformed_matrix_1.col, 'weight_q1': transformed_matrix_1.data})[['index', 'term_id', 'weight_q1']].sort_values(['index', 'term_id']).reset_index(drop = True)
weights_dataframe_2 = pd.DataFrame({'index': transformed_matrix_2.row, 'term_id': transformed_matrix_2.col, 'weight_q2': transformed_matrix_2.data})[['index', 'term_id', 'weight_q2']].sort_values(['index', 'term_id']).reset_index(drop = True)
weights_dataframe_3 = pd.DataFrame({'index': transformed_common_weights_matrix.row, 'term_id': transformed_common_weights_matrix.col, 'common_weight': transformed_common_weights_matrix.data})[['index', 'term_id', 'common_weight']].sort_values(['index', 'term_id']).reset_index(drop = True)

#Summing the weights of each token in each question to get the summed weight of the question
sum_weights_1, sum_weights_2, sum_weights_3 = weights_dataframe_1.groupby('index').sum(), weights_dataframe_2.groupby('index').sum(), weights_dataframe_3.groupby('index').sum()

weights = sum_weights_1.join(sum_weights_2, how = 'outer', lsuffix = '_q1', rsuffix = '_q2').join(sum_weights_3, how = 'outer', lsuffix = '_cw', rsuffix = '_cw')
weights = weights.fillna(0)
del weights['term_id_q1'], weights['term_id_q2'], weights['term_id']

print (weights[:20])


# This feature is designed to account for questions that are quite similar as strings but are different in meaning. The difference usually comes from a small amount of very significant terms. Example pair id 0:
# 
#  - What is the step by step guide to invest in share market in india?
#  - What is the step by step guide to invest in share market?
# 
# As is obvious from the data, these two questions have a 91% similarity and 90% pruned similarity. However, the one word that significantly differentiates them is 'india.' The way tf-idf is supposed to address this issue is by applying a larger weight to the 'india' term than to the others. This changes significantly the weight sum of the first and second questions (as is evident from the data above).
# 
# In addition, we also derive the 'common weight' of the two questions, i.e. the sum of the weight of all the tokens that the two questions share. As we can see this weight is very similar to the weight of the second question which also agrees with our observations.
# 
# Next, we'll join the features we derived, shuffle and scale them:

# In[ ]:


X = naive_similarity.join(weights, how = 'inner')

#Creating a random train-test split
y = train_sample['is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)

#Scaling the features
sc = StandardScaler()
for frame in (X_train, X_test):
    sc.fit(frame)
    frame = pd.DataFrame(sc.transform(frame), index = frame.index, columns = frame.columns)

print (X_train[:20])


# We train our algorithm (gradient boosting classifier) and print the logarithmic loss:

# In[ ]:


#Training the algorithm and making a prediction
gbc = GradientBoostingClassifier(n_estimators = 8000, learning_rate = 0.3, max_depth = 3)
gbc.fit(X_train, y_train.values.ravel())
prediction = pd.DataFrame(gbc.predict(X_test), columns = ['is_duplicate'], index = X_test.index)

#Inspecting our mistakes
prediction_actual = prediction.join(y_test, how = 'inner', lsuffix = '_predicted', rsuffix = '_actual').join(train_sample[['question1', 'question2']], how = 'inner').join(X_test, how = 'inner')

print ('The log loss is %s' % log_loss(y_test, prediction))


# As we can see, the log loss is abysmal for the 30 question pairs in the sample, but it actually goes down substantially if the algorithm is trained over most of the training data.
# 
# Finally, we evaluate our mistakes:

# In[ ]:


print (prediction_actual[prediction_actual['is_duplicate_predicted'] != prediction_actual['is_duplicate_actual']][:10])


# As we can see, this approach needs to be supplemented by other metrics. The types of errors we are likely to encounter are:
# 
#  - Cases where the weights of two contextually different expressions are similar (e.g. pair 28). In this case 'ask for' and 'make' may have very similar weights due to similar counts of the term throughout the corpus, but have a fundamentally different meaning.
#  - Algorithmic errors - where the features indicate difference to an observer but not to the algorithm (e.g. pair 24 where the similarity is 48% and the weight ratio is 75%). This could potentially be improved by tweaking the training parameters, adding more training data and executing more epochs.
# 
# In addition, our data derivation has several shortcomings. Namely: we have done no canonization of the terms in the corpus. This means that the following terms will be considered different (and have different counts and weights according to the tf-idf):
# 
#  - 2016-12-01 and 1st of December 2016
#  - Youtube and YouTube
#  - india and India
# 
# This problem can be solved through a similarity matching and some regular expressions.
# 
# Another issue we haven't addressed is the semantic closeness of terms in the question pairs for cases like:
# 
#  - Holland and The Netherlands
#  - Holland and France (both may have equal frequency in the corpus and equal weights but have different meaning)
# 
# This problem can be resolved through vectorization of the terms and taking cosine of their values.
# 
# Unfortunately those tasks are beyond the allocated time or hardware of my current participation (30 hours and Acer Revo One, respectively), but had time been abundantly available, I would work on the following additional features:
# 
#  - Regular expression parser to canonize the training and test corpus
#  - Cosine of the terms of each question pair
#  - N-gram derivation and comparison
# 
# I'm eager to hear your constructive criticism and suggestions for improvement!
