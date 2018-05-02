
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd

import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import re
import string

pd.options.mode.chained_assignment = None

URL = '../input/Tweets.csv'
def load_data(url=URL):
	return pd.read_csv(url)

df = load_data()
columns_to_keep = [u'airline_sentiment',u'retweet_count', u'airline', u'text']

df = df[columns_to_keep]

df.loc[:,'sentiment'] = df.airline_sentiment.map({'negative':0,'neutral':2,'positive':4})
df = df.drop(['airline_sentiment'], axis=1)

df = df[df['retweet_count'] <= 2]


# In[ ]:


def clean_tweet(s):
	'''
	:s : string; a tweet

	:return : list; words that don't contain url, @somebody, and in utf-8 and lower case
	'''
	s = re.sub(clean_tweet.pattern_airline, '', s, 1)
	remove_punctuation_map = dict.fromkeys(map(ord, string.punctuation))
	s = s.translate(remove_punctuation_map)
	sents = sent_tokenize(s)

	words = [word_tokenize(s) for s in sents]
	words = [e for sent in words for e in sent]
	return [clean_tweet.stemmer.stem(e.lower()) for e in words]

clean_tweet.stemmer = PorterStemmer()
clean_tweet.pattern_airline = re.compile(r'@\w+')
df.loc[:,'text'] = df.loc[:,'text'].map(clean_tweet)


# In[ ]:


def get_stop_words(s, n):
	'''
	:s : pd.Series; each element as a list of words from tokenization
	:n : int; n most frequent words are judged as stop words 

	:return : list; a list of stop words
	'''
	from collections import Counter
	l = get_corpus(s)
	l = [x for x in Counter(l).most_common(n)]
	return l

def get_corpus(s):
	'''
	:s : pd.Series; each element as a list of words from tokenization

	:return : list; corpus from s
	'''
	l = []
	s.map(lambda x: l.extend(x))
	return l

freqwords = get_stop_words(df['text'],n=100)

freq = [s[1] for s in freqwords]

plt.title('frequency of top 100 most frequent words')
plt.plot(freq)
plt.xlim([-1,100])
plt.ylim([0,1.1*max(freq)])
plt.ylabel('frequency')
plt.show()


# In[ ]:


print(freqwords[:18])


# In[ ]:


stopwords = [w[0] for w in freqwords[:18]]
remove_stop_words = lambda x: [e for e in x if e not in stopwords]
df.loc[:,'text'] = df.loc[:,'text'].map(remove_stop_words)


# In[ ]:


import numpy as np

airlines = df['airline'].unique()
dfs = [df[(df['airline'] == a)] for a in airlines]

dfs = [df for df in dfs if len(df) >= 10]
dfs = [df.reindex(np.random.permutation(df.index)) for df in dfs]
dfs = [(df.text, df.sentiment) for df in dfs]


# In[ ]:


import random
class lm(object):
	"""
	statistical language model based on MLE method. Both jelinek-mercer and dirichlet smoothing methods are implemented
	"""
	def __init__(self, a=0.1, smooth_method='jelinek_mercer'):
		super(lm, self).__init__()
		'''
		:a : float; discount parameter; should be tuned via cross validation
		:smooth_method: function; method selected to discount the probabilities	
		'''
		self.a = a
		smooth_method = getattr(self, smooth_method)
		self.smooth_method = smooth_method

		#self.counter = 0

	def df_to_ct(self, df):
		from collections import Counter	
		l = []
		df.map(lambda x: l.extend(x))
		return pd.Series(dict(Counter(l)))

	def ct_to_prob(self, d):
		total_occur = d.sum()
		return d/float(total_occur)

	def df_to_prob(self, df):
		'''
		df: list of lists; each containing a document of words, like [[a],[b,c],...]
		out: pd.Series; the probabilities of each word, like ({a:0.3,b:0.3,...})
		'''
		return self.ct_to_prob(self.df_to_ct(df))	


	def fit(self, X, Y):
		'''
		:X : pd.Series; features; features are actually a list of words, standing for the document.
		:Y : pd.Series; labels

		:return : pd.DataFrame; language model
		'''
		if len(Y) != 0  and len(X) != 0:
			from math import log
			cats = Y.unique()	
			p_ref = self.df_to_prob(X)
			model = pd.DataFrame()
			model['unseen'] = (p_ref*self.a).map(log)
			for c in cats:
				idx = Y[Y == c].index
				ct = self.df_to_ct(X.loc[idx])
				p_ml = self.ct_to_prob(ct)
				model[c] = self.smooth_method(ct, p_ml,p_ref)
				model[c].fillna(model['unseen'],inplace=True)
			model.drop(['unseen'],axis=1,inplace=True)
			self.model = model
		else: print('input is empty')

	def jelinek_mercer(self, ct, p_ml,p_ref,a=0.1):
		from math import log
		log_p_s = (p_ml*(1-a)+p_ref.loc[p_ml.index]*a).map(log)
		return log_p_s

	def dirichlet(self, ct, p_ml,p_ref,a=0.1):
		from math import log
		d = len(p_ml)
		u = a / (1+a)*d
		log_p_s = ((ct+u*p_ref.loc[ct.index])/(d+u)).map(log)
		return log_p_s

	
	def predict_item(self, l, N):
		model = self.model
		# self.counter += 1
		# if self.counter % 200 == 0: print self.counter/float(N)
		in_list = [e for e in l if e in model.index]
		if not in_list: 
			return model.columns[random.randint(0,len(model.columns)-1)]
		selected_model =  model.loc[in_list,:]
		s = selected_model.sum(axis=0)

		label = s.loc[s==s.max()].index[0]
		word = selected_model.loc[selected_model[label] == selected_model[label].max(),:].index[0]
		self.predwords[label].append(word)
		return label

	def predict(self, df):
		self.predwords = dict(zip(self.model.columns,[[] for _ in range((len(self.model.columns)))])) #tricky
		return df.map(lambda x: self.predict_item(x,len(df)))

	def get_params(self):
		return (self.a, self.smooth_method.__name__)
		
	def get_predictive_words(self, n=3):
		from collections import Counter
		total_words = {k:len(v) for k,v in self.predwords.items()}
		most_predictive_words = {k:Counter(v).most_common(n) for k, v in self.predwords.items()}
		most_predictive_words = {label:{w:v/float(length) for w, v in words} for label, length, words in zip(total_words.keys(), total_words.values(), most_predictive_words.values())}
		return most_predictive_words


# In[ ]:


def cross_validation(clf, X, Y, cv=5, avg=False):
	'''
	:clf : classifier with fit() and predict() method
	:X : pd.DataFrame; features
	:Y : pd.DataFrame(1 column) or pd.Series; labels
	:cv : int; cross validation folders

	:return : list of float; cross validation scores
	'''

	k = [int((len(X))/cv*j) for j in range(cv+1)]
	score = [0.0]*cv
	for i in range(cv):	
		train_x, train_y = pd.concat([X[:k[i]],X[k[i+1]:]]), pd.concat([Y[:k[i]],Y[k[i+1]:]])
		test_x, test_y = X[k[i]:k[i+1]], Y[k[i]:k[i+1]]

		clf.fit(X,Y)
		pred = clf.predict(test_x)

		score[i] = (pred == test_y).sum()/float(len(test_y))
	if avg: return sum(score)/float(len(score))
	return score


models = [lm()]*len(dfs)
avg_score = [cross_validation(model, X, Y, avg=True, cv=2) for model, (X, Y) in zip(models, dfs)]
print(avg_score)


# In[ ]:


clf = lm()

axarr = [None]*len(dfs)

for i in range(len(dfs)):
	X, Y = dfs[i]
	pt = int(len(X)/2)
	clf.fit(X[:pt],Y[:pt])
	_ = clf.predict(X[pt+1:])
	predictive_words = clf.get_predictive_words(n=3)
	f, axarr[i] = plt.subplots()
	f.suptitle('words with top 3 predictive contributions',fontweight='bold')

	ax = axarr[i]
	ax.set_title(airlines[i])

	for senti, words in predictive_words.items():
		tot_frac = 0.0
		colors = ['red','blue','green']
		ct = 0
		ax.set_xticks([0,2,4])
		ax.set_xticklabels(('negative','netural','positive'))
		width = 1.2
		# print words
		for w, frac in words.items():
			y = tot_frac + frac / 2.0
			x = senti
			ax.text(x, y, w, color='white',fontweight='bold',ha='center')
			if tot_frac == 0.0: ax.bar(x - width/2.0, frac, color = colors[ct], alpha=0.7, width = width)
			else: ax.bar(x - width/2.0, frac, bottom = tot_frac, color = colors[ct], alpha=0.7, width = width)
			ct += 1
			tot_frac += frac
plt.show()

