
# coding: utf-8

# # Introduction
# 
# ![](https://i.pinimg.com/736x/7f/d8/04/7fd8042f4c23f6f1b8effc7c9de0ade9--witch-art-the-raven.jpg)
# *Sure, that's a raven, it must be Poe. But wait, is that man over there Dr. Frankenstein?*

# This notebook aims to illustrate the usage of a few well known classification algorithms in this very interesting NLP problem. And since a picture is worth a thousand words, I'll try to plot every message I want to convey.
# 
# Still, we can use an outline. There will be two parts, plus maybe a third one.
# 
# 1. **Bag-of-words appraoch**
# In this first simplistic attempt, we will only keep track of the word counts in each sentence. Still, this will set a benchmark that is not easy to beat with more sophisticated models.
# 
# 2. **Grammar**
# Sometimes less is more, so let's discard the actual words and only use their grammar meaning.
# 
# 3. **And neural networks?**
# Well, they should sometimes be better, but unfortunately it didn't work well here. The (keras) code is provided, but not run, because proably there is a bug somwhere...
# 
# Since we do many test runs, the 1h limit would be violated. Hence some runs are commented and the results reported from an off-line run. Feel free to uncomment them and run locally if you want to check!

# # 1. Bag-of-words
# ## 1.1. Visualization: word counts
# 
# There are already many excellent kernels that explain basic statistics of the text, I will try to add value instead of repeating them. For example, if you haven't already, you are encouraged to visit [this one](https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial) to see about basic statistics (e.g. authors are represented as 40%-31%-29%), stemming, and stopwords. I'll go ahead, plot first, explain then.

# In[50]:


# General useful packages - plotting, data.
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import holoviews as hv
hv.extension('bokeh', 'matplotlib')
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,  AnnotationBbox)
from tqdm import trange, tqdm
from matplotlib import cm
from scipy.sparse import hstack

# NLP imports
from nltk.stem.snowball import EnglishStemmer
from wordcloud import WordCloud
import nltk, re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# scikit-learn classification imports
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
from sklearn.ensemble import VotingClassifier, BaggingClassifier


# In[2]:


csv = pd.read_csv('../input/train.csv')

# Split into words
csv['split'] = csv.text.apply(nltk.word_tokenize)

# Save a mapping of all (unique) csv words to integers
unique_words = {word for sentence in csv.split.values for word in sentence}
dict_csv = {s: i for i,s in enumerate(unique_words)}

# Apply stemming (Snowball). We also exclude stopwords in this step,
# as well as punctuation
stopwords = nltk.corpus.stopwords.words('english') + ['']
stemmer = EnglishStemmer()
# Create dictionaries from words to stem and conversely
stemmer_dict = {u: stemmer.stem(u) for u in unique_words}
reverse_stemmer = {u: [] for u in stemmer_dict.values()}
for k in stemmer_dict:
    reverse_stemmer[stemmer_dict[k]].append(k)
csv['stemmed'] = csv['split'].apply(lambda x: [stemmer_dict[y] for y in x
                                               if re.sub('[^a-z]+','',y.lower()) not in stopwords])

# Count words overall and by author
word_count = {'ALL': pd.Series([y for x in csv.stemmed for y in x]).value_counts()}
authors = ['MWS', 'EAP', 'HPL']
authors_dict = {auth: i for i, auth in enumerate(authors)}
for auth in authors:
    word_count[auth] = pd.Series([y for x in csv.loc[csv.author==auth, 'stemmed']
                                  for y in x]).value_counts()
word_count = pd.DataFrame(word_count).fillna(0).astype(int).sort_values('ALL', ascending=False)[['ALL']+authors]

print('Count for the most common words (excl. stopwords)')

word_count.head(10).style.background_gradient(subset=authors,
                                              cmap=LinearSegmentedColormap.from_list('', ['ivory','yellow']))


# Colors are nice, but plots are nicer!

# In[3]:


plt.style.use('ggplot')
plt.rcParams['font.size'] = 16
plt.figure(figsize=(20,10))
bottom = np.zeros((20))
ind = np.arange(20)
df = word_count.head(20)
for auth in authors:
    # Stacked bar with actual numbers.
    # Uncomment the below for percentages instead.
    vals = df[auth]# / df['ALL']
    plt.bar(ind, vals, bottom=bottom, label = auth)
    bottom += vals

# If using percentages, replace the two "df['ALL']" by "np.ones(df['ALL'].shape)"
plt.plot(ind, df['ALL'] * word_count[authors[0]].sum() / word_count['ALL'].sum(), 'k--',
         label='Expected cutoffs for\nuninformative words')
plt.plot(ind, df['ALL'] * word_count[authors[:2]].values.sum() / word_count['ALL'].sum(), 'k--', label='')
plt.xticks(ind, df.index, rotation='vertical')
#plt.yticks(np.arange(0,1.1,0.2), ['{:.0%}'.format(x) for x in np.arange(0,1.1,0.2)])
plt.legend(fontsize=24)
plt.title('Top 20 word count split by author (dotted lines is the global average)', fontsize=24)
plt.xlim([-0.7,19.7])
plt.show()


# **Good** *old* Lovecraft certainly has a *thing* for the *thing*, and Poe loves writing *upon* the word *upon*.
# 
# **Bad** *time*, however, if *one* wants to distinguish: some words are distributed almost as the general population.
# 
# The most common words above allow us to capture trends that we may find in many sentences (because they are, well, common), but that's only part of the picture. If in a sentence we find
# 
# *"[...] for Mr. Kirwin hastened to say, "Immediately upon your being taken ill, [...]"*
# 
# and, that is, if we remember that Mr. Kirwin is a character in Frankenstein, we'll hopefully not mark this as written by Poe. For this reason, let's try to take a look at the most characteristic words for each author.

# ## 1.2. Visualization: characteristic words for an author

# In[4]:


def most_characteristic(auth, m=10, head=20):
    ''' Compute most charactersitic words for the author "auth"
        and returns them in a DataFrame of length "head". '''
    df = ((word_count[auth]+m) / (word_count[[a for a in authors if a != auth]].sum(
        axis=1) + 2*m)).sort_values(ascending=False).head(head).to_frame()
    df.columns=['Score']
    df['Words'] = pd.Series(df.index).apply(lambda i: reverse_stemmer[i]).values
    return df

wc = WordCloud(width=1000, height=1000)
f, axes = plt.subplots(3,3,figsize=(15,15))
for i in range(3):
    for j,m in enumerate([1, 10, 1000]):
        wc.generate_from_frequencies(most_characteristic(authors[i], m=m, head=50)['Score'].to_dict())
        axes[i,j].imshow(wc.recolor(colormap='Set3'))
        axes[i,j].set_title('Characteristic words\nfor {} m={}'.format(authors[i], m))
        axes[i,j].set_ylabel(authors[i])
        axes[i,j].set_xlabel('m={}'.format(m))
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])
plt.tight_layout()


# Maybe some explanation: **why three plots per author?**
# 
# We need a measure for what it means to be "characteristic". The one I've chosen is something like the following, for each word $w$ author:
# $$
# score_{MWS}(w) = \frac{m+n_{MWS}(w)}{2m+n_{HPL}(w)+n_{EAP}(w)},
# $$
# where $n_{auth}(\cdot)$ is the word number count per author.
# 
# **Why the $m$?** Well, you need some regularization, or each word appearing in one author but not the others will get infinite score. Not very distinguishable. The reason why I chose a regularization of this form is (very) loosely inspired by [this blog post](http://julesjacobs.github.io/2015/08/17/bayesian-scoring-of-ratings.html). It explains that $m$ should be the expected number of appearances; hence broadly speaking a given $m$ allows us to focus on words appearing that amount of time.
# 
# **So, what does it do?** Small $m$ gives little normalization, thus favouring words that essentially only appear in one author; highest scores are for proper names there (Perdita, Adrian, Raymond for MWS; Dupin for EAP; Gilman, Whateleys, Innsmouth for HPL). When $m$ is 1000, it matters little whether a given name appears 0 or 10 times for another author; thus we find again common words, because now it's the numerator that matters most.
# 
# Or, if you prefer some math, you can compute the limits and see that:
# $$
# \lim_{m\to 0} score_{MWS} = \frac{n_{MWS}(w)}{n_{HPL}(w)+n_{EAP}(w)}; \qquad
# score_{MWS} \approx \frac{1}{2} + \frac{1}{2m} \bigg(n_{MWS}(w) - \frac{1}{2}\big(n_{HPL}(w)+n_{EAP}(w)\big)\bigg) \text{ when } m \to \infty.
# $$
# Thus it is an interpolation between "geometric" $(m=0)$ and "linear" ($m \to \infty$) frequencies.
# 
# To see better the effect of $m$, let's use an interactive plot:

# In[5]:


def remove_duplicates(seq):
    ''' Nice little function to remove duplicates while preserving the order.
        Taken from https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-whilst-preserving-order '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

mc = {}
keys = np.exp(np.linspace(0, np.log(1000)))
indices = {auth: {} for auth in authors}
for auth in authors:
    mc[auth] = {}
    for m in keys:
        mc[auth][m] = most_characteristic(auth, m=m, head=10000)
    indices[auth] = remove_duplicates([mc[auth][m].index[0] for m in keys])
    i=0
    # We want to include only the top results, but since we remove
    # duplicates we don't know exactly how many there are.
    # The following code is very non-pythonic, but anyway it does
    # at most 20 loops, so...
    while len(indices[auth]) < 20:
        indices[auth] = remove_duplicates(indices[auth] + [mc[auth][m].index[i] for m in keys])[:20]
        i += 1


# In[6]:


get_ipython().run_cell_magic('opts', "Layout [fig_size=300 sublabel_format='' vspace=0.1]", "# If using Bokeh backend, for better result (incl. hovering text):\n# - remove fig_size=300 above\n# - uncomment the following\n# %%opts Bars [height=300 width=800 aspect=6 tools=['hover']]\n# But since we are forced to matplotlib...\n%%opts Bars [aspect=6]\nmost_characteristic('MWS').loc['perdita','Words'][0]\n\n# Interactive plots with HoloViews. http://holoviews.org/\ndef barplot(auth, col):\n    return hv.Bars([(mc[auth][m].loc[i, 'Words'][0],\n                     np.log(mc[auth][m].loc[i, 'Score'] - 0.5)+np.log(m))\n                    for i in indices[auth]]).opts(plot={'xrotation': 90, 'yaxis': None}, style={'color': col}\n                                                 ).relabel(group=auth)\n\nhmap = hv.HoloMap(kdims='m')\nfor m in keys:\n    hmap[m] = (barplot('MWS', 'red') + barplot('EAP', 'blue') + barplot('HPL', 'purple')).cols(1)\nhmap")


# Sliding the bar from left to right, we can see common words like "Life" for MWS, "say" (and derivates) for EAP and "thing" for HPL rise from negligible to most characteristic. *There's no perfect $m$ here, each gives valuable information.*
# 
# Remark that the score is tweaked to give nicer plots (logarithmic scale, further shifted to avoid too negative numbers), therefore the actual numbers are practically meaningless; the only important parameter is the relative value among words for a fixed $m$.

# ## 1.3. Implementing scikit-learn straightforward approach
# 
# Enough with the chit-chat, we've understood that there's power to distinguish, let's see if stock solutions work fine already. We'll start by a literally stock solution, i.e. by barely modifying [this great script from scikit-learn website](http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html). Main differences here are that:
# - To keep things short and fast, output has been reduced, and the slowest methods abandoned (I tried them offline, they didn't even score well).
# - We try different regularization parameters ($\alpha$ or $C$ depending on the model)
# - Some fantasy, and few actual, log losses have been computed, too. More on this below.

# In[7]:


scores = []
fantasy_log_losses = []
true_log_losses = []
model_descr = []
model_param = []

def benchmark(clf, additional='', param=''):
    ''' Function that trains clf on X_train, y_train, tests 
        it on y_train, and saves everything on the lists. '''
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    l = metrics.accuracy_score(y_test, pred)
    scores.append(l)
    
    # For multi class log-loss we "cheat" by giving max probability
    # equal to the accuracy, which in practice we can't know.
    # See 1.4. for justification of the formula.
    fantasy_log_losses.append(-np.mean(np.log([l if p==y else (1-l)/2 for p,y in zip(pred, y_test)])))

    model_descr.append(' '.join((str(clf).split('(')[0], str(additional))))
    
    try:
        proba = clf.predict_proba(X_test)
    except:
        true_log_losses.append(np.nan)
    else:
        true_log_losses.append(-np.mean(np.log(proba[np.arange(len(proba)), y_test])))
    
    model_param.append(param)
    
# 10 loops of splitting the input csv in train and test
this_range = trange(10)

for test_loop in this_range:
    test = csv.iloc[int(len(csv)*test_loop/10) : int(len(csv)*(test_loop+1)/10)]
    train = csv.loc[list(set(csv.index).difference(test.index))]

    # split a training set and a test set
    y_train = train.author.apply(lambda auth: authors_dict[auth]).values
    y_test = test.author.apply(lambda auth: authors_dict[auth]).values

    # built-in vectorizer, automatically removes stop words.
    # Let's just accept it for the moment.
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(train.text.values)
    X_test = vectorizer.transform(test.text.values)

    # mapping from integer feature name to original token string
    feature_names = np.array(vectorizer.get_feature_names())

    # Ridge classifier:
    this_range.set_postfix(working_on='RidgeClassifier')
    for alpha in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
        benchmark(RidgeClassifier(alpha=alpha, solver='sag'), param=alpha)
        
    # Perceptron skipped as it's a bit too rough
    
    # Passive-Aggressive
    this_range.set_postfix(working_on='Passive-Aggressive')
    for C in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
        benchmark(PassiveAggressiveClassifier(C=C, tol=1e-3), param=C)
    
    # k-neighbors and RandomForests skipped as they are very slow.
    # They also performed quite badly!
    
    # LinearSVC (l2 loss and dual=True)
    this_range.set_postfix(working_on='SVC-L2')
    for C in [0.01, 0.1, 0.3, 1, 3, 10, 30]:
        benchmark(LinearSVC(C=C), additional='L2 penalty', param=C)
        
    # Now l1
    this_range.set_postfix(working_on='SVC-L1')
    for C in [0.01, 0.1, 0.3, 1, 3, 10, 30]:
        benchmark(LinearSVC(C=C, penalty='l1', dual=False), additional='L1 penalty', param=C)
    
    # SGDClassifier skipped as by default it does the same as LinearSVC but non-deterministic...
    
    # But let's try Elastic Net, because that's the same loss but different penalty
    this_range.set_postfix(workin_on='Elastic Net')
    for alpha in [1e-7, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
        benchmark(SGDClassifier(alpha=alpha, max_iter=100, penalty='elasticnet', tol=1e-3),
                  additional='Elastic Net', param=alpha)
    
    # Naive Bayes
    this_range.set_postfix(workin_on='Naive Bayes')
    for alpha in [1e-3, 3e-3, 0.01,0.03,0.1,0.3,1]:
        benchmark(MultinomialNB(alpha=alpha), param=alpha)
        benchmark(BernoulliNB(alpha=alpha), param=alpha)
        
    # L2 SVC with L1 feature selection
    this_range.set_postfix(workin_on='L2 SVC with L1 feature selection')
    for C in [0.01, 0.1, 0.3, 1, 3, 10]:
        benchmark(Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(C=C, penalty="l1", dual=False, tol=1e-3))),
            ('classification', LinearSVC(C=C, penalty="l2"))]),
                  additional='SVC L2 with L1 feature selection', param=C)


# In[8]:


results = pd.DataFrame({'Names': model_descr, 'Accuracy': scores, 'Param': np.log10(model_param),
                  'Fantasy LogLoss': fantasy_log_losses, 'LogLoss': true_log_losses})
results = results.groupby(['Names', 'Param']).mean().reset_index().sort_values(['Names', 'Param'])
# These results are in a "long format", need to pivot them along the parameters.
# But beware that the parameters are not the same for all models, so there will
# be nans.

plt.style.use('ggplot')
plt.rcParams['font.size'] = 18
# Need to interpolate to avoid skipping values. However, do not want to extrapolate, so small hack
output = results[['Names','Param','Accuracy']].pivot_table(index='Param', values='Accuracy', columns='Names')
mask = output.fillna(method='bfill').notnull().replace(0, 'nan')
output = output.interpolate('index') * mask
output.plot(figsize=(15,10), linewidth=3, colormap='tab10')
plt.xlabel('Parameter (C or $\\alpha$)')
yl = list(plt.ylim())
yl[0] = max(yl[0], .75)
yl[1] = min(yl[1], 1)
plt.xticks(np.arange(-7,2), ['$10^{%d}$'%x for x in np.arange(-7,2)])
plt.yticks(np.arange(0,1,0.02), ['{:.0%}'.format(x) for x in np.arange(0,1,0.02)])
plt.ylim(yl)
plt.title('Out-of-sample accuracy of different methods')
plt.ylabel('Accuracy')
plt.show()


# **It seems that Naive Bayes has the best out-of-the-box accuracy**, followed by L2 SVC, Ridge regression, and Passive Aggressive classifier. It's also the least sensitive to changes in the regularization parameter.
# 
# While nearly 84% accuracy is not impressive, it's still a rather good result, considering that, under the hood, everything that we are checking is which words appear more often in an author's writing.
# 
# Let's take a look at the corresponding log losses, we'll discuss after what they actually mean.

# In[9]:


# Need to interpolate to avoid skipping values. However, do not want to extrapolate, so small hack
output = results[['Names','Param','Fantasy LogLoss']].pivot_table(index='Param', values='Fantasy LogLoss', columns='Names')
mask = output.fillna(method='bfill').notnull().replace(False, np.nan)
output = output.interpolate('index') * mask
output.plot(figsize=(15,10), linewidth=3, colormap='tab10')
plt.xlabel('Parameter (C or $\\alpha$)')
true_output = results[['Names','Param','LogLoss']].dropna().pivot_table(index='Param', values='LogLoss', columns='Names')
for c in true_output.columns:
    plt.plot(true_output[c], '--', label=c + ' "true" log loss')
plt.legend()
plt.xticks(np.arange(-7,3), ['$10^{%d}$'%x for x in np.arange(-7,3)])
plt.ylim([.3,.7])
plt.title('Out-of-sample "log-loss" of different methods')
plt.ylabel('Log Loss')
plt.show()


# As expected, the artificially constructed Log Loss reflects the same results as the accuracy plot; *but it goes nowhere near the "true" log loss, computed via predict_proba*.
# 
# Before proceeding further, let's explain shortly how that "fantasy" log loss was computed, since this will give some insight on how to push that nice dotted blue line slightly further down.

# ## 1.4. Intermezzo. Some math about log loss.
# 
# A very true statement is that when you introduce a metric for reward, you are incentivizing people to optimize that metric and not necessarily improving the overall quality. So that's exactly what we're going to do here, try to squeeze out the bestt possible log loss without improving the accuracy of our models.
# 
# **Problem:** Suppose that we have a non-probabilistic classifier that has a certain accuracy, say $\lambda = 84\%$. What answers $p_{ij}$ should it give to minimize the log loss:
# $$
# logloss = -\frac{1}{N} \sum_{i=1}^N \log(p_{i,y_i}),
# $$
# where, for each $i$, $y_i$ is the true class to which the $i$-th test element belongs?
# 
# Since our classifier is non-probabilistic, for each $i$, we only know the estimate $\hat{y_i}$, the most likely prediction for the $i$-th element. We'd be tempted to say simply:
# $$
# p_{ij} = \left\{ \begin{array}{ll} 1 & \text{ if } j=\hat{y_i}\\ 0 & \text{ otherwise.}\end{array} \right .
# $$
# But this is clearly wrong: a single misclassified element $\hat y_i \neq y_i$ would contribute $-\log(0) = \infty$ to the mean; hard to bring that down.
# 
# (actually Kaggle helps us in this, bounding all numbers at $10^{-15}$; thus we'd "only" contribute the error by $\log(10^{15}) \approx 34.5$...)
# 
# We now make some very simplifying assumptions to quickly give a short answer. Suppose that we only have two classes, both with 50% chance, and our estimator accuracy is the same for both classes. Then we only need estimate one parameter $p=p_{i,\hat y_i} \in [0.5,1]$, the output corresponding to a predicted class (the other one then being $1-p$). The overall logloss is then:
# $$
# logloss = -\frac{1}{N} \sum_{i=1}^N \log(p) \mathbb{1}_{y_i = \hat y_i} + \log(1-p) \mathbb{1}_{y_i \neq \hat y_i} = \lambda \log(p) + (1-\lambda) \log(1-p).
# $$
# Deriving the above one gets $\frac{\lambda}{p} = \frac{1-\lambda}{1-p}$, that without any computation is true when $p=\lambda$. However we said plots, not words, so:

# In[10]:


l = 0.84
p = np.linspace(0.5,1-1e-6,100)
plt.figure(figsize=(15,7))
plt.plot(p, -l*np.log(p) - (1-l)*np.log(1-p), label='log-loss')
plt.plot([0.84,0.84], [0.3, 1.2], '--', label='$p = \lambda = 84\%$')
plt.xlabel('$p$')
plt.ylabel('logloss')
plt.ylim([0.3, .7])
plt.xticks([.5, .6, .7, .8, .84, .9, 1])
plt.legend()
plt.title('Plot of logloss $-\lambda \log(p) - (1-\lambda)\log(1-p)$, $\lambda=84\%$')
plt.show()


# If nothing else, the plot shows that calibrating $p$ is useful, but the function is sufficiently permissive, any value between $p=0.8$ and $p=0.88$ does not blow up too much.
# 
# When we have 3 classes, uneven probabilities, or uneven accuracy, the computations become more cumbersome, as the problem has now in general 6 free variables (three positive outputs given three possible prediction, each with the condition that the sum be 1). Lagrange multipliers help here, but that's cumbersome; that's why above we took a shortcut, simply setting the predicted probability at $p=\lambda$ and the other two at $\frac{1-\lambda}{2}$.
# 
# However, the process is intrinsically wrong: our model *should* know better than us how sure it is if itself. If a sentence refers to Cthulhu, we probably don't want it to have 8% chance to be by MWS and 8% by EAP. And indeed, for Naive Bayes, using predict_proba method automatically gave much better results.
# 
# Was then all this digression for nothing? Not really! Because [probabilistic estimators are well known to have biases in estimation](http://scikit-learn.org/stable/modules/calibration.html). For example, if you train a neural network long enough, it will overfit and become very sure of itself, tending to predict $p\approx 100\%$. For this reason, we'll now improve our models by means of calibrations, that under the hood just squeeze the $p$ like we did in this toy example.

# ## 1.5. Calibrating probabilities
# We'll have to restrict to probabilistic estimators; that's Naive Bayes above. We'll throw in two more probabilistic classifiers, good old Logit and modified Huber (that is just a more robust version thereof).
# 
# (again, two others should be added, but they're excluded because too slow: Random Forests and Support Vector Machines. Also, all tests I've done offline result in bad results, so you're free to try, please do share a comment if you find a version with satisfying answers!)
# 
# Also, scikit-learn comes with two built-in calibrations, Sigmoid and Isotonic. In our situation, both simply calibrate a function:
# $$
# f \colon D^2 \to D^2,
# $$
# where $D^2 = \{(x,y,z) \in \mathbb R_+^3 \ :\ x+y+z=1\}$ is the 2-dimensional simplex. This function takes as input the prediction of a model and produces an output built to minimize the log-loss; in the first case the choice is restricted to some sigmoid functions, hence only a few parameters are chosen. The second case is non-parametric (step-wise approximation), thus allowing for more flexibility, but also risking more overfit if the data is not enough.

# In[11]:


def test_with_calibration(vectorizer=TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english'),
                          C_logit = [0.3, 1, 3, 10, 30], alpha_SGD = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                          alpha_NB = [0.01, 0.03, 0.1, 0.3, 1]):
    this_range = trange(10)
    model_names = []
    params = []
    targets = []
    preds = []
    preds_calib_sig = []
    preds_calib_iso = []

    models = [LogisticRegression(C=c) for c in C_logit] + [
        SGDClassifier(loss='modified_huber', alpha=a, max_iter=10000, tol=1e-4) for a in alpha_SGD] + [
        MultinomialNB(alpha=a) for a in alpha_NB] + [
        BernoulliNB(alpha=a) for a in alpha_NB]
    
    for test_loop in this_range:
        # Partition: 90% into train, 10% to test.
        test = csv.iloc[int(len(csv)*test_loop/10) : int(len(csv)*(test_loop+1)/10)]
        train = csv.loc[list(set(csv.index).difference(test.index))]

        y_train = train.author.apply(lambda auth: authors_dict[auth]).values
        y_test = test.author.apply(lambda auth: authors_dict[auth]).values

        # For more flexibility, vectorizer can be changed as a parameter.
        X_train = vectorizer.fit_transform(train.text.values)
        X_test = vectorizer.transform(test.text.values)

        for m in models:
            name = str(m).split('(')[0]

            if name.endswith('NB') or name == 'SGDClassifier':
                param = m.alpha
            elif name == 'SVC' or name == 'LogisticRegression':
                param = m.C

            this_range.set_postfix(working_on=name, step='base')

            m.fit(X_train, y_train)
            targets.append(y_test)
            model_names.append([name] * len(y_test))
            params.append([param] * len(y_test))
            preds.append(m.predict_proba(X_test))

            this_range.set_postfix(working_on=name, step='sigmoid')

            # Sigmoid calibration
            m_sigmoid = CalibratedClassifierCV(m, method='sigmoid')
            m_sigmoid.fit(X_train, y_train)
            preds_calib_sig.append(m_sigmoid.predict_proba(X_test))

            this_range.set_postfix(working_on=name, step='isotonic')
            # Isotonic calibration
            m_isotonic = CalibratedClassifierCV(m, method='isotonic')
            m_isotonic.fit(X_train, y_train)
            preds_calib_iso.append(m_isotonic.predict_proba(X_test))


    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    preds_calib_sig = np.concatenate(preds_calib_sig)
    preds_calib_iso = np.concatenate(preds_calib_iso)
    params = np.log10(np.concatenate(params))
    model_names = np.concatenate(model_names)

    log_losses = -np.log(np.clip(preds[np.arange(len(preds)), targets], 1e-15, 1-1e-15))
    log_losses_sig = -np.log(np.clip(preds_calib_sig[np.arange(len(preds_calib_sig)), targets], 1e-15, 1-1e-15))
    log_losses_iso = -np.log(np.clip(preds_calib_iso[np.arange(len(preds_calib_iso)), targets], 1e-15, 1-1e-15))

    final_results = pd.DataFrame({'LogLoss': log_losses, 'Param': params, 'Names': model_names,
                                  'Target': targets, 'Prediction': np.argmax(preds, axis=1),
                                  'LogLoss Sigmoid': log_losses_sig, 'LogLoss Isotonic': log_losses_iso})
    final_results['Accuracy'] = final_results['Target'] == final_results['Prediction']

    return pd.concat((final_results[[c, 'Names', 'Param']].groupby(['Names', 'Param']).mean()
                      for c in ('LogLoss', 'LogLoss Sigmoid', 'LogLoss Isotonic', 'Accuracy')),
                     axis=1).reset_index()


# In[12]:


calibrated_results = test_with_calibration()


# In[63]:


def plot_calibrated_results(calibrated_results, title='Calibrated log-loss', ylim=[.3,.6]):
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 18

    # Use the same color for the same model, different styles per calibration
    colors = [c['color'] for c in list(plt.rcParams['axes.prop_cycle'])]
    n_cols = len(colors)
    
    logloss_columns = ['LogLoss', 'LogLoss Sigmoid', 'LogLoss Isotonic']

    plt.figure(figsize=(15,10))
    for j,n in enumerate(calibrated_results.Names.unique()):
        for c, style in zip(logloss_columns, [':','--','-']):
            plt.plot(calibrated_results.loc[calibrated_results.Names==n, 'Param'],
                     calibrated_results.loc[calibrated_results.Names==n, c], style,
                     color=colors[j%n_cols], linewidth=3,
                     label=n + ('' if ' ' not in c else ' '+c.split()[1]))
    plt.xlabel('Parameter (C or $\\alpha$)')
    xt = [x for x in np.arange(-10,10)
          if x >= calibrated_results.Param.min() and x <= calibrated_results.Param.max()]
    plt.xticks(xt, ['$10^{%d}$'%x for x in xt])
    plt.ylim(ylim)
    plt.xlim(calibrated_results.Param.min(), calibrated_results.Param.max())
    plt.title(title)
    plt.ylabel('Log Loss')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()


# In[14]:


plot_calibrated_results(calibrated_results)


# Results improved a bit, which is good because it's free (recall that we didn't change the accuracy a single bit here).
# - Isotonic calibration seems generally similar to sigmoid one, with the puzzing exception of BernoulliNB.
# - SGD greatly profits from calibration, but still remains inferior.
# - Logit regression theoretically has a sound probability estimate, so, as expected, it doesn't profit very much (even losing some).
# - Bernoulli Naive Bayes definitely needs some calibration; this is less manifest for Multinomial one.

# ## 1.6. A peek into the future
# 
# We've done some testing with various models, but they all started from the same step: built-in Tf-idf vectorization, with the parameters from the example on scikit-learn documentation. In section 2, we'll start questioning this choice more seriously. For the moment, let's just try a few more built-in options, maybe we'll get a lucky shot... A few options to analyze:
# - *max_df*: it removes too frequent words. Actually 0.5 is already a high value, and it's not removing anything. Could we lower it?
#     - (**Pro**) Akin to stopwords removal
#     - (**Con**) Could lose some information...
# - *stop_words*: Do we really want to remove them? (similar pros and cons)
# - *sublinear_tf*: for term frequency, we log(count) instead of count. Can we remove it?
#     - (**Pro**) Since stopwords are already removed, log might be an overkill.
#     - (**Con**) We could then give too much weight to common words; even more so if we brought stopwords back.
# - *token_pattern*: By default, words must have at least 2 non-punctuation characters. Can change this to a more "basic" definition of word.
#     - (**Pro**) Lose less information
#     - (**Con**) Might get some noise in.
# - *ngram_range*: Use n-grams instead of words, i.e. consecutive group of words.
#     - (**Pro**) Makes much intuitive sense to look for typical expressions instead of just typical words.
#     - (**Con**) Will increase the number of features significantly, risking more overfit and longer fitting times.
# - *CountVectorizer*: Many people don't like Tf-idf at all and prefer to work with raw word counts instead. It has largely the same structure, but the _tf and _df options above don't apply here.
#     - (**Pro**) Somehow more natural, sometimes gives better results.
#     - (**Con**) Leads to some big numbers, lacking any normalization. Also causes some overflow problems with Pratt's (sigmoid) calibration, which uses exponentials.

# To cut things short, reducing max_df or removing sublinear_tf does more harm than good (although the very best scores are the same), so I'll not report them. Actually, the following plot shows that the same also holds for passing to CountVectorizer. This surprised me a bit since I've seen other notebooks claiming the contrary. If you have different experiences, feel free to comment!

# In[15]:


# Offline run to save time. Uncomment this and comment tue following to run.
# using_countvectorizer = test_with_calibration(vectorizer=CountVectorizer(stop_words='english'))

using_countvectorizer = pd.DataFrame([['BernoulliNB', -2.0, 0.706319987941122, 0.4813479864258888,  0.43078230217300867, 0.8297155115174422], ['BernoulliNB', -1.5228787452803376, 0.6125400411922719,  0.4693183991047648, 0.4200070176089392, 0.8346697992747332], ['BernoulliNB', -1.0, 0.5318631184911502, 0.45957841327517474,  0.4124031543488292, 0.8377853822973594], ['BernoulliNB', -0.5228787452803376, 0.4897262986685063,  0.4636773714716324, 0.41451718613477395, 0.8371724807191379], ['BernoulliNB', 0.0, 0.5178300753228159, 0.5162830243131579,  0.44378952304323976, 0.8171510291639001], ['LogisticRegression', -0.5228787452803376, 0.5439896178579512,  0.5206278056309603, 0.5189494603320542, 0.799070432606364], ['LogisticRegression', 0.0, 0.4939695963535932, 0.5048351798532839,  0.5019940867981258, 0.8081618060166504], ['LogisticRegression', 0.47712125471966244, 0.49769625473481144,  0.5084358776109752, 0.5058884868022172, 0.8038204198375811], ['LogisticRegression', 1.0, 0.5655966032375217, 0.5237789358787479,  0.5223114794515888, 0.795444098268553], ['LogisticRegression', 1.4771212547196624, 0.6916563102635536,  0.5419143219215302, 0.5405129033294451, 0.7849226211757495], ['MultinomialNB', -2.0, 0.718011588213635, 0.4843884436595753,  0.4345146438138747, 0.8253230502068543], ['MultinomialNB', -1.5228787452803376, 0.6217299212692091,  0.47175176740564567, 0.4240145043205851, 0.830685939016293], ['MultinomialNB', -1.0, 0.5364765812733492, 0.45961192052773553,  0.4158300934342699, 0.8346187241432147], ['MultinomialNB', -0.5228787452803376, 0.48446107569802527,  0.45566722031583384, 0.41677265516149553, 0.8357423770366209], ['MultinomialNB', 0.0, 0.45877407274879933, 0.464806847973281,  0.43570686348654986, 0.8304305633587007], ['SGDClassifier', -5.0, 4.885052266624309, 0.5405027466565611,  0.5385120105442537, 0.7667909494866949], ['SGDClassifier', -4.522878745280337, 4.150779202151389,  0.5416658943504635, 0.5397880680058839, 0.7801215588130139], ['SGDClassifier', -4.0, 2.692209778404653, 0.5327215220018812,  0.5304161262567021, 0.793350017876296], ['SGDClassifier', -3.5228787452803374, 1.3803166326665117,  0.5182917369567382, 0.517008856903903, 0.8023392410235456], ['SGDClassifier', -3.0, 0.6988518162337475, 0.507113596537479,  0.5036106107336277, 0.8055058991776903]], columns=['Names', 'Param', 'LogLoss', 'LogLoss Sigmoid', 'LogLoss Isotonic', 'Accuracy'])

plot_calibrated_results(using_countvectorizer, title='Using CountVectorizer')


# Interstingly and, somewhat, surprisingly, adding back stopwords actually helps quite a bit, allowing to go **beyond 0.4 logloss**!

# In[16]:


plot_calibrated_results(test_with_calibration(vectorizer=TfidfVectorizer(sublinear_tf=True)),
                       title='Not discarding stopwords')


# Following the same line of though, why should we limit words to 2+ characters? Indeed, we shouldn't, and dropping that also helps a tiny bit:

# In[17]:


# Offline run to save time. Uncomment this and comment tue following to run.
# custom_token_pattern = test_with_calibration(vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}',sublinear_tf=True))
custom_token_pattern = pd.DataFrame([['BernoulliNB', -2.0, 0.6635507999191133, 0.44788502436384,  0.39315348567426417, 0.8456509525512028], ['BernoulliNB', -1.5228787452803376, 0.5788288927657376,  0.435446783320394, 0.3830651254165367, 0.8497369630726799], ['BernoulliNB', -1.0, 0.5078835075618886, 0.42725610915188744,  0.3759747559507382, 0.8528014709637878], ['BernoulliNB', -0.5228787452803376, 0.47314354690823984,  0.4340981141919862, 0.37993489818399007, 0.8516778180703816], ['BernoulliNB', 0.0, 0.5019734806584952, 0.48646348803130657,  0.41263612786031756, 0.8354359262475101], ['LogisticRegression', -0.5228787452803376, 0.7026691796469897,  0.5869410485168362, 0.5905181513515557, 0.7765462996067215], ['LogisticRegression', 0.0, 0.5692719996490762, 0.5083127564046314,  0.5092162033832176, 0.8146483477194953], ['LogisticRegression', 0.47712125471966244, 0.4792948573113742,  0.45906101187564213, 0.46239255113852323, 0.8330353950661423], ['LogisticRegression', 1.0, 0.4290701429588527, 0.4375943934428565,  0.43948342537860563, 0.8397773124265795], ['LogisticRegression', 1.4771212547196624, 0.43300570627987706,  0.4385391369298042, 0.4408027000067626, 0.8358445272996577], ['MultinomialNB', -2.0, 0.4125941620437241, 0.4190215749565747,  0.40828040804656934, 0.8393687113744318], ['MultinomialNB', -1.5228787452803376, 0.40714249110758255,  0.40129174326712436, 0.39360636427980883, 0.8455998774196843], ['MultinomialNB', -1.0, 0.4259516450862784, 0.3884868773956777,  0.383110501574009, 0.8498901884672353], ['MultinomialNB', -0.5228787452803376, 0.47630251030213644,  0.39167800671646685, 0.3901468642181604, 0.8466724551815721], ['MultinomialNB', 0.0, 0.5812538136471457, 0.42287631504181267,  0.4247365110254928, 0.8193983349507125], ['SGDClassifier', -5.0, 2.644901132462858, 0.47825526055714196,  0.47794256422203635, 0.8129117932478677], ['SGDClassifier', -4.522878745280337, 1.2549952199372436,  0.4619323425203225, 0.4603783489936237, 0.8290004596761836], ['SGDClassifier', -4.0, 0.6341338181070553, 0.4339657370855429,  0.4342455996997695, 0.8413095663721334], ['SGDClassifier', -3.5228787452803374, 0.5985628556207552,  0.4483802897245992, 0.44981922818186937, 0.8316052913836253], ['SGDClassifier', -3.0, 0.6680392917833494, 0.5073434790084068,  0.5091260173524894, 0.8030032177332856]], columns=['Names', 'Param', 'LogLoss', 'LogLoss Sigmoid', 'LogLoss Isotonic', 'Accuracy'])
plot_calibrated_results(custom_token_pattern, title='Custom token_pattern')


# Finally, let's be brave and **use also 2-grams**.
# 
# Going up from single words will somehow be the focus of the second part of this notebook, so for now let's just try it without explanations. It'll double the running time, as now we consider many more "words". But we will be somewhat recompensated:

# In[18]:


# Offline run to save time. Uncomment this and comment tue following to run.
# using_2grams = test_with_calibration(vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2)), C_logit=[3,10,30,100,300])
using_2grams = pd.DataFrame([['BernoulliNB', -2.0, 1.1829537822268923, 0.42882184350651664,  0.3574336854854024, 0.8605648909545942], ['BernoulliNB', -1.5228787452803376, 1.0034099170691535,  0.4238300133137205, 0.3521584466856666, 0.8617396189795189], ['BernoulliNB', -1.0, 0.8584855983611743, 0.4329739587553948,  0.35381801965661336, 0.8620971449001481], ['BernoulliNB', -0.5228787452803376, 0.907776895154225,  0.5143400658599186, 0.38069723614289763, 0.8418713928188365], ['BernoulliNB', 0.0, 2.5284020189561827, 0.8665639570532223,  0.48821064408572684, 0.6911997548393687], ['LogisticRegression', 0.47712125471966244, 0.5166478797790038,  0.48547839857229863, 0.5037727689750268, 0.8290515348077021], ['LogisticRegression', 1.0, 0.4392387650003737, 0.45016674100276843,  0.461878375648529, 0.8435057970274273], ['LogisticRegression', 1.4771212547196624, 0.40915645510723675,  0.4352609278613576, 0.4472301715832008, 0.8477450329434598], ['LogisticRegression', 2.0, 0.40765827822750805,  0.42700452030205444, 0.4389093348868385, 0.8498901884672353], ['LogisticRegression', 2.4771212547196626, 0.42449890413885477,  0.42276079942698247, 0.43303743984503246, 0.850962766229123], ['MultinomialNB', -2.0, 0.3578113672136049, 0.37545929030279757,  0.36030367415241854, 0.8583175851677818], ['MultinomialNB', -1.5228787452803376, 0.3475764766027269,  0.36321418518356896, 0.3577637730813878, 0.8640890750293682], ['MultinomialNB', -1.0, 0.37991004939354456, 0.36445150749047184,  0.37139598970530746, 0.8616885438480004], ['MultinomialNB', -0.5228787452803376, 0.472218907236443,  0.3899353777583827, 0.40325147650530285, 0.835538076510547], ['MultinomialNB', 0.0, 0.6367635084522356, 0.4483081235646716,  0.46652352411338466, 0.7606108585729608], ['SGDClassifier', -5.0, 1.875315599401462, 0.4362438404927373,  0.43556807875290676, 0.8412074161090964], ['SGDClassifier', -4.522878745280337, 0.7585651162455638,  0.4224121891513341, 0.427068782985493, 0.8499923387302722], ['SGDClassifier', -4.0, 0.5828856388906344, 0.4246164926556892,  0.4358121664644361, 0.8480514837325707], ['SGDClassifier', -3.5228787452803374, 0.6079532216632112,  0.4659431255610908, 0.48242465809839463, 0.8286940088870729], ['SGDClassifier', -3.0, 0.7164340295311806, 0.547004057582182,  0.562190687042716, 0.7859951989376373]], columns=['Names', 'Param', 'LogLoss', 'LogLoss Sigmoid', 'LogLoss Isotonic', 'Accuracy'])

plot_calibrated_results(using_2grams, title='Using 2-grams')


# If anything, this leaves us with a conundrum: should we use calibration, or should we not? Is that 0.01 gained logloss to be trusted, or is it just a data glitch and we should encourage the constant quality of the calibrated method (also much less dependent on the exact value of $\alpha$?
# 
# Anyway we **touched the 0.35 threshold** which is definitely good enough for this very simple model. And the next section will give the solution to the conundrum...
# 
# By the way, this is as far as we can push this line of inquiry. Increasing to 3-grams or 4-grams will actually increase the log-loss (wasting much more time in doing so), probably because no method is safe from overfitting.
# 
# Time to build up a solution to submit, and as is a must in Kaggle's competitions this means using some...

# ## 1.7. Ensembling and submission
# As with anything else, there's many ways of doing ensembling. In the following, we'll just use two built-in ones, voting and bagging. Again, we'll compare by cross validation.

# In[19]:


previous_best = [('MultiNB', MultinomialNB(alpha=0.03)),
     ('Calibrated MultiNB', CalibratedClassifierCV(MultinomialNB(alpha=0.03), method='isotonic')),
     ('Calibrated BernoulliNB', CalibratedClassifierCV(BernoulliNB(alpha=0.03), method='isotonic')),
     ('Calibrated Huber', CalibratedClassifierCV(
         SGDClassifier(loss='modified_huber', alpha=1e-4, max_iter=10000, tol=1e-4), method='sigmoid')),
     ('Logit', LogisticRegression(C=30))]

def test_ensembling():
    ''' Compare two ways of ensembling:
        - Voting, among the best scorers above (possibly with weights)
        - Bagging, multiple times the best above (with and without calibration) '''
    this_range = trange(10)
    model_names = []
    targets = []
    preds = []
    vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))

    voting_model_uniform = VotingClassifier(previous_best, voting='soft')
    
    voting_model_NB = VotingClassifier([(s,m) for s,m in previous_best if 'NB' in s], voting='soft')
    
    voting_model = VotingClassifier(previous_best, voting='soft', weights=[3,3,3,1,1])
    
    bagging_model = BaggingClassifier(MultinomialNB(alpha=0.03))
    
    bagging_calibrated_model = BaggingClassifier(
        CalibratedClassifierCV(MultinomialNB(alpha=0.03), method='isotonic'))
    
    models = [voting_model_uniform, voting_model_NB,
              voting_model, bagging_model, bagging_calibrated_model]
    these_names = ['Voting Uniform Weights', 'Voting NB Only',
                   'Voting Merit Weights', 'MultinomialNB Bagging', 'Calibration + Bagging']
    
    for test_loop in this_range:
        test = csv.iloc[int(len(csv)*test_loop/10) : int(len(csv)*(test_loop+1)/10)]
        train = csv.loc[list(set(csv.index).difference(test.index))]
        
        y_train = train.author.apply(lambda auth: authors_dict[auth]).values
        y_test = test.author.apply(lambda auth: authors_dict[auth]).values
        
        vectorizer = TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
        X_train = vectorizer.fit_transform(train.text.values)
        X_test = vectorizer.transform(test.text.values)
        
        for m,name in zip(models, these_names):
            this_range.set_postfix(working_on=name)

            m.fit(X_train, y_train)
            targets.append(y_test)
            model_names.append([name] * len(y_test))
            preds.append(m.predict_proba(X_test))

    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    model_names = np.concatenate(model_names)

    log_losses = -np.log(np.clip(preds[np.arange(len(preds)), targets], 1e-15, 1-1e-15))

    final_results = pd.DataFrame({'LogLoss': log_losses, 'Names': model_names,
                                  'Target': targets, 'Prediction': np.argmax(preds, axis=1)})
    final_results['Accuracy'] = final_results['Target'] == final_results['Prediction']

    return final_results[['LogLoss', 'Names']].groupby('Names').mean().reset_index()


# In[20]:


ensembling_results = test_ensembling()


# In[21]:


plt.rcParams['font.size'] = 20
plt.figure(figsize=(15,10))
y = ensembling_results.LogLoss
y2 = (max(y)-y) / (max(y) - min(y))
colors = cm.RdYlGn(y2)
plt.bar(np.arange(len(ensembling_results)), ensembling_results.LogLoss, color=colors)
plt.xticks(np.arange(len(ensembling_results)), ensembling_results.Names.str.replace(' ', '\n'))
plt.title('Log losses of ensembling methods\n(the greener the better)')
plt.show()


# As usual, putting more models together and voting seems to help a little bit. Giving more weight to the better models is a good idea, but still excluding the suboptimal ones is not the best.
# 
# *Bagging seems to be damaging* in this situation. Given this fact, it is probably not surprising that Random Forests performed poorly. One reason I can think of is that sampling with repetitions will inevitably lack some words, thus leaving us with many rather poor models.
# 
# Anyway, we've got our winner, let's use it to submit a prediction.

# In[22]:


vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(previous_best, voting='soft', weights=[3,3,3,1,1])
X_train = vectorizer.fit_transform(csv.text.values)
y_train = csv.author.apply(lambda auth: authors_dict[auth]).values
clf.fit(X_train, y_train)

test = pd.read_csv('../input/test.csv', index_col=0)
X_test = vectorizer.transform(test.text.values)
result = clf.predict_proba(X_test)
pd.DataFrame(result, index=test.index, columns=authors).to_csv('tfidf_results.csv')


# Final score is 0.334. Yuppie!

# # 2. Grammar
# ## 2.1. Further data investigation
# We went a long way simply using words (not even words, but stems), or at most pairs thereof, and disregarding their order. But if we want to squeeze all the possible information, throwing away all the other information seems wasteful. Some things that we are disregarding right now:
# 
# - Grammar / Words terminations (singular vs. plural, present vs. past, etc.);
# - Groups of more than two words;
# - Punctuation;
# - Sentence length.
# 
# However, there is a reason: implementing all these features (especially the second one) would make the dimensionality of the problem blow up.
# 
# Before seeing if neural networks allow us to pick such features seemlessly, let's investigate a bit more these features to see if they are useful at all.
# 
# ### 2.1.1. Punctuation and grammar
# First, let's take a look at the sentences of our authors, to see if their grammar gives any hint. For this, we'll use nltk.pos_tag (that tags the interpreted grammar of the sentence)

# In[53]:


for df in [csv, test]:
    df['split'] = df.text.apply(nltk.word_tokenize)
    df['PosTag'] = [[x[1] for x in nltk.pos_tag(y)] for y in df.split]


# In[24]:


tags_exp = nltk.data.load('help/tagsets/upenn_tagset.pickle')
tags = pd.Series([x for y in csv.PosTag for x in y]).value_counts().to_frame()
tags.columns = ['Count']
tags['Explanation'] = [tags_exp[i][0] for i in tags.index]


# In[25]:


tags['Percentage'] = tags['Count'] / tags['Count'].sum()
for aut in authors:
    these_tags = pd.Series([x for y in csv.loc[csv.author==aut, 'PosTag'] for x in y]).value_counts().to_frame()
    these_tags.columns = ['Percentage_' + aut]
    these_tags.iloc[:,0] = these_tags.iloc[:,0] / these_tags.iloc[:,0].sum()
    tags = pd.merge(tags, these_tags, left_index=True, right_index=True, how='left').fillna(0)


# In[26]:


max_diff = np.max(tags[['Percentage_'+x for x in ['MWS', 'HPL', 'EAP']]].values - np.expand_dims(tags['Percentage'].values, 1))

print('The maximum difference between any author percentage and the overall is {:.3f} percentage points'.format(
    max_diff*100))

tags.head()


# Let's visualize these percentages to see if they are significantly different
# 
# (the following hidden cells are a hack to type a 100x100 image as a numpy array)

# In[27]:


frankenstein1 = np.array([0]*52+[67]+[131]+[188]+[229]+[248]+[254]+[246]+[228]+[196]+[163]+[149]+[149]+[149]+[149]+[149]+[149]+[149]+[149]+[160]+[185]+[188]+[172]+[141]+[97]+[60]+[21]+[0]*62+[29]+[90]+[139]+[138]+[126]+[126]+[126]+[126]+[126]+[126]+[137]+[255]*28+[193]+[143]+[126]+[126]+[126]+[126]+[126]+[121]+[54]+[16]+[0]*49+[4]+[222]+[255]*49+[135]+[0]*47+[17]+[247]+[255]*51+[205]+[0]*45+[13]+[248]+[255]*53+[186]+[0]*44+[43]+[255]*54+[235]+[0]*44+[82]+[255]*54+[242]+[1]+[0]*43+[127]+[255]*54+[247]+[1]+[0]*43+[168]+[255]*54+[254]+[1]+[0]*43+[209]+[255]*55+[9]+[0]*43+[249]+[255]*55+[42]+[0]*43+[255]*56+[80]+[0]*42+[17]+[255]*56+[114]+[0]*42+[35]+[255]*56+[150]+[0]*42+[53]+[255]*56+[187]+[0]*42+[69]+[255]*56+[223]+[0]*42+[86]+[255]*56+[252]+[0]*42+[107]+[255]*56+[254]+[3]+[0]*41+[128]+[255]*57+[9]+[0]*41+[153]+[255]*57+[15]+[0]*41+[176]+[255]*57+[22]+[0]*41+[195]+[255]*57+[28]+[0]*41+[211]+[255]*57+[34]+[0]*41+[229]+[255]*57+[47]+[0]*41+[243]+[255]*57+[76]+[0]*41+[251]+[255]*57+[111]+[0]*41+[255]*58+[142]+[0]*41+[252]+[255]*57+[227]+[27]+[0]*40+[246]+[255]*59+[103]+[0]*39+[236]+[255]*59+[240]+[0]*39+[219]+[255]*60+[29]+[0]*38+[202]+[255]*60+[60]+[0]*38+[180]+[255]*60+[29]+[0]*38+[157]+[255]*59+[99]+[0]*39+[127]+[255]*58+[69]+[0]*40+[97]+[255]*57+[35]+[0]*41+[73]+[255]*57+[92]+[0]*41+[49]+[255]*58+[101]+[0]*40+[19]+[255]*59+[96]+[0]*40+[255]*60+[54]+[0]*39+[191]+[255]*59+[249]+[8]+[0]*38+[92]+[255]*60+[229]+[1]+[0]*37+[25]+[255]*61+[105]+[0]*38+[249]+[255]*61+[38]+[0]*37+[106]+[255]*61+[184]+[0]*38+[251]+[255]*60+[243]+[0]*38+[213]+[255]*60+[182]+[0]*38+[42]+[255]*60+[69]+[0]*39+[197]+[255]*56+[241]+[122]+[35]+[0]*40+[146]+[255]*54+[107]+[27]+[0]*43+[93]+[255]*54+[26]+[0]*44+[41]+[255]*54+[23]+[0]*45+[255]*54+[31]+[0]*45+[255]*54+[51]+[0]*45+[255]*54+[84]+[0]*44+[37]+[255]*54+[154]+[0]*44+[126]+[255]*55+[0]*44+[213]+[255]*55+[40]+[0]*42+[53]+[255]*54+[230]+[9]+[0]*41+[26]+[118]+[223]+[255]*55+[86]+[0]*40+[1]+[238]+[255]*58+[22]+[0]*39+[129]+[255]*57+[251]+[124]+[0]*39+[102]+[255]*20+[198]+[18]+[34]+[232]+[255]*34+[70]+[0]*37+[83]+[215]+[240]+[255]*20+[197]+[0]*4+[233]+[255]*33+[217]+[0]*35+[133]+[235]+[255]*23+[107]+[0]*4+[180]+[255]*34+[33]+[0]*31+[6]+[101]+[255]*26+[153]+[0]*4+[209]+[255]*34+[114]+[0]*26+[1]+[22]+[52]+[139]+[227]+[255]*29+[73]+[7]+[11]+[124]+[255]*35+[222]+[0]*23+[1]+[20]+[151]+[254]+[255]*73+[0]*21+[3]+[155]+[234]+[255]*75+[241]+[0]*19+[54]+[187]+[255]*60+[203]+[255]*17+[127]+[0]*17+[45]+[187]+[255]*61+[211]+[0]+[1]+[101]+[178]+[230]+[255]*12+[216]+[4]+[0]*15+[8]+[147]+[255]*63+[38]+[0]*6+[79]+[145]+[197]+[220]+[225]+[229]+[230]+[230]+[229]+[227]+[164]+[0]*16+[57]+[255]*64+[119]+[0]*33+[168]+[255]*64+[240]+[0]*32+[2]+[205]+[255]*65+[38]+[0]*31+[23]+[238]+[255]*65+[214]+[0]*31+[24]+[255]*67+[202]+[0]*30+[9]+[250]+[255]*67+[202]+[0]*30+[240]+[255]*68+[204]+[0]*28+[4]+[234]+[255]*69+[58]+[0]*28+[189]+[255]*69+[107]+[0]*28+[124]+[255]*69+[169]+[0]*29+[255]*70+[0]*30+[255]*69+[123]+[0]*30+[255]*69+[85]+[0]*30+[255]*69+[69]+[0]*30+[255]*69+[28]+[0]*30+[255]*69+[33]+[0]*30+[255]*69+[144]+[0]*30+[255]*70+[11]+[0]*29+[255]*7+[245]+[212]+[184]+[158]+[125]+[93]+[68]+[50]+[35]+[26]+[19]+[20]+[27]+[38]+[53]+[73]+[97]+[127]+[159]+[184]+[211]+[241]+[255]*41+[150]+[0]*29+[255]+[255]+[248]+[229]+[184]+[94]+[12]+[0]*22+[1]+[75]+[154]+[221]+[239]+[255]*36+[251]+[0]*29+[202]+[63]+[1]+[0]*30+[1]+[7]+[108]+[225]+[255]*34+[185]+[0]*65+[14]+[43]+[162]+[255]*31+[253]+[8]+[0]*67+[3]+[69]+[168]+[255]*29+[216]+[0]*70+[3]+[123]+[218]+[255]*27+[63]+[0]*72+[47]+[212]+[250]+[255]*23+[232]+[68]+[0]*75+[128]+[255]*20+[248]+[80]+[0]*78+[4]+[34]+[139]+[255]*15+[179]+[44]+[4]+[0]*83+[37]+[84]+[132]+[183]+[218]+[243]+[253]+[252]+[239]+[209]+[170]+[117]+[64]+[9]+[0]*32)/255.0


# In[28]:


cthulhu1 = np.array([0]*44+[18]+[42]+[53]+[68]+[82]+[93]+[93]+[82]+[68]+[53]+[42]+[17]+[0]*82+[8]+[64]+[118]+[156]+[195]+[234]+[255]*12+[234]+[194]+[155]+[117]+[63]+[7]+[0]*72+[1]+[56]+[123]+[189]+[246]+[255]*22+[246]+[188]+[122]+[55]+[1]+[0]*66+[31]+[119]+[220]+[255]*30+[226]+[136]+[38]+[0]*61+[5]+[97]+[199]+[255]*12+[240]+[208]+[192]+[181]+[169]+[153]+[153]+[169]+[181]+[192]+[208]+[240]+[255]*12+[197]+[95]+[5]+[0]*56+[9]+[108]+[225]+[255]*9+[215]+[150]+[102]+[64]+[24]+[0]*12+[24]+[64]+[102]+[150]+[216]+[255]*9+[224]+[106]+[8]+[0]*52+[5]+[115]+[232]+[255]*7+[240]+[160]+[91]+[26]+[0]*22+[26]+[92]+[161]+[241]+[255]*7+[230]+[112]+[5]+[0]*49+[62]+[212]+[255]*6+[253]+[187]+[94]+[11]+[0]*28+[8]+[85]+[179]+[250]+[255]*6+[217]+[68]+[0]*46+[24]+[164]+[254]+[255]*5+[242]+[138]+[25]+[0]*34+[26]+[139]+[243]+[255]*5+[254]+[162]+[23]+[0]*43+[78]+[237]+[255]*5+[237]+[128]+[18]+[0]*38+[19]+[130]+[239]+[255]*5+[236]+[76]+[0]*40+[6]+[142]+[254]+[255]*4+[252]+[145]+[15]+[0]*15+[14]+[88]+[150]+[212]+[243]+[161]+[140]+[241]+[221]+[161]+[99]+[24]+[0]*15+[16]+[148]+[252]+[255]*4+[254]+[139]+[5]+[0]*37+[31]+[200]+[255]*5+[199]+[49]+[0]*15+[68]+[167]+[246]+[255]*10+[252]+[183]+[87]+[4]+[0]*14+[51]+[201]+[255]*5+[199]+[30]+[0]*35+[49]+[232]+[255]*4+[247]+[108]+[2]+[0]*13+[2]+[88]+[206]+[255]*16+[223]+[109]+[8]+[0]*13+[2]+[110]+[248]+[255]*4+[230]+[47]+[0]*33+[64]+[240]+[255]*4+[218]+[48]+[0]*14+[68]+[206]+[255]*20+[227]+[102]+[3]+[0]*8+[3]+[0]*4+[49]+[219]+[255]*4+[243]+[69]+[0]*31+[83]+[248]+[255]*4+[167]+[13]+[0]+[0]+[1]+[105]+[212]+[247]+[222]+[130]+[12]+[0]*4+[23]+[179]+[255]*24+[203]+[42]+[0]*4+[3]+[112]+[219]+[253]+[234]+[147]+[14]+[0]+[0]+[14]+[169]+[255]*4+[247]+[82]+[0]*29+[83]+[252]+[255]*4+[128]+[0]*4+[154]+[255]*5+[194]+[1]+[0]+[0]+[13]+[217]+[255]*26+[238]+[34]+[0]*3+[146]+[255]*5+[205]+[15]+[0]*3+[130]+[255]*4+[252]+[82]+[0]*27+[64]+[247]+[255]*3+[252]+[105]+[0]*4+[97]+[255]+[255]+[238]+[255]*4+[65]+[0]+[0]+[129]+[255]*28+[172]+[0]+[0]+[19]+[248]+[255]*3+[230]+[249]+[255]+[159]+[0]*4+[106]+[253]+[255]*3+[249]+[69]+[0]*25+[49]+[241]+[255]*3+[248]+[84]+[0]*5+[210]+[193]+[39]+[0]+[106]+[255]*3+[133]+[0]+[7]+[234]+[255]*28+[254]+[36]+[0]+[77]+[255]*3+[152]+[0]+[15]+[147]+[251]+[18]+[0]*4+[85]+[249]+[255]*3+[240]+[47]+[0]*23+[31]+[232]+[255]*3+[252]+[84]+[0]*6+[183]+[9]+[0]+[0]+[25]+[255]*3+[161]+[0]+[95]+[255]*30+[136]+[0]+[104]+[255]*3+[78]+[0]*3+[131]+[45]+[0]*5+[86]+[253]+[255]*3+[230]+[30]+[0]*21+[5]+[197]+[255]*4+[110]+[0]*11+[27]+[255]*3+[150]+[0]+[199]+[255]*30+[234]+[7]+[92]+[255]*3+[82]+[0]*11+[104]+[255]*4+[201]+[6]+[0]*20+[142]+[255]*4+[127]+[0]*12+[72]+[255]*3+[99]+[37]+[255]*32+[80]+[40]+[255]*3+[129]+[0]*12+[130]+[255]*4+[138]+[0]*19+[78]+[254]+[255]*3+[167]+[0]*13+[162]+[255]+[255]+[244]+[19]+[129]+[255]*32+[170]+[0]+[203]+[255]+[255]+[221]+[7]+[0]*12+[169]+[255]*3+[254]+[76]+[0]*17+[24]+[237]+[255]*3+[218]+[13]+[0]+[0]+[5]+[74]+[133]+[151]+[123]+[67]+[2]+[0]*3+[83]+[254]+[255]+[255]+[142]+[0]+[219]+[255]*32+[247]+[14]+[82]+[255]*3+[147]+[0]*3+[1]+[67]+[135]+[153]+[133]+[74]+[5]+[0]+[0]+[14]+[219]+[255]*3+[236]+[22]+[0]*16+[164]+[255]*3+[248]+[50]+[0]+[0]+[74]+[221]+[255]*5+[207]+[52]+[0]+[86]+[250]+[255]+[255]+[224]+[13]+[48]+[255]*34+[89]+[0]+[177]+[255]*3+[137]+[2]+[50]+[206]+[255]*5+[220]+[72]+[0]+[0]+[46]+[246]+[255]*3+[160]+[0]*15+[65]+[254]+[255]*3+[108]+[0]+[0]+[97]+[254]+[255]*7+[248]+[96]+[238]+[255]+[255]+[247]+[59]+[0]+[128]+[255]*34+[170]+[0]+[21]+[222]+[255]*3+[176]+[176]+[255]*7+[254]+[95]+[0]+[0]+[110]+[255]*3+[254]+[62]+[0]*13+[6]+[214]+[255]*3+[202]+[2]+[0]+[83]+[252]+[255]+[255]+[239]+[205]+[229]+[255]*4+[252]+[122]+[251]+[252]+[86]+[0]+[0]+[209]+[255]*34+[243]+[8]+[0]+[41]+[233]+[255]*3+[211]+[137]+[251]+[255]+[233]+[219]+[239]+[255]+[255]+[252]+[81]+[0]+[2]+[199]+[255]*3+[212]+[5]+[0]*12+[114]+[255]*3+[252]+[49]+[0]+[18]+[238]+[255]+[186]+[55]+[0]*3+[56]+[207]+[255]*3+[239]+[138]+[88]+[0]+[0]+[32]+[255]*36+[76]+[0]+[0]+[43]+[231]+[255]*3+[238]+[122]+[66]+[0]*3+[56]+[188]+[255]+[237]+[16]+[0]+[51]+[252]+[255]*3+[112]+[0]*11+[8]+[230]+[255]*3+[145]+[0]+[0]+[128]+[255]+[168]+[2]+[0]*4+[100]+[161]+[205]+[255]*3+[188]+[0]*3+[114]+[255]*36+[157]+[0]*3+[105]+[206]+[255]*4+[162]+[10]+[0]*3+[3]+[171]+[255]+[126]+[0]+[0]+[148]+[255]*3+[231]+[9]+[0]*10+[108]+[255]*3+[237]+[15]+[0]+[0]+[195]+[236]+[20]+[0]*3+[15]+[169]+[255]+[255]+[128]+[252]+[255]*3+[61]+[0]+[0]+[188]+[255]*36+[231]+[0]+[0]+[64]+[255]+[164]+[185]+[255]*4+[211]+[38]+[0]*3+[21]+[237]+[193]+[0]+[0]+[16]+[239]+[255]*3+[106]+[0]*9+[5]+[224]+[255]*3+[127]+[0]*3+[226]+[147]+[0]*3+[36]+[214]+[255]*3+[223]+[179]+[255]*3+[134]+[0]+[0]+[230]+[255]*37+[17]+[0]+[137]+[255]+[255]+[195]+[155]+[255]*4+[242]+[78]+[0]*3+[150]+[225]+[0]*3+[131]+[255]*3+[225]+[6]+[0]*8+[97]+[255]*3+[242]+[18]+[0]*3+[225]+[65]+[0]+[0]+[61]+[238]+[255]*4+[166]+[99]+[255]*3+[177]+[0]+[0]+[153]+[255]*36+[195]+[0]+[0]+[181]+[255]*3+[92]+[113]+[252]+[255]*3+[252]+[111]+[0]+[0]+[67]+[225]+[0]*3+[19]+[243]+[255]*3+[96]+[0]*8+[199]+[255]*3+[137]+[0]*4+[108]+[2]+[0]+[55]+[242]+[255]*4+[134]+[1]+[47]+[255]*3+[204]+[0]+[0]+[9]+[205]+[255]*34+[236]+[32]+[0]+[0]+[208]+[255]*3+[37]+[0]+[78]+[244]+[255]*4+[107]+[0]+[2]+[108]+[0]*4+[140]+[255]*3+[197]+[0]*7+[38]+[255]*3+[251]+[26]+[0]*6+[45]+[238]+[255]*3+[249]+[96]+[0]+[0]+[26]+[255]*3+[214]+[0]*3+[32]+[236]+[255]*32+[252]+[70]+[0]*3+[217]+[255]*3+[8]+[0]+[0]+[50]+[229]+[255]*3+[254]+[92]+[0]*6+[27]+[252]+[255]+[255]+[254]+[37]+[0]*6+[131]+[255]*3+[181]+[0]*6+[24]+[224]+[255]*3+[240]+[64]+[0]*3+[7]+[255]*3+[211]+[0]*4+[69]+[252]+[255]*31+[115]+[0]*4+[214]+[255]*3+[1]+[0]*3+[28]+[211]+[255]*3+[249]+[62]+[0]*6+[185]+[255]*3+[129]+[0]*5+[1]+[222]+[255]*3+[88]+[0]*5+[6]+[200]+[255]*3+[239]+[54]+[0]*4+[24]+[255]*3+[194]+[0]*5+[127]+[255]*30+[174]+[1]+[0]*4+[198]+[255]*3+[9]+[0]*4+[21]+[210]+[255]*3+[236]+[32]+[0]*5+[92]+[255]*3+[223]+[1]+[0]*4+[56]+[255]*3+[241]+[9]+[0]*5+[140]+[255]*3+[245]+[54]+[0]*5+[55]+[255]*3+[162]+[0]*5+[1]+[173]+[255]*28+[215]+[14]+[0]*5+[168]+[255]*3+[38]+[0]*5+[21]+[218]+[255]*3+[197]+[4]+[0]*4+[11]+[242]+[255]*3+[54]+[0]*4+[123]+[255]*3+[160]+[0]*5+[68]+[254]+[255]*3+[89]+[0]*6+[89]+[255]*3+[122]+[0]+[1]+[112]+[1]+[0]+[0]+[22]+[214]+[255]*26+[242]+[50]+[0]*3+[111]+[14]+[0]+[126]+[255]*3+[84]+[0]*6+[42]+[242]+[255]*3+[127]+[0]*5+[164]+[255]*3+[121]+[0]*4+[189]+[255]*3+[91]+[0]*4+[5]+[216]+[255]*3+[141]+[0]*7+[151]+[255]*3+[63]+[1]+[146]+[255]+[11]+[0]+[0]+[156]+[74]+[241]+[255]*24+[254]+[93]+[184]+[0]*3+[209]+[199]+[15]+[67]+[255]*3+[145]+[0]*7+[80]+[255]*3+[248]+[34]+[0]*4+[95]+[255]*3+[186]+[0]*3+[7]+[246]+[255]*3+[27]+[0]*4+[110]+[255]*3+[215]+[7]+[0]*6+[3]+[228]+[255]+[255]+[242]+[6]+[108]+[255]+[255]+[40]+[0]+[0]+[88]+[221]+[95]+[254]+[255]*23+[130]+[188]+[140]+[0]+[0]+[3]+[238]+[255]+[171]+[7]+[244]+[255]+[255]+[226]+[2]+[0]*7+[163]+[255]*3+[172]+[0]*4+[29]+[255]*3+[245]+[7]+[0]+[0]+[65]+[255]*3+[215]+[0]*4+[4]+[230]+[255]*3+[78]+[0]*7+[72]+[255]*3+[161]+[32]+[244]+[255]+[255]+[123]+[0]+[0]+[26]+[255]+[198]+[145]+[255]*22+[186]+[158]+[255]+[79]+[0]+[0]+[72]+[255]*3+[83]+[164]+[255]*3+[68]+[0]*7+[28]+[247]+[255]+[255]+[254]+[38]+[0]*4+[218]+[255]*3+[62]+[0]+[0]+[118]+[255]*3+[150]+[0]*4+[74]+[255]*3+[212]+[1]+[0]*7+[192]+[255]*3+[56]+[147]+[255]*3+[242]+[47]+[0]+[2]+[252]+[255]+[170]+[186]+[255]*20+[223]+[132]+[255]+[255]+[53]+[0]+[24]+[219]+[255]*3+[206]+[59]+[255]*3+[186]+[0]*8+[156]+[255]*3+[131]+[0]*4+[152]+[255]*3+[116]+[0]+[0]+[157]+[255]*3+[101]+[0]*4+[139]+[255]*3+[119]+[0]*7+[72]+[255]*3+[194]+[10]+[243]+[255]*4+[244]+[150]+[129]+[255]*3+[130]+[227]+[255]*18+[246]+[108]+[253]+[255]+[255]+[158]+[152]+[234]+[255]*5+[56]+[195]+[255]*3+[70]+[0]*7+[63]+[255]*3+[194]+[0]*4+[104]+[255]*3+[154]+[0]+[0]+[196]+[255]*3+[62]+[0]*4+[190]+[255]*3+[54]+[0]*6+[9]+[219]+[255]+[255]+[254]+[53]+[65]+[255]*7+[205]+[255]*3+[251]+[102]+[250]+[255]*17+[111]+[235]+[255]*3+[204]+[255]*7+[122]+[55]+[254]+[255]+[255]+[218]+[9]+[0]*6+[7]+[248]+[255]+[255]+[244]+[0]*4+[65]+[255]*3+[193]+[0]+[0]+[235]+[255]*3+[23]+[0]*4+[217]+[255]*3+[13]+[0]*6+[145]+[255]*3+[164]+[0]+[117]+[255]*6+[246]+[220]+[255]*4+[217]+[117]+[255]*16+[153]+[176]+[255]*4+[242]+[225]+[255]*6+[175]+[0]+[167]+[255]*3+[143]+[0]*7+[215]+[255]*3+[16]+[0]*3+[26]+[255]*3+[232]+[0]+[18]+[255]*3+[240]+[0]*5+[232]+[255]*3+[2]+[0]*5+[66]+[254]+[255]+[255]+[247]+[31]+[0]+[134]+[255]*11+[230]+[228]+[125]+[172]+[255]*14+[209]+[81]+[233]+[207]+[255]*11+[191]+[0]+[32]+[247]+[255]+[255]+[254]+[71]+[0]*6+[205]+[255]*3+[31]+[0]*3+[1]+[242]+[255]*3+[16]+[42]+[255]*3+[207]+[0]*5+[226]+[255]*3+[23]+[0]*4+[15]+[225]+[255]*3+[131]+[0]+[0]+[138]+[255]*10+[243]+[12]+[2]+[110]+[18]+[216]+[255]*12+[242]+[39]+[97]+[9]+[0]+[196]+[255]*10+[192]+[0]+[0]+[134]+[255]*3+[224]+[14]+[0]*5+[227]+[255]*3+[25]+[0]*4+[209]+[255]*3+[40]+[55]+[255]*3+[192]+[0]*5+[207]+[255]*3+[68]+[0]*4+[160]+[255]*3+[214]+[8]+[0]+[0]+[101]+[255]*10+[252]+[30]+[0]*3+[46]+[246]+[255]*10+[254]+[83]+[0]*3+[6]+[224]+[255]*10+[155]+[0]+[0]+[7]+[212]+[255]*3+[158]+[0]*4+[20]+[253]+[255]+[255]+[252]+[9]+[0]*4+[195]+[255]*3+[53]+[68]+[255]*3+[180]+[0]*5+[165]+[255]*3+[150]+[0]*3+[79]+[255]*3+[251]+[53]+[0]*3+[46]+[255]*11+[169]+[0]*4+[103]+[255]*10+[144]+[0]*4+[120]+[255]*11+[98]+[0]*3+[50]+[250]+[255]*3+[83]+[0]*3+[101]+[255]*3+[219]+[0]*5+[182]+[255]*3+[66]+[81]+[255]*3+[167]+[0]*5+[108]+[255]*3+[242]+[23]+[0]+[20]+[232]+[255]*3+[132]+[0]*5+[198]+[255]*3+[214]+[142]+[163]+[250]+[255]*4+[122]+[0]*4+[172]+[255]*8+[213]+[9]+[0]*3+[81]+[250]+[255]*4+[176]+[126]+[185]+[255]*3+[238]+[11]+[0]*4+[135]+[255]*3+[234]+[23]+[0]+[3]+[214]+[255]*3+[160]+[0]*5+[169]+[255]*3+[79]+[94]+[255]*3+[154]+[0]*5+[25]+[252]+[255]*3+[156]+[0]+[171]+[255]*3+[214]+[8]+[0]*5+[60]+[250]+[255]+[155]+[5]+[0]+[0]+[88]+[255]*5+[152]+[13]+[0]+[0]+[21]+[232]+[255]*6+[249]+[47]+[0]+[0]+[5]+[120]+[252]+[255]*4+[137]+[0]*3+[94]+[254]+[255]+[104]+[0]*5+[7]+[211]+[255]*3+[176]+[0]+[107]+[255]*4+[74]+[0]*5+[155]+[255]*3+[93]+[94]+[255]*3+[154]+[0]*6+[166]+[255]*4+[119]+[217]+[255]+[255]+[251]+[52]+[0]*7+[106]+[217]+[5]+[0]*3+[51]+[255]*6+[232]+[120]+[19]+[0]+[84]+[255]*6+[126]+[0]+[11]+[105]+[219]+[255]*6+[108]+[0]*4+[162]+[152]+[0]*7+[53]+[252]+[255]*3+[101]+[243]+[255]*3+[212]+[4]+[0]*5+[155]+[255]*3+[93]+[81]+[255]*3+[167]+[0]*6+[29]+[244]+[255]*4+[157]+[152]+[245]+[133]+[0]*9+[9]+[0]*3+[26]+[209])/255.0


# In[29]:


cthulhu2 = np.array([255]*8+[245]+[150]+[29]+[177]+[255]*4+[217]+[23]+[133]+[237]+[255]*8+[240]+[61]+[0]*3+[5]+[0]*9+[147]+[255]*3+[240]+[144]+[255]+[255]+[254]+[69]+[0]*6+[169]+[255]*3+[79]+[68]+[255]*3+[180]+[0]*7+[74]+[250]+[255]*4+[240]+[149]+[65]+[3]+[0]*10+[23]+[116]+[238]+[255]*11+[240]+[81]+[244]+[198]+[161]+[254]+[85]+[222]+[255]*11+[253]+[152]+[43]+[0]*11+[47]+[229]+[255]*3+[178]+[219]+[255]+[116]+[0]*7+[182]+[255]*3+[66]+[55]+[255]*3+[192]+[0]*8+[81]+[243]+[255]*6+[235]+[182]+[144]+[109]+[93]+[85]+[85]+[100]+[122]+[157]+[208]+[251]+[255]*14+[220]+[72]+[29]+[12]+[84]+[176]+[255]*15+[226]+[180]+[149]+[118]+[98]+[85]+[97]+[114]+[140]+[180]+[231]+[255]+[128]+[255]*4+[124]+[119]+[0]*8+[195]+[255]*3+[53]+[42]+[255]*3+[207]+[0]*9+[106]+[204]+[255]*32+[83]+[0]+[0]+[33]+[253]+[255]*27+[198]+[216]+[255]*3+[188]+[0]*9+[209]+[255]*3+[40]+[17]+[255]*3+[240]+[0]*8+[31]+[254]+[191]+[134]+[231]+[255]*30+[163]+[0]+[0]+[111]+[255]*29+[140]+[255]*3+[254]+[36]+[0]*7+[1]+[242]+[255]*3+[15]+[0]+[234]+[255]*3+[24]+[0]*7+[121]+[255]+[255]+[253]+[149]+[101]+[190]+[252]+[255]*27+[233]+[2]+[0]+[181]+[255]*27+[252]+[193]+[81]+[237]+[255]*3+[116]+[0]*7+[27]+[255]*3+[231]+[0]+[0]+[195]+[255]*3+[63]+[0]*7+[186]+[255]*3+[161]+[0]+[0]+[22]+[80]+[129]+[176]+[199]+[220]+[238]+[238]+[238]+[238]+[225]+[203]+[178]+[151]+[126]+[103]+[93]+[102]+[120]+[139]+[178]+[251]+[255]*7+[50]+[7]+[245]+[255]*6+[254]+[188]+[139]+[119]+[91]+[80]+[90]+[111]+[134]+[159]+[181]+[205]+[229]+[227]+[221]+[221]+[221]+[200]+[162]+[124]+[85]+[20]+[0]+[0]+[161]+[255]*3+[189]+[0]*7+[66]+[255]*3+[192]+[0]+[0]+[156]+[255]*3+[102]+[0]*7+[233]+[255]*3+[88]+[0]*16+[186]+[255]*5+[239]+[127]+[206]+[255]*6+[115]+[60]+[255]*6+[226]+[121]+[229]+[255]*5+[223]+[1]+[0]*15+[96]+[255]*3+[239]+[1]+[0]*6+[105]+[255]*3+[153]+[0]+[0]+[117]+[255]*3+[151]+[0]*6+[26]+[255]*4+[19]+[0]*5+[6]+[17]+[0]*8+[63]+[251]+[255]*7+[121]+[215]+[255]*5+[155]+[101]+[255]*5+[239]+[109]+[252]+[255]*7+[93]+[0]*7+[2]+[25]+[15]+[0]*5+[30]+[255]*4+[28]+[0]*6+[153]+[255]*3+[115]+[0]+[0]+[64]+[255]*3+[215]+[0]*6+[57]+[255]*3+[239]+[0]*3+[5]+[131]+[219]+[254]+[255]+[248]+[186]+[76]+[0]*4+[96]+[249]+[255]*6+[246]+[205]+[145]+[66]+[255]*5+[180]+[126]+[255]*5+[116]+[122]+[200]+[242]+[255]*6+[254]+[126]+[1]+[0]*3+[57]+[178]+[246]+[255]+[255]+[240]+[166]+[25]+[0]*3+[234]+[255]*3+[48]+[0]*6+[219]+[255]*3+[61]+[0]+[0]+[7]+[246]+[255]*3+[27]+[0]*5+[69]+[255]*3+[216]+[0]*3+[118]+[255]*7+[161]+[4]+[2]+[135]+[255]*6+[206]+[80]+[69]+[161]+[185]+[77]+[220]+[255]*4+[189]+[135]+[255]*4+[250]+[93]+[185]+[160]+[65]+[68]+[191]+[255]*6+[166]+[8]+[0]+[123]+[254]+[255]*6+[179]+[0]*3+[208]+[255]*3+[61]+[0]*5+[29]+[255]*3+[245]+[7]+[0]*3+[188]+[255]*3+[92]+[0]*5+[63]+[255]*3+[196]+[0]*3+[205]+[255]+[252]+[209]+[230]+[255]*4+[114]+[172]+[255]*5+[253]+[132]+[5]+[4]+[252]+[255]+[255]+[214]+[134]+[255]*4+[190]+[134]+[255]*4+[184]+[216]+[255]+[255]+[252]+[2]+[1]+[110]+[248]+[255]*5+[188]+[103]+[255]*4+[234]+[204]+[242]+[255]+[251]+[13]+[0]+[0]+[200]+[255]*3+[60]+[0]*5+[96]+[255]*3+[185]+[0]*4+[122]+[255]*3+[161]+[0]*5+[48]+[255]*3+[206]+[0]*3+[234]+[255]+[104]+[0]+[0]+[77]+[229]+[255]+[163]+[203]+[255]*5+[228]+[65]+[0]+[0]+[2]+[255]*4+[109]+[255]*4+[176]+[121]+[255]*4+[156]+[255]*3+[249]+[0]*3+[43]+[212]+[255]*4+[134]+[244]+[255]+[255]+[245]+[103]+[2]+[0]+[44]+[253]+[255]+[34]+[0]+[0]+[211]+[255]*3+[36]+[0]*5+[165]+[255]*3+[120]+[0]*4+[55]+[255]*3+[241]+[10]+[0]*4+[13]+[254]+[255]+[255]+[238]+[0]*3+[223]+[254]+[18]+[0]*3+[40]+[138]+[203]+[255]*5+[185]+[24]+[0]*3+[21]+[255]*4+[113]+[255]*4+[161]+[106]+[255]*4+[150]+[255]*4+[13]+[0]*3+[13]+[163]+[255]+[255]+[214]+[187]+[255]+[255]+[253]+[81]+[0]*4+[213]+[255]+[22]+[0]+[2]+[242]+[255]+[255]+[254]+[11]+[0]*4+[11]+[243]+[255]*3+[53]+[0]*4+[1]+[221]+[255]*3+[89]+[0]*5+[224]+[255]*3+[40]+[0]+[0]+[138]+[223]+[0]*4+[22]+[211]+[255]*4+[247]+[109]+[1]+[0]*4+[80]+[255]*4+[115]+[255]*4+[147]+[92]+[255]*4+[149]+[255]*4+[73]+[0]*5+[96]+[244]+[130]+[255]*3+[161]+[39]+[0]*4+[163]+[187]+[0]+[0]+[45]+[255]*3+[214]+[0]*5+[93]+[255]*3+[223]+[1]+[0]*5+[130]+[255]*3+[182]+[0]*5+[158]+[255]*3+[124]+[0]+[0]+[4]+[40]+[0]*3+[14]+[209]+[255]*4+[232]+[56]+[0]*6+[185]+[255]*4+[123]+[255]*4+[132]+[77]+[255]*4+[153]+[255]*4+[179]+[0]*6+[40]+[165]+[255]+[255]+[251]+[133]+[227]+[29]+[0]*3+[21]+[12]+[0]+[0]+[130]+[255]*3+[149]+[0]*5+[186]+[255]*3+[128]+[0]*6+[37]+[254]+[255]+[255]+[252]+[26]+[0]*4+[84]+[255]*3+[231]+[11]+[0]*5+[5]+[185]+[255]*4+[210]+[142]+[45]+[0]*5+[83]+[255]*5+[101]+[253]+[255]*3+[110]+[54]+[255]*4+[135]+[255]*5+[91]+[0]*5+[5]+[240]+[255]+[255]+[189]+[223]+[255]+[211]+[15]+[0]*5+[13]+[235]+[255]*3+[81]+[0]*4+[28]+[252]+[255]+[255]+[254]+[36]+[0]*7+[198]+[255]*3+[138]+[0]*4+[8]+[236]+[255]*3+[147]+[0]*5+[155]+[255]*4+[211]+[158]+[255]+[118]+[0]*4+[50]+[240]+[255]*4+[208]+[17]+[253]+[255]*3+[77]+[21]+[255]*4+[68]+[209]+[255]*4+[242]+[49]+[0]*4+[65]+[255]*3+[138]+[255]*3+[186]+[2]+[0]*4+[151]+[255]*3+[227]+[5]+[0]*4+[142]+[255]*3+[196]+[0]*8+[95]+[255]*3+[243]+[19]+[0]*4+[124]+[255]*4+[106]+[0]*3+[73]+[255]*4+[220]+[154]+[255]+[255]+[177]+[0]*3+[69]+[238]+[255]*4+[171]+[11]+[0]+[254]+[255]*3+[43]+[0]+[242]+[255]*3+[56]+[11]+[172]+[255]*4+[242]+[73]+[0]*3+[126]+[255]*3+[132]+[255]*4+[109]+[0]*3+[111]+[255]*4+[116]+[0]*4+[20]+[244]+[255]*3+[94]+[0]*8+[5]+[223]+[255]*3+[129]+[0]*4+[10]+[218]+[255]*4+[140]+[8]+[12]+[225]+[255]*3+[244]+[39]+[207]+[255]+[255]+[226]+[0]+[14]+[144]+[252]+[255]*3+[254]+[127]+[1]+[0]+[8]+[255]*3+[254]+[11]+[0]+[209]+[255]*3+[65]+[0]+[1]+[129]+[254]+[255]*3+[252]+[149]+[19]+[0]+[170]+[255]+[255]+[254]+[32]+[229]+[255]*3+[244]+[32]+[9]+[144]+[255]*4+[207]+[7]+[0]*4+[133]+[255]*3+[225]+[5]+[0]*9+[106]+[255]*3+[238]+[16]+[0]*4+[50]+[244]+[255]*4+[220]+[119]+[150]+[225]+[255]+[255]+[97]+[0]+[171]+[255]+[255]+[253]+[89]+[237]+[255]*4+[250]+[95]+[0]*3+[40]+[255]*3+[231]+[0]+[0]+[176]+[255]*3+[93]+[0]*3+[97]+[250]+[255]*4+[237]+[136]+[127]+[203]+[253]+[229]+[0]+[63]+[252]+[255]*3+[150]+[202]+[255]*4+[244]+[40]+[0]*4+[18]+[240]+[255]*3+[104]+[0]*10+[7]+[229]+[255]*3+[147]+[0]*5+[69]+[244]+[255]*5+[246]+[186]+[143]+[132]+[85]+[85]+[147]+[255]*3+[142]+[255]*4+[242]+[70]+[0]*4+[74]+[255]*3+[194]+[0]+[0]+[138]+[255]*3+[122]+[0]*4+[71]+[243]+[255]*5+[254]+[209]+[159]+[130]+[90]+[85]+[153]+[255]*3+[245]+[154]+[255]*3+[244]+[67]+[0]*5+[151]+[255]*3+[230]+[8]+[0]*11+[112]+[255]*3+[252]+[51]+[0]*5+[48]+[219]+[255]*10+[145]+[255]*3+[141]+[255]*3+[223]+[49]+[0]*5+[109]+[255]*3+[146]+[0]+[0]+[90]+[255]*3+[154]+[0]*5+[50]+[225]+[255]*10+[137]+[248]+[255]*3+[131]+[255]+[255]+[219]+[47]+[0]*5+[53]+[253]+[255]*3+[109]+[0]*12+[4]+[209]+[255]*3+[203]+[2]+[0]*5+[9]+[133]+[243]+[255]*8+[137]+[255]*3+[139]+[255]+[255]+[155]+[13]+[0]*6+[144]+[255]*3+[98]+[0]+[0]+[41]+[255]*3+[199]+[0]*6+[17]+[165]+[254]+[255]*8+[230]+[177]+[255]*3+[155]+[242]+[132]+[9]+[0]*5+[2]+[200]+[255]*3+[211]+[5]+[0]*13+[62]+[254]+[255]*3+[110]+[0]*7+[19]+[128]+[215]+[255]*6+[134]+[255]*3+[133]+[162]+[49]+[0]*4+[9]+[43]+[49]+[11]+[192]+[255]*3+[50]+[0]+[0]+[3]+[245]+[255]+[255]+[223]+[15]+[56]+[48]+[14]+[0]*4+[53]+[170]+[248]+[255]*7+[145]+[255]*3+[178]+[16]+[0]*7+[112]+[255]*3+[254]+[60]+[0]*15+[162]+[255]*3+[248]+[50]+[0]*7+[150]+[197]+[138]+[131]+[151]+[183]+[203]+[204]+[119]+[255]*3+[111]+[0]*4+[72]+[175]+[247]+[255]+[255]+[168]+[248]+[255]+[255]+[249]+[7]+[0]*3+[199]+[255]+[156]+[155]+[250]+[255]+[255]+[251]+[186]+[87]+[2]+[0]*3+[42]+[134]+[148]+[176]+[198]+[204]+[193]+[179]+[149]+[255]*3+[183]+[0]*7+[52]+[248]+[255]*3+[158]+[0]*16+[23]+[236]+[255]*3+[219]+[14]+[0]*6+[130]+[255]*3+[227]+[4]+[0]+[0]+[59]+[255]*3+[134]+[0]+[0]+[54]+[195]+[255]*5+[137]+[255]*3+[191]+[0]*4+[137]+[134]+[223]+[255]*7+[209]+[70]+[0]+[0]+[80]+[255]+[255]+[241]+[94]+[0]*3+[196]+[255]*3+[154]+[0]*6+[15]+[221]+[255]*3+[235]+[21]+[0]*17+[76]+[254]+[255]*3+[169]+[0]*6+[72]+[255]*4+[91]+[0]+[0]+[31]+[255]*3+[162]+[6]+[132]+[251]+[255]*5+[246]+[154]+[255]*3+[117]+[0]*4+[55]+[235]+[255]*9+[254]+[153]+[13]+[109]+[255]*3+[75]+[0]+[0]+[60]+[255]*4+[104]+[0]*5+[1]+[172]+[255]*3+[254]+[74]+[0]*19+[140]+[255]*4+[130]+[0]*5+[9]+[236]+[255]*3+[241]+[41]+[0]+[2]+[248]+[255]+[227]+[120]+[208]+[255]*4+[254]+[196]+[155]+[124]+[241]+[255]*3+[126]+[51]+[0]+[0]+[31]+[235]+[255]*3+[249]+[188]+[153]+[187]+[252]+[255]*4+[212]+[165]+[255]*3+[37]+[0]+[22]+[226]+[255]*3+[249]+[22]+[0]*5+[133]+[255]*4+[136]+[0]*20+[4]+[196]+[255]*4+[111]+[0]*5+[133]+[255]*4+[231]+[86]+[12]+[132]+[137]+[170]+[252]+[255]*4+[206]+[50]+[0]+[0]+[120]+[255]*3+[208]+[197]+[190]+[0]+[0]+[163]+[255]*3+[178]+[145]+[161]+[0]+[0]+[37]+[186]+[255]*3+[175]+[233]+[255]+[255]+[251]+[16]+[75]+[220]+[255]*4+[162]+[0]*5+[105]+[255]*4+[200]+[5]+[0]*21+[30]+[230]+[255]*3+[253]+[85]+[0]*4+[13]+[229]+[255]*12+[254]+[144]+[7]+[0]+[0]+[40]+[242]+[255]*3+[128]+[255]+[252]+[19]+[8]+[247]+[255]+[255]+[155]+[212]+[255]+[255]+[94]+[0]+[0]+[2]+[118]+[249]+[255]+[127]+[255]*3+[212]+[228]+[255]*5+[244]+[28]+[0]*4+[87]+[253]+[255]*3+[229]+[29]+[0]*23+[47]+[240]+[255]*3+[249]+[85]+[0]*4+[64]+[249]+[255]*10+[223]+[70]+[0]*3+[46]+[231]+[255]*3+[236]+[139]+[255]+[255]+[76]+[44]+[255]+[255]+[213]+[156]+[255]*3+[253]+[104]+[0]*3+[49]+[148]+[203]+[255]*3+[164]+[255]*6+[98]+[0]*4+[87]+[249]+[255]*3+[239]+[46]+[0]*25+[63]+[247]+[255]*3+[254]+[114]+[0]*4+[84]+[242]+[255]*7+[219]+[133]+[135]+[14]+[0]+[4]+[105]+[244]+[255]*4+[108]+[69]+[255]+[255]+[85]+[54]+[255]+[255]+[99]+[47]+[252]+[255]*4+[158]+[27]+[0]+[9]+[155]+[255]*4+[166]+[255]*4+[252]+[113]+[0]*4+[107]+[253]+[255]*3+[247]+[62]+[0]*27+[82]+[252]+[255]*4+[130]+[0]*4+[33]+[152]+[227]+[255]+[249]+[216]+[156]+[135]+[185]+[254]+[255]+[238]+[193]+[234]+[255]*5+[207]+[4]+[14]+[255]+[255]+[79]+[49]+[255]+[255]+[42]+[0]+[148]+[255]*5+[248]+[207]+[239]+[255]*4+[195]+[184]+[241]+[255]+[237]+[168]+[51]+[0]*3+[1]+[132]+[255]*4+[252]+[81]+[0]*29+[82]+[247]+[255]*4+[170]+[14]+[0]*6+[4]+[0]+[21]+[230]+[255]*10+[237]+[36]+[0]+[20]+[255]+[255]+[28]+[5]+[246]+[255]+[47]+[0]+[7]+[196]+[255]*10+[251]+[46]+[0]+[0]+[3]+[0]*5+[15]+[172]+[255]*4+[247]+[81]+[0]*31+[63]+[240]+[255]*4+[224]+[55]+[0]*8+[57]+[239]+[255]*8+[222]+[47]+[0]+[0]+[99]+[255]+[221]+[0]+[0]+[189]+[255]+[126]+[0]+[0]+[16]+[182]+[255]*8+[251]+[87]+[0]*8+[50]+[219]+[255]*4+[239]+[62]+[0]*33+[47]+[230]+[255]*4+[248]+[110]+[2]+[0]*7+[38]+[186]+[255]*5+[247]+[142]+[17]+[0]+[0]+[20]+[230]+[255]+[122]+[0]+[0]+[90]+[255]+[244]+[38]+[0]+[0]+[2]+[102]+[227]+[255]*5+[204]+[60]+[0]*7+[2]+[112]+[248]+[255]*4+[229]+[46]+[0]*35+[30]+[198]+[255]*5+[201]+[51]+[0]*8+[37]+[103]+[130]+[123]+[82]+[14]+[0]*3+[20]+[203]+[255]+[220]+[13]+[0]+[0]+[3]+[198]+[255]+[220]+[34]+[0]*3+[3]+[58]+[107]+[120]+[101]+[45]+[0]*8+[53]+[203]+[255]*5+[196]+[29]+[0]*37+[5]+[139]+[254]+[255]*4+[252]+[148]+[16]+[0]*14+[26]+[217]+[254]+[195]+[34]+[0]*4+[19]+[181]+[252]+[231]+[43]+[0]*14+[18]+[151]+[253]+[255]*4+[254]+[136]+[4]+[0]*40+[76]+[236]+[255]*5+[239]+[130]+[20]+[0]*12+[3]+[28]+[12]+[0]*8+[10]+[28]+[5]+[0]*12+[20]+[132]+[240]+[255]*5+[235]+[74]+[0]*43+[22]+[161]+[254]+[255]*5+[243]+[140]+[27]+[0]*34+[28]+[141]+[244]+[255]*5+[254]+[159]+[21]+[0]*46+[61]+[211]+[255]*6+[253]+[196]+[102]+[12]+[0]*28+[8]+[85]+[179]+[251]+[255]*6+[216]+[67]+[0]*49+[5]+[112]+[230]+[255]*7+[242]+[164]+[94]+[28]+[0]*22+[29]+[95]+[165]+[243]+[255]*7+[228]+[109]+[4]+[0]*52+[8]+[106]+[224]+[255]*9+[218]+[152]+[104]+[65]+[26]+[0]*12+[27]+[66]+[104]+[153]+[219]+[255]*9+[222]+[104]+[7]+[0]*56+[5]+[96]+[197]+[255]*12+[242]+[209]+[194]+[182]+[170]+[153]+[153]+[170]+[182]+[194]+[209]+[242]+[255]*11+[254]+[196]+[94]+[5]+[0]*61+[30]+[119]+[219]+[255]*30+[226]+[136]+[37]+[0]*66+[1]+[54]+[121]+[186]+[245]+[255]*22+[245]+[186]+[120]+[53]+[1]+[0]*72+[7]+[62]+[116]+[154]+[192]+[233]+[255]*12+[232]+[192]+[154]+[116]+[61]+[6]+[0]*82+[16]+[40]+[52]+[68]+[81]+[93]+[93]+[81]+[68]+[52]+[39]+[15]+[0]*44)/255.0


# In[30]:


raven1 = np.array([0]*200+[0]*200+[0]*77+[16]+[100]+[179]+[221]+[238]+[226]+[193]+[160]+[125]+[73]+[12]+[0]*87+[20]+[142]+[247]+[255]*9+[245]+[164]+[44]+[0]*83+[1]+[95]+[238]+[255]*13+[253]+[143]+[7]+[0]*74+[21]+[57]+[95]+[129]+[136]+[150]+[155]+[201]+[255]*17+[197]+[15]+[0]*69+[1]+[77]+[162]+[233]+[255]*26+[200]+[9]+[0]*67+[48]+[208]+[255]*30+[158]+[0]*66+[41]+[243]+[255]*32+[88]+[0]*65+[65]+[253]+[255]*32+[232]+[18]+[0]*65+[38]+[105]+[157]+[208]+[251]+[255]*29+[159]+[0]*69+[9]+[59]+[130]+[204]+[254]+[255]*25+[254]+[66]+[0]*72+[23]+[98]+[179]+[250]+[255]*23+[207]+[2]+[0]*74+[20]+[149]+[253]+[255]*22+[90]+[0]*76+[79]+[252]+[255]*21+[210]+[1]+[0]*76+[144]+[255]*22+[69]+[0]*76+[28]+[253]+[255]*21+[162]+[0]*77+[204]+[255]*21+[242]+[6]+[0]*76+[176]+[255]*22+[55]+[0]*75+[5]+[229]+[255]*22+[108]+[0]*75+[125]+[255]*23+[132]+[0]*74+[90]+[252]+[255]*23+[146]+[0]*73+[112]+[252]+[255]*24+[126]+[0]*71+[11]+[166]+[255]*26+[100]+[0]*70+[46]+[211]+[255]*27+[68]+[0]*68+[3]+[120]+[248]+[255]*28+[47]+[0]*67+[37]+[197]+[255]*30+[47]+[0]*65+[1]+[102]+[243]+[255]*31+[62]+[0]*64+[20]+[182]+[255]*33+[89]+[0]*63+[60]+[229]+[255]*34+[114]+[0]*62+[95]+[248]+[255]*35+[140]+[0]*60+[1]+[128]+[255]*37+[164]+[0]*59+[8]+[166]+[255]*38+[188]+[0]*58+[26]+[203]+[255]*39+[213]+[0]*57+[47]+[225]+[255]*40+[237]+[0]*56+[82]+[245]+[255]*41+[248]+[0]*55+[120]+[253]+[255]*42+[244]+[0]*53+[3]+[151]+[255]*44+[233]+[0]*52+[8]+[171]+[255]*45+[198]+[0]*51+[20]+[199]+[255]*46+[160]+[0]*50+[33]+[216]+[255]*47+[96]+[0]*49+[46]+[228]+[255]*47+[253]+[24]+[0]*48+[72]+[243]+[255]*48+[200]+[0]*47+[1]+[111]+[250]+[255]*49+[107]+[0]*46+[14]+[172]+[255]*50+[249]+[20]+[0]*45+[51]+[221]+[255]*51+[170]+[0]*44+[1]+[109]+[249]+[255]*52+[67]+[0]*43+[22]+[178]+[255]*53+[210]+[1]+[0]*42+[82]+[235]+[255]*54+[94]+[0]*41+[14]+[153]+[255]*55+[229]+[6]+[0]*40+[64]+[224]+[255]*56+[117]+[0]*39+[7]+[136]+[252]+[255]*56+[231]+[10]+[0]*38+[47]+[210]+[255]*58+[108]+[0]*37+[4]+[119]+[248]+[255]*58+[225]+[7]+[0]*36+[51]+[209]+[255]*60+[99]+[0]*35+[5]+[119]+[249]+[255]*60+[204]+[3]+[0]*34+[56]+[214]+[255]*61+[253]+[55]+[0]*33+[9]+[136]+[252]+[255]*62+[147]+[0]*33+[59]+[216]+[255]*63+[232]+[16]+[0]*31+[18]+[154]+[254]+[255]*64+[82]+[0]*30+[1]+[94]+[233]+[255]*65+[149]+[0]*30+[48]+[193]+[255]*66+[209]+[8]+[0]*28+[27]+[153]+[252]+[255]*66+[244]+[41]+[0]*27+[12]+[131]+[245]+[255]*67+[254]+[91]+[0]*26+[2]+[94]+[231]+[255]*69+[124]+[0]*26+[61]+[212]+[255]*70+[146]+[0]*25+[6]+[135]+[252]+[255]*70+[145]+[1]+[0]*24+[41]+[205]+[255]*70+[254]+[128]+[0]*25+[87]+[246]+[255]*70+[233]+[76]+[0]*25+[113]+[253]+[255]*69+[254]+[167]+[21]+[0]*24+[1]+[141]+[255]*70+[198]+[58]+[0]*26+[137]+[255]*42+[246]+[210]+[194]+[196]+[214]+[242]+[255]*21+[221]+[86]+[1]+[0]*26+[83]+[255]*38+[254]+[216]+[161]+[106]+[49]+[4]+[0]*4+[2]+[39]+[96]+[177]+[247]+[255]*16+[136]+[6]+[0]*27+[4]+[228]+[255]*31+[243]+[220]+[198]+[169]+[132]+[96]+[58]+[17]+[0]*13+[12]+[85]+[226]+[255]*14+[66]+[0]*28+[27]+[230]+[184]+[254]+[241]+[207]+[154]+[116]+[205]+[255]*20+[185]+[75]+[31]+[9]+[0]*23+[197]+[255]*14+[70]+[0]*31+[7]+[0]+[0]+[91]+[236]+[255]*18+[253]+[166]+[48]+[0]*27+[202]+[255]*7+[206]+[179]+[255]*5+[74]+[0]*32+[25]+[177]+[255]*18+[254]+[171]+[42]+[0]*29+[206]+[255]*4+[251]+[151]+[51]+[0]+[107]+[255]*5+[81]+[0]*31+[85]+[236]+[255]*18+[187]+[51]+[0]*31+[208]+[255]*4+[240]+[0]*3+[109]+[255]*5+[98]+[0]*29+[24]+[175]+[255]*18+[185]+[54]+[0]*33+[210]+[255]*4+[250]+[0]*3+[110]+[255]*5+[119]+[0]*28+[88]+[236]+[255]*17+[191]+[60]+[0]*35+[213]+[255]*5+[5]+[0]+[0]+[77]+[255]*5+[212]+[5]+[0]*25+[22]+[172]+[255]*17+[198]+[77]+[0]*37+[215]+[255]*5+[14]+[0]+[0]+[2]+[178]+[255]*5+[143]+[0]*24+[64]+[234]+[255]*16+[202]+[72]+[0]*39+[205]+[255]*5+[81]+[0]*3+[12]+[190]+[255]*5+[97]+[0]*22+[88]+[249]+[255]*15+[206]+[77]+[1]+[0]*40+[98]+[255]*5+[216]+[13]+[0]*3+[9]+[179]+[255]*4+[250]+[78]+[0]*11+[9]+[12]+[0]*8+[237]+[255]*14+[197]+[75]+[1]+[0]*43+[144]+[255]*5+[172]+[1]+[0]*3+[5]+[167]+[255]*4+[248]+[72]+[0]*6+[49]+[135]+[197]+[244]+[255]+[255]+[244]+[168]+[24]+[0]*5+[85]+[164]+[158]+[122]+[75]+[208]+[255]*6+[241]+[159]+[58]+[0]*47+[135]+[255]*5+[141]+[0]*4+[2]+[150]+[255]*4+[249]+[91]+[0]*3+[73]+[202]+[255]*4+[212]+[150]+[154]+[232]+[191]+[0]*9+[96]+[255]*4+[232]+[153]+[73]+[7]+[0]*50+[120]+[255]*5+[125]+[0]*5+[134]+[255]*4+[254]+[191]+[167]+[188]+[255]*4+[246]+[97]+[2]+[0]+[0]+[14]+[198]+[5]+[0]*8+[194]+[255]+[219]+[148]+[64]+[3]+[0]*54+[105]+[253]+[255]*4+[134]+[0]+[13]+[48]+[56]+[74]+[229]+[255]*11+[213]+[50]+[3]+[0]*3+[7]+[0]*9+[22]+[18]+[0]*59+[84]+[249]+[255]*4+[223]+[253]+[255]*17+[246]+[191]+[99]+[4]+[0]*72+[96]+[255]*26+[208]+[41]+[0]*66+[67]+[150]+[197]+[221]+[225]+[249]+[255]*16+[252]+[204]+[139]+[74]+[40]+[18]+[0]+[22]+[61]+[132]+[226]+[234]+[50]+[0]*63+[32]+[186]+[255]*22+[215]+[56]+[0]*8+[6]+[107]+[207]+[18]+[0]*61+[35]+[230]+[255]+[215]+[132]+[82]+[72]+[96]+[136]+[195]+[247]+[255]*6+[252]+[201]+[145]+[121]+[97]+[102]+[126]+[176]+[246]+[250]+[88]+[0]*9+[6]+[4]+[0]*60+[1]+[199]+[231]+[95]+[2]+[0]*6+[7]+[56]+[106]+[136]+[147]+[122]+[77]+[14]+[0]*7+[14]+[121]+[234]+[85]+[0]*70+[48]+[171]+[20]+[0]*25+[14]+[73]+[0]*200+[0]*200+[0]*13)/255.0


# In[31]:


frankenstein = np.concatenate((np.ones((100,100,1)), np.zeros((100,100,2)), frankenstein1.reshape(100,100,1)), axis=2)
cthulhu = np.concatenate((.6*np.ones((100,100,2)), .8*np.ones((100,100,1)), np.concatenate((cthulhu1, cthulhu2)).reshape(100,100,1)), axis=2)
raven = np.concatenate((np.zeros((100,100,2)), 0.5*np.ones((100,100,1)), raven1.reshape((100,100,1))), axis=2)


# In[32]:


auth_to_img = {'MWS': frankenstein, 'EAP': raven, 'HPL': cthulhu}
f, axes = plt.subplots(1,3,figsize=(15,5))
for i, auth in enumerate(authors):
    axes[i].imshow(auth_to_img[auth])
    axes[i].set_title(auth)
    axes[i].set_axis_off()


# In[33]:


def add_image(ax, arr_img, x, y, cmap='Greys'):
    xy = (x,y)
    imagebox = OffsetImage(arr_img, zoom=0.2, cmap=cmap)
    imagebox.image.axes = ax

    ab = AnnotationBbox(imagebox, xy,
                        xycoords='data',
                        pad=0,
                        frameon=False)

    ax.add_artist(ab)
    
print('Grammar distribution per author')
plt.rcParams['font.size'] = 20
    
f, axes = plt.subplots(len(tags), 1, figsize=(15, 1.2*len(tags)))
for j,i in enumerate(tags.index):
    axes[j].barh(0, tags.loc[i, 'Percentage'], color='lightcoral')
    axes[j].set_xlim([tags.loc[i, 'Percentage'] - max_diff*1.2, tags.loc[i, 'Percentage'] + max_diff*1.2])
    axes[j].set_yticks([0])
    axes[j].set_yticklabels([tags.loc[i, 'Explanation']])
    axes[j].set_xticklabels(['{:g}%'.format(x*100) for x in axes[j].get_xticks()])
    m, M = axes[j].get_ylim()
    add_image(axes[j], frankenstein, tags.loc[i, 'Percentage_MWS'], 0)
    add_image(axes[j], raven, tags.loc[i, 'Percentage_EAP'], 0)
    add_image(axes[j], cthulhu, tags.loc[i, 'Percentage_HPL'], 0)
plt.tight_layout()


# There are some interesting things to remark. Differences in prepositions, determiners, adjectives and comma are statistically very significant, so we shouldn't discard this information.

# ### 2.1.2. Groups of words.
# We'll just naively count all the groups of consecutive words by size. We'll name them [as twins are named](https://en.wikipedia.org/wiki/List_of_multiple_births) because they're so close... and since "Once Is Chance, Twice is Coincidence, Third Time Is A Pattern", we'll look especially closely at those that happen at least three times in the train set.
# 
# We'll work here with stemmed words, but without *really* throwing away punctuation. Also, we'll remove too rare words, as they will make the number of pairs, etc blow up unnecessarily.

# In[54]:


for df in [csv, test]:
    df['stemmed'] = df['split'].apply(lambda x: [stemmer.stem(y) for y in x])


# In[35]:


csv.iloc[0]


# In[36]:


vc = pd.Series([y for x in csv.stemmed for y in x]).value_counts()

words_to_remove = [w for w in vc.index if vc[w] <= 4]
print('This will reduce to {} words'.format(
    len(set(vc.index).difference(words_to_remove)), len(tags)))

unique_other_words = sorted(list(set(
    [word for sentence in test.split for word in sentence]).union(
    [word for sentence in test.split for word in sentence]).difference(
    unique_words)))

word_removal_dict = {w: '' for w in {y for x in test.stemmed for y in x}}
word_removal_dict.update({w: w for w in vc.index})
word_removal_dict.update({w: '' for w in words_to_remove})
word_removal_dict[''] = ''

good_words = set(vc.index).difference(words_to_remove)

for df in [csv, test]:
    df['reduced'] = df['stemmed'].apply(lambda x: [word_removal_dict[y] for y in x])
    
for df in [csv, test]:
    df['reduced2'] = [[y if y in good_words else df.loc[i,'PosTag'][j] for j,y in enumerate(df.loc[i,'stemmed'])] for i in df.index];


# In[37]:


# Creates pairs, triplets, etc.
def group(x, l):
    return ['_'.join(x[i:i+l])  for i in range(len(x)-l+1)]

twin_names = ['pairs', 'triplets', 'quadruplets', 'quintuplets',
              'sextuplets', 'septuplets', 'octuplets', 'nonuplets']
for i,name in enumerate(twin_names):
    l = i+2
    print(l)
    for df in [csv, test]:
        df[name] = df['reduced2'].apply(group, args=(l,))


# In[ ]:


for name in twin_names:
    vc = pd.Series([y for x in csv[name] for y in set(x)]).value_counts()
    print('There are {} {}, {} appearing more than once, {} appearing more than twice, {} appearing more than 4 times'.format(
        len(vc), name, len(vc[vc>1]), len(vc[vc>2]), len(vc[vc>4])))


# In[ ]:


plt.style.use('ggplot')
plt.rcParams['font.size'] = 16

for i,name in enumerate(twin_names):
    l = i+2
    
    # Count twins overall and by author
    twin_count = {'ALL': pd.Series([y for x in csv[name] for y in set(x)]).value_counts()}
    for auth in authors:
        twin_count[auth] = pd.Series([y for x in csv.loc[csv.author==auth, name]
                                      for y in x]).value_counts()
    twin_count = pd.DataFrame(twin_count).fillna(0).astype(int).sort_values('ALL', ascending=False)[['ALL']+authors]

    plt.figure(figsize=(20,10))
    bottom = np.zeros((20))
    ind = np.arange(20)
    df = twin_count.head(20)
    for auth in authors:
        vals = df[auth]
        plt.bar(ind, vals, bottom=bottom, label = auth)
        bottom += vals

    plt.plot(ind, df['ALL'] * twin_count[authors[0]].sum() / twin_count['ALL'].sum(), 'k--',
             label='Expected cutoffs for\nuninformative words')
    plt.plot(ind, df['ALL'] * twin_count[authors[:2]].values.sum() / twin_count['ALL'].sum(), 'k--', label='')
    plt.xticks(ind, df.index, rotation='vertical')
    #plt.yticks(np.arange(0,1.1,0.2), ['{:.0%}'.format(x) for x in np.arange(0,1.1,0.2)])
    plt.legend(fontsize=24)
    plt.title('Top 20 {} count split by author (dotted lines is the global average)'.format(name), fontsize=24)
    plt.xlim([-0.7,19.7])
    plt.show()


# **Now things are getting interesting**
# 
# Common pairs have little specificity, for example and "and" after a comma is no big surprise (although some patterns already appear, like the "of the" from Poe).
# 
# But going on patterns do emerge more clearly, and I must say, especially for Poe. Already with three words we see a very characteristic "[...], however, [...]" that can't be a chance; similarly for the "[...], and, [...]" or "[...], in the [...]".
# 
# Increasing the number the specificity grows: expressions like "[...], said I" or "that is to say" are nearly certain indicators for Poe. Any combination from 7 on that is not unique is basically always an indicator for Poe (although admittedly most of them are proper names, so this information seems quite redundant).
# 
# Other authors are not exempt from repetitions, although less dramatically: Shelley likes to emphasize a text by saying '[...]", he cried, [...]'. Lovecraft likes to write "where the sea meets the sky" (probably a citation, though), and to list proper names. Shelley appears to be the only one never to write "[...], on the other hand, [...]".

# ### 2.1.3. Length of sentences
# This pretty basic statistics has already been observed by several people, but anyway, we listed it, so here it is. Remark that this is a bit backward, because sentences are split by Kaggle's team. On the other hand, they are split not randomly but close to full stops and the like, so the length can reflect some writing style of the authors.

# In[43]:


lengths = csv.split.apply(len)
# We remove the 0.5% extreme quantiles as it's just noise.
lengths = np.clip(lengths, *lengths.quantile([0.005, 0.995]))
lengths.index = csv.author
plt.figure(figsize=(15,10))
vals, ticks, _ = plt.hist([np.log(lengths.loc[auth].values) for auth in authors],
                          stacked=True, bins=20, label=authors)
plt.xticks(np.arange(1.5,5,.5), ['{:.0f}'.format(x) for x in np.exp(np.arange(1.5,5,.5))])
plt.legend()
plt.title('Count of sentence length by author')
plt.show()


# Doesn't look tremendously informative, let's scale the plot uniformly to see if there is some overall tendency

# In[44]:


colors = [c['color'] for c in list(plt.rcParams['axes.prop_cycle'])]
plt.figure(figsize=(15,10))
bottom = np.zeros(vals[0].shape)
for i in range(3):
    v = vals[0]/vals[2] if i == 0 else vals[i]/vals[2] - vals[i-1]/vals[2]
    plt.bar(ticks[:-1], v, color=colors[i], align='edge',
            width=ticks[1]-ticks[0], bottom=bottom, label=authors[i])
    bottom = vals[i] / vals[2]
plt.legend()
plt.xticks(np.arange(1.5,5,.5), ['{:.0f}'.format(x) for x in np.exp(np.arange(1.5,5,.5))])
plt.show()


# So **there is actually some clear tendency**, albeit a weird one: Lovecraft seems to have much more uniform sentence length (well centred between, say, 20 and 55 words), while Poe is a bit more represented among the very short (< 20 words) and very long (> 60 words) sentences.

# ## 2.2 Adding them to the model
# 
# Now that we saw that grammar was a useful feature, let's create simple models implementing it (and sentence length). We'll use the same kind of cross validation as above, which unfortunately leads to some unnecessary code duplication.

# In[55]:


# We transform all the "PosTags" to numbers.
# This is just because several of them are punctuation, and sklearn vectorizers
# remove punctuation by default.
dict_tag = {x: i for i,x in enumerate(tags_exp.keys())}
for df in [csv, test]:
    df['PosTagNum'] = df['PosTag'].apply(lambda x: [dict_tag[y] for y in x])
    df['JoinTagNum'] = df['PosTagNum'].apply(lambda x: ' '.join([str(y) for y in x]))
    
# For the length, let's use logarithmic length, as by eye it gave more
# information.
csv['len'] = csv.split.apply(len).apply(np.log)
test['len'] = test.split.apply(len).apply(np.log)


# In[46]:


def test_with_grammar(vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2)),
                      C_logit = [0.3, 1, 3, 10, 30], alpha_SGD = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                      alpha_NB = [0.01, 0.03, 0.1, 0.3, 1]):
    this_range = trange(10)
    model_names = []
    params = []
    targets = []
    preds = []
    preds_calib_sig = []
    preds_calib_iso = []

    models = [LogisticRegression(C=c) for c in C_logit] + [
        SGDClassifier(loss='modified_huber', alpha=a, max_iter=10000, tol=1e-4) for a in alpha_SGD] + [
        MultinomialNB(alpha=a) for a in alpha_NB] + [
        BernoulliNB(alpha=a) for a in alpha_NB]
    
    for test_loop in this_range:
        # Partition: 90% into train, 10% to test.
        test = csv.iloc[int(len(csv)*test_loop/10) : int(len(csv)*(test_loop+1)/10)]
        train = csv.loc[list(set(csv.index).difference(test.index))]

        y_train = train.author.apply(lambda auth: authors_dict[auth]).values
        y_test = test.author.apply(lambda auth: authors_dict[auth]).values

        # For more flexibility, vectorizer can be changed as a parameter.
        X_train = vectorizer.fit_transform(train.text.values)
        X_test = vectorizer.transform(test.text.values)
        
        # Now add also grammar.
        grammar_train = vectorizer.fit_transform(train.JoinTagNum.values)
        grammar_test = vectorizer.transform(test.JoinTagNum.values)
        
        # And finally put them together, adding also word count
        X_train = hstack([X_train, grammar_train, train.len.values.reshape(-1,1)])
        X_test = hstack([X_test, grammar_test, test.len.values.reshape(-1,1)])

        for m in models:
            name = str(m).split('(')[0]

            if name.endswith('NB') or name == 'SGDClassifier':
                param = m.alpha
            elif name == 'SVC' or name == 'LogisticRegression':
                param = m.C

            this_range.set_postfix(working_on=name, step='base')

            m.fit(X_train, y_train)
            targets.append(y_test)
            model_names.append([name] * len(y_test))
            params.append([param] * len(y_test))
            preds.append(m.predict_proba(X_test))

            this_range.set_postfix(working_on=name, step='sigmoid')

            # Sigmoid calibration
            m_sigmoid = CalibratedClassifierCV(m, method='sigmoid')
            m_sigmoid.fit(X_train, y_train)
            preds_calib_sig.append(m_sigmoid.predict_proba(X_test))

            this_range.set_postfix(working_on=name, step='isotonic')
            # Isotonic calibration
            m_isotonic = CalibratedClassifierCV(m, method='isotonic')
            m_isotonic.fit(X_train, y_train)
            preds_calib_iso.append(m_isotonic.predict_proba(X_test))


    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    preds_calib_sig = np.concatenate(preds_calib_sig)
    preds_calib_iso = np.concatenate(preds_calib_iso)
    params = np.log10(np.concatenate(params))
    model_names = np.concatenate(model_names)

    log_losses = -np.log(np.clip(preds[np.arange(len(preds)), targets], 1e-15, 1-1e-15))
    log_losses_sig = -np.log(np.clip(preds_calib_sig[np.arange(len(preds_calib_sig)), targets], 1e-15, 1-1e-15))
    log_losses_iso = -np.log(np.clip(preds_calib_iso[np.arange(len(preds_calib_iso)), targets], 1e-15, 1-1e-15))

    final_results = pd.DataFrame({'LogLoss': log_losses, 'Param': params, 'Names': model_names,
                                  'Target': targets, 'Prediction': np.argmax(preds, axis=1),
                                  'LogLoss Sigmoid': log_losses_sig, 'LogLoss Isotonic': log_losses_iso})
    final_results['Accuracy'] = final_results['Target'] == final_results['Prediction']

    return pd.concat((final_results[[c, 'Names', 'Param']].groupby(['Names', 'Param']).mean()
                      for c in ('LogLoss', 'LogLoss Sigmoid', 'LogLoss Isotonic', 'Accuracy')),
                     axis=1).reset_index()


# In[47]:


# Off-line calculation to save time
# with_grammar = test_with_grammar(C_logit = [10, 30, 100], alpha_SGD = [3e-5, 1e-4, 3e-4], alpha_NB = [1e-3, 3e-3, 0.01, 0.03, 0.1])
# Parameters were already tested to be around their log-loss minimum

with_grammar = pd.DataFrame([['BernoulliNB', -3.0, 1.6471344235434087, 0.4435087577077471, 0.3577320896648269, 0.8567853312222279], ['BernoulliNB', -2.5228787452803374, 1.4710867854088956, 0.43929662596014907, 0.3543656366534263, 0.859441238061188], ['BernoulliNB', -2.0, 1.2902726232856094, 0.4369954213267632, 0.3525028669614839, 0.8606159660861127], ['BernoulliNB', -1.5228787452803376, 1.1501147846426674, 0.44044722351873644, 0.3559070371653456, 0.8597987639818172], ['BernoulliNB', -1.0, 1.059813417422428, 0.4611156884810041, 0.3675970332149561, 0.8557638285918586], ['LogisticRegression', 1.0, 0.41495735633115627, 0.4444363365240902, 0.4519791696222668, 0.8443740742632412], ['LogisticRegression', 1.4771212547196624, 0.3967254346672535, 0.4279426091513384, 0.43648381860509605, 0.850196639256346], ['LogisticRegression', 2.0, 0.4030843012131652, 0.41870106490803444, 0.42726604153139947, 0.8536186730680831], ['MultinomialNB', -3.0, 0.43307769203579094, 0.3996930781027156, 0.3665698887670437, 0.8550998518821186], ['MultinomialNB', -2.5228787452803374, 0.3890398733992301, 0.3871984611130643, 0.3619716036094982, 0.859339087798151], ['MultinomialNB', -2.0, 0.3545547603091124, 0.37540682385085566, 0.3560837402829932, 0.8625568210838143], ['MultinomialNB', -1.5228787452803376, 0.34351097136696496, 0.3723802076864194, 0.3610605741745939, 0.8634250983196282], ['MultinomialNB', -1.0, 0.3740066997854423, 0.3944609074299107, 0.3932247328330751, 0.8538740487256755], ['SGDClassifier', -4.522878745280337, 2.834158930282594, 0.4398856555186023, 0.44728487427768865, 0.8379386076919149], ['SGDClassifier', -4.0, 1.1306407647492276, 0.4348249833675688, 0.44502743957344704, 0.8303794882271822], ['SGDClassifier', -3.5228787452803374, 0.7603235276871617, 0.4656250015696377, 0.4707740721717643, 0.8110730885132029]], columns=['Names', 'Param', 'LogLoss', 'LogLoss Sigmoid', 'LogLoss Isotonic', 'Accuracy'])

plot_calibrated_results(with_grammar)


# **Results are similar but slightly better**
# 
# Let's try to make a submission based on them.
# 
# (to be sure, we should also check ensembling methods like we did above, but the results are essentially comparable so I won't spam by repeating them.)

# In[48]:


previous_best = [('MultiNB', MultinomialNB(alpha=0.03)),
     ('Calibrated MultiNB', CalibratedClassifierCV(MultinomialNB(alpha=0.01), method='isotonic')),
     ('Calibrated BernoulliNB', CalibratedClassifierCV(BernoulliNB(alpha=0.01), method='isotonic')),
     ('Calibrated Huber', CalibratedClassifierCV(
         SGDClassifier(loss='modified_huber', alpha=1e-4, max_iter=10000, tol=1e-4), method='sigmoid')),
     ('Logit', LogisticRegression(C=30))]


# In[59]:


vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(previous_best, voting='soft', weights=[3,3,3,1,1])
X_train = vectorizer.fit_transform(csv.text.values)
X_test = vectorizer.transform(test.text.values)

# Now add also grammar.
grammar_train = vectorizer.fit_transform(csv.JoinTagNum.values)
grammar_test = vectorizer.transform(test.JoinTagNum.values)

# And finally put them together, adding also word count
X_train = hstack([X_train, grammar_train, csv.len.values.reshape(-1,1)])
X_test = hstack([X_test, grammar_test, test.len.values.reshape(-1,1)])

y_train = csv.author.apply(lambda auth: authors_dict[auth]).values

clf.fit(X_train, y_train)

result = clf.predict_proba(X_test)
pd.DataFrame(result, index=test.index, columns=authors).to_csv('grammar_results.csv')


# # Wait, weren't there neural networks too?
# As a matter of fact, yes, but I ran out of time before making them run appropriately. So here below is the code without too many explanations, if anybody finds any use to it I'd be happy!
# 
# The main idea is to use a CNN to create new features from group of words. CNN tend to overfit terribly with "only" 20k input, so an extremely simple architecture should be used. Furthermore, sentences must be cut at a fixed length (and padded if shorter) for them to work.
# 
# The basic idea I used was: split the training set in 5 pieces, run 5 times, each time use 4 to train the CNN, then remove the last layer and use the previous step to create some features (210 in my architecture) for the remaining fifth. For the training set, put these features to 0, otherwise we'd massively overfit in the next step. Joining all this will add 210*5 = 1050 features in total.
# 
# However, results are shown below and look very bad, so I must have made some implementation error...

# In[60]:


from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


# In[61]:


def force_length(df, max_length):
    df['tokens'] = df.split.apply(len)
    dfs = [df[df.tokens <= max_length].copy()]
    dfs[0]['Part'] = 1
    dfs[0]['text'] = dfs[0]['split'].apply(lambda s: ' '.join(s))
    for mult, cur_min in enumerate(range(max_length, df.tokens.max(), max_length)):
        for i in range(mult+2):
            dfs.append(df[np.logical_and(df.tokens >= cur_min, df.tokens < cur_min + max_length)].copy())
            dfs[-1]['Part'] = i+1
            dfs[-1]['split'] = dfs[-1]['split'].apply(lambda s: s[int(len(s)*i/(mult+2)):int(len(s)*(i+1)/(mult+2))])
            dfs[-1]['text'] = dfs[-1]['split'].apply(lambda s: ' '.join(s))
    return pd.concat(dfs).sort_index()

def single_one_hot(i, max_features):
    ''' Transforms a single number in a one hot-encoded vector. '''
    res = np.zeros(max_features)
    # Exclude negative numbers and nans.
    if i >= 0 and i == i:
        res[i] = 1
    return res

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]

def extract_all_features(train, test, loglen=True, input_means_file=None, output_means_file=None):
    result_train = train.reset_index().copy()
    result_test = test.reset_index().copy()
    
    for df in [train, test]:
        df['split'] = df.text.apply(nltk.word_tokenize)
        
    all_train_words = [word for sentence in train.split.values for word in sentence]
    
    unique_other_words = sorted(list(set(
        [word for sentence in test.split for word in sentence]).difference(
        all_train_words)))
    
    unique_train_words = sorted(list(set(all_train_words)))
    
    def one_hot(l, max_length=60, max_features=len(unique_train_words)):
        ''' Transforms a list l in a one hot-encoded array. '''
        ''' encoded = np.array([single_one_hot(x, max_features=max_features) for x in l])
        res = np.zeros((max_length, max_features))
        res[:len(encoded)] = encoded[:len(res)]
        return res'''
        return np.array([single_one_hot(x, max_features=max_features) for x in l])


    dict_train_words = {s: i for i,s in enumerate(unique_train_words)}
    
    def set_numeric(x, max_length=60, d=dict_train_words):
        res = np.zeros(max_length, dtype=int)-1
        true_res = np.array([d.setdefault(y, -1) for y in x])
        res[:len(true_res)] = true_res
        return res
    
    test = force_length(test, 60)
    train = force_length(train, 60)
    
    for df in [train, test]:
        df['numeric'] = df['split'].apply(set_numeric)
        
    if input_means_file is None:
        means = np.zeros((60, len(unique_train_words)))
        for n in tqdm(train.numeric):
            means += one_hot(n) / len(train)
            
        if output_means_file is not None:
            save_array(output_means_file, means)
    else:
        means = load_array(input_means_file)
        
    first_step = lambda x: K.one_hot(K.cast(x, tf.int32), len(unique_train_words)) - means
        
    #def first_step(x, depth=len(unique_train_words)):
    #    return K.one_hot(K.cast(x, tf.int32), depth) - means

    authors = ['HPL','EAP','MWS']

    x_train = np.array(list(train.numeric.values))
    y_train = keras.utils.to_categorical(train.author.apply(authors.index).values)
    x_test = np.array(list(test.numeric.values))
    
    split_size=5
    
    cutoffs = np.linspace(0,len(x_train),split_size+1).astype(int)
    
    
    for i in range(split_size):
        train['CNN_features'+str(i)] = None
        
        model = Sequential([
            Lambda(first_step, input_shape=(60,), output_shape=(60, len(unique_train_words))),
            Conv1D(30, 1, input_shape=(60, len(unique_train_words))),
            ZeroPadding1D(1),
            Conv1D(30, 3, activation='relu'),
            MaxPooling1D(strides=2),
            ZeroPadding1D(1),
            Conv1D(30, 3, activation='relu'),
            MaxPooling1D(strides=2),
            ZeroPadding1D(1),
            Conv1D(30, 3, activation='relu'),
            MaxPooling1D(strides=2),
            Flatten(),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        pred_idx = np.arange(cutoffs[i], cutoffs[i+1])
        train_idx = sorted(list(set(np.arange(len(x_train))).difference(pred_idx)))
        
        model.fit(x_train[train_idx], y_train[train_idx], validation_data=(x_train[pred_idx], y_train[pred_idx]),
                 batch_size=64, epochs=100, callbacks=[EarlyStopping(patience=1)])

        # Make a copy of the model, but without the last layers (or Dropout, of course) and last activation
        partial_model = Sequential(model.layers[:-5] + [Conv1D(30, 3), MaxPooling1D(strides=2), Flatten()])
        partial_model.layers[-3].set_weights(model.layers[-5].get_weights())
        
        # Need to pad to zero to allow pandas to set the whole df at once.
        output = np.zeros((len(train), partial_model.output_shape[1]))
        output[pred_idx] = partial_model.predict(x_train[pred_idx], batch_size=64)
        train['CNN_features'+str(i)] = list(output)
        
        test['CNN_features'+str(i)] = list(partial_model.predict(x_test, batch_size=64))
        
        K.clear_session()
        collect()
        
    cnn_cols = [c for c in train.columns if c.startswith('CNN_features')]
    for c in cnn_cols:
        train[c+'_nonneg'] = train[c].apply(lambda x: np.maximum(x,0))
        test[c+'_nonneg'] = test[c].apply(lambda x: np.maximum(x,0))
    
    for df, result_df in zip([train,test],[result_train,result_test]):
        for c in [k for k in train.columns if k.startswith('CNN_features')]:
            result_df[c] = result_df['id'].apply(
                lambda x: df.loc[x, c] if isinstance(df.loc[x], pd.Series)
                else np.max([list(y) for y in df.loc[x, c]], axis=0)
            )
    
    ## Now add len feature
        
    for df in [result_train, result_test]:
        df['len'] = df['text'].apply(len if not loglen else lambda x: np.log(len(x)))

    m = result_train.len.mean()
    s = result_train.len.std()

    for df in [result_train, result_test]:
        df['len'] = (df['len'] - m) / s
        
    ## Finally, integrate the tf-idf.
    vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
    
    X_train = vectorizer.fit_transform(result_train.text.values)
    y_train = result_train.author.apply(authors.index).values
    X_test = vectorizer.transform(result_test.text.values)
    
    print('Increasing from {} features...'.format(X_train.shape[1]))
    
    m = min(result_train.len.values.min(), result_test.len.values.min())
    
    X_train_nonneg = hstack([X_train, result_train.len.values.reshape(-1,1) - m + 1e-2,] +                             [np.array([list(y) for y in result_train[c]])
                              for c in result_train.columns if c.startswith('CNN_') and c.endswith('_nonneg')])
    X_train = hstack([X_train, result_train.len.values.reshape(-1,1)] +                      [np.array([list(y) for y in result_train[c]])
                       for c in result_train.columns if c.startswith('CNN_') and not c.endswith('_nonneg')])

    X_test_nonneg = hstack([X_test, result_test.len.values.reshape(-1,1) - m + 1e-2,] +                             [np.array([list(y) for y in result_test[c]])
                              for c in result_test.columns if c.startswith('CNN_') and c.endswith('_nonneg')])
    X_test = hstack([X_test, result_test.len.values.reshape(-1,1)] +                      [np.array([list(y) for y in result_test[c]])
                       for c in result_test.columns if c.startswith('CNN_') and not c.endswith('_nonneg')])
    #X_test_nonneg = hstack((X_test, result_test.len.values.reshape(-1,1) - m + 1e-2,
    #                        np.array([list(y) for y in result_test.CNN_features_nonneg]) + 1e-2))
    #X_test = hstack((X_test, result_test.len.values.reshape(-1,1),
    #                 np.array([list(y) for y in result_test.CNN_features])))
    
    print('... to {}'.format(X_train.shape[1]))
    
    # If test is actually validation, also return y_test
    try:
        return (X_train, X_train_nonneg, y_train, X_test, X_test_nonneg,
                result_test.author.apply(authors.index).values)
    except:
        return (X_train, X_train_nonneg, y_train, X_test, X_test_nonneg)


# In[62]:


def test_with_CNN_features(C_logit = [0.3, 1, 3, 10, 30], alpha_SGD = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                           alpha_NB = [0.01, 0.03, 0.1, 0.3, 1],
                           input_means_file=None, output_means_file=None, csv=None, loops=10):
    model_names = []
    params = []
    targets = []
    preds = []
    preds_calib_sig = []
    preds_calib_iso = []
    
    models = [LogisticRegression(C=c) for c in C_logit] + [
        SGDClassifier(loss='modified_huber', alpha=a, max_iter=10000, tol=1e-4) for a in alpha_SGD] + [
        MultinomialNB(alpha=a) for a in alpha_NB] + [
        BernoulliNB(alpha=a) for a in alpha_NB]
    
    tstart = time()
    
    if csv is None:
        csv = pd.read_csv('../input/train.csv', index_col=0)
    
    for test_loop in range(min(loops,10)):
        print('\n\n\n **** Start of loop {}/10 ****\n\n'.format(test_loop))
        
        # Partition: 90% into train, 10% to test.
        test = csv.iloc[int(len(csv)*test_loop/10) : int(len(csv)*(test_loop+1)/10)].copy()
        train = csv.loc[list(set(csv.index).difference(test.index))].copy()
        
        X_train, X_train_nonneg, y_train, X_test, X_test_nonneg, y_test =            extract_all_features(train, test, input_means_file=input_means_file, 
                                 output_means_file=output_means_file)

        for m in models:
            name = str(m).split('(')[0]
            
            print('Working on {}'.format(name))

            if name.endswith('NB') or name == 'SGDClassifier':
                param = m.alpha
            elif name == 'SVC' or name == 'LogisticRegression':
                param = m.C
                
            targets.append(y_test)
            model_names.append([name] * len(y_test))
            params.append([param] * len(y_test))
            
            for mod, preds_list, text in zip([m, CalibratedClassifierCV(m, method='sigmoid'),
                                        CalibratedClassifierCV(m, method='isotonic')],
                                       [preds, preds_calib_sig, preds_calib_iso],
                                            ['base','sigmoid','isotonic']):
                # For NB, only non-negative values make sense
                if name.endswith('NB'):
                    mod.fit(X_train_nonneg, y_train)
                    preds_list.append(mod.predict_proba(X_test_nonneg))
                else:
                    mod.fit(X_train, y_train)
                    preds_list.append(mod.predict_proba(X_test))
                    
                print('... {}, log-loss: {:.3f}'.format(
                    text, -np.log(np.clip(preds_list[-1][np.arange(len(preds_list[-1])), targets[-1]],
                                         1e-15, 1-1e-15)).mean()))
            
        print('ETA: {:.0f}s'.format((time()-tstart)/(test_loop+1)*(10-test_loop)))
        
    targets = np.concatenate(targets)
    preds = np.concatenate(preds)
    preds_calib_sig = np.concatenate(preds_calib_sig)
    preds_calib_iso = np.concatenate(preds_calib_iso)
    params = np.log10(np.concatenate(params))
    model_names = np.concatenate(model_names)

    log_losses = -np.log(np.clip(preds[np.arange(len(preds)), targets], 1e-15, 1-1e-15))
    log_losses_sig = -np.log(np.clip(preds_calib_sig[np.arange(len(preds_calib_sig)), targets], 1e-15, 1-1e-15))
    log_losses_iso = -np.log(np.clip(preds_calib_iso[np.arange(len(preds_calib_iso)), targets], 1e-15, 1-1e-15))

    final_results = pd.DataFrame({'LogLoss': log_losses, 'Param': params, 'Names': model_names,
                                  'Target': targets, 'Prediction': np.argmax(preds, axis=1),
                                  'LogLoss Sigmoid': log_losses_sig, 'LogLoss Isotonic': log_losses_iso})
    final_results['Accuracy'] = final_results['Target'] == final_results['Prediction']

    return pd.concat((final_results[[c, 'Names', 'Param']].groupby(['Names', 'Param']).mean()
                      for c in ('LogLoss', 'LogLoss Sigmoid', 'LogLoss Isotonic', 'Accuracy')),
                     axis=1).reset_index()


# In[64]:


# Off-line, long, computation
# results_with_CNN = test_with_CNN_features()

results_with_CNN = pd.DataFrame([['BernoulliNB', -2.0, 3.8188118076260196, 0.5096711927616674,  1.1795356262216476, 0.8534654476735277], ['BernoulliNB', -1.5228787452803376, 3.880750213815746,  0.5202376688083046, 1.358177667710076, 0.8495326625466061], ['BernoulliNB', -1.0, 3.975367932540496, 0.5328807805309641,  1.5471285282204623, 0.8448848255784258], ['BernoulliNB', -0.5228787452803376, 4.102530886106262,  0.5488427953682987, 1.72837367456449, 0.8387047346646918], ['BernoulliNB', 0.0, 4.388301526696273, 0.5843221269485471,  1.89007974002896, 0.8267531538893713], ['LogisticRegression', -0.5228787452803376, 0.8276233877874041,  0.7780217174913097, 1.8656785329386454, 0.8600030645078911], ['LogisticRegression', 0.0, 0.8491437229898623, 0.7675657932520066,  1.8398311431440328, 0.8601562899024465], ['LogisticRegression', 0.47712125471966244, 0.8789745283747518,  0.7318205609236829, 1.7467091350857022, 0.8610756422697788], ['LogisticRegression', 1.0, 0.9796890811277373, 0.6927679221099151,  1.625134327727758, 0.8607691914806681], ['LogisticRegression', 1.4771212547196624, 1.128075722131628,  0.6694903303020543, 1.5419854402760016, 0.8604627406915573], ['MultinomialNB', -2.0, 4.010611404986249, 0.5025973693229974,  1.3490547019869024, 0.8581643597732265], ['MultinomialNB', -1.5228787452803376, 4.023318560125922,  0.5044575782125705, 1.3563465081027166, 0.8577557587210787], ['MultinomialNB', -1.0, 4.040370929027427, 0.5076699429652193,  1.376520800310606, 0.8564278053015987], ['MultinomialNB', -0.5228787452803376, 4.07137326642833,  0.5109088091542248, 1.4032173932607042, 0.8558149037233771], ['MultinomialNB', 0.0, 4.156411238990353, 0.5167148250594493,  1.4397098896229865, 0.8543848000408601], ['SGDClassifier', -5.0, 3.233127283422876, 0.5836388395620052,  1.0881375025505777, 0.8118392154859799], ['SGDClassifier', -4.522878745280337, 3.4527652042491987,  0.5732323286185356, 0.9354805341131621, 0.8217988661320803], ['SGDClassifier', -4.0, 3.6765661324614975, 0.5661220086181918,  0.7931251697056672, 0.8088257827263905], ['SGDClassifier', -3.5228787452803374, 2.3424274242460297,  0.5352699359423028, 0.631858188229672, 0.8021860156289903], ['SGDClassifier', -3.0, 1.4268986988840726, 0.467931669235592,  0.6224331109971302, 0.8058123499668012]],       columns=['Names', 'Param', 'LogLoss', 'LogLoss Sigmoid', 'LogLoss Isotonic', 'Accuracy'])

plot_calibrated_results(results_with_CNN, ylim=[0,2])


# **0.5... Scary, isn't it?**
