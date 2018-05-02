
# coding: utf-8

# # Quora Question-pair classification
# 
# This competition is about modelling whether a pair of questions on Quora is asking the same question. For this problem we have about **400.000** training examples. Each row consists of two sentences and a binary label that indicates to us whether the two questions were the same or not.
# 
# Inspired by this nice [kernel](https://www.kaggle.com/arthurtok/d/mcdonalds/nutrition-facts/super-sized-we-macdonald-s-nutritional-metrics) from [Anisotropic](https://www.kaggle.com/arthurtok) I've added a few interactive 2D and 3D scatter plots.
# To get an insight into how the duplicates evolve over the number of words in the questions, I've added a plotly animation that encodes number of words and word share similarity in a scatter plot.
# 
# **We will be looking in detail at:**
# 
# * question pair TF-IDF encodings
# * basic feature engineering and their embeddings in lower dimensional spaces
# * parallel coordinates visualization
# * model selection and evaluation + sample submission.
# 
# If you like this kernel, please upvote it :D, thanks!
# 
# 
# ----------
# 
# 
# Added a final section for cross-validated model selection and evaluation. We will look at standard binary classification metrics, like ROC and PR curves and their AUCs. The best (linear) model that we found then generates a submission.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output

get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/train.csv").fillna("")
df.head() 


# So we have six columns in total one of which is the label.

# In[ ]:


df.info()


# In[ ]:


df.shape


# We have a fairly balanced dataset here.

# In[ ]:


df.groupby("is_duplicate")['id'].count().plot.bar()


# # Feature construction
# 
# We will now construct a basic set of features that we will later use to embed our samples with.
# 
# The first we will be looking at is rather standard TF-IDF encoding for each of the questions. In order to limit the computational complexity and storage requirements we will only encode the top terms across all documents with TF-IDF and also look at a subsample of the data.

# In[ ]:


dfs = df[0:2500]
dfs.groupby("is_duplicate")['id'].count().plot.bar()


# The subsample still has a very similar label distribution, ok to continue like that, without taking a deeper look how to achieve better sampling than just taking the first rows of the dataset.
# 
# Create a dataframe where the top 50% of rows have only question 1 and the bottom 50% have only question 2, same ordering per halve as in the original dataframe.

# In[ ]:


dfq1, dfq2 = dfs[['qid1', 'question1']], dfs[['qid2', 'question2']]
dfq1.columns = ['qid1', 'question']
dfq2.columns = ['qid2', 'question']

# merge two two dfs, there are two nans for question
dfqa = pd.concat((dfq1, dfq2), axis=0).fillna("")
nrows_for_q1 = dfqa.shape[0]/2
dfqa.shape


# Transform questions by TF-IDF.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
mq1 = TfidfVectorizer(max_features = 256).fit_transform(dfqa['question'].values)
mq1


# Since we are looking at pairs of data, we will be taking the difference of all question one and question two pairs with this. This will result in a matrix that again has the same number of rows as the subsampled data and one vector that describes the relationship between the two questions.

# In[ ]:


diff_encodings = np.abs(mq1[::2] - mq1[1::2])
diff_encodings


# # 3D t-SNE embedding
# 
# We will use t-SNE to embed the TF-IDF vectors in three dimensions and create an interactive scatter plot with them.

# In[ ]:


from sklearn.manifold import TSNE
tsne = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=200,
    verbose=2,
    angle=0.5
).fit_transform(diff_encodings.toarray())


# In[ ]:


trace1 = go.Scatter3d(
    x=tsne[:,0],
    y=tsne[:,1],
    z=tsne[:,2],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = dfs['is_duplicate'].values,
        colorscale = 'Portland',
        colorbar = dict(title = 'duplicate'),
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.75
    )
)

data=[trace1]
layout=dict(height=800, width=800, title='test')
fig=dict(data=data, layout=layout)
py.iplot(fig, filename='3DBubble')


# That three dimensional embedding looks nice, but is not telling us much about the structure of the space that we created. There seem to be no clusters of either class present, so let's go on to the next section.

# ## Feature EDA
# 
# Let us now construct a few features
# 
# * character length of questions 1 and 2
# * number of words in question 1 and 2
# * normalized word share count.
# 
# We can then have a look at how well each of these separate the two classes.

# In[ ]:


df['q1len'] = df['question1'].str.len()
df['q2len'] = df['question2'].str.len()

df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))

def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))


df['word_share'] = df.apply(normalized_word_share, axis=1)

df.head()


# The distributions for normalized word share have some overlap on the far right hand side, meaning there are quite a lot of questions with high word similarity but are both duplicates and non-duplicates.

# In[ ]:


plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
sns.violinplot(x = 'is_duplicate', y = 'word_share', data = df[0:50000])
plt.subplot(1,2,2)
sns.distplot(df[df['is_duplicate'] == 1.0]['word_share'][0:10000], color = 'green')
sns.distplot(df[df['is_duplicate'] == 0.0]['word_share'][0:10000], color = 'red')


# Scatter plot of question pair character lengths where color indicates duplicates and the size the word share coefficient we've calculated earlier.

# In[ ]:


df_subsampled = df[0:2000]

trace = go.Scatter(
    y = df_subsampled['q2len'].values,
    x = df_subsampled['q1len'].values,
    mode='markers',
    marker=dict(
        size= df_subsampled['word_share'].values * 60,
        color = df_subsampled['is_duplicate'].values,
        colorscale='Portland',
        showscale=True,
        opacity=0.5,
        colorbar = dict(title = 'duplicate')
    ),
    text = np.round(df_subsampled['word_share'].values, decimals=2)
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Scatter plot of character lengths of question one and two',
    hovermode= 'closest',
        xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    yaxis=dict(
        title= 'Question 2 length',
        ticklen= 5,
        gridwidth= 2,
        showgrid=False,
        zeroline=False,
        showline=False,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatterWords')


# # Animation over average number of words
# 
# For that we will calculate the average number of words in both questions for each row.
# 
# In the end we want to have a scatter plot, just like the one above, but giving us one more dimension, in that case the average number of words in both questions. That will allow us to see the dependence on that variable. We also expect that as the number of words is increased, the character lengths of Q1 and Q2 will increase.

# In[ ]:


from IPython.display import display, HTML

df_subsampled['q_n_words_avg'] = np.round((df_subsampled['q1_n_words'] + df_subsampled['q2_n_words'])/2.0).astype(int)
print(df_subsampled['q_n_words_avg'].max())
df_subsampled = df_subsampled[df_subsampled['q_n_words_avg'] < 20]
df_subsampled.head()


# In[ ]:


word_lens = sorted(list(df_subsampled['q_n_words_avg'].unique()))
# make figure
figure = {
    'data': [],
    'layout': {
        'title': 'Scatter plot of char lenghts of Q1 and Q2 (size ~ word share similarity)',
    },
    'frames': []#,
    #'config': {'scrollzoom': True}
}

# fill in most of layout
figure['layout']['xaxis'] = {'range': [0, 200], 'title': 'Q1 length'}
figure['layout']['yaxis'] = {
    'range': [0, 200],
    'title': 'Q2 length'#,
    #'type': 'log'
}
figure['layout']['hovermode'] = 'closest'

figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 300, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Avg. number of words in both questions:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

# make data
word_len = word_lens[0]
dff = df_subsampled[df_subsampled['q_n_words_avg'] == word_len]
data_dict = {
    'x': list(dff['q1len']),
    'y': list(dff['q2len']),
    'mode': 'markers',
    'text': list(dff['is_duplicate']),
    'marker': {
        'sizemode': 'area',
        #'sizeref': 200000,
        'colorscale': 'Portland',
        'size': dff['word_share'].values * 120,
        'color': dff['is_duplicate'].values,
        'colorbar': dict(title = 'duplicate')
    },
    'name': 'some name'
}
figure['data'].append(data_dict)

# make frames
for word_len in word_lens:
    frame = {'data': [], 'name': str(word_len)}
    dff = df_subsampled[df_subsampled['q_n_words_avg'] == word_len]

    data_dict = {
        'x': list(dff['q1len']),
        'y': list(dff['q2len']),
        'mode': 'markers',
        'text': list(dff['is_duplicate']),
        'marker': {
            'sizemode': 'area',
            #'sizeref': 200000,
            'size': dff['word_share'].values * 120,
            'colorscale': 'Portland',
            'color': dff['is_duplicate'].values,
            'colorbar': dict(title = 'duplicate')
        },
        'name': 'some name'
    }
    frame['data'].append(data_dict)

    figure['frames'].append(frame)
    slider_step = {'args': [
        [word_len],
        {
            'frame': {'duration': 300, 'redraw': False},
            'mode': 'immediate',
            'transition': {'duration': 300}
        }
     ],
     'label': word_len,
     'method': 'animate'}
    sliders_dict['steps'].append(slider_step)

    
figure['layout']['sliders'] = [sliders_dict]

py.iplot(figure)


# What is interesting about that, is that as the number of words increases, the distribution of character lengths of the first and second question becomes less and less spherical.

# # Embedding with engineered features
# 
# We will now revisit the t-SNE embedding with the manually engineered features.
# 
# For that we use the number of words in both questions, character lengths and their word share coefficient. t-SNE is sensitive to scaling of different dimensions and we want all of the dimensions to contribute equally to the distance measure that t-SNE is trying to preserve.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

df_subsampled = df[0:3000]
X = MinMaxScaler().fit_transform(df_subsampled[['q1_n_words', 'q1len', 'q2_n_words', 'q2len', 'word_share']])
y = df_subsampled['is_duplicate'].values


# In[ ]:


tsne = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=200,
    verbose=2,
    angle=0.5
).fit_transform(X)


# In[ ]:


trace1 = go.Scatter3d(
    x=tsne[:,0],
    y=tsne[:,1],
    z=tsne[:,2],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = y,
        colorscale = 'Portland',
        colorbar = dict(title = 'duplicate'),
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.75
    )
)

data=[trace1]
layout=dict(height=800, width=800, title='3d embedding with engineered features')
fig=dict(data=data, layout=layout)
py.iplot(fig, filename='3DBubble')


# The embedding of the engineered features has much more structure than the previous one where we were only computing differences of TF-IDF encodings.
# 
# In the cluster of the negatives we have few positives whereas in the cluster of positives we have a lot more negatives. That matches our observation from the boxplot of word share coefficient above, where we could see that the negative class has a lot of overlap with the positive class for high word share coefficients.

# # Parallel Coordinates
# 
# We now want to get another perspective on high dimensional data, such as the TF-IDF encoded questions. For that purpose I'll encode the concatenated questions into a set of N dimensions, s.t. each row in the dataframe then has one N dimensional vector associated to it.
# With this we can then have a look at how these coordinates (or TF-IDF dimensions) vary by label.
# 
# There are many EDA methods to visualize high dimensional data, I'll show parallel coordinates here.
# 
# To make a nice looking plot, I've chosen N to be quite small, much smaller actually than you would encode it in a machine learning algorithm.

# In[ ]:


from pandas.tools.plotting import parallel_coordinates

df_subsampled = df[0:500]

N = 64

#encoded = HashingVectorizer(n_features = N).fit_transform(df_subsampled.apply(lambda row: row['question1']+' '+row['question2'], axis=1).values)
encoded = TfidfVectorizer(max_features = N).fit_transform(df_subsampled.apply(lambda row: row['question1']+' '+row['question2'], axis=1).values)
# generate columns in the dataframe for each of the 32 dimensions
cols = ['hashed_'+str(i) for i in range(encoded.shape[1])]
for idx, col in enumerate(cols):
    df_subsampled[col] = encoded[:,idx].toarray()

plt.figure(figsize=(12,8))
kws = {
    'linewidth': 0.5,
    'alpha': 0.7
}
parallel_coordinates(
    df_subsampled[cols + ['is_duplicate']],
    'is_duplicate',
    axvlines=False, colormap=plt.get_cmap('plasma'),
    **kws
)
#plt.grid(False)
plt.xticks([])
plt.xlabel("encoded question dimensions")
plt.ylabel("value of dimension")


# In the parallel coordinates we can see that there are some dimensions that have high TF-IDF features values for duplicates and others high values for non-duplicates.

# # Question character length correlations by duplication label
# 
# The pairplot of character length of both questions by duplication label is showing us that, duplicated questions seem to have a somewhat similar amount of characters in them.
# 
# Also we can see something quite intuitive, that there is rather strong correlation in the number of words and the number of characters in a question.

# In[ ]:


n = 10000
sns.pairplot(df[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'is_duplicate']][0:n], hue='is_duplicate')


# # Model starter
# 
# Train a model with the basic feature we've constructed so far.
# 
# For that we will use Logisitic regression, for which we will do a quick parameter search with CV, plot ROC and PR curve on the holdout set and finally generate a submission.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(df[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_share']])

X = scaler.transform(df[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_share']])
y = df['is_duplicate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Run cross-validation with a few hyper parameters.

# In[ ]:


clf = LogisticRegression()
grid = {
    'C': [1e-6, 1e-3, 1e0],
    'penalty': ['l1', 'l2']
}
cv = GridSearchCV(clf, grid, scoring='neg_log_loss', n_jobs=-1, verbose=1)
cv.fit(X_train, y_train)


# Print validation results. Here we see that the strongly regularized model has much worse negative log loss than the other two models, regardless of which regularizer we've used.

# In[ ]:


for i in range(1, len(cv.cv_results_['params'])+1):
    rank = cv.cv_results_['rank_test_score'][i-1]
    s = cv.cv_results_['mean_test_score'][i-1]
    sd = cv.cv_results_['std_test_score'][i-1]
    params = cv.cv_results_['params'][i-1]
    print("{0}. Mean validation neg log loss: {1:.3f} (std: {2:.3f}) - {3}".format(
        rank,
        s,
        sd,
        params
    ))


# In[ ]:


print(cv.best_params_)
print(cv.best_estimator_.coef_)


# ### ROC
# 
# Receiver operator characteristic, used very commonly to assess the quality of models for binary classification.
# 
# We will look at at three different classifiers here, a strongly regularized one and two with weaker regularization. The heavily regularized model has parameters very close to zero and is actually worse than if we would pick the labels for our holdout samples randomly.

# In[ ]:


colors = ['r', 'g', 'b', 'y', 'k', 'c', 'm', 'brown', 'r']
lw = 1
Cs = [1e-6, 1e-4, 1e0]

plt.figure(figsize=(12,8))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for different classifiers')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

labels = []
for idx, C in enumerate(Cs):
    clf = LogisticRegression(C = C)
    clf.fit(X_train, y_train)
    print("C: {}, parameters {} and intercept {}".format(C, clf.coef_, clf.intercept_))
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=colors[idx])
    labels.append("C: {}, AUC = {}".format(C, np.round(roc_auc, 4)))

plt.legend(['random AUC = 0.5'] + labels)


# # Precision-Recall Curve
# 
# Also used very commonly, but more often in cases where we have class-imbalance. We can see here, that there are a few positive samples that we can identify quite reliably. On in the medium and high recall regions we see that there are also positives samples that are harder to separate from the negatives.

# In[ ]:


pr, re, _ = precision_recall_curve(y_test, cv.best_estimator_.predict_proba(X_test)[:,1])
plt.figure(figsize=(12,8))
plt.plot(re, pr)
plt.title('PR Curve (AUC {})'.format(auc(re, pr)))
plt.xlabel('Recall')
plt.ylabel('Precision')


# # Prepare submission
# 
# Here we read the test data and apply the same transformations that we've used for the training data. We also need to scale the computed features again.

# In[ ]:


dftest = pd.read_csv("../input/test.csv").fillna("")

dftest['q1len'] = dftest['question1'].str.len()
dftest['q2len'] = dftest['question2'].str.len()

dftest['q1_n_words'] = dftest['question1'].apply(lambda row: len(row.split(" ")))
dftest['q2_n_words'] = dftest['question2'].apply(lambda row: len(row.split(" ")))

dftest['word_share'] = dftest.apply(normalized_word_share, axis=1)

dftest.head()


# We use the best estimator found by cross-validation and retrain it, using the best hyper parameters, on the whole training set.

# In[ ]:


retrained = cv.best_estimator_.fit(X, y)

X_submission = scaler.transform(dftest[['q1len', 'q2len', 'q1_n_words', 'q2_n_words', 'word_share']])

y_submission = retrained.predict_proba(X_submission)[:,1]

submission = pd.DataFrame({'test_id': dftest['test_id'], 'is_duplicate': y_submission})
submission.head()


# In[ ]:


sns.distplot(submission.is_duplicate[0:2000])


# In[ ]:


submission.to_csv("submission.csv", index=False)


# Most likely this submission will not score very good, we've only used a small set of features and didn't really dive into feature engineering that much. :)
