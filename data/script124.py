
# coding: utf-8

# In most competition I am way too greedy searching for a better rank on the so-called Public Leaderboard ;-)
# 
# My personnal algorithm goes like this :
# 1. Choose an algorithm at random but contains boost: XGBoost, LightGBM or CatBoost? Or whateverBoost
# 2. Perform 5/10 fold CV mean scoring
# 3. Search for the holly grail feature and check if mean score has improved by 1e-11
# 4. Submit to the LB
# 5. If rank is up, keep the change, if its down I'm sad and I drop the change
# 6. Go to 3 and loop until the end of the competition
# 
# This works not so bad in many challenges but I've seen at least 2 competitions where this was clearly not the way to go :
# 1. [santander-customer-satisfaction ](http://www.kaggle.com/c/santander-customer-satisfaction)
# 2. [mercedes-benz-greener-manufacturing](http://www.kaggle.com/c/mercedes-benz-greener-manufacturing)
# 
# I'm wondering if Porto Seguro’s Safe Driver Prediction competition will end up in these competitions since sometimes Public LB scores are really off compared to local CV.
# 
# The question then ibecomes how can we make sure that what we are doing locally is significant?
# 
# I came across an article by Thomas G. Dietterich written in 1997 about 5 iterations of 2-fold cross-validation used for a Null hypothesis statistical test.
# 
# The paper is [here](http://sci2s.ugr.es/keel/pdf/algorithm/articulo/dietterich1998.pdf) so you can have a look at it if you don't know about it already.
# 
# I'm not saying thisis good or bad I just would like to start a thread on Null hypothesis testing and see how you carry out these things yourself in your day to day competition tasks...

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier


# In[ ]:


trn = pd.read_csv("../input/train.csv")
target = trn.target
del trn["target"]


# In[ ]:


clf1 = LGBMClassifier(n_estimators=100, n_jobs=2)
clf2 = LGBMClassifier(n_estimators=100, reg_alpha=1, reg_lambda=1, min_split_gain=2, n_jobs=2)


# ## 5 by 2-fold CV t paired test
# We will first use the 5 by 2-fold CV t paired test
# 
# As its name says the test runs five 2-fold cross validation for each classifier. Score differences are then used to compute the folowing t statistic :
# $$
# t = \frac{p_1^{(1)}}{\sqrt{\frac{1}{5}\sum_{i=1}^{5}{s_i^2}}}
# $$
# 
# where :
# - $p_1^{(1)}$ is the classifiers' scores difference for the first fold of the first iteration 
# - $s_i^2$ is the estimated variance of the score difference for $i^{th}$ iteration. This variance computes as $ \left(p_i^{(1)} - \overline{p_i} \right)^2 + \left(p_i^{(2)} - \overline{p_i} \right)^2$ 
# - $p_i^{(j)}$ is the classifiers' scores difference for the $i^{th}$ iteration and fold $j$
# - $\overline{p}_i = \left( p_i^{1} +  p_i^{2} \right) / 2$
# 
# Hopefully this will become clear as you see the code.
# 
# What we need to know is that under the null hypothesis (i.e. both classifiers are statistically equal) the score difference between the two classifiers in each fold is assumed to follow a normal distribution. With this assumption statistic $t$ is assumed to follow a t distribution with 5 degrees of freedom. The proof of this is in the paper itself ;-)
# 
# To test the null hypothesis we compute the value of $t$ and check if it statisfies a t distribution with 5 degree of freedom. Namely we check if the value looks like an outlier or not. If the value stays close enough to 0 then the null hypothesis is satisfied and classifiers are asummed to be equal.
# 
# The thresholds for various t distributions are available on [this web page](http://www.medcalc.org/manual/t-distribution.php).
# 
# Please note that most of the statistical test for classifiers use the accuracy score and that I extend this to Gini without being sure the test still holds (I would have to check the distribution of score differences over several 2-fold iterations).
# 
# Let's look at the code now

# In[ ]:


# Choose seeds for each 2-fold iterations
seeds = [13, 51, 137, 24659, 347]
# Initialize the score difference for the 1st fold of the 1st iteration 
p_1_1 = 0.0
# Initialize a place holder for the variance estimate
s_sqr = 0.0
# Initialize scores list for both classifiers
scores_1 = []
scores_2 = []
diff_scores = []
# Iterate through 5 2-fold CV
for i_s, seed in enumerate(seeds):
    # Split the dataset in 2 parts with the current seed
    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    # Initialize score differences
    p_i = np.zeros(2)
    # Go through the current 2 fold
    for i_f, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
        # Split the data
        trn_x, trn_y = trn.iloc[trn_idx], target.iloc[trn_idx]
        val_x, val_y = trn.iloc[val_idx], target.iloc[val_idx]
        # Train classifiers
        clf1.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)
        clf2.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)
        # Compute scores
        preds_1 = clf1.predict_proba(val_x, num_iteration=clf1.best_iteration_)[:, 1]
        score_1 = roc_auc_score(val_y, preds_1)
        preds_2 = clf2.predict_proba(val_x, num_iteration=clf2.best_iteration_)[:, 1]
        score_2 = roc_auc_score(val_y, preds_2)
        # keep score history for mean and stdev calculation
        scores_1.append(score_1)
        scores_2.append(score_2)
        diff_scores.append(score_1 - score_2)
        print("Fold %2d score difference = %.6f" % (i_f + 1, score_1 - score_2))
        # Compute score difference for current fold  
        p_i[i_f] = score_1 - score_2
        # Keep the score difference of the 1st iteration and 1st fold
        if (i_s == 0) & (i_f == 0):
            p_1_1 = p_i[i_f]
    # Compute mean of scores difference for the current 2-fold CV
    p_i_bar = (p_i[0] + p_i[1]) / 2
    # Compute the variance estimate for the current 2-fold CV
    s_i_sqr = (p_i[0] - p_i_bar) ** 2 + (p_i[1] - p_i_bar) ** 2 
    # Add up to the overall variance
    s_sqr += s_i_sqr
    
# Compute t value as the first difference divided by the square root of variance estimate
t_bar = p_1_1 / ((s_sqr / 5) ** .5) 

print("Classifier 1 mean score and stdev : %.6f + %.6f" % (np.mean(scores_1), np.std(scores_1)))
print("Classifier 2 mean score and stdev : %.6f + %.6f" % (np.mean(scores_2), np.std(scores_2)))
print("Score difference mean + stdev : %.6f + %.6f" 
      % (np.mean(diff_scores), np.std(diff_scores)))


# Again, under the null hypothesis t_bar is assumed to follow a t distribution with 5 degrees of freedom. 
# As such its value should remain in a given confidence interval. 
# 
# This interval is **2.571** for a 5% threshold and **3.365** for a 2% thresholds (value taken from [this web page](http://www.medcalc.org/manual/t-distribution.php))

# In[ ]:


"t_value for the current test is %.6f" % t_bar


# **t value** is within the confidence interval so we can say both classifiers are not statistically different based on a 5 iteration of 2-fold cross validation t test.
# 
# 

# ## k-fold cross-validated paired t test
# I believe this is the most used statistical test.
# 
# In this test we use a simple k-fold cross validation (usually 10) where both classifiers are trained and tested on each fold we then compute the following statistics t: 
# $$
# t = \frac{\overline{p}.\sqrt{n}}{\sqrt{\frac{\sum_{i=1}^{n-1}{\left(p^{(i)} - \overline{p}\right)^2}}{n-1}}}
# $$
# 
# where $\overline{p}$ is the mean difference of scores between classifier 1 and 2 over the folds and $p^{(i)}$ is the score difference for the $i^{th}$ fold.
# 
# Under the null hypothesis, $t$ has a t distribution with k-1 degrees of freedom. The null hypothesis can be rejected if $\left |  t \right | > t_{k-1} $ is greater than for a 95% :
# - for 10-fold CV :  $t_{9, 0.95} =  2.262$ or $t_{9, 0.98} =  2.821$
# - for 7-fold CV : $t_{6, 0.95} =  2.447$ or $t_{6, 0.98} =  3.143$
# - for 5-fold CV : $t_{4, 0.95} =  2.776$ or $t_{4, 0.98} =  3.747$
# 
# Let's check this on a 5-fold CV

# In[ ]:


n_splits = 10 
scores_1 = []
scores_2 = []
oof_1 = np.zeros(len(trn))
oof_2 = np.zeros(len(trn))
diff_scores = []
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=15)
p_i = np.zeros(2)
for i_f, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
    trn_x, trn_y = trn.iloc[trn_idx], target.iloc[trn_idx]
    val_x, val_y = trn.iloc[val_idx], target.iloc[val_idx]
    # Train classifiers
    clf1.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)
    clf2.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=20, verbose=0)
    # Compute scores
    preds_1 = clf1.predict_proba(val_x, num_iteration=clf1.best_iteration_)[:, 1]
    oof_1[val_idx] = preds_1
    score_1 = roc_auc_score(val_y, preds_1)
    preds_2 = clf2.predict_proba(val_x, num_iteration=clf2.best_iteration_)[:, 1]
    score_2 = roc_auc_score(val_y, preds_2)
    oof_2[val_idx] = preds_2
    # keep score history for mean and stdev calculation
    scores_1.append(score_1)
    scores_2.append(score_2)
    diff_scores.append(score_1 - score_2)
    print("Fold %2d score difference = %.6f" % (i_f + 1, diff_scores[i_f]))
# Compute t value
centered_diff = np.array(diff_scores) - np.mean(diff_scores)
t = np.mean(diff_scores) * (n_splits ** .5) / (np.sqrt(np.sum(centered_diff ** 2) / (n_splits - 1)))
print("OOF score for classifier 1 : %.6f" % roc_auc_score(target, oof_1))
print("OOF score for classifier 2 : %.6f" % roc_auc_score(target, oof_2))
print("t statistic for %2d-fold CV = %.6f" % (n_splits, t))


# The t statistic is below the threshold of $t_{9, 0.95} =  2.262$ so both classifiers are statistically equal under the 10-fold CV paired t test.

# You are more than welcome to comment on this and please shout if you see anything wrong in the notebook. As I said my goal is to open a discussion on the subject.
