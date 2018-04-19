
# coding: utf-8

# After publishing a [LGBM kernel](https://www.kaggle.com/ogrellier/lgbm-with-words-and-chars-n-gram), 
# [@Sergei Fironov](https://www.kaggle.com/sergeifironov]) pointed out substantial differences between AUC scores averaged by fold and full OOF AUC, which is mainly due to the fact AUC is not linear.
# 
# So I decided to publish a kernel showing significant distribution differences between each fold predictions.
# 
# I believe we have to tackle this issue before successfully stacking models.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.io import output_file, show, output_notebook
from bokeh.layouts import column, gridplot
from bokeh.plotting import figure
from bokeh.palettes import brewer
output_notebook()
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold


# Read Out-Of-Fold predictions

# In[2]:


oof_dir = '../input/lgbm-with-words-and-chars-n-gram/'
oof = pd.read_csv(oof_dir +"lvl0_lgbm_clean_oof.csv")


# In[3]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
class_preds = [c_ + "_oof" for c_ in class_names]
folds = KFold(n_splits=4, shuffle=True, random_state=1)


# To show what's happening I often like to display F1 scores against probability thresholds. This shows how different each fold behaves for the same threshold. 
# 
# When folds do not behave properly for the same threshold the overall AUC will usually degrade and optimal weights found on OOF data may not yield good results when applied to test predictions.

# In[4]:


figures = []
for i_class, class_name in enumerate(class_names):
    # create a new plot for current class
    # Compute full score :
    full = roc_auc_score(oof[class_names[i_class]], oof[class_preds[i_class]])
    # Compute average score
    avg = 0.0
    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        avg += roc_auc_score(oof[class_names[i_class]].iloc[val_idx], oof[class_preds[i_class]].iloc[val_idx]) / folds.n_splits
    
    s = figure(plot_width=750, plot_height=300, 
               title="F1 score vs threshold for %s full oof %.6f / avg fold %.6f" % (class_name, full, avg))
    
    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        # Get False positives, true positives and the list of thresholds used to compute them
        fpr, tpr, thresholds = roc_curve(oof[class_names[i_class]].iloc[val_idx], 
                                         oof[class_preds[i_class]].iloc[val_idx])
        # Compute recall, precision and f1_score
        recall = tpr
        precision = tpr / (fpr + tpr + 1e-5)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-5)
        # Finally plot the f1_scores against thresholds
        s.line(thresholds, f1_scores, name="Fold %d" % n_fold, color=brewer["Set1"][4][n_fold])
    figures.append(s)

# put the results in a column and show
show(column(figures))


# We can now clearly see problems on severe_toxic, threat and identity_hate.
# 
# Another way to look at this uses the AUC curve directly.

# In[5]:


figures = []
for i_class, class_name in enumerate(class_names):
    # create a new plot for current class
    # Compute full score :
    full = roc_auc_score(oof[class_names[i_class]], oof[class_preds[i_class]])
    # Compute average score
    avg = 0.0
    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        avg += roc_auc_score(oof[class_names[i_class]].iloc[val_idx], oof[class_preds[i_class]].iloc[val_idx]) / folds.n_splits
    
    s = figure(plot_width=400, plot_height=400, 
               title="%s ROC curves OOF %.6f / Mean %.6f" % (class_name, full, avg))
    
    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        # Get False positives, true positives and the list of thresholds used to compute them
        fpr, tpr, thresholds = roc_curve(oof[class_names[i_class]].iloc[val_idx], 
                                         oof[class_preds[i_class]].iloc[val_idx])
        s.line(fpr, tpr, name="Fold %d" % n_fold, color=brewer["Set1"][4][n_fold])
        s.line([0, 1], [0, 1], color='navy', line_width=1, line_dash="dashed")

    figures.append(s)

# put the results in a column and show
show(gridplot(np.array_split(figures, 3)))


# ROC curves are even clearer on that matter as we clearly see curves have the same shape for toxic, obscene and insult while there are significant differences for the last 3 classes.
# 
# The problem now is we don't know if there are even further differences between OOF probabilities and Test predictions. If this were the case this would undermine any stacking attempt. 
# 
# As a matter of fact we can't use ROC curves or F1 scores since Kaggle teams do not let us access test ground truth, and this really is a shame ;-)  
# 
# Can we see anything interesting in the probability distributions themselves?

# In[7]:


# Read submission data 
sub = pd.read_csv(oof_dir +"lvl0_lgbm_clean_sub.csv")
figures = []
for i_class, class_name in enumerate(class_names):
    s = figure(plot_width=600, plot_height=300, 
               title="Probability logits for %s" % class_name)

    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        probas = oof[class_preds[i_class]].values[val_idx]
        p_log = np.log((probas + 1e-5) / (1 - probas + 1e-5))
        hist, edges = np.histogram(p_log, density=True, bins=50)
        s.line(edges[:50], hist, legend="Fold %d" % n_fold, color=brewer["Set1"][4][n_fold])
    
    oof_probas = oof[class_preds[i_class]].values
    oof_logit = np.log((oof_probas + 1e-5) / (1 - oof_probas + 1e-5))
    hist, edges = np.histogram(oof_logit, density=True, bins=50)
    s.line(edges[:50], hist, legend="Full OOF", color=brewer["Paired"][6][1], line_width=3)
    
    sub_probas = sub[class_name].values
    sub_logit = np.log((sub_probas + 1e-5) / (1 - sub_probas + 1e-5))
    hist, edges = np.histogram(sub_logit, density=True, bins=50)
    s.line(edges[:50], hist, legend="Test", color=brewer["Paired"][6][5], line_width=3)
    figures.append(s)

# put the results in a column and show
show(column(figures))


# I can hear you shout :  he should have started here... Agreed !
# 
# Things are cristal clear now and we have a way to make sure what we do in OOF will translate to test probabilities... or not ! 
# 
# 
# As a conclusion I would say that using these OOF outputs for stacking may not be the best idea, especially for severe_toxic and threat.
# 
# Probabilities need to be aligned before any stacking and Im' planning on using a simple LogisticRegression for this purpose.
# 
# More on this later...
# 

# UPDATE: 
# 
# I'm currently running a fork of LightGBM kernel trying to align probabilities.  LogisticRegression gives even worse results but I may be using wrong parameters. I decided to try pd.Series().rank(), which is appropriate for AUC metric. Things are looking better at least on the OOF side but I still need to check the submission predictions. Once the kernel looks right I'll use the output to add a few graphs here.

# In fact I don't need to wait for the kernell to complete since I can simply use the rank() method directly on the OOF data. So let's have a try!

# In[11]:


figures = []
for i_class, class_name in enumerate(class_names):
    s = figure(plot_width=600, plot_height=300, 
               title="Probability logits for %s using rank()" % class_name)

    for n_fold, (_, val_idx) in enumerate(folds.split(oof)):
        probas = (1 + oof[class_preds[i_class]].iloc[val_idx].rank().values) / (len(val_idx) + 1)
        p_log = np.log((probas + 1e-5) / (1 - probas + 1e-5))
        hist, edges = np.histogram(p_log, density=True, bins=50)
        s.line(edges[:50], hist, legend="Fold %d" % n_fold, color=brewer["Set1"][4][n_fold])
    
    oof_probas = (1 + oof[class_preds[i_class]].rank().values) / (oof.shape[0] + 1)
    oof_logit = np.log((oof_probas + 1e-5) / (1 - oof_probas + 1e-5))
    hist, edges = np.histogram(oof_logit, density=True, bins=50)
    s.line(edges[:50], hist, legend="Full OOF", color=brewer["Paired"][6][1], line_width=3)
    
    sub_probas = (1 + sub[class_name].rank().values) / (sub.shape[0] + 1)
    sub_logit = np.log((sub_probas + 1e-5) / (1 - sub_probas + 1e-5))
    hist, edges = np.histogram(sub_logit, density=True, bins=50)
    s.line(edges[:50], hist, legend="Test", color=brewer["Paired"][6][5], line_width=3)
    figures.append(s)

# put the results in a column and show
show(column(figures))


# Ok I think the figures speak for themselves. However I ould still urge you to check your OOF and test predictions since I have a few models that have weird behaviors on the far left or right side of the graph even with rank() !
# 
