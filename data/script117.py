
# coding: utf-8

# ## Introduction

# There are two very different strong baselines currently in the kernels for this competition:
#     
# - An *LSTM* model, which uses a recurrent neural network to model state across each text, with no feature engineering
# - An *NB-SVM* inspired model, which uses a simple linear approach on top of naive bayes features
# 
# In theory, an ensemble works best when the individual models are as different as possible. Therefore, we should see that even a simple average of these two models gets a good result. Let's try it! First, we'll load the outputs of the models (in the Kaggle Kernels environment you can add these as input files directly from the UI; otherwise you'll need to download them first).

# In[3]:


import numpy as np, pandas as pd

f_lstm = '../input/improved-lstm-baseline-glove-dropout/submission.csv'
f_nbsvm = '../input/nb-svm-strong-linear-baseline/submission.csv'


# In[4]:


p_lstm = pd.read_csv(f_lstm)
p_nbsvm = pd.read_csv(f_nbsvm)


# Now we can take the average of the label columns.

# In[5]:


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p_lstm.copy()
p_res[label_cols] = (p_nbsvm[label_cols] + p_lstm[label_cols]) / 2


# And finally, create our CSV.

# In[6]:


p_res.to_csv('submission.csv', index=False)


# As we hoped, when we submit this to Kaggle, we get a great result - much better than the individual scores of the kernels we based off. This is currently the best Kaggle kernel submission that runs within the kernels sandbox!
