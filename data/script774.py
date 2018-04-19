
# coding: utf-8

# With an unsupervised postprocessing technique I could manage to gain around 0.005 in my best ensemble model, which made ~0.1325 to ~1275. (Maybe I could stay in top 1% thanks to it.)

# In[ ]:


import numpy as np
import pandas as pd

df_train =  pd.read_csv('../input/train.csv')
df_test =  pd.read_csv('../input/test.csv')


# After you obtain your best model, use your submission for a better submission:)

# In[ ]:


#REPLACE it with:
#test_label = np.array(pd.read_csv('your_best_solution.csv')["is_duplicate"])
test_label = np.random.rand(len(df_test))


# In[ ]:


from collections import defaultdict

REPEAT = 2 #a reasonable number which can consider your updates iteratively but not ruin the predictions

DUP_THRESHOLD = 0.5 #classification threshold for duplicates
NOT_DUP_THRESHOLD = 0.1 #classification threshold for non-duplicates
#Since the data is unbalanced, our mean prediction is around 0.16. So this is the reason of unbalanced thresholds

MAX_UPDATE = 0.2 # maximum update on the dup probability (a high choice may ruin the predictions)
DUP_UPPER_BOUND = 0.98 # do not update dup probabilities above this threshold
NOT_DUP_LOWER_BOUND = 0.01 # do not update dup probabilities below this threshold
# There is no significant gain between 0.98 and 1.00 for a dup 
# but there is significant loss if it is not really a dup


# This part is nothing magic but basic logic. If A is a duplicate of B and C is a duplicate of B, then A is a duplicate of C.

# In[ ]:


for i in range(REPEAT):
    dup_neighbors = defaultdict(set)

    for dup, q1, q2 in zip(df_train["is_duplicate"], df_train["question1"], df_train["question2"]): 
        if dup:
            dup_neighbors[q1].add(q2)
            dup_neighbors[q2].add(q1)
    
    for dup, q1, q2 in zip(test_label, df_test["question1"], df_test["question2"]): 
        if dup > DUP_THRESHOLD:
            dup_neighbors[q1].add(q2)
            dup_neighbors[q2].add(q1)

    count = 0
    for index, (q1, q2) in enumerate(zip(df_test["question1"], df_test["question2"])): 
        dup_neighbor_count = len(dup_neighbors[q1].intersection(dup_neighbors[q2]))
        if dup_neighbor_count > 0 and test_label[index] < DUP_UPPER_BOUND:
            update = min(MAX_UPDATE, (DUP_UPPER_BOUND - test_label[index])/2)
            test_label[index] += update
            count += 1

    print("Edited:", count)


# This part is the magic part, because having a non-duplicate common neighbor does not mean that these questions are not duplicates but if you read https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs, you may find some insights.
# 
# > Our original sampling method returned an imbalanced dataset with many more true examples of duplicate pairs than non-duplicates. **Therefore, we supplemented the dataset with negative examples.** One source of negative examples were pairs of “related questions” which, although **pertaining to similar topics**, are not truly semantically equivalent.

# In[ ]:


for i in range(REPEAT):
    not_dup_neighbors = defaultdict(set)

    for dup, q1, q2 in zip(df_train["is_duplicate"], df_train["question1"], df_train["question2"]): 
        if not dup:
            not_dup_neighbors[q1].add(q2)
            not_dup_neighbors[q2].add(q1)
    
    for dup, q1, q2 in zip(test_label, df_test["question1"], df_test["question2"]): 
        if dup < NOT_DUP_THRESHOLD:
            not_dup_neighbors[q1].add(q2)
            not_dup_neighbors[q2].add(q1)

    count = 0
    for index, (q1, q2) in enumerate(zip(df_test["question1"], df_test["question2"])): 
        dup_neighbor_count = len(not_dup_neighbors[q1].intersection(not_dup_neighbors[q2]))
        if dup_neighbor_count > 0 and test_label[index] > NOT_DUP_LOWER_BOUND:
            update = min(MAX_UPDATE, (test_label[index] - NOT_DUP_LOWER_BOUND)/2)
            test_label[index] -= update
            count += 1

    print("Edited:", count)


# Prepare the submission

# In[ ]:


submission = pd.DataFrame({'test_id':df_test["test_id"], 'is_duplicate':test_label})
#submission.to_csv('submission.csv', index=False)


# I will also provide the repository of my relatively lightweight solution when I have time.
