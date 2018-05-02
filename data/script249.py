
# coding: utf-8

# ## Summary
# 
# I tried out a neural network for classification because Machine Learning. With the large number of samples and 1000 samples per class, the task seems to be well-suited for Deep Learning. As expected, the model achieved significant results. There is still some room for improvement, so maybe combining the model with XGBoost can help there.
# 
# ## Approach
# 
# I used the subm.csv output file from ZFTurbo's [Greedy children baseline [0.8168]](https://www.kaggle.com/zfturbo/greedy-children-baseline-0-8168) kernel. As features, I used the children's wishlists and their id resulting in 11 features.

# In[ ]:


import pandas as pd
from keras.layers import Dense
from keras.models import Sequential

INPUT_PATH = '../input/'

child_wishes = pd.read_csv(INPUT_PATH + 'santa-gift-matching/child_wishlist.csv', header=None)
targets = pd.read_csv(INPUT_PATH + 'greedy-children-baseline-0-8168/subm.csv')['GiftId']
child_wishes['target'] = targets


# We only need 10% of the available data to predict the rest because Artificial Intelligence is very powerful. So, now we make sure that 10% of the labels of each class appear in the training and validation set.

# In[ ]:


# For each of the 1000 gifts, put 80 samples are in the training data and 20 in 
# the validation data
train_data = pd.DataFrame()
valid_data = pd.DataFrame()
for gift_id in range(1000):
    train_split = child_wishes.loc[child_wishes['target'] == gift_id].iloc[:80]
    valid_split = child_wishes.loc[child_wishes['target'] == gift_id].iloc[80:100]
    train_data = train_data.append(train_split)
    valid_data = valid_data.append(valid_split)

# Shuffle the training data
train_data = train_data.sample(frac=1)


# In[ ]:


# Assign the inputs and targets 
y_train = pd.get_dummies(train_data['target']).values
X_train = train_data.drop('target', axis=1).values
y_valid = pd.get_dummies(valid_data['target']).values
X_valid = valid_data.drop('target', axis=1).values

print('Shapes: X_train: %s, y_train: %s, X_valid: %s, y_valid: %s' % 
      (X_train.shape, y_train.shape, X_valid.shape, y_valid.shape))


# In[ ]:


# Train the model!
model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(11,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(1000, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=2)


# Looks good! We see significant progress being made during training. Now, let's unleash the beast!

# In[ ]:


# Predict the whole dataset
X_all = child_wishes.drop('target', axis=1).values
predicted_probs = model.predict_proba(X_all)


# The problem here is, that there is no guarantee that the model's predictions conform to the 1000 kids per gift constraint and the twin constraint. To keep things simple, we now assign the gifts greedily to the kids based on the probabilites predicted by the model. Deep Learning may yield further improvement at this step.

# In[ ]:


# Assign gifts greedily until there are no gifts left
child_to_gift = {}
# Keep track of how many gifts there are left to give
gift_counts = dict((gift_id, 1000) for gift_id in range(1000))
available_gifts = gift_counts.keys()

for i, gift_probs in enumerate(predicted_probs):
    child_id = child_wishes.iloc[i, 0]

    # Ignore children (twins) that already have a gift
    if child_id in child_to_gift:
        continue

    candidate_gifts = available_gifts
    # If this is a twin we need the gift two times
    if child_id < 4000:
        candidate_gifts = [g for g in available_gifts if gift_counts[g] >= 2]

    # Get the candidate gift with the highest probability
    gift_id = max(candidate_gifts, key=lambda gift_id: gift_probs[gift_id])

    child_to_gift[child_id] = gift_id
    gift_counts[gift_id] -= 1

    # If this is a twin, assign the gift to his other sibling as well
    if child_id < 4000:
        sibling_id = child_id + 1 if child_id % 2 == 0 else child_id - 1
        child_to_gift[sibling_id] = gift_id
        gift_counts[gift_id] -= 1

    # Recalculate the available gifts
    available_gifts = [g for g in available_gifts if gift_counts[g] > 0]
    
pred = sorted(child_to_gift.items(), key= lambda t: t[0])


# Done! We have our final predictions. Let's find out by how much this model outperforms the others.

# In[ ]:


import numpy as np
from collections import Counter

n_children = 1000000  # n children to give
n_gift_type = 1000  # n types of gifts available
n_gift_quantity = 1000  # each type of gifts are limited to this quantity
n_gift_pref = 10  # number of gifts a child ranks
n_child_pref = 1000  # number of children a gift ranks
twins = int(0.004 * n_children)  # 0.4% of all population, rounded to the closest even number
ratio_gift_happiness = 2
ratio_child_happiness = 2


def avg_normalized_happiness(pred, child_pref, gift_pref):
    # check if number of each gift exceeds n_gift_quantity
    gift_counts = Counter(elem[1] for elem in pred)
    for count in gift_counts.values():
        assert count <= n_gift_quantity

    # check if twins have the same gift
    for t1 in range(0, twins, 2):
        twin1 = pred[t1]
        twin2 = pred[t1 + 1]
        assert twin1[1] == twin2[1]

    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type)

    for row in pred:
        child_id = row[0]
        gift_id = row[1]

        # check if child_id and gift_id exist
        assert child_id < n_children
        assert gift_id < n_gift_type
        assert child_id >= 0
        assert gift_id >= 0
        child_happiness = (n_gift_pref - np.where(gift_pref[child_id] == gift_id)[0]) * ratio_child_happiness
        if not child_happiness:
            child_happiness = -1

        gift_happiness = (n_child_pref - np.where(child_pref[gift_id] == child_id)[0]) * ratio_gift_happiness
        if not gift_happiness:
            gift_happiness = -1

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id] += gift_happiness

    # print(max_child_happiness, max_gift_happiness
    print('normalized child happiness=',
          float(total_child_happiness) / (float(n_children) * float(max_child_happiness)), \
          ', normalized gift happiness', np.mean(total_gift_happiness) / float(max_gift_happiness * n_gift_quantity))
    return float(total_child_happiness) / (float(n_children) * float(max_child_happiness)) + np.mean(
        total_gift_happiness) / float(max_gift_happiness * n_gift_quantity)


# In[ ]:


gift_pref = pd.read_csv(INPUT_PATH + 'santa-gift-matching/child_wishlist.csv', header=None).drop(0, 1).values
child_pref = pd.read_csv(INPUT_PATH + 'santa-gift-matching/gift_goodkids.csv', header=None).drop(0, 1).values
score = avg_normalized_happiness(pred, child_pref, gift_pref)


# In[ ]:


print(score)


# ## Conclusion
# 
# Wow! A simple first try with Deep Learning already achieved a whopping -4% Average Normalized Happiness. There is still room for improvement. For example, the 11 features could be preprocessed by a Convolutional Neural Network.. Also, the greedy matching at the end might be replaced with a Blockchain mechanism. But all in all, this path looks very promising. I am curious as to what further enhancements of this approach will yield.
