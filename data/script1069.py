
# coding: utf-8

# Welcome Kaggler!
# 
# With this notebook I invite you to travel with me through a gradient attack using a simpler model than inception v3 and the MNIST data. When I read about the competition I was really excited to learn, how to fool a machine learning model. But I felt a bit lost given so much new content: 
# 
# * inception model
# * math and concepts behind attacks and defenses
# * papers to read 
# 
# ...
# 
# So I decided that I first like to learn, how to fool a more simpler model than inception v3. For my own I started with binary logistic regression and I was surprised and as well shocked how easy this was with the scikit-learn breast-cancer dataset. As this competition has multiple classes and works with images I decided to switch to multiclass logistic regression. And now I like to share my results and what I have learned with you. 
# 
# If you like my kernel, you can make me very happy with an upvote! ;-) 
# 

# ## Walktrough contents
# 
# * Loading data and packages
# * Attack methods and class
# * Training the model
# * Non-Targeted-Attack: 
#     * Maximizing the output-target discrepance
#     * Natural fooling targets (Reworked)
# * Targeted-Attack: 
#     * One image example
#     * Natural vs non-natural fooling targets & accuracy score breakdown for all images (Reworked)
# * Comparison to Fast Gradient Method: 
#     * Thoughts: Maximizing discrepance vs. minimizing likelihood
# * The Jacobian enters the door
# * Conclusion
# 
# **Latest Changes:**
# * Rework: Natural vs non-natural fooling
# * Skip LDA as kernel has already too much content

# ## Loading packages

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import datasets as dt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import pandas as pd
import numpy as np
import seaborn as sns


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

from skimage.io import imread, imshow

X,y = dt.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)


# To see that are trained model can be good at prediction of original images but bad for perturbed I split the data into train and test. Then we can compare the prediction for original test data and perturbed test data. ;-)   

# ## Loading data
# 
# We will use the digits of the digit-recognizer competition. Add it to your data source and it should be in your input folder. Check: 

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


df = pd.read_csv("../input/digit-recognizer/train.csv")
df.head()


# In[ ]:


y = df.label.values
X = df.drop("label",axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# Before we start with building targeted and non-targeted attacks, let's have a look at the first digits of the test set:

# In[ ]:


fig1, ax1 = plt.subplots(1,15, figsize=(15,10))
for i in range(15):
    ax1[i].imshow(X_test[i].reshape((28,28)))
    ax1[i].axis('off')
    ax1[i].set_title(y_test[i])


# Some digits are not easily recognized by human eyes... 

# ## Code - methods and classes
# 
# As I don't like to fill this kernel with same code used for different cases again and again, I will store it here. I hope, this makes it also easier for you to play with this kernel if you like to fork :-)

# ### Attack methods 

# In[ ]:


def calc_output_weighted_weights(output, w):
    for c in range(len(output)):
        if c == 0:
            weighted_weights = output[c] * w[c]
        else:
            weighted_weights += output[c] * w[c]
    return weighted_weights

def targeted_gradient(foolingtarget, output, w):
    ww = calc_output_weighted_weights(output, w)
    for k in range(len(output)):
        if k == 0:
            gradient = foolingtarget[k] * (w[k]-ww)
        else:
            gradient += foolingtarget[k] * (w[k]-ww)
    return gradient

def non_targeted_gradient(target, output, w):
    ww = calc_output_weighted_weights(output, w)
    for k in range(len(target)):
        if k == 0:
            gradient = (1-target[k]) * (w[k]-ww)
        else:
            gradient += (1-target[k]) * (w[k]-ww)
    return gradient

def non_targeted_sign_gradient(target, output, w):
    gradient = non_targeted_gradient(target, output, w)
    return np.sign(gradient)


# ### Attack class

# In[ ]:


class Attack:

    def __init__(self, model):
        self.fooling_targets = None
        self.model = model
    
    def prepare(self, X_train, y_train, X_test, y_test):
        self.images = X_test
        self.true_targets = y_test
        self.num_samples = X_test.shape[0]
        self.train(X_train, y_train)
        print("Model training finished.")
        self.test(X_test, y_test)
        print("Model testing finished. Initial accuracy score: " + str(self.initial_score))
    
    def set_fooling_targets(self, fooling_targets):
        self.fooling_targets = fooling_targets
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.weights = self.model.coef_
        self.num_classes = self.weights.shape[0]

    def test(self, X_test, y_test):
        self.preds = self.model.predict(X_test)
        self.preds_proba = self.model.predict_proba(X_test)
        self.initial_score = accuracy_score(y_test, self.preds)
    
    def create_one_hot_targets(self, targets):
        self.one_hot_targets = np.zeros(self.preds_proba.shape)
        for n in range(targets.shape[0]):
            self.one_hot_targets[n, targets[n]] = 1
            
    def attack(self, attackmethod, epsilon):
        perturbed_images, highest_epsilon = self.perturb_images(epsilon, attackmethod)
        perturbed_preds = self.model.predict(perturbed_images)
        score = accuracy_score(self.true_targets, perturbed_preds)
        return perturbed_images, perturbed_preds, score, highest_epsilon

    def perturb_images(self, epsilon, gradient_method):
        perturbed = np.zeros(self.images.shape)
        max_perturbations = []
        for n in range(self.images.shape[0]):
            perturbation = self.get_perturbation(epsilon, gradient_method, self.one_hot_targets[n], self.preds_proba[n])
            perturbed[n] = self.images[n] + perturbation
            max_perturbations.append(np.max(perturbation))
        highest_epsilon = np.max(np.array(max_perturbations))
        return perturbed, highest_epsilon
    
    def get_perturbation(self, epsilon, gradient_method, target, pred_proba):
        gradient = gradient_method(target, pred_proba, self.weights)
        inf_norm = np.max(gradient)
        perturbation = epsilon/inf_norm * gradient
        return perturbation
        
    def attack_to_max_epsilon(self, attackmethod, max_epsilon):
        self.max_epsilon = max_epsilon
        self.scores = []
        self.epsilons = []
        self.perturbed_images_per_epsilon = []
        self.perturbed_outputs_per_epsilon = []
        for epsilon in range(0, self.max_epsilon):
            perturbed_images, perturbed_preds, score, highest_epsilon = self.attack(attackmethod, epsilon)
            self.epsilons.append(highest_epsilon)
            self.scores.append(score)
            self.perturbed_images_per_epsilon.append(perturbed_images)
            self.perturbed_outputs_per_epsilon.append(perturbed_preds)
        


# ## Training the model
# 
# First of all we need a model for multiclass logistic regression:

# In[ ]:


model = LogisticRegression(multi_class='multinomial', solver='lbfgs', fit_intercept=False)


# And we will pass it to our class and call prepare. This way we train our model on training data and we will obtain the initial accuracy score on test data. Later on we want to break down this score by perturbing the test data.

# In[ ]:


attack = Attack(model)
attack.prepare(X_train, y_train, X_test, y_test)


# Ok, around 90 percent of test data was classified correctly and for us this is sufficient to play with. In multiclass logistic regression the probability that the model outputs $y_{n}$ of $N$ inputs $x_{n}$ matches their targets $t_{n}$ is given by:
# 
# $$ 
# p(t|y(x,w)) = \prod_{n=1}^{N} \prod_{k=1}^{K} y_{n,k}^{t_{n,k}} 
# $$
# 
# We assumed that inputs and class memberships are independent and identically distributed. The target $t_{n}$ of one input $x_{n}$ is a vector with K elements following one-hot-encoding (the true label class is 1, all others are 0). Maximizing the probability of matches above is also called the maximum likelihood approach. Each class in a multiclass logistic regression has its own weight vector and inputs are passed with weights through the softmax function to obtain the model output:
# 
# $$ 
# y_{n,k} = \frac{\exp(w_{k}^{T}x_{n})} {\sum_{c=1}^{K}\exp(w_{c}^{T}x_{n})}
# $$

# Let's check if we have as much weight vectors as classes:

# In[ ]:


weights = attack.weights
weights.shape


# In[ ]:


num_classes = len(np.unique(y_train))
num_classes


# :-)

# ## Non-targeted Attack: Maximizing output-target discrepance
# 
# In analogy to the maximum likelihood approach I define a discrepancy function as my objective. Usually the likelihood gives us the probability that our model output y matches the target t. Thus maximizing the likelihood yields us the best matches. Creating an attack one has to think the other way round: We want to maximize the probability that the outputs do not match the targets. Let's write down this descrepance function by using the multinoulli distribution again:
# 
# $$ D(t|y(w,x)) = \prod_{n=1}^{N} \prod_{k=1}^{K} y_{n,k}^{1-t_{n,k}} $$

# Notice that in contrast to the likelihood function we now have 1-t instead of t. If our true target $t_{n,k}$ of class k for input $x_{n}$ is 1, but our model predicts $y_{n,k}=0$, then we have $0^{1-1}=1$ and vice versa $1^{1-0}=1$.  

# To train a model we usually would maximize the likelihood with respect to the weight parameters whereas inputs are fixed. In our case we already have a trained model and fixed weights. But we can add tiny perturbations to our input images such that we maximize our discrepance function. Let's do that and for making things simpler we use the log! :-)
# 
# $$\nabla_{x} log D = \nabla_{x} \sum_{k=1}^{K} (1-t_{k}) \log y_{k} $$
# 
# $$ \partial_{x} \log D = \sum_{k=1}^{K} \frac {\partial D}{\partial y_{k}} \frac {\partial y_{k}}{\partial x} $$

# The first part is easy:
# 
# $$ \frac {\partial D}{\partial y_{k}}  = \frac {1-t_{k}} {y_{k}}$$

# For the second part we should keep in mind the gradient of activations yields the class weight vector:
# 
# $$ a_{c} = \vec{w_{c}}^{T} \vec{x} = \vec{x}^{T} \vec{w_{c}} $$
# 
# $$ \nabla_{x} a_{c} = \vec{w_{c}} $$
# 
# $$ y_{k} = \frac{\exp(a_{k})} {\sum_{c=1}^{K}\exp(a_{c})}$$

# Now this second part is a bit cumbersome (... hopefully I didn't make a mistake ...):
# 
# $$ \frac {\partial y_{k}}{\partial x} = \frac{\Sigma \cdot \partial_{x} \exp(a_{k}) - \exp(a_{k}) \partial_{x} \Sigma}{\Sigma^{2}} = \frac {\sum_{c=1}^{K} \exp(a_{c}) \cdot \vec{w}_{k} \exp(a_{k}) - \exp(a_{k}) \sum_{c=1}^{K} \vec{w}_{c} \exp(a_{c})} {\sum_{c=1}^{K}\exp(a_{c}) \cdot \sum_{c=1}^{K}\exp(a_{c})} $$
# 
# $$ \frac {\partial y_{k}}{\partial x} =  y_{k} \vec{w}_{k} - y_{k} \cdot \sum_{c=1}^{K} y_{c} \vec{w}_{c}  $$

# Finally I end up with:
# 
# $$ \nabla_{x} \log D = \sum_{k=1}^{K} (1-t_{k}) \cdot  (\vec{w}_{k} - \sum_{c=1}^{K} y_{c} \vec{w}_{c}) $$

# Let's try to understand this: For the true label the summand is 0 whereas all other classes contribute to the gradient with their class weight vector $\vec{w_{k}}$ reduced by an output "weighted" sum of all other class weights. What does that mean? ...

# To maximize the discrepance for playing around we can use gradient ascent. Though we have to find a sufficient rate $\eta$. Given an input $x_{m}$ we will then add a perturbation with:

# $$ x_{p, m} = x_{m} + \delta x_{m} = x_{m} + \eta \cdot \nabla_{x_{m,c}} D_{m,c} $$
# 
# $$ x_{p, m} = x_{m} + \eta \sum_{k=1}^{K} (1-t_{m,k}) \cdot  (\vec{w}_{k} - \sum_{c=1}^{K} y_{m,c} \vec{w}_{c}) $$

# Let's give it a try! :-)

# ### Adding tiny perturbations

# First of all we need to calculate the perturbations for each image in the test set. To do this we have to transform our true targets to one-hot-targets and call attack :-). As I want to see, how much epsilon we need to create a good breakdown, I use the attack_to_max_epsilon method. 

# In[ ]:


attack.create_one_hot_targets(y_test)
attack.attack_to_max_epsilon(non_targeted_gradient, 30)
non_targeted_scores = attack.scores


# In[ ]:


plt.figure()
plt.plot(attack.epsilons, attack.scores, 'g*')
plt.ylabel('accuracy_score')
plt.xlabel('epsilon')
plt.title('Accuracy score breakdown - non-targeted attack')


# Uii, the threshold is given by a max of 16 pixel that are allowed to be added as perturbation per pixel per image. Given this $\epsilon$ we would end up with a model that still predicts around 40 % correctly. If we would use max $\epsilon=30$ the model would fail with almoast 90 % digits in the test set :-) . Let's have a look at one example of successful fooling for a range of epsilons until max of $\epsilon = 16$.

# In[ ]:


eps = 16
attack.epsilons[eps]


# We need the perturbed images as well as the fooling results of that epsilon:

# In[ ]:


example_images = attack.perturbed_images_per_epsilon[eps]
example_preds = attack.perturbed_outputs_per_epsilon[eps]


# And I will store results in a pandas dataframe such that we can easily find successful foolings:

# In[ ]:


example_results = pd.DataFrame(data=attack.true_targets, columns=['y_true'])
example_results['y_fooled'] = example_preds
example_results['y_predicted'] = attack.preds
example_results['id'] = example_results.index.values
example_results.head()


# In[ ]:


success_df = example_results[example_results.y_fooled != example_results.y_true]
success_df.head()


# Ok, we will choose one of these successful examples and plot its related perturbed image over a range of epsilons: 

# In[ ]:


example_id = success_df.id.values[0]
example_id


# In[ ]:


fig2, ax2 = plt.subplots(4,4, figsize=(10,10))
for i in range(4):
    for j in range(4):
        image = attack.perturbed_images_per_epsilon[i*4 + j][example_id]
        y_fooled = attack.perturbed_outputs_per_epsilon[i*4 + j][example_id]
        epsilon = attack.epsilons[i*4 +j]
        ax2[i,j].imshow(image.reshape((28,28)))
        ax2[i,j].axis('off')
        ax2[i,j].set_title("true: " + str(y_test[example_id]) + ", fooled: " + str(y_fooled)  + "\n" 
                           + "epsilon: " + str(epsilon))


# Yeah! :-) We can still see the true target and not the fooling target. That's amazing. But we can also see, that the background has increased intensitiy. Let's visualize the difference between the original true label and the adversarial image for $\epsilon = 16$: 

# In[ ]:


fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15,5))
axB.imshow(example_images[example_id].reshape((28,28)), cmap='Greens')
axB.set_title("Non-targeted attack result: " + str(example_preds[example_id]))
axA.imshow(X_test[example_id].reshape((28,28)), cmap='Greens')
axA.set_title("True label: " + str(y_test[example_id]))
axC.imshow((X_test[example_id]-example_images[example_id]).reshape((28,28)), cmap='Reds')
axC.set_title("Perturbation: epsilon 16")


# ### The gradient travel guide - natural fooling targets
# 
# I'm happy that it was possible to fool our model but it's still diffuse and unclear where the one-step-gradient guides us through (remember we do not iterate with gradient ascent, we just take one step and size is given by strength of gradient times eta). I assume that some numbers are closer to each other in weight space than to others. As the model training draws decision boundaries dependent on the quality of the input data and flexibility of model architecture, there will be regions where a 3 is not predicted as 3 but as 8. Those regions where the model makes an incorrect prediction. And I think, that there are preffered numbers to be wrong predictions given a digit input image. Perhaps the fooling gradients drives us to those "natural" fooling target numbers? 

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='y_fooled', data=example_results[example_results.y_true != example_results.y_fooled])


# Ok, we see that 8 was selected most often as fooling target. But 9, 3, 5 and 2 have high counts as well in contrast to 0, 1, 6 and 7. If our assumption is true that the gradient drives us to targets where the model tends to fail in prediction we should see a similar pattern of counts for wrong predictions:

# In[ ]:


wrong_predictions = example_results[example_results.y_true != example_results.y_predicted]
wrong_predictions.shape


# In[ ]:


X_test.shape


# Ok, so out of 16800 samples, the model failed to predict around 1600. That's why our intital accuracy score is close to 90 % (means 10 % failing). Now, which digit was selected as wrong prediction result most often?

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='y_predicted', data=wrong_predictions)


# Yes, that's the same pattern as for the fooling targets. As this is caused by the difficulty of our model to draw good decision boundaries we should see this pattern as well for the true labels of those digits that were wrong predicted:

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='y_true', data=wrong_predictions)


#  Now I want to see it in more detail: Which are the natural fooling targets (for successful foolings) for each digit?

# In[ ]:


attacktargets = example_results[example_results.y_true != example_results.y_fooled].groupby('y_true').y_fooled.value_counts().unstack()
attacktargets = attacktargets.fillna(0.0)

f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(attacktargets, annot=True, ax=ax)


# We found out that each digit has its natural fooling target.

# ## Targeted-Attack
# 
# We have seen that fooling the multiclass logistic regression model was easy with gradient ascent and the only cumbersome part was to calculate the gradient with respect to the inputs of our discrepance function. Instead of forcing the function to yield the maximum discrepance we could have also construct it such that outputs have to match a specific false target. To emphasize the difference to the true target $t_{n,k}$ let's call it $f_{n,k}$ for "fooling" target.
# 
# $$ D(t|y(w,x)) = \prod_{n=1}^{N} \prod_{k=1}^{K} y_{n,k}^{f_{n,k}} $$
# 
# Luckily we do not have to calculate everything again, the only thing that changes is:
# 
# $$ \frac {\partial D}{\partial y_{k}}  = \frac {f_{k}} {y_{k}} $$
# 
# Consequently we have:
# 
# $$ \nabla_{x} \log D = \sum_{k=1}^{K} f_{k} \cdot  (\vec{w}_{k} - \sum_{c=1}^{K} y_{c} \vec{w}_{c}) $$
# 
# $$ x_{p, m} = x_{m} + \eta \sum_{k=1}^{K} f_{k} \cdot  (\vec{w}_{k} - \sum_{c=1}^{K} y_{m,c} \vec{w}_{c}) $$

# ### One example image
# 
# To play around, let's select one input of $X_{test}$ and try to make targeted attacks for each class $f_{k}$ except for the true label target $t_{k}$. 

# In[ ]:


example = X_test[0]
imshow(example.reshape((28,28)), cmap='Greens')


# In[ ]:


print("true label target: " + str(y_test[0]))


# ### Generating the fooling/attack classes and targets

# First of all, we need some fooling targets. For our example digit all others are possible:

# In[ ]:


fooling_classes = []
for k in range(num_classes):
    if k != y_test[0]:
        fooling_classes.append(k)
fooling_classes


# In[ ]:


foolingtargets = np.zeros((len(fooling_classes), num_classes))
for n in range(len(fooling_classes)):
    foolingtargets[n,fooling_classes[n]] = 1
foolingtargets


# ### Attacking the model
# 
# I will force the attack to success by allowing an epsilon high enough to yield all targets. This way we can still find out, if we can see the true label or the fooling target.

# In[ ]:


eps=100
targeted_perturbed_images = []
targeted_perturbed_predictions = []
for fooling_target in foolingtargets:   
    targeted_perturbation = attack.get_perturbation(eps, targeted_gradient, fooling_target, attack.preds_proba[0])
    targeted_perturbed_image = X_test[0] + targeted_perturbation
    targeted_perturbed_prediction = attack.model.predict(targeted_perturbed_image.reshape(1, -1))
    targeted_perturbed_images.append(targeted_perturbed_image)
    targeted_perturbed_predictions.append(targeted_perturbed_prediction)


# In[ ]:


targeted_perturbed_predictions


# In[ ]:


fig3, ax3 = plt.subplots(3,3, figsize=(9,9))
for i in range(3):
    for j in range(3):
        ax3[i,j].imshow(targeted_perturbed_images[i*3+j].reshape((28,28)))
        ax3[i,j].axis('off')
        ax3[i,j].set_title("fooling result: " + str(targeted_perturbed_predictions[i*3+j][0]))


# Even though we can see high background noise the true label is not destroyed. I can still see the true label with my eyes whereas the model predicts the desired fooling target (0 to 9, except true label). :-) That's cool.

# ## Natural vs. non-natural targeted attack
# 
# Now, I like to see what happens with the accuracy score if we fool the model for each image in the test set. 
# By analyzing non-targeted attacks we found that some digits are more used as "fooling" target than others and that each digit has its fooling digit counterpart. I assume that fooling takes place in regions where the model fails to draw good decision boundaries. Using targeted attack we should see that we can breakdown the accuracy score easier with natural fooling targets than with the other digits. Let's try this! :-)

# ### Prepare natural and non-natural fooling targets
# 
# The gradient travel guide showed us the occurences of fooling target digits for each true digit. The highest count stands for the natural fooling target whereas the lowest corresponds to the non-natural fooling target. Given the heatmap we could create the targets by argmin and argmax per row (y_true) as follows:

# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(attacktargets, annot=True, ax=ax)


# In[ ]:


natural_targets_dict = {}
non_natural_targets_dict = {}
for ix, series in attacktargets.iterrows():
    natural_targets_dict[ix] = series.argmax()
    non_natural_targets_dict[ix] = series.drop(ix).argmin()


# In[ ]:


natural_targets_dict


# In[ ]:


natural_foolingtargets = np.zeros((y_test.shape[0]))
non_natural_foolingtargets = np.zeros((y_test.shape[0]))

for n in range(len(natural_foolingtargets)):
    target = y_test[n]
    natural_foolingtargets[n] = natural_targets_dict[target]
    non_natural_foolingtargets[n] = non_natural_targets_dict[target]


# In[ ]:


attack.create_one_hot_targets(natural_foolingtargets.astype(np.int))
attack.attack_to_max_epsilon(targeted_gradient, 30)
natural_scores = attack.scores
attack.create_one_hot_targets(non_natural_foolingtargets.astype(np.int))
attack.attack_to_max_epsilon(targeted_gradient, 30)
non_natural_scores = attack.scores


# In[ ]:


plt.figure()
nf, = plt.plot(attack.epsilons, natural_scores, 'g*', label='natural fooling')
nnf, = plt.plot(attack.epsilons, non_natural_scores, 'b*', label='non-natural fooling')
plt.legend(handles=[nf, nnf])
plt.ylabel('accuracy_score')
plt.xlabel('epsilon')
plt.title('Accuracy score breakdown: natural vs non-natural targeted attack')


# Ahhh! :-) We can clearly see that it was easier to fool the model with natural fooling targets.

# ## Comparison to Fast Gradient Method
# 
# 
# So far we have used gradient ascent to maximize the probability of no-matches (non-targeted & targeted) and we derived the gradient with respect to inputs for objectives that are only slightly changed in comparison to the model defining likelihood function. This way we are very close to the Fast Gradient Method given in the cleverhans library:
# 

# * Maximizing probability of no-matches:
# 
# $$ x_{p, m} = x_{m} + \delta x_{m} = x_{m} + \eta \cdot \nabla_{x_{m}} D(f_{m}|y_{m}(x_{m},w)) $$
# 
# * Fast Gradient Method:
# 
# $$ x_{p,m} = x_{m} + \delta x_{m} = x_{m} + \eta \cdot sign (\nabla_{x_{m}} J(w,x,y)) $$

# Whereas $J(w,x,y)$ stands for the cost function used to train the model.  In many cases this is $E=- \log (L(t|y(x,w))$, the negative log likelihood function. One might think that this should also be the case for this example and that we could have also simply maximize the cost funtion. As this would be the same as to minimize the log-likelihood, one might think:

# ** ... instead of defining a discrepance function one could have also just minimize the likelihood function to yield the lowest probability of matches... **

# Ohoh! Is this true? Sounds so nice, but... 

# ### Maximizing discrepance = minimizing likelihood?
# 
# Normally we use the likelihood to maximize the probability and we want to count those outputs $y_{n,c}$ related to target-values of $t_{n,c}=1$. Look at one example image $m$ that belongs to class 2 of 3, then for this one we would have:
# 
# $$ l = y_{1}^{0} * y_{2}^{1} * y_{3}^{0} = y_{2} $$
# 
# Hence this image contribute to the overall product over $n$ by a factor of $y_{2}$. Consequently each image $n$ would give us a y-factor of its true class whereas all the others only give us a factor of 1 such that it does not disturb our nice probability. If we would now use the same likelihood to represent the probability of no-matches we encounter a problem: To minimize the likelihood the only thing that can be done now is to set $y_{2}=0$ for image m. We would do this for all outputs that enter the likelihood by themselves. But what is with all the others? What to do with $y_{1}$ and $y_{3}$ for image m? Looking for a minimum of $y_{2}$ we could select anything for $y_{1}$ and $y_{3}$ as they just stay 1. That's ill-defined in contrast to our discrepance probability. There the situation for image m would look like this:
# 
# $$ l = y_{1}^{1} * y_{2}^{0} * y_{3}^{1} = y_{1} * y_{3} $$
# 
# And also our targeted-attack looks well for image m, if we define class 1 as our fooling class:
# 
# $$ l = y_{1}^{1} * y_{2}^{0} * y_{3}^{0} = y_{1} $$
# 
# We have made sure that only those output-values enter the likelihood that are not related to the true target and we made sure that they carry their values and not just a value of 1. 

# ## Comparison to "discrepance" Gradient Sign Method

# Some small experiment: Let's only take the sign of the gradient of our attack. I think we will not be as goog as we could be with the full gradient:

# In[ ]:


attack.create_one_hot_targets(y_test)
attack.attack_to_max_epsilon(non_targeted_sign_gradient, 30)


# In[ ]:


plt.figure()
gm, = plt.plot(attack.epsilons, non_targeted_scores, 'g*', label='gradient method')
gsm, = plt.plot(attack.epsilons, attack.scores, 'r*', label='gradient sign method')
plt.ylabel('accuracy_score')
plt.xlabel('eta')
plt.legend(handles=[gm, gsm])
plt.title('Accuracy score breakdown')


# Uiih!! :-o That's a surprise. 

# ### One step further: The Jacobian matrix

# What if we would use the likelihood instead of the discrepance function to yield the perturbations? If you look at the equations, you can find a clear similarity:

# $$ D(t|y(w,x)) = \prod_{n=1}^{N} \prod_{k=1}^{K} y_{n,k}^{f_{n,k}} $$
# 
# $$ \nabla_{x} \log D = \sum_{k=1}^{K} f_{k} \cdot  (\vec{w}_{k} - \sum_{c=1}^{K} y_{c} \vec{w}_{c}) $$
# 
# $$ L(t|y(x,w)) = \prod_{n=1}^{N} \prod_{k=1}^{K} y_{n,k}^{t_{n,k}} $$
# 
# $$ \nabla_{x} \log L = \sum_{k=1}^{K} t_{k} \cdot  (\vec{w}_{k} - \sum_{c=1}^{K} y_{c} \vec{w}_{c}) $$

# The difference is the use of either the true label $t_{n,k}$ or the fooling target $f_{n,k}$. What remains the same is the gradient of output $y_{k}$ of class $k$ with respect to all features $x$ (pixels) of one single image.
# 
# $$ \frac {\partial y_{k}}{\partial x} =  y_{k} \vec{w}_{k} - y_{k} \cdot \sum_{c=1}^{K} y_{c} \vec{w}_{c}  $$

# Taking the derivative of the outputs for each class $\partial y_{k}$ with respect to the input features $\partial x_{d}$ (with d in range for 1 to 64 pixel) we have arrived at the **Jacobian matrix**: 

# $$
# \begin{bmatrix}
# \frac{\partial y_{1}}{\partial x_{1}}	& \frac{\partial y_{1}}{\partial x_{2}}	& \dots	 & \frac{\partial y_{1}}{\partial x_{64}}     \\
# \frac{\partial y_{2}}{\partial x_{1}}	& \frac{\partial y_{2}}{\partial x_{2}} 	& \dots  & \frac{\partial y_{2}}{\partial x_{64}} 	  \\
# \vdots	& \vdots 	& \ddots & \vdots \\
# \frac{\partial y_{10}}{\partial x_{1}} 	& \dots & \dots	 & \frac{\partial y_{10}}{\partial x_{64}}
# \end{bmatrix}
# $$ 

# Now, in each row we have one vector of $\frac {\partial y_{k}}{\partial x}$. But that's not all! Have a closer look at the equations of $ \nabla_{x} \log L $ and $ \nabla_{x} \log D$. What are $t_{n,k}$ and $f_{n,k}$ doing here? ...
# 
# They are working like a mask. :-)
# 
# ...
# 
# Not every class contributes to the gradient! As we use one-hot-encoding, only the class with the $1$ yields a summand which is not 0. Looking at the Jacobian this means that all rows are 0 except for the class that is represented by 1 in the encoding of $t_{n,k}$ or $f_{n,k}$. Let's call this specific class $\hat{k}$, then we have:
# 
# $$ \nabla_{x} \log D =  \vec{w}_{\hat{k}} - \sum_{c=1}^{K} y_{c} \vec{w}_{c} $$
# 
# $$ \nabla_{x} \log L =  \vec{w}_{\hat{k}} - \sum_{c=1}^{K} y_{c} \vec{w}_{c} $$
# 
# But take in mind: $\hat{k}$ is different for the both equations: one is the fooling class and the other the true label class and of course they are not the same! :-)
# 
# ### Why could this be important?
# 
# 

# This looks like playing with equations but we gained an insight: We have found that we could express the perturbation by a masking of the Jacobian matrix. The letter tells us more generally how the outputs of all classes change with input perturbations. This is already a hint that we could built attacks and defenses on approaches based on the Jacobian. Besides the Fast Gradient Attack you can find a Jacobian-based approach as well. Have fun to explore! :-)

# # Conclusion
# 
# My public journey has come to end. I have still ideas to play with but this kernel is already too long for an introduction. Feel free to fork, try different epsilons, only use the sign method, use dimensionality reductions as influence property.... there is so much to do :-) . For me it was the first time to fool a machine learning model and I learnt:
# 
# 1. One could easily fool a model if one has access to the learned weights and by defining own objectives that maximize the discrepance between targets and model outputs. (One could also fool without knowing weights or model architecture, but that's another topic...)
# 2. Fooling takes place in regions where the model fails to draw good decision boundaries which of course depends on the model architecture/flexibility but on the input data quality and preprocessing as well.
# 3. Due to the fact that some inputs are closer to each other in meanings of decision boundaries, there exist natural and non-natural fooling targets.
# 
# 
# Have fun and good luck, so far :-)
