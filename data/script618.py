
# coding: utf-8

# One way to approach the competition is to look for a solution structure that has a good chance to yield good submission.  A solution structure is defined by a number of bag types, plus a number of occurrence of each bag type.  A bag type is defined by the number of gifts of each type it contains. For instance 3 blocks and 1 train.
# 
# We can focus on bag types because all bags have the same capacity (50 pounds).
# 
# There is a finite number of bag types that are possible.  We define one random variables for each bag type. 
# 
# All we need is an estimate the expected value and the variance of each possible bag type.  Then we use two properties to find a combination of bags that maximizes a combination of expected value and standard deviation:
# 
# - the expected value of a sum of random variables is the sum of the expected values of the random variables
# - the variance of a sum of independent random variables is the sum of the variances of the random variable
# 
# Kernels or scripts with similar approaches have been proposed by [Dominic Breuker](https://www.kaggle.com/breuker/santas-uncertain-bags/can-we-improve-by-increasing-variance) and [Ben Gorman](https://www.kaggle.com/ben519/santas-uncertain-bags/merry-christmas-y-all).
# 
# The difference is that here we find the optimal solution in a probabilistic sense.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# First some definitions.

# In[ ]:


gift_types = ['horse', 'ball', 'bike', 'train', 'coal', 'book', 'doll', 'blocks', 'gloves']
ngift_types = len(gift_types)

horse, ball, bike, train, coal, book, doll, blocks, gloves = range(ngift_types)


# We will use Monte Carlo simulation quite a bit. Let's agree on the number of samples to use.  Set it to a higher value to get more accurate results.

# In[ ]:


nsample=10000


# Let's look at bags composed of a single gift type. We use a vectorized version of the original numpy distributions.
# 

# In[ ]:


def gift_weights(gift, ngift, n=nsample):
    if ngift == 0:
        return np.array([0.0])
    np.random.seed(2016)
    if gift == horse:
        dist = np.maximum(0, np.random.normal(5,2,(n, ngift))).sum(axis=1)
    if gift == ball:
        dist = np.maximum(0, 1 + np.random.normal(1,0.3,(n, ngift))).sum(axis=1)
    if gift == bike:
        dist = np.maximum(0, np.random.normal(20,10,(n, ngift))).sum(axis=1)
    if gift == train:
        dist = np.maximum(0, np.random.normal(10,5,(n, ngift))).sum(axis=1)
    if gift == coal:
        dist = 47 * np.random.beta(0.5,0.5,(n, ngift)).sum(axis=1)
    if gift == book:
        dist = np.random.chisquare(2,(n, ngift)).sum(axis=1)
    if gift == doll:
        dist = np.random.gamma(5,1,(n, ngift)).sum(axis=1)
    if gift == blocks:
        dist = np.random.triangular(5,10,20,(n, ngift)).sum(axis=1)
    if gift == gloves:
        gloves1 = 3.0 + np.random.rand(n, ngift)
        gloves2 = np.random.rand(n, ngift)
        gloves3 = np.random.rand(n, ngift)
        dist = np.where(gloves2 < 0.3, gloves1, gloves3).sum(axis=1)
    return dist


# Let's find a reasonable upper bound on the number of gifts in the bag. For this we compute the expected score for bags with an increasing number of toys until the score decreases. The bag with largest score is determining the maximum value. This is fine when optimizing the expected value, as adding additional toys uses more toys without improving the objective function.

# In[ ]:


epsilon = 1
max_type = np.zeros(ngift_types).astype('int')

for gift, gift_type in enumerate(gift_types):
    best_value = 0.0
    for j in range(1, 200):
        weights = gift_weights(gift, j, nsample)
        raw_value = np.where(weights <= 50.0, weights, 0.0)
        value = raw_value.mean()
        if value > best_value:
            best_value = value
        else:
            break
    max_type[gift] = j
max_type


# We can now look at more general bag types. First we precompute weights of bags with a single type. The code is similar to the above one.
# 
# For each gift type , we create a 2D array with nsample rows, and ntype columns. Column j contains the weights of a bag made of j+1 toys of the given gift type.

# In[ ]:


def gift_distributions(gift, ngift, n=nsample):
    if ngift == 0:
        return np.array([0.0])
    np.random.seed(2016)
    if gift == horse:
        dist = np.maximum(0, np.random.normal(5,2,(n, ngift)))
    if gift == ball:
        dist = np.maximum(0, 1 + np.random.normal(1,0.3,(n, ngift)))
    if gift == bike:
        dist = np.maximum(0, np.random.normal(20,10,(n, ngift)))
    if gift == train:
        dist = np.maximum(0, np.random.normal(10,5,(n, ngift)))
    if gift == coal:
        dist = 47 * np.random.beta(0.5,0.5,(n, ngift))
    if gift == book:
        dist = np.random.chisquare(2,(n, ngift))
    if gift == doll:
        dist = np.random.gamma(5,1,(n, ngift))
    if gift == blocks:
        dist = np.random.triangular(5,10,20,(n, ngift))
    if gift == gloves:
        gloves1 = 3.0 + np.random.rand(n, ngift)
        gloves2 = np.random.rand(n, ngift)
        gloves3 = np.random.rand(n, ngift)
        dist = np.where(gloves2 < 0.3, gloves1, gloves3)
    for j in range(1, ngift):
        dist[:,j] += dist[:,j-1]
    return dist

distributions = dict()
    
for gift in range(ngift_types):
    distributions[gift] = gift_distributions(gift, max_type[gift])


# We can now compute expected value of complex bags with lookups of precomputed weight distributions. With a slight change it code it is easy to compute additional statistics like the variance of the weight.

# In[ ]:


def gift_distributions(gift, ngift):
    if ngift <= 0:
        return 0
    if ngift >= max_type[gift]:
        return 51
    return distributions[gift][:,ngift-1]

def gift_value(ntypes):
    weights = np.zeros(nsample)
    for gift in range(ngift_types):
        dist = gift_distributions(gift, ntypes[gift])
        weights += dist
    weights = np.where(weights <= 50.0, weights, 0.0)
    return weights.mean(), weights.std()


# We can now generate bag types. The idea is to start with an empty bag, and to add one item at a time. We do it until the expected value of the bag decreases. When this happens then we can discard the newly created bag, as it uses more items and yields a lower expected value.
# We use a queue and some dictionaries to keep track of what bag types are relevant.
# 
# Once the relevant bags are found we put all of them in a dataframe.  We remove those with less than three elements.
# 
# This takes a time roughly proportional to nsample.  With 10,000 is takes less than a minute.   Go grab a coffee if you set nsample to a larger value, say 100,000.

# In[ ]:


from collections import deque

def get_update_value(bag, bag_stats):
    if bag in bag_stats:
        bag_mean, bag_std = bag_stats[bag]
    else:
        bag_mean, bag_std = gift_value(bag)
        bag_stats[bag] = (bag_mean, bag_std)
    return bag_mean, bag_std

def gen_bags():
    bag_stats = dict()
    queued = dict()
    queue = deque()
    bags = []
    bag0 = (0,0,0,0,0,0,0,0,0)
    queue.append(bag0)
    queued[bag0] = True
    bag_stats[bag0] = (0,0)
    counter = 0
    try:
        while True:
            if counter % 1000 == 0:
                print(counter, end=' ')
            counter += 1
            bag = queue.popleft()
            bag_mean, bag_std = get_update_value(bag, bag_stats)
            bags.append(bag+(bag_mean, bag_std ))
            for gift in range(ngift_types):
                new_bag = list(bag)
                new_bag[gift] = 1 + bag[gift]
                new_bag = tuple(new_bag)
                if new_bag in queued:
                    continue
                new_bag_mean, new_bag_std = get_update_value(new_bag, bag_stats)
                if new_bag_mean > bag_mean:
                    queue.append(new_bag)
                    queued[new_bag] = True
                    
    except:
        return bags

    
bags = gen_bags()

nbags = len(bags)

bags = pd.DataFrame(columns=gift_types+['mean', 'std'], 
                    data=bags)

bags['var'] = bags['std']**2

bags = bags[bags[gift_types].sum(axis=1) >= 3].reset_index(drop=True)

bags.head()


# We have about 40k bags.

# In[ ]:


bags.shape[0]


# Let's now look at the available gifts.  We one hot encode the gift type.

# In[ ]:


gifts = pd.read_csv('../input/gifts.csv')

for gift in gift_types:
    gifts[gift] = 1.0 * gifts['GiftId'].str.startswith(gift)

gifts.head()


# The number of gift of each type is easy to get.

# In[ ]:


allgifts = gifts[gift_types].sum()

allgifts


# We can now look for a combination of bag types that optimizes the expected value, or a combination of the expected value and the standard deviation.
# 
# The mathematical formulation is as follows.
# 
# $$
# \begin{align}
# & \text{maximize} && mean + \alpha \cdot std& \\
# & \text{s.t.} 
# && \sum_{i=1}^n g_{ij} \cdot x_i \leq capa_j && \forall j = 1,\ldots,m \\
# &&& \sum_{i=1}^nx_i \leq 1000  \\
# &&& \sum_{i=1}^n mean_i \cdot x_i = mean & \\
# &&& \sum_{i=1}^n var_i \cdot x_i = var & \\
# &&& std^2 = var & \\
# &&& x_{i} \geq 0&& \forall i = 1,\ldots,n\\
# \end{align}
# $$ 
# where:
# - $n$ is the number of bag types
# - $m$ the number of gift types
# - $\alpha$ is the relative importance of std vs mean in the objective function
# - $w_i$ the expected value of the weight of bag type $i$
# - $var_i$ the variance of the weight of bag type $i$
# - $g_{ij}$ the number of gifts of type $j$ in bag type $i$
# - $capa_j$ the number of available gifts of type $j$
# - $x_i$ is an integer decision variable that takes the value $a$ if bag type $i$ is used $a$ times
# - $std$ a decision variable representing the standard deviation of the solution
# - $var$ a decision variable representing the variance of the solution
# 
# Constraints (1) ensure that the solution does not use more gifts than available. Constraints (2) states that there are at most 1,000 bags in the solution.  Constraint (3) computes the mean the solution, constraint (4) computes the variance of the solution, and constraint (5) compute its standard deviation.
# 
# The trick 
# here is that the last constraint is a quadratic constraint.  We cannot use an open source LP solver because of
# it, which is why I use CPLEX.  CPLEX is not available on Kaggle kernel, but one can use the [feely available DoCplexCloud trial](https://developer.ibm.com/docloud/try-docloud-free/) run the following code.

# In[ ]:


from docplex.mp.model import Model

def qcpmip_solve(gift_types, bags, std_coef):
    mdl = Model('Santa')

    rbags = range(bags.shape[0])
    x_names = ['x_%d' % i for i in range(bags.shape[0])]
    x = mdl.integer_var_list(rbags, lb=0, name=x_names)
    
    var = mdl.continuous_var(lb=0, ub=mdl.infinity, name='var')
    std = mdl.continuous_var(lb=0, ub=mdl.infinity, name='std')
    mean = mdl.continuous_var(lb=0, ub=mdl.infinity, name='mean')
                                  
    mdl.maximize(mean + std_coef * std)
    
    for gift in gift_types:
        mdl.add_constraint(mdl.sum(bags[gift][i] * x[i] for i in rbags) <= allgifts[gift])
        
    mdl.add_constraint(mdl.sum(x[i] for i in rbags) <= 1000)

    mdl.add_constraint(mdl.sum(bags['mean'][i] * x[i] for i in rbags) == mean)
    
    mdl.add_constraint(mdl.sum(bags['var'][i] * x[i] for i in rbags) == var)

    mdl.add_constraint(std**2 <= var)
    
    mdl.parameters.mip.tolerances.mipgap = 0.00001
    
    s = mdl.solve(log_output=False)
    assert s is not None
    mdl.print_solution()
    


# We can now solve the problem if we have CPLEX installed. For instance, if we are only looking for the best expected value:

# In[ ]:


qcpmip_solve(gift_types, bags, 0.0)


# Let me paste output here given it does not run on Kaggle kernel:
# 
#     objective: 35618.561
#       x_14954=1
#       x_368=63
#       x_773=395
#       x_3936=93
#       x_39480=47
#       x_36998=1
#       x_474=24
#       x_11628=55
#       mean=35618.561
#       x_2089=76
#       x_264=46
#       x_2091=199

# We see that the best solution structure has an expected value around 35620.  I noticed that as the sample size is increased (the value of nsample), then the objective value decreases.

# For instance, when using `nsample=100000`, we get
#     
#    
# 
#      objective: 35545.217
#       x_2315=6
#       x_36888=1
#       x_2301=61
#       x_39358=47
#       x_8792=115
#       x_5991=2
#       x_420=2
#       x_822=1
#       x_315=286
#       x_2137=303
#       mean=35545.217
#       var=106992.386
#       x_418=87
#       x_3965=89

# We see that the expected value is now about 35545. When I run the code with even larger sample, for instance with nsample=1000000 I get below 35540.

# If we want to maximize the expected value plus 3 times the standard deviation:
# 

# In[ ]:


qcpmip_solve(gift_types, bags, 3.0)


# Output with `nsample=100000` is:
# 
#     objective: 36533.143
#       var=113259.323
#       mean=35523.522
#       x_3965=100
#       x_36888=1
#       x_523=1
#       x_2137=299
#       x_315=199
#       x_6121=49
#       x_39358=47
#       x_2301=200
#       x_8768=1
#       x_1035=95
#       std=336.540
#       x_418=8
# 

# This time we get a lower mean, but a larger std. 
# 
# Solution structure found this way are the best if given enough submissions, as they maximize the likelihood of a good submission.  But the competition shows that switching to a local search approach using feedback from LB is more effective.
