
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

amt = 10000

class Horse:
    def __init__(self,id):
        self.weight = max(0, np.random.normal(5,2,1)[0])
        self.name = 'horse_' + str(id)

class Ball:
    def __init__(self,id):
        self.weight = max(0, 1 + np.random.normal(1,0.3,1)[0])
        self.name = 'ball_' + str(id)

class Bike:
    def __init__(self,id):
        self.weight = max(0, np.random.normal(20,10,1)[0])
        self.name = 'bike_' + str(id)

class Train:
    def __init__(self,id):
        self.weight = max(0, np.random.normal(10,5,1)[0])
        self.name = 'train_' + str(id)
        
class Coal:
    def __init__(self,id):
        self.weight = 47 * np.random.beta(0.5,0.5,1)[0]
        self.name = 'coal_' + str(id)
        
class Book:
    def __init__(self,id):
        self.weight = np.random.chisquare(2,1)[0]
        self.name = "book_" + str(id)
        
class Doll:
    def __init__(self,id):
        self.weight = np.random.gamma(5,1,1)[0]
        self.name = "doll_" + str(id)

class Block:
    def __init__(self,id):
        self.weight = np.random.triangular(5,10,20,1)[0]
        self.name = "blocks_" + str(id)
        
class Gloves:
    def __init__(self,id):
        self.weight = 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
        self.name = "gloves_" + str(id)

books = [Book(x) for x in range(amt)]
horses = [Horse(x) for x in range(amt)]
bikes = [Bike(x) for x in range(amt)]
trains = [Train(x) for x in range(amt)]
coals = [Coal(x) for x in range(amt)]
dolls = [Doll(x) for x in range(amt)]
balls = [Ball(x) for x in range(amt)]
blocks = [Block(x) for x in range(amt)]
gloves = [Gloves(x) for x in range(amt)]

def plot_gift(g,i):
    wvec = [x.weight for x in eval(g)]
    plt.figure(i)
    plt.suptitle(g + " = " + str(sum(wvec)))
    plt.hist(wvec)

plot_gift('horses', 0)
plot_gift('balls', 1)
plot_gift('bikes', 2)
plot_gift('trains', 3)
plot_gift('coals', 4)
plot_gift('books', 5)
plot_gift('dolls', 6)
plot_gift('blocks', 7)
plot_gift('gloves', 8)

