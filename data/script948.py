
# coding: utf-8

# Hey guys!
# 
# In the subsequent kernel I am using the *PCA algorithm* in order to gain further insights into the various atom-configurations contained in the xyz-files. I hope that this might be a help for you in order to explore new features which advance you in the underlying problem statement. I like to add that I am not a domain expert in the chemistry field.

# In[ ]:


# General libraries
import pandas as pd
import numpy as np

# Plotting and Visualization Library
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Math Libraries
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull


# In[ ]:


# Load the main train and test data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train_id = train["id"]
test_id = test["id"]
label = train[["formation_energy_ev_natom", "bandgap_energy_ev"]]

train = train.drop(["id", "formation_energy_ev_natom", "bandgap_energy_ev"], axis = 1)
test = test.drop("id", axis = 1)


# In[ ]:


# Read and split of xyz-file
# Adapted from Tony Y: https://www.kaggle.com/tonyyy

def get_xyz_data(filename, ids):
    
    A = pd.DataFrame(columns=list('ABCDE'))
    B = pd.DataFrame(columns=list('ABCE'))
    
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':

                newrowA = pd.DataFrame([[x[1],x[2],x[3],x[4],ids]], columns=list('ABCDE'))
                A = A.append(newrowA)
                
            elif x[0] == 'lattice_vector':
                
                newrowB = pd.DataFrame([[x[1],x[2],x[3],ids]], columns=list('ABCE'))
                B = B.append(newrowB)

    return A, B


# In[ ]:


# plot_pca performs pca on the atom configuration and plots it
# Moreover, the convex hull of the projection is computed and also visualized in the plot

def plot_pca(index):
    
    fn = "../input/train/{}/geometry.xyz".format(index)
    train_xyz, train_lat = get_xyz_data(fn, index)
    color_dict = { 'Ga':'black', 'Al':'blue', 'O':'red', 'In':'green' }
    
    matrix = train_xyz
    colour = matrix["D"]
    matrix = matrix[["A","B","C"]].as_matrix()
    matrix = matrix.astype(float)
    
    pca = PCA(n_components=3)
    X_r = pca.fit(matrix).transform(matrix)
    df_ = pd.DataFrame(np.round(X_r,2))
        
    x = np.array(matrix[:,0])
    y = np.array(matrix[:,1])
    z = np.array(matrix[:,2])

    fig = plt.figure(figsize=(20,20))

    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(x,y,z, c=[ color_dict[i] for i in colour ], marker='o', s=70)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(label.loc[index])
    
    ax = fig.add_subplot(222)
    plt.scatter(X_r[:, 0], X_r[:, 1], color=[color_dict[i] for i in colour], alpha=.8, lw=1, s=70)
    hull = ConvexHull(X_r[:,[0,1]])
    volume_1 = hull.volume
    plt.plot(X_r[hull.vertices,0], X_r[hull.vertices,1], 'r--', lw=1)
    plt.title(label.loc[index])
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title(label.loc[index])
    
    ax = fig.add_subplot(223)
    plt.scatter(X_r[:, 0], X_r[:, 2], color=[color_dict[i] for i in colour], alpha=.8, lw=1, s=70)
    hull = ConvexHull(X_r[:,[0,2]])
    volume_2 = hull.volume
    plt.plot(X_r[hull.vertices,0], X_r[hull.vertices,2], 'r--', lw=1)
    plt.title(label.loc[index])
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Third Principal Component')
    ax.set_title(label.loc[index])
    
    ax = fig.add_subplot(224)
    plt.scatter(X_r[:, 1], X_r[:, 2], color=[color_dict[i] for i in colour], alpha=.8, lw=1, s=70)
    hull = ConvexHull(X_r[:,[1,2]])
    volume_3 = hull.volume
    plt.plot(X_r[hull.vertices,1], X_r[hull.vertices,2], 'r--', lw=1)
    plt.title(label.loc[index])
    ax.set_xlabel('Second Principal Component')
    ax.set_ylabel('Third Principal Component')
    ax.set_title(label.loc[index])
    
    plt.show()
    
    print("On the first principle component are approx. " + str(len(df_[0].unique())) + " distinct coordinates with atoms")
    print("On the second principle component are approx. " + str(len(df_[1].unique())) + " distinct coordinates with atoms")
    print("On the third principle component are approx. " + str(len(df_[2].unique())) + " distinct coordinates with atoms")
    print("")
    print("Area covered by the first and second principal component: " + str(volume_1))
    print("Area covered by the first and third principal component: " + str(volume_2))
    print("Area covered by the second and third principal component: " + str(volume_3))


# In[ ]:


ind = 200
plot_pca(index=ind)


# **Interesting**
# 
# The projection on the first and second principal component reveals a (approx. symmetric) circular structure. I recognized that this isn't the only atom-configuration in which that structure occurs.

# In[ ]:


ind = 500
plot_pca(index=ind)


# **Interesting**
# 
# On the second principal component axis it suffies to have four coordinates order to explain the atom distribution on that axis.

# In[ ]:


ind = 2350
plot_pca(index=ind)


# **Interesting**
# 
# The projection on the second and third principal component reveals this very concentrated structure distributed to a few clusters. A pattern which also occured in several train/test samples.

# So these are just some insights and measurements based on PCA. Further measurements could be
# 
# 1. Variance in the projection plane
# 2. Measurement of symmetry in the projection plane
# 3. Amount of Clusters in the projection plane
# 
# Personally I discovered some interesting patterns and correlations to the target. I hope this also helps you to gain further insights.
# 
# Best, Max
