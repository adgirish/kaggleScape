
# coding: utf-8

# This notebook shows a way to obtain atomic coordinates from xyz files. We can calculate bond lengths, bond angles, coordination numbers, and so on from atomic coordinates. Some of them might be able to be used as good features. 
# 
# ![monoclinic_and_hexagonal](https://github.com/Tony-Y/MaterialsAreSimilarToDocuments/blob/master/monoclinic_and_hexagonal.png?raw=true)
# 
# 1. Reading geometry files
# 2. Reduced coordinates
# 3. Geometrical properties

# # Reading geometry files
# 
# [Jmol](http://jmol.sourceforge.net) is an open source molecular viewer. Jmol can load xyz geometry files directly.
# 
# ![train 1 geometry](https://github.com/Tony-Y/MaterialsAreSimilarToDocuments/blob/master/geometry.jpg?raw=true)
# 
# This figure is a visualization example of **train/1/gemometry.xyz**.
# 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df_train = pd.read_csv("../input/train.csv")
df_train["dataset"] = "train"
df_test = pd.read_csv("../input/test.csv")
df_test["dataset"] = "test"
df_crystals = pd.concat([df_train, df_test], ignore_index=True)


# In[3]:


df_crystals.head()


# In[4]:


df_crystals.tail()


# In[5]:


# Orthorhombic (kappa-Al2O3 type)
row_id = 0

# Monoclinic (beta-Ga2O3 type)
#row_id = 19

# Hexagonal (hcp type)
#row_id = 660

# The first row in the test dataset
#row_id = len(df_train)

row_id


# In[6]:


lattice_columns = ["lattice_vector_1_ang", "lattice_vector_2_ang", "lattice_vector_3_ang", "lattice_angle_alpha_degree", "lattice_angle_beta_degree", "lattice_angle_gamma_degree"]
df_crystals.loc[row_id, lattice_columns]


# You can see the lattice constants $a$, $b$, $c$, $\alpha$, $\beta$, and $\gamma$ in the above visualization equal to ones in the CSV file. 

# In[7]:


def get_xyz_data(filename):
    pos_data = []
    lat_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)


# In[8]:


idx = df_crystals.id.values[row_id]
dataset = df_crystals.dataset.values[row_id]
fn = "../input/{}/{}/geometry.xyz".format(dataset, idx)
crystal_xyz, crystal_lat = get_xyz_data(fn)


# In[9]:


crystal_xyz


# This list conatins pairs of the position vector $(X, Y, Z)$ and the element symbol.

# In[10]:


crystal_lat


# This matirx is $(\textbf{a}_1, \textbf{a}_2, \textbf{a}_3)^t$ where $\textbf{a}_i$ is $i$-th lattice vector.

# In[11]:


def length(v):
    return np.linalg.norm(v)

def unit_vector(vector):
    return vector / length(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_deg_between(v1, v2):
    return np.degrees(angle_between(v1, v2))

def get_lattice_constants(lattice_vectors):
    lat_const_series = pd.Series()
    for i in range(3):
        lat_const_series["lattice_vector_"+str(i+1)+"_ang"] = length(lattice_vectors[i])
    lat_const_series["lattice_angle_alpha_degree"] = angle_deg_between(lattice_vectors[1],lattice_vectors[2])
    lat_const_series["lattice_angle_beta_degree"] = angle_deg_between(lattice_vectors[2],lattice_vectors[0])
    lat_const_series["lattice_angle_gamma_degree"] = angle_deg_between(lattice_vectors[0],lattice_vectors[1])
    return lat_const_series


# In[12]:


get_lattice_constants(crystal_lat)


# You can see the lattice constants again.

# # Reduced coordinates

# When the reduced coordinate is donated as $\textbf{r} = (x, y, z)^t$, the position vector $\textbf{R}$ is expressed using the lattice vectors $\textbf{a}_i$ as following:
# 
# $\textbf{R} = (X, Y, Z)^t = x \textbf{a}_1 + y \textbf{a}_2 + z \textbf{a}_3 = A \textbf{r}$
# 
# where $A =(\textbf{a}_1, \textbf{a}_2, \textbf{a}_3)$. The reduced coordinate is obtained from:
# 
# $\textbf{r} = A^{-1} \textbf{R}$.
# 
# $B = A^{-1} =(\textbf{b}_1, \textbf{b}_2, \textbf{b}_3)^t$ where $\textbf{b}_i$ is the $i$-th reciprocal lattice vector.

# In[13]:


A = np.transpose(crystal_lat)
R = crystal_xyz[0][0]
print("The lattice vectors:")
print(A)
print("The position vector:")
print(R)


# In[14]:


from numpy.linalg import inv
B = inv(A)
print("The reciprocal lattice vectors:")
print(B)


# In[15]:


r = np.matmul(B, R)
print("The reduced coordinate vector:")
print(r)


# # Geometrical properties

# Crystalline materials are made of periodic cells. Therefore, geometrical properties must be calculated with consideration for the periodicity. The vector between two atoms $i$ and $j$ is expressed as:
# 
# $\textbf{r}_{ij} =  \textbf{r}_{i} - \textbf{r}_{j} + (l, m, n)^t$
# 
# where $l$, $m$, and $n$ are any integer value. The real-space vector of the atom pair is
# 
# $\textbf{R}_{ij} = A \textbf{r}_{ij}$.
# 
# ## Spatial distance
# 
# $d_{ij} = \min_{l,m,n} \mid \textbf{R}_{ij} \mid$

# In[16]:


def get_shortest_distances(reduced_coords, amat):
    natom = len(reduced_coords)
    dists = np.zeros((natom, natom))
    Rij_min = np.zeros((natom, natom, 3))

    for i in range(natom):
        for j in range(i):
            rij = reduced_coords[i][0] - reduced_coords[j][0]
            d_min = np.inf
            R_min = np.zeros(3)
            for l in range(-1, 2):
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        r = rij + np.array([l, m, n])
                        R = np.matmul(amat, r)
                        d = length(R)
                        if d < d_min:
                            d_min = d
                            R_min = R
            dists[i, j] = d_min
            dists[j, i] = dists[i, j]
            Rij_min[i, j] = R_min
            Rij_min[j, i] = -Rij_min[i, j]
    return dists, Rij_min


# In[17]:


crystal_red = [[np.matmul(B, R), symbol] for (R, symbol) in crystal_xyz]
crystal_dist, crystal_Rij = get_shortest_distances(crystal_red, A)
crystal_dist


# In[18]:


import seaborn as sns
sns.heatmap(crystal_dist)


# In[19]:


def get_min_length(distances, A_atoms, B_atoms):
    A_B_length = np.inf
    for i in A_atoms:
        for j in B_atoms:
            d = distances[i, j]
            if d > 1e-8 and d < A_B_length:
                A_B_length = d
    
    return A_B_length


# In[20]:


natom = len(crystal_red)
al_atoms = [i for i in range(natom) if crystal_red[i][1] == 'Al']
ga_atoms = [i for i in range(natom) if crystal_red[i][1] == 'Ga']
in_atoms = [i for i in range(natom) if crystal_red[i][1] == 'In']
o_atoms = [i for i in range(natom) if crystal_red[i][1] == 'O']


# In[21]:


if len(al_atoms):
    print("Al-O min length:", get_min_length(crystal_dist, al_atoms, o_atoms))
if len(ga_atoms):
    print("Ga-O min length:", get_min_length(crystal_dist, ga_atoms, o_atoms))
if len(in_atoms):
    print("In-O min length:", get_min_length(crystal_dist, in_atoms, o_atoms))


# ~~The minimun lengths between metal and oxygen atoms can be used for estimating the threshold of connections in the crystal graph.  However, the current version of this notebook does not use the suggested method yet.~~ (I introduced the function get_factor which appears below.)

# ## Histogram of distances

# In[22]:


hist_dist = plt.hist(crystal_dist.flatten(), bins=100)
_ = plt.title("Histogram of the shortest distances")


# In[23]:


def get_distances(r, amat, l_max=3, m_max=3, n_max=3, R_max=20.0):
    distances = []
    for l in range(-l_max, l_max+1):
        for m in range(-m_max, m_max+1):
            for n in range(-n_max, n_max+1):
                R = np.matmul(amat, r + np.array([l, m, n]))
                d = length(R)
                if d < R_max:
                    distances.append(d)
                    
    return distances


# Note that optimal values of `l_max`, `m_max`, and `n_max` depend on both the maximum radius `R_max` and the lattice matrix `amat`. A larger `R_max` or a smaller cell requires larger `l_max`, `m_max`, and `n_max`. You can simply estimate them from $l_{\rm max} > R_{\rm max} / a$ where $a$ is the lattice constant. In general, $l_{\rm max} >  R_{\rm max} \cdot \mid{\bf b}_1\mid$, $m_{\rm max} >  R_{\rm max} \cdot \mid{\bf b}_2\mid$, and $n_{\rm max} >  R_{\rm max} \cdot \mid{\bf b}_3\mid$ where ${\bf b }_i$ is the $i$-th reciprocal lattice vector.

# In[24]:


def get_optimal_lmn(bmat, R_max=20.0):
    lmn = dict()
    lmn["l_max"] = int(length(bmat[0]) * R_max) + 1
    lmn["m_max"] = int(length(bmat[1]) * R_max) + 1
    lmn["n_max"] = int(length(bmat[2]) * R_max) + 1
    lmn["R_max"] = R_max

    return lmn


# In[25]:


opt_lmn = get_optimal_lmn(B)

print(opt_lmn)


# In[26]:


natom = len(crystal_red)
m_atoms = [i for i in range(natom) if crystal_red[i][1] != 'O']
o_atoms = [i for i in range(natom) if crystal_red[i][1] == 'O']

m_o_distances = []
for i in m_atoms:
    for j in o_atoms:
        rij = np.matmul(B, crystal_Rij[i, j])
        m_o_distances += get_distances(rij, A, **opt_lmn)


# In[27]:


m_m_distances = []
for i in m_atoms:
    for j in m_atoms:
        rij = np.matmul(B, crystal_Rij[i, j])
        m_m_distances += get_distances(rij, A, **opt_lmn)


# In[28]:


o_o_distances = []
for i in o_atoms:
    for j in o_atoms:
        rij = np.matmul(B, crystal_Rij[i, j])
        o_o_distances += get_distances(rij, A, **opt_lmn)


# In[29]:


plt.figure(figsize=(6,12))

ax1 = plt.subplot(311)
hist_m_o_dist = plt.hist(m_o_distances, bins=100, range=(0, 20))
plt.text(0.1, 0.9, "Metal-Oxygen", fontsize=12, transform=ax1.transAxes)

plt.title("Histogram of distances")

ax2 = plt.subplot(312)
hist_m_m_dist = plt.hist(m_m_distances, bins=100, range=(0, 20))
plt.text(0.1, 0.9, "Metal-Metal", fontsize=12, transform=ax2.transAxes)

ax3 = plt.subplot(313)
hist_o_o_dist = plt.hist(o_o_distances, bins=100, range=(0, 20))
plt.xlabel("Radius (Å)", fontsize=12)
_ = plt.text(0.1, 0.9, "Oxygen-Oxygen", fontsize=12, transform=ax3.transAxes)


# ## Radiral distribution function
# 
# You can find the definition at Wikipedia: [Radial distribution function](https://en.wikipedia.org/wiki/Radial_distribution_function) and [Pair distribution function](https://en.wikipedia.org/wiki/Pair_distribution_function).

# In[30]:


def get_rdf(hist_x, hist_r, density, natom):
    dr = hist_r[1] - hist_r[0]
    factor = 1.0 / ( 4 * np.pi * dr * density * natom)
    rad = []
    rdf = []
    for i in range(len(hist_x)):
        r = (hist_r[i] + hist_r[i+1])/2
        rad.append(r)
        v = factor * hist_x[i] / r**2
        rdf.append(v)
    
    return rad, rdf


# In[31]:


vol = np.linalg.det(A)
print("Volume:", vol)
m_count = len(m_atoms)
m_density = m_count/vol
print("Metal count and density:", m_count, m_density)
o_count = len(o_atoms)
o_density = o_count/vol
print("Oxygen count and density:", o_count, o_density)

m_o_hist_x, m_o_hist_r, _ = hist_m_o_dist
m_o_rad, m_o_rdf = get_rdf(m_o_hist_x, m_o_hist_r, o_density, m_count)

m_m_hist_x, m_m_hist_r, _ = hist_m_m_dist
m_m_hist_x[0] = 0
m_m_rad, m_m_rdf = get_rdf(m_m_hist_x, m_m_hist_r, m_density, m_count)

o_o_hist_x, o_o_hist_r, _ = hist_o_o_dist
o_o_hist_x[0] = 0
o_o_rad, o_o_rdf = get_rdf(o_o_hist_x, o_o_hist_r, o_density, o_count)


# In[32]:


plt.hlines(1, 0, 20)
plt.plot(m_o_rad, m_o_rdf, label="M-O")
plt.plot(m_m_rad, m_m_rdf, label="M-M")
plt.plot(o_o_rad, o_o_rdf, label="O-O")
plt.xlim(0, 20)
plt.ylim(0, 6)
plt.legend()
plt.xlabel("Radius (Å)", fontsize=12)
_ = plt.title("Radial Distribution Functions")


# You can see RDFs converge on 1 upon the increase of the radius.

# ## Crystal graph
# 
# If you are in a hurry, please read the paper about the crytal graph: https://arxiv.org/abs/1710.10324
# 
# When at least one of the cell edges is smaller than double the minumun distance between metal and oxygen atoms, the crytal graph becomes unpreferable. In such cases, this notebook uses [the supercell method][a].
# 
# [a]: https://en.wikipedia.org/wiki/Supercell_(crystal)

# In[33]:


def get_supercell(reduced_coords, amat, l_max, m_max, n_max):
    sc_indeces = np.array([l_max, m_max, n_max])
    sc_amat = amat * sc_indeces
    sc_red = []
    for l in range(l_max):
        for m in range(m_max):
            for n in range(n_max):
                for rc in reduced_coords:
                    x = rc[0] + np.array([l, m, n])
                    x /= sc_indeces
                    sc_red.append([x, rc[1]])

    return sc_red, sc_amat


# In[34]:


sc_lmn = get_optimal_lmn(B, R_max=3.6)
print("Supercell:", sc_lmn)

del sc_lmn["R_max"]
sc_red, sc_A = get_supercell(crystal_red, A, **sc_lmn)

sc_dist, sc_Rij = get_shortest_distances(sc_red, sc_A)

natom = len(sc_red)
m_atoms = [i for i in range(natom) if sc_red[i][1] != 'O']
o_atoms = [i for i in range(natom) if sc_red[i][1] == 'O']


# In[35]:


import networkx as nx
#
# Database of Ionic Radii
# http://abulafia.mt.ic.ac.uk/shannon/ptable.php
#
# Coordination IV
R_O = 1.35
#
# Coordination VI
R_Al = 0.535
R_Ga = 0.62
R_In = 0.8
#
R_ionic = { "O" : R_O, "Al" : R_Al, "Ga" : R_Ga, "In" : R_In }

def get_crytal_graph(reduced_coords, dists, factor=1.5):
    natom = len(reduced_coords)
    G = nx.Graph()
    for i in range(natom):
        symbol_i = reduced_coords[i][1]
        for j in range(i):
            symbol_j = reduced_coords[j][1]
            if (symbol_i == "O" and symbol_j != "O") or (symbol_i != "O" and symbol_j == "O"):
                node_i = symbol_i + "_" + str(i)
                node_j = symbol_j + "_" + str(j)
                R_max = (R_ionic[symbol_i] + R_ionic[symbol_j]) * factor
                if dists[i, j] < R_max:
                    G.add_edge(node_i, node_j)
    
    return G


# If you find lack or excess of connections, you shoud adjust `factor`. Maybe, close-packed lattices require a lower value than the defalut value. For a hcp-like lattice, I used `factor=1.2`.
# 
# I recommend the factor that depends on the spacegroup and the gamma:

# In[36]:


def get_factor(spacegroup, gamma):
    if spacegroup == 12:
        return 1.4
    elif spacegroup == 33:
        return 1.4
    elif spacegroup == 167:
        return 1.5
    elif spacegroup == 194:
        return 1.3
    elif spacegroup == 206:
        return 1.5
    elif spacegroup == 227:
        if gamma < 60:
            return 1.4
        else:
            return 1.5
    else:
        raise NameError('get_factor does not support the spacegroup: {}'.format(spacegroup))


# In[39]:


spacegroup = df_crystals.spacegroup.values[row_id]
angle_gamma = df_crystals.lattice_angle_gamma_degree.values[row_id]
cg_factor = get_factor(spacegroup, angle_gamma)
G = get_crytal_graph(sc_red, sc_dist, factor=cg_factor)

print("Node count:", G.number_of_nodes())
print("Edge count:", G.number_of_edges())

for i in range(natom):
    symbol_i = sc_red[i][1]
    node_i = symbol_i + "_" + str(i)
    crdn_i = list(G.neighbors(node_i))
    print(node_i, len(crdn_i), crdn_i)


# There are 4, 6-coordinated Al and Ga atoms, and 3, 4, 5-coordinated O atoms.

# In[40]:


plt.figure(figsize=(10,10)) 
nx.draw_spring(G, with_labels=True, node_size=800, font_size=8)


# All edges of the crystal graph are imported as bonds in the below gemometry figure.
# ![Crystal graph to Jmol viewer](https://github.com/Tony-Y/MaterialsAreSimilarToDocuments/blob/master/geometry_add2.jpg?raw=true)
# 
# You can generate the bonds for Jmol using the following script:  

# In[41]:


def generate_jmol_bonds(G, xyz_coords, reduced_coords, amat,
                        jmol_bonds_fn="./jmol_bonds.spt",
                        jmol_xyz_fn = "./jmol_geometry.xyz"):
    add_atoms = []
    na = len(reduced_coords)
    with open(jmol_bonds_fn, "w") as f:
        f.write("set autobond off\n")
        f.write("load \"{}\"\n".format(jmol_xyz_fn))
        f.write("background [x0000cc]\n")
        f.write("set bondradiusmilliangstroms 100\n")
        f.write("set perspectiveDepth true\n")
        for e in G.edges():
            e1 = e[0].split("_")
            e2 = e[1].split("_")
            i = int(e1[1])
            j = int(e2[1])
            red = reduced_coords[i][0] - reduced_coords[j][0]
            if np.sum(np.abs(red) > 0.5) > 0:
                sign_vec = [0] * 3
                for k in range(3):
                    if red[k] > 0.5:
                        sign_vec[k] = 1
                    elif red[k] < -0.5:
                        sign_vec[k] = -1
                lat_vec = np.matmul(amat, sign_vec)
                xyz_i = xyz_coords[i][0] - lat_vec
                ii = na
                na += 1
                add_atoms.append([xyz_i, e1[0]])
                xyz_j = xyz_coords[j][0] + lat_vec
                jj = na
                na += 1
                add_atoms.append([xyz_j, e2[0]])
                atom1 = e1[0]+str(i+1)
                atom2 = e2[0]+str(jj+1)
                f.write("connect ({}) ({}) single\n".format(atom1, atom2))
                atom1 = e1[0]+str(ii+1)
                atom2 = e2[0]+str(j+1)
                f.write("connect ({}) ({}) single\n".format(atom1, atom2))
            else:
                atom1 = e1[0]+str(i+1)
                atom2 = e2[0]+str(j+1)
                f.write("connect ({}) ({}) single\n".format(atom1, atom2))

    with open(jmol_xyz_fn, "w") as f:
        f.write("#=================================\n")
        f.write("#Created using the Crystal Graph\n")
        f.write("#=================================\n")
        for i in range(3):
            f.write("lattice_vector {} {} {}\n".format(amat[0,i], amat[1,i], amat[2,i]))
        for atom in xyz_coords:
            f.write("atom {} {} {} {}\n".format(atom[0][0], atom[0][1], atom[0][2], atom[1]))
        for atom in add_atoms:
            f.write("atom {} {} {} {}\n".format(atom[0][0], atom[0][1], atom[0][2], atom[1]))


# In[42]:


sc_xyz = [ [np.matmul(sc_A, atom[0]), atom[1]] for atom in sc_red]
generate_jmol_bonds(G, sc_xyz, sc_red, sc_A)


# In[43]:


# Please see `Output` of this notebook or uncomment the following two lines.
#with open("jmol_bonds.spt") as f:
#    print(f.read())


# In[44]:


# Please see `Output` of this notebook or uncomment the following two lines.
#with open("jmol_geometry.xyz") as f:
#    print(f.read())


# Put the two files, **jmol_bonds.spt** and **jmol_geometry.xyz** on the same directory, and run Jmol as:
# 
# `java -Xmx512m -jar /your/installed/directory/Jmol.jar jmol_bonds.spt`
# 
# Note that `/your/installed/directory` is different for each person.

# ## Graph distance

# In[45]:


path_lengths = np.ones((natom, natom), dtype=np.int)
for i in range(natom):
    atom_i = sc_red[i][1] + "_" + str(i)
    for j in range(i):
        atom_j = sc_red[j][1] + "_" + str(j)
        path_lengths[i, j] = nx.shortest_path_length(G, atom_i, atom_j)
        path_lengths[j, i] = path_lengths[i, j]
path_lengths


# In[46]:


import seaborn as sns
sns.heatmap(path_lengths)


# ## Histogram of angles

# In[49]:


def get_angles(G, Rij, atom1):
    angles = []
    crdn1 = list(G.neighbors(atom1))
    crdn1_indeces = [int(atom.split("_")[1]) for atom in crdn1]
    i1 = int(atom1.split("_")[1])
    for i in range(len(crdn1)):
        i2 = crdn1_indeces[i]
        v2 = Rij[i2, i1]
        for j in range(i):
            i3 = crdn1_indeces[j]
            v3 = Rij[i3, i1]
            angle = angle_deg_between(v2, v3)
            angles.append(angle)
            #print(atom1, crdn1[i], crdn1[j], angle, length(v2), length(v3))
    return angles       


# In[50]:


o_m_o_angles = []
for i in m_atoms:
    atom = sc_red[i][1] + "_" + str(i)
    o_m_o_angles += get_angles(G, sc_Rij, atom)

m_o_m_angles = []
for i in o_atoms:
    atom = sc_red[i][1] + "_" + str(i)
    m_o_m_angles += get_angles(G, sc_Rij, atom)


# In[51]:


plt.figure(figsize=(6,8))

ax1 = plt.subplot(211)
hist_m_angle = plt.hist(o_m_o_angles, bins=100, range=(60,180))
plt.text(0.55, 0.9, "Oxygen-Metal-Oxygen", fontsize=12, transform=ax1.transAxes)
plt.title("Histogram of angles")

ax2 = plt.subplot(212)
hist_o_angle = plt.hist(m_o_m_angles, bins=100, range=(60,180))
plt.text(0.55, 0.9, "Metal-Oxygen-Metal", fontsize=12, transform=ax2.transAxes)
_ = plt.xlabel("θ (degree)", fontsize=12)


# ## Histogram of dihedral angles
# 
# You can find the definition at Wikipedia: [Dihedral_angle](https://en.wikipedia.org/wiki/Dihedral_angle)

# In[52]:


def get_dihedral_angles(G, Rij, atom1, atom2):
    dihedral_angles = []
    crdn1 = list(G.neighbors(atom1))
    crdn2 = list(G.neighbors(atom2))
    crdn1.remove(atom2)
    crdn2.remove(atom1)
    for c1 in crdn1:
        for c2 in crdn2:
            if c1 == c2: continue
            j1 = int(atom1.split("_")[1])
            j2 = int(atom2.split("_")[1])
            i1 = int(c1.split("_")[1])
            i2 = int(c2.split("_")[1])
            v0 = Rij[i1, i2]
            v1 = Rij[i1, j1]
            v2 = Rij[i2, j2]
            uv0 = unit_vector(v0)
            w1 = v1 - np.dot(v1, uv0) * uv0
            w2 = v2 - np.dot(v2, uv0) * uv0
            if length(w1) < 1e-8 or length(w2) < 1e-8: continue
            angle = angle_deg_between(w1, w2)
            dihedral_angles.append(angle)
            #print(atom1, atom2, c1, c2, angle, length(v0), length(v1), length(v2), length(w1), length(w2))
    
    return dihedral_angles


# In[53]:


train_dihedral = []
for i in m_atoms:
    atom1 = sc_red[i][1] + "_" + str(i)
    for atom2 in G.neighbors(atom1):
        train_dihedral += get_dihedral_angles(G, sc_Rij, atom1, atom2)


# In[54]:


hist_dihedral = plt.hist(train_dihedral, bins=100)
ax = plt.axes()
plt.text(0.45, 0.9, "Metal-Oxygen-Metal-Oxygen", fontsize=12, transform=ax.transAxes)
_ = plt.title("Histogram of dihedral angles")
_ = plt.xlabel("φ (degree)", fontsize=12)

