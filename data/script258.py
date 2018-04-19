
# coding: utf-8

# # Nomad: an initial exploration in Python with domain knowledge
# 
# In this kernel I'd like to introduce the basics of data exploration in Python, combined with some domain knowledge to help you get started with this Kaggle competition. - Michael Sluydts (updated 21 dec 2017)
# 
# Let's start by loading the essential modules

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Loading the data
# Next it's time to load in the CSV data into a pandas dataframe, which combines the speed of numpy with the accessibility of a spreadsheet. We'll have a look at the geometries later. The column names have been conventionally shortened for easier access:
# * natoms is the number of atoms
# * al, ga and in are the fractions of the respective elements
# * lattice vectors are now a,b and c
# * angles are alpha, beta and gamma
# * E0 is the energy
# * bandgap is the bandgap
# 

# In[ ]:



train = pd.read_csv('../input/train.csv',names=['id', 'spacegroup', 'natoms', 'al',
       'ga', 'in', 'a',
       'b', 'c',
       'alpha', 'beta',
       'gamma', 'E0',
       'bandgap'],header=0,sep=',')
test = pd.read_csv('../input/test.csv',names=['id', 'spacegroup', 'natoms', 'al',
       'ga', 'in', 'a',
       'b', 'c',
       'alpha', 'beta',
       'gamma'],header=0,sep=',')

full = pd.concat([train,test])


# ## Exploring the data
# 
# ### The geometry
# Let's start by having a look at the geometric parameters. This is of course not as good as looking at the actual geometries, but hey there are thousands of those so let's start simple.
# 
# #### The spacegroup
# Space groups contain a lot of information. Sadly enough their number doesn't tell you very much unless you're quite familiar with them. There are 230 spacegroups, but likely only a small subset has been studied. Let's have a look.

# In[ ]:


train['spacegroup'].value_counts(normalize=True).plot.bar()
plt.title('Spacegroup distribution in the training set')
plt.ylabel('% of crystals in spacegroup')
plt.xlabel('Space group')
plt.show()


# As expected, only a handful of spacegroups have been studied. A first good question is, are these the same for the test set?

# In[ ]:


test['spacegroup'].value_counts(normalize=True).plot.bar()
plt.title('Spacegroup distribution in the test set')
plt.ylabel('% of crystals in spacegroup')
plt.xlabel('Space group')
plt.show()


# Okay, we're in luck. The same spacegroup distribution is roughly the same between test and training set, so our models are likely to be relevant. Now let's look at the individual space groups with some information from the <a href="http://www.cryst.ehu.es/">Bilbao Crystallographic Server</a>.
# 
# Space groups get more pleasant as the number becomes higher. Pleasant here means a higher symmetry.
# 
# ##### Spacegroup 12: C2/m
# This is a prismatic monoclinic spacegroup:
# * the base is a parallellogram, i.e. 2 vectors have the same length
# * two lattice vectors are perpendicular, i.e. there is one 90 degree angle
# * it is centrosymmetric, meaning there is a point in the cell with respect to which the atomic positions will be mirrored
# 
# ![Monoclinic symmetry (WikiPedia)](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Clinorhombic_prismC.svg/125px-Clinorhombic_prismC.svg.png)
# 
# Let's investigate how the free angles and lattice parameters differ:
# 
# 

# In[ ]:


train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 12].describe()


# In[ ]:


train[train['spacegroup'] == 12].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))
plt.show()


# We see that alpha and gamma are effectively 90 degrees with small numerical errors, while beta varies between 103.7° to 106.2°. This variation is minimal. The variations on the lattice vectors are more significant. a varies from 12 Å to 24 Å (1 Å = 10^{-10} m), b varies from 2.9Å to 6.7Å and c from 5.7 to 6.9 Å. As shown in the histograms a and b come in two distinct groups which may indicate two different substructures where one has nearly double the length in lattice vector. It is possible the large one is effectively the small one, but with the minimal unit cell doubled and then lowered in symmetry. Keep in mind these boxes are periodically repeated to describe the true geometry.

# ##### Spacegroup 33: Pna2_1
# This is a pyramidal orthorhombic spacegroup:
# * the three lattice vectors have different length
# * the three angles are 90 degrees
# 
# ![Orthorhombic symmetry (WikiPedia)](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Orthorhombic.svg/108px-Orthorhombic.svg.png)
# 
# 
# 
# 

# In[ ]:


train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 33].describe()


# In[ ]:


train[train['spacegroup'] == 33].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))
plt.show()


# Again, there is some numerical noise on the angles. This is not likely to be significant. The a lattice vector again has two groups of which one is roughly double the other.
# a varies between 2.4 and 11.3 Å, b between 8.4 and 9.6 Å and c from 9.04 to 10.26 Å.

# ##### Spacegroup 167: R-3c
# 
# This is a rhombohedral trigonal spacegroup:
# * The three lattice vectors have the same length.
# * The three angles are different from 90 degrees in the rhombohedral from. It would seem this is the hexagonal representation of the cell with two 90 degree angles and one 120 degree angle.
# * A trifold symmetry axis is present.
# 
# ![Orthorhombic symmetry (WikiPedia)](https://upload.wikimedia.org/wikipedia/commons/7/76/Hexagonal_latticeR.svg)

# In[ ]:


train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 167].describe()


# In[ ]:


train[train['spacegroup'] == 167].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))
plt.show()


# The angles are two times 90 and one times 120 with some numerical noise. This is the hexagonal equivalent of the rhombohedral cell. The a lattice vector again has two groups of which one is roughly double the other.
# a varies between 4.83 and 11.04 Å, b between 4.83 and 5.58 Å and c from 13.17 to 14.75 Å.

# ##### Spacegroup 194: P6_3/mmc
# This is a hexagonal spacegroup (dihexagonal dipyramidal): 
# * one angle is 120 degrees, two are 90
# * two lattice vectors have equal length
# 
# ![Hexagonal lattice (WikiPedia)](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Hexagonal_latticeFRONT.svg/160px-Hexagonal_latticeFRONT.svg.png)    

# In[ ]:


train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 194].describe()


# In[ ]:


train[train['spacegroup'] == 194].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))
plt.show()


# a and b have equal length and vary from 3 to 7.1 Å, again divided into two groups, one of which double in length. c varies from 11.67 Å to 25.34 Å. Angles are what we expected, but again with numerical noise.

# ##### Spacegroup 206: Ia-3
# This is a cubic spacegroup (diploidal):
# * all angles are 90 degrees
# * all lattice vectors have the same length
# * I stands for Innenzentriert (body-centered cubic, bcc)
# ![BCC (wikipedia)](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Cubic-body-centered.svg/109px-Cubic-body-centered.svg.png)
#     

# In[ ]:


train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 206].describe()


# In[ ]:


train[train['spacegroup'] == 206].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))
plt.show()


# Angles are now fixed by symmetry, length varies from 9 to 10.3 Å.

# ##### Spacegroup 227: Fd-3m
# This is again a cubic spacegroup (hexoctahedral):
# * all angles are 90 degrees
# * all lattice vectors have the same length
# * F stands for Flachenzentriert (face-centered cubic, fcc)
# ![FCC (wikipedia)](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Cubic-face-centered.svg/109px-Cubic-face-centered.svg.png)
# 

# In[ ]:


train[['a','b','c','alpha','beta','gamma']][train['spacegroup'] == 227].describe()


# In[ ]:


train[train['spacegroup'] == 227].hist(figsize=(12,8),column = ['a','b','c','alpha','beta','gamma'],layout =(2,3))
plt.show()


# Matters are a bit more complicated here as likely a cell which is not cubic has been chosen to represent the cubic pattern, due to the hexagonal subsymmetry. There however is a lot of variation on alpha and beta, which should be further investigated. gamma angles of 30 and 120 point to a hexagonal cell.
# a varies from 8.6 to 11.1 Å, in two groups, b varies from 5.9 to 6.5 Å, c from 14.5 to 16 Å.

# ### Composition
# Let's now have a look at the composition of each of the materials in these space groups

# In[ ]:


train.groupby(['spacegroup'])['al'].describe()


# In[ ]:


train.groupby(['spacegroup'])['ga'].describe()


# In[ ]:


train.groupby(['spacegroup'])['in'].describe()


# Clear compositional variations are visible in the spacegroups
# 
# A summary of the averages:
# * 12: 32% Al, 39% Ga and 29% In
# * 33: 41% Al, 26% Ga and 33% In
# * 167: 43% Al, 32% Ga and 24% In
# * 194: 39% Al, 32% Ga and 28 % In
# * 206: 33% Al, 29% Ga and 38% In
# * 227:  44% Al, 29 % Ga and 28% In
# 
# Note that the standard deviation is very large and not all materials contain Al, Ga and in at the same time. Likely larger cells contain more different elements and smaller cells can have a same starting geometry, but with fewer elements (and thus higher symmetry). It is recommended to generate these tables for the subsets of each spacegroup to get a better idea of their composition.

# ### Stability
# 
# Stability is one of the most important properties. It is gauged by the formation energy here in units of electronVolts (eV). When the formation energy is negative a material will spontaneously form. If it is positive it will cost energy to synthesize and gain energy by falling apart into other materials. A material of four elements can either transform into another quaternary material or fall apart in any combination of ternary (three elements), binary (two elements) or unary (one element) materials provided a reaction can be made where the total number of atoms remains the same before and after the reaction. The formation energy of a new material thus represents the difference in energy stored in its bonds, with respect to known materials. If the formation energy is positive, but not too high it may still exist as either a truly stable material (both methodological and numerical errors are likely) or a metastable material. Metastable materials will eventually convert to stable materials, but this can take a very long time (i.e. diamond will turn into pencil lead).
# 
# I explain this in more detail in 10 mins during my <a href="https://youtu.be/6ueXoqJgpZ0?list=PLvuieV8mZakAuZNX61h6EKertCcs_45xh&t=1252">PhD Defense</a>.
# 
# Now let's have a look whether every spacegroup equally stable.
# 

# In[ ]:


train.groupby(['spacegroup'])['E0'].describe()



# We see a wide range of formation energies in each spacegroup, materials with formation energies < 0.05 are likely to be stable. Materials with energies of 0.5 eV are very likely to be unstable. In some cases this can lead to imperfect bonding which may have geometric effects (which could lead to some geometrical noise).
# 
# Note: the exact definition of this energy should be checked still i.e. it is likely per atom and with respect to binaries and ternaries, but it could also be simply with respect to unary materials.
# 
# How many of these may be stable if we assume 0.05 eV as a cutoff?

# In[ ]:


train[train['E0'] < 0.05].groupby(['spacegroup'])['E0'].describe()


# We see that this is clearly not equal among spacegroup. Spacegroup 12 has most stable materials, spacegroups 194 and 227 has nearly none.

# ### The bandgap
# Finally let's look at the bandgap. The bandgap determines the electronic properties of the material and will be linked both to geometry and spacegroup. Three cases are typically considered:
# * a metal: there is no bandgap
# * a semiconductor: the bandgap is small (Si is 1.12 eV)
# * an insulator: the bandgap is large (can be 10 eV and higher)
# 
# 
# 

# In[ ]:


train.groupby(['spacegroup'])['bandgap'].describe()


# Spacegroup 227 seems to have the lowest gaps, including near-metals, 167 has the highest bandgaps, going up to 5.3 eV.
# 
# To have a transparent conducting oxide the bandgap should be larger than that of light (red light is 1.8, blue light is 3.1 eV). If it is not the case, electrons from lower energy band may use the energy of the light to bridge the bandgap, above which there are empty bands they can access. This means we want a bandgap greater than around 3.2 eV.

# In[ ]:


train[train['bandgap'] >= 3.2].groupby(['spacegroup'])['bandgap'].describe()


# We see that group 167 has most possible TCO's, but how many are likely to be stable?

# In[ ]:


train[(train['bandgap'] >= 3.2) & (train['E0'] < 0.05)].groupby(['spacegroup'])['bandgap'].describe()


# Only three spacegroups have proper candidates here, with 167 showing most promise. Of course it is also interesting to see what the fraction of possible TCO's is compared with the total number of stable materials in each group.

# In[ ]:


train[(train['bandgap'] >= 3.2) & (train['E0'] < 0.05)].groupby(['spacegroup'])['bandgap'].agg([ 'count'])/train[(train['E0'] < 0.05)].groupby(['spacegroup'])['bandgap'].agg([ 'count'])


# We see that all stable materials in group 167 are possible TCO's, despite only having 34 stable entries. Group 12 which had over 100 possible stable materials only has 16% possible TCO's.

# # Conclusions
# I'll end this kernel here for now, I hope it will help some of you both in getting used to pandas and the field of computational materials science. If you have any further questions you would like answered or have suggestions to improve the kernel feel free to let me know.
# 
# ## Some additional references (blatant self-advertising, albeit in a useful way)
# *  **<a href="https://youtu.be/6ueXoqJgpZ0?list=PLvuieV8mZakAuZNX61h6EKertCcs_45xh&t=1252">My Phd Defense</a>**, with booklet in the description. We do research which is highly similar to the Kaggle competition. While the status of the research included in my PhD is now a bit aged it still provides a good start.
# 
# * A **<a href="http://www.compmatphys.org/">mooc</a>** on computational materials physics by my supervisor at the Center for Molecular Modeling:  . It explains all the concepts used to generate a dataset like this, in bite-sized videos.
# 
# * Our Machine Learning in DFT Mendeley group, which gathers machine learning applications on this type of data:
#     * Public: https://www.mendeley.com/community/machine-learning-in-dft/
#     * Private: https://www.mendeley.com/community/ml-in-dft/ - this one is the same as the public one, but has slightly better organization of articles in folders. I'm afraid it didn't let me convert this one to public. :(
# 
# * **<a href="http://www.cryst.ehu.es/">Bilbao crystallographic server</a>**: everything on spacegroups, including their free positions and subgroup relations
# * Geometries are dumps of **<a href="https://wiki.fysik.dtu.dk/ase/">ASE</a>** atoms objects and can be loaded in again with this python package normally, though I did not immediately find the right format in my version of ASE
# EDIT: As Lian has mentioned in the comments, the aims format is what you are looking for. :)
# * Using io.write(atoms,format='vasp') in ase you should be able to convert the geometries back to POSCAR format. You can visualize these with **<a href="http://jp-minerals.org/vesta/en/">VESTA</a>**. There are some tools which also let you do it inside jupyter notebooks, but they'll never be as nice. If you do open VESTA, be sure to edit object => boundaries and get a feel for the periodicity of these materials.
# * My **<a href="https://twitter.com/ePotentia">company twitter account</a>**, for those scientists who want some more machine learning in their life. :) 
# 
# 
