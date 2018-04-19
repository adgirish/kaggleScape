
# coding: utf-8

# In[ ]:


# import packages

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.unicode_minus'] = False


# In[ ]:


# read data into DataFrame

def make_df(fin):
    """
    Args:
        fin (str) - file name with training or test data
    Returns:
        DataFrame with renamed columns (personal preference)
    """
    df = pd.read_csv(fin)
    df = df.rename(columns={'spacegroup' : 'sg',
                            'number_of_total_atoms' : 'Natoms',
                            'percent_atom_al' : 'x_Al',
                            'percent_atom_ga' : 'x_Ga',
                            'percent_atom_in' : 'x_In',
                            'lattice_vector_1_ang' : 'a',
                            'lattice_vector_2_ang' : 'b',
                            'lattice_vector_3_ang' : 'c',
                            'lattice_angle_alpha_degree' : 'alpha',
                            'lattice_angle_beta_degree' : 'beta',
                            'lattice_angle_gamma_degree' : 'gamma',
                            'formation_energy_ev_natom' : 'E',
                            'bandgap_energy_ev' : 'Eg'})
    return df

# folder which contains data folders
input_dir = os.path.join('..', 'input')
# folder which contains competition data
path_to_train_data = os.path.join(input_dir, 'nomad2018-predict-transparent-conductors')
# training data
f_train = os.path.join(path_to_train_data, 'train.csv')
# make DataFrame of training data
df_train = make_df(f_train)
df_train.head()


# In[ ]:


# retrieve list of elemental properties

def get_prop_list(path_to_element_data):
    """
    Args:
        path_to_element_data (str) - path to folder of elemental property files
    Returns:
        list of elemental properties (str) which have corresponding .csv files
    """
    return [f[:-4] for f in os.listdir(path_to_element_data)]

# folder which contains element data
path_to_element_data = os.path.join(input_dir, 'elemental-properties')
# get list of properties which have data files
properties = get_prop_list(path_to_element_data)
print(sorted(properties))


# In[ ]:


# retrieve elemental properties

def get_prop(prop, path_to_element_data):
    """
    Args:
        prop (str) - name of elemental property
        path_to_element_data (str) - path to folder of elemental property files
    Returns:
        dictionary of {element (str) : property value (float)}
    """
    fin = os.path.join(path_to_element_data, prop+'.csv')
    with open(fin) as f:
        all_els = {line.split(',')[0] : float(line.split(',')[1][:-1]) for line in f}
        my_els = ['Al', 'Ga', 'In']
        return {el : all_els[el] for el in all_els if el in my_els}

# make nested dictionary which maps {property (str) : {element (str) : property value (float)}}
prop_dict = {prop : get_prop(prop, path_to_element_data) for prop in properties}
print('The mass of aluminum is %.2f amu' % prop_dict['mass']['Al'])


# In[ ]:


# average each property using the composition

def avg_prop(x_Al, x_Ga, x_In, prop):
    """
    Args:
        x_Al (float or DataFrame series) - concentration of Al
        x_Ga (float or DataFrame series) - concentration of Ga
        x_In (float or DataFrame series) - concentration of In
        prop (str) - name of elemental property
    Returns:
        average property for the compound (float or DataFrame series), 
        weighted by the elemental concentrations
    """
    els = ['Al', 'Ga', 'In']
    concentration_dict = dict(zip(els, [x_Al, x_Ga, x_In]))
    return np.sum(prop_dict[prop][el] * concentration_dict[el] for el in els)

# add averaged properties to DataFrame
for prop in properties:
    df_train['_'.join(['avg', prop])] = avg_prop(df_train['x_Al'], 
                                                 df_train['x_Ga'],
                                                 df_train['x_In'],
                                                 prop)
list(df_train)


# In[ ]:


# calculate the volume of the structure

def get_vol(a, b, c, alpha, beta, gamma):
    """
    Args:
        a (float) - lattice vector 1
        b (float) - lattice vector 2
        c (float) - lattice vector 3
        alpha (float) - lattice angle 1 [radians]
        beta (float) - lattice angle 2 [radians]
        gamma (float) - lattice angle 3 [radians]
    Returns:
        volume (float) of the parallelepiped unit cell
    """
    return a*b*c*np.sqrt(1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                           - np.cos(alpha)**2
                           - np.cos(beta)**2
                           - np.cos(gamma)**2)

# convert lattice angles from degrees to radians for volume calculation
lattice_angles = ['alpha', 'beta', 'gamma']
for lang in lattice_angles:
    df_train['_'.join([lang, 'r'])] = np.pi * df_train[lang] / 180
    
# compute the cell volumes 
df_train['vol'] = get_vol(df_train['a'], df_train['b'], df_train['c'],
                          df_train['alpha_r'], df_train['beta_r'], df_train['gamma_r'])
df_train[['a','b','c','alpha_r','beta_r','gamma_r','vol']].head()


# In[ ]:


# calculate the atomic density

# this is known to correlate with stability or bonding strength
df_train['atomic_density'] = df_train['Natoms'] / df_train['vol']   

df_train[['a','b','c','alpha','beta','gamma','vol', 'Natoms', 'atomic_density']].head()


# In[ ]:


# function to visualize data using scatter plots

def plot_scatter(x, y, xlabel, ylabel):
    """
    Args:
        x (str) - DataFrame column for x-axis
        y (str) - DataFrame column for y-axis
        xlabel (str) - name for x-axis
        ylabel (str) - name for y-axis
    Returns:
        matplotlib scatter plot of y vs x
    """
    s = 75
    lw = 0
    alpha = 0.05
    color = 'blue'
    marker = 'o'
    axis_width = 1.5
    maj_tick_len = 6
    fontsize = 16
    label = '__nolegend__'
    ax = plt.scatter(df_train[x].values, df_train[y].values,
                     marker=marker, color=color, s=s, 
                     lw=lw, alpha=alpha, label=label)
    xrange = abs(df_train[x].max() - df_train[x].min())
    yrange = abs(df_train[y].max() - df_train[y].min())
    cushion = 0.1
    xmin = df_train[x].min() - cushion*xrange
    xmax = df_train[x].max() + cushion*xrange
    ymin = df_train[y].min() - cushion*yrange
    ymax = df_train[y].max() + cushion*yrange
    ax = plt.xlim([xmin, xmax])
    ax = plt.ylim([ymin, ymax])
    ax = plt.xlabel(xlabel, fontsize=fontsize)
    ax = plt.ylabel(ylabel, fontsize=fontsize)
    ax = plt.xticks(fontsize=fontsize)
    ax = plt.yticks(fontsize=fontsize)
    ax = plt.tick_params('both', length=maj_tick_len, width=axis_width, 
                         which='major', right=True, top=True)
    return ax


# In[ ]:


# visualize the relationship between our target properties and the atomic density

fig1 = plt.figure(1, figsize=(8, 4))
ax1 = plt.subplot(121)
ax1 = plot_scatter('atomic_density', 'E', 
                   'atomic density (atoms/vol)', 'formation energy (eV/atom)')
ax2 = plt.subplot(122)
ax2 = plot_scatter('atomic_density', 'Eg', 
                   'atomic density (atoms/vol)', 'band gap (eV)')
plt.tight_layout()
plt.show()
plt.close()


# In[ ]:


# visualize some of the element-based features

fig2 = plt.figure(2, figsize=(8, 7))
ax1 = plt.subplot(221)
ax1 = plot_scatter('avg_electronegativity', 'E', 
                   '', 'formation energy (eV/atom)')
ax2 = plt.subplot(222)
ax2 = plot_scatter('avg_IP', 'E', 
                   '', '')
ax3 = plt.subplot(223)
ax3 = plot_scatter('avg_electronegativity', 'Eg', 
                   'average electronegativity', 'band gap (eV)')
ax4 = plt.subplot(224)
ax4 = plot_scatter('avg_IP', 'Eg', 
                   'average ionization potential (eV)', '')
plt.tight_layout()
plt.show()
plt.close()


# In[ ]:


# visualize the effect of composition on the atomic density and averaged atomic properties

fig3 = plt.figure(3, figsize=(9, 9))
y1_label = 'atomic density'
y2_label = 'average mass (amu)'
y3_label = 'average LUMO (eV)'
ax1 = plt.subplot(331)
ax1 = plot_scatter('x_Al', 'atomic_density', 
                   '', y1_label)
ax2 = plt.subplot(332)
ax2 = plot_scatter('x_Ga', 'atomic_density', 
                   '', '')
ax3 = plt.subplot(333)
ax3 = plot_scatter('x_In', 'atomic_density', 
                   '', '')
ax4 = plt.subplot(334)
ax4 = plot_scatter('x_Al', 'avg_mass', 
                   '', y2_label)
ax5 = plt.subplot(335)
ax5 = plot_scatter('x_Ga', 'avg_mass', 
                   '', '')
ax6 = plt.subplot(336)
ax6 = plot_scatter('x_In', 'avg_mass', 
                   '', '')
ax7 = plt.subplot(337)
ax7 = plot_scatter('x_Al', 'avg_LUMO', 
                   'Al concentration', y3_label)
ax8 = plt.subplot(338)
ax8 = plot_scatter('x_Ga', 'avg_LUMO', 
                   'Ga concentration', '')
ax9 = plt.subplot(339)
ax9 = plot_scatter('x_In', 'avg_LUMO', 
                   'In concentration', '')
plt.tight_layout()
plt.show()
plt.close() 


# In[ ]:


# use random forests to quantify the importances of each feature

# list of columns not to be used for training
non_features = ['id', 'E', 'Eg',
               'alpha_r', 'beta_r', 'gamma_r']

# list of columns to be used for training each model
features = [col for col in list(df_train) if col not in non_features]
print('%i features: %s' % (len(features), features))

# make feature matrix
X = df_train[features].values

# make target columns for each target property
y_E = df_train['E'].values
y_Eg = df_train['Eg'].values

# split into training and test for the purposes of this demonstration
test_size = 0.2
rstate = 42
X_train_E, X_test_E, y_train_E, y_test_E = train_test_split(X, y_E, 
                                                            test_size=test_size,
                                                            random_state=rstate)
X_train_Eg, X_test_Eg, y_train_Eg, y_test_Eg = train_test_split(X, y_Eg, 
                                                                test_size=test_size, 
                                                                random_state=rstate)

# number of base decision tree estimators
n_est = 100
# maximum depth of any given decision tree estimator
max_depth = 5
# random state variable
rstate = 42
# initialize a random forest algorithm
rf_E = RandomForestRegressor(n_estimators=n_est, 
                             max_depth=max_depth,
                             random_state=rstate)
rf_Eg = RandomForestRegressor(n_estimators=n_est, 
                             max_depth=max_depth,
                             random_state=rstate)
# fit to training data
rf_E.fit(X_train_E, y_train_E)
rf_Eg.fit(X_train_Eg, y_train_Eg)


# In[ ]:


# report the most important featuers for predicting each target

# collect ranking of most "important" features for E
importances_E =  rf_E.feature_importances_
descending_indices_E = np.argsort(importances_E)[::-1]
sorted_importances_E = [importances_E[idx] for idx in descending_indices_E]
sorted_features_E = [features[idx] for idx in descending_indices_E]
print('most important feature for formation energy is %s' % sorted_features_E[0])

# collect ranking of most "important" features for Eg
importances_Eg =  rf_Eg.feature_importances_
descending_indices_Eg = np.argsort(importances_Eg)[::-1]
sorted_importances_Eg = [importances_Eg[idx] for idx in descending_indices_Eg]
sorted_features_Eg = [features[idx] for idx in descending_indices_Eg]
print('most important feature for band gap is %s' % sorted_features_Eg[0])


# In[ ]:


# plot the feature importances

def plot_importances(X_train, sorted_features, sorted_importances):
    """
    Args:
        X_train (nd-array) - feature matrix of shape (number samples, number features)
        sorted_features (list) - feature names (str)
        sorted_importances (list) - feature importances (float)
    Returns:
        matplotlib bar chart of sorted importances
    """
    axis_width = 1.5
    maj_tick_len = 6
    fontsize = 14
    bar_color = 'lightblue'
    align = 'center'
    label = '__nolegend__'
    ax = plt.bar(range(X_train.shape[1]), sorted_importances,
                 color=bar_color, align=align, label=label)
    ax = plt.xticks(range(X_train.shape[1]), sorted_features, rotation=90)
    ax = plt.xlim([-1, X_train.shape[1]])
    ax = plt.ylabel('Average impurity decrease', fontsize=fontsize)
    ax = plt.tick_params('both', length=maj_tick_len, width=axis_width, 
                         which='major', right=True, top=True)
    ax = plt.xticks(fontsize=fontsize)
    ax = plt.yticks(fontsize=fontsize)
    ax = plt.tight_layout()
    return ax

fig3 = plt.figure(3, figsize=(11,6))
ax1 = plt.subplot(121)
ax1 = plot_importances(X_train_E, sorted_features_E, sorted_importances_E)
ax1 = plt.legend(['formation energy'], fontsize=14, frameon=False)
ax2 = plt.subplot(122)
ax2 = plot_importances(X_train_Eg, sorted_features_Eg, sorted_importances_Eg)
ax2 = plt.legend(['band gap'], fontsize=14, frameon=False)
plt.tight_layout()
plt.show()
plt.close()


# In[ ]:


# evaluate performance of the random forest models

def rmsle(actual, predicted):
    """
    Args:
        actual (1d-array) - array of actual values (float)
        predicted (1d-array) - array of predicted values (float)
    Returns:
        root mean square log error (float)
    """
    return np.sqrt(np.mean(np.power(np.log1p(actual)-np.log1p(predicted), 2)))

def plot_actual_pred(train_actual, train_pred, 
                     test_actual, test_pred,
                     target):
    """
    Args:
        train_actual (1d-array) - actual training values (float)
        train_pred (1d-array) - predicted training values (float)
        test_actual (1d-array) - actual test values (float)
        test_pred (1d-array) - predicted test values (float)
        target (str) - target property
    Returns:
        matplotlib scatter plot of actual vs predicted
    """
    s = 75
    lw = 0
    alpha = 0.2
    train_color = 'orange'
    train_marker = 's'
    test_color = 'red'
    test_marker = '^'
    axis_width = 1.5
    maj_tick_len = 6
    fontsize = 16
    label = '__nolegend__'
    ax = plt.scatter(train_pred, train_actual,
                     marker=train_marker, color=train_color, s=s, 
                     lw=lw, alpha=alpha, label='train')
    ax = plt.scatter(test_pred, test_actual,
                     marker=test_marker, color=test_color, s=s, 
                     lw=lw, alpha=alpha, label='test')
    ax = plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.4)    
    all_vals = list(train_pred) + list(train_actual) + list(test_pred) + list(test_actual)
    full_range = abs(np.max(all_vals) - np.min(all_vals))
    cushion = 0.1
    xmin = np.min(all_vals) - cushion*full_range
    xmax = np.max(all_vals) + cushion*full_range
    ymin = xmin
    ymax = xmax    
    ax = plt.xlim([xmin, xmax])
    ax = plt.ylim([ymin, ymax])
    ax = plt.plot([xmin, xmax], [ymin, ymax], 
                  lw=axis_width, color='black', ls='--', 
                  label='__nolegend__')
    ax = plt.xlabel('predicted ' + target, fontsize=fontsize)
    ax = plt.ylabel('actual ' + target, fontsize=fontsize)
    ax = plt.xticks(fontsize=fontsize)
    ax = plt.yticks(fontsize=fontsize)
    ax = plt.tick_params('both', length=maj_tick_len, width=axis_width, 
                         which='major', right=True, top=True)
    return ax  

y_train_E_pred = rf_E.predict(X_train_E)
y_test_E_pred = rf_E.predict(X_test_E)
target_E = 'formation energy (eV/atom)'
print('RMSLE for formation energies = %.3f eV/atom (training) and %.3f eV/atom (test)' 
      % (rmsle(y_train_E, y_train_E_pred),  (rmsle(y_test_E, y_test_E_pred))))
y_train_Eg_pred = rf_Eg.predict(X_train_Eg)
y_test_Eg_pred = rf_Eg.predict(X_test_Eg)
target_Eg = 'band gap (eV)'
print('RMSLE for band gaps = %.3f eV (training) and %.3f eV (test)' 
      % (rmsle(y_train_Eg, y_train_Eg_pred), (rmsle(y_test_Eg, y_test_Eg_pred))))
fig4 = plt.figure(4, figsize=(11,5))
ax1 = plt.subplot(121)
ax1 = plot_actual_pred(y_train_E, y_train_E_pred,
                       y_test_E, y_test_E_pred,
                       target_E)
ax2 = plt.subplot(122)
ax2 = plot_actual_pred(y_train_Eg, y_train_Eg_pred,
                       y_test_Eg, y_test_Eg_pred,
                       target_Eg)
plt.tight_layout()
plt.show()
plt.close()

