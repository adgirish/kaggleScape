
# coding: utf-8

# ![title_img](https://cdn.midwestsupplies.com/img/lph/beer-recipe-kits.jpg)
# image from [Midwest Supplies](https://www.midwestsupplies.com/)
# 
# # Beer Recipes Exploratory Analysis
# 
# I created this kernel in a attempt to improve my current exploratory analysis skills as well as my plotting skills. I'm looking at coming up with some reusable code snippets that I'll be able to use in other kernels for analysing other datasets.
# 
# What I'm trying to say is that it may be a bit... messy I guess... so bear with me.
# 
# ## Table of Contents
# ---
# 1. [Imports](#import)  
# 1.1 [Importing the necessary librairies for this kernel](#import_librairies)  
# 1.2 [Importing the dataset into a pandas DataFrame](#import)
# 
# 2. [High Level feel for the dataset](#feel)  
# 2.1 [Shape](#shape)  
# 2.2 [Missing Values](#missing)  
#   * [The Story behind Priming Method and Amount](#priming)  
# 
#   2.3 [Categorical Features](#cat_feats)    
# 2.4 [Numerical Features](#num_feats)    
# 2.5 [Class Imbalance](#class)  
#     * [About Pie Charts ...](#pie)
# 
# 3. [Correlations](#corr)  
# 3.1 [ABV vs. OG](#abv_vs_og)
# 
# 4. [Building a simple classifier](#clf)  
# 4.1 [Preprocessing](#clf_process)  
# 4.2 [Scale](#clf_scale)  
# 4.3 [Train](#clf_train)  
# 4.4 [Test](#clf_test)

# # Imports  <a class="anchor" id="import"></a>
# ## Import necessary librairies  <a class="anchor" id="import_librairies"></a>
# I'll be using the following librairies for my exploratory analysis.
# A librairy that I don't see often in other kernel is [missingno](https://github.com/ResidentMario/missingno). I'm using this library to look at missing values and where they are in the dataset. I like the easiness and the visual representations provided.
# 
# The creator of the librairy says in the readme that it can accomodate up to 50 variables before the visualisation becomes unreadable.  So it's nice for a small dataset like the one here, but for biggger dataset you'll have to watchout for the number of features.

# In[1]:


import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


# ## Importing the dataset and first look <a class="anchor" id="import_dataset"></a>
# We have to watch out for special characters in the input file (forward slashes), hence the use of a different encoding that the default one.

# In[2]:


beer_recipe = pd.read_csv('../input/recipeData.csv', index_col='BeerID', encoding='latin1')
beer_recipe.head()


# # High Level feel for the dataset <a class="anchor" id="feel"></a>
# 
# This section is going to look at very basic information regarding the dataset.
# 
# Since I don't know anything about the data, I want to get an idea of...
# * how big it is
# * how "good" it is
# * etc.
# 
# Let's start with how big  
# 
# ## Shape<a class="anchor" id="shape"></a>

# In[3]:


print(beer_recipe.info(verbose=False))


# OK, so 73K entries and 21 features is not too bad, it's a pretty "small" dataset and it will be easy to build visualisations on it, hopefully :)
# ___
# 
# ## Missing Values <a class="anchor" id="missing"></a>
# 
# Next, let's look at how good the data is... well, right now I'll focus on **missing values**. 
# 
# Just by looking at the first 5 rows above, we can see that there's a couple of null values.  I think it's safe to expect a couple of null value throughout the whole data set.  let's see how many we have.
# 

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
msno.matrix(beer_recipe.sample(500))


# 
# 
# You can also see that the visualization above is only looking at 500 rows of the whole dataset, which might not seem like a lot, but I did check with higher number and the distribution of null values is pretty similar.
# 
# ---
# Anyhow, what have we learned here ?
# 
# * There seems to be some null values in the Style column, which is suprising, because the StyleID columns doesn't seem to have any nulls
#     * The reason behind this is that the Style field is not mandatory, but the website does have an ID (111) for "No Style" (N/A)
# * There are columns that are mostly null (Priming Method / Priming Amount)
#     * it does make sense, based on the fact that they both relate to priming and if one is null, the other is most likely too (which is not always the case)
# * Other columns also have a large number of nulls, but they are not as bad as the Priming ones

# <a class="anchor" id="priming"></a>
# ![priming](https://www.highgravitybrew.com/store/pc/catalog/New-to-Homebrew-Large.jpg)
# 
# ## Let's investigate the worst offenders in missing values found above 
# ### Priming Method & Priming Amount 
# First, what is Priming ???
# 
# Priming refers to adding sugars (glucose of any form) to the fermented beer before bottling.  This will help to carbonate the beer naturally when homebrewing, where specialized equipement to achieve this is unavailable. Corn syrup is the most widely used ingredient for priming and it said to be included in most home brewing kits. This is something we'll try to confirm below).
# 
# Doing a bit of research on homebrewing, it seems like priming is an integral part of the process and that it should included in pretty much every recipe. However, it's important to note that giudelines for homebrewing are suggesting to use different amount of sugars depending on the type (style) of beer you're brewing.  The different styles of beer requires different amount of carbonation, so that's not surprising.
# 
# I still find it very strange that this information is not available for most of the dataset.

# In[5]:


null_priming = beer_recipe['PrimingMethod'].isnull()
print('Priming Method is null on {} rows out of {}, so {} % of the time'.format(null_priming.sum(), len(beer_recipe), round((null_priming.sum()/len(beer_recipe))*100,2)))


# wow... it's null 90% of the time.
# 
# Let's see if it the same across all styles

# In[6]:


style_cnt = beer_recipe.loc[:,['Style','PrimingMethod']]
style_cnt['NullPriming'] = style_cnt['PrimingMethod'].isnull()
style_cnt['Count'] = 1
style_cnt_grp = style_cnt.loc[:,['Style','Count','NullPriming']].groupby('Style').sum()

style_cnt_grp = style_cnt_grp.sort_values('NullPriming', ascending=False)
style_cnt_grp.reset_index(inplace=True)

def stacked_bar_plot(df, x_total, x_sub_total, sub_total_label, y):
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the total
    sns.set_color_codes("pastel")
    sns.barplot(x=x_total, y=y, data=df, label="Total", color="b")

    # Plot
    sns.set_color_codes("muted")
    sns.barplot(x=x_sub_total, y=y, data=df, label=sub_total_label, color="b")
    
    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    sns.despine(left=True, bottom=True)
    
    return f, ax
    
f, ax = stacked_bar_plot(style_cnt_grp[:20], 'Count', 'NullPriming', 'Priming Method is null', 'Style')
ax.set(title='Missing Values in PrimingMethod column, per style', ylabel='', xlabel='Count of Beer Recipes')
sns.despine(left=True, bottom=True)


# Upon further reading of the data notes, it seems that the priming fields are not standardized like the other fields. Which might implied that they were added later, while the database was already created.  This could be a reason for so much missing values.
# 
# Another reason would be that these fields are in a funny place in the Brewer's Friend website. When adding a new recipe to the database, the Priming Method and Priming Amount are under the section **Other Ingredients** which might not be entered most of the time.  It might also be because, as explained earlier, this is such a given that most brewers are not just taking the time to enter it assuming that everyone will prime their own beers (if they try the recipe) like they want to and with what they're used to.
# 

# ## Categorical Features <a class="anchor" id="cat_feats"></a>
# Let's have a look at the categorical features in the dataset.
# 
# In this case, there aren't many, but it's important to understand how many values each of this features have and what's the distribution of each of those values within each features.
# 
# Let's just start by looking at what features are of the object type in our dataframe:

# In[7]:


print( list(beer_recipe.select_dtypes(include=object).columns))


# First, we don't have to look at **Name** or **URL** as these columns should be unique per type of beer.
# 
# Second, **Style** is going to be our class or label, so this is not a feature.
# 
# Third, that leaves us with the following:
# * SugarScale
# * BrewMethod
# * PrimingMethod
# * PrimingAmount
# 
# It's odd that we have **PrimingAmount** here, since we would think that this would be a numerical feature based on name.  Let's look at that before moving on.
# 

# In[8]:


print(beer_recipe.PrimingAmount.unique())


# We can see that the reason this shows up as an *object* dtype is because the unit is included with the amount. If we wanted to use that information in the classifer we would have to clean that up.  But since we found out that 90% of the data doesn't have values in those fields, we'll let that go for now.  We may come back to this if we find out that the fact that the information is present may help us classify some styles of beers.
# 
# ---
# So then let's have a closer look at the 3 remaining categorical features:
# 

# In[9]:


ax = sns.countplot(x='SugarScale', data=beer_recipe)
ax.set(title='Frequency table of possible values in SugarScale')
sns.despine(left=True, bottom=True)

print('SugarScale has {} null values'.format(beer_recipe.SugarScale.isnull().sum()))


# ### Quick note on SugarScale
# As Aaron Santos pointed out to me in the comments, this feature is giving us more information on the type of measurement used for numerical feature concerning gravity (**OG**, **FG** and **BoilGravity**).  
# I'll use that information to recalculate the proper Specific Gravity where a *Plato* scale was used.

# In[10]:


ax = sns.countplot(x='BrewMethod', data=beer_recipe)
ax.set(title='Frequency table of possible values in BrewMethod')
sns.despine(left=True, bottom=True)

print('BrewMethod has {} null values'.format(beer_recipe.BrewMethod.isnull().sum()))


# In[11]:


print('PrimingMethod has {} unique values'.format(beer_recipe.PrimingMethod.nunique()))
print(beer_recipe.PrimingMethod.unique()[:20])


# **SugarScale** and **BrewMethod** have generated pretty nice plot (and have no null values), however since the categories are few in both those fields a clear majority, it will be hard to use these in a classifier.
# 
# **PrimingMethod** did not generate a good plot and that's why I changed the outpout to a count and a list of unique values.  The reason why we see this situation (even with the high number of nulls in that column) is because the field on the website to enter this information is freeform.  If that feature had more information, we could clean this up a bit and try to group together data that looks similar, but given our current knowledge about the data, it doesn't seem like it's worth the effort

# ## Numerical Features <a class="anchor" id="num_feats"></a>
# 
# Let's have a quick look at the features that are in that group:

# In[12]:


print( list( beer_recipe.select_dtypes(exclude=object)))


# First, **StyleID** is part of our class, so this is not a feature.
# 
# But all the other features seems like they are good numerical features and don't have any "problems" that we've uncovered yet.
# 
# ---
# As mentionned above (see my exploration of SugarScale), I've got to align the values in the gravity related columns based on SugarScale used.  
# *The formula I'm using is defined on Brewer's Friends website: [here](https://www.brewersfriend.com/plato-to-sg-conversion-chart/)*

# In[37]:


def get_sg_from_plato(plato):
    sg = 1 + (plato / (258.6 - ( (plato/258.2) *227.1) ) )
    return sg

beer_recipe['OG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['OG']) if row['SugarScale'] == 'Plato' else row['OG'], axis=1)
beer_recipe['FG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['FG']) if row['SugarScale'] == 'Plato' else row['FG'], axis=1)
beer_recipe['BoilGravity_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['BoilGravity']) if row['SugarScale'] == 'Plato' else row['BoilGravity'], axis=1)


# In[38]:


num_feats_list = ['Size(L)', 'OG_sg', 'FG_sg', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime', 'BoilGravity_sg', 'Efficiency', 'MashThickness', 'PitchRate', 'PrimaryTemp']
beer_recipe.loc[:, num_feats_list].describe().T


# With just this table, we can see the following:
# * There are a couple of features that have null values (we'll have to watchout for **MashThickness** and **PitchRate**, since they have a pretty high ratio of nulls)
# * The numerical features in this dataset are on different scales (**OG** and **FG** have low std, but other fields like **Size** or **BoilSize** have very high std).  This means that some sort of scaling is absolutely necessary
# * Every numerical feature will have some important outliers (the only exception being **PitchRate**) because the max value is always very far away from the 75 percentile.
# 
# That third point is pretty important, we should have a look at boxplots for these features.

# In[43]:


# I should define a function that will categorize the features automatically
vlow_scale_feats = ['OG_sg', 'FG_sg', 'BoilGravity_sg', 'PitchRate']
low_scale_feats = ['ABV', 'MashThickness']
mid_scale_feats = ['Color', 'BoilTime', 'Efficiency', 'PrimaryTemp']
high_scale_feats = ['IBU', 'Size(L)',  'BoilSize']


# In[44]:


f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, vlow_scale_feats], orient='h')
ax.set(title='Boxplots of very low scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)


# In[45]:


f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, low_scale_feats], orient='h')
ax.set(title='Boxplots of low scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)


# In[46]:


f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, mid_scale_feats], orient='h')
ax.set(title='Boxplots of medium scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)


# In[47]:


f, ax = plt.subplots(figsize=(12, 8))
ax = sns.boxplot(data=beer_recipe.loc[:, high_scale_feats], orient='h')
ax.set(title='Boxplots of high scale features in Beer Recipe dataset')
sns.despine(left=True, bottom=True)


# I've seperated the fields by scale so that each plot would make a bit of sense and not have any features "overpowered" by other... but even by doing that, the sheer number of outliers in each features just distort each and every plot.
# 
# I think this type of situation is to be expected when dealing with so many different classes, the data for each of those features when seperated by style might make more sense. We'll have to investigate that further.

# ## Class Imbalance <a class="anchor" id="class"></a>
# If the final purpose of this exploratory analysis is to help in setting up a classifier, we have to look at the classes that we have. It's important to define if we have some sort of class imbalance within the dataset.
# 
# The classes in this dataset are the individual **Style** under which each beer is classified and each style is part of a group of styles.  On the Brewers' Friends website, the data is entered in 2 drop-down lists. Here in the data, we only have the individual style under the **Style** and **StyleID** columns.

# In[19]:


print('There are {} different styles of beer'.format(beer_recipe.StyleID.nunique()))


# As we can see we have a lot of different classes.
# 
# I've used the **StyleID** column to get the number, because there's an actual value in there for "No Style" - so that count would be missing if looking at the style columns.
# 
# We've already seen the most popular styles above when looking at missing values in **PrimingMethod** and **PrimingAmount**, but let's see how much of the data these most popular styles represent.

# In[20]:


# Get top10 styles
top10_style = list(style_cnt_grp['Style'][:10].values)

# Group by current count information computed earlier and group every style not in top20 together
style_cnt_other = style_cnt_grp.loc[:, ['Style','Count']]
style_cnt_other.Style = style_cnt_grp.Style.apply(lambda x: x if x in top10_style else 'Other')
style_cnt_other = style_cnt_other.groupby('Style').sum()

# Get ratio of each style
style_cnt_other['Ratio'] = style_cnt_other.Count.apply(lambda x: x/float(len(beer_recipe)))
style_cnt_other = style_cnt_other.sort_values('Count', ascending=False)

f, ax = plt.subplots(figsize=(8, 8))
explode = (0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0)
plt.pie(x=style_cnt_other['Ratio'], labels=list(style_cnt_other.index), startangle = 180, autopct='%1.1f%%', pctdistance= .9, explode=explode)
plt.title('Ratio of styles across dataset')
plt.show()


# Just with the top 10 styles, we can see that they make up about 45% of the data. 
# 
# But even then, the ratio drops severely just after the 2 first style: *American IPA* and *American Pale Ale*. 
# 
# There's a clear imbalance between the classes and we'll have to deal with that when comes time to build the classifier.
# It would be cool to have the "grouping" of styles that the Brewers' Friend website uses in the first dropdown to reduce the number of classes to a more manageable level. I'll see if I can figure out a quick way to bring in that data.

# ### About Pie Charts ...  <a class="anchor" id="pie"></a>
# Well after researching about how to make my pie chart a bit sexier, I found out they have a bad reputation :)
# 
# Even if I still think that the pie chart above still does what it's supposed to do (presenting the importance of the top classes in the dataset), I'll try a different approach and you can tell me in the comments which one looks / works best.
# 
# Let's try to present the same thing using a bar chart:

# In[21]:


#plt.barh(list(style_cnt_other.index), style_cnt_other['Count'])
style_cnt_other['Ratio'].plot(kind='barh', figsize=(12,6),)
plt.title('Ratio of styles across dataset')
sns.despine(left=True, bottom=True)
plt.gca().invert_yaxis()


# <a class="anchor" id="corr"></a>
# ![beer_n_diapers](https://blog.a4everyone.com/wp-content/uploads/2016/06/beer-diapers-correlation-data-analytics-sales-forecasting.jpg)
# # Correlations
# 
# Are there any correlations between the fields in the dataset ???
# 
# For this part of the analysis, I'll focus on the fields which we know are more "reliable" (as per the notes and our previous analysis of null values):
# * Original Gravity (OG)
# * Final Gravity (FG)
# * Alcohol by Volume (ABV)
# * Internation Bitterness Units (IBU)
# * Color

# In[48]:


# create specific df that only contains the fields we're interested in
pairplot_df = beer_recipe.loc[:, ['Style','OG_sg','FG_sg','ABV','IBU','Color']]

# create the pairplot
sns.set(style="dark")
sns.pairplot(data=pairplot_df)
plt.show()


# This might not be the best way to look at this type of information...
# 
# The diagonal plots are really weird and it's hard to see if there is any semblance of distributions. The main reason seems to be based on the scale of the x axis that's always too large.  So this points to the presence of significant outliers within each of those fields.  
# 
# These outliers could be related to certains styles having extreme values, or just plain user error, since all these recipes are entered manually.  Let's see if we can spot those outliers in the most popular styles...

# In[57]:


style_cnt_grp = style_cnt_grp.sort_values('Count', ascending=False)
top5_style = list(style_cnt_grp['Style'][:5].values)

top5_style_df = pairplot_df[pairplot_df['Style'].isin(top5_style)]

f, ax = plt.subplots(figsize=(12, 8))
sns.violinplot(x='Style', y='OG_sg',data=top5_style_df)
plt.show()


# The plots for all the fields and for at least the top20 styles all look like the one above.  Each fields, even when split by style have a significant numbers of outliers, which will make it difficult for a classifier to use.
# 
# The five fields that were mentionned in the notes as being standardized are that way because they are automatically calculated by Brewer's Friend when the recipe is entered.  They depends on the ingredients that are specified and their quantity.
# 
# The reason behind the outliers seems to be because of the *disconnect* between the style of beer and the ingredients used during brewing.  There is certainly a connection, however, the same ingredient could be added to the beer in different forms, which are all going to affect the **Original Gravity** and **Final Gravity** in their own way.  For example:
# 
# * Flaked Barley
# * Rolled Barley
# * Black Barley
# 
# The same quantity of these fermentables added to the recipe will affect the calculated fieds in a different way.
# 

# ## ABV vs. Original Gravity <a class="anchor" id="abv_vs_og"></a>
# Note that the text below was before I realigned the gravity related fields to be all Specific Gravity.  
# The 2 linear relationship that I mention below was because there was 2 different scale in the same feature (managed through the feature SugarScale).  
# Once re-aligned, there's only a single linear relationship between the 2 features.
# 
# ---
# 
# I want to look further at the correlation between these 2 features, since there seems to be something there, but it's hard to make out in the plot seen above.
# 
# I mean, there seems to be 2 different linear relationship between the variable. I'm wondering if these relationship are not based on styles, meaning that the correlation would be stronger for certain specific styles.
# Let's look at the Top5 styles to keep things simple:

# In[24]:


# Get Top5 styles
top5_style = list(style_cnt_grp['Style'][:5].values)
beer_recipe['Top5_Style'] = beer_recipe.Style.apply(lambda x: x if x in top5_style else 'Other')

# Create Reg plot
sns.lmplot(x='ABV', y='OG', hue='Top5_Style', col='Top5_Style', col_wrap=3, data=beer_recipe, n_boot=100)


# Doesn't seem to be the case (that the linear relationship is based on styles - each of the Top5 styles each have values in both "lines".
# 
# ---
# 
# Now let's look at the same thing but with the realigned OG field and see the difference:

# In[49]:


# Create Reg plot
sns.lmplot(x='ABV', y='OG_sg', hue='Top5_Style', col='Top5_Style', col_wrap=3, data=beer_recipe, n_boot=100)


# # Building a simple classifier <a class="anchor" id="clf"></a>
# I think it's time to take a jab at building a quick classifier with the data that we have to see what kind of accuracy we can get out of a simple classifier.
# 
# I'll be using a DecisionTree based model like XGBoost or RandomForest first to see how it performs on this dataset.
# 
# ## Preprocessing the data <a class="anchor" id="clf_process"></a>
# But before we can start looking at the classifier, we'll have to preprocess our data to be able to use it to train our model.
# 
# The steps I'll need to go through are:
# * Label-encoding the categorical features I'll use
# * Fill null values in some of the numerical features (if I decide to use them)
# * Seperate Target Classes from the Features
# * Perform a Train-Test Split so we can evaluate the classifier on part of the data

# In[50]:


# imports
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.model_selection import train_test_split

# Get only the features to be used from original dataset
features_list= ['StyleID', #target
                'OG_sg','FG_sg','ABV','IBU','Color', #standardized fields
                'SugarScale', 'BrewMethod', #categorical features
                'Size(L)', 'BoilSize', 'BoilTime', 'BoilGravity_sg', 'Efficiency', 'MashThickness', 'PitchRate', 'PrimaryTemp' # other numerical features
                ]

clf_data = beer_recipe.loc[:, features_list]

# Label encoding
cat_feats_to_use = list(clf_data.select_dtypes(include=object).columns)
for feat in cat_feats_to_use:
    encoder = LabelEncoder()
    clf_data[feat] = encoder.fit_transform(clf_data[feat])

# Fill null values
num_feats_to_use = list(clf_data.select_dtypes(exclude=object).columns)
for feat in num_feats_to_use:
    imputer = Imputer(strategy='median')
    clf_data[feat] = imputer.fit_transform(clf_data[feat].values.reshape(-1,1))
    
# Seperate Targets from Features
X = clf_data.iloc[:, 1:]
y = clf_data.iloc[:, 0] #the target were the first column I included

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=35)


# In[51]:


#sanity check making sure everything is in number format and no null values
X.info()


# ## Scaling features <a class="anchor" id="clf_scale"></a>
# Since we've seen earlier that the numerical features are on totally different scales, we need to bring them to the same scale.

# In[52]:


# imports
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[53]:


#sanity check again
sanity_df = pd.DataFrame(X_train, columns = X.columns)
sanity_df.describe().T


# ## Training the classifier <a class="anchor" id="clf_train"></a>

# In[54]:


#imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

clf = RandomForestClassifier()
#clf = LogisticRegression()
#clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)


# ## Testing the classifier <a class="anchor" id="clf_test"></a>

# In[ ]:


from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)
print('Accuracy: {}'.format(score))


# Well, that didn't go well at all... 32% accuracy.
# 
# I wasn't expecting much based on the exploration results we've seen above, but this is much lower than what I was thinking we would see.
# 
# ---
# I've tried other models and here are their accuracy score:
# * RandomForest: 32.5 %
# * LogisticRegression: 25.1 %
# * XGBoost: 37 %
# 
# I'll leave the RandomForest in the Kernel so that it runs quicker.
# 
# ---
# 
# Let's see what features the model used to differentiate between the styles.

# In[31]:


feats_imp = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=['FeatureImportance'])
feats_imp = feats_imp.sort_values('FeatureImportance', ascending=False)

feats_imp.plot(kind='barh', figsize=(12,6), legend=False)
plt.title('Feature Importance from RandomForest Classifier')
sns.despine(left=True, bottom=True)
plt.gca().invert_yaxis()


# Make sense that **Color** and **IBU** are at the top of that list. 
# 
# These features are the ones that may vary the most between the different styles.

# ## Still to be done... <a class="anchor" id="tbd"></a>
# 1. Continue investigating correlations between standardized features
# 2. Investigate other types of classifier and their respective score.
# 3. Maybe refine the RandomForest (hyper parameters tuning) to see how much we can get out of it
