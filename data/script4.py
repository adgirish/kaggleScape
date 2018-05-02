
# coding: utf-8

# # CLEANING DATA OFF 

# IN THIS NOTEBOOK WE ARE GETTING RID OFF ERROR AND NOISE BY CLEANING OUT INCONSISTENCIES, DETECTING MISPLACED VALUES AND PUTTING THEM INTO THE RIGHT CELLS.

# #### DEALING WITH THE FOLLOWING FEATURES
# * ---------------------------------------------------------------
# * price_doc: sale price (this is the target variable)
# * id: transaction id
# * timestamp: date of transaction
# * full_sq: total area in square meters, including loggias, balconies and other non-residential areas
# * life_sq: living area in square meters, excluding loggias, balconies and other non-residential areas
# * floor: for apartments, floor of the building
# * max_floor: number of floors in the building
# * material: wall material
# * build_year: year built
# * num_room: number of living rooms
# * kitch_sq: kitchen area
# * state: apartment condition
# * product_type: owner-occupier purchase or investment
# * sub_area: name of the district

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'nbagg')
import xgboost as xgb
import seaborn as sns


# In[ ]:


train = pd.read_csv("../input/train.csv", encoding= "utf_8")
test = pd.read_csv("../input/test.csv", encoding= "utf_8")


# # IMPORTANT NOTES ABOUT FEATURES
# THESE ARE SOME OF THE SUMMARY NOTES GAINED FROM KAGGLE DISCUSSIONS, QUESTIONS AND ANSWERS FROM SBERBANK
# 
# * CHECK LIFE SQ, FULL SQ, KITCH SQ FOR CONSISTENCY (DONE)
# * BUILD YEAR CAN BE IN FUTURE - PRE INVESTMENT TYPE (DONE)
# * BUILD YEAR 0 AND 1 ARE MISTAKES (DONE)
# * CHECK TRAIN AND TEST PRODUCT TYPES (DONE)
# * CHECK NUM OF ROOMS FOR CONSISTENCY (DONE)
# * MATERIAL EXPLAINED: 1 - panel, 2 - brick, 3 - wood, 4 - mass concrete, 5 - breezeblock, 6 - mass concrete plus brick
# * STATE EXPLAINED: 4 BEST 1 WORST
# * KITCHEN INCLUDED IN LIFE SQ CHECK INCONSISTENCY (DONE)
# * FULL SQ > LIFE SQ (MOST PROBABLY) (DONE)
# * KM DISTANCES ARE AIRLINE DISTANCES
# * RAION POPUL AND FULL ALL ARE SAME CALC FROM DIFF SOURCES

# ### FIRST SET OF FEATURES

# In[ ]:


first_feat = ["id","timestamp","price_doc", "full_sq", "life_sq",
"floor", "max_floor", "material", "build_year", "num_room",
"kitch_sq", "state", "product_type", "sub_area"]


# In[ ]:


first_feat = ["id","timestamp", "full_sq", "life_sq",
"floor", "max_floor", "material", "build_year", "num_room",
"kitch_sq", "state", "product_type", "sub_area"]


# #### CORRECTIONS RULES FOR FULL_SQ AND LIFE_SQ (APPLY TO TRAIN AND TEST):
#  * IF LIFE SQ >= FULL SQ MAKE LIFE SQ NP.NAN
#  * IF LIFE SQ < 5 NP.NAN
#  * IF FULL SQ < 5 NP.NAN 
#  * KITCH SQ < LIFE SQ
#  * IF KITCH SQ == 0 OR 1 NP.NAN
#  * CHECK FOR OUTLIERS IN LIFE SQ, FULL SQ AND KITCH SQ
#  * LIFE SQ / FULL SQ MUST BE CONSISTENCY (0.3 IS A CONSERVATIVE RATIO)

# In[ ]:


bad_index = train[train.life_sq > train.full_sq].index
train.ix[bad_index, "life_sq"] = np.NaN


# In[ ]:


equal_index = [601,1896,2791]
test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]


# In[ ]:


bad_index = test[test.life_sq > test.full_sq].index
test.ix[bad_index, "life_sq"] = np.NaN


# In[ ]:


bad_index = train[train.life_sq < 5].index
train.ix[bad_index, "life_sq"] = np.NaN


# In[ ]:


bad_index = test[test.life_sq < 5].index
test.ix[bad_index, "life_sq"] = np.NaN


# In[ ]:


bad_index = train[train.full_sq < 5].index
train.ix[bad_index, "full_sq"] = np.NaN


# In[ ]:


bad_index = test[test.full_sq < 5].index
test.ix[bad_index, "full_sq"] = np.NaN


# In[ ]:


kitch_is_build_year = [13117]
train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]


# In[ ]:


bad_index = train[train.kitch_sq >= train.life_sq].index
train.ix[bad_index, "kitch_sq"] = np.NaN


# In[ ]:


bad_index = test[test.kitch_sq >= test.life_sq].index
test.ix[bad_index, "kitch_sq"] = np.NaN


# In[ ]:


bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.ix[bad_index, "kitch_sq"] = np.NaN


# In[ ]:


bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.ix[bad_index, "kitch_sq"] = np.NaN


# In[ ]:


bad_index = train[(train.full_sq > 210) * (train.life_sq / train.full_sq < 0.3)].index
train.ix[bad_index, "full_sq"] = np.NaN


# In[ ]:


bad_index = test[(test.full_sq > 150) * (test.life_sq / test.full_sq < 0.3)].index
test.ix[bad_index, "full_sq"] = np.NaN


# In[ ]:


bad_index = train[train.life_sq > 300].index
train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN


# In[ ]:


bad_index = test[test.life_sq > 200].index
test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN


# #### BUILDYEAR CAN BE IN FUTURE (TYPE OF PRODUCTS)
# * CHECK BUILD YEAR FOR EACH PRODUCT TYPE
# * CHECK BUILD YEAR FOR CONSISTENCY (IF BUILD YEAR < 1500)

# In[ ]:


train.product_type.value_counts(normalize= True)


# In[ ]:


test.product_type.value_counts(normalize= True)


# In[ ]:


bad_index = train[train.build_year < 1500].index
train.ix[bad_index, "build_year"] = np.NaN


# In[ ]:


bad_index = test[test.build_year < 1500].index
test.ix[bad_index, "build_year"] = np.NaN


# #### CHECK NUM OF ROOMS
# * IS THERE A OUTLIER ?
# * A VERY SMALL OR LARGE NUMBER ?
# * LIFE SQ / ROOM > MIN ROOM SQ (LET'S SAY 5 SQ FOR A ROOM MIGHT BE OK)
# * IF NUM ROOM == 0 SET TO NP.NAN
# * DETECT ABNORMAL NUM ROOMS GIVEN LIFE AND FULL SQ

# In[ ]:


bad_index = train[train.num_room == 0].index 
train.ix[bad_index, "num_room"] = np.NaN


# In[ ]:


bad_index = test[test.num_room == 0].index 
test.ix[bad_index, "num_room"] = np.NaN


# In[ ]:


bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.ix[bad_index, "num_room"] = np.NaN


# In[ ]:


bad_index = [3174, 7313]
test.ix[bad_index, "num_room"] = np.NaN


# #### CHECK FLOOR AND MAX FLOOR 
# * FLOOR == 0 AND MAX FLOOR == 0 POSSIBLE ??? WE DON'T HAVE IT IN TEST SO NP.NAN
# * FLOOR == 0 0R MAX FLOOR == 0 ??? WE DON'T HAVE IT IN TEST SO NP.NAN (NP.NAN IF MAX FLOOR == 0 FOR BOTH TEST TRAIN)
# * CHECK FLOOR < MAX FLOOR (IF FLOOR > MAX FLOOR -> MAX FLOOR NP.NAN)
# * CHECK FOR OUTLIERS

# In[ ]:


bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.ix[bad_index, ["max_floor", "floor"]] = np.NaN


# In[ ]:


bad_index = train[train.floor == 0].index
train.ix[bad_index, "floor"] = np.NaN


# In[ ]:


bad_index = train[train.max_floor == 0].index
train.ix[bad_index, "max_floor"] = np.NaN


# In[ ]:


bad_index = test[test.max_floor == 0].index
test.ix[bad_index, "max_floor"] = np.NaN


# In[ ]:


bad_index = train[train.floor > train.max_floor].index
train.ix[bad_index, "max_floor"] = np.NaN


# In[ ]:


bad_index = test[test.floor > test.max_floor].index
test.ix[bad_index, "max_floor"] = np.NaN


# In[ ]:


train.floor.describe(percentiles= [0.9999])


# In[ ]:


bad_index = [23584]
train.ix[bad_index, "floor"] = np.NaN


# CHECK MATERIAL

# In[ ]:


train.material.value_counts()


# In[ ]:


test.material.value_counts()


# CHECK STATE

# In[ ]:


train.state.value_counts()


# In[ ]:


bad_index = train[train.state == 33].index
train.ix[bad_index, "state"] = np.NaN


# In[ ]:


test.state.value_counts()


# ### SAVE TEST AND TRAIN AS CLEAN

# In[ ]:


test.to_csv("test_clean.csv", index= False, encoding= "utf_8")
train.to_csv("train_clean.csv", index = False, encoding= "utf_8")

