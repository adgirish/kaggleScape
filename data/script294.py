
# coding: utf-8

# Looking at the [leaderboard][1], I was wondering what is the maximum possible score one would be able to achieve since the top score currently is **0.0303** (on 13th Nov, 2016).
# 
# I thought if we could get a ball park value for the maximum possible score, it will be nice to know. So in this notebook, let us do some computation for the same. 
# 
# **Objective:**
# 
# We have train data from 2015-01-28 to 2016-05-28 and the test data is from 2016-06-28.
# 
# The objective is to predict what **additional** products a customer will get in the last month, 2016-06-28, in addition to what they already have at 2016-05-28. 
# 
# **Evaluation:**
# 
# Evaluation metric is Mean Average Precision @ 7 (MAP@7) as seen from this [evaluation page][2].
# 
# If the number of added products for the given user at that time point is 0 (which is from May 2016 to June 2016), then the precision is defined to be 0.
# 
# **Maximum possible score:**
# 
# All those customers who did not buy a product in the given one month will have a map@7 score of 0 though they will be counted in the mean calculation.  So the maximum possible score for the competition is not 1 and a value lesser than 1. We do not know the additional products bought in June 2016 to compute the maximum possible score for June.
# 
# So let us compute the maximum possible score for May 2016 by taking in to account what the customers already have in April 2016. This will get us a fairly good idea on the maximum possible score one would be able to achieve in this competition.
# 
# 
#   [1]: https://www.kaggle.com/c/santander-product-recommendation/leaderboard
#   [2]: https://www.kaggle.com/c/santander-product-recommendation/details/evaluation

# In[ ]:


# importing the modules needed #
import csv
from operator import sub 


# In[ ]:


# name of the target columns #
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1',
               'ind_cco_fin_ult1','ind_cder_fin_ult1',
               'ind_cno_fin_ult1','ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1','ind_ctop_fin_ult1',
               'ind_ctpp_fin_ult1','ind_deco_fin_ult1',
               'ind_deme_fin_ult1','ind_dela_fin_ult1',
               'ind_ecue_fin_ult1','ind_fond_fin_ult1',
               'ind_hip_fin_ult1','ind_plan_fin_ult1',
               'ind_pres_fin_ult1','ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1','ind_valo_fin_ult1',
               'ind_viv_fin_ult1','ind_nomina_ult1',
               'ind_nom_pens_ult1','ind_recibo_ult1']

def getTarget(row):
    """
    Function to fetch the target columns as a list
    """
    tlist = []
    for col in target_cols:
        if row[col].strip() in ['', 'NA']:
            target = 0
        else:
            target = int(float(row[col]))
        tlist.append(target)
    return tlist

data_path = "../input/"
train_file = open(data_path+"train_ver2.csv")

cust_dict = {}
cust_count = 0
map_count = 0
for row in csv.DictReader(train_file):
    cust_id = int(row['ncodpers'])
    date = row['fecha_dato']
    
    if date != '2016-05-28':
        cust_dict[cust_id] = getTarget(row)  
    elif date == '2016-05-28':
        new_products = getTarget(row)
        existing_products = cust_dict.get(cust_id, [0]*24)
        num_new_products = sum([max(x1 - x2,0) for (x1, x2) in zip(new_products, existing_products)])
        if num_new_products >= 1:
            map_count += 1
        cust_count += 1
print("Number of customers in May 2016 : ",cust_count)
print("Number of customers with new products in May 2016 : ",map_count)

train_file.close()


# Though there were **931453 customers in May 2016, only 29712 of them got new products in that month**.
# 
# Considering we are predicting all these new products correctly, we will get a MAP@7 score of

# In[ ]:


print("Max possible MAP@7 score for May 2016: ",29712./931453.)


# Wow.! It is **0.0319 for May 2016** and I think that it will be around the same number for June 2016 as well. 
# 
# I did the same for other months as well and the maximum possible scores for other months are as follows:
# 
# 1. Apr 2016 - 0.0309
# 2. Mar 2016 - 0.0330
# 3. Feb 2016 - 0.0422
# 4. Jan 2016 - 0.0330
# 5. Dec 2015 - 0.0406
# 6. Nov 2015 - 0.0404
# 7. Oct 2015 - 0.0543
# 
# **Leaderboard Probing Methodology:** 
# 
# As discussed in this [forum post][1], the idea is to make submission with 1 product on all rows to determine the maximum score.
# 
# Panos was kind enough to share his results (sparing us 21 submissions.!) of such an exercise. Thanks to him and the scores are:
# 
# 1. ind_cco_fin_ult1 - 0.0096681
# 
# 2. ind_recibo_ult1  - 0.0086845
# 
# 3. ind_tjcr_fin_ult1  - 0.0041178
# 
# 4. ind_reca_fin_ult1  - 0.0032092
# 
# 5. ind_nom_pens_ult1  - 0.0021801
# 
# 6. ind_nomina_ult1  - 0.0021478
# 
# 7. ind_ecue_fin_ult1  - 0.0019961
# 
# 8. cno  - 0.0017839
# 
# 9. ctma  - 0.0004488
# 
# 10. valo  - 0.000278
# 
# 11. ctop  - 0.0001949
# 
# 12. ctpp  - 0.0001142
# 
# 13. fond  - 0.000104
# 
# 14. ctju  - 0.0000502
# 
# 15. hip  - 0.0000161
# 
# 16. plan  - 0.0000126
# 
# 17. pres  - 0.0000054
# 
# 18. cder  - 0.000009
# 
# 19. viv  - 0
# 
# 20. deco  - 0
# 
# 21. deme  - 0
# 
# The last 3 will most probably be 0 as well. 
# 
# So summing them all, we get a score of **0.0350207** for the month of June 2016.
# 
# **Inference:**
# 
# From the script, we see that the maximum possible score varies between 0.0309 to 0.0543 for the previous months.
# 
# From the Public LB probing, the maximum possible score is  ~0.0350207
# 
# **So if the distribution of private LB is same as that of public LB, then we expect the maximum possible score to be around 0.035**
# 
# Happy Kaggling.!
# 
# 
#   [1]: https://www.kaggle.com/c/santander-product-recommendation/forums/t/25727/question-about-map-7
