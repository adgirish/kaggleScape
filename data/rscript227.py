# Weighted Ensemble of Two Public Kernels (LB:0.519)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import warnings
import time

start_time = time.time()
tcurrent   = start_time

warnings.filterwarnings("ignore")

# Ensemble of two public kernels
# Median-based from Paulo Pinto: https://www.kaggle.com/paulorzp/log-ma-and-days-of-week-means-lb-0-529
# LGBM from Ceshine Lee: https://www.kaggle.com/ceshine/lgbm-starter

filelist1 = ['../input/ensemble/LGBM.csv', '../input/ensemble/Median-based.csv']

outs1 = [pd.read_csv(f, index_col=0) for f in filelist1]
concat_df1 = pd.concat(outs1, axis=1)
concat_df1.columns = ['submission1', 'submission2']
print('concat_df1',concat_df1.head())


# Results of public kernels 
# 3) https://www.kaggle.com/tarobxl/how-the-test-set-is-split-lb-0-532?scriptVersionId=1728938/output v2 LB 0.532
# split_verification.csv
# 4) https://www.kaggle.com/dongxu027/time-series-ets-starter-lb-0-556 v17 LB 0.556
# sub_ets_log
# 5) https://www.kaggle.com/captcalculator/variation-on-dseverything-s-ets-starter-script v14 LB 0.564
# submission_v0.csv

filelist2 = ['../input/ensemble-grocery-01/split_verification.csv',
             '../input/ensemble-grocery-01/sub_ets_log.csv',
             '../input/ensemble-grocery-01/submission_v0.csv']

outs2 = [pd.read_csv(f, index_col=0) for f in filelist2]
concat_df2 = pd.concat(outs2, axis=1)
concat_df2.columns = ['submission3', 'submission4','submission5']
print('\nconcat_df2',concat_df2.head())


#------------- weighted ensemble approach 
print('\n\nEnsemble of Public Kernels with different weights\n')

v                       = 59                        # version
w1                      = np.linspace(0.42,0.58,5)  # weighting (INITIAL test with % of public kernels) - 90%
w2                      = [0.06,0.03,0.01]          # weighting of 3 results using public kernels - 10%


print('Adopted weights options:', w1, '\n')

for i in range(len(w1)-1):
     print ('Ensemble ' + str(i+1) + ': weight of LGBM = ', w1[i], ', Median = ', (0.9-w1[i]),
            ', Other regressors = [ ', w2[0],w2[1],w2[2],' ]')
     
     concat_df1["unit_sales"] = w1[i]*concat_df1['submission1'] + (0.9-w1[i])*concat_df1['submission2'] + \
                                w2[0]*concat_df2['submission3'] + \
                                w2[1]*concat_df2['submission4'] + \
                                w2[2]*concat_df2['submission5'] 

     print('Ensemble resulting:',concat_df1['unit_sales'].head())     
     
     file_name = 'Ens 5 public kernels - ' + str(i+1) + ' - v' + str(v) + '.csv'
     print('File name', file_name, '\n')
     
     concat_df1[["unit_sales"]].to_csv(file_name)
     
t = (time.time() - start_time)/60
print ("Total processing time %s min" % t)