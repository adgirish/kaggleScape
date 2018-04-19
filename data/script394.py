
# coding: utf-8

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


# In[ ]:


"""
Contributions from:
DSEverything - Mean Mix - Math, Geo, Harmonic (LB 0.493) 
https://www.kaggle.com/dongxu027/mean-mix-math-geo-harmonic-lb-0-493
JdPaletto - Surprised Yet? - Part2 - (LB: 0.503)
https://www.kaggle.com/jdpaletto/surprised-yet-part2-lb-0-503
hklee - weighted mean comparisons, LB 0.497, 1ST
https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st
Pranav Pandya - Surprise me! H2O autoML version
Johannesss, the1owl, festa78

Also all comments for changes, encouragement, and forked scripts rock

Keep the Surprise Going
"""

sub1 = pd.read_csv('../input/surprise-me-h2o-automl-version/submission.csv')
sub2 = pd.read_csv('../input/multiple-lightgbm/LGB_sub.csv')
sub3 = pd.read_csv('../input/surprise-me/submission.csv')
sub4 = pd.read_csv('../input/simple-xgboost-lb-0-495/xgb0_submission.csv')
sub5 = pd.read_csv('../input/baseline-3-month-mean-time-series/dumb_result.csv')
sub6 = pd.read_csv('../input/baseline-lb-0-497/dumb_result.csv')
sub7 = pd.read_csv('../input/rrv-modelling-trials/submission.csv')

sub8 = pd.DataFrame()
sub8['id'] = sub1['id']
sub8['visitors'] = 0.4*sub1['visitors']+0.1*sub2['visitors']+0.25*sub3['visitors']+0.25*sub4['visitors']

sub8.to_csv('SubmissonK.csv',index=False)

