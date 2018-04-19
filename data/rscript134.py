# All credits go to original authors!

import pandas as pd
import numpy as np

print('2 momths are left to end this competition and **blends** are giving better results than single models.')
print('looks like this competition will also end up like toxic comment classifier challenge. Blends, blends everywhere !!')

test_files = ['../input/lewis-undersampler-9562-version/pred.csv',
              '../input/weighted-app-chanel-os/subnew.csv',
              '../input/single-xgboost-lb-0-966/xgb_sub.csv',
              '../input/swetha-s-xgboost-revised/xgb_sub5.csv',
              '../input/lightgbm-fixing-unbalanced-data-lb-0-9680/sub_lgb_balanced99.csv',
              '../input/lightgbm-with-count-features/sub_lgb_balanced99.csv',
              '../input/deep-learning-support-imbalance-architect-9671/imbalanced_data.csv',
              '../input/rank-averaging-on-talkingdata/rank_averaged_submission.csv',
              '../input/lightgbm-smaller/submission.csv',
              '../input/adding-to-the-blender-lb-0-9690/average_result.csv',
              '../input/do-not-congratulate/sub_mix_logits_ranks.csv']

ll = []
for test_file in test_files:
    print('read ' + test_file)
    ll.append(pd.read_csv(test_file, encoding='utf-8'))
n_models = len(ll)
pp = n_models
weights = [0.2*0.05, 0.2*0.15, 0.2*0.12, 0.2*0.05, 0.2*0.33, 0.2*0.3, 0.2, 0.1, 0.05, 0.20, 0.25]
cc = 'is_attributed'
print(np.corrcoef([ll[pp - 5][cc], ll[pp - 4][cc], ll[pp - 3][cc], ll[pp - 2][cc], ll[pp - 1][cc]]))
print('ALWAYS BLEND NON CORRELATED RESULTS TO PREVENT OVERFITTING..')

print('predict')
test_predict_column = [0.] * len(ll[0][cc])
for ind in range(0, n_models):
    test_predict_column += ll[ind][cc] * weights[ind]

print('make result')
final_result = ll[0]['click_id']
final_result = pd.concat((final_result, pd.DataFrame(
    {cc: test_predict_column})), axis=1)
final_result.to_csv("blend_3.csv", index=False)
