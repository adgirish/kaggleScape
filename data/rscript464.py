from sklearn.utils import shuffle
from collections import Counter
from multiprocessing import *
from random import randint
import pandas as pd
import numpy as np
import copy, random
import math

gp = pd.read_csv('../input/santa-gift-matching/child_wishlist_v2.csv',header=None).drop(0, 1).values
cp = pd.read_csv('../input/santa-gift-matching/gift_goodkids_v2.csv',header=None).drop(0, 1).values

def ANH_SCORE(pred):
    gift_counts = Counter(elem[1] for elem in pred)
    for count in gift_counts.values():
        assert count <= 1000

    for t1 in np.arange(0,5001,3):
        triplet1 = pred[t1]
        triplet2 = pred[t1+1]
        triplet3 = pred[t1+2]
        assert triplet1[1] == triplet2[1] and triplet2[1] == triplet3[1]
    
    for t1 in np.arange(5001,45001, 2):
        twin1 = pred[t1]
        twin2 = pred[t1+1]
        assert twin1[1] == twin2[1]

    tch = 0
    tgh = np.zeros(1000)
    
    for row in pred:
        cid, gid = row

        assert cid < 1e6
        assert gid < 1000
        assert cid >= 0 
        assert gid >= 0
        
        ch = (100 - np.where(gp[cid]==gid)[0]) * 2
        if not ch:
            ch = -1

        gh = (1000 - np.where(cp[gid]==cid)[0]) * 2
        if not gh:
            gh = -1

        tch += ch
        tgh[gid] += gh
    return float(math.pow(tch*10,3) + math.pow(np.sum(tgh),3)) / 8e+27

#print(ANH_SCORE(test))

def ANH_SCORE_ROW(pred):
    tch = 0
    tgh = np.zeros(1000)
    for row in pred:
        cid, gid = row
        ch = (100 - np.where(gp[cid]==gid)[0]) * 2
        if not ch:
            ch = -1
        gh = (1000 - np.where(cp[gid]==cid)[0]) * 2
        if not gh:
            gh = -1
        tch += ch
        tgh[gid] += gh
    return float(math.pow(tch*10,3) + math.pow(np.sum(tgh),3)) / 8e+27 #math.pow(float(tch)/2e8,2) + math.pow(np.mean(tgh)/2e6,2)

def metric_function(c1, c2):
    cid1, gid1 = c1
    cid2, gid2 = c2
    return [ANH_SCORE_ROW([c1,c2]), ANH_SCORE_ROW([[cid1,gid2],[cid2,gid1]])]

def objective_function_swap(otest):
    otest = otest.values
    otest = shuffle(otest, random_state=2017)
    #score1 = ANH_SCORE_ROW(otest)
    for b in range(len(otest)):
        for j in range(b+1,len(otest)):
            mf = metric_function(otest[b], otest[j])
            if mf[0] < mf[1]:
                temp = int(otest[b][1])
                otest[b][1] = int(otest[j][1])
                otest[j][1] = temp
                break
    #score2 = ANH_SCORE_ROW(otest)
    #if score2 > score1:
        #print(score2 - score1)
    otest = pd.DataFrame(otest)
    return otest

def multi_transform(mtest):
    p = Pool(cpu_count())
    mtest = p.map(objective_function_swap, np.array_split(mtest, cpu_count()*30))
    mtest = pd.concat(mtest, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    return mtest

if __name__ == '__main__':
    test = pd.read_csv('../input/baseline-python-ortools-algo-0-933795/submit_verGS.csv')
    test2 = multi_transform(shuffle(test[45001:100000].copy(), random_state=2017))
    test = pd.concat([pd.DataFrame(test[:45001].values), pd.DataFrame(test2), pd.DataFrame(test[100000:].values)], axis=0, ignore_index=True).reset_index(drop=True).values
    test = pd.DataFrame(test)
    test.columns = ['ChildId','GiftId']
    print(ANH_SCORE(test.values))
    test.to_csv('02_public_subm.csv', index=False)