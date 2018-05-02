import pandas as pd
import numpy as np

INPUT_PATH = '../input/'
           
wish = pd.read_csv(INPUT_PATH + 'child_wishlist.csv', header=None).as_matrix()[:, 1:]
gift = pd.read_csv(INPUT_PATH + 'gift_goodkids.csv', header=None).as_matrix()[:, 1:]
answ = np.zeros((len(wish)), dtype=np.int32)
answ[:] = -1
gift_count = np.zeros((len(gift)), dtype=np.int32)

#singles
for k in range(10):
    for i in range(1000):
        for j in range(100):
            c = gift[i, k*100+j]
            if gift_count[i] < 1000 and answ[c] == -1 and c >= 4000:
                answ[c] = i
                gift_count[i] += 1
    for i in range(4000, len(answ)):
        g = wish[i, k]
        if gift_count[g] < 1000 and answ[i] == -1:
            answ[i] = g
            gift_count[g] += 1
#twins
for i in range(0, 4000, 2):
    g = -1
    for j in range(10):
        if gift_count[wish[i][j]] < 999:
            g = wish[i, j];
            break
        elif gift_count[wish[i+1][j]] < 999:
            g = wish[i+1, j]
            break
    if g == -1:
        g = np.argmin(gift_count)
    answ[i] = g
    answ[i+1] = g
    gift_count[g] += 2

#unhappy children
for i in range(4000, len(answ)):
    if answ[i] == -1:
        g = np.argmin(gift_count)
        answ[i] = g
        gift_count[g] += 1
        
out = open('sub.csv', 'w')
out.write('ChildId,GiftId\n')
for i in range(len(answ)):
    out.write(str(i) + ',' + str(answ[i]) + '\n')
out.close()