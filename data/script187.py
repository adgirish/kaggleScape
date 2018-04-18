
# coding: utf-8

# Here's a quick attempt at exploring why the train set average response rate could be so different to that observed in the Public LB
# 
# If we make the assumption that there's an underlying temporal pattern to the data, and use the qid values as a proxy for it (higher qid value implies more recent question), then re-sorting the train set by increasing qid and plotting the sliding window of mean response rate should show us some pattern.
# 

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df = pd.read_csv( "../input/train.csv")

df["qmax"]      = df.apply( lambda row: max(row["qid1"], row["qid2"]), axis=1 )
df              = df.sort_values(by=["qmax"], ascending=True)
df["dupe_rate"] = df.is_duplicate.rolling(window=500, min_periods=500).mean()
df["timeline"]  = np.arange(df.shape[0]) / float(df.shape[0])

df.plot(x="timeline", y="dupe_rate", kind="line")
plt.show()


# The above pattern, and the ~16.5% LB response rate reported by others, imply that the Public LB (and possibly Private LB) are potentially sourced from more recent data than the training set.
# 
# If this holds, then we could also use this concept to more appropriately construct validation data splits and training data sampling
# 
# 
