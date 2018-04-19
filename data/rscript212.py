import pandas as pd
import numpy as np
import kagglegym
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.structure.modules import TanhLayer
from pybrain.supervised import BackpropTrainer

env = kagglegym.make()
o = env.reset()
o.train = o.train.tail(10000)
col = [c for c in o.train.columns if c not in [env.TARGET_COL_NAME]]

train = pd.read_hdf('../input/train.h5')
train = train[col]
d_mean= train.median(axis=0)
train = []

ds = SupervisedDataSet(len(col), 1)
ds.setField('input' , o.train[col].fillna(d_mean))
ds.setField('target' , pd.DataFrame(o.train['y']))
nn = buildNetwork(len(col), 2, 1, hiddenclass=TanhLayer) #hiddenclass=SigmoidLayer, outclass=LinearLayer
tr = BackpropTrainer(nn, ds, learningrate = 0.001, momentum = 0.0000001, verbose = True)
tr.trainUntilConvergence(maxEpochs=5, continueEpochs=2, verbose = True, validationProportion=0.35)

while True:
    test = o.features[col].fillna(d_mean)
    pred = o.target
    pred['y'] = [float(nn.activate(row)) for row in test.values]
    o, reward, done, info = env.step(pred)
    if done:
        print("Info Result: ", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)