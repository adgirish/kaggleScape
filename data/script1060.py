
# coding: utf-8

# #Sequence prefixes lookup
# 
# This approach is described in Neil Sloane's book [Encyclopedia of Integer Sequences](http://neilsloane.com/doc/EIStext.pdf) (chapter 2.5: analysis of differences).
# 
# Suppose we have a sequence `[ 1, 8, 27, 64, 125, 216 ]` and want to predict next term. This sequence is cubes: $ f(n)=n^3 $.
# 
# An array of differences is `[ 8 - 1, 27 - 8, 64 - 27, 125 - 64, 216 - 125 ] = [ 7, 19, 37, 61, 91 ]`. A simple quick look at this sequence doesn't let us recognize it. We'll define it as $ f_1( n ) = f_0( n + 1 ) - f_0( n ) $ where $ f_0( n ) = f( n ) $ is the original sequence. If we calculate differences once again, we get: `[ 19 - 7, 37 - 19, 61 - 37, 91 - 61 ] = [ 12, 18, 24, 30 ]`. And this pattern is already recognizable. Let's take one more: `[ 18 - 12, 24 - 18, 30 - 24 ] = [ 6, 6, 6 ]`. A constant value of `6`. How to get a next element? We need to add one more constant `6` to the array of third differences. Then the next element of second differences $ f_2( n ) = f_2( n - 1 ) + f_3( n - 1 ) $. So, $ f_2( 5 ) = f_2( 4 ) + f_3( 4 ) = 30 + 6 = 36 $. Analogously, $ f_1( 6 ) = f_1( 5 ) + f_2( 5 ) = 91 + 36 = 127 $. Finally, $ f_0( 7 ) = f_0( 6 ) + f_1( 6 ) = 216 + 127 = 343 $. Check: $ f( 7 ) = 7^3 = 343 = f_0( 7 ) $ – correct.
# 
# Another example: `[ 1, 2, 4, 8, 16, 32, 64 ]`. Differences: `[ 2 - 1, 4 - 2, 8 - 4, 16 - 8, 32 - 16, 64 - 32 ] = [ 1, 2, 4, 8, 16, 32 ]`. The result is the original sequence itself. We can take the next element from the first array: `64`. Then the next element of original sequence is $ f_0( 8 ) = f_0( 7 ) + f_1( 7 ) = 64 + 64 = 128 $.
# 
# In many cases a next item of sequence can be predicted by calculating a next item of differences array. And if that array is recognizable then we are able to easily find the next item. It's possible to recognize an array by looking it up in a prefix tree (trie).
# 
# That trie would contain not sequences themselves but their signatures. We'll use a definition of signature function suggested by [Nina Chen](https://www.kaggle.com/ncchen) in [this article](https://www.kaggle.com/ncchen/integer-sequence-learning/match-test-set-to-training-set) with a slight change: $$ signature( seq ) = sign( seq ) \star \frac{ seq }{ GCD( seq ) } $$.
# 
# Here, `sign` is a sign of the first non-zero item of sequence `seq` (+1 if the item is positive, or -1 otherwise) and `GCD` is a greatest common divisor of all sequence elements.
# 
# Example: sequence `[ 2, 4, 6, 8 ]` would be stored in a trie as `[ 1, 2, 3, 4 ]` because `GCD( [ 2, 4, 6, 8 ] ) = 2` and we divide each element of sequence by GCD.
# 
# __A GCD function definition (taken from Nina Chen's article):__

# In[ ]:


import math

def findGCD(seq):
    gcd = seq[0]
    for i in range(1,len(seq)):
        gcd=math.gcd(gcd, seq[i])
    return gcd

print(findGCD([2,4,6,8]))


# __Signature function (slightly modified version of Nina Chen's function):__

# In[ ]:


def findSignature(seq):
    nonzero_seq = [d for d in seq if d!=0]
    if len(nonzero_seq)==0:
        return seq
    sign = 1 if nonzero_seq[0]>0 else -1
    gcd = findGCD(seq)
    return [sign*x//gcd for x in seq]

print(findSignature([0,2,4,6,8]))


# __A function to find differences array:__

# In[ ]:


def findDerivative(seq):
    return [0] if len(seq)<=1 else [seq[i]-seq[i-1] for i in range(1,len(seq))]

print(findDerivative([1,1,2,3,5,8,13,21]))


# __Trie class implementation:__

# In[ ]:


def addAll(seq, node, list):
    if 'value' in node:
        list.append( ( seq, node['value'] ) )
    for key in node:
        if key != 'value':
            addAll(seq + [key], node[key], list)

class prefixTree:
    def __init__(self):
        self.data={}
        self.puts=0
        self.nodes=0
    
    def put(self, seq, value):
        node=self.data
        nodeCreated=False
        for i in range(0,len(seq)):
            item=seq[i]
            if not item in node:
                node[item]={}
                if 'value' in node:
                    del node['value']
                self.nodes+=1
                nodeCreated=True
            node=node[item]
        if nodeCreated:
            node['value']=value
            self.puts+=1
        elif 'value' in node:
            node['value']=max(node['value'], value)
    
    def prefix(self, seq):
        list=[]
        node=self.data
        for i in range(0,len(seq)):
            item=seq[i]
            if item in node:
                node=node[item]
            else:
                return list
        addAll(seq, node, list)
        return list
    
    def hasPrefix(self, seq):
        node=self.data
        for i in range(0,len(seq)):
            item=seq[i]
            if item in node:
                node=node[item]
            else:
                return False
        return True

sampleTrie=prefixTree()
sampleTrie.put([1,2,3], 50)
sampleTrie.put([1,2,4,9], 30)
sampleTrie.put([2,3,4], 20)
print(sampleTrie.prefix([1,2]))


# Note that this is a showcase implementation. It's inefficient in memory consumption and performance terms. A better production version needs to be a C-wrapper of a string-based trie. I used a [datrie](https://pypi.python.org/pypi/datrie) library (the code is commented because Kaggle environment doesn't include this library):

# In[ ]:


"""
import datrie

class prefixTree:
    def __init__(self):
        self.data=datrie.Trie(',-0123456789')
        self.puts=0
        self.nodes=0
    
    def put(self, seq, value):
        key=','.join(map(str,seq))+','
        if key in self.data:
            self.data[key]=max(self.data[key],value)
        elif not self.data.has_keys_with_prefix(key):
            self.data[key]=value
            self.puts+=1
    
    def prefix(self, seq):
        ret=[]
        keys=self.data.keys(','.join(map(str,seq))+',')
        for k in keys:
            ret.append( ( list( map( int, k[:-1].split(',') ) ), self.data[ k ] ) )
        return ret
    
    def hasPrefix(self, seq):
        return self.data.has_keys_with_prefix(','.join(map(str,seq))+',')
"""
print()


# __Data loading:__

# In[ ]:


import pandas as pd

train_df= pd.read_csv('../input/train.csv', index_col="Id", nrows=100)
test_df = pd.read_csv('../input/test.csv', index_col="Id", nrows=100)

train_df= train_df['Sequence'].to_dict()
test_df= test_df['Sequence'].to_dict()
seqs={0: [1 for x in range(0,400)]}

for key in train_df:
    seq=train_df[key]
    seq=[int(x) for x in seq.split(',')]
    seqs[key]=seq

for key in test_df:
    seq=test_df[key]
    seq=[int(x) for x in seq.split(',')]
    seqs[key]=seq

for key in range(2, 5):
    print('ID = '+str(key)+':')
    print(seqs[key])
    print()


# We take both train and test arrays and combine them. Arrays are limited to 100 rows for a demonstration. Processing of the full files took 24 hours on my computer. I don't know the reason. Tried to rewrite the same script to Java using apache-commons-collections4 PatriciaTrie class and it takes several minutes to process.
# 
# Also, note that we add an extra sequence with ID = 0 consisting of 400 ones (`[ 1, 1, 1, ..., 1 ]`). Any polynomial sequence would be collapsed to constants array by taking differences recursively. The signature of constants array is `[ 1, 1, ..., 1 ]`. The sequence with ID = 0 would help us to deal with it. It would easily return 1 as a next element prediction.
# 
# __Prefixes preparation:__

# In[ ]:


import json

trie=prefixTree()
#Caching turned off.
#if not trie.load('trie'):
if True:
    for id in seqs:
        der=seqs[id]
        for derAttempts in range(4):
            seq=der
            firstInTrie=False
            for subseqAttempts in range(4-derAttempts):
                while len(seq)>0 and seq[0]==0:
                    seq=seq[1:]
                signature=findSignature(seq)
                if trie.hasPrefix( signature ):
                    if subseqAttempts==0:
                        firstInTrie=True
                    break
                trie.put( signature, len(seq)*100//len(der) )
                if len(seq)<=3:
                    break
                seq=seq[1:]
            if firstInTrie:
                break
            der=findDerivative(der)
    #trie.save('trie')

print(json.dumps(trie.prefix([2,3,6]),sort_keys=True,indent=2))


# This code adds a signature of each sequence to a trie. Also, it shifts a sequence (removing first items) and adds a signature of a shifted sequence too. Look at the Fibonacci sequence differences: `[ 1, 1, 2, 3, 5, 8, 13, 21 ] -> [ 1 - 1, 2 - 1, 3 - 2, 5 - 3, 8 - 5, 13 - 8, 21 - 13 ] = [ 0, 1, 1, 2, 3, 5, 8 ]`. We get the original sequence itself but shifted right by a new zero item.
# 
# So, we need to store the same sequences with additional shifts up to 4 elements to let searching for shifted sequences.
# 
# And we store differences arrays up to order 4.
# 
# Each stored sequence maps to a weight. This weight is calculated as a percent of length comparing to the original length of a sequence. When we try to find a next element for a sequence `[ 2, 3, 6 ]`, we get an array of two elements. Then we sort it by weight, so shifted sequences would have less priority than original sequences. An array `[ 2, 3, 6 ]` has two candidate sequences with the same 100 % weight. It means that such a short sequence is not enough to precisely predict a next element. We'll choose an element from a longest sequence.
# 
# Let's define a function that finds best candidate sequence and retrieves a next element from it:

# In[ ]:


from functools import reduce

def findNext(seq, trie):
    while True:
        nonZeroIndex=-1
        for i in range(0,len(seq)):
            if seq[i]!=0:
                nonZeroIndex=i
                break
        if nonZeroIndex<0:
            return 0
        signature=findSignature(seq)
        list=trie.prefix( signature )
        list=filter(lambda x: len(x[0])>len(signature), list)
        item=next(list, None)
        if item!=None:
            best=reduce(lambda a, b: a if a[1]>b[1] else b if b[1]>a[1] else a if len(b[0])<=len(a[0]) else b, list, item)
            nextElement=best[0][len(seq)]
            nextElement*=seq[nonZeroIndex]//signature[nonZeroIndex]
            return nextElement
        if len(seq)<=3:
            break
        seq=seq[1:]
    return None

print(findNext([2,3,6],trie))


# This function takes a sequence `seq` as an argument. We need to predict a next element of this sequence. At first step, we find all sequences which have `seq` in prefix. Then, the best one of them is chosen (sorting by weight then by length). If there are no sequences with `seq` prefix then we cut the first element and try to find a shifted sequence.
# 
# Here's a function that calls `findNext` for the original sequence. If a next element cannot be predicted then a differences array is calculated and prediction is performed for it. When a differences next element is found then it's used to recursively find a next element for the original sequence.

# In[ ]:


def findNextAndDerive(seq, trie):
    nextElement=findNext(seq, trie)
    if nextElement==None:
        der=findDerivative(seq)
        if len(der)<=3:
            return None
        nextElement=findNextAndDerive(der, trie)
        if nextElement==None:
            return None
        return seq[len(seq)-1]+nextElement
    return nextElement

print(findNextAndDerive([1,1,2,3,5,8,13],trie))


# An answer `22` is returned instead of expected `21`. Whoa!
# 
# The reason is the limited showcase dictionary. First rows do not contain Fibonacci sequence. If we manually add the sequence ourselves, the result would be correct. Note that we add a sequence ending at 13 without the item 21:

# In[ ]:


trie.put([1,1,2,3,5,8,13],100)
print(findNextAndDerive([1,1,2,3,5,8,13],trie))


# Now the answer is correct. Function tries these lookups:
# 
#     Original sequence and its shifts:
#     [ 1, 1, 2, 3, 5, 8, 13 ] -> not found
#     [ 1, 2, 3, 5, 8, 13 ] -> not found
#     [ 2, 3, 5, 8, 13 ] -> not found
#     [ 3, 5, 8, 13 ] -> not found
#     [ 5, 8, 13 ] -> not found
#     
#     First differences array and its shifts:
#     [ 0, 1, 1, 2, 3, 5 ] -> not found
#     [ 1, 1, 2, 3, 5 ] -> 2 candidates:
#         [ 1, 1, 2, 3, 5, 8, 13 ]: 100 %
#         [ 1, 1, 2, 3, 5, 9, 16, 28, 51, ... ]: 95 % - sequence ID = 59
# 
# And takes the first candidate `[ 1, 1, 2, 3, 5, 8, 13 ]` as a best fit. The next element of first differences array is 8. And the next element of the original sequence is 8 + 13 = 21.
# 
# The final script that outputs result to a file:

# In[ ]:


total=0
guessed=0
with open('prefix_lookup.csv', 'w+') as output:
    output.write('"Id","Last"\n')
    for id in test_df:
        der=seqs[id]
        nextElement=findNextAndDerive(der, trie)
        output.write(str(id))
        output.write(',')
        total+=1
        if nextElement==None:
            output.write('0')
        else:
            output.write(str(nextElement))
            guessed+=1
        output.write('\n')

print('Total %d' %total)
print('Guessed %d' %guessed)
print('Percent %d' %int(guessed*100//total))


# This script gives about 0.22 as a final result. It took 24 hours to prepare prefix tree and 1 hour to predict next elements for test sequences. The resulting file contains about 44 % of filled rows. So, about a half of the answers are correct. A Java port of this script is not finished yet but it seems to work for just several minutes.
