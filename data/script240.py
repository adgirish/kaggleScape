
# coding: utf-8

# <h1>This notebook shows how I reduce the size of the properties dataset by selecting smaller datatypes. </h1> <p> I noticed the size of the properties dataset is pretty big for a lower/mid-range laptop so I made a script to make the dataset smaller without losing information. </p>
# <h3>This notebook uses the following approach: </h3>
# <ol>
# <li>Iterate over every column </li>
# <li>Determine if the column is numeric </li>
# <li>Determine if the column can be represented by an integer </li>
# <li>Find the min and the max value </li>
# <li>Determine and apply the smallest datatype that can fit the range of values </li>
# </ol>
# 
# <h3> This reduces the dataset from approx. 1.3 GB to 466 MB </h3>

# <h2> 1 | load packages </h2>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# <h2> 2 | Function for reducing memory usage of a pandas dataframe </h2>

# In[ ]:


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist


# <h2> 3 | Load Data </h2>

# In[ ]:


props = pd.read_csv(r"../input/properties_2016.csv")  #The properties dataset

#train = pd.read_csv(r"../input/train_2016_v2.csv")   # The parcelid's with their outcomes
#samp = pd.read_csv(r"../input/sample_submission.csv")  #The parcelid's for the testset


# <h2> 4 | Run function </h2>

# In[ ]:


props, NAlist = reduce_mem_usage(props)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)

