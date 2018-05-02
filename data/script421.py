
# coding: utf-8

# Public Leader-board of 0.89094
# ====================================================

# Save train and test images to normalized numpy arrays once for running multiple neural network configuration tests

# In[ ]:


from PIL import ImageFilter, ImageStat, Image, ImageDraw
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import glob
import cv2

np.random.seed(17)

def im_multi(path):
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': [0,0]}]

def im_stats(im_stats_df):
    im_stats_d = {}
    p = Pool(cpu_count())
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        im_stats_d[ret[i][0]] = ret[i][1]
    im_stats_df['size'] = im_stats_df['path'].map(lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))
    return im_stats_df

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (16, 16), cv2.INTER_LINEAR) #change to (64, 64)
    return [path, resized]

def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    fdata = fdata.transpose((0, 3, 1, 2))
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata

train = glob.glob('../input/train/**/*.jpg') + glob.glob('../input/additional/**/*.jpg')
train = pd.DataFrame([[p.split('/')[3],p.split('/')[4],p] for p in train], columns = ['type','image','path'])

#new for stage2
test_stg1_labels = pd.read_csv('../input/solution_stg1_release.csv')
test_stg1_labels['path'] = test_stg1_labels['image_name'].map(lambda x: '../input/test/' + x)
test_stg1_labels['image'] = test_stg1_labels['image_name']
test_stg1_labels['type'] = test_stg1_labels.apply(lambda r: 'Type_1' if r['Type_1'] == 1 else '', axis=1)
test_stg1_labels['type'] = test_stg1_labels.apply(lambda r: 'Type_2' if r['Type_2'] == 1 else r['type'], axis=1)
test_stg1_labels['type'] = test_stg1_labels.apply(lambda r: 'Type_3' if r['Type_3'] == 1 else r['type'], axis=1)
test_stg1_labels = test_stg1_labels[['type','image','path']]
print(len(train), len(test_stg1_labels))
train = pd.concat((train, test_stg1_labels), axis=0, ignore_index=True)

#new for stage2
test = glob.glob('../input/test_stg2/*.jpg')
test = pd.DataFrame([[p.split('/')[3],p] for p in test], columns = ['image','path'])

train = im_stats(train)
train = train[train['size'] != '0 0'].reset_index(drop=True) #remove bad images
train_data = normalize_image_features(train['path'])
np.save('train.npy', train_data, allow_pickle=True, fix_imports=True)

le = LabelEncoder()
train_target = le.fit_transform(train['type'].values)
print(le.classes_)
np.save('train_target.npy', train_target, allow_pickle=True, fix_imports=True)

test_data = normalize_image_features(test['path'])
np.save('test.npy', test_data, allow_pickle=True, fix_imports=True)

test_id = test.image.values
np.save('test_id.npy', test_id, allow_pickle=True, fix_imports=True)

train_data = np.load('train.npy')
train_target = np.load('train_target.npy')

x_train,x_val_train,y_train,y_val_train = train_test_split(train_data,train_target,test_size=0.4, random_state=17)


# Start your neural network high performance engines
# 
#  - I'll admit as you've no doubt noticed by now that I still don't know what I am doing with Neural Networks 'yet'.

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')
K.set_floatx('float32')

def create_model(opt_='adamax'):
    model = Sequential()
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th', input_shape=(3, 16, 16))) #change to (3, 64, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model

datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)
datagen.fit(train_data)

model = create_model()
model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True), nb_epoch=100, samples_per_epoch=len(x_train), verbose=1, validation_data=(x_val_train, y_val_train))

test_data = np.load('test.npy')
test_id = np.load('test_id.npy')

pred = model.predict_proba(test_data)
df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
df['image_name'] = test_id

#new
test_stg1_labels = pd.read_csv('../input/solution_stg1_release.csv')
df = pd.concat((df, test_stg1_labels), axis=0, ignore_index=True)
df.to_csv('submission.csv', index=False)

