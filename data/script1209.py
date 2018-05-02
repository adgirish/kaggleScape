
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# # Load Datasets
# Since we will be using a generator we don't need to actually load in any files into memory, all we need is the filepaths :)

# In[2]:


path = "../input/train/"

def load_train(path):
    train_set = pd.read_csv('../input/train_labels.csv')
    train_label = np.array(train_set['invasive'].iloc[: ])
    train_files = []
    for i in range(len(train_set)):
        train_files.append(path + str(int(train_set.iloc[i][0])) +'.jpg')
    train_set['name'] = train_files
    return train_files, train_set, train_label

train_files, train_set, train_label = load_train(path)

train_set.head()


# In[4]:


path = "../input/test/"

def load_test(path):
    test_set = pd.read_csv('../input/sample_submission.csv')
    test_files = []
    for i in range(len(test_set)):
        test_files.append(path + str(int(test_set.iloc[i][0])) +'.jpg')
    return test_files, test_set

test_files, test_set = load_test(path)

test_set.head()


# # Define CNN Model Architecture
# *Kaggle can't access the weights file*

# In[ ]:


img_height = 800
img_width = 800
img_channels = 3
img_dim = (img_height, img_width, img_channels)

def inceptionv3(img_dim=img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = InceptionV3(include_top=False,
                   weights='imagenet',
                   input_shape=img_dim)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model

model = inceptionv3()
model.summary()


# # Train Model
# Here we use 5-fold cross-validation to train the model. Submission file is saved with the average of all folds. Additionally, prediction arrays are saved for each fold in case we want to hand-pick results from an individual fold.

# In[5]:


def train_model(model, batch_size, epochs, img_size, x, y, test, n_fold, kf):
    roc_auc = metrics.roc_auc_score
    preds_train = np.zeros(len(x), dtype = np.float)
    preds_test = np.zeros(len(test), dtype = np.float)
    train_scores = []; valid_scores = []

    i = 1

    for train_index, test_index in kf.split(x):
        x_train = x.iloc[train_index]; x_valid = x.iloc[test_index]
        y_train = y[train_index]; y_valid = y[test_index]

        def augment(src, choice):
            if choice == 0:
                # Rotate 90
                src = np.rot90(src, 1)
            if choice == 1:
                # flip vertically
                src = np.flipud(src)
            if choice == 2:
                # Rotate 180
                src = np.rot90(src, 2)
            if choice == 3:
                # flip horizontally
                src = np.fliplr(src)
            if choice == 4:
                # Rotate 90 counter-clockwise
                src = np.rot90(src, 3)
            if choice == 5:
                # Rotate 180 and flip horizontally
                src = np.rot90(src, 2)
                src = np.fliplr(src)
            return src

        def train_generator():
            while True:
                for start in range(0, len(x_train), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(x_train))
                    train_batch = x_train[start:end]
                    for filepath, tag in train_batch.values:
                        img = cv2.imread(filepath)
                        img = cv2.resize(img, img_size)
                        img = augment(img, np.random.randint(6))
                        x_batch.append(img)
                        y_batch.append(tag)
                    x_batch = np.array(x_batch, np.float32) / 255.
                    y_batch = np.array(y_batch, np.uint8)
                    yield x_batch, y_batch

        def valid_generator():
            while True:
                for start in range(0, len(x_valid), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(x_valid))
                    valid_batch = x_valid[start:end]
                    for filepath, tag in valid_batch.values:
                        img = cv2.imread(filepath)
                        img = cv2.resize(img, img_size)
                        img = augment(img, np.random.randint(6))
                        x_batch.append(img)
                        y_batch.append(tag)
                    x_batch = np.array(x_batch, np.float32) / 255.
                    y_batch = np.array(y_batch, np.uint8)
                    yield x_batch, y_batch

        def test_generator():
            while True:
                for start in range(0, len(test), batch_size):
                    x_batch = []
                    end = min(start + batch_size, len(test))
                    test_batch = test[start:end]
                    for filepath in test_batch:
                        img = cv2.imread(filepath)
                        img = cv2.resize(img, img_size)
                        x_batch.append(img)
                    x_batch = np.array(x_batch, np.float32) / 255.
                    yield x_batch

        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=1, 
                               verbose=1, min_lr=1e-7),
             ModelCheckpoint(filepath='inception.fold_' + str(i) + '.hdf5', verbose=1,
                             save_best_only=True, save_weights_only=True, mode='auto')]

        train_steps = len(x_train) / batch_size
        valid_steps = len(x_valid) / batch_size
        test_steps = len(test) / batch_size
        
        model = model

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', 
                      metrics = ['accuracy'])

        model.fit_generator(train_generator(), train_steps, epochs=epochs, verbose=1, 
                            callbacks=callbacks, validation_data=valid_generator(), 
                            validation_steps=valid_steps)

        model.load_weights(filepath='inception.fold_' + str(i) + '.hdf5')

        print('Running validation predictions on fold {}'.format(i))
        preds_valid = model.predict_generator(generator=valid_generator(),
                                      steps=valid_steps, verbose=1)[:, 0]

        print('Running train predictions on fold {}'.format(i))
        preds_train = model.predict_generator(generator=train_generator(),
                                      steps=train_steps, verbose=1)[:, 0]

        valid_score = roc_auc(y_valid, preds_valid)
        train_score = roc_auc(y_train, preds_train)
        print('Val Score:{} for fold {}'.format(valid_score, i))
        print('Train Score: {} for fold {}'.format(train_score, i))

        valid_scores.append(valid_score)
        train_scores.append(train_score)
        print('Avg Train Score:{0:0.5f}, Val Score:{1:0.5f} after {2:0.5f} folds'.format
              (np.mean(train_scores), np.mean(valid_scores), i))

        print('Running test predictions with fold {}'.format(i))

        preds_test_fold = model.predict_generator(generator=test_generator(),
                                              steps=test_steps, verbose=1)[:, -1]

        preds_test += preds_test_fold

        print('\n\n')

        i += 1

        if i <= n_fold:
            print('Now beginning training for fold {}\n\n'.format(i))
        else:
            print('Finished training!')

    preds_test /= n_fold


    return preds_test


# In[ ]:


batch_size = 5
epochs = 50
n_fold = 5
img_size = (img_height, img_width)
kf = KFold(n_splits=n_fold, shuffle=True)

test_pred = train_model(model, batch_size, epochs, img_size, train_set, 
                        train_label, test_files, n_fold, kf)

test_set['invasive'] = test_pred
test_set.to_csv('./submission.csv', index = None)

