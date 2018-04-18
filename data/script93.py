
# coding: utf-8

# # This is a TF Estimator end-to-end baseline solution
# 
# **For local run**
# 
# Tested with
# 
# ```
# numpy==1.13.3
# scipy==0.19.1
# tensorflow-gpu==1.4.0
# tqdm
# ```
# 
# 
# I want to show usage of Estimators with custom python datagenerators.
# 
# 
# Detailed documentation you can find at https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
# 
# I also recommend to read source code  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/estimator/estimator.py

# Suppose we have following project structure:
# ```
# .
# ├── data
# │   ├── test            # extracted
# │   │   └── audio          # all test
# │   ├── test.7z         # downloaded
# │   ├── train           # extracted
# │   │   ├── audio          # folder with all train command/file.wav
# │   │   ├── LICENSE
# │   │   ├── README.md
# │   │   ├── testing_list.txt
# │   │   └── validation_list.txt
# │   └── train.7z         # downloaded
# ├── kernel.ipynb      # this ipynb  
# └── model-k           # folder for model, checkpoints, logs and submission.csv
# ```

# In[ ]:


DATADIR = './data' # unzipped train and test data
OUTDIR = './model-k' # just a random name
# Data Loading
import os
import re
from glob import glob


POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}


def load_data(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val

trainset, valset = load_data(DATADIR)


# Let me introduce pythonic datagenerator.
# It is just a python/numpy/... function **without tf** that yields dicts such that
# ```
# {
#   'x': np.array(...),
#   'str_key': np.string_(...),
#   'label': np.int32(...),
# }
# ```
# 
# Be sure, every value in this dict has `.dtype` method.

# In[ ]:


import numpy as np
from scipy.io import wavfile

def data_generator(data, params, mode='train'):
    def generator():
        if mode == 'train':
            np.random.shuffle(data)
        # Feel free to add any augmentation
        for (label_id, uid, fname) in data:
            try:
                _, wav = wavfile.read(fname)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                L = 16000  # be aware, some files are shorter than 1 sec!
                if len(wav) < L:
                    continue
                # let's generate more silence!
                samples_per_file = 1 if label_id != name2id['silence'] else 20
                for _ in range(samples_per_file):
                    if len(wav) > L:
                        beg = np.random.randint(0, len(wav) - L)
                    else:
                        beg = 0
                    yield dict(
                        target=np.int32(label_id),
                        wav=wav[beg: beg + L],
                    )
            except Exception as err:
                print(err, label_id, uid, fname)

    return generator


# 
# Suppose, we have spectrograms and want to write feature extractor that produces logits.
# 
# 
# Let's write some simple net, treat sound as a picture.
# 
# 
# **Spectrograms** (input x) have shape `(batch_size, time_frames, freq_bins, 2)`.
# 
# **Logits** is a tensor with shape `(batch_size, num_classes)`.

# In[ ]:


import tensorflow as tf
from tensorflow.contrib import layers

def baseline(x, params, is_training):
    x = layers.batch_norm(x, is_training=is_training)
    for i in range(4):
        x = layers.conv2d(
            x, 16 * (2 ** i), 3, 1,
            activation_fn=tf.nn.elu,
            normalizer_fn=layers.batch_norm if params.use_batch_norm else None,
            normalizer_params={'is_training': is_training}
        )
        x = layers.max_pool2d(x, 2, 2)

    # just take two kind of pooling and then mix them, why not :)
    mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)
    apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    x = 0.5 * (mpool + apool)
    # we can use conv2d 1x1 instead of dense
    x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)
    x = tf.nn.dropout(x, keep_prob=params.keep_prob if is_training else 1.0)
    
    # again conv2d 1x1 instead of dense layer
    logits = layers.conv2d(x, params.num_classes, 1, 1, activation_fn=None)
    return tf.squeeze(logits, [1, 2])


# We need to write a model handler for three regimes:
# - train
# - eval
# - predict
# 
# Loss function, train_op, additional metrics and summaries should be defined.
# 
# Also, we need to convert sound waveform into spectrograms (we could do it with numpy/scipy/librosa in data generator, but TF has new signal processing API)

# In[ ]:


from tensorflow.contrib import signal

# features is a dict with keys: tensors from our datagenerator
# labels also were in features, but excluded in generator_input_fn by target_key

def model_handler(features, labels, mode, params, config):
    # Im really like to use make_template instead of variable_scopes and re-usage
    extractor = tf.make_template(
        'extractor', baseline,
        create_scope_now_=True,
    )
    # wav is a waveform signal with shape (16000, )
    wav = features['wav']
    # we want to compute spectograms by means of short time fourier transform:
    specgram = signal.stft(
        wav,
        400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
        160,  # 16000 * 0.010 -- default stride
    )
    # specgram is a complex tensor, so split it into abs and phase parts:
    phase = tf.angle(specgram) / np.pi
    # log(1 + abs) is a default transformation for energy units
    amp = tf.log1p(tf.abs(specgram))
    
    x = tf.stack([amp, phase], axis=3) # shape is [bs, time, freq_bins, 2]
    x = tf.to_float(x)  # we want to have float32, not float64

    logits = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        # some lr tuner, you could use move interesting functions
        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
            learning_rate_decay_fn=learning_rate_decay_fn,
            clip_gradients=params.clip_gradients,
            variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        specs = dict(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        prediction = tf.argmax(logits, axis=-1)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(
            labels, prediction, params.num_classes)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        specs = dict(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                acc=(acc, acc_op),
            )
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'label': tf.argmax(logits, axis=-1),  # for probability just take tf.nn.softmax()
            'sample': features['sample'], # it's a hack for simplicity
        }
        specs = dict(
            mode=mode,
            predictions=predictions,
        )
    return tf.estimator.EstimatorSpec(**specs)


def create_model(config=None, hparams=None):
    return tf.estimator.Estimator(
        model_fn=model_handler,
        config=config,
        params=hparams,
    )


# Define some params. Move model hyperparams (optimizer, extractor, num of layers, activation fn, ...) here

# In[ ]:


params=dict(
    seed=2018,
    batch_size=64,
    keep_prob=0.5,
    learning_rate=1e-3,
    clip_gradients=15.0,
    use_batch_norm=True,
    num_classes=len(POSSIBLE_LABELS),
)

hparams = tf.contrib.training.HParams(**params)
os.makedirs(os.path.join(OUTDIR, 'eval'), exist_ok=True)
model_dir = OUTDIR

run_config = tf.contrib.learn.RunConfig(model_dir=model_dir)


# **Let's run training!**

# In[ ]:


# it's a magic function :)
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn
            
train_input_fn = generator_input_fn(
    x=data_generator(trainset, hparams, 'train'),
    target_key='target',  # you could leave target_key in features, so labels in model_handler will be empty
    batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
    queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
)

val_input_fn = generator_input_fn(
    x=data_generator(valset, hparams, 'val'),
    target_key='target',
    batch_size=hparams.batch_size, shuffle=True, num_epochs=None,
    queue_capacity=3 * hparams.batch_size + 10, num_threads=1,
)
            

def _create_my_experiment(run_config, hparams):
    exp = tf.contrib.learn.Experiment(
        estimator=create_model(config=run_config, hparams=hparams),
        train_input_fn=train_input_fn,
        eval_input_fn=val_input_fn,
        train_steps=10000, # just randomly selected params
        eval_steps=200,  # read source code for steps-epochs ariphmetics
        train_steps_per_iteration=1000,
    )
    return exp

tf.contrib.learn.learn_runner.run(
    experiment_fn=_create_my_experiment,
    run_config=run_config,
    schedule="continuous_train_and_eval",
    hparams=hparams)


# 
# While it trains (~10-20min on i5 + 1080), you could start tensorboard on model_dir and see live chart like this
# 
# ![Tensorboard](https://pp.userapi.com/c841329/v841329524/3db60/fdNDyRMJHMQ.jpg)
# 
# 
# Now we want to predict testset and make submission file.
# 
# 1. Create datagenerator and input_function
# 2. Load model
# 3. Iterate over predictions and store results

# In[ ]:


from tqdm import tqdm
# now we want to predict!
paths = glob(os.path.join(DATADIR, 'test/audio/*wav'))

def test_data_generator(data):
    def generator():
        for path in data:
            _, wav = wavfile.read(path)
            wav = wav.astype(np.float32) / np.iinfo(np.int16).max
            fname = os.path.basename(path)
            yield dict(
                sample=np.string_(fname),
                wav=wav,
            )

    return generator

test_input_fn = generator_input_fn(
    x=test_data_generator(paths),
    batch_size=hparams.batch_size, 
    shuffle=False, 
    num_epochs=1,
    queue_capacity= 10 * hparams.batch_size, 
    num_threads=1,
)

model = create_model(config=run_config, hparams=hparams)
it = model.predict(input_fn=test_input_fn)


# last batch will contain padding, so remove duplicates
submission = dict()
for t in tqdm(it):
    fname, label = t['sample'].decode(), id2name[t['label']]
    submission[fname] = label

with open(os.path.join(model_dir, 'submission.csv'), 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))


# ## About tf.Estimators
# 
# **Pros**:
# - no need to control Session
# - datagenerator feeds model via queues without explicit queue coding :)
# - you could naturaly export models into production
#     
# **Cons**:
# - it's very hard to debug computational graph (use `tf.add_check_numerics()` and `tf.Print` in case of problems)
# - boilerplate code
# - need to read source code for making interesting things
# 
# 
# **Conclusion**:
# Estimator is a nice abstraction with some boilerplate code :)
# 
# 
# ## About Speech Recognition Challenge:
# 
# You could start from this end-to-end ipynb, improving several functions for much better results.
# 
# 
# 
# May the gradient flow be with you. 
