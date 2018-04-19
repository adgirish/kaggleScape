
# coding: utf-8

# In[ ]:


# make sure you downloaded the files correctly
import hashlib
import os.path as path

def sha256(fname):
    hash_sha256 = hashlib.sha256()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

filenames = ['category_names.csv', 'sample_submission.csv', 'train_example.bson', 
             'test.bson', 'train.bson']
hashes = ['84fe1e7334836d50ed04d475cfc525bccbe420f7242f85ca301b3f69294632c6',
          'a4ea875b408504bb9e981a7462a41f7d03cc0f68eecc8b222ecf0afc8e43e688',
          '5d54291c3704a755178d9c1cd8f877eaa6adbf207713988ca2bb5cd52aab7bdb',
          '844d3e13fa785498c2b153bc0edc942d14bbc95b92f30c827487ef096fd28a53',
          '2b9ac4157e67fc96ab85ca99679b3b25cada589c4da6bb128fa006085b4cc42b']
data_root = path.join('..', 'input')  # make sure you set up this path correctly

# this may take a few minutes
for filename, hash_ in zip(filenames, hashes):
    computed_hash = sha256(path.join(data_root, filename))
    if computed_hash == hash_:
        print('{}: OK'.format(filename))
    else:
        print('{}: fail'.format(filename))
        print('expected: {}'.format(hash_))
        print('computed: {}'.format(computed_hash))

