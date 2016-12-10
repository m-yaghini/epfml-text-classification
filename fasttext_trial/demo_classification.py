from fasttext_util import text2vec_fast
from cleaners import clean_data
import json
import pickle
import os.path
import shutil
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC

DATA_FOLDER = 'data/'
FROM_SCRATCH = False
ALREADY_TRAINED = True

K = 100  # number of vector features

# Concatenating positive and negative files
with open(DATA_FOLDER + 'train.txt', 'wb') as wfd:
    for f in [DATA_FOLDER + 'train_pos.txt', DATA_FOLDER + 'train_neg.txt']:
        with open(f, 'rb') as fd:
            shutil.copyfileobj(fd, wfd, 1024 * 1024 * 10)
            # 10MB per writing chunk to avoid reading big file into memory.

# Extracting vocabulary and word vectors with fasttext
if not os.path.isfile('vectors_indices.json') or FROM_SCRATCH:
    with open('vectors_indices.json', 'w') as jsonfile:
        vector_dict, index_dict = text2vec_fast(DATA_FOLDER + 'train.txt')
        json.dump((vector_dict, index_dict), jsonfile)  # serializing the vectorization output for future use
else:
    with open('vectors_indices.json', 'r') as jsonfile:
        vector_dict, index_dict = json.load(jsonfile)

if not ALREADY_TRAINED:
    # Labeling training data
    x_train = []
    y_train = []
    for file in ['train_pos.txt', 'train_neg.txt']:
        cleanedfile = open(DATA_FOLDER + 'cleaned' + file, 'wb')
        clean_data(DATA_FOLDER + file, DATA_FOLDER + 'cleaned' + file)
        cleanedfile.close()
        with open(DATA_FOLDER + 'cleaned' + file, 'r') as file:
            for tweet in file:
                word_count = 0
                tweet_mean_vec = []
                sum_vec = np.zeros((K, 1))
                for word in tweet.split():
                    try:
                        word_vec = np.array([float(num) for num in vector_dict[word].split()]).reshape(K, 1)
                        sum_vec += word_vec
                        word_count += 1
                        tweet_mean_vec = sum_vec / word_count
                    except KeyError:
                        continue
                if file == 'train_pos.txt':
                    tweet_label = 1
                else:
                    tweet_label = -1

                x_train.append(tweet_mean_vec)
                y_train.append(tweet_label)

    with open('trained_data.pkl', 'wb') as trained_picklefile:
        pickle.dump((x_train, y_train), trained_picklefile)

else:
    with open('trained_data.pkl', 'rb') as trained_picklefile:
        x_train, y_train = pickle.load(trained_picklefile)

# Classification
rfc = RFC()
rfc.fit(x_train, y_train)
print(RFC.predict(vector_dict['good']))
