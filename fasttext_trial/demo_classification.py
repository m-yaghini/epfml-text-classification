from fasttext_util import text2vec_fast, extract_features
from cleaners import clean_data
from helpers import linecount

import pickle
import os.path
import shutil
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC

DATA_FOLDER = 'data/'
FROM_SCRATCH = False
ALREADY_TRAINED = True

K = 100  # number of vector features

# # Concatenating positive and negative files
# with open(DATA_FOLDER + 'train.txt', 'wb') as wfd:
#     for f in [DATA_FOLDER + 'train_pos.txt', DATA_FOLDER + 'train_neg.txt']:
#         with open(f, 'rb') as fd:
#             shutil.copyfileobj(fd, wfd, 1024 * 1024 * 10)
#             # 10MB per writing chunk to avoid reading big file into memory.

# Extracting vocabulary and word vectors with fasttext
if not os.path.isfile('vectors_indices.pkl') or FROM_SCRATCH:
    with open('vectors_indices.pkl', 'wb') as vec_indice_file:
        vector_dict, index_dict = text2vec_fast(DATA_FOLDER + 'train.txt')
        pickle.dump((vector_dict, index_dict), vec_indice_file)  # serializing the vectorization output for future use
else:
    with open('vectors_indices.pkl', 'rb') as vec_indice_file:
        vector_dict, index_dict = pickle.load(vec_indice_file)

if not ALREADY_TRAINED:

    # Cleaning tweets and finding the total number of 'sample tweets'
    N = 0  # total number of 'cleaned' tweets
    for file in ['train_pos.txt', 'train_neg.txt']:
        cleanedfile = open(DATA_FOLDER + 'cleaned' + file, 'wb')
        clean_data(DATA_FOLDER + file, DATA_FOLDER + 'cleaned' + file)
        cleanedfile.close()
        N += linecount(DATA_FOLDER + 'cleaned' + file)

    print(K, N)
    x_train = np.zeros((K, N))  # pre-allocating memory to large matrices to boost performance
    y_train = np.zeros((N, 1))

    for (file_no, file) in enumerate(['train_pos.txt', 'train_neg.txt']):
        with open(DATA_FOLDER + 'cleaned' + file, 'r') as file:
            for (index, tweet) in enumerate(file, start=0):
                word_count = 0
                sum_vec = np.zeros((K, 1))
                for word in tweet.split():
                    try:
                        # word_vec = np.array(float(num) for num in vector_dict[word].split())
                        word_vec = np.array(vector_dict[word]).reshape(K, 1)
                        # print(word_vec.shape, sum_vec.shape)
                        sum_vec += word_vec
                        word_count += 1
                    except KeyError:  # no errors for words not in the dictionary (they were probably
                        # omitted by fasttext)
                        continue

                if word_count == 0:  # handle the exceptional case where tweet is empty.
                    tweet_mean_vec = np.zeros((K, 1))
                else:
                    tweet_mean_vec = sum_vec / word_count

                if file_no == 0:  # assign positive labels
                    tweet_label = 1
                elif file_no == 1:  # assign negative labels
                    tweet_label = -1
                else:
                    raise "Out Of Range File Error"

                x_train[:, index] = tweet_mean_vec.flatten()
                y_train[index] = tweet_label

    with open('train_data.pkl', 'wb') as train_data_picklefile:
        pickle.dump((x_train, y_train), train_data_picklefile)

else:
    with open('train_data.pkl', 'rb') as train_data_picklefile:
        x_train, y_train = pickle.load(train_data_picklefile)

# Classification
rfc = RFC()
print(extract_features('I am happy', K))
# print(vector_dict['good'])
# print(x_train.shape, y_train.shape)
# rfc.fit(np.transpose(x_train), y_train.flatten())
# print(rfc.predict(np.array(vector_dict['good']).reshape(1, K)))
