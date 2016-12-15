from cleaners import clean_data
import fasttext
import os
import shutil
import numpy as np
from helpers import linecount


def extract_word_vectors(vector_filename):
    # Extracts word vectors from fasttext 'model.vec', outputs two dictionaries searchable by words, vector
    # dictionary and index dictionary, respectively.
    with open(vector_filename, 'r') as vec_file:
        skipped = False
        vec_dict = {}
        index_dict = {}
        index = 0
        for line in vec_file:
            if not skipped:  # skipping the first line
                skipped = True
                continue
            word, vec_char = line.split(sep=' ', maxsplit=1)
            vec = np.fromstring(vec_char, dtype=float, sep=' ')
            vec_dict.__setitem__(word, vec)
            index_dict.__setitem__(word, index)
            index += 1
    return vec_dict, index_dict


def text2vec_fast(data_file, method='cbow', modelname='model', **kargs):
    clean_data(data_file, 'cleaned.txt')
    if method == 'skipgram':
        model = fasttext.skipgram('cleaned.txt', modelname, **kargs)
    else:
        model = fasttext.cbow('cleaned.txt', modelname, **kargs)

    vector_dict, index_dict = extract_word_vectors(modelname + '.vec')
    os.remove('cleaned.txt')
    return vector_dict, index_dict


def extract_features(infile_instring, K, isfile=False, **kargs):
    if not isfile:
        # Copy tweet to temporary text file
        tweet_file = open('tweets.txt', 'w')
        tweet_file.write(infile_instring)
        tweet_file.close()
        vector_dict, index_dict = text2vec_fast('tweets.txt', **kargs)
        N = 1  # Just one tweet
    else:
        shutil.copyfile(infile_instring, 'tweets.txt')
        N = linecount('tweets.txt')  # TODO: Have we cleaned this?
        vector_dict, index_dict = text2vec_fast('tweets.txt', **kargs)
        x_test = np.zeros((K, N))  # pre-allocating memory to large matrices to boost performance

    with open('tweets.txt') as file:
        for (index, tweet) in enumerate(file, start=0):
            word_count = 0
            sum_vec = np.zeros((K, 1))
            for word in tweet.split():
                try:
                    # word_vec = np.array(float(num) for num in vector_dict[word].split())
                    word_vec = np.array(vector_dict[word]).reshape(K, 1)
                    sum_vec += word_vec
                    word_count += 1
                except KeyError:  # no errors for words not in the dictionary (they were probably
                    # omitted by fasttext)
                    continue

            if word_count == 0:  # handle the exceptional case where tweet is empty.
                tweet_mean_vec = np.zeros((K, 1))
            else:
                tweet_mean_vec = sum_vec / word_count
            if not isfile:
                x_test = tweet_mean_vec.flatten()
            else:
                x_test[:, index] = tweet_mean_vec.flatten()

    os.remove('tweets.txt')
    return x_test
