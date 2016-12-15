from scipy.sparse import *
import numpy as np
import pickle
import random
import os
import subprocess
from shutil import copyfile
from cleaners import *
from helpers import *

def prepare_data_files(run_path, ratio):

    print("Preparing the files ...")
    if not os.path.exists('../data/cleaned/train_pos_full.txt') \
    or not os.path.exists('../data/cleaned/train_neg_full.txt'):
        print('Cleaned files do not exist')
        print('Cleaning the files')
        clean_files_big()
    if not os.path.exists('../data/cleaned/test_data.txt'):
        print('Cleaned files do not exist')
        print('Cleaning the files')
        clean_files_test()
    print("Files cleaned")

    training_outputs_path = run_path + 'outputs/train/'
    testing_outputs_path = run_path + 'outputs/test/'
    if not os.path.exists(run_path):
        os.makedirs(run_path)
        os.makedirs(run_path + 'data/')
        os.makedirs(training_outputs_path)
        os.makedirs(testing_outputs_path)

    print('Creating a training set files')
    split_data('../data/cleaned/train_pos_full.txt', run_path + 'data/train_pos.txt', ratio)
    split_data('../data/cleaned/train_neg_full.txt', run_path + 'data/train_neg.txt', ratio)
    print('Training sets created')
    copyfile('../data/cleaned/test_data.txt', run_path + 'data/test_data.txt')


    print("Creating the vocab files")
    subprocess.call(['../scripts/create_vocab.sh', str(run_path)])

    pickle_vocab(training_outputs_path)
    pickle_vocab(testing_outputs_path)
    print("Vocab files created")

    print("Creating co-occurrences matrices")
    print("This operation can take some time")
    cooc_matrix(run_path, 'train')
    cooc_matrix(run_path, 'test')
    print("Co-occurrences matrices created")


    print("Creating the embeddings")
    print("This operation can take some time")
    glove_embeddings(training_outputs_path, embedding_dim=100)
    glove_embeddings(testing_outputs_path, embedding_dim=100)
    print("Embeddings created")

def prepare_data_sets(run_path):

    training_path = run_path + 'outputs/train/'
    testing_path = run_path + 'outputs/test/'
    with open(training_path + 'vocab.pkl', 'rb') as f:
        vocab_dict = pickle.load(f)

    with open(testing_path + 'vocab.pkl', 'rb') as f:
        vocab_dict_test = pickle.load(f)

    W = np.load(training_path + 'embeddings.npy')
    W_test = np.load(testing_path + 'embeddings.npy')

    train, labels, test = load_data_and_labels(run_path + 'data/train_pos.txt', run_path + '/data/train_neg.txt', run_path + 'data/test_data.txt')

    ls = []
    for sent in train:
        ls_temp = []
        for word in list(sent.split()):
            try:
                ls_temp.append(vocab_dict[word])
            except:
                ls_temp.append(0)
        ls.append(ls_temp)

    ls_sum = []
    for ls_in in ls:
        sum_vect = 0
        for index in ls_in:
            sum_vect += W[index]
        ls_sum.append(sum_vect)
    X_train = ls_sum

    ls = []
    for sent in test:
        ls_temp = []
        for word in list(sent.split()):
            try:
                ls_temp.append(vocab_dict_test[word])
            except:
                ls_temp.append(0)
        ls.append(ls_temp)

    ls_sum = []
    for ls_in in ls:
        sum_vect = 0
        for index in ls_in:
            sum_vect += W_test[index]
        ls_sum.append(sum_vect)
    X_test = ls_sum
    print("Data loaded")
    return X_train, labels, X_test