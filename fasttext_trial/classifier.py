#classifier

import fasttext
from cleaners import *
from preprocess import *

INPUT_FILE_NAME = 'train_pos.txt'

# Cleaning the data set

# TODO: Removing duplicate lines
clean_duplicates('train_pos.txt', 'no_dup.txt')

# TODO: Remove <user> tags.
clean_user_tags('no_dup.txt', 'no_user.txt')

# TODO: Add _label_
add_label('no_user.txt', 'pos', 'labeled.txt')

# train classifier
classifier = fasttext.supervised('labeled.txt', 'supervised_model')


# test the classifier
texts = ['Im angry', "I'm happy"]
labels = classifier.predict(texts)
print(labels)
