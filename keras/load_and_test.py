from __future__ import print_function
import os
import csv
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import sys


BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove/'
TEXT_DATA_TEST_DIR = BASE_DIR + 'data/data_test'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100

MODEL_NAME = 'model_base_100.h5'

texts_test = []  # list of text samples
for name in sorted(os.listdir(TEXT_DATA_TEST_DIR)):
    path = os.path.join(TEXT_DATA_TEST_DIR, name)
    if os.path.isdir(path):
        print(path)
        for fname in sorted(os.listdir(path)):
             #if fname.isdigit():
             fpath = os.path.join(path, fname)
             if sys.version_info < (3,):
                f = open(fpath)
             else:
                f = open(fpath, encoding='latin-1')
             texts_test.append(f.read())
             f.close()

print('Found %s texts.' % len(texts_test))


# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_test)
sequences_test = tokenizer.texts_to_sequences(texts_test)


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)


model = load_model(MODEL_NAME)

probas = model.predict(data_test, batch_size=32)
print(probas[0:10])

predictions = []
no = 1
for tweet in probas:
    if tweet[0] >= tweet[1]:
        predictions.append([no, 1])
    else:
        predictions.append([no, -1])
    no += 1

print(predictions)

sum = 0
for pred in predictions:
    sum += pred[1]
print(sum)

with open('submission.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    sub_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    index = 0
    sub_writer.writeheader()
    for res in predictions:
        index += 1
        sub_writer.writerow({'Id': str(index), 'Prediction': str(res[1])})
print("Submission file created")