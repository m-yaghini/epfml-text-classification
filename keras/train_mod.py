from __future__ import print_function
import os
import numpy as np
import csv


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys

np.random.seed(1337)

#BASE_DIR = '/home/sylb/data/'
BASE_DIR = ''
GLOVE_DIR = 'glove/'
TEXT_DATA_TRAIN_DIR = BASE_DIR + 'data/data'
TEXT_DATA_TEST_DIR = BASE_DIR + 'data/data_test'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts_train = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_TRAIN_DIR)):
    path = os.path.join(TEXT_DATA_TRAIN_DIR, name)
    if os.path.isdir(path):
        print(path)
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
             fpath = os.path.join(path, fname)
             if sys.version_info < (3,):
                f = open(fpath)
             else:
                f = open(fpath, encoding='latin-1')
             texts_train.append(f.read())
             f.close()
             labels.append(label_id)

print('Found %s train texts.' % len(texts_train))

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

print('Found %s test texts.' % len(texts_test))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_train)
sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)



word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data train tensor:', data_train.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data_train.shape[0])
np.random.shuffle(indices)
data_train = data_train[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data_train.shape[0])


x_train = data_train[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data_train[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

'''
x = Flatten()(embedded_sequences)
x = Dense(60, init='normal', activation='relu')(x)
x = Dense(30, init='normal', activation='relu')(x)
#x = Dense(1, init='normal', activation='sigmoid')(x)
preds = Dense(len(labels_index), activation='softmax')(x)
'''

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=2, batch_size=128)

model.save('model_base_100.h5')

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