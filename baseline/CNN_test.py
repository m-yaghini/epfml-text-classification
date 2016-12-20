from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D
from helpers import load_data_and_labels
from keras.preprocessing.text import Tokenizer

# set parameters:
# We can play with those parameters to get better results
max_features = 5000
maxlen = 400
batch_size = 1000
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 50
nb_epoch = 2


print('Loading data...')
x_train, labels, x_test = load_data_and_labels('data/train_pos.txt', 'data/train_neg.txt', 'data/test_data.txt')

# Tokenize the words and creating sequences and padding sequences
tokenizer = Tokenizer(nb_words=maxlen)
tokenizer.fit_on_texts(x_train)
sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)

print(len(x_train), 'train sequences')
print(len(labels), 'labels')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
# This basic model was taken from an online example, we can add
# much more layers and use the Standford embeddings to improve it

model = Sequential()
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=2))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, labels,
          batch_size=batch_size,
          nb_epoch=nb_epoch)

model.save('model_CNN_test.h5')
