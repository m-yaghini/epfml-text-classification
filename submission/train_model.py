from __future__ import print_function
import numpy as np

# Import useful elements to preprocess the input
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# Import models and layers from Keras to build the neural nets
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, MaxPooling1D
from keras.layers import Convolution1D, Flatten

# Import of our other functions
from helpers import load_data_and_labels

# Arbitrarly defined parameters deduced from experimentations
np.random.seed(1337)
max_features = 20000
maxlen = 1000
batch_size = 300
embedding_dims = 100
nb_filter = 100
filter_length = 4
hidden_dims = 128
nb_epoch = 1

# Loads the embeddings created with Fasttext
embeddings_index = {}
f = open('data/embeddings_model.vec')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loading data...')
x_train, labels, x_test = load_data_and_labels('data/train_pos_full.txt', 'data/train_neg_full.txt', 'data/test_data.txt')

# Tokenizes the words, creates and pads sequences
tokenizer = Tokenizer(nb_words=maxlen)
tokenizer.fit_on_texts(x_train)
sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)

word_index = tokenizer.word_index

print(len(x_train), 'train sequences')
print(len(labels), 'labels')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test = sequence.pad_sequences(sequences_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# Creates an embedding matrix based on the embeddings from Fasttext
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words+1, embedding_dims))
print(nb_words)
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embeddings file will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Build model...')

model = Sequential()
# Starts by creating an embedding layer for the input sequences
model.add(Embedding(nb_words + 1,
                    embedding_dims,
                    weights=[embedding_matrix],
                    input_length=maxlen,
                    dropout=0.2))

# Creates a Convolution layer with the relu activation function
model.add(Convolution1D(nb_filter, filter_length, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=1))
model.add(Flatten())
model.add(Dropout(0.25))

# Creates a Convolution layer with the relu activation function
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dropout(0.5))

# Adds a final layer of size 2 to have the predictions for our binary classification
model.add(Dense(2, activation='relu'))
model.add(Activation('softmax'))

# Compiles the model using the crossentropy binary function and the adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fits and saves the model to reuse it later to do the predictions
model.fit(X_train, labels, batch_size=batch_size, nb_epoch=nb_epoch)
model.save('model.h5')