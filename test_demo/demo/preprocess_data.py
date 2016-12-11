from scipy.sparse import *
import numpy as np
import pickle
import random
import subprocess
from cleaners import clean_files
from helpers import *

print("Starting ...")
clean_files()
print("Files cleaned")

print("Creating the vocab files")
subprocess.call(['../scripts/create_vocab.sh'])
pickle_vocab('train')
pickle_vocab('test')
print("Vocab files created")

print("Creating co-occurrences matrices")
print("This operation can take some time")
cooc_train()
cooc_test()
print("Co-occurrences matrices created")

print("Creating the embeddings")
glove_embeddings('train')
glove_embeddings('test')
print("Embeddings created")

