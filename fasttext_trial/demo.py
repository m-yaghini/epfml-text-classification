from fasttext_util import text2vec_fast
import shutil

DATA_FOLDER = 'data/'

# Concatenating positive and negative files
with open(DATA_FOLDER + 'train.txt', 'wb') as wfd:
    for f in [DATA_FOLDER + 'train_pos.txt', DATA_FOLDER + 'train_neg.txt']:
        with open(DATA_FOLDER + 'train_pos.txt','rb') as fd:
            shutil.copyfileobj(fd, wfd, 1024*1024*10)



# Extracting vocabulary and word vectors with fasttext
vector_dict, index_dict = text2vec_fast(DATA_FOLDER + 'train.txt')
