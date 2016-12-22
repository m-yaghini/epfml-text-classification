import fasttext


def text2vec_fast(data_file, method='cbow', modelname='model', **kargs):
    if method == 'skipgram':
        model = fasttext.skipgram('cleaned.txt', modelname, **kargs)
    else:
        model = fasttext.cbow('cleaned.txt', modelname, **kargs)

    vector_dict, index_dict = extract_word_vectors(modelname + '.vec')
    os.remove('cleaned.txt')
    return vector_dict, index_dict

