#fasttext

import fasttext

model = fasttext.cbow('train_pos.txt', 'model')
print(model['like'])

