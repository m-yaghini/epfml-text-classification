# fasttext

import fasttext

model = fasttext.load_model('model.bin')
print(model['good'])