import os
import pickle

#with open('../data/train_data.pkl', 'rb') as train_data_picklefile:
#    x_train, y_train = pickle.load(train_data_picklefile)

#with open('data/full/train_pos_full.txt', 'r') as train_data:
#    x_train_pos = train_data.readlines()

#with open('data/full/train_neg_full.txt', 'r') as test_data:
#    x_test_neg = test_data.readlines()

'''
if not os.path.exists('data/data/pos/'):
    os.makedirs('data/data/pos/')
    counter = 0
    lines = []
    for line in x_train:
        counter += 1
        lines.append(line)
        if counter % 100 == 0:
            with open('data/data/pos/' + str(counter) + '.txt', 'w') as out:
                for li in lines:
                    out.write(li)
                lines = []
if not os.path.exists('data/data/neg/'):
    os.makedirs('data/data/neg/')
    counter = 0
    lines = []
    for line in x_train:
        counter += 1
        lines.append(line)
        if counter % 100 == 0:
            with open('data/data/neg/' + str(counter) + '.txt', 'w') as out:
                for li in lines:
                    out.write(li)
                lines = []

'''


if not os.path.exists('data/data/pos/'):
    os.makedirs('data/data/pos/')
    with open('data/data/train_pos_full.txt', 'r') as f:
        counter = 0
        lines = []
        for line in f.readlines():
            counter += 1
            lines.append(line)
            if counter % 100 == 0:
                with open('data/data/pos/' + str(counter) + '.txt', 'w') as out:
                    for li in lines:
                        out.write(li)
                    lines = []

if not os.path.exists('data/data/neg/'):
    os.makedirs('data/data/neg/')
    with open('data/data/train_neg_full.txt', 'r') as f:
        counter = 0
        lines = []
        for line in f.readlines():
            counter += 1
            lines.append(line)
            if counter % 100 == 0:
                with open('data/data/neg/' + str(counter) + '.txt', 'w') as out:
                    for li in lines:
                        out.write(li)
                    lines = []

if not os.path.exists('data/data_test/test/'):
    os.makedirs('data/data_test/test/')
    with open('data/data/test_data.txt', 'r') as f:
        counter = 0
        for line in f.readlines():
            counter += 1

            with open('data/data_test/test/' + str(counter) + '.txt', 'w') as out:
                 out.write(line)