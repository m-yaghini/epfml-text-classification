import shutil


def give_labels(in_file, out_file, label):
    with open(in_file, 'r') as infile:
        with open(out_file, 'w') as outfile:
            for line in infile:
                outfile.write('__label__' + label + ' ' + line)


def seperate(in_file, out_file):
    with open(in_file, 'r') as infile:
        with open(out_file, 'w') as outfile:
            for line in infile:
                out = line.split(',', maxsplit=1)[1]
                outfile.write(out)


def prepare_submission(in_file, out_file):
    with open(in_file, 'r') as infile:
        with open(out_file, 'w') as outfile:
            # outfile.write('Id,Prediction\n')
            # for (index, line) in enumerate(infile, start=1):
            #     if (line.find('__label__1')):
            #         out = str(index) + ',1\n'
            #     elif (line.find('__label__0')):
            #         out = str(index) + ',-1\n'
            #     outfile.write(out)
            for (index, line) in enumerate(infile, start=1):
                line.replace('__label__1',)


# give_labels("data/train_pos_full.txt", "data/train_pos_full_labeled.txt", '1')
# give_labels("data/train_neg_full.txt", "data/train_neg_full_labeled.txt", '0')

# seperate('data/test_data.txt', 'data/test_data_.txt')

prepare_submission('predict.txt', 'submission.csv')