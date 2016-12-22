import shutil

def give_labels(in_file, out_file, label):
    with open(in_file, 'r') as infile:
        with open(out_file, 'w') as outfile:
            for line in infile:
                outfile.write('___label___' + label + ' ' + line)


give_labels("data/train_pos.txt", "data/train_pos_labeled.txt", 'positive')
give_labels("data/train_neg.txt", "data/train_neg_labeled.txt", 'negative')
