import csv
import os
import sys
sys.path.insert(0, '../scripts/')
from process_data import prepare_data_files, prepare_data_sets

def load_data(run_path, ratio=0.001):
    if not os.path.exists(run_path + 'data/train_pos.txt') \
    or not os.path.exists(run_path + 'data/train_neg.txt') \
    or not os.path.exists(run_path + 'outputs/train/embeddings.npy') \
    or not os.path.exists(run_path + 'outputs/test/embeddings.npy'):
        prepare_data_files(run_path, ratio, )
    return prepare_data_sets(run_path)

def make_submission(run_path, results, file_name='submission'):
    with open(run_path + str(file_name) + '.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        sub_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        index = 0
        sub_writer.writeheader()
        for res in results:
            index += 1
            sub_writer.writerow({'Id': str(index), 'Prediction': str(res[0])})
    print("Submission file created")
