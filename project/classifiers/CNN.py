import numpy as np
import time
from helpers_functions import load_data, make_submission
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import pickle
import sklearn.neural_network

run_Id = 'CNN'
ratio = 0.1

print("Starting ...")
time_start = time.time()
print('Starting Time : ' + str(time_start))
run_path = '../runs/' + str(run_Id) + '/'
X_train, labels, X_test = load_data(run_path, ratio)

n_samples, n_features = 10, 5
clf = sklearn.neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, labels)

all_results = []
for test_sent in X_test:
    test_sent = np.reshape(test_sent, (1, -1))
    res = clf.predict(test_sent)
    if res < 0:
        res_bin = [-1]
    else:
        res_bin = [1]
    all_results.append(res)
print("Predictions done")


make_submission(run_path, all_results)
print('Duration : ' + str(time.time() - time_start))