import numpy as np
import time
from helpers_functions import load_data, make_submission
from sklearn import svm
from sklearn.grid_search import GridSearchCV

# Define a runId for the current classification
run_Id = 'svm'
ratio = 0.2

print("Starting ...")
time_start = time.time()
print('Starting Time : ' + str(time_start))
run_path = '../runs/' + str(run_Id) + '/'
# X_train, labels, X_test = load_data(run_path, ratio)


param_grid = [{'C': [1.0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': [1,2,3]}]
# clf_cv = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, cv=3, verbose=10)
clf_cv = svm.SVC()
clf_cv.fit(X_train, labels)
print("Cross-validation using RF done")
print('Best Score : ' + str(clf_cv.best_score_))
print('Best Parameters :' + str(clf_cv.best_params_))

all_results = []
for test_sent in X_test:
    test_sent = np.reshape(test_sent, (1, -1))
    res = clf_cv.predict(test_sent)
    all_results.append(res)
print("Predictions done")

make_submission(run_path, all_results)
print('Duration : ' + str(time.time() - time_start))