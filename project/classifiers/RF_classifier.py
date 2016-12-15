import numpy as np
import time
from helpers_functions import load_data, make_submission
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

# Define a runId for the current classification
run_Id = 201
ratio = 0.1

print("Starting ...")
time_start = time.time()
print('Starting Time : ' + str(time_start))
run_path = '../runs/' + str(run_Id) + '/'
X_train, labels, X_test = load_data(run_path, ratio)

param_grid = [{'n_estimators': [10,15,20,30,50,70],
               'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 'log2']}]
rfc_cv = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3, verbose=10)
rfc_cv.fit(X_train, labels)
print("Cross-validation using RF done")
print('Best Score : ' + str(rfc_cv.best_score_))
print('Best Parameters :' + str(rfc_cv.best_params_))

all_results = []
for test_sent in X_test:
    test_sent = np.reshape(test_sent, (1, -1))
    res = rfc_cv.predict(test_sent)
    all_results.append(res)
print("Predictions done")

make_submission(run_path, all_results)
print('Duration : ' + str(time.time() - time_start))