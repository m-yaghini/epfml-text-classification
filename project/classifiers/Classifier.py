import numpy as np
import time
from helpers_functions import load_data, make_submission
from sklearn import svm
import pickle
from sklearn.model_selection import GridSearchCV

print("Starting ...")
with open('../data/train_data.pkl', 'rb') as train_data_picklefile:
    x_train, y_train = pickle.load(train_data_picklefile)

with open('../data/test_data.pkl', 'rb') as test_data_picklefile:
    x_test = pickle.load(test_data_picklefile)

X_train = np.transpose(x_train)
labels = y_train.ravel()
X_test = np.transpose(x_test)

print(X_train.shape)
print(labels.shape)
print(X_test.shape)


print("Starting ...")
time_start = time.time()
print('Starting Time : ' + str(time_start))

param_grid = [{'C': [1.0], 'kernel': ['linear', 'rbf', 'sigmoid']}]
clf_cv = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, cv=2, verbose=10)
clf_cv = svm.SVC(kernel='rbf')
clf_cv.fit(X_train, labels)
print("Cross-validation using RF done")
#print('Best Score : ' + str(clf_cv.best_score_))
#print('Best Parameters :' + str(clf_cv.best_params_))

all_results = []
for test_sent in X_test:
    test_sent = np.reshape(test_sent, (1, -1))
    res = clf_cv.predict(test_sent)
    all_results.append(res)
print("Predictions done")

make_submission('', all_results)
print('Duration : ' + str(time.time() - time_start))