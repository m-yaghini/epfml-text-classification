from prepare_data import prepare_data
import numpy as np
import csv

print("Starting")
X_train, labels, X_test = prepare_data()
print("Data loaded")

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, labels)
print("Fitting with RF done")
all_res = []
for test_sent in X_test:
    test_sent = np.reshape(test_sent, (1, -1))
    res = rfc.predict(test_sent)
    all_res.append(res)
print("Predictions done")

with open('../outputs/submission01.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    sub_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    index = 0
    sub_writer.writeheader()
    for res in all_res:
        index += 1
        sub_writer.writerow({'Id': str(index), 'Prediction': str(res[0])})
print("Submission file created")