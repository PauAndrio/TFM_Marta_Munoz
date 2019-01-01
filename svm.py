import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    cross_val_predict, cross_val_score, train_test_split
)
from sklearn import svm


def load_and_preprocessing_data():

    filename = 'Lynch_data_all.xlsx'

    # read_excel to spicific columns use: names=['__', '__']
    data = pd.read_excel(filename, header=0)

    # preprocessing string class to int.
    # Fields: 'Gene', 'Region', 'DNA_Change_Type', 'Category_1'
    le = preprocessing.LabelEncoder()
    le.fit(data['Gene'].values)
    gene_int = le.transform(data['Gene'].values)

    le.fit(data['Region'].values)
    region_int = le.transform(data['Region'].values)

    le.fit(data['DNA_Change_Type'].values)
    dna_change_type_int = le.transform(data['DNA_Change_Type'].values)

    le.fit(data['Category_1'].values)
    category_int = le.transform(data['Category_1'].values)

    # replace np.nan value to empty string and transform string to number.
    # Fields: 'AA_Change_Type' and 'Disease_1'
    aa_change_type_int = data['AA_Change_Type'].replace(np.nan, 'UNKNOWN', regex=True)
    le.fit(aa_change_type_int.values)
    aa_change_type_int = le.transform(aa_change_type_int.values)

    disease_1_int = data['Disease_1'].replace(np.nan, 'Healthy', regex=True)
    le.fit(disease_1_int.values)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    disease_1_int = le.transform(disease_1_int.values)

    # create design matrix X and target vector y
    X = np.array([gene_int, region_int, dna_change_type_int, category_int, aa_change_type_int])
    X = np.transpose(X)

    # reshape
    y = np.array(disease_1_int).reshape(402,)

    return X, y


def apply_linear_svc(X_train, X_test, y_train, y_test):
    svc = svm.LinearSVC()
    svc.fit(X_train, y_train)

    accuracy_cross_test = cross_val_score(svc, X, y, cv=5)
    return accuracy_cross_test.mean()


def apply_svc(X_train, X_test, y_train, y_test, kernel_type):
    svc = svm.SVC(kernel=kernel_type)
    svc.fit(X_train, y_train)

    accuracy_cross_test = cross_val_score(svc, X, y, cv=5)
    return accuracy_cross_test.mean()


def apply_svc_poly(X_train, X_test, y_train, y_test, degree_n):
    svc = svm.SVC(kernel='poly', degree=degree_n, gamma='scale')
    svc.fit(X_train, y_train)

    accuracy_cross_test = cross_val_score(svc, X, y, cv=5)
    return accuracy_cross_test.mean()


# START CODE
X, y = load_and_preprocessing_data()

# 20% data used to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

accuracy_test_linear_svc = apply_linear_svc(X_train, X_test, y_train, y_test)
accuracy_test_linear = apply_svc(X_train, X_test, y_train, y_test, 'linear')
accuracy_test_rbf = apply_svc(X_train, X_test, y_train, y_test, 'rbf')
accuracy_test_poly = apply_svc(X_train, X_test, y_train, y_test, 'poly')

print(accuracy_test_linear_svc)
print(accuracy_test_linear)
print(accuracy_test_rbf)
print(accuracy_test_poly)

k_range = range(1, 10)
scores_test_poly = []
for k in k_range:
    score_test_k = apply_svc_poly(X_train, X_test, y_train, y_test, k)
    scores_test_poly.append(score_test_k)

plt.figure()
plt.title('SVC Polynomial')
plt.xlabel('Degree')
plt.ylabel('Cross-validated accuracy')

plt.plot(k_range, scores_test_poly)

plt.xticks(k_range)
plt.show()

# BEST SVM
degree_best = scores_test_poly.index(max(scores_test_poly)) + 1
print(' degree_best : {}'.format(degree_best))
svm_poly_best = svm.SVC(kernel='poly', degree=degree_best, gamma='scale')
svm_poly_best.fit(X_train, y_train)
prediction = svm_poly_best.predict(X_test)
prediction = cross_val_predict(svm_poly_best, X, y, cv=5)
print(confusion_matrix(y_test, prediction))
