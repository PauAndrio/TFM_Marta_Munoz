import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import spline
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split


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
    disease_1_int = le.transform(disease_1_int.values)

    # create design matrix X and target vector y
    X = np.array([gene_int, region_int, dna_change_type_int, category_int, aa_change_type_int])
    X = np.transpose(X)

    # reshape
    y = np.array(disease_1_int).reshape(402,)

    return X, y


best_values_dict = {}

# START CODE
X, y = load_and_preprocessing_data()

# 20% data used to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# n_estimators
trees_range = range(1, 200)
accuracy_trees = []
for tree in trees_range:
    clf1 = RandomForestClassifier(n_estimators=tree)
    clf1.fit(X_train, y_train)
    accuracy_trees.append(cross_val_score(clf1, X, y, cv=5).mean())

plt.figure()
poly = np.polyfit(trees_range, accuracy_trees, 10)
poly_y = np.poly1d(poly)(trees_range)
plt.plot(trees_range, poly_y)
plt.title('n_estimators')
plt.ylabel('Cross-validated accuracy')
plt.show()

best_values_dict['n_estimators'] = trees_range[
    accuracy_trees.index(max(accuracy_trees))
]

# max_features
features = np.arange(0.10, 1.00, 0.10)
accuracy_features = []
for feature in features:
    clf2 = RandomForestClassifier(max_features=feature)
    clf2.fit(X_train, y_train)
    accuracy_features.append(cross_val_score(clf2, X, y, cv=5).mean())

clf3 = RandomForestClassifier()
clf3.fit(X_train, y_train)
accuracy_features.append(cross_val_score(clf3, X, y, cv=5).mean())

clf4 = RandomForestClassifier(max_features='sqrt')
clf4.fit(X_train, y_train)
accuracy_features.append(cross_val_score(clf4, X, y, cv=5).mean())

features_list = list(map(lambda x: float("{0:.2f}".format(x)), features))
features_list.append(1.0)
features_list.append('sqrt')

plt.cla()
plt.title('max_features')
plt.ylabel('Cross-validated accuracy')
plt.plot(features_list, accuracy_features)
plt.show()

best_values_dict['max_features'] = features_list[
    accuracy_features.index(max(accuracy_features))
]

# min_sample_leaf
leafs_range = range(1, 50)
accuracy_leafs = []
for leaf in leafs_range:
    clf5 = RandomForestClassifier(min_samples_leaf=leaf)
    clf5.fit(X_train, y_train)
    accuracy_leafs.append(cross_val_score(clf5, X, y, cv=5).mean())

plt.cla()
plt.title('min_samples_leaf')
plt.ylabel('Cross-validated accuracy')
plt.plot(leafs_range, accuracy_leafs)
plt.show()
best_values_dict['min_samples_leaf'] = leafs_range[
    accuracy_leafs.index(max(accuracy_leafs))
]

# min_sample_split
splits_range = range(2, 50)
accuracy_splits = []
for split in splits_range:
    clf6 = RandomForestClassifier(min_samples_split=split)
    clf6.fit(X_train, y_train)
    accuracy_splits.append(cross_val_score(clf6, X, y, cv=5).mean())

plt.cla()
plt.title('min_samples_split')
plt.ylabel('Cross-validated accuracy')
plt.plot(splits_range, accuracy_splits)
plt.show()
best_values_dict['min_samples_split'] = splits_range[
    accuracy_splits.index(max(accuracy_splits))
]

# Best values of each parameter
print(best_values_dict)

# Utilizar con los mejores parámetros obtenidos
clf_best_values = RandomForestClassifier()
clf_best_values.set_params(**best_values_dict)
clf_best_values.fit(X_train, y_train)

# Predecimos para los valores del grupo Test
predictions = clf_best_values.predict(X_test)
print(confusion_matrix(y_test, predictions))

accuracy_test = cross_val_score(clf_best_values, X, y, cv=5).mean()
print('Test {}'.format(accuracy_test))

# ExtraTreesClassifier
model = ExtraTreesClassifier()
model.set_params(**best_values_dict)
model.fit(X_train, y_train)

list_feature_importances = [
    'Gene', 'Region', 'DNA_Change_Type', 'Category_1', 'AA_Change_Type'
]
plt.title('features importances')
plt.bar(list_feature_importances, model.feature_importances_)
plt.show()
