import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pydot

from pydotplus import graph_from_dot_data
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphvix\bin'


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
    print(data.groupby('Disease_1').size())
    le.fit(disease_1_int.values)
    disease_1_int = le.transform(disease_1_int.values)

    # create design matrix X and target vector y
    X = np.array([gene_int, region_int, dna_change_type_int, category_int, aa_change_type_int])
    X = np.transpose(X)

    # reshape
    y = np.array(disease_1_int).reshape(402,)

    return X, y


# START CODE
X, y = load_and_preprocessing_data()

# 20% data used to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_values_dict = {}

# criterion
clf1 = DecisionTreeClassifier(criterion='entropy')
clf1.fit(X_train, y_train)
accuracy_test1 = cross_val_score(clf1, X, y, cv=5).mean()
print('First test {}'.format(accuracy_test1.mean()))
# Predecimos para los valores del grupo Test
predictions1 = clf1.predict(X_test)
print(confusion_matrix(y_test, predictions1))

clf2 = DecisionTreeClassifier(criterion='gini')
clf2.fit(X_train, y_train)
accuracy_test2 = cross_val_score(clf2, X, y, cv=5).mean()
print('Second test {}'.format(accuracy_test2.mean()))
predictions2 = clf2.predict(X_test)
print(confusion_matrix(y_test, predictions2))

best_values_dict['criterion'] = 'entropy' if accuracy_test1>=accuracy_test2 else 'gini'

# class_weight
clf3 = DecisionTreeClassifier(class_weight={0: 1.77, 1: 1.0})
clf3.fit(X_train, y_train)
accuracy_test3 = cross_val_score(clf3, X, y, cv=5).mean()
print('Third test (class_weight={{0:1.77}}) {}'.format(accuracy_test3.mean()))
predictions3 = clf3.predict(X_test)
print(confusion_matrix(y_test, predictions3))

best_values_dict['class_weight'] = {0: 1.77, 1: 1.0}

# min_samples_split
accuracy_test4 = []
split_range = range(5, 25)
for split in split_range:
	clf3 = DecisionTreeClassifier(min_samples_split=split)
	clf3.fit(X_train, y_train) 
	accuracy_test4.append(cross_val_score(clf3, X, y, cv=5).mean())

plt.figure()
plt.xlabel('k')
plt.ylabel('Cross-validated accuracy')
plt.title('min_samples_split')
plt.scatter(split_range, accuracy_test4)
plt.xticks(split_range)
# plt.show()

best_values_dict['min_samples_split'] = split_range[accuracy_test4.index(max(accuracy_test4))]

# min_samples_leaf
accuracy_test5 = []
leaf_range = range(1, 20)
for leaf in leaf_range:
	clf5 = DecisionTreeClassifier(min_samples_leaf=leaf)
	clf5.fit(X_train, y_train) 
	accuracy_test5.append(cross_val_score(clf5, X, y, cv=5).mean())

plt.figure()
plt.xlabel('k')
plt.ylabel('Cross-validated accuracy')
plt.title('min_samples_leaf')
plt.scatter(leaf_range, accuracy_test5)
plt.xticks(leaf_range)
# plt.show()

best_values_dict['min_samples_leaf'] = leaf_range[accuracy_test5.index(max(accuracy_test5))]

# max_depth
accuracy_test6 = []
depth_range = range(1, 20)
for depth in depth_range:
    clf6 = DecisionTreeClassifier(max_depth=depth)
    clf6.fit(X_train, y_train) 
    accuracy_test6.append(cross_val_score(clf6, X, y, cv=5).mean())

plt.figure()
plt.xlabel('k')
plt.ylabel('Cross-validated accuracy')
plt.title('max_depth')
plt.scatter(depth_range, accuracy_test6)
plt.xticks(depth_range)
plt.show()

best_values_dict['max_depth'] = depth_range[accuracy_test6.index(max(accuracy_test6))]

# Best values of each parameter
print(best_values_dict)

clf_best_values = DecisionTreeClassifier()
clf_best_values.set_params(**best_values_dict)
clf_best_values.fit(X_train, y_train)
accuracy_best_values = cross_val_score(clf_best_values, X, y, cv=5).mean()
print('Best values accuracy {}'.format(accuracy_best_values))
predictions_best_values = clf_best_values.predict(X_test)
print(confusion_matrix(y_test, predictions_best_values))

features = ['Gene', 'Region', 'DNA Change Type', 'Category', 'AA Change Type']
dot_data = export_graphviz(
	clf_best_values, out_file=None, feature_names=features, filled=True, rounded=True,
    proportion=True
)

graph = graph_from_dot_data(dot_data)
graph.write_png('decision_tree.png')
