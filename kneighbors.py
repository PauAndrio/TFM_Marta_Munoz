import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def load_and_preprocessing_data():

    filename = 'Lynch_data_all.xlsx'

    # read_excel to spicific columns use: names=['__', '__']
    data = pd.read_excel(filename, header=0)

    # preprocessing string class to int. Fields: 'Gene', 'Region', 'DNA_Change_Type', 'Category_1'
    le = preprocessing.LabelEncoder()

    le.fit(data['Gene'].values)
    gene_int = le.transform(data['Gene'].values)

    le.fit(data['Region'].values)
    region_int = le.transform(data['Region'].values)

    le.fit(data['DNA_Change_Type'].values)
    dna_change_type_int = le.transform(data['DNA_Change_Type'].values)

    le.fit(data['Category_1'].values)
    category_int = le.transform(data['Category_1'].values)

    # replace np.nan value to empty string and transform string to number. Fields: 'AA_Change_Type' and 'Disease_1'
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

    return X,y


def apply_kneighbors(neighbors, X_train, X_test, y_train, y_test, algorithm):
    knn = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algorithm)
    knn.fit(X_train, y_train) 

    accuracy_cross_test = cross_val_score(knn, X, y, cv=5)
    return accuracy_cross_test.mean()

# START CODE
X, y = load_and_preprocessing_data()

# 20% data used to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

k_range = range(1, 40)
scores_test_brute = []
scores_test_ball_tree = []
scores_test_kd_tree = []

for k in k_range:
    scores_test_knn_1 = apply_kneighbors(k, X_train, X_test, y_train, y_test, 'brute') 
    scores_test_brute.append(scores_test_knn_1)

    scores_test_knn_2 = apply_kneighbors(k, X_train, X_test, y_train, y_test, 'ball_tree') 
    scores_test_ball_tree.append(scores_test_knn_2)
    
    scores_test_knn_3 = apply_kneighbors(k, X_train, X_test, y_train, y_test, 'kd_tree') 
    scores_test_kd_tree.append(scores_test_knn_3)

plt.figure()
plt.xlabel('k')
plt.ylabel('Cross-validated accuracy')
plt.title('knn classifier')

plt.scatter(k_range, scores_test_brute, s=400, c='green', alpha=0.4, label='Brute')
plt.scatter(k_range, scores_test_ball_tree, s=300, c='blue', alpha=0.3, label='Ball tree')
plt.scatter(k_range, scores_test_kd_tree, s=100, c='red', alpha=0.2, label='KD tree')

plt.xticks(k_range)
plt.legend()
plt.show()

# Best accuracy
best_values_dict = {}

best_values_dict['brute'] = k_range[scores_test_brute.index(max(scores_test_brute))]
best_values_dict['ball_tree'] = k_range[scores_test_ball_tree.index(max(scores_test_ball_tree))]
best_values_dict['kd_tree'] = k_range[scores_test_kd_tree.index(max(scores_test_kd_tree))]

knn_brute = KNeighborsClassifier(
    n_neighbors=best_values_dict['brute'], algorithm='brute'
)
knn_brute.fit(X_train, y_train)
prediction_brute = knn_brute.predict(X_test)
print(confusion_matrix(y_test, prediction_brute))

knn_ball_tree = KNeighborsClassifier(
    n_neighbors=best_values_dict['ball_tree'], algorithm='ball_tree'
)
knn_ball_tree.fit(X_train, y_train)
prediction_ball_tree = knn_ball_tree.predict(X_test)
print(confusion_matrix(y_test, prediction_ball_tree))

knn_kd_tree = KNeighborsClassifier(
    n_neighbors=best_values_dict['kd_tree'], algorithm='kd_tree'
)
knn_kd_tree.fit(X_train, y_train)
prediction_kd_tree = knn_kd_tree.predict(X_test)
print(confusion_matrix(y_test, prediction_kd_tree))
