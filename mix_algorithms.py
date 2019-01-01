import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB


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
    X = np.array([
        gene_int, region_int, dna_change_type_int, category_int, aa_change_type_int
    ])
    X = np.transpose(X)

    # reshape
    y = np.array(disease_1_int).reshape(402,)

    return X, y


def lda(X, y, X_train, y_train, X_test, y_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    accuracy_lda = cross_val_score(lda, X, y).mean()
    print('Score: LDA {}'.format(accuracy_lda))
    predictions = lda.predict(X_test)
    print(confusion_matrix(y_test, predictions))


def qda(X, y, X_train, y_train, X_test, y_test):
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    accuracy_qda = cross_val_score(qda, X, y).mean()
    print('Score: QDA {}'.format(accuracy_qda))
    predictions = qda.predict(X_test)
    print(confusion_matrix(y_test, predictions))


def bernoulli_nb(X, y, X_train, y_train, X_test, y_test):
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    accuracy_bnb = cross_val_score(bnb, X, y).mean()
    print('Score: BernoulliNB {}'.format(accuracy_bnb))
    predictions = bnb.predict(X_test)
    print(confusion_matrix(y_test, predictions))


def gaussian_process_classifier(X, y, X_train, y_train, X_test, y_test):
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    accuracy_gpc = cross_val_score(gpc, X, y).mean()
    print('Score: GaussianProcessClassifier {}'.format(accuracy_gpc))
    predictions = gpc.predict(X_test)
    print(confusion_matrix(y_test, predictions))


# START CODE
X, y = load_and_preprocessing_data()

# 20% data used to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

bernoulli_nb(X, y, X_train, y_train, X_test, y_test)
lda(X, y, X_train, y_train, X_test, y_test)
qda(X, y, X_train, y_train, X_test, y_test)
gaussian_process_classifier(X, y, X_train, y_train, X_test, y_test)
