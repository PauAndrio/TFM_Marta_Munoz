import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns


filename = 'Lynch_data_all.xlsx'

# read_excel to spicific columns use: names=['__', '__']
data = pd.read_excel(filename, header=0)

# preprocessing string class to int
le = preprocessing.LabelEncoder()

# fit the data
le.fit(data['Gene'].values)
# tranform string data to number data
gene_int = le.transform(data['Gene'].values)
# A standard pie plot
plt.bar(list(le.classes_), data.groupby('Gene').size(), color='mediumspringgreen')
plt.title('Gene')
plt.show()

le.fit(data['Region'].values)
region_int = le.transform(data['Region'].values)
plt.bar(list(le.classes_), data.groupby('Region').size())
plt.xticks(list(le.classes_), rotation='vertical', ha='center')
plt.title('Region')
plt.show()

le.fit(data['DNA_Change_Type'].values)
dna_change_type_int = le.transform(data['DNA_Change_Type'].values)
plt.bar(list(le.classes_), data.groupby('DNA_Change_Type').size(), color='lightsalmon')
plt.xticks(list(le.classes_), rotation='horizontal', ha='center')
plt.title('DNA Change Type')
plt.show()

aa_change_type_int = data['AA_Change_Type'].replace(np.nan, 'UNKNOWN', regex=True)
le.fit(aa_change_type_int.values)
aa_change_type_int = le.transform(aa_change_type_int.values)
plt.bar(list(le.classes_), data.groupby('AA_Change_Type').size(), color='lightblue')
plt.xticks(list(le.classes_), rotation='horizontal', ha='center')
plt.title('AA Change Type')
plt.show()

le.fit(data['Category_1'].values)
category_int = le.transform(data['Category_1'].values)
plt.bar(list(le.classes_), data.groupby('Category_1').size(), color='gold')
plt.xticks(list(le.classes_), rotation='vertical', ha='center')
plt.title('Category')
plt.show()

disease_1_int = data['Disease_1'].replace(np.nan, 'Healthy', regex=True)
le.fit(disease_1_int.values)
disease_1_int = le.transform(disease_1_int.values)
plt.bar(['Healthy', 'Lynch Syndrome'], data.groupby('Disease_1').size(), color='plum')
plt.title('Disease')
plt.show()

corelation_table = pd.crosstab(
    index=data["Category_1"], columns=data["Disease_1"]
)
corelation_table.plot(kind="bar", stacked=False)
print(corelation_table)
plt.xlabel('Category')
plt.title('Relation Disease-Category')
plt.legend(['Healthy', 'Lynch syndrome'])
plt.show()

# correlation map
# data[['Gene', 'Region', 'DNA_Change_Type', 'AA_Change_Type', 'Category_1']]
features_dict = {
    'Gene': gene_int, 'Region': region_int, 'DNA_Change_Type': dna_change_type_int,
    'AA_Change_Type': aa_change_type_int, 'Category_1': category_int
}
features_data = pd.DataFrame(features_dict)
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(features_data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.yticks(rotation=0)
plt.show()
