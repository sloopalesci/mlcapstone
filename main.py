import pandas as pd
from sklearn import linear_model, metrics, model_selection
import numpy as np
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

names = ['Marital status', 'Application mode', 'Application order', 'Course', 'evening attendance',
         'Previous qualification', 'Nationality', 'Mother\'s qualification', 'Father\'s qualification', 'Mother\'s occupation',
         'Father\'s occupation', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
         'Gender', 'Scholarship holder', 'Age at enrollment', 'International', 'CU 1st sem (enrolled)',
         'CU 1st sem (approved)', 'CU 1st sem (evaluations)', 'CU 1st sem (grade)', 'CU 1st sem (credited)', 'CU 1st sem (without evaluations)',
         'CU 2nd sem (credited)', 'CU 2nd sem (enrolled)', 'CU 2nd sem (evaluations)', 'CU 2nd sem (approved)', 'CU 2nd sem (grade)',
         'CU 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP', 'Output']
# Load the data
df = pd.read_csv('data2.csv', names=names)

mylog_model = linear_model.LogisticRegression()
y = df.values[:, 34]
X = df.values[:, 0:34]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2,)

mylog_model.fit(X_train, y_train)

y_pred = mylog_model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))

print(mylog_model.predict([[1,8,5,2,1,1,1,13,10,6,10,1,0,0,1,1,0,20,0,0,0,0,0,0,0,0,0,0,0,0,0,10.8,1.4,1.74]]))

# print(df['Output'].value_counts())
# print(y)
# print(df.columns)