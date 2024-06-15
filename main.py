import pandas as pd
from sklearn import linear_model, metrics, model_selection
import numpy as np
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

names = ['Marital status','Application mode','Application order','evening attendance','Previous qualification',
         'Nationality,Mothers qualification','Fathers qualification','Mothers occupation','Fathers occupation',
         'Educational special needs','Debtor','Tuition fees up to date','Gender','Scholarship holder',
         'Age at enrollment','International','Unemployment rate','Inflation rate','GDP','Output']
# Load the data
df = pd.read_csv('data2.csv',names = names)

mylog_model = linear_model.LogisticRegression()
y = df.values[:, 19]
X = df.values[:, 0:19]

# mylog_model.fit(X, y)