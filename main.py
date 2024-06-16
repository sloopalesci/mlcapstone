import pandas as pd
from sklearn import linear_model, metrics, model_selection, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier

import numpy as np
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

names = ['Marital status', 'Application mode', 'Application order', 'Course', 'evening attendance',
         'Previous qualification', 'Nationality', 'Mother\'s qualification', 'Father\'s qualification',
         'Mother\'s occupation',
         'Father\'s occupation', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
         'Gender', 'Scholarship holder', 'Age at enrollment', 'International', 'CU 1st sem (enrolled)',
         'CU 1st sem (approved)', 'CU 1st sem (evaluations)', 'CU 1st sem (grade)', 'CU 1st sem (credited)',
         'CU 1st sem (without evaluations)',
         'CU 2nd sem (credited)', 'CU 2nd sem (enrolled)', 'CU 2nd sem (evaluations)', 'CU 2nd sem (approved)',
         'CU 2nd sem (grade)',
         'CU 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP', 'Output']
# Load the data
df = pd.read_csv('data3.csv', names=names)

# create model using linear regression
mylog_model = linear_model.LogisticRegression()
# create model using random forest
myrf_model = RandomForestClassifier()
# create the model using support vector machine
mysvm_model = svm.SVC(kernel='linear')
# create the model using KNeighborsClassifier
myknn_model = KNeighborsClassifier()
# create the model using GradientBoostingClassifier
mygb_model = GradientBoostingClassifier()
# create the model using xgboost
myxgb_model = xgb.XGBClassifier()
# create the model using GaussianNB
mygnb_model = GaussianNB()
# create the model using DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
# create the model using AdaBoostClassifier
ab_model = AdaBoostClassifier()
# create the model using ExtraTreesClassifier
et_model = ExtraTreesClassifier()
# create the model using SGDClassifier
sgd_model = SGDClassifier()

y = df.values[:, 34]
X = df.values[:, 0:34]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, )

# train the linear regression model
mylog_model.fit(X_train, y_train)
# train the random forest model
myrf_model.fit(X_train, y_train)
# train the support vector machine model
mysvm_model.fit(X_train, y_train)
# train the KNeighborsClassifier model
myknn_model.fit(X_train, y_train)
# train the GradientBoostingClassifier model
mygb_model.fit(X_train, y_train)
# train the xgboost model
myxgb_model.fit(X_train, y_train)
# train the GaussianNB model
mygnb_model.fit(X_train, y_train)
# train the DecisionTreeClassifier model
dt_model.fit(X_train, y_train)
# train the AdaBoostClassifier model
ab_model.fit(X_train, y_train)
# train the ExtraTreesClassifier model
et_model.fit(X_train, y_train)
# train the SGDClassifier model
sgd_model.fit(X_train, y_train)

# predict the output using the linear regression model
y_pred = mylog_model.predict(X_test)
# predict the output using the random forest model
y_pred_rf = myrf_model.predict(X_test)
# predict the output using the support vector machine model
y_pred_svm = mysvm_model.predict(X_test)
# predict the output using the KNeighborsClassifier model
y_pred_knn = myknn_model.predict(X_test)
# predict the output using the GradientBoostingClassifier model
y_pred_gb = mygb_model.predict(X_test)
# predict the output using the xgboost model
y_pred_xgb = myxgb_model.predict(X_test)
# predict the output using the GaussianNB model
y_pred_gnb = mygnb_model.predict(X_test)
# predict the output using the DecisionTreeClassifier model
y_pred_dt = dt_model.predict(X_test)
# predict the output using the AdaBoostClassifier model
y_pred_ab = ab_model.predict(X_test)
# predict the output using the ExtraTreesClassifier model
y_pred_et = et_model.predict(X_test)
# predict the output using the SGDClassifier model
y_pred_sgd = sgd_model.predict(X_test)

# print the accuracy of the linear regression model
print("linear regression:", metrics.accuracy_score(y_test, y_pred))
# print the accuracy of the random forest model
print("random forest:", metrics.accuracy_score(y_test, y_pred_rf))
# print the accuracy of the support vector machine model
print("support vector machine:", metrics.accuracy_score(y_test, y_pred_svm))
# print the accuracy of the KNeighborsClassifier model
print("KNeighborsClassifier:", metrics.accuracy_score(y_test, y_pred_knn))
# print the accuracy of the GradientBoostingClassifier model
print("GradientBoostingClassifier:", metrics.accuracy_score(y_test, y_pred_gb))
# print the accuracy of the xgboost model
print("xgboost:", metrics.accuracy_score(y_test, y_pred_xgb))
# print the accuracy of the GaussianNB model
print("GaussianNB:", metrics.accuracy_score(y_test, y_pred_gnb))
# print the accuracy of the DecisionTreeClassifier model
print("Decision Tree:", metrics.accuracy_score(y_test, y_pred_dt))
# print the accuracy of the AdaBoostClassifier model
print("AdaBoost:", metrics.accuracy_score(y_test, y_pred_ab))
# print the accuracy of the ExtraTreesClassifier model
print("Extra Trees:", metrics.accuracy_score(y_test, y_pred_et))
# print the accuracy of the SGDClassifier model
print("Stochastic Gradient Descent:", metrics.accuracy_score(y_test, y_pred_sgd))

# print out  the prediction of the same output for all the models
dataset = [
    [1, 8, 5, 2, 1, 1, 1, 13, 10, 6, 10, 1, 0, 0, 1, 1, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10.8, 1.4, 1.74]]



print("linear regression predicts:", mylog_model.predict(dataset), "and has an accuracy of:",
      mylog_model.score(X_test, y_test))
print("random forest predicts:", myrf_model.predict(dataset), "and has an accuracy of:",
      myrf_model.score(X_test, y_test))
print("support vector machine predicts:", mysvm_model.predict(dataset), "and has an accuracy of:",
      mysvm_model.score(X_test, y_test))
print("KNeighborsClassifier predicts:", myknn_model.predict(dataset), "and has an accuracy of:",
      myknn_model.score(X_test, y_test))
print("GradientBoostingClassifier predicts:", mygb_model.predict(dataset), "and has an accuracy of:",
      mygb_model.score(X_test, y_test))
print("xgboost predicts:", myxgb_model.predict(dataset), "and has an accuracy of:", myxgb_model.score(X_test, y_test))
print("GaussianNB predicts:", mygnb_model.predict(dataset), "and has an accuracy of:",
      mygnb_model.score(X_test, y_test))
print("Decision Tree predicts:", dt_model.predict(dataset), "and has an accuracy of:", dt_model.score(X_test, y_test))
print("AdaBoost predicts:", ab_model.predict(dataset), "and has an accuracy of:", ab_model.score(X_test, y_test))
print("Extra Trees predicts:", et_model.predict(dataset), "and has an accuracy of:", et_model.score(X_test, y_test))
print("Stochastic Gradient Descent predicts:", sgd_model.predict(dataset), "and has an accuracy of:",
      sgd_model.score(X_test, y_test))

# print(df['Output'].value_counts())
# print(y)
# print(df.columns)
