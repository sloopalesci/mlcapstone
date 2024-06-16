import pandas as pd
from sklearn import linear_model, metrics, model_selection, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import SGDClassifier
# import numpy as np
# from matplotlib import pyplot
# from pandas.plotting import scatter_matrix

names = ['Marital status', 'Application mode', 'Application order', 'Course', 'evening attendance',
         'Previous qualification', 'Nationality', 'Mother\'s qualification',
         'Father\'s qualification', 'Mother\'s occupation', 'Father\'s occupation',
         'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date',
         'Gender', 'Scholarship holder', 'Age at enrollment', 'International', 'CU 1st sem (enrolled)',
         'CU 1st sem (approved)', 'CU 1st sem (evaluations)', 'CU 1st sem (grade)', 'CU 1st sem (credited)',
         'CU 1st sem (without evaluations)',
         'CU 2nd sem (credited)', 'CU 2nd sem (enrolled)', 'CU 2nd sem (evaluations)', 'CU 2nd sem (approved)',
         'CU 2nd sem (grade)',
         'CU 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP', 'Output']
# Load the data
df = pd.read_csv('data3.csv', names=names)

# create models
mylog_model = linear_model.LogisticRegression(max_iter=2000)  # linear regression model
myrf_model = RandomForestClassifier()  # random forest model
mysvm_model = svm.SVC(kernel='linear', probability=True)  # support vector machine model
mygb_model = GradientBoostingClassifier()  # GradientBoostingClassifier model
myxgb_model = xgb.XGBClassifier()  # xgboost model
ab_model = AdaBoostClassifier(algorithm='SAMME')  # AdaBoostClassifier model
et_model = ExtraTreesClassifier()  # ExtraTreesClassifier model
# myknn_model = KNeighborsClassifier()  # KNeighborsClassifier model
# dt_model = DecisionTreeClassifier()  # DecisionTreeClassifier model
# sgd_model = SGDClassifier(loss='modified_huber')  # SGDClassifier model
# mygnb_model = GaussianNB()  # GaussianNB model

y = df.values[:, 34]
X = df.values[:, 0:34]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, )

# train the  models
mylog_model.fit(X_train, y_train)  # linear regression model
myrf_model.fit(X_train, y_train)  # random forest model
mysvm_model.fit(X_train, y_train)  # support vector machine model
mygb_model.fit(X_train, y_train)  # GradientBoostingClassifier model
myxgb_model.fit(X_train, y_train)  # xgboost model
ab_model.fit(X_train, y_train)  # AdaBoostClassifier model
et_model.fit(X_train, y_train)  # ExtraTreesClassifier model
# myknn_model.fit(X_train, y_train)  # KNeighborsClassifier model
# dt_model.fit(X_train, y_train)  # DecisionTreeClassifier model
# sgd_model.fit(X_train, y_train)  # SGDClassifier model
# mygnb_model.fit(X_train, y_train)  # GaussianNB model

# predict the outputs using the test data
y_pred = mylog_model.predict(X_test)  # linear regression model
y_pred_rf = myrf_model.predict(X_test)  # random forest model
y_pred_svm = mysvm_model.predict(X_test)  # support vector machine model
y_pred_gb = mygb_model.predict(X_test)  # GradientBoostingClassifier model
y_pred_xgb = myxgb_model.predict(X_test)  # xgboost model
y_pred_ab = ab_model.predict(X_test)  # AdaBoostClassifier model
y_pred_et = et_model.predict(X_test)  # ExtraTreesClassifier model
# y_pred_sgd = sgd_model.predict(X_test)  # SGDClassifier model
# y_pred_knn = myknn_model.predict(X_test)  # KNeighborsClassifier model
# y_pred_gnb = mygnb_model.predict(X_test)  # GaussianNB model
# y_pred_dt = dt_model.predict(X_test)  # DecisionTreeClassifier model

# print out  the prediction of the same output for all the models
dataset = [
    [1, 6, 1, 11, 1, 1, 1, 1, 3, 4, 4, 1, 0, 0, 0, 1, 0, 19, 0, 6, 6, 6, 14, 0, 0, 0, 6, 6, 6, 13.66666667, 0, 13.9,
     -0.3, 0.79]]

# create variables to store the prediction of the output for all the models
linear_prediction = mylog_model.predict(dataset)
random_forest_prediction = myrf_model.predict(dataset)
svm_prediction = mysvm_model.predict(dataset)
gbc_prediction = mygb_model.predict(dataset)
xgboost_prediction = myxgb_model.predict(dataset)
ab_prediction = ab_model.predict(dataset)
et_prediction = et_model.predict(dataset)
# sgd_prediction = sgd_model.predict(dataset)
# knn_prediction = myknn_model.predict(dataset)
# gnb_prediction = mygnb_model.predict(dataset)
# dt_prediction = dt_model.predict(dataset)

print("linear regression predicts:", linear_prediction, "and has an accuracy of:",
      mylog_model.score(X_test, y_test))
print("random forest predicts:", random_forest_prediction, "and has an accuracy of:",
      myrf_model.score(X_test, y_test))
print("support vector machine predicts:", svm_prediction, "and has an accuracy of:",
      mysvm_model.score(X_test, y_test))
print("GradientBoostingClassifier predicts:", gbc_prediction, "and has an accuracy of:",
      mygb_model.score(X_test, y_test))
print("xgboost predicts:", xgboost_prediction, "and has an accuracy of:", myxgb_model.score(X_test, y_test))
print("AdaBoost predicts:", ab_prediction, "and has an accuracy of:", ab_model.score(X_test, y_test))
print("Extra Trees predicts:", et_prediction, "and has an accuracy of:", et_model.score(X_test, y_test))
# print("Stochastic Gradient Descent predicts:", sgd_prediction, "and has an accuracy of:",
#       sgd_model.score(X_test, y_test))
# print("KNeighborsClassifier predicts:", knn_prediction, "and has an accuracy of:",
#       myknn_model.score(X_test, y_test))
# print("GaussianNB predicts:", gnb_prediction, "and has an accuracy of:",
#       mygnb_model.score(X_test, y_test))
# print("Decision Tree predicts:", dt_prediction, "and has an accuracy of:", dt_model.score(X_test, y_test))

# print out the accuracy of the models
voting_clf = VotingClassifier(estimators=[('lr', mylog_model), ('rf', myrf_model), ('svm', mysvm_model),
                                          ('gbc', mygb_model), ('xgb', myxgb_model), ('ab', ab_model),
                                          ('et', et_model)], voting='soft')
voting_clf.fit(X_train, y_train)

print("Voting Classifier predicts:", voting_clf.predict(dataset), "and has an accuracy of:",
      voting_clf.score(X_test, y_test))
