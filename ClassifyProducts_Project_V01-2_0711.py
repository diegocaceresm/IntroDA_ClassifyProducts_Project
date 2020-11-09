import os
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

os.chdir("/Users/lhc0537/Desktop/Project")
overall_data = pd.read_csv('ClassifyProducts.csv')

#################
# DATA OVERVIEW #
#################
print ("\n")
print ("                   DATA OVERVIEW","\n")

print(overall_data[0:5],"\n")
print("Content")
print(overall_data.shape,"\n")
x_overall = overall_data.iloc[:, 1:-1].values # Values of X are only considering feat 1 - feat 93
y_overall = overall_data.iloc[:, -1].values   # Values of Y are only considering the Classes
print("Distribution of Features Lines, Columns",x_overall.shape)
print("Distribution of 9 Classes in lines",y_overall.shape)
print("_________________________________________________________")

#################
# DATA BARGRAPH #
#################

df_pivot = overall_data.pivot_table(index=['target'], aggfunc='size')
y = df_pivot.values
x = df_pivot.index

plt.bar(x,y,align = 'center', alpha = 0.5)
plt.ylabel('Products')
plt.title('Product Distribution by Product Target')
plt.show()


#################
#   DATA SPLIT  #
#################
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, train_size=0.8, random_state = 0)
for train_index, test_index in sss.split(x_overall, y_overall):
    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = x_overall[train_index], x_overall[test_index]
    y_train, y_test = y_overall[train_index], y_overall[test_index]

###########
#  PCA    #
###########
from sklearn.decomposition import PCA
import seaborn as sns

X_train_N = (X_train-X_train.min())/(X_train.max()-X_train.min())

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_N)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel("Principal Component")
plt.ylabel("Cumulative Proportion of Variance Explained")

plt.show()

#create report dataframe
report = pd.DataFrame(columns=['Model','Mean Acc. Training','Standard Deviation','Acc. Test'])


###############
# Naive Bayes #
###############

from sklearn.naive_bayes import GaussianNB
nbmodel = GaussianNB()
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(nbmodel, X_train, y_train, scoring='accuracy', cv = 10)
print("Accuracies = ", accuracies)
nbmodel.fit(X_train, y_train)
y_test_pred = nbmodel.predict(X_test)
accte = accuracy_score(y_test, y_test_pred)
report.loc[len(report)] = ['Naive Bayes', accuracies.mean(), accuracies.std(), accte]
print(report.loc[len(report)-1])


################################
# Gradient Boosting Classifier #
################################

from sklearn.ensemble import GradientBoostingClassifier
gbmodel = GradientBoostingClassifier(random_state=0)
from sklearn.model_selection import GridSearchCV
param_grid = { 
    'max_depth': [ 3., 4., 5.],
    'subsample': [0.7, 0.8, 0.9],
    'n_estimators': [50, 100,150],
    'learning_rate': [0.1, 0.2, 0.3]
}
CV_gbmodel = GridSearchCV(estimator=gbmodel, param_grid=param_grid, cv=10)
CV_gbmodel.fit(X_train, y_train)
print(CV_gbmodel.best_params_)
#use the best parameters
gbmodel = gbmodel.set_params(**CV_gbmodel.best_params_)
gbmodel.fit(X_train, y_train)
y_test_pred = gbmodel.predict(X_test)
accte = accuracy_score(y_test, y_test_pred)
report.loc[len(report)] = ['Gradient Boosting (grid)', 
                          CV_gbmodel.cv_results_['mean_test_score'][CV_gbmodel.best_index_], 
                          CV_gbmodel.cv_results_['std_test_score'][CV_gbmodel.best_index_], accte]
print(report.loc[len(report)-1])

##################
# Decision Trees #
##################
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
dtmodel = DecisionTreeClassifier(random_state=0)
param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': np.arange(10,100,10),
              'min_samples_leaf': np.arange(1,10,2)}
CV_dtmodel = GridSearchCV(estimator=dtmodel, param_grid=param_grid, cv=10)
CV_dtmodel.fit(X_train, y_train)
print(CV_dtmodel.best_params_)

#use the best parameters
dtmodel = dtmodel.set_params(**CV_dtmodel.best_params_)
dtmodel.fit(X_train, y_train)
y_test_pred = dtmodel.predict(X_test)
accte = accuracy_score(y_test, y_test_pred)
report.loc[len(report)] = ['Decision Trees (grid)', 
                          CV_dtmodel.cv_results_['mean_test_score'][CV_dtmodel.best_index_], 
                          CV_dtmodel.cv_results_['std_test_score'][CV_dtmodel.best_index_], accte]
print(report.loc[len(report)-1])

###########
# XGBoost #
###########
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
xgmodel = XGBClassifier()
param_grid = {'max_depth': [ 2, 4, 6, 8],
              'colsample_bytree': [0.7],
              'n_estimators': [50, 100, 150, 200],
              'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3]}
CV_xgmodel = GridSearchCV(estimator=xgmodel, param_grid=param_grid, cv=10)
CV_xgmodel.fit(X_train, y_train)
print(CV_xgmodel.best_params_)

#use the best parameters
xgmodel = xgmodel.set_params(**CV_xgmodel.best_params_)
xgmodel.fit(X_train, y_train)
y_test_pred = xgmodel.predict(X_test)
accte = accuracy_score(y_test, y_test_pred)
report.loc[len(report)] = ['XGBoost (grid)', 
                          CV_xgmodel.cv_results_['mean_test_score'][CV_xgmodel.best_index_], 
                          CV_xgmodel.cv_results_['std_test_score'][CV_xgmodel.best_index_], accte]
print(report.loc[len(report)-1])


################
# Final Report #
################

print(report)








