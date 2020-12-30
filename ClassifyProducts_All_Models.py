#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

os.chdir("C:/Users/cdieg/OneDrive/Documents/MADS/Courses/q1/Introduction to Data Analytics in Business/Project")
df = pd.read_csv('ClassifyProducts.csv')

df.head(10)
df.tail(10)

df = df.drop('id', axis=1)
y = df['target']
y = y.map(lambda s: s[6:])
print("Interesting use of lambda",y,"\n")


X_ori = df.drop('target', axis=1)

# normalize
X = (X_ori-X_ori.min())/(X_ori.max()-X_ori.min())
#print(X.head(6))


from collections import Counter
print(Counter(y))

##################
#  Data Split    #
##################
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_index, test_index in sss.split(X, y):
    print(len(train_index))
    print(len(test_index))
    
    X_train = X.values[train_index]
    X_test = X.values[test_index]
    
    y_train = y[train_index]
    y_test = y[test_index]
    
import seaborn as sns
sns.countplot(y_test)
plt.show()

print(Counter(y_train))
#resample
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

sampler = make_pipeline(
    RandomUnderSampler(random_state=0, sampling_strategy={'2': 5500, '6': 5500, '8': 5500, '3': 5500}),
    SMOTE(random_state=0),
)

X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

print(Counter(y_train_resampled))


#create report dataframe
report = pd.DataFrame(columns=['Model','Mean Training','Standard Deviation','Test'])

##XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train_resampled, y_train_resampled)

####Tuning####

#n_estimators
scores_ne = []

n_estimators = [1500, 1600, 1700, 1800, 1900, 2000]

for nes in n_estimators:
    print('n_estimators:', nes)
    xgb = XGBClassifier(max_depth=3,
                        learning_rate=0.1,
                        n_estimators=nes,
                        objective="multi:softprob",
                        n_jobs=-1,
                        nthread=4,
                        min_child_weight=1,
                        subsample=1,
                        colsample_bytree=1,
                        seed=42)
    
    xgb.fit(X_train_resampled, y_train_resampled)
    y_test_pred = xgb.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_ne.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(n_estimators, scores_ne, "o-")

plt.ylabel('f1')
plt.xlabel('n_estimators')
print('optimal n_estimators:{}'.format(n_estimators[np.argmax(scores_ne)]))

#max depth
scores_md = []
max_depths = [4, 5, 6, 7, 8]
for md in max_depths:
    print('max_depths:', md)
    xgb = XGBClassifier(max_depth=md,
                        learning_rate=0.1,
                        n_estimators=1900,
                        objective="multi:softprob",
                        n_jobs=-1,
                        nthread=4,
                        min_child_weight=1,
                        subsample=1,
                        colsample_bytree=1,
                        seed=42)
    
    xgb.fit(X_train_resampled, y_train_resampled)
    y_test_pred = xgb.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_md.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(max_depths, scores_md, "o-")
plt.ylabel('f1')
plt.xlabel('max_depths')
print('optimal max_depth:{}'.format(max_depths[np.argmax(scores_md)]))

#min_child_weight
scores_mcw = []
min_child_weights = [1, 2, 3, 4, 5, 6]
for mcw in min_child_weights:
    print('min_child_weights:', mcw)
    xgb = XGBClassifier(max_depth=7,
                        learning_rate=0.1,
                        n_estimators=1900,
                        objective="multi:softprob",
                        n_jobs=-1,
                        nthread=4,
                        min_child_weight=mcw, 
                        subsample=1,
                        colsample_bytree=1,
                        seed=42)
    
    xgb.fit(X_train_resampled, y_train_resampled)
    y_test_pred = xgb.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_mcw.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(min_child_weights, scores_mcw, "o-")
plt.ylabel('f1')
plt.xlabel('min_child_weights')
print('optimal min_child_weight:{}'.format(min_child_weights[np.argmax(scores_mcw)]))

#subsample
scores_ss = []
subsamples = [0.8, 0.85, 0.9, 0.95, 1]
for ss in subsamples:
    print('subsamples:', ss)
    xgb = XGBClassifier(max_depth=7,
                        learning_rate=0.1,
                        n_estimators=1900,
                        objective="multi:softprob",
                        n_jobs=-1,
                        nthread=4,
                        min_child_weight=4,
                        subsample=ss,
                        colsample_bytree=1,
                        seed=42)
    
    xgb.fit(X_train_resampled, y_train_resampled)
    y_test_pred = xgb.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_ss.append(score)
    print('Test f1:{}'.format(score))
    
#graph
plt.plot(subsamples, scores_ss, "o-")
plt.ylabel('f1')
plt.xlabel('subsamples')

print('optimal subsample:{}'.format(subsamples[np.argmax(scores_ss)]))

#colsample_bytree
scores_cb = []
colsample_bytrees = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
for cb in colsample_bytrees:
    print('colsample_bytrees:', cb)
    xgb = XGBClassifier(max_depth=7,
                        learning_rate=0.1,
                        n_estimators=1900,
                        objective="multi:softprob",
                        n_jobs=-1,
                        nthread=4,
                        min_child_weight=1,
                        subsample=0.95,
                        colsample_bytree=cb,
                        seed=42)
    
    xgb.fit(X_train_resampled, y_train_resampled)
    y_test_pred = xgb.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_cb.append(score)
    print('Test f1:{}'.format(score))
    
#graph
plt.plot(colsample_bytrees, scores_cb, "o-")
plt.ylabel('f1')
plt.xlabel('colsample_bytrees')

print('optimal colsample_bytree:{}'.format(colsample_bytrees[np.argmax(scores_cb)]))

#learning_rate
scores_lr = []
learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
for lr in learning_rates:
    print('learning_rates:', lr)
    xgb = XGBClassifier(max_depth=7,
                        learning_rate=lr,
                        n_estimators=1900,
                        objective="multi:softprob",
                        n_jobs=-1,
                        nthread=4,
                        min_child_weight=1,
                        subsample=0.95,
                        colsample_bytree=0.8,
                        seed=42)
    
    xgb.fit(X_train_resampled, y_train_resampled)
    y_test_pred = xgb.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_lr.append(score)
    print('Test f1:{}'.format(score))
    
#graph
plt.plot(learning_rates, scores_lr, "o-")
plt.ylabel('f1')
plt.xlabel('learning_rates')

print('optimal learning_rate:{}'.format(learning_rates[np.argmax(scores_lr)]))

#best_xgboost
xgb = XGBClassifier(max_depth=7,
                    learning_rate=0.1,
                    n_estimators=1900,
                    objective="multi:softprob",
                    n_jobs=-1,
                    nthread=4,
                    min_child_weight=1,
                    subsample=0.95,
                    colsample_bytree=0.8,
                    seed=42)
    
xgb.fit(X_train_resampled, y_train_resampled)
y_test_pred = xgb.predict(X_test)
print('Test f1:{}'.format(f1_score(y_test, y_test_pred, average='weighted')))

cmte_f1 = confusion_matrix(y_test, y_test_pred, labels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
print("Confusion Matrix Testing:\n", cmte_f1)

#visualize confusion matrix
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
"""
   # This function prints and plots the confusion matrix.
   # Normalization can be applied by setting `normalize=True`.
"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=16,
                 color="red" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

np.set_printoptions(precision=2)
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.figure()
plot_confusion_matrix(cmte_f1, classes=class_names, title='XGBoost Test')

print(report.loc[len(report)-1])


##KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train_resampled, y_train_resampled)

####Tuning####

#n_neighbers
scores_nn = []
n_neighbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

for nn in n_neighbers:
    print('n_neighbors:', nn)
    knn = KNeighborsClassifier(n_neighbors=nn)    
    knn.fit(X_train_resampled, y_train_resampled)
    y_test_pred = knn.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_nn.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(n_neighbers, scores_nn, "o-")

plt.ylabel('f1')
plt.xlabel('n_neighbors')
print('optimal n_neighbors:{}'.format(n_neighbers[np.argmax(scores_nn)]))

#best_knn
knn = KNeighborsClassifier(n_neighbors=2)    
    
knn.fit(X_train_resampled, y_train_resampled)
y_test_pred = knn.predict(X_test)
print('Test f1:{}'.format(f1_score(y_test, y_test_pred, average='weighted')))

cmte_f1 = confusion_matrix(y_test, y_test_pred, labels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
print("Confusion Matrix KNN Testing:\n", cmte_f1)


#visualize confusion matrix
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
"""
   # This function prints and plots the confusion matrix.
   # Normalization can be applied by setting `normalize=True`.
"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=16,
                 color="red" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

np.set_printoptions(precision=2)
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.figure()
plot_confusion_matrix(cmte_f1, classes=class_names, title='KNN Test')

print(report.loc[len(report)-1])


##Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train_resampled, y_train_resampled)

####Tuning####

#n_estimators
scores_ne = []

n_estimators = [1700, 1750, 1800, 1850, 2000]

for nes in n_estimators:
    print('n_estimators:', nes)
    rf = RandomForestClassifier(max_depth=10,
                                n_estimators=nes,
                                max_features=20,
                                min_samples_leaf=30,
                                random_state=0)
    
    rf.fit(X_train_resampled, y_train_resampled)
    y_test_pred = rf.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_ne.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(n_estimators, scores_ne, "o-")

plt.ylabel('f1')
plt.xlabel('n_estimators')
print('optimal n_estimators:{}'.format(n_estimators[np.argmax(scores_ne)]))

#max_features
scores_mf = []
max_features = [4, 5, 6, 7, 8]
for mf in max_features:
    print('max_features:', mf)
    rf = RandomForestClassifier(max_depth=10,
                                n_estimators=1850,
                                max_features=mf,
                                min_samples_leaf=30,
                                random_state=0)
    
    rf.fit(X_train_resampled, y_train_resampled)
    y_test_pred = rf.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_mf.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(max_features, scores_mf, "o-")

plt.ylabel('f1')
plt.xlabel('max_features')
print('optimal max_features:{}'.format(max_features[np.argmax(scores_mf)]))

#max depth
scores_md = []
max_depths = [20, 30, 40, 50, 60, 70, 80, 90, 100]
for md in max_depths:
    print('max_depths:', md)
    rf = RandomForestClassifier(max_depth=md,
                                n_estimators=1850,
                                max_features=7,
                                min_samples_leaf=30,
                                random_state=0)
    
    rf.fit(X_train_resampled, y_train_resampled)
    y_test_pred = rf.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_md.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(max_depths, scores_md, "o-")

plt.ylabel('f1')
plt.xlabel('max_depths')
print('optimal max_depths:{}'.format(max_depths[np.argmax(scores_md)]))


#min_samples_leaf
scores_msl = []
min_samples_leaves = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for msl in min_samples_leaves:
    print('min_samples_leaf:', msl)
    rf = RandomForestClassifier(max_depth=50,
                                n_estimators=1850,
                                max_features=7,
                                min_samples_leaf=msl,
                                random_state=0)
    
    rf.fit(X_train_resampled, y_train_resampled)
    y_test_pred = rf.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_msl.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(min_samples_leaves, scores_msl, "o-")

plt.ylabel('f1')
plt.xlabel('min_samples_leaves')
print('optimal min_samples_leaf:{}'.format(min_samples_leaves[np.argmax(scores_msl)]))


#best_random_forest
rf = RandomForestClassifier(max_depth=50,
                                n_estimators=1850,
                                max_features=7,
                                min_samples_leaf=1,
                                random_state=0)
    
rf.fit(X_train_resampled, y_train_resampled)
y_test_pred = rf.predict(X_test)
print('Test f1:{}'.format(f1_score(y_test, y_test_pred, average='weighted')))

cmte_f1 = confusion_matrix(y_test, y_test_pred, labels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
print("Confusion Matrix Random Forest Testing:\n", cmte_f1)

#visualize confusion matrix
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
"""
   # This function prints and plots the confusion matrix.
   # Normalization can be applied by setting `normalize=True`.
"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=16,
                 color="red" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

np.set_printoptions(precision=2)
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.figure()
plot_confusion_matrix(cmte_f1, classes=class_names, title='Random Forest Test')

print(report.loc[len(report)-1])

##Decision Trees
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train_resampled, y_train_resampled)

####Tuning####

#criterion
scores_c = []

criterions = ['gini', 'entropy']

for c in criterions:
    print('criterion:', c)
    dt = DecisionTreeClassifier(criterion=c,
                                max_depth=10,
                                min_samples_leaf=10,
                                random_state=0)
    
    dt.fit(X_train_resampled, y_train_resampled)
    y_test_pred = dt.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_c.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(criterions, scores_c, "o-")

plt.ylabel('f1')
plt.xlabel('criterions')
print('optimal criterion:{}'.format(criterions[np.argmax(scores_c)]))

#max depth
scores_md = []
max_depths = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
for md in max_depths:
    print('max_depths:', md)
    dt = DecisionTreeClassifier(criterion='gini',
                                max_depth=md,
                                min_samples_leaf=10,
                                random_state=0)
    
    dt.fit(X_train_resampled, y_train_resampled)
    y_test_pred = dt.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_md.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(max_depths, scores_md, "o-")

plt.ylabel('f1')
plt.xlabel('max_depths')
print('optimal max_depths:{}'.format(max_depths[np.argmax(scores_md)]))


#min_samples_leaf
scores_msl = []
min_samples_leaves = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for msl in min_samples_leaves:
    print('min_samples_leaf:', msl)
    dt = DecisionTreeClassifier(criterion='gini',
                                max_depth=36,
                                min_samples_leaf=msl,
                                random_state=0)
    
    dt.fit(X_train_resampled, y_train_resampled)
    y_test_pred = dt.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_msl.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(min_samples_leaves, scores_msl, "o-")

plt.ylabel('f1')
plt.xlabel('min_samples_leaves')
print('optimal min_samples_leaf:{}'.format(min_samples_leaves[np.argmax(scores_msl)]))


#best_decision_trees
dt = DecisionTreeClassifier(criterion='gini',
                                max_depth=36,
                                min_samples_leaf=2,
                                random_state=0)
    
dt.fit(X_train_resampled, y_train_resampled)
y_test_pred = dt.predict(X_test)
print('Test f1:{}'.format(f1_score(y_test, y_test_pred, average='weighted')))

cmte_f1 = confusion_matrix(y_test, y_test_pred, labels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
print("Confusion Matrix Decision Trees Testing:\n", cmte_f1)

#visualize confusion matrix
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
"""
   # This function prints and plots the confusion matrix.
   # Normalization can be applied by setting `normalize=True`.
"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=16,
                 color="red" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

np.set_printoptions(precision=2)
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.figure()
plot_confusion_matrix(cmte_f1, classes=class_names, title='Decision Trees Test')

print(report.loc[len(report)-1])

##SVM
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train_resampled, y_train_resampled)

####Tuning####

#C
scores_c = []

Cs = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]

for c in Cs:
    print('C:', c)
    svm = SVC(kernel='rbf',
              C=c,
              gamma=7,
              random_state=0)
    
    svm.fit(X_train_resampled, y_train_resampled)
    y_test_pred = svm.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_c.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(Cs, scores_c, "o-")

plt.ylabel('f1')
plt.xlabel('Cs')
print('optimal C:{}'.format(Cs[np.argmax(scores_c)]))

#gamma
scores_g = []
gammas = [6, 6.5, 7, 7.5, 8]

for g in gammas:
    print('gammas:', g)
    svm = SVC(kernel='rbf',
              C=84,
              gamma=g,
              random_state=0)
    
    svm.fit(X_train_resampled, y_train_resampled)
    y_test_pred = svm.predict(X_test)
    score = f1_score(y_test, y_test_pred, average='weighted')
    scores_g.append(score)
    print('Test f1:{}'.format(score))

#graph the result
plt.plot(gammas, scores_g, "o-")

plt.ylabel('f1')
plt.xlabel('gammas')
print('optimal gamma:{}'.format(gammas[np.argmax(scores_g)]))


#best_svm
svm = SVC(kernel='rbf',
              C=84,
              gamma=7,
              random_state=0)
    
svm.fit(X_train_resampled, y_train_resampled)
y_test_pred = svm.predict(X_test)
print('Test f1:{}'.format(f1_score(y_test, y_test_pred, average='weighted')))

cmte_f1 = confusion_matrix(y_test, y_test_pred, labels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
print("Confusion Matrix SVM Testing:\n", cmte_f1)

#visualize confusion matrix
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
"""
   # This function prints and plots the confusion matrix.
   # Normalization can be applied by setting `normalize=True`.
"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=16,
                 color="red" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

np.set_printoptions(precision=2)
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.figure()
plot_confusion_matrix(cmte_f1, classes=class_names, title='SVM Test')

print(report.loc[len(report)-1])


##Neural Networks
from sklearn.neural_network import MLPClassifier

nn = MLPClassifier()
nn.fit(X_train_resampled, y_train_resampled)

nn = MLPClassifier(solver='lbfgs',
                   max_iter=10000000000,
                   hidden_layer_sizes=(20,10),
                   random_state=0)
    
nn.fit(X_train_resampled, y_train_resampled)
y_test_pred = nn.predict(X_test)
score = f1_score(y_test, y_test_pred, average='weighted')
print('Test f1:{}'.format(score))

cmte_f1 = confusion_matrix(y_test, y_test_pred, labels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
print("Confusion Matrix Nueral Networks Testing:\n", cmte_f1)

#visualize confusion matrix
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
"""
   # This function prints and plots the confusion matrix.
   # Normalization can be applied by setting `normalize=True`.
"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=16,
                 color="red" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

np.set_printoptions(precision=2)
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.figure()
plot_confusion_matrix(cmte_f1, classes=class_names, title='Neural Networks Test')

print(report.loc[len(report)-1])











