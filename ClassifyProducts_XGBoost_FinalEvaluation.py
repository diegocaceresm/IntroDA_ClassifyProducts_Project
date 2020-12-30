# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 22:17:50 2020

@author: cdieg
"""

import os
import numpy as np
import pandas as pd
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
print("Interesting use of lambda", y, "\n")

X_ori = df.drop('target', axis=1)

# normalize
X = (X_ori-X_ori.min())/(X_ori.max()-X_ori.min())
# print(X.head(6))


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

##################
#  Data Balance  #
##################

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)


#############
#  XGBoost  #
#############

from xgboost import XGBClassifier

xgb = XGBClassifier(class_weight=class_weights)
xgb.fit(X_train, y_train)

# best_xgboost
xgb = XGBClassifier(max_depth=7,
                    learning_rate=0.1,
                    n_estimators=1900,
                    objective="multi:softprob",
                    n_jobs=-1,
                    nthread=4,
                    min_child_weight=4,
                    subsample=0.95,
                    colsample_bytree=0.8,
                    seed=42)

xgb.fit(X_train, y_train)
y_test_pred = xgb.predict(X_test)
print('Test f1:{}'.format(f1_score(y_test, y_test_pred, average='weighted')))

cmte_f1 = confusion_matrix(y_test, y_test_pred, labels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
print("Confusion Matrix Testing:\n", cmte_f1)

# visualize confusion matrix
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
       # This function prints and plots the confusion matrix.
       # Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

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
plt.show()

