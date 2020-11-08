import os
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

os.chdir("C:/Users/cdieg/OneDrive/Documents/MADS/Courses/q1/Introduction to Data Analytics in Business/Project")
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


##############################
#  CROSS VALIDATION - KNN    #
##############################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
print("______________________________")
print("KNN - Cross Validation","\n")
#create a new KNN model
for k in range(6, 14, 2):
    knnmodel = KNeighborsClassifier(n_neighbors=k)
    knnmodel.fit(X_train, y_train)
    cv_scores = cross_val_score(knnmodel, X_train, y_train, cv=10)
    cv_mean = np.mean(cv_scores)
    print("In the scenario of ",k, " Neighbors, the ACC CV scores =", cv_scores)
    print("For scenario",k," we have a mean ",cv_mean,"\n")
print("______________________________")

