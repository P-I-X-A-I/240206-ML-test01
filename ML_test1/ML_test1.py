
print("ML test 01")

from re import A
from matplotlib import markers
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
import sklearn
from sklearn.model_selection import train_test_split


# generate test data
X, y = mglearn.datasets.make_forge() #X is ndarray(26, 2), y is ndarray(26,)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)
print(X_train.shape)
print(y_train.shape)

# test k-nearest-neighbor
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)

# setup knn
knn.fit(X_train, y_train)

# Classify
print(knn.predict(X_test))

# score the result
print("accuracy : {}".format(knn.score(X_test, y_test)))


# show decision boundary
fig, axes = plt.subplots(1, 3, figsize=(10, 3)) # figsize in inch...
print(axes)

for i in range(3):
    print("process : {}".format(i))
    clf = KNeighborsClassifier(n_neighbors=(i+1)*2).fit(X_train, y_train)
    # draw boundary
    mglearn.plots.plot_2d_separator(clf, X_train, fill=True, eps=0.5, ax=axes[i], alpha=.4)
    # draw point
    mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train, ax=axes[i])
    # setup graph
    axes[i].set_title("graph : {}".format((i+1)*2))
    axes[i].set_xlabel("feature 0")
    axes[i].set_ylabel("feature 1")
    
plt.show()


"""
# plot data to plt
mglearn.discrete_scatter(X[:,0], X[:, 1], y, markers=['o', '^'], c=['red', 'yellow'])

plt.legend(labels=["ClassA", "ClassB"], loc="lower right")
plt.xlabel("1st feature")
plt.ylabel("2nd feature")
plt.show()
"""

'''
# generate wave data
X, y = mglearn.datasets.make_wave(n_samples=40)
# print(X) : feature data
# print(y) : target

# plot data
plt.plot(X, y, 'x')
plt.ylim(-5, 5)
plt.xlim(-4, 4)
plt.xlabel("Feature")
plt.ylabel("Target")

plt.show()
'''


# test data loading
from sklearn.datasets import load_breast_cancer

cancer_dataset = load_breast_cancer()
print(cancer_dataset.keys())
print("*********")
print("num features:{}\n".format(len(cancer_dataset.feature_names)))
print(cancer_dataset.feature_names)
print("*********")
print(cancer_dataset.target_names)
print("*********")
print(cancer_dataset.data.shape)
print(cancer_dataset.target.shape)
# print(cancer_dataset.data[:20, :5])

# split data
Data_train, Data_test, label_train, label_test = train_test_split(
    cancer_dataset.data,
    cancer_dataset.target,
    stratify=cancer_dataset.target, # keep ratio of label
    random_state=11)

# try knn from 1 to 10
iter = range(1, 10)
acc_train = []
acc_test = []

for i in iter:
    # setup
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(Data_train, label_train)
    # record accuracy
    acc_train.append(clf.score(Data_train, label_train))
    # record accuracy 2
    acc_test.append(clf.score(Data_test, label_test))
    print("loop :", i)


# plot accuracy result
plt.clf()
plt.plot(iter, acc_train, label="train")
plt.plot(iter, acc_test, label="test")

plt.show()