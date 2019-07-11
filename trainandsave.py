#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Shashi Bhushan <sbhushan1@outlook.com>

# Standard scientific Python imports
import numpy as np

import pickle

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
# fetch original mnist dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def main():
    # it creates openml folder in your root project folder
    mnist = fetch_openml('mnist_784', version=1, data_home='./')

    data = mnist.data
    target = mnist.target

    # show_random_digits(data, target)

    # ---------------- classification begins -----------------
    random_index = np.random.choice(data.shape[0], 1000)

    # Random dataset classification
    X = data[random_index]
    y = target[random_index]

    # Full dataset classification
    # X = data
    # y = target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

    # Create a SVM Classifier
    svm_classifier = svm.SVC(C=5, gamma=0.5)

    # Training SVM

    print("Training SVM")
    svm_classifier.fit(X_train, y_train)

    pickle.dump(svm_classifier, open('svm.sav', 'wb'))

    #y_predicted = svm_classifier.predict(X_test)
    print("Saved Model")


if __name__ == "__main__":
    main()
