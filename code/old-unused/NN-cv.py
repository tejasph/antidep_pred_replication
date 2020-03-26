# -*- coding: utf-8 -*-
"""
CPSC 532M project
Yihan, John-Hose, Teyden
"""

from __future__ import print_function
import numpy as np
from utility import subsample
from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold


def NNEnsemble():
    """ Train an ensemble of trees and report the accuracy as they did in the paper
    """
    # read data and chop the header
    pathData = r'C:\Users\y374zhou\Documents\GitHub\antidep-project\code\data\X_lvl2_rem_qids01__final.csv'
    pathLabel = r'C:\Users\y374zhou\Documents\GitHub\antidep-project\code\data\y_lvl2_rem_qids01__final.csv'
    # read data and chop the header
    X = np.genfromtxt(pathData, delimiter=',')
    y = np.genfromtxt(pathLabel, delimiter=',')[1:,1]
    X = X[1:,1:]
    
    n,m = X.shape
    kf = KFold(n_splits=10, shuffle=True)

    j=1
    accu = np.empty([10,], dtype=float)
    for train_index, test_index in kf.split(X):
        
        print("Fold:", j)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Feature selection
        #features = featureSelectionChi(X_train,y_train,75,100)
        #features = featureSelectionELAS(X_train,y_train,75)
        #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),10)
        
        # Subsampling data
        X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
        training, label = subsample(X_combined, t=30)
        
        # Train an ensemble of 30 decision trees
        clf = [None]*10
        for i in range(10):
            clf[i] = MLPClassifier(solver='sgd',alpha=1e-3,hidden_layer_sizes=(50,)
            ,max_iter=2000,learning_rate='adaptive')
            #clf[i].fit(training[i][:,features],label[i])
            clf[i].fit(training[i],label[i])

        # Prediction
        n = X_test.shape[0]
        # calculting the average probabilty of each class
        pred_prob = np.zeros((n,2))
        for i in range(10):
            #pred_prob += clf[i].predict_proba(X_test[:,features])
            pred_prob += clf[i].predict_proba(X_test)
            
        pred_prob = pred_prob/10
        # Pick the class with the greastest probability to be the prediction
        pred = np.argmax(pred_prob,axis=1)
        y_score = pred_prob[:,1]
    
        # Report accuracy and draw ROC curve
        drawROC(y_test, y_score)
        score = sum(pred==y_test)/n
        print("Accuracy is:", score)
        accu[j-1] = score
        j = j+1
    print("Average accuracy is:",sum(accu)/10)

NNEnsemble()
