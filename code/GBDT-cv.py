# -*- coding: utf-8 -*-
"""
CPSC 532M project
Yihan, John-Hose, Teyden
"""

from utility import subsample
from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


def GBDTEnsemble():
    """ Train an ensemble of trees and report the accuracy as they did in the paper
    """
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
    auc = np.empty([10,], dtype=float)
    bscore = np.empty([10,], dtype=float)
    for train_index, test_index in kf.split(X):
        print("Fold:", j)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Feature selection
        #features = featureSelectionChi(X_train,y_train,30,50)
        features = featureSelectionELAS(X_train,y_train,31)
        #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),10)
        #features = np.arange(m)

        # Subsampling data
        X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
        training, label = subsample(X_combined, t=30)
 
        # Train an ensemble of 30 decision trees
        clf = [None]*30
        for i in range(30):
            clf[i] = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
            clf[i].fit(training[i][:,features],label[i])
            #clf[i].fit(training[i],label[i])

        # Prediction
        n = X_test.shape[0]
        # calculting the average probabilty of each class
        pred_prob = np.zeros((n,2))
        for i in range(30):
            pred_prob += clf[i].predict_proba(X_test[:,features])
            #pred_prob += clf[i].predict_proba(X_test)
            
        pred_prob = pred_prob/30
        # Pick the class with the greastest probability to be the prediction
        pred = np.argmax(pred_prob,axis=1)
        y_score = pred_prob[:,1]
    
        # Report accuracy and draw ROC curve
        auc[j-1] = drawROC(y_test, y_score)
        bscore[j-1] = balanced_accuracy_score(y_test, pred)
        score = sum(pred==y_test)/n
        print("Accuracy is:", score)
        print("Balanced Accuracy is:", bscore[j-1])
        accu[j-1] = score
        j = j+1
    print("Average accuracy is:",sum(accu)/10)
    print("Average AUC is:",sum(auc)/10)
    print("Average balanced accuracy is:",sum(bscore)/10)

GBDTEnsemble()