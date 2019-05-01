# -*- coding: utf-8 -*-
"""
CPSC 532M project
Yihan, John-Hose, Teyden
"""

from utility import subsample
from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
from sklearn.model_selection import train_test_split
from scipy import stats
import xgboost as xgb
from sklearn import cross_validation
import numpy as np


def XGBTEnsemble():
    """ Train an ensemble of trees and report the accuracy as they did in the paper
    """
    param = {'nthread':5, 'booster': 'gbtree', 'max_depth':3, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric': 'error', 'colsample_bytree':0.8, 'lambda':0.5, 'lambda_bias': 0.5, 'subsample' : 1}
    num_round = 5
    
    pathData = '../data/canbind-clean-aggregated-data.with-id.csv'
    pathLabel = '../data/targets.csv'
    # read data and chop the header
    X = np.genfromtxt(pathData, delimiter=',')
    y = np.genfromtxt(pathLabel, delimiter=',')[:,1]
    X = X[1:,]
    X = X[:,2:]
    
    n,m = X.shape
    kf = cross_validation.KFold(n, n_folds=10, shuffle=True)

    j=1
    accu = np.empty([10,], dtype=float)
    for train_index, test_index in kf:
        print("Fold:", j)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Feature selection
        #features = featureSelectionChi(X_train,y_train,15,25)
        features = featureSelectionELAS(X_train,y_train,50)
        #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),10)

        # Subsampling data
        X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
        training, label = subsample(X_combined, t=30)
 
        # Train an ensemble of 30 decision trees
        clf = [None]*30
        for i in range(30):
            dtrain = xgb.DMatrix(training[i][:,features], label=label[i])      
            bst = xgb.train(param, dtrain, num_round)
            clf[i] = bst

        # Prediction
        n = X_test.shape[0]
        dtest = xgb.DMatrix(X_test[:,features], label=y_test)
        # calculting the average probabilty of each class
        pred_prob = np.zeros((n,))
        for i in range(30):
            pred_prob += clf[0].predict(dtest)
            
        pred_prob = pred_prob/30
        # Pick the class with the greastest probability to be the prediction
        pred = (pred_prob>=0.5)
        y_score = pred_prob
    
        # Report accuracy and draw ROC curve
        drawROC(y_test, y_score)
        score = sum(pred==y_test)/n
        print("Accuracy is:", score)
        accu[j-1] = score
        j = j+1
    print("Average accuracy is:",sum(accu)/10)

XGBTEnsemble()
