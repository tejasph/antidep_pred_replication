# -*- coding: utf-8 -*-
"""
CPSC 532M project
Yihan, John-Hose, Teyden
"""

from utility import subsample
from utility import featureSelectionChi, featureSelectionELAS, featureSelectionAgglo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
    


def plott():
    """ Train an ensemble of trees and report the accuracy as they did in the paper
    """
    pathData = '../data/X_stard_overlapping_v2.csv'
    pathLabel = '../data/y-stard-overlapping-targets.csv'
    testData = '../data/X_canbind-overlapping.csv'
    testLabel = '../data/y_canbind_targets.csv'
    # read data and chop the header
    X_train = np.genfromtxt(pathData, delimiter=',')[1:,1:]
    y_train = np.genfromtxt(pathLabel, delimiter=',')[1:,2]
    
    X_test = np.genfromtxt(testData, delimiter=',')[1:,1:]
    y_test = np.genfromtxt(testLabel, delimiter=',')[1:,2]
    
    _,m = X_train.shape
    # Feature selection
    features = range(m)
    #features = featureSelectionChi(X_train,y_train,30,50)
    #features = featureSelectionELAS(X_train,y_train,31)
    #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),50)
        
    # Subsampling data
    X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
    training, label = subsample(X_combined, t=30)

    # Train an ensemble of 30 decision trees
    clf = [None]*30
    for i in range(30):
        clf[i] = RandomForestClassifier(n_estimators=50, n_jobs = 5)
        clf[i].fit(training[i][:,features],label[i])

    # Prediction
    n = X_test.shape[0]
    # calculting the average probabilty of each class
    pred_prob = np.zeros((n,2))
    for i in range(30):
        pred_prob += clf[i].predict_proba(X_test[:,features])
            
    pred_prob = pred_prob/30
    # Pick the class with the greastest probability to be the prediction
    y_score = pred_prob[:,1]
    
    # Report accuracy and draw ROC curve
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    rf, = plt.plot(fpr, tpr)
    
     # Feature selection
    #features = featureSelectionChi(X_train,y_train,30,50)
    #features = featureSelectionELAS(X_train,y_train,31)
    #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),50)

    # Subsampling data
    X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
    training, label = subsample(X_combined, t=30)
 
    # Train an ensemble of 30 decision trees
    clf = [None]*30
    for i in range(30):
        clf[i] = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
        clf[i].fit(training[i][:,features],label[i])
        
     # Prediction
    n = X_test.shape[0]
    # calculting the average probabilty of each class
    pred_prob = np.zeros((n,2))
    for i in range(30):
        pred_prob += clf[i].predict_proba(X_test[:,features])
            
    pred_prob = pred_prob/30
    # Pick the class with the greastest probability to be the prediction
    y_score = pred_prob[:,1]
    
    # Report accuracy and draw ROC curve
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    gdbt, = plt.plot(fpr, tpr)
    
    param = {'nthread':5, 'booster': 'gbtree', 'max_depth':3, 'eta':0.1, 'silent':1, 'objective':'binary:logistic', 'eval_metric': 'error', 'colsample_bytree':0.8, 'lambda':0.5, 'lambda_bias': 0.5, 'subsample' : 1}
    num_round = 5
    # Feature selection
    #features = featureSelectionChi(X_train,y_train,30,50)
    #features = featureSelectionELAS(X_train,y_train,31)
    #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),50)

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
    y_score = pred_prob
    
    # Report accuracy and draw ROC curve
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    xg, = plt.plot(fpr, tpr)
    
    # Feature selection
    #features = featureSelectionChi(X_train,y_train,30,100)
    #features = featureSelectionELAS(X_train,y_train,31)
    #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),50)

    # Subsampling data
    X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
    training, label = subsample(X_combined, t=30)
        
    # Train an ensemble of 30 decision trees
    clf = [None]*30
    for i in range(30):
        clf[i] = LogisticRegression()
        clf[i].fit(training[i][:,features],label[i])
        
     # Prediction
    n = X_test.shape[0]
    # calculting the average probabilty of each class
    pred_prob = np.zeros((n,2))
    for i in range(30):
        pred_prob += clf[i].predict_proba(X_test[:,features])
            
    pred_prob = pred_prob/30
    # Pick the class with the greastest probability to be the prediction
    y_score = pred_prob[:,1]
    
    # Report accuracy and draw ROC curve
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    logis, = plt.plot(fpr, tpr)
    
    # Feature selection
    #features = featureSelectionChi(X_train,y_train,15,25)
    #features = featureSelectionELAS(X_train,y_train,31)
    #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),50)

    #Subsampling data
    X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
    training, label = subsample(X_combined, t=30)
 
    # Train an ensemble of 30 decision trees
    clf = [None]*30
    for i in range(30):
        clf[i] = SGDClassifier(loss='log', penalty='elasticnet', max_iter=500, alpha=0.01, l1_ratio=0.15)
        clf[i].fit(training[i][:,features],label[i])
        
    # Prediction
    n = X_test.shape[0]
    # calculting the average probabilty of each class
    pred_prob = np.zeros((n,2))
    for i in range(30):
        pred_prob += clf[i].predict_proba(X_test[:,features])
            
    pred_prob = pred_prob/30
    # Pick the class with the greastest probability to be the prediction
    y_score = pred_prob[:,1]
    
    # Report accuracy and draw ROC curve
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    elas, = plt.plot(fpr, tpr)
    
    # Feature selection
    #features = featureSelectionChi(X_train,y_train,30,75)
    #features = featureSelectionELAS(X_train,y_train,31)
    #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),50)
        
    # Subsampling data
    X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
    training, label = subsample(X_combined, t=10)
        
    # Train an ensemble of 30 decision trees
    clf = [None]*10
    for i in range(10):
        clf[i] = MLPClassifier(solver='sgd',alpha=1e-3,hidden_layer_sizes=(50,)
        ,max_iter=2000,learning_rate='adaptive')
        #clf[i].fit(training[i][:,features],label[i])
        clf[i].fit(training[i][:,features],label[i])
    
    # Prediction
    n = X_test.shape[0]
    # calculting the average probabilty of each class
    pred_prob = np.zeros((n,2))
    for i in range(10):
        #pred_prob += clf[i].predict_proba(X_test[:,features])
        pred_prob += clf[i].predict_proba(X_test[:,features])
            
    pred_prob = pred_prob/10
    # Pick the class with the greastest probability to be the prediction
    y_score = pred_prob[:,1]
    
    # Report accuracy and draw ROC curve
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    NN, = plt.plot(fpr, tpr)
    
    # Feature selection
    #features = range(m)
    #features = featureSelectionChi(X_train,y_train,30,100)
    #features = featureSelectionELAS(X_train,y_train,31)
    #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),50)

    # Subsampling data
    X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
    training, label = subsample(X_combined, t=30)
 
    # Train an ensemble of 30 decision trees
    clf = [None]*30
    for i in range(30):
        clf[i] = SVC(probability=True)
        clf[i].fit(training[i][:,features],label[i])
        
    # Prediction
    n = X_test.shape[0]
    # calculting the average probabilty of each class
    pred_prob = np.zeros((n,2))
    for i in range(30):
        pred_prob += clf[i].predict_proba(X_test[:,features])
            
    pred_prob = pred_prob/30
    # Pick the class with the greastest probability to be the prediction
    y_score = pred_prob[:,1]
    
    # Report accuracy and draw ROC curve
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    svm, = plt.plot(fpr, tpr)
    
    # Feature selection
    #features = range(m)
    #features = featureSelectionChi(X_train,y_train,30,100)
    #features = featureSelectionELAS(X_train,y_train,31)
    #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),50)

    # Subsampling data
    X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
    training, label = subsample(X_combined, t=50)
 
    # Train an ensemble of 30 decision trees
    clf = [None]*50
    for i in range(10):
        clf[i] = SGDClassifier(loss='log', penalty='elasticnet', max_iter=50, alpha=0.01, l1_ratio=0.15)
        clf[i].fit(training[i][:,features],label[i])
    for i in range(10,20):
        clf[i] = RandomForestClassifier(n_estimators=50, n_jobs = 5)
        clf[i].fit(training[i][:,features],label[i])
    for i in range(20,30):
        clf[i] = MLPClassifier(solver='sgd',alpha=1e-3,hidden_layer_sizes=(50,)
        ,max_iter=2000,learning_rate='adaptive')
        #clf[i].fit(training[i][:,features],label[i])
        clf[i].fit(training[i][:,features],label[i])
    for i in range(30,40):
        clf[i] = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=10, random_state=0)
        clf[i].fit(training[i][:,features],label[i])
    for i in range(40,50):
        clf[i] = SVC(probability=True)
        clf[i].fit(training[i][:,features],label[i])
        
    # Prediction
    n = X_test.shape[0]
    # calculting the average probabilty of each class
    pred_prob = np.zeros((n,2))
    for i in range(50):
        pred_prob += clf[i].predict_proba(X_test[:,features])
        
    pred_prob = pred_prob/50
    # Pick the class with the greastest probability to be the prediction
    print(pred_prob)
    y_score = pred_prob[:,1]
    
    # Report accuracy and draw ROC curve
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    ultra, = plt.plot(fpr, tpr)
    plt.title('ROC curve for different models on Canbind')
    plt.legend([rf,gdbt,xg,logis,elas,NN,svm,ultra],["Random Forest","Gradient Descent Boosting Tree",'l_2-penalized Logistic Regression','Elastic Net', "Neural Network","SVM","Ultra Ensemble"],loc=4)
    plt.tight_layout()
    plt.savefig("../figs/comparisonCan.eps",format="eps")
    
plott()
