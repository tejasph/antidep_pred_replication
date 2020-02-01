# -*- coding: utf-8 -*-
"""
CPSC 532M project
Yihan, John-Hose, Teyden
"""

from utility import subsample
from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import time

# Simplified imputation
    ##pathData = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\532M_project\data\teyden-git\code\data-cleaning\final_datasets\to_run_experiment_simple_imput\X_lvl2_rem_qids01__final_simple_imputation.csv'
    ##pathLabel = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\532M_project\data\teyden-git\code\data-cleaning\final_datasets\to_run_experiment_simple_imput\y_lvl2_rem_qids01__final_simple_imputation.csv'
    # Full features
pathData = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\code\data-cleaning\final_datasets\to_run_20201016\1_Replication\X_lvl2_rem_qids01__final.csv'
pathLabel = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\code\data-cleaning\final_datasets\to_run_20201016\1_Replication\y_lvl2_rem_qids01__final.csv'
    


def RandomForrestEnsemble():
    """ Train an ensemble of trees and report the accuracy as they did in the paper
    """
    
    #pathData = r'C:\Users\y374zhou\Documents\GitHub\antidep-project\code\data\X_lvl2_rem_qids01__final.csv'
    #pathLabel = r'C:\Users\y374zhou\Documents\GitHub\antidep-project\code\data\y_lvl2_rem_qids01__final.csv'
    
    
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
        ##print("Fold:", j)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Feature selection
        #features = featureSelectionChi(X_train,y_train,30,50)
        features = featureSelectionELAS(X_train,y_train,31)
        #features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),20)
        #features = np.arange(m)
        
        # Subsampling data
        X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
        training, label = subsample(X_combined, t=30)
 
        # Train an ensemble of 30 decision trees
        clf = [None]*30
        for i in range(30):
            clf[i] = RandomForestClassifier(n_estimators=50, n_jobs = 5)
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
        ##print("Accuracy is:", score)
        ##print("Balanced Accuracy is:", bscore[j-1])
        accu[j-1] = score
        j = j+1
    
    avg_acc = sum(accu)/10
    avg_auc = sum(auc)/10
    avg_bacc = sum(bscore)/10
    
    
    ##print("Average accuracy is:",sum(accu)/10)
    ##print("Average AUC is:",sum(auc)/10)
    ##print("Average balanced accuracy is:",sum(bscore)/10)
    
    return avg_acc, avg_auc, avg_bacc

runs = 300
avg_acc = 0
avg_auc = 0 
avg_bacc = 0


for i in range(runs):
    a, b, c = RandomForrestEnsemble()
    avg_acc = avg_acc + a
    avg_auc = avg_auc + b
    avg_bacc = avg_bacc + c
    print("Finished run: " + str(i) + " of " + str(runs) + "\n")


pathResults = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\AJP Submission\model_results.txt'

f = open(pathResults, "a")
f.write("Writing results of running Random Forest model with no feature selection\n")
f.write("X is: " + pathData + "\n")
f.write("y is: " + pathLabel + "\n")
f.write("After " + str(runs) + "runs of 10 fold CV, average accuracy is:" + str(avg_acc/runs) +"\n")
f.write("After " + str(runs) + "runs of 10 fold CV, average AUC is:" + str(avg_auc/runs) +"\n")
f.write("After " + str(runs) + "runs of 10 fold CV, average balanced accuracy is:" + str(avg_bacc/runs) +"\n")
f.write("----------------------------------------------------------\n")
f.close()

#print("After 10 runds of 10 fold CV, average accuracy is:" + str(avg_acc/10))
#print("After 10 runds of 10 fold CV, average AUC is:" + str(avg_auc/10))
#print("After 10 runds of 10 fold CV, average balanced accuracy is:" + str(avg_bacc/10))
