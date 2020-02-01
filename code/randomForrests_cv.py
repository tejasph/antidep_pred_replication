# -*- coding: utf-8 -*-
"""
CPSC 532M project
Yihan, John-Hose, Teyden
"""

from utility import subsample
from utility import featureSelectionChi, featureSelectionELAS, drawROC, featureSelectionAgglo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import numpy as np


def RandomForrestEnsemble(pathData, pathLabel, f_select):
    """ Train an ensemble of trees and report the accuracy as they did in the paper
    """
    ##pathData = r'C:\Users\y374zhou\Documents\GitHub\antidep-project\code\data\X_lvl2_rem_qids01__final.csv'
    ##pathLabel = r'C:\Users\y374zhou\Documents\GitHub\antidep-project\code\data\y_lvl2_rem_qids01__final.csv'
    f = open(r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\results\testresults.txt', 'w')
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
    specificity = np.empty([10,], dtype=float)
    sensitivity = np.empty([10,], dtype=float)
    precision = np.empty([10,], dtype=float)
    f1 = np.empty([10,], dtype=float)
    features_n = np.empty([10,], dtype=float)
    features_importance =  np.empty([10,], dtype=float)
    
    for train_index, test_index in kf.split(X):
        print("Fold:", j)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Feature selection
        if f_select == "chi":
            features = featureSelectionChi(X_train,y_train,30,50)
        elif f_select == "elas":
            features = featureSelectionELAS(X_train,y_train,31)
        elif f_select == "agglo":
            features,_ = featureSelectionAgglo(np.append(y_train.reshape(-1,1),X_train , axis=1),20)
        elif f_select == "full":
            features = np.arange(m)
        
        # Array to store the importance of features, whether feature was used, and an int to stores number of features per classifier
        featureimportance = np.zeros(len(features))
        features_used = np.zeros(len(features))
        features_n_fold = 0
        
        # Subsampling data
        X_combined = np.append(y_train.reshape(-1,1),X_train,axis=1)
        training, label = subsample(X_combined, t=30)
 
        # Train an ensemble of 30 RF classifiers
        clf = [None]*30
        for i in range(30):
            clf[i] = RandomForestClassifier(n_estimators=50, n_jobs = -1)
            clf[i].fit(training[i][:,features],label[i])
            #clf[i].fit(training[i],label[i])

        # Prediction
        n = X_test.shape[0]
        # calculting the average probabilty of each class, as well as feature use/importance
        pred_prob = np.zeros((n,2))
        for i in range(30):
            pred_prob += clf[i].predict_proba(X_test[:,features])
            #pred_prob += clf[i].predict_proba(X_test)
            featureimportance += clf[i].feature_importances_
            features_n_fold += clf[i].n_features_ 
            features_used += features_used_in_rf(clf[i],  len(features))
            
            #Testing
            if 1 == 2:
                print(featureimportance)
                indic, n_nodes_ptr = clf[i].decision_path(X_test[:,features])
                #f.write(str(indic.todense().shape))
                f.write(str(clf[i].estimators_[1].tree_.feature))
                f.write("-------------------------------------------------------------------------")
                #f.write(str(n_nodes_ptr))
                #print(indic)
                #print(n_nodes_ptr)
                f.close()
                raise ValueError("Testing complete")
            
            
        pred_prob = pred_prob/30
        featureimportance = featureimportance/30
        features_n[j-1] = features_n_fold/30
        print("Elements non-zero per feature importance:")
        print(np.count_nonzero(featureimportance))
        print("Elements non-zero per feature used:")
        print(np.count_nonzero(features_used))
        ##print("Feature importance is:", featureimportance)
        # Pick the class with the greastest probability to be the prediction
        pred = np.argmax(pred_prob,axis=1)
        y_score = pred_prob[:,1]
    
        # Store performance metrics accuracy and draw ROC curve
        auc[j-1] = drawROC(y_test, y_score)
        bscore[j-1] = balanced_accuracy_score(y_test, pred)
        tn, fp, fn, tp = confusion_matrix(y_test,pred).ravel()
        specificity[j-1] = tn/(tn+fp)
        sensitivity[j-1] = tp/(tp+fn)
        precision[j-1] = tp/(tp+fp)
        f1[j-1] = 2*precision[j-1]*sensitivity[j-1]/(precision[j-1]+sensitivity[j-1])
        ##print("Specificity is:", specificity[j-1])
        ##print("Sensitivity is:", sensitivity[j-1])
        ##print("Precision is:", precision[j-1])
        ##print("F1 is:", f1[j-1])
        score = sum(pred==y_test)/n
        ##print("Accuracy is:", score)
        ##print("Balanced Accuracy is:", bscore[j-1])
        accu[j-1] = score
        
        
        j = j+1
    ##print("Average accuracy is:",sum(accu)/10)
    ##print("Average AUC is:",sum(auc)/10)
    ##print("Average balanced accuracy is:",sum(bscore)/10)
    ##print("Specificity is:", sum(specificity)/10)
    ##print("Sensitivity is:", sum(sensitivity)/10)
    ##print("Precision is:", sum(precision)/10)
    ##print("F1 is:", sum(f1)/10)
    
    avg_accu = sum(accu)/10
    avg_bal_acc = sum(bscore)/10
    avg_auc = sum(auc)/10
    avg_sens = sum(sensitivity)/10
    avg_spec = sum(specificity)/10
    avg_prec = sum(precision)/10
    avg_f1 = sum(f1)/10
    avg_features_n = sum(features_n)/10
    avg_feature_importance = sum(feature_importance)/10
    
    return(avg_accu, avg_bal_acc, avg_auc, avg_sens, avg_spec, avg_prec, avg_f1, avg_features_n, avg_feature_importance)
    
def features_used_in_rf(rf_clf, n_of_features):
    """
    Takes in a random forest classifier, the number of features for this model
    
    Returns an array as long as number of features, with how many times a feature was used as node in any 
    of the decision trees in this random forest classifier
    
    Ordering of features is same as order in X
    
    Function currently unsed as checking for feature_importance_ non-zeros leads to what wanted to use this for anyways
    
    
    """
    features_used_in_rf = np.zeros(n_of_features)
    
    for tree in rf_clf:
        for feature in tree.tree_.feature:
            if feature != -2:
                features_used_in_rf[feature] += 1
            
    return features_used_in_rf
    