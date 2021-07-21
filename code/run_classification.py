# run_classification.py
"""
Run 10-fold classification on the training data

"""
import os
import datetime
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from run_globals import REG_MODEL_DATA_DIR, OPTIMIZED_MODELS
from utility import subsample

def RunClassRun(classifier, X_train_path, y_train_path,y_label, out_path, runs):
    ensemble_n = 30
    startTime = datetime.datetime.now()
    sns.set(style = "darkgrid")
    result_filename = "{}_{}_{}_{}_{}".format(classifier, runs,X_train_path, y_label, datetime.datetime.now().strftime("%Y%m%d-%H%M"))

    X_path = os.path.join(REG_MODEL_DATA_DIR, X_train_path + ".csv")
    y_path = os.path.join(REG_MODEL_DATA_DIR, y_train_path + ".csv")

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    y_response = pd.read_csv('data/y_wk8_resp_qids_sr__final.csv').set_index('subjectkey')
     #  might want to make this a variable
    y_remission = pd.read_csv('data/y_wk8_rem_qids_sr__final.csv').set_index('subjectkey')

    for r in range(runs):

        kf = KFold(10, shuffle = True)

        fold = 1
        scores = {'fold':[], 'model':[],'valid_bal_acc':[]}

        for train_index, valid_index in kf.split(X):

            # Subset the data with the split-specific indexes
            X_train, X_valid  = X.loc[train_index].set_index('subjectkey'), X.loc[valid_index].set_index('subjectkey')
            y_train, y_valid = y.loc[train_index].set_index('subjectkey'), y.loc[valid_index].set_index('subjectkey')
        
            # Assign appropriate y label
            if y_label == "response":
                print("Response as ylabel")
                y_train = y_train.join(y_response)
                y_valid = y_valid.join(y_response)
            elif y_label == "remission":
                print("Remission as y_label")
                y_train = y_train.join(y_remission)
                y_valid = y_valid.join(y_remission)
            else: raise Exception("Invalid y_label variable")
            print(y_valid.target.value_counts())
            # Get rid of columns only useful for regression
            y_train = y_train.drop(columns = ['baseline_score','target_change','target_score'])
            y_valid = y_valid.drop(columns = ['baseline_score','target_change','target_score'])
            print(X_train.shape)
            training, label = subsample(y_train.join(X_train).to_numpy(), 30)

            clf = [None]*ensemble_n
            for i in range(ensemble_n):
                if classifier == "rf":
                    clf[i] = RandomForestClassifier(n_estimators=50, n_jobs = -1)
                    clf[i].fit(training[i],label[i])

            n = X_valid.shape[0]
            pred_prob = np.zeros((n,2))

            for i in range(ensemble_n):
                pred_prob += clf[i].predict_proba(X_valid)

            # Avg probabilities across the ensemble
            pred_prob = pred_prob/ensemble_n

            # Prediction is class with highest probability
            pred = np.argmax(pred_prob, axis = 1)

            scores['fold'].append(fold)
            scores['model'].append(classifier)
            scores['valid_bal_acc'].append(balanced_accuracy_score(y_valid, pred))
            fold +=1

        results = pd.DataFrame(scores)
        print(results)
        print(results['valid_bal_acc'].mean())
        print(results['valid_bal_acc'].std())
        
        # print(training.shape)            
        # print(label.shape)
        # print(pred_prob)
        # print(pred_prob.shape)

    return