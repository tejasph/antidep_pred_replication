# -*- coding: utf-8 -*-
"""
Runs 1 run of the specified ML training and evaluation

"""
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, r2_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np

def RunRegRun(regressor, X_train_path, y_train_path, runs):

    # Read in the data
    X = pd.read_csv(X_train_path)
    y = pd.read_csv(y_train_path)
    y_response = pd.read_csv('data/y_wk8_resp_qids_sr__final.csv').set_index('subjectkey') #  might want to make this a variable
    print(y_response.head())
    # alter Column slightly
    y_response.columns = ['actual_resp']

    for r in range(runs):
        print(f"Run {r}")
        # Establish 10 fold crossvalidation splits
        kf = KFold(10, shuffle = True)
        scores = {'fold':[], 'model':[], 'train_RMSE':[], 'train_bal_acc':[], 'valid_RMSE':[], 'valid_bal_acc':[], 'valid_R2': [],'specificity':[], 'sensitivity':[], 'precision':[]}
        fold = 1
        for train_index, valid_index in kf.split(X):

            # Subset the data with the split-specific indexes
            X_train, X_valid  = X.loc[train_index].set_index('subjectkey'), X.loc[valid_index].set_index('subjectkey')
            y_train, y_valid = y.loc[train_index].set_index('subjectkey'), y.loc[valid_index].set_index('subjectkey')

            # append actual_resp for later comparison --> grabs actual response fore each matching subjectkey
            y_train = y_train.join(y_response)
            y_valid = y_valid.join(y_response)

            # Establish the model
            if regressor == 'rf':
                model = RandomForestRegressor()


            # Make our predictions (t_results = training, v_results = validation)
            t_results, v_results = assess_model(model, X_train, y_train, X_valid, y_valid)

            # Calculate Scores
            scores['fold'].append(fold)
            scores['model'].append(regressor)
            
            scores['train_RMSE'].append(mean_squared_error(t_results.target, t_results.pred, squared = False))
            scores['train_bal_acc'].append(balanced_accuracy_score(t_results.actual_resp, t_results.pred_response))
            
            scores['valid_RMSE'].append(mean_squared_error(v_results.target,v_results.pred, squared = False))
            scores['valid_bal_acc'].append(balanced_accuracy_score(v_results.actual_resp,v_results.pred_response))
            scores['valid_R2'].append(r2_score(v_results.target, v_results.pred))

            tn, fp, fn, tp = confusion_matrix(v_results.actual_resp,v_results.pred_response).ravel()
            scores['specificity'].append(tn/(tn+fp)) 
            scores['sensitivity'].append(tp/(tp+fn))
            scores['precision'].append(tp/(tp+fp))
            fold += 1
            

        # Avg the scores across the 10 folds
        results = pd.DataFrame(scores)
        print(results.mean())
        
    # average the scores across the r runs and get standard deviations



    # Write text file, or output dataframes to appropriate folders

def assess_model(model, X_train, y_train, X_valid, y_valid):
    train_results = y_train.copy()
    valid_results = y_valid.copy()
    
    # Fit the model
    model.fit(X_train, y_train['target'])
    
    # Make Predictions
    train_results['pred'] = model.predict(X_train)
    valid_results['pred'] = model.predict(X_valid)
    
    # Calculate response percentage
    train_results['response_pct'] = train_results['pred']/train_results['baseline_score']
    valid_results['response_pct'] = valid_results['pred']/valid_results['baseline_score']
    
    # Determine whether model predictions are established as response or not
    train_results['pred_response'] = np.where(train_results['response_pct'] <= -.50, 1.0, 0.0)
    valid_results['pred_response'] = np.where(valid_results['response_pct'] <= -.50, 1.0, 0.0)
    
#     print(f"Balanced Accuracy: {balanced_accuracy_score(results['actual_resp'], results['pred_response'])}")
    return train_results, valid_results