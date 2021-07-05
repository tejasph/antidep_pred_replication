# -*- coding: utf-8 -*-
"""
Runs 1 run of the specified ML training and evaluation

"""
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, r2_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PowerTransformer
from scipy.stats import kurtosistest

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import datetime
import os

def RunRegRun(regressor, X_train_path, y_train_path, out_path, runs, class_y = "response", test_data = False):

    result_filename = "{}_{}_{}_{}_{}".format(regressor, runs,X_train_path, y_train_path, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    # Read in the data
    X = pd.read_csv("data/modelling/"+ X_train_path + ".csv")
    y = pd.read_csv("data/modelling/" + y_train_path + ".csv")


    if class_y == "response":
        y_response = pd.read_csv('data/y_wk8_resp_qids_sr__final.csv').set_index('subjectkey') #  might want to make this a variable
    elif class_y == "remission":
        y_response = pd.read_csv('data/y_wk8_rem_qids_sr__final.csv').set_index('subjectkey')
    else: raise Exception("Not a valid classification label")

    # alter Column name to avoid mixup
    y_response.columns = ['actual_resp']


    print(kurtosistest(y['target']))

    run_scores = {'run':[], 'model':[], 'avg_train_RMSE':[], 'avg_train_bal_acc':[], 'avg_train_R2':[], 'avg_valid_RMSE':[], 'avg_valid_bal_acc':[], 'avg_valid_R2':[]}
    for r in range(runs):
        print(f"Run {r}")
        # Establish 10 fold crossvalidation splits
        kf = KFold(10, shuffle = True)
        scores = {'fold':[], 'model':[], 'train_RMSE':[], 'train_bal_acc':[],'train_R2':[], 'valid_RMSE':[], 'valid_bal_acc':[], 'valid_R2': [],'specificity':[], 'sensitivity':[], 'precision':[]}
        fold = 1
        for train_index, valid_index in kf.split(X):

            # Subset the data with the split-specific indexes
            X_train, X_valid  = X.loc[train_index].set_index('subjectkey'), X.loc[valid_index].set_index('subjectkey')
            y_train, y_valid = y.loc[train_index].set_index('subjectkey'), y.loc[valid_index].set_index('subjectkey')

            # append actual_resp for later comparison --> grabs actual response fore each matching subjectkey
            y_train = y_train.join(y_response)
            y_valid = y_valid.join(y_response)

            #Transform y to make the distribution more Gaussian
            # yeo_transformer = PowerTransformer()
            # y_train['yeo_target'] = yeo_transformer.fit_transform(y_train[['target']])
            # y_valid['target'] = yeo_transformer.transform(y_valid[['target']])

            # Establish the model
            if regressor == 'rf':
                # optimized for overlapping features
                # model = RandomForestRegressor(max_features=0.33, max_samples=0.9,
                #       min_samples_leaf=11, min_samples_split=7, n_jobs=-1)

                # optimized for non-overlapping X
                model = RandomForestRegressor(max_depth=30, max_samples=0.8, min_samples_leaf=5,
                      min_samples_split=10, n_jobs=-1)

                # Basic Model
                # model = RandomForestRegressor()
            elif regressor == 'svr':
                model = SVR()
            elif regressor == 'gbr':
                model = GradientBoostingRegressor()


            # Make our predictions (t_results = training, v_results = validation)
            if class_y == "response":
                t_results, v_results = assess_model(model, X_train, y_train, X_valid, y_valid)
                t_classification = t_results.pred_response
                v_classification = v_results.pred_response

            elif class_y == "remission":
                t_results, v_results = assess_on_remission(model, X_train, y_train, X_valid, y_valid)
                t_classification = t_results.pred_remission
                v_classification = v_results.pred_remission

            # Calculate Scores
            scores['fold'].append(fold)
            scores['model'].append(regressor)
            
            scores['train_RMSE'].append(mean_squared_error(t_results.target, t_results.pred_change, squared = False))
            scores['train_bal_acc'].append(balanced_accuracy_score(t_results.actual_resp, t_classification)) # t_classification used to be t_results.pred_response
            scores['train_R2'].append(r2_score(t_results.target, t_results.pred_change))
            
            scores['valid_RMSE'].append(mean_squared_error(v_results.target,v_results.pred_change, squared = False))
            scores['valid_bal_acc'].append(balanced_accuracy_score(v_results.actual_resp, v_classification))
            scores['valid_R2'].append(r2_score(v_results.target, v_results.pred_change))

            tn, fp, fn, tp = confusion_matrix(v_results.actual_resp,v_classification).ravel()
            scores['specificity'].append(tn/(tn+fp)) 
            scores['sensitivity'].append(tp/(tp+fn))
            scores['precision'].append(tp/(tp+fp))
            fold += 1
            
        # Generate a histogram of predictions for the run 
        # v_plot = sns.histplot(data = v_results, x = 'pred')

        sns.set(style = "darkgrid")
        # All thx to python-graph-gallery.com
        fig, axs = plt.subplots(2,2, figsize = (7,7))
        sns.histplot(data = v_results, x = 'pred_change', ax = axs[0,0], kde = True,  color = "red", label = "Validation")
        sns.histplot(data = v_results, x = 'target',ax = axs[0,1], kde = True,color = "red", label = "Validation")
        sns.histplot(data = t_results, x = 'pred_change', ax = axs[1,0], kde = True ,color = "blue", label = "Training", alpha = 0.5)
        sns.histplot(data = t_results, x = 'target', ax = axs[1,1], kde = True,  color = "blue", label = "Training")
        plt.legend()
        plt.savefig(out_path + "prediction_plots.png", bbox_inches = 'tight')

    

        # Avg the scores across the 10 folds
        results = pd.DataFrame(scores)
        print(f"Run {r} Results")
        print(results.mean())
        run_scores['run'].append(r)
        run_scores['model'].append(regressor)

        run_scores['avg_train_RMSE'].append(results['train_RMSE'].mean())
        run_scores['avg_train_bal_acc'].append(results['train_bal_acc'].mean())
        run_scores['avg_train_R2'].append(results['train_R2'].mean())

        run_scores['avg_valid_RMSE'].append(results['valid_RMSE'].mean())
        run_scores['avg_valid_bal_acc'].append(results['valid_bal_acc'].mean())
        run_scores['avg_valid_R2'].append(results['valid_R2'].mean())
        
    # average the scores across the r runs and get standard deviations
    final_score_df = pd.DataFrame(run_scores)
    print(final_score_df.mean())
    print(final_score_df.std())

    

    # Write text file, or output dataframes to appropriate folders

    f = open(os.path.join(out_path, result_filename + '.txt'), 'w')
    f.write("Model Results for: {}\n\n".format(result_filename))
    f.write("Average training RMSE is {:.4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_train_RMSE'].mean(), final_score_df['avg_train_RMSE'].std()))
    f.write("Average training balanced_accuracy is {:.4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_train_bal_acc'].mean(), final_score_df['avg_train_bal_acc'].std()))
    f.write("Average training R2 is {:.4f} with standard deviation of {:6f}.\n\n".format(final_score_df['avg_train_R2'].mean(), final_score_df['avg_train_R2'].std()))

    f.write("Average validation RMSE is {:.4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_valid_RMSE'].mean(), final_score_df['avg_valid_RMSE'].std()))
    f.write("Average validation balanced accuracy is {:.4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_valid_bal_acc'].mean(), final_score_df['avg_valid_bal_acc'].std()))
    f.write("Average validation R2 is {:.4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_valid_R2'].mean(), final_score_df['avg_valid_R2'].std()))

    if test_data == True: # need to add transformer if its proven to work out
        X_test = pd.read_csv("data/modelling/X_test_norm.csv").set_index('subjectkey')
        y_test = pd.read_csv("data/modelling/y_test.csv").set_index('subjectkey')
        print("Predicting on unseen data")

        y = y.join(y_response)
        y_test = y_test.join(y_response)

        train_results, test_results = assess_model(model, X.set_index('subjectkey'), y.set_index('subjectkey'), X_test, y_test)

        f.write("Test RMSE is {:4f}\n".format(mean_squared_error(test_results.target,test_results.pred, squared = False)))
        f.write("Test balanced accuracy is {:4f}\n".format(balanced_accuracy_score(test_results.actual_resp,test_results.pred_response)))
        f.write("Test R2 is {:4f}\n".format(r2_score(test_results.target, test_results.pred)))

def assess_model(model, X_train, y_train, X_valid, y_valid):
    train_results = y_train.copy()
    valid_results = y_valid.copy()
    
    # Fit the model to transformed y
    model.fit(X_train, y_train['target'])
    
    # Make Predictions
    train_results['pred_change'] = model.predict(X_train)
    valid_results['pred_change'] = model.predict(X_valid)

    # # Inverse Transform predictions
    # train_results['pred'] = (train_results['pred']**2) + y_min
    # valid_results['pred'] = (valid_results['pred']**2) + y_min
    # print(train_results.head())

    # Inverse transform prediction back into qids scale
    # train_results['pred_change'] = yeo_transformer.inverse_transform(train_results[['pred']])
    # valid_results['pred_change'] = yeo_transformer.inverse_transform(valid_results[['pred']])
    
    # Calculate response percentage
    train_results['response_pct'] = train_results['pred_change']/train_results['baseline_score']
    valid_results['response_pct'] = valid_results['pred_change']/valid_results['baseline_score']
    
    # Determine whether model predictions are established as response or not
    train_results['pred_response'] = np.where(train_results['response_pct'] <= -.50, 1.0, 0.0)
    valid_results['pred_response'] = np.where(valid_results['response_pct'] <= -.50, 1.0, 0.0)
    print(valid_results.head())
#     print(f"Balanced Accuracy: {balanced_accuracy_score(results['actual_resp'], results['pred_response'])}")
    return train_results, valid_results

def assess_on_remission(model, X_train, y_train, X_valid, y_valid): #Merge this with above

    train_results = y_train.copy()
    valid_results = y_valid.copy()
    
    # Fit the model to transformed y
    model.fit(X_train, y_train['target'])

    # Make Predictions
    train_results['pred_change'] = model.predict(X_train)
    valid_results['pred_change'] = model.predict(X_valid)

    train_results['pred_score'] = train_results['baseline_score'] + train_results['pred_change']
    valid_results['pred_score'] = valid_results['baseline_score'] + valid_results['pred_change']

    # Determine whether model predictions are established as response or not
    train_results['pred_remission'] = np.where(train_results['pred_score'] <= 5, 1.0, 0.0)
    valid_results['pred_remission'] = np.where(valid_results['pred_score'] <= 5, 1.0, 0.0)

    print(valid_results.head())
    return train_results, valid_results