# -*- coding: utf-8 -*-
"""
Runs 1 run of the specified ML training and evaluation

"""
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, r2_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.preprocessing import PowerTransformer
from scipy.stats import kurtosistest

from run_globals import REG_MODEL_DATA_DIR, OPTIMIZED_MODELS

import seaborn as sns
import matplotlib.pyplot as plt
import shap

import pandas as pd
import numpy as np
import datetime
import pickle
import os

def RunRegRun(regressor, X_train_path, y_train_path, y_proxy, out_path, runs):

    startTime = datetime.datetime.now()
    sns.set(style = "darkgrid")
    result_filename = "{}_{}_{}_{}_{}".format(regressor, runs,X_train_path, y_proxy, datetime.datetime.now().strftime("%Y%m%d-%H%M"))

    # Set up paths
    X_path = os.path.join(REG_MODEL_DATA_DIR, X_train_path + ".csv")
    y_path = os.path.join(REG_MODEL_DATA_DIR, y_train_path + ".csv")

    # Read in the data
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    print(y.head())

    # Assign target based on specified y_proxy
    if y_proxy == "score_change":
        y['target'] = y['target_change']
    elif y_proxy == 'final_score':
        y['target'] = y['target_score']
    else: raise Exception("Invalid proxy y selection")   
    y = y.drop(columns = ['target_change', 'target_score']) 

    
    y_response = pd.read_csv('data/y_wk8_resp_qids_sr__final.csv').set_index('subjectkey') #  might want to make this a variable
    y_remission = pd.read_csv('data/y_wk8_rem_qids_sr__final.csv').set_index('subjectkey')
    

    # Alter Column name to avoid mixup
    y_response.columns = ['actual_resp']
    y_remission.columns = ['true_rem']

    # This dict will store all results across the runs
    run_scores = {'run':[], 'model':[], 'avg_train_RMSE':[], 'avg_train_R2':[], 'avg_valid_RMSE':[], 'avg_valid_R2':[],
                'avg_train_resp_bal_acc':[], 'avg_valid_resp_bal_acc':[],'avg_valid_resp_auc':[], 'avg_valid_resp_sens':[], 'avg_valid_resp_spec':[], 'avg_valid_resp_prec':[], 'avg_valid_resp_NPV':[],
                 'avg_train_rem_bal_acc':[], 'avg_valid_rem_bal_acc':[], 'avg_valid_rem_sens':[], 'avg_valid_rem_spec':[], 'avg_valid_rem_prec':[], 'avg_valid_rem_NPV':[]}

    for r in range(runs):
        print(f"Run {r}")
        # Establish 10 fold crossvalidation splits
        kf = KFold(10, shuffle = True)

        scores = {'fold':[], 'model':[], 'train_RMSE':[],'train_R2':[], 'valid_RMSE':[], 'valid_R2': [],
            'train_resp_bal_acc':[], 'valid_resp_bal_acc':[], 'valid_resp_auc':[],  'resp_specificity':[], 'resp_sensitivity':[], 'resp_precision':[], 'resp_NPV':[],
            'train_rem_bal_acc':[], 'valid_rem_bal_acc':[],  'rem_specificity':[], 'rem_sensitivity':[], 'rem_precision':[], 'rem_NPV':[]}

        fold = 1
        for train_index, valid_index in kf.split(X):

            # Subset the data with the split-specific indexes
            X_train, X_valid  = X.loc[train_index].set_index('subjectkey'), X.loc[valid_index].set_index('subjectkey')
            y_train, y_valid = y.loc[train_index].set_index('subjectkey'), y.loc[valid_index].set_index('subjectkey')

            # append actual_resp for later comparison --> grabs actual response fore each matching subjectkey
            y_train = y_train.join(y_response)
            y_train = y_train.join(y_remission)

            y_valid = y_valid.join(y_response)
            y_valid = y_valid.join(y_remission)
            
            # Establish the model
            model_filename = "{}_{}_{}".format(regressor, X_train_path, y_proxy) # can use this to directly import the optimized param model
            model_path = os.path.join(OPTIMIZED_MODELS, model_filename + ".pkl")
            model = pickle.load(open(model_path,'rb'))


            # Below is best performing model for overlapping features --> 70ish bal acc
            # model = RandomForestRegressor(max_features=0.33, max_samples=0.9,
            #           min_samples_leaf=11, min_samples_split=7, n_jobs=-1)

            # Optimized for X_train_norm w/ final score
            print(model)
            
        
            # Make our predictions (t_results = training, v_results = validation)
            if y_proxy == "score_change":
                
                t_results, v_results = assess_on_score_change(model, X_train, y_train, X_valid, y_valid, out_path)

                # All thx to python-graph-gallery.com
                fig, axs = plt.subplots(2,2, figsize = (7,7))
                sns.histplot(data = v_results, x = 'pred_change', ax = axs[0,0], kde = True,  color = "red", label = "Validation")
                sns.histplot(data = v_results, x = 'target',ax = axs[0,1], kde = True,color = "red", label = "Validation")
                sns.histplot(data = t_results, x = 'pred_change', ax = axs[1,0], kde = True ,color = "blue", label = "Training", alpha = 0.5)
                sns.histplot(data = t_results, x = 'target', ax = axs[1,1], kde = True,  color = "blue", label = "Training")
                plt.legend()
                plt.savefig(out_path + "/prediction_plots.png", bbox_inches = 'tight')
                plt.close()

                scores['train_RMSE'].append(mean_squared_error(t_results.target, t_results.pred_change, squared = False))
                scores['train_R2'].append(r2_score(t_results.target, t_results.pred_change))
                scores['valid_RMSE'].append(mean_squared_error(v_results.target,v_results.pred_change, squared = False))
                scores['valid_R2'].append(r2_score(v_results.target, v_results.pred_change))

            elif y_proxy == "final_score":

                t_results, v_results = assess_on_final_score(model, X_train, y_train, X_valid, y_valid, out_path)

                fig, axs = plt.subplots(2,2, figsize = (7,7))
                sns.histplot(data = v_results, x = 'pred_score', ax = axs[0,0], kde = True,  color = "red", label = "Validation")
                sns.histplot(data = v_results, x = 'target',ax = axs[0,1], kde = True,color = "red", label = "Validation")
                sns.histplot(data = t_results, x = 'pred_score', ax = axs[1,0], kde = True ,color = "blue", label = "Training", alpha = 0.5)
                sns.histplot(data = t_results, x = 'target', ax = axs[1,1], kde = True,  color = "blue", label = "Training")
                plt.legend()
                plt.savefig(out_path + "/prediction_plots.png", bbox_inches = 'tight')
                plt.close()

                scores['train_RMSE'].append(mean_squared_error(t_results.target, t_results.pred_score, squared = False))
                scores['train_R2'].append(r2_score(t_results.target, t_results.pred_score))
                scores['valid_RMSE'].append(mean_squared_error(v_results.target,v_results.pred_score, squared = False))
                scores['valid_R2'].append(r2_score(v_results.target, v_results.pred_score))

                fpr, tpr, thresholds = roc_curve(v_results.actual_resp, v_results.pred_score)
                print(fpr)
                print(tpr)
                print(thresholds)
                print(f"AUC: {auc(fpr, tpr)}")
                scores['valid_resp_auc'].append(auc(fpr, tpr))

            # Calculate Regression Scores
            scores['fold'].append(fold)
            scores['model'].append(regressor)

            # Calculate Response Classification Accuracy
            scores['train_resp_bal_acc'].append(balanced_accuracy_score(t_results.actual_resp, t_results.pred_response)) 
            scores['valid_resp_bal_acc'].append(balanced_accuracy_score(v_results.actual_resp, v_results.pred_response))
            

            tn, fp, fn, tp = confusion_matrix(v_results.actual_resp, v_results.pred_response).ravel()
            scores['resp_specificity'].append(tn/(tn+fp)) 
            scores['resp_sensitivity'].append(tp/(tp+fn))
            scores['resp_precision'].append(tp/(tp+fp))
            scores['resp_NPV'].append(tn/(fn + tn))

            # Calculate Remission Classification Accuracy 
            scores['train_rem_bal_acc'].append(balanced_accuracy_score(t_results.true_rem, t_results.pred_remission)) 
            scores['valid_rem_bal_acc'].append(balanced_accuracy_score(v_results.true_rem, v_results.pred_remission))

            tn, fp, fn, tp = confusion_matrix(v_results.true_rem, v_results.pred_remission).ravel()
            scores['rem_specificity'].append(tn/(tn+fp)) 
            scores['rem_sensitivity'].append(tp/(tp+fn))
            scores['rem_precision'].append(tp/(tp+fp))
            scores['rem_NPV'].append(tn/(fn + tn))
            fold += 1


        sns.set(style = "darkgrid")
        # All thx to python-graph-gallery.com
        t_results['correct_resp'] = np.where(t_results['pred_response'] == t_results['actual_resp'], 1, 0)
        print(t_results.head())
        sns.scatterplot(data = t_results, x = 'target', y = 'pred_change', hue = 'correct_resp', alpha = 0.3)
        plt.plot([-25,10], [-25,10])
        plt.savefig(out_path + "/prediction_vs_actual.png", bbox_inches = 'tight')
        plt.close()

        # Avg the scores across the 10 folds
        results = pd.DataFrame(scores)
        print(f"Run {r} Results")
        print(results.mean())
        run_scores['run'].append(r)
        run_scores['model'].append(regressor)

        run_scores['avg_train_RMSE'].append(results['train_RMSE'].mean())
        run_scores['avg_train_R2'].append(results['train_R2'].mean())
        run_scores['avg_valid_RMSE'].append(results['valid_RMSE'].mean())
        run_scores['avg_valid_R2'].append(results['valid_R2'].mean())

        
        run_scores['avg_train_resp_bal_acc'].append(results['train_resp_bal_acc'].mean())
        run_scores['avg_valid_resp_bal_acc'].append(results['valid_resp_bal_acc'].mean())
        run_scores['avg_valid_resp_auc'].append(results['valid_resp_auc'].mean())
        run_scores['avg_valid_resp_sens'].append(results['resp_sensitivity'].mean())
        run_scores['avg_valid_resp_spec'].append(results['resp_specificity'].mean())
        run_scores['avg_valid_resp_prec'].append(results['resp_precision'].mean())
        run_scores['avg_valid_resp_NPV'].append(results['resp_NPV'].mean())

        run_scores['avg_train_rem_bal_acc'].append(results['train_rem_bal_acc'].mean())
        run_scores['avg_valid_rem_bal_acc'].append(results['valid_rem_bal_acc'].mean())
        run_scores['avg_valid_rem_sens'].append(results['rem_sensitivity'].mean())
        run_scores['avg_valid_rem_spec'].append(results['rem_specificity'].mean())
        run_scores['avg_valid_rem_prec'].append(results['rem_precision'].mean())
        run_scores['avg_valid_rem_NPV'].append(results['rem_NPV'].mean())

    # average the scores across the r runs and get standard deviations
    final_score_df = pd.DataFrame(run_scores)
    print(final_score_df.mean())
    print(final_score_df.std()) 


    # Write text file, or output dataframes to appropriate folders
 
    f = open(os.path.join(out_path, result_filename + '.txt'), 'w')
    # Regression Related metrics
    f.write("Regression Model Results for: {}\n\n".format(result_filename))
    f.write("Model used: {}\n\n".format(model))
    f.write("Number of runs: {}\n".format(runs))
    f.write("Average training RMSE is {:.4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_train_RMSE'].mean(), final_score_df['avg_train_RMSE'].std()))
    f.write("Average training R2 is {:.4f} with standard deviation of {:6f}.\n\n".format(final_score_df['avg_train_R2'].mean(), final_score_df['avg_train_R2'].std()))
    f.write("Average validation RMSE is {:.4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_valid_RMSE'].mean(), final_score_df['avg_valid_RMSE'].std()))
    f.write("Average validation R2 is {:.4f} with standard deviation of {:6f}.\n\n".format(final_score_df['avg_valid_R2'].mean(), final_score_df['avg_valid_R2'].std()))

    # Response Classification Performance
    f.write("Response Classification Results: \n")
    f.write("Average training response balanced_accuracy is {:.4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_train_resp_bal_acc'].mean(), final_score_df['avg_train_resp_bal_acc'].std()))
    f.write("Average validation response balanced accuracy is {:.4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_valid_resp_bal_acc'].mean(), final_score_df['avg_valid_resp_bal_acc'].std()))
    f.write("Average validation sensitivity is {:4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_valid_resp_sens'].mean(), final_score_df['avg_valid_resp_sens'].std()))
    f.write("Average validation specificity is {:4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_valid_resp_spec'].mean(), final_score_df['avg_valid_resp_spec'].std()))
    f.write("Average validation precision is {:4f} with standard deviation of {:6f}.\n\n".format(final_score_df['avg_valid_resp_prec'].mean(), final_score_df['avg_valid_resp_prec'].std()))

    # Remission Classification Performance 
    f.write("Remission Classification Results: \n")
    f.write("Average training remission balanced_accuracy is {:.4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_train_rem_bal_acc'].mean(), final_score_df['avg_train_rem_bal_acc'].std()))
    f.write("Average validation remission balanced accuracy is {:.4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_valid_rem_bal_acc'].mean(), final_score_df['avg_valid_rem_bal_acc'].std()))
    f.write("Average validation sensitivity is {:4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_valid_rem_sens'].mean(), final_score_df['avg_valid_rem_sens'].std()))
    f.write("Average validation specificity is {:4f} with standard deviation of {:6f}.\n".format(final_score_df['avg_valid_rem_spec'].mean(), final_score_df['avg_valid_rem_spec'].std()))
    f.write("Average validation precision is {:4f} with standard deviation of {:6f}.\n\n".format(final_score_df['avg_valid_rem_prec'].mean(), final_score_df['avg_valid_rem_prec'].std()))
    
    print("Completed after seconds: \n")
    print(datetime.datetime.now() - startTime)
    return final_score_df

def evaluate_on_test(regressor, X_train_type, y_proxy, out_path, runs = 10):
    print("Predicting on unseen data...")
    result_filename = "test_{}_{}_{}_{}".format(regressor, X_train_type, y_proxy, datetime.datetime.now().strftime("%Y%m%d-%H%M"))

    test_run_scores = {'run':[], 'model':[], 'train_RMSE':[], 'train_R2':[], 'test_RMSE':[], 'test_R2':[],
                'train_resp_bal_acc':[], 'test_resp_bal_acc':[],'test_resp_sens':[], 'test_resp_spec':[], 'test_resp_prec':[],'test_resp_NPV':[],
                 'train_rem_bal_acc':[], 'test_rem_bal_acc':[], 'test_rem_sens':[], 'test_rem_spec':[], 'test_rem_prec':[], 'test_rem_NPV':[]}

    # Adding path to test data
    if X_train_type == "X_train_norm":
        X_train_path = os.path.join(REG_MODEL_DATA_DIR, "X_train_norm.csv")
        X_test_path = os.path.join(REG_MODEL_DATA_DIR, "X_test_norm.csv")
    elif X_train_type == "X_train_norm_over":
        X_train_path = os.path.join(REG_MODEL_DATA_DIR, "X_train_norm_over.csv")
        X_test_path = os.path.join(REG_MODEL_DATA_DIR, "X_test_norm_over.csv")
    elif X_train_type == "X_train_norm_select":
        X_train_path = os.path.join(REG_MODEL_DATA_DIR, "X_train_norm_select.csv")
        X_test_path = os.path.join(REG_MODEL_DATA_DIR, "X_test_norm_select.csv")
    elif X_train_type == "X_train_norm_over_select":
        X_train_path = os.path.join(REG_MODEL_DATA_DIR, "X_train_norm_over_select.csv")
        X_test_path = os.path.join(REG_MODEL_DATA_DIR, "X_test_norm_over_select.csv")       
        
    y_train_path =os.path.join(REG_MODEL_DATA_DIR, "y_train.csv")
    y_test_path = os.path.join(REG_MODEL_DATA_DIR, "y_test.csv")

    # Read in the Data
    X_train = pd.read_csv(X_train_path).set_index('subjectkey')
    X_test = pd.read_csv(X_test_path).set_index('subjectkey')
    y_train = pd.read_csv(y_train_path).set_index('subjectkey')
    y_test = pd.read_csv(y_test_path).set_index('subjectkey')
    y_response = pd.read_csv('data/y_wk8_resp_qids_sr__final.csv').set_index('subjectkey') #  might want to make this a variable
    y_remission = pd.read_csv('data/y_wk8_rem_qids_sr__final.csv').set_index('subjectkey')
    

    # Alter Column name to avoid mixup
    y_response.columns = ['actual_resp']
    y_remission.columns = ['true_rem']

    # Prep the y labels (y -refers to entire training data) 
    if y_proxy == "score_change":
        y_train['target'] = y_train['target_change']
        y_test['target'] = y_test['target_change']
    elif y_proxy == 'final_score':
        y_train['target'] = y_train['target_score']
        y_test['target'] = y_test['target_score']
    else: raise Exception("Invalid proxy y selection")   

    y_train = y_train.drop(columns = ['target_change','target_score'])
    y_test = y_test.drop(columns = ['target_change', 'target_score'])
    
    y_train = y_train.join(y_response)
    y_train = y_train.join(y_remission)

    y_test = y_test.join(y_response)
    y_test = y_test.join(y_remission)

    # Load Model
    model_filename = "{}_{}_{}".format(regressor, X_train_type, y_proxy) # can use this to directly import the optimized param model
    model_path = os.path.join(OPTIMIZED_MODELS, model_filename + ".pkl")
    model = pickle.load(open(model_path,'rb'))

    for run in range(runs):
        # Train and Assess
        test_run_scores['run'].append(run)
        test_run_scores['model'].append(regressor + "_" + X_train_type)

        if y_proxy == "score_change":
            train_results, test_results = assess_on_score_change(model, X_train, y_train, X_test, y_test, out_path)
            test_run_scores['train_RMSE'].append(mean_squared_error(train_results.target, train_results.pred_change, squared = False))
            test_run_scores['train_R2'].append(r2_score(train_results.target, train_results.pred_change))
            test_run_scores['test_RMSE'].append(mean_squared_error(test_results.target, test_results.pred_change, squared = False))
            test_run_scores['test_R2'].append(r2_score(test_results.target, test_results.pred_change))

        elif  y_proxy == "final_score":
            train_results, test_results = assess_on_final_score(model, X_train, y_train, X_test, y_test, out_path)
            test_run_scores['train_RMSE'].append(mean_squared_error(train_results.target, train_results.pred_score, squared = False))
            test_run_scores['train_R2'].append(r2_score(train_results.target, train_results.pred_score))
            test_run_scores['test_RMSE'].append(mean_squared_error(test_results.target, test_results.pred_score, squared = False))
            test_run_scores['test_R2'].append(r2_score(test_results.target, test_results.pred_score))

          
        # Calculate Response Classification Accuracy
        test_run_scores['train_resp_bal_acc'].append(balanced_accuracy_score(train_results.actual_resp, train_results.pred_response))
        test_run_scores['test_resp_bal_acc'].append(balanced_accuracy_score(test_results.actual_resp, test_results.pred_response))

        tn, fp, fn, tp = confusion_matrix(test_results.actual_resp, test_results.pred_response).ravel()
        test_run_scores['test_resp_spec'].append(tn/(tn+fp)) 
        test_run_scores['test_resp_sens'].append(tp/(tp+fn))
        test_run_scores['test_resp_prec'].append(tp/(tp+fp))
        test_run_scores['test_resp_NPV'].append(tn/(fn + tn))

        # Calculate Remission Classification Accuracy 
        train_rem_bal_acc = balanced_accuracy_score(train_results.true_rem, train_results.pred_remission)
        test_rem_bal_acc = balanced_accuracy_score(test_results.true_rem, test_results.pred_remission)

        test_run_scores['train_rem_bal_acc'].append(balanced_accuracy_score(train_results.true_rem, train_results.pred_remission))
        test_run_scores['test_rem_bal_acc'].append(balanced_accuracy_score(test_results.true_rem, test_results.pred_remission))
        
        tn, fp, fn, tp = confusion_matrix(test_results.true_rem, test_results.pred_remission).ravel()
        test_run_scores['test_rem_spec'].append(tn/(tn+fp)) 
        test_run_scores['test_rem_sens'].append(tp/(tp+fn))
        test_run_scores['test_rem_prec'].append(tp/(tp+fp))
        test_run_scores['test_rem_NPV'].append(tn/(fn + tn))

    test_results_df = pd.DataFrame(test_run_scores)
    return test_results_df

def assess_on_score_change(model, X_train, y_train, X_valid, y_valid, out_path):
    train_results = y_train.copy()
    valid_results = y_valid.copy()
    
    # Fit the model to transformed y
    model.fit(X_train, y_train['target'])
    
    # Make Predictions
    train_results['pred_change'] = model.predict(X_train)
    valid_results['pred_change'] = model.predict(X_valid)

    # Calculate the predicted score
    train_results['pred_score'] = train_results['baseline_score'] + train_results['pred_change']
    valid_results['pred_score'] = valid_results['baseline_score'] + valid_results['pred_change']
    
    # Calculate response percentage
    train_results['response_pct'] = train_results['pred_change']/train_results['baseline_score']
    valid_results['response_pct'] = valid_results['pred_change']/valid_results['baseline_score']
    
    # Determine whether model predictions are established as response or not
    train_results['pred_response'] = np.where(train_results['response_pct'] <= -.50, 1.0, 0.0)
    valid_results['pred_response'] = np.where(valid_results['response_pct'] <= -.50, 1.0, 0.0)


    # Determine whether model predictions are established as remission or not
    train_results['pred_remission'] = np.where(train_results['pred_score'] <= 5, 1.0, 0.0)
    valid_results['pred_remission'] = np.where(valid_results['pred_score'] <= 5, 1.0, 0.0)
    
    #Temporary exploratory analysis
    sns.scatterplot(data = valid_results, x = 'baseline_score', y = 'target', hue = 'actual_resp')
    plt.plot([0,25],[0,0])
    plt.savefig(out_path + "/baseline_vs_target.png", bbox_inches = 'tight')
    plt.close()

    sns.scatterplot(data = valid_results, x = 'baseline_score', y = 'target', hue = 'pred_response')
    plt.plot([0,25],[0,0])
    plt.savefig(out_path + "/baseline_vs_target_pred.png", bbox_inches = 'tight')
    plt.close()

    return train_results, valid_results

def assess_on_final_score(model, X_train, y_train, X_valid, y_valid, out_path):
    train_results = y_train.copy()
    valid_results = y_valid.copy()

    # Fit the model to transformed y
    model.fit(X_train, y_train['target'])

    # Make Predictions
    train_results['pred_score'] = model.predict(X_train)
    valid_results['pred_score'] = model.predict(X_valid)

    # Calculate predicted score change
    train_results['pred_change'] = train_results['pred_score'] - train_results['baseline_score']
    valid_results['pred_change'] = valid_results['pred_score'] - valid_results['baseline_score']

    # Calculate response percentage (calculated in similar manner to train_on_score_change())
    train_results['response_pct'] = train_results['pred_change']/train_results['baseline_score']
    valid_results['response_pct'] = valid_results['pred_change']/valid_results['baseline_score']

    # Determine whether model predictions are established as response or not (note: response_pct is calculated slightly differently as assess_model())
    train_results['pred_response'] = np.where(train_results['response_pct'] <= -.50, 1.0, 0.0)
    valid_results['pred_response'] = np.where(valid_results['response_pct'] <= -.50, 1.0, 0.0)

    # Determine whether model predictions are established as remission or not
    train_results['pred_remission'] = np.where(train_results['pred_score'] <= 5, 1.0, 0.0)
    valid_results['pred_remission'] = np.where(valid_results['pred_score'] <= 5, 1.0, 0.0)

    
    return train_results, valid_results
