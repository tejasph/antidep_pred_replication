#Regression Results
from run_regression import RunRegRun, evaluate_on_test
from run_globals import REG_RESULTS_DIR
# from run_classification import RunClassRun
import pandas as pd
import os

"""
Run Experiment script

Change variables and experiment name using this script. 
"""


#########################################
# Instructions
# There are a few variable options that can be used/altered in each experiment. See below for list of options:
#
regressor_options = ['rf', 'sgdReg', 'gbdt', 'knn','svr_rbf'] # which regressor model to use
X_path_options = ['X_train_norm_over', 'X_train_norm', 'X_train_norm_select', 'X_train_norm_over_select'] # whether to use overlapping features or full-set features

X_path_mapping = {'X_train_norm_over':'Overlapping', 'X_train_norm':'All', 'X_train_norm_select':'All_RFE', 'X_train_norm_over_select':'Overlapping_RFE'}
y_proxy_options = ['score_change', 'final_score']  # what the target for the regressors will be

if __name__ == "__main__":

    if True:
        exp_name = "test_jj_auc2"
        out_path = os.path.join(REG_RESULTS_DIR, exp_name)

        runs = 10
        regressors = ['sgdReg']
        X_paths = ['X_train_norm_over', 'X_train_norm', 'X_train_norm_select', 'X_train_norm_over_select']
        y = "y_train"
        y_proxies = ["final_score"]

        if all(option in regressor_options for option in regressors):
            print("Validated Regressor Options")
        else: raise Exception(f"Invalid regressor option. Valid options are {regressor_options}")

        if all(option in X_path_options for option in X_paths):
            print("Validated X_path")
        else: raise Exception(f"Invalid X_path option. Valid options are {X_path_options}")

        if all(option in y_proxy_options for option in y_proxies):
            print("Validated y_proxy")
        else: raise Exception(f"Invalid y_proxy option. Valid options are {y_proxy_options}")

        # Makes sure not to overwrite any files
        if os.path.isdir(out_path):
            raise Exception("Name already exists")
        else:
            os.mkdir(out_path + "/")

        exp_summary = {'model':[],'target':[], 'features':[], 'train_RMSE':[],'train_R2':[], 'valid_RMSE':[], 'valid_R2':[], 
                            'CV_train_resp_bal_acc':[], 'CV_valid_resp_bal_acc':[], 'valid_resp_auc':[],  'CV_train_rem_bal_acc':[], 'CV_valid_rem_bal_acc':[] }
        test_summaries = []
        for regressor in regressors:
            for y_proxy in y_proxies:
                for X_path in X_paths:

                    df_filename = "{}_{}_{}".format(regressor, X_path, y_proxy)
                    run_results = RunRegRun(regressor, X_path, y, y_proxy, out_path,   runs)
                    test_results = evaluate_on_test(regressor, X_path, y_proxy, out_path)
                    test_summaries.append(test_results)
                    exp_summary['model'].append(regressor)
                    exp_summary['target'].append(y_proxy)
                    exp_summary['features'].append(X_path_mapping[X_path])
                    exp_summary['train_RMSE'].append(run_results['avg_train_RMSE'].mean())
                    exp_summary['train_R2'].append(run_results['avg_train_R2'].mean())
                    exp_summary['valid_RMSE'].append(run_results['avg_valid_RMSE'].mean())
                    exp_summary['valid_R2'].append(run_results['avg_valid_R2'].mean())
                    exp_summary['valid_resp_auc'].append(run_results['avg_valid_resp_auc'].mean())
                    exp_summary['CV_train_resp_bal_acc'].append(run_results['avg_train_resp_bal_acc'].mean())
                    exp_summary['CV_valid_resp_bal_acc'].append(run_results['avg_valid_resp_bal_acc'].mean())

                    exp_summary['CV_train_rem_bal_acc'].append(run_results['avg_train_rem_bal_acc'].mean())
                    exp_summary['CV_valid_rem_bal_acc'].append(run_results['avg_valid_rem_bal_acc'].mean())


                    run_results.to_csv(out_path + '/' + df_filename + ".csv", index = False)
                    test_results.to_csv(out_path + '/test_'+ df_filename + ".csv", index = False)
        
        exp_df = pd.DataFrame(exp_summary)
        print(exp_df)
        exp_df.to_csv(out_path + '/summary.csv', index = False)

        test_summary_df = pd.concat(test_summaries, axis = 0).drop(columns = ['run'])
        test_summary_df = test_summary_df.groupby('model').agg('mean')
        test_summary_df.to_csv(out_path + '/test_summary.csv', index = True)
    
    # if True:

    #     runs = 1
    #     classifiers = ["rf"]
    #     X_paths = ["X_train_norm"]
    #     y = "y_train"
    #     y_labels = ['response']

    #     for classifier in classifiers:
    #         for X_path in X_paths:
    #             for y_label in y_labels:
    #                 RunClassRun(classifier,X_path, y, y_label, "blank", runs)

# On how to do ROC
    # However, you could use a decision rule to assign that example to a class. 
    # The obvious decision rule is to assign it to the more likely class:
    #  the positive one if the probability is at least a half, and the negative one otherwise. 
    #  By varying this decision rule (e.g., an example is in the positive class if P(class=+)>{0.25,0.5,0.75,etc},
    #   you can turn the TP/FP knob and generate an ROC curve.



