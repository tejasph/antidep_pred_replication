#Regression Results
from run_regression import RunRegRun, evaluate_on_test
from run_globals import REG_RESULTS_DIR
import pandas as pd
import os

"""
Run Experiment script

Change variables and experiment name using this script. 
"""

if __name__ == "__main__":

    exp_name = "12_RMSE_prelim_run"
    out_path = os.path.join(REG_RESULTS_DIR, exp_name)

    # Makes sure not to overwrite any files
    if os.path.isdir(out_path):
        raise Exception("Name already exists")
    else:
        os.mkdir(out_path + "/")

    runs = 10
    regressors = ["rf", "gbdt", "sgdReg"]
    X_paths = ["X_train_norm", "X_train_norm_over"]
    y = "y_train"
    y_proxies = ["score_change", "final_score"]
    test_data = True

    exp_summary = {'model':[],'target':[], 'features':[], 'train_RMSE':[], 'valid_RMSE':[],
                        'CV_train_resp_bal_acc':[], 'CV_valid_resp_bal_acc':[],  'CV_train_rem_bal_acc':[], 'CV_valid_rem_bal_acc':[] }

    for regressor in regressors:
        for y_proxy in y_proxies:
            for X_path in X_paths:
                df_filename = "{}_{}_{}".format(regressor, X_path, y_proxy)
                run_results = RunRegRun(regressor, X_path, y, y_proxy, out_path,  runs , test_data)
                test_results = evaluate_on_test(regressor, X_path, y_proxy, out_path)
                exp_summary['model'].append(regressor)
                exp_summary['target'].append(y_proxy)
                exp_summary['features'].append("All" if X_path == "X_train_norm" else "Overlapping")
                exp_summary['train_RMSE'].append(run_results['avg_train_RMSE'].mean())
                exp_summary['valid_RMSE'].append(run_results['avg_valid_RMSE'].mean())
                exp_summary['CV_train_resp_bal_acc'].append(run_results['avg_train_resp_bal_acc'].mean())
                exp_summary['CV_valid_resp_bal_acc'].append(run_results['avg_valid_resp_bal_acc'].mean())

                exp_summary['CV_train_rem_bal_acc'].append(run_results['avg_train_rem_bal_acc'].mean())
                exp_summary['CV_valid_rem_bal_acc'].append(run_results['avg_valid_rem_bal_acc'].mean())


                run_results.to_csv(out_path + '/' + df_filename + ".csv", index = False)
                test_results.to_csv(out_path + '/test_'+ df_filename + ".csv", index = False)
    
    exp_df = pd.DataFrame(exp_summary)
    print(exp_df)
    exp_df.to_csv(out_path + '/summary.csv', index = False)



