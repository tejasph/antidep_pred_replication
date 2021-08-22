# prepare_data.py
# June 24th 2021


'''This script split data with a set random_state for reproducibility. It also creates scaled versions of the data 

Example Usage: python code/split_data.py -over data/jj_processed/X_overlap_tillwk4_qids_sr.csv

Run from root of the repo

''' 

import pandas as pd
import numpy as np
import datetime
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from run_globals import REG_MODEL_DATA_DIR, REG_PROCESSED_DATA, ALL_CONT_VARS, ALL_CAT_DICT, ALL_CAT_VARS, ALL_ORD_VARS, ALL_BINARY_VARS, OVER_BINARY_VARS, OVER_CAT_VARS, OVER_CAT_DICT, OVER_CONT_VARS, OVER_ORD_VARS


def prepare_data(X_path, name):

    startTime = datetime.datetime.now()

    if name == "_over":
        CAT_VARS = OVER_CAT_VARS
        CAT_DICT = OVER_CAT_DICT
        ORD_VARS = OVER_ORD_VARS
        CONT_VARS = OVER_CONT_VARS
        BINARY_VARS = OVER_BINARY_VARS
    else:
        CAT_VARS = ALL_CAT_VARS
        CAT_DICT = ALL_CAT_DICT
        ORD_VARS = ALL_ORD_VARS
        CONT_VARS = ALL_CONT_VARS
        BINARY_VARS = ALL_BINARY_VARS
    
    y_path = os.path.join(REG_PROCESSED_DATA, "y_wk8_resp_mag_qids_sr__final.csv")

    if not os.path.exists(REG_MODEL_DATA_DIR):
        os.mkdir(REG_MODEL_DATA_DIR)
    out_path = REG_MODEL_DATA_DIR
    
    # Read in cleaned csv
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    # Get a copy that will be for external validation
    X_ext_train = X.copy()

    # Split the data at a 8:2 ratio, with random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
    
    print(f"X_train shape is {X_train.shape}")
    print(f"y_train shape is {y_train.shape}")

    print(f"X_test shape is {X_test.shape}")
    print(f"y_test shape is {y_test.shape}")
    
    #Checks that same number of rows are in the X and y dfs
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    
    # checks that all subject ids match between X and y
    assert X_train.subjectkey.compare(y_train.subjectkey).shape[0] == 0
    assert X_test.subjectkey.compare(y_test.subjectkey).shape[0] == 0

    # Associate subjectkey as the index for easier tracking/manipulation
    X_train = X_train.set_index('subjectkey')
    X_test = X_test.set_index('subjectkey')
    X_ext_train = X_ext_train.set_index('subjectkey')


    print("Centering and Scaling Data")
    # Center Binary Variables
    for var in BINARY_VARS:
        X_train[var] = X_train[var].apply(lambda x:-0.5 if x == 0 else 0.5)
        X_test[var] = X_test[var].apply(lambda x:-0.5 if x == 0 else 0.5)
        X_ext_train[var] = X_ext_train[var].apply(lambda x:-0.5 if x == 0 else 0.5)

    # Center Categorical one-hot encoded data
    for key, value in CAT_DICT.items():
        sub_vals = {"zero":-1/value, "one": 1-(1/value)}
        subset_feats = []
        for var in CAT_VARS:
            if (key + "||") in var:
                X_train[var] = X_train[var].apply(lambda x:sub_vals["zero"] if x == 0 else sub_vals["one"])
                X_test[var] = X_test[var].apply(lambda x:sub_vals["zero"] if x == 0 else sub_vals["one"])
                X_ext_train[var] = X_ext_train[var].apply(lambda x:sub_vals["zero"] if x == 0 else sub_vals["one"])

    # Center ordinal data by median
    for var in ORD_VARS:
        median = np.median(X_train[var])
        ext_median = np.median(X_ext_train[var])

        X_train[var] = X_train[var].apply(lambda x: x- median)
        X_test[var] = X_test[var].apply(lambda x: x- median)
        X_ext_train[var] = X_ext_train[var].apply(lambda x: x- ext_median)

    # Bounds ordinal data within a certain range to be on comparables scale with other categories
    minmax = MinMaxScaler(feature_range = (-1,1))
    X_train_ord = pd.DataFrame(minmax.fit_transform(X_train[ORD_VARS]), columns = ORD_VARS, index = X_train.index)
    X_test_ord = pd.DataFrame(minmax.transform(X_test[ORD_VARS]), columns = ORD_VARS, index = X_test.index)
    X_ext_train_ord = pd.DataFrame(minmax.fit_transform(X_ext_train[ORD_VARS]), columns = ORD_VARS, index = X_ext_train.index)


    # Centers continuous data using mean and std.
    stand_scaler = StandardScaler()
    X_train_cont = pd.DataFrame(stand_scaler.fit_transform(X_train[CONT_VARS]), columns = CONT_VARS, index = X_train.index)
    X_test_cont = pd.DataFrame(stand_scaler.transform(X_test[CONT_VARS]), columns = CONT_VARS, index = X_test.index)
    X_ext_train_cont = pd.DataFrame(stand_scaler.fit_transform(X_ext_train[CONT_VARS]), columns = CONT_VARS, index = X_ext_train.index)

    # potentially rename
    X_train_norm = pd.concat([X_train_cont, X_train_ord, X_train[BINARY_VARS], X_train[CAT_VARS]], axis =1 )
    X_test_norm = pd.concat([X_test_cont, X_test_ord, X_test[BINARY_VARS], X_test[CAT_VARS]], axis = 1)

    X_ext_train_processed = pd.concat([X_ext_train_cont, X_ext_train_ord, X_ext_train[BINARY_VARS], X_ext_train[CAT_VARS]], axis =1 )
    print(X_ext_train_processed.shape)

    

    #Temporary variable drops d/t poor dists
    potential_removals = ['dm01_enroll__resm','dm01_enroll__relat','dm01_enroll__frend','dm01_enroll__thous','dm01_w0__mempl','dm01_w0__massist','dm01_w0__munempl','dm01_w0__minc_other',
 'wpai01__wpai02','wpai01__wpai03','wpai01__wpai04','wpai01__wpai_totalhrs','wpai01__wpai_pctmissed','wpai01__wpai_pctworked','wpai01__wpai_pctwrkimp','phx01__epino','phx01__episode_date', 'ucq01__ucq030',
                     'ucq01__ucq091','ucq01__ucq100','ucq01__ucq130','ucq01__ucq150','ucq01__ucq170','ucq01__ucq050','ucq01__ucq070']
    # X_train_norm = X_train_norm.drop(columns = potential_removals)
    # X_test_norm = X_test_norm.drop(columns = potential_removals)
    print(X_train_norm.head())



    #Below is from the original script#################################
    # Create normalized version of X_train
    # scaler = MinMaxScaler()
    # X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns, index = X_train.index)
    # X_test_norm = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns, index = X_test.index)

    # # Crude way of determining categorical variables
    # cat_cols = []
    # num_cols = []
    # for col in X_train.columns:
    #     val_nums = len(X_train[col].unique())
    #     if val_nums <= 2:
    #         cat_cols.append(col)
    #     else:
    #         num_cols.append(col)  

    # # Create standardized version of X_train and X_test
    # num_transformer = Pipeline([('standardize', StandardScaler())])
    # ct = ColumnTransformer([('stand', num_transformer, num_cols)], remainder = 'passthrough')

    # X_train_stand = pd.DataFrame(ct.fit_transform(X_train), columns = X_train.columns, index = X_train.index)
    # X_test_stand = pd.DataFrame(ct.transform(X_test), columns = X_test.columns, index = X_test.index)

    # # Create standardized/normalized version of X_train and X_test
    # num_transformer = Pipeline([('standardize', StandardScaler()),('normalize',MinMaxScaler())])
    # ct = ColumnTransformer([('stand_norm', num_transformer, num_cols)], remainder = 'passthrough')
    
    # X_train_stand_norm = pd.DataFrame(ct.fit_transform(X_train), columns = X_train.columns, index = X_train.index)
    # X_test_stand_norm = pd.DataFrame(ct.transform(X_test), columns = X_test.columns, index = X_test.index)

    
    # Output csv files
    #X_train is now modified d/t to recent changes
    X_train.to_csv(out_path + "/X_train" + name + ".csv", index = True)
    X_train_norm.to_csv(out_path + "/X_train_norm" + name + ".csv", index = True)

    if name == "_over": # only output the df if overlapping features are selected
        X_ext_train_processed.to_csv(out_path + "/X_ext_train_processed.csv", index = True)
    # X_train_stand.to_csv(out_path + "/X_train_stand" + name + ".csv" , index = True)
    # X_train_stand_norm.to_csv(out_path + "/X_train_stand_norm" + name + ".csv", index = True)
    y_train.to_csv(out_path + "/y_train" + ".csv", index = False) # y_train and y_test aren't affected by overlapping feats --> that's why no name variable used

    X_test.to_csv(out_path + "/X_test" +name + ".csv", index = False)
    X_test_norm.to_csv(out_path + "/X_test_norm" + name + ".csv", index = True)
    # X_test_stand.to_csv(out_path + "/X_test_stand" + name + ".csv", index = True)
    # X_test_stand_norm.to_csv(out_path + "/X_test_stand_norm" + name + ".csv", index = True)
    y_test.to_csv(out_path + "/y_test" + ".csv", index = False) 

    print(f"Finished data prep in {datetime.datetime.now() - startTime}")

if __name__ == "__main__":
    print(sys.argv[1])
    print(sys.argv[2])
    if sys.argv[1] == "-over":
        name = "_over"
    elif sys.argv[1] == "-all":
        name = ""
    else:
        print("option not typed correctly")

    # Need a check on this
    prepare_data(sys.argv[2], name)
 
