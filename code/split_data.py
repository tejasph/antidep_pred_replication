# prepare_data.py
# June 24th 2021


'''This script split data with a set random_state for reproducibility. It also creates scaled versions of the data 

Usage: split_data.py --X_path=<X_path> --y_path=<y_path>


Options: 
--X_path=<X_path>   :   Relative folder path for features
--y_path=<y_path>   :   Relative folder path for labels

''' 

import pandas as pd
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline


if __name__ == "__main__":

    startTime = datetime.datetime.now()

    out_path = "data/modelling_data"
    X_path = "data/X_tillwk4_qids_sr__final.csv"
    y_path = "data/y_wk8_resp_mag_qids_sr__final.csv"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Read in cleaned csv
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

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
    
    # Create normalized version of X_train
    scaler = MinMaxScaler()
    X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns, index = X_train.index)
    X_test_norm = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns, index = X_test.index)

    # Crude way of determining categorical variables
    cat_cols = []
    num_cols = []
    for col in X_train.columns:
        val_nums = len(X_train[col].unique())
        if val_nums <= 2:
            cat_cols.append(col)
        else:
            num_cols.append(col)  

    # Create standardized version of X_train and X_test
    num_transformer = Pipeline([('standardize', StandardScaler())])
    ct = ColumnTransformer([('stand', num_transformer, num_cols)], remainder = 'passthrough')

    X_train_stand = pd.DataFrame(ct.fit_transform(X_train), columns = X_train.columns, index = X_train.index)
    X_test_stand = pd.DataFrame(ct.transform(X_test), columns = X_test.columns, index = X_test.index)

    # Create standardized/normalized version of X_train and X_test
    num_transformer = Pipeline([('standardize', StandardScaler()),('normalize',MinMaxScaler())])
    ct = ColumnTransformer([('stand_norm', num_transformer, num_cols)], remainder = 'passthrough')
    
    X_train_stand_norm = pd.DataFrame(ct.fit_transform(X_train), columns = X_train.columns, index = X_train.index)
    X_test_stand_norm = pd.DataFrame(ct.transform(X_test), columns = X_test.columns, index = X_test.index)

  
    # Output csv files
    X_train.to_csv(out_path + "/X_train.csv", index = True)
    X_train_norm.to_csv(out_path + "/X_train_norm.csv", index = True)
    X_train_stand.to_csv(out_path + "/X_train_stand.csv" , index = True)
    X_train_stand_norm.to_csv(out_path + "/X_train_stand_norm.csv", index = True)
    y_train.to_csv(out_path + "/y_train.csv", index = False) 

    X_test.to_csv(out_path + "/X_test.csv", index = False)
    X_test_norm.to_csv(out_path + "/X_test_norm.csv", index = True)
    X_test_stand.to_csv(out_path + "/X_test_stand.csv", index = True)
    X_test_stand_norm.to_csv(out_path + "/X_test_stand_norm.csv", index = True)
    y_test.to_csv(out_path + "/y_test.csv", index = False) 

print(f"Finished data prep in {datetime.datetime.now() - startTime}")
