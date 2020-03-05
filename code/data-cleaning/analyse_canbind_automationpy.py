import os
import sys
import pandas as pd
import numpy as np

#Quick script to ensure the automatically made version of canbind data is same as the manually made


from utils import *

old_canbind = pd.read_csv(r'C:/Users/jjnun/Documents/Sync/Research/1_CANBIND Replication/teyden-git/data/canbind_data_full_auto/canbind-clean-aggregated-data.with-id.contains-blanks-with-qidssr_TOREPLICATE.csv')
new_canbind = pd.read_csv(r'C:/Users/jjnun/Documents/Sync/Research/1_CANBIND Replication/teyden-git/data/canbind_data_full_auto/canbind-clean-aggregated-data.with-id.contains-blanks.csv')


all_columns_found = True
for col in old_canbind.columns:
    if col in new_canbind.columns:
        continue
    else:
        print("This column was not found in new version of CANBIND data: " + col)
        all_columns_found = False

print("Columns found in both?: " + str(all_columns_found))

cols_equal = 0
cols_unequal = 0
all_columns_equal = True
for col in old_canbind.columns:
    old_col = old_canbind[col].replace(np.nan,0)
    new_col = new_canbind[col].replace(np.nan,0)
    
    if old_col.dtype == 'int64':
        old_col = old_col.astype('float64')
    if old_col.equals(new_col):
        cols_equal = cols_equal + 1
    elif col == 'QIDS_RESP_WK8': 
        continue
    elif (old_col - new_col).abs().sum(axis=0) == 0: # do it this way due to some odd stuff
        cols_equal = cols_equal + 1
    else:
        print("This column was not equal to the old version of CANBIND data: " + col)
        cols_unequal = cols_unequal + 1
        all_columns_equal = False

for i in range(len(old_canbind.index)):
    col_test = 'QIDS_SR_6_week2'
    if (old_canbind.replace(np.nan,0).at[i, col_test] != new_canbind.replace(np.nan,0).at[i, col_test]):
        print("Error at index: " + str(i) + " of patient: " + str(old_canbind.at[i,"SUBJLABEL"]) + " with values : " + str(old_canbind.at[i, col_test]) + " " + str(new_canbind.at[i,col_test]))

print("Columns equal in both?: " + str(all_columns_equal))
print("Columns equal: " + str(cols_equal))
print("Columns unequal: " + str(cols_unequal))

#print((old_canbind['EMPLOY_STATUS_1.0'] - new_canbind['EMPLOY_STATUS_1.0']).sum())
#print(new_canbind['EMPLOY_STATUS_1.0'])