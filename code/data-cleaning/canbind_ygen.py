import os
import csv
import pandas as pd
import numpy as np
import sys

from canbind_globals import ORIGINAL_SCALE_FILENAMES, COL_NAME_PATIENT_ID,COL_NAME_EVENTNAME, COL_NAMES_WHITELIST_PSYHIS, COL_NAME_GROUP, GROUP_WHITELIST, YGEN_EVENTNAME_WHITELIST,  QLESQ_COL_MAPPING, COL_NAMES_ONE_HOT_ENCODE, COL_NAMES_BLACKLIST_UNIQS, TARGET_MAP, VALUE_REPLACEMENT_MAPS, YGEN_COL_NAMES_TO_CONVERT, COL_NAMES_BLACKLIST_PSYHIS, COL_NAMES_BLACKLIST_DARS, COL_NAMES_BLACKLIST_SHAPS
from canbind_globals import COL_NAMES_NEW_FROM_EXTENSION, COL_NAMES_TO_DROP_FROM_EXTENSION
from canbind_utils import get_event_based_value, aggregate_rows, finalize_blacklist, one_hot_encode, merge_columns, add_columns_to_blacklist
from canbind_utils import is_number, replace_target_col_values, replace_target_col_values_to_be_refactored, collect_columns_to_extend
from utils import get_valid_subjects
""" 
Generates a y-matrix from the CAN-BIND data. Similiar code to canbind_data_processor, separated to ensure no week8 contamination to data matrix

Example usages

    Basic:
        python canbind_ygen.py /path/to/data/folders

This will output a single CSV file containing the y-matrix

The method expects CSV files to be contained within their own subdirectories from the root directory, as is organized
in the ZIP provided.
"""
def ygen(root_dir):
    global UNIQ_COLUMNS
    global COL_NAMES_CATEGORICAL
    global COL_NAMES_NA
    global FILENAMES
    global NUM_DATA_FILES
    global NUM_DATA_ROWS
    global NUM_DATA_COLUMNS

    global COL_NAMES_BLACKLIST_PSYHIS
    global COL_NAMES_BLACKLIST_DARS
    global COL_NAMES_BLACKLIST_SHAPS
    global COL_NAMES_DARS_TO_CONVERT

    uniq_columns = {}
    col_names_categorical = {}
    col_names_na = {}

    filenames = []

    num_data_files = 0
    num_data_rows = 0
    num_data_columns = 0

    merged_df = pd.DataFrame([])

    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            file_path = os.path.join(subdir, filename)

            if filename in ORIGINAL_SCALE_FILENAMES:
                filenames.append(filename)
                num_data_files += 1
                
                # Convert to csv if data file is an xlsx
                root, ext = os.path.splitext(file_path)
                if (ext == '.xlsx'):
                    read_xlsx = pd.read_excel(file_path)
                    # IPAQ File uses "EVENTME" instead of "EVENTNAME", so replace
                    if "IPAQ" in filename:
                        read_xlsx = read_xlsx.rename({'EVENTME' : 'EVENTNAME'}, axis='columns', errors='raise')
                    file_path = root + '.csv'
                    read_xlsx.to_csv(file_path, index = None, header=True)
                elif (ext != '.csv'):
                    raise Exception("Provided a data file that is neither an xlsx or csv")
                
                # Track counts and column names for sanity check
                with open(file_path, 'rt') as csvfile:
                    col_names = []
                    csv_reader = csv.reader(csvfile)
                    for i, row in enumerate(csv_reader):
                        num_data_rows += 1

                        # Store the column names
                        if i == 0:
                            col_names = row
                            num_data_columns += len(row)
                            for field in row:
                                field = field.upper()
                                if field in uniq_columns:
                                    uniq_columns[field] += 1
                                else:
                                    uniq_columns[field] = 1

                                if field.startswith("DARS_"):
                                    COL_NAMES_BLACKLIST_DARS.append(field)
                                    continue
                                if field.startswith("SHAPS_"):
                                    COL_NAMES_BLACKLIST_SHAPS.append(field)
                                    continue
                                if field.startswith("PSYHIS_") and field not in COL_NAMES_WHITELIST_PSYHIS:
                                    COL_NAMES_BLACKLIST_PSYHIS.append(field)
                                    continue

                                # Collect names of columns that will be extended with extra columns based on event value
                                collect_columns_to_extend(field)
                        else:
                            # Determine all columns with categorical values
                            for j, field_value in enumerate(row):
                                col_name = col_names[j].upper()
                                if field_value == "":
                                    continue
                                elif is_number(field_value):
                                    continue
                                elif field_value == "NA":
                                    col_names_na[col_name] = True
                                else:
                                    col_names_categorical[col_name] = True

                csvfile.close()

                df = pd.read_csv(file_path)

                # Convert all column names to upper case to standardize names
                df.rename(columns=lambda x: x.upper(), inplace=True)

                # Append the CSV dataframe
                merged_df = merged_df.append(df, sort=False)

    # Sort the rows by the patient identifier
    merged_df = merged_df.sort_values(by=[COL_NAME_PATIENT_ID])

    # Back up full merged file for debugging purposes
    merged_df.to_csv(root_dir + "/merged-data.unprocessed_ygen.csv")

    #### FILTER ROWS AND COLUMNS ####

    # Filter out rows that are controls
    if COL_NAME_GROUP in merged_df:
        merged_df = merged_df.loc[~merged_df.GROUP.str.lower().isin(GROUP_WHITELIST)]
    
    # Filter out rows that were recorded beyond Week 8
    if COL_NAME_EVENTNAME in merged_df:
        merged_df = merged_df.loc[merged_df.EVENTNAME.str.lower().isin(YGEN_EVENTNAME_WHITELIST)]

    #### CREATE NEW COLUMNS AND MERGE ROWS ####

    # Handle column extension based on EVENTNAME or VISITSTATUS
    merged_df = extend_columns_eventbased(merged_df)

    # Collapse/merge patient rows
    merged_df = aggregate_rows(merged_df)

    # Handle replacing values in specific columns, see @VALUE_REPLACEMENT_MAPS
    merged_df = replace_target_col_values_to_be_refactored(merged_df, VALUE_REPLACEMENT_MAPS)

    # Merge QLESQ columns
    merged_df = merge_columns(merged_df, QLESQ_COL_MAPPING)

    # First replace empty strings with np.nan, as pandas knows to ignore creating one-hot columns for np.nan
    # This step is necessary for one-hot encoding and for replacing nan values with a median
    merged_df = merged_df.replace({"": np.nan})

    # One-hot encode specific columns, see @COL_NAMES_ONE_HOT_ENCODE
    merged_df = one_hot_encode(merged_df, COL_NAMES_ONE_HOT_ENCODE)

    # Finalize the blacklist, then do a final drop of columns (original ones before one-hot and blacklist columns)
    add_columns_to_blacklist
    finalize_blacklist()
    merged_df.drop(COL_NAMES_BLACKLIST_UNIQS, axis=1, inplace=True)
    
    
    
    # Create y target, eliminate invalid subjects in both X and y (those who don't make it to week 8), convert responder/nonresponder string to binary
    merged_df = get_valid_subjects(merged_df)
    merged_df = merged_df.drop(["RESPOND_WK8"], axis=1)
    ##print(merged_df)
    merged_df = replace_target_col_values(merged_df, [TARGET_MAP])
    merged_df = merged_df.sort_values(by=[COL_NAME_PATIENT_ID])
    merged_df = merged_df.reset_index(drop=True)
    
    # Fix a value in the data that was messed up in a recent version (pt had age of 56, switched to 14 recently, so switched back)
    if merged_df.at[68, 'AGE'] == 16:
        merged_df.at[68, 'AGE'] = 56
        print("Replaced misrecorded age")
        
    # Rename the column that will be used for the y value (target)
    merged_df = merged_df.rename({"QIDS_RESP_WK8_week 2":"QIDS_RESP_WK8"},axis='columns',errors='raise')
    
    # Replace missing "QIDS_RESP_WK8" values by manually checking criteria
    for i, row in merged_df.iterrows():
        if "QIDS_RESP_WK8" in row:
            if np.isnan(row["QIDS_RESP_WK8"]):
                baseline_qids_sr = row['QIDS_OVERL_SEVTY_baseline']
                week2_qids_sr = row['QIDS_OVERL_SEVTY_week 2']
                week8_qids_sr = row['QIDS_OVERL_SEVTY_week 8']
                
                if week2_qids_sr <= baseline_qids_sr*0.50 or week8_qids_sr <= baseline_qids_sr*0.50:
                    merged_df.at[i, 'QIDS_RESP_WK8'] = 1
                else:
                    merged_df.at[i, 'QIDS_RESP_WK8'] = 0
    
    
    y_df = merged_df[['SUBJLABEL','QIDS_RESP_WK8']]
    y_df.to_csv(root_dir + "/canbind-targets.csv", index=False)
    
    # Save the version containing NaN values just for debugging, not otherwise used
    merged_df.to_csv(root_dir + "/canbind-clean-aggregated-data.with-id.contains-blanks-ygen.csv")



def extend_columns_eventbased(orig_df):
    """
    Handles adding extra columns based on a condition the value of another column.

    :param orig_df: the original dataframe
    :return: a new, modified dataframe
    """
    global COL_NAMES_NEW_FROM_EXTENSION
    global COL_NAMES_TO_DROP_FROM_EXTENSION
        
    # Create extra columns with name of event appended, initialized blank
    for scale_group in YGEN_COL_NAMES_TO_CONVERT:
        scale_name = scale_group[0]
        scale_events_whitelist = scale_group[1]
        col_names = scale_group[2]

        for col_name in col_names:
            for event in scale_events_whitelist:
                # Only add extra columns for entries with a non-empty and valid event
                if type(event) != type("") or is_number(event):
                    continue

                new_col_name = col_name + "_" + event

                # Add columns to this list
                COL_NAMES_NEW_FROM_EXTENSION.append(new_col_name)

                # Set the value for the new column
                orig_df[new_col_name] = orig_df.apply(lambda row: get_event_based_value(row, event, col_name, scale_name), axis=1)

        COL_NAMES_TO_DROP_FROM_EXTENSION.extend(col_names)

    return orig_df


if __name__ == "__main__":
    if len(sys.argv) == 1 and os.path.isdir(sys.argv[1]):
        ygen(sys.argv[1])
    elif len(sys.argv) == 0:
        pathData = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\canbind_data_full_auto\\'
        ygen(pathData, verbose=False)
    else:
        print("Enter valid arguments\n"
               "\t path: the path to a real directory\n")
