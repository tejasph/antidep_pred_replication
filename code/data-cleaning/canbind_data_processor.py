import os
import csv
import pandas as pd
import re

from canbind_globals import *
from utils import * #ORIGINAL_SCALE_FILENAMES

# TODO ADD TESTS - SANITY CHECKS FOR DATA INTEGRITY BETWEEN UNPROCESSSED AND FINAL SET

""" 
Cleans and aggregates CAN-BIND data.

Example usages

    Basic:
        python canbind_data_processor.py /path/to/data/folders

    Verbose:
        python canbind_data_processor.py -v /path/to/data/folders

    Super verbose:
        python canbind_data_processor.py -v+ /path/to/data/folders


This will output a single CSV file containing the merged and clean data.

The method expects CSV files to be contained within their own subdirectories from the root directory, as is organized
in the ZIP provided.
"""
def aggregate_and_clean(root_dir, verbose=False, extra=False):
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
    merged_df.to_csv(root_dir + "/merged-data.unprocessed.csv")

    #### FILTER ROWS AND COLUMNS ####

    # Filter out rows that are controls
    if COL_NAME_GROUP in merged_df:
        merged_df = merged_df.loc[~merged_df.GROUP.str.lower().isin(GROUP_WHITELIST)]

    # Filter out rows that were recorded beyond Week 2
    if COL_NAME_EVENTNAME in merged_df:
        merged_df = merged_df.loc[merged_df.EVENTNAME.str.lower().isin(EVENTNAME_WHITELIST)]

    #### CREATE NEW COLUMNS AND MERGE ROWS ####

    # Handle column extension based on EVENTNAME or VISITSTATUS
    merged_df = extend_columns_eventbased(merged_df)

    # Collapse/merge patient rows
    merged_df = aggregate_rows(merged_df)

    # Handle replacing values in specific columns, see @VALUE_REPLACEMENT_MAPS
    merged_df = replace_target_col_values_to_be_refactored(merged_df, VALUE_REPLACEMENT_MAPS) # TODO no time to refactor atm

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
    ##print(merged_df)
    ##targets = merged_df[[COL_NAME_PATIENT_ID, "QIDS_RESP_WK8_week 2"]]
    ##targets.to_csv(root_dir + "/canbind-targets.csv", index=False)
    ##merged_df.drop(["QIDS_RESP_WK8"], axis=1, inplace=True)
    
    
    
    # Rename the column that will be used for the y value (target)
    merged_df = merged_df.rename({"QIDS_RESP_WK8_week 2":"QIDS_RESP_WK8"},axis='columns',errors='raise')
    # Replace "week 2" with "week2" in column names
    merged_df = merged_df.rename(columns=lambda x: re.sub('week 2','week2',x))
    # Save the version containing NaN values
    merged_df.to_csv(root_dir + "/canbind-clean-aggregated-data.with-id.contains-blanks.csv")

    ## Handle imputation with separate file
    
    # Replace all NaN values with median for a column
    ##merged_df_without_blanks = replace_nan_with_median(merged_df)

    # Save the version without NaN values
    ##merged_df_without_blanks.to_csv(root_dir + "/canbind-clean-aggregated-data.with-id.csv")

    # Remove IDs and write to CSVs
    ##merged_df.drop([COL_NAME_PATIENT_ID], axis=1).to_csv(root_dir + "/canbind-clean-aggregated-data.contains-blanks.csv")
    ##merged_df_without_blanks.drop([COL_NAME_PATIENT_ID], axis=1).to_csv(root_dir + "/canbind-clean-aggregated-data.csv")

    if verbose:
        UNIQ_COLUMNS = uniq_columns
        COL_NAMES_CATEGORICAL = col_names_categorical
        COL_NAMES_NA = col_names_na
        FILENAMES = filenames
        NUM_DATA_FILES = num_data_files
        NUM_DATA_ROWS = num_data_rows
        NUM_DATA_COLUMNS = num_data_columns
        print_info(merged_df, extra)

def create_sum_column(df, scale_col_names, new_col_name):
    new_col = []
    for index, row in df.iterrows():
        sum = 0
        for sub_col in scale_col_names:
            val = row[sub_col]
            if val == "":
                continue
            elif not is_number(val):
                print("\t%s - %s: not a number [%s]" % (row[COL_NAME_PATIENT_ID], sub_col, str(val)))
                continue
            sum += row[sub_col]
        new_col.append(sum)
    df[new_col_name] = new_col
    return df

def replace_nan_with_median(df):
    for col_name in df.columns.values:
        if col_name == COL_NAME_PATIENT_ID or col_name == "RESPOND_WK8":
            continue
        df[col_name] = df[col_name].replace(np.nan, df[col_name].median())
    return df

def one_hot_encode(df, columns):
    # Convert categorical variables to indicator variables via one-hot encoding
    for col_name in columns:
        df = pd.concat([df, pd.get_dummies(df[col_name], prefix=col_name)], axis=1)

    add_columns_to_blacklist(columns)
    return df

def add_columns_to_blacklist(col_names):
    global COL_NAMES_BLACKLIST_UNIQS
    COL_NAMES_BLACKLIST_UNIQS.extend(col_names)

def aggregate_patient_rows(df):
    """
    Aggregates groups of patient rows corresponding to a single patient to a single row.

    :param df: the dataframe
    :return: a new dataframe consisting of one row per patient
    """
    new_df = pd.DataFrame()
    grouped = df.groupby([COL_NAME_PATIENT_ID])
    for patient_id, group_of_rows_df in grouped:
        agg_patient_vals = [(COL_NAME_PATIENT_ID, [patient_id])]

        # Iterate over columns to grab the values for aggregation, and determine which to keep
        for column, values in group_of_rows_df.iteritems():
            if column == COL_NAME_PATIENT_ID:
                continue

            uniqs_counter = {}
            val_to_keep = ""

            for val in values:
                if val == None:
                    continue
                if val is np.nan or val != val:
                    continue
                if val == float('nan'):
                    continue
                if val == "" or val == "NA":
                    continue

                # Standardize with lowercases
                if column == "RESPOND_WK8" and type(val) == type(""):
                    val = val.lower()

                # For debugging purposes later
                if val in uniqs_counter:
                    uniqs_counter[val] += 1
                else:
                    uniqs_counter[val] = 1

                val_to_keep = val

            # Decide which value to store for this column
            # If num uniqs is 0 then saves a blank, if 1 then saves the single value. If greater than 1, then saves "collision".
            if len(uniqs_counter) > 1:
                agg_patient_vals.append((column, ["[collision]" + str(uniqs_counter)]))
            else:
                agg_patient_vals.append((column, [val_to_keep]))

        new_df = new_df.append(pd.DataFrame.from_items(agg_patient_vals))

    print_progress_completion(aggregate_patient_rows, "aggregated groups of patient rows to a single row")
    return new_df


def aggregate_rows(df):
    """
    Aggregates groups of patient rows corresponding to a single patient to a single row.

    :param df: the dataframe
    :return: a new dataframe consisting of one row per patient
    """
    new_df = pd.DataFrame()
    grouped = df.groupby([COL_NAME_PATIENT_ID])
    i = 0
    num_collisions = 0
    num_collisions_handled = 0
    collisions = {}
    conversions = {}
    for patient_id, group_of_rows_df in grouped:
        agg_patient_vals = [(COL_NAME_PATIENT_ID, [patient_id])]

        # Iterate over columns to grab the values for aggregation, and determine which to keep
        for column, values in group_of_rows_df.iteritems():
            if column == COL_NAME_PATIENT_ID:
                continue

            column_collisions = []
            column_conversions = []

            uniqs_counter = {}
            val_to_keep = ""

            for val in values:
                # Skip blank/NA/NAN values. Only want to grab the real values to save.
                if val == None:
                    continue
                if val is np.nan or val != val:
                    continue
                if val == float('nan'):
                    continue
                if val == "" or val == "NA" or val == "nan":
                    continue

                # Standardize with lowercases
                if column == "RESPOND_WK8" and type(val) == type(""):
                    val = val.lower()

                # For debugging purposes later
                _val = val
                if is_number(val):
                    _val = float(val)
                    conversion = ["[conversion]", val, type(val), "to", _val, type(_val)]
                    column_conversions += conversion

                if _val in uniqs_counter:
                    uniqs_counter[_val] += 1
                else:
                    uniqs_counter[_val] = 1

                val_to_keep = _val

            # Decide which value to store for this column
            # If num uniqs is 0 then saves a blank, if 1 then saves the single value. If greater than 1, then saves "collision".
            if len(uniqs_counter) > 1:
                num_collisions += 1
                collision = ["[collision]" + str(uniqs_counter)]
                column_collisions += collision

                max_freq = max(uniqs_counter.values())
                for key, val in uniqs_counter.items():
                    if val == max_freq:
                        val_to_keep = val
                        num_collisions_handled += 1
                        break

            agg_patient_vals.append((column, [val_to_keep]))

            collisions[column] = column_collisions
            conversions[column] = column_conversions

        new_df = new_df.append(pd.DataFrame.from_items(agg_patient_vals))

        if i % 100 == 0:
            print("Batch: [%d] subjects have been aggregated thus far with [%d] total collisions" % (i, num_collisions))
        i += 1

    for col, collisionz in collisions.items():
        if len(collisionz) > 0:
            print(col)
        for x in collisionz:
            print("\t", x)
    for col, conversionz in conversions.items():
        if len(conversionz) > 0:
            print(col)
        for x in conversionz:
            print("\t", x)

    return new_df


def get_event_based_value(row, curr_event, curr_feature, scale_name):
    """
    Helper function to get the value in a column given that the value in another column for that row
    meets a specific condition.

    For example, given that...
        - row is a patient entry
        - curr_event is 'Time K'
        - curr_feature is 'MADRS_XYZ'

    If the patient entry has the value curr_event at its COL_NAME_EVENTNAME column, then return
    the value stored for that patient in the feature in question.

    If the given row is an entry for 'Time A' and not 'Time K', then it will return an empty value.

    :param row: the row representing a patient in the table
    :param curr_event: a value of the EVENTNAME column
    :param curr_feature: a column which needs to be extended based on the value of the event
    :return:
    """
    if row[COL_NAME_EVENTNAME].lower() == curr_event.lower():
        return row[curr_feature]
    else:
        return ""


def extend_columns_eventbased(orig_df):
    """
    Handles adding extra columns based on a condition the value of another column.

    :param orig_df: the original dataframe
    :return: a new, modified dataframe
    """
    global COL_NAMES_NEW_FROM_EXTENSION
    global COL_NAMES_TO_DROP_FROM_EXTENSION

    # Create extra columns with name of event appended, initialized blank
    for scale_group in COL_NAMES_TO_CONVERT:
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

    print_progress_completion(extend_columns_eventbased, "added extra columns based on event/visit")
    return orig_df

def merge_columns(df, column_mapping):
    """
    Handles merging pairs of columns. If col A is "" or "NA" or np.nan and col B is "z" then col AB will contain "z".
    If both columns are non-empty but do not match then it will take the value of the first column.
    :param df: the dataframe to modify
    :param column_mapping: key-value pair mapping for pairs of columns that will get merged
    :return: the modified df
    """
    df.reset_index(drop=True, inplace=True)
    for col1, col2 in column_mapping.items():
        merged_col_name = col1 + "_" + col2 + "_merged"
        df[merged_col_name] = ""

    blacklist = []
    for i, row in df.iterrows():
        for col1, col2 in column_mapping.items():
            val1 = row[col1]
            val2 = row[col2]
            merged_col_name = col1 + "_" + col2 + "_merged"
            if is_empty_value(val1) and is_empty_value(val2):
                df.set_value(i, merged_col_name, np.nan)
            elif not is_empty_value(val1) and not is_empty_value(val2):
                df.set_value(i, merged_col_name, val1)
            elif not is_empty_value(val1) and is_empty_value(val2):
                df.set_value(i, merged_col_name, val1)
            elif not is_empty_value(val2) and is_empty_value(val1):
                df.set_value(i, merged_col_name, val2)
            blacklist.extend([col1, col2])
    add_columns_to_blacklist(blacklist)
    print_progress_completion(merge_columns, "merged QLESQ columns")
    return df

def print_progress_completion(f, msg):
    print("Progress completion: [", f, "]", msg)

def is_number(s):
    """
    Checks if the variable is a number.

    :param s: the variable
    :return: True if it is, otherwise False
    """
    try:
        # Don't need to check for int, if it can pass as a float then it's a number
        float(s)
        return True
    except ValueError:
        return False


def replace_all_values_in_col(df, replacement_maps):
    """
    Converts the values in a column based on mappings defined in VALUE_REPLACEMENT_MAPS.

    :param df: the dataframe
    :return: the dataframe
    """
    for dict in replacement_maps:
        values_map = dict["values"]
        if "col_names" in dict:
            for col_name in dict["col_names"]:
                df[col_name] = df[col_name].map(values_map)
    return df

def replace_target_col_values(df, replacement_maps):
    """
    Converts the values in a column based on mappings defined in VALUE_REPLACEMENT_MAPS_USE_REPLACE by replacing
    single values.

    :param df: the dataframe
    :return: the dataframe
    """
    for dict in replacement_maps:
        if "col_names" in dict:
            for col_name in dict["col_names"]:
                values_map = dict["values"]
                df[col_name] = df[col_name].replace(to_replace=values_map)
    return df

def replace_target_col_values_to_be_refactored(df, replacement_maps):
    """
    Converts the values in a column based on mappings defined in VALUE_REPLACEMENT_MAPS_USE_REPLACE by replacing
    single values.

    :param df: the dataframe
    :return: the dataframe
    """
    for dict in replacement_maps:
        if "col_names" in dict:
            for col_name in dict["col_names"]:
                values_map = dict["values"]
                df[col_name] = df[col_name].replace(to_replace=values_map)

                # Hard-code this exceptional case separately
                if col_name == "EDUC":
                    vals_less_than_14 = {}
                    for key, value in df[col_name].iteritems():
                        if value <= 13:
                            vals_less_than_14[value] = value - 1
                    df[col_name] = df[col_name].replace(to_replace=vals_less_than_14)

    # Hard-code replacing values with median for some columns
    col_name = "HSHLD_INCOME"
    df[col_name] = df[col_name].replace("", np.nan)
    df[col_name] = df[col_name].replace(9999, df[col_name].median()) # TODO redundant, replace these values 9999 etc with np.nan instead, and median will be handled elsewhere
    df[col_name] = df[col_name].replace(9998, df[col_name].median())
    col_name = "EDUC"
    df[col_name] = df[col_name].replace(9999, df[col_name].median())
    return df

def finalize_blacklist():
    add_columns_to_blacklist(COL_NAMES_BLACKLIST_IPAQ)
    add_columns_to_blacklist(COL_NAMES_BLACKLIST_QIDS)
    add_columns_to_blacklist(COL_NAMES_BLACKLIST_LEAPS)
    add_columns_to_blacklist(COL_NAMES_BLACKLIST_MINI)
    add_columns_to_blacklist(COL_NAMES_BLACKLIST_DEMO)
    add_columns_to_blacklist(COL_NAMES_BLACKLIST_DARS)
    add_columns_to_blacklist(COL_NAMES_BLACKLIST_SHAPS)
    add_columns_to_blacklist(COL_NAMES_BLACKLIST_PSYHIS)
    add_columns_to_blacklist(COL_NAMES_TO_DROP_FROM_EXTENSION)
    add_columns_to_blacklist(COL_NAMES_BLACKLIST_COMMON)

def collect_columns_to_extend(field):
    if field.startswith("MADRS_"):
        COL_NAMES_MADRS_TO_CONVERT.append(field)
    elif field.startswith("HCL_"):
        COL_NAMES_HCL_TO_CONVERT.append(field)
    elif field.startswith("GAD7_"):
        COL_NAMES_GAD7_TO_CONVERT.append(field)
    elif field.startswith("QIDS_"):
        COL_NAMES_GAD7_TO_CONVERT.append(field)
    elif field.startswith("QLESQ"):
        COL_NAMES_QLESQ_TO_CONVERT.append(field)

def print_info(merged_df, extra):
    # TODO update and refactor this
    print("\n____Data cleaning summary_____________________________________\n")
    print("Final dimension of the merged table:", merged_df.shape)
    print("Total data files merged:", NUM_DATA_FILES)
    if extra:
        for filename in FILENAMES:
            print("\t", filename)

    print("\nTotal data rows merged:", NUM_DATA_ROWS)
    print("Total data columns merged:", NUM_DATA_COLUMNS)

    repeats = 0
    print("\nColumns that appear more than once across files, which were merged:")
    for col_name, count in UNIQ_COLUMNS.items():
        if count > 1:
            print("\t", col_name, "-", count, "times")
            repeats += count

    if extra:
        print("\nPatient duplicate rows:")
        print(merged_df.groupby(['SUBJLABEL']).size().reset_index(name='Count'))

    print("\nThere are %d columns that have NA values" % len(COL_NAMES_NA))
    for col_name in COL_NAMES_NA:
        print("\t", col_name)

    print("\nThere are %d columns with categorical values:" % len(COL_NAMES_CATEGORICAL))
    for col_name in COL_NAMES_CATEGORICAL:
        if extra:
            print("\t", col_name)

    print("\nThere are %d columns with that had data collisions for a group of patient rows:" % len(COLLISION_MANAGER))
    for col_name in COLLISION_MANAGER:
        if extra:
            print("\t", col_name, COLLISION_MANAGER[col_name])


# if __name__ == "__main__":
#     if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
#         aggregate_and_clean(sys.argv[1], verbose=False, extra=False)

#     elif len(sys.argv) == 3 and sys.argv[1] == "-v" and os.path.isdir(sys.argv[2]):
#         aggregate_and_clean(sys.argv[2], verbose=True, extra=False)

#     elif len(sys.argv) == 3 and sys.argv[1] == "-v+" and os.path.isdir(sys.argv[2]):
#         aggregate_and_clean(sys.argv[2], verbose=True, extra=True)
#     else:
#         print("Enter valid arguments\n"
#               "\t options: -v for verbose, -v+ for super verbose\n"
#               "\t path: the path to a real directory\n")



pathData = r'C:\Users\jjnun\Documents\Sync\Research\1_CANBIND Replication\teyden-git\data\canbind_data_full_auto\\'
aggregate_and_clean(pathData, "-v")