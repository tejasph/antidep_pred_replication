import pandas as pd
import numpy as np
from canbind_globals import COL_NAME_PATIENT_ID,COL_NAME_EVENTNAME, COL_NAMES_BLACKLIST_DARS
from canbind_globals import COL_NAMES_TO_CONVERT, COL_NAMES_BLACKLIST_IPAQ, COL_NAMES_BLACKLIST_QIDS, COL_NAMES_BLACKLIST_LEAPS, COL_NAMES_BLACKLIST_MINI, COL_NAMES_BLACKLIST_DEMO
from canbind_globals import COL_NAMES_BLACKLIST_SHAPS, COL_NAMES_BLACKLIST_PSYHIS, COL_NAMES_TO_DROP_FROM_EXTENSION, COL_NAMES_BLACKLIST_COMMON 
from canbind_globals import COL_NAMES_HCL_TO_CONVERT, COL_NAMES_GAD7_TO_CONVERT, COL_NAMES_QIDS_TO_CONVERT, COL_NAMES_QLESQ_TO_CONVERT, COL_NAMES_MADRS_TO_CONVERT
from canbind_globals import COL_NAMES_NEW_FROM_EXTENSION, COL_NAMES_BLACKLIST_UNIQS, COL_NAMES_BLACKLIST_YGEN
from utils import is_empty_value
from collections import OrderedDict

"""
Helper functions for the CAN-BIND data preprocessing
"""



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
                df.at[i, merged_col_name]=np.nan
            elif not is_empty_value(val1) and not is_empty_value(val2):
                df.at[i, merged_col_name]=val1
            elif not is_empty_value(val1) and is_empty_value(val2):
                df.at[i, merged_col_name]=val1
            elif not is_empty_value(val2) and is_empty_value(val1):
                df.at[i, merged_col_name]=val2
            blacklist.extend([col1, col2])
    add_columns_to_blacklist(blacklist)
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
    if col_name in df.columns:
        df[col_name] = df[col_name].replace("", np.nan)
        df[col_name] = df[col_name].replace(9999, df[col_name].median()) 
        df[col_name] = df[col_name].replace(9998, df[col_name].median())
    col_name = "EDUC"
    if col_name in df.columns:
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
        COL_NAMES_QIDS_TO_CONVERT.append(field)
    elif field.startswith("QLESQ"):
        COL_NAMES_QLESQ_TO_CONVERT.append(field)
        

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

        new_df = new_df.append(pd.DataFrame.from_dict(OrderedDict(agg_patient_vals)))

    ##print_progress_completion(aggregate_patient_rows, "aggregated groups of patient rows to a single row")
    return new_df


def aggregate_rows(df, verbose=False):
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

        new_df = new_df.append(pd.DataFrame.from_dict(OrderedDict(agg_patient_vals)))

        if i % 100 == 0:
            if verbose: print("Batch: [%d] subjects have been aggregated thus far with [%d] total collisions" % (i, num_collisions))
        i += 1

    for col, collisionz in collisions.items():
        if len(collisionz) > 0:
            if verbose: print(col)
        for x in collisionz:
            if verbose: print("\t", x)
    for col, conversionz in conversions.items():
        if len(conversionz) > 0:
            if verbose: print(col)
        for x in conversionz:
            if verbose: print("\t", x)

    return new_df
