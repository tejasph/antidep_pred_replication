import os
import sys
import pandas as pd
import numpy as np

import stard_globals as gls
from utils import *

COL_NAMES_TO_DROP_FROM_EXTENSION = []
COL_NAMES_NEW_FROM_EXTENSION = []
COL_NAMES_BLACKLIST_UNIQS = []

""" 
Cleans and aggregates STAR*D data.
    
Example usages

    Basic:
        python canbind_data_processor.py /path/to/data/folders
        
        python canbind_data_processor.py -targetsoverlapping /path/to/data/folders

        python canbind_data_processor.py -targetsoverlapping /path/to/data/folders

    Verbose:
        python canbind_data_processor.py -v /path/to/data/folders

    Super verbose:
        python canbind_data_processor.py -v+ /path/to/data/folders


This will output a single CSV file containing the merged and clean data. Can also output target (y matrix) files for 
STAR*D data as well as the targets for the overlapping features between STAR*D and CAN-BIND.

The method expects CSV files to be contained within their own subdirectories from the root directory, as is organized
in the ZIP provided.
"""
def aggregate(root_dir, verbose=False, extra=False):
    merged_df = pd.DataFrame([])
    output_dir = root_dir + "/output_files"

    for filename in os.listdir(root_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # TODO grab rows by condition in level, days_baseline and week, for each scale
        scale_name = filename.split(".")[0]
        if scale_name not in gls.SCALES:
            continue
        # if scale_name != "dm01":
        #     continue

        modified_scale_df = pd.DataFrame([])

        file_path = root_dir + "/" + filename

        # Read in the txt file + preliminary processing
        df = pd.read_csv(file_path, sep='\s+')
        df = df.iloc[1:] # Drop the second row
        df.rename(columns=lambda x: x.lower(), inplace=True)

        # This has introduced bugs. We should filter these patients out with excel instead of here. No time to fix these bugs.
        # df = eliminate_early_leavers(df)

        # Grab configuration information for the scale
        scale_config = gls.SCALES[scale_name]
        levels = scale_config[gls.COL_NAME_LEVEL]
        days_baselines = scale_config[gls.COL_NAME_DAYS_BASELINE]
        weeks = scale_config[gls.COL_NAME_WEEK]
        whitelist = gls.SCALES[scale_name]["whitelist"]

        print("*************************************************************")
        print("Handling scale = ", scale_name)

        # Handle ccv01 and qids separately
        if scale_name == "ccv01":
            modified_scale_df = extend_columns_ccv01(df, whitelist, levels, weeks)
        elif scale_name == "qids01":
            modified_scale_df = extend_columns_qids_mod(df, whitelist, levels, days_baselines,
                                                    scale_config[gls.COL_NAME_VERSION_FORM])
        elif scale_name in gls.SCALES_NO_ROW_RESTRICTIONS:
            modified_scale_df = df
        else:
            if scale_name == "dm01":
                s_name = scale_name + "_alt_partial"
                l = gls.SCALES[s_name][gls.COL_NAME_LEVEL]
                w = gls.SCALES[s_name][gls.COL_NAME_WEEK]
                rows = filter_rows_on_conditions(df, scale_name, l, w, None)
                if len(rows) > 0:
                    modified_scale_df = modified_scale_df.append(rows)
                else:
                    print("Scale [%s] did not have any patients meeting the conditions." % (scale_name))

            list_of_rows = filter_rows_on_conditions(df, scale_name, levels, weeks, days_baselines)
            if len(list_of_rows) > 0:
                modified_scale_df = modified_scale_df.append(list_of_rows)
            else:
                print("Scale [%s] did not have any patients meeting the conditions." % (scale_name))

        print(len(modified_scale_df))
        modified_scale_df.to_csv(output_dir + "/" + scale_name + "-reduced.csv")
        merged_df = merged_df.append(modified_scale_df, ignore_index=True)

    merged_df.to_csv(output_dir + "/stard-so-far-before-row-aggregation.csv")
    merged_df = aggregate_rows(merged_df)
    merged_df.to_csv(output_dir + "/stard-so-far-after-row-aggregation-before-whitelist.csv")
    merged_df = merged_df[gls.WHITELIST + ["days_baseline"]]
    merged_df.to_csv(output_dir + "/stard-so-far.csv")

def filter_rows_on_conditions(df, scale_name, levels, weeks, days_baselines):
    """
    Returns a list of DataFrame rows selected based on conditions.
    :param scale_name: name of the scale
    :param group:
    :param levels:
    :param weeks:
    :param days_baselines:
    :return:
    """
    rows_grouped = df.groupby([gls.COL_NAME_SUBJECTKEY])
    list_of_rows = []
    for subject_id, group in rows_grouped:
        row = get_preferred_entry(subject_id, group, levels, weeks, days_baselines)
        if row is not None:
            list_of_rows += [row]
    return list_of_rows

def get_preferred_entry(subject_id, group, levels, weeks, days_baselines):
    """
    Finds the preferred subject entry for a given scale, based on the conditions of different columns.
    :param group: a Series containing rows for a single subject
    :return: the preferred entry/row
    """
    # Filter out the rows not in the possible range
    if levels != None:
        group = group.loc[group.level.isin(levels)]

    # Sort it so that the higher priority level comes first (most cases it's just one)
    group = group.sort_values(by=[gls.COL_NAME_LEVEL])

    if levels is not None:
        if days_baselines is None and weeks is None:
            # Iterate through and grab the first one
            for i, row in group.iterrows():
                for level in levels:
                    if row[gls.COL_NAME_LEVEL] == level:
                        # TODO return a list of rows that meet the condition, or the first one that meets the preferred
                        return row
        elif days_baselines is not None and weeks is None:
            # Filter out the rows not in the possible range
            # group = group.loc[group.days_baseline.isin(days_baselines)] <---- This doesn't actually work because the values are strings!!!

            # Iterate through and grab the first one
            for i, row in group.iterrows():
                for level in levels:
                    for days_baseline in days_baselines:
                        if row[gls.COL_NAME_LEVEL] == level and int(row[gls.COL_NAME_DAYS_BASELINE]) == days_baseline:
                            return row
        elif weeks is not None and days_baselines is None:
            # Filter out the rows not in the possible range
            group = group.loc[group.week.isin(weeks)]

            # Iterate through and grab the first one
            for i, row in group.iterrows():
                for level in levels:
                    for week in weeks:
                        if row[gls.COL_NAME_LEVEL] == level and row[gls.COL_NAME_WEEK] == week:
                            return row
    else:
        raise ValueError("Should've met at least one condition")
    return None

def extend_columns_ccv01(scale_df, whitelist, levels, weeks):
    """
    Handles adding extra columns based on a condition the value of another column.
    :param orig_df: the original dataframe
    :return: a modified dataframe
    """
    global COL_NAMES_TO_DROP_FROM_EXTENSION
    global COL_NAMES_NEW_FROM_EXTENSION

    def get_value(row, level, week, curr_col_to_extend):
        if row[gls.COL_NAME_LEVEL].lower() == level.lower() \
                and row[gls.COL_NAME_WEEK] == week:
            return row[curr_col_to_extend]
        return ""

    # Filter out the rows not in the possible range
    scale_df = scale_df.loc[scale_df.level.isin(levels)]
    scale_df = scale_df.loc[scale_df.week.isin(weeks)]

    for col_name in whitelist:
        level = "Level 1"
        week_01 = 0.1
        week_2 = 2

        new_col_name_1 = col_name + "_" + level + "_" + str(week_01)
        new_col_name_2 = col_name + "_" + level + "_" + str(week_2)

        # Add these for tracking
        COL_NAMES_NEW_FROM_EXTENSION.extend([new_col_name_1, new_col_name_2])

        # Set the value for the new column
        scale_df[new_col_name_1] = scale_df.apply(lambda row: get_value(row, level, week_01, col_name), axis=1)
        scale_df[new_col_name_2] = scale_df.apply(lambda row: get_value(row, level, week_2, col_name), axis=1)

    # Add the original column names to drop them later
    COL_NAMES_TO_DROP_FROM_EXTENSION.extend(whitelist)

    return scale_df

def extend_columns_qids(scale_df, whitelist, levels, days_baselines, version_forms):
    """
    Handles adding extra columns based on a condition the value of another column.
    :param orig_df: the original dataframe
    :return: a modified dataframe
    """
    global COL_NAMES_TO_DROP_FROM_EXTENSION
    global COL_NAMES_NEW_FROM_EXTENSION

    days_baselines_week0 = [-14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    days_baselines_week2 = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    def get_value(row, days_baselines, version_form, curr_col_to_extend):
        if row[gls.COL_NAME_VERSION_FORM].lower() == version_form.lower():
            for days_baseline in days_baselines:
                if int(row[gls.COL_NAME_DAYS_BASELINE]) == days_baseline and not is_empty_value(row[curr_col_to_extend]):
                    # No need to check for lowest days baseline because the lists should be in ascending order already
                    return row[curr_col_to_extend]
        return ""

    # Filter the rows in the possible range
    scale_df = scale_df.loc[scale_df.days_baseline.isin(days_baselines)]
    scale_df = scale_df.loc[scale_df.version_form.isin(version_forms)]

    # rows_grouped = df.groupby([gls.COL_NAME_SUBJECTKEY])
    # list_of_rows = []
    # for subject_id, group in rows_grouped:
    #     row = get_preferred_entry(subject_id, group, levels, weeks, days_baselines)
    #     if row is not None:
    #         list_of_rows += [row]
    # return list_of_rows
    
    for col_name in whitelist:
        version_form_self = "Self Rating"
        version_form_clin = "Clinician"

        new_col_name_1 = col_name + "_" + "week0" + "_" + version_form_self
        new_col_name_2 = col_name + "_" + "week0" + "_" + version_form_clin
        new_col_name_3 = col_name + "_" + "week2" + "_" + version_form_self
        new_col_name_4 = col_name + "_" + "week2" + "_" + version_form_clin

        # Add these for tracking
        COL_NAMES_NEW_FROM_EXTENSION.extend([new_col_name_1, new_col_name_2, new_col_name_3, new_col_name_4])

        # Set the value for the new columns
        scale_df[new_col_name_1] = scale_df.apply(lambda row: get_value(row, days_baselines_week0, version_form_self, col_name), axis=1)
        scale_df[new_col_name_2] = scale_df.apply(lambda row: get_value(row, days_baselines_week0, version_form_clin, col_name), axis=1)
        scale_df[new_col_name_3] = scale_df.apply(lambda row: get_value(row, days_baselines_week2, version_form_self, col_name), axis=1)
        scale_df[new_col_name_4] = scale_df.apply(lambda row: get_value(row, days_baselines_week2, version_form_clin, col_name), axis=1)

    # Add the original column names to drop them later
    COL_NAMES_TO_DROP_FROM_EXTENSION.extend(whitelist)
    return scale_df


def extend_columns_qids_mod(scale_df, whitelist, levels, days_baselines, version_forms):
    """
    Handles adding extra columns based on a condition the value of another column.
    :param orig_df: the original dataframe
    :return: a modified dataframe
    """
    global COL_NAMES_TO_DROP_FROM_EXTENSION
    global COL_NAMES_NEW_FROM_EXTENSION

    days_baselines_week0 = [-14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    days_baselines_week2 = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    def get_value(row, subject_id, days_baselines, version_form, curr_col_to_extend):
        if row[gls.COL_NAME_SUBJECTKEY] == subject_id \
                and row[gls.COL_NAME_VERSION_FORM].lower() == version_form.lower():
            for days_baseline in days_baselines:
                if int(row[gls.COL_NAME_DAYS_BASELINE]) == days_baseline and not is_empty_value(row[curr_col_to_extend]):
                    # No need to check for lowest days baseline because the lists should be in ascending order already
                    return row[curr_col_to_extend]
        return ""

    # Filter the rows in the possible range
    scale_df = scale_df.loc[scale_df.days_baseline.isin(days_baselines)]
    scale_df = scale_df.loc[scale_df.version_form.isin(version_forms)]
    for col_name in whitelist:
        version_form_self = "Self Rating"
        version_form_clin = "Clinician"

        new_col_name_1 = col_name + "_" + "week0" + "_" + version_form_self
        new_col_name_2 = col_name + "_" + "week0" + "_" + version_form_clin
        new_col_name_3 = col_name + "_" + "week2" + "_" + version_form_self
        new_col_name_4 = col_name + "_" + "week2" + "_" + version_form_clin

        # Add these for tracking
        COL_NAMES_NEW_FROM_EXTENSION.extend([new_col_name_1, new_col_name_2, new_col_name_3, new_col_name_4])

        # Set the value for the new columns
        scale_df[new_col_name_1] = ""
        scale_df[new_col_name_2] = ""
        scale_df[new_col_name_3] = ""
        scale_df[new_col_name_4] = ""

    rows_grouped = scale_df.groupby([gls.COL_NAME_SUBJECTKEY])
    for subject_id, group in rows_grouped:
        for col_name in whitelist:
            version_form_self = "Self Rating"
            version_form_clin = "Clinician"
            new_col_name_1 = col_name + "_" + "week0" + "_" + version_form_self
            new_col_name_2 = col_name + "_" + "week0" + "_" + version_form_clin
            new_col_name_3 = col_name + "_" + "week2" + "_" + version_form_self
            new_col_name_4 = col_name + "_" + "week2" + "_" + version_form_clin
            for i, row in group.iterrows():
                if is_empty_value(row[new_col_name_1]):
                    scale_df.set_value(i, new_col_name_1, get_value(row, subject_id, days_baselines_week0, version_form_self, col_name))
                if is_empty_value(row[new_col_name_2]):
                    scale_df.set_value(i, new_col_name_2, get_value(row, subject_id, days_baselines_week0, version_form_clin, col_name))
                if is_empty_value(row[new_col_name_3]):
                    scale_df.set_value(i, new_col_name_3, get_value(row, subject_id, days_baselines_week2, version_form_self, col_name))
                if is_empty_value(row[new_col_name_4]):
                    scale_df.set_value(i, new_col_name_4, get_value(row, subject_id, days_baselines_week2, version_form_clin, col_name))

    # Add the original column names to drop them later
    COL_NAMES_TO_DROP_FROM_EXTENSION.extend(whitelist)
    return scale_df

def does_exceed(tracker):
    for key, val in tracker.items():
        if val > 1:
            return True
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
    for key, dict in replacement_maps.items():
        if key == "minus":
            continue
        for col_name in dict["col_names"]:
            if "col_extenders" in dict:
                for extender in dict["col_extenders"]:
                    converted_col_name = col_name + extender
                    df[converted_col_name] = df[converted_col_name].astype("object") # Deathly necessary. lol
                    conversion_map = dict["values"]
                    df[converted_col_name] = df[converted_col_name].replace(to_replace=conversion_map)
            else:
                # Change dtype to object. This fixes the issue between comparing mismatched types. Pandas infers a type
                # when it initially loads in the data, which may/may not be correct.
                df[col_name] = df[col_name].astype("object")
                conversion_map = dict["values"]
                df[col_name] = df[col_name].replace(to_replace=conversion_map)
    return df

def convert_sfhs01_cols(df):
    config = gls.VALUE_CONVERSION_MAP["minus"]
    for key, list_of_cols in config.items():
        for col_name in list_of_cols:
            conversion_map = {}
            for k, value in df[col_name].iteritems():
                if key == 6 or key == 3:
                    conversion_map[value] = key - value
                elif key == 1:
                    conversion_map[value] = value - 1
            df[col_name] = df[col_name].replace(to_replace=col_name)
    return df

def convert_months_to_year(df, col_name):
    conversion_map = {}
    df[col_name] = df[col_name].astype("float")
    for key, value in df[col_name].iteritems():
        converted = None
        if not is_number(value):
            continue
        if value not in conversion_map:
            converted = np.floor(value/12)
            conversion_map[value] = converted
    df[col_name] = df[col_name].replace(to_replace=conversion_map)
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

def aggregate_rows(df):
    """
    Aggregates groups of patient rows corresponding to a single patient to a single row.

    :param df: the dataframe
    :return: a new dataframe consisting of one row per patient
    """
    new_df = pd.DataFrame()
    grouped = df.groupby(gls.COL_NAME_SUBJECTKEY)
    i = 0
    num_collisions = 0
    collisions = {}
    conversions = {}
    for patient_id, group_of_rows_df in grouped:
        agg_patient_vals = [(gls.COL_NAME_SUBJECTKEY, [patient_id])]

        # Iterate over columns to grab the values for aggregation, and determine which to keep
        for column, values in group_of_rows_df.iteritems():
            if column == gls.COL_NAME_SUBJECTKEY:
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
                        break

                agg_patient_vals.append((column, [val_to_keep]))
            else:
                agg_patient_vals.append((column, [val_to_keep]))

            collisions[column] = column_collisions
            conversions[column] = column_conversions

        new_df = new_df.append(pd.DataFrame.from_items(agg_patient_vals))

        if i % 1000 == 0:
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

def clean(root_dir):
    output_dir = root_dir + "/output_files"
    df = pd.read_csv(output_dir + "/stard-so-far.csv")
    df = df.drop(["Unnamed: 0"], axis=1)

    # # Handle early leavers
    qids_df = pd.read_csv(root_dir + "/" + "qids01.txt", sep='\s+')
    qids_df = qids_df.iloc[1:]  # Drop the second row
    qids_df.rename(columns=lambda x: x.lower(), inplace=True)
    qids_df.days_baseline = qids_df.days_baseline.astype("float")  # Deathly necessary. lol
    qids_df.qstot = qids_df.qstot.astype("float")
    reduced_qids_df = eliminate_early_leavers(qids_df)
    whitelist_subjects = reduced_qids_df["subjectkey"].unique()

    final_whitelist = []
    keys = df["subjectkey"].unique()
    for id in whitelist_subjects:
        if id in keys:
            final_whitelist.append(id)

    df = df.loc[df.subjectkey.isin(final_whitelist)]

    # Value conversions
    df = replace_target_col_values(df, gls.VALUE_CONVERSION_MAP)
    df = convert_months_to_year(df, "interview_age")
    df = convert_sfhs01_cols(df)
    df = convert_to_zero_if_scale_is_nonempty(df, "wpai01")
    df = convert_to_zero_if_scale_is_nonempty(df, "phx01")

    # One hot encoding
    df = one_hot_encode(df, gls.COL_NAMES_ONE_HOT_ENCODE)
    df = df.drop(gls.COL_NAMES_ONE_HOT_ENCODE, axis=1)

    # Final drops - too sparse (very last minute add)
    final_blacklist = ['mempl', 'assist', 'massist', 'unempl', 'munempl', 'otherinc', 'minc_other', 'totincom',
                       'medication1_dosage_Level 1_0.1', 'days_baseline']
    df = df.drop(final_blacklist, axis=1)

    df = df.sort_values(by=["subjectkey"])
    df = df.reset_index(drop=True)
    df.to_csv(output_dir + "/stard-clean-aggregated-data.csv")

    for column in df.columns.values:
        print("\t", column)

def convert_to_zero_if_scale_is_nonempty(df, scale_name):
    """
    If the scale has at least one non empty value for all of its features, then any empty values for that scale
    will get converted to 0.
    :param df:
    :param scale_name:
    :return:
    """
    whitelist = gls.SCALES[scale_name]["whitelist"]
    subjects_grouped = df.groupby([gls.COL_NAME_SUBJECTKEY])
    for subject_id, group in subjects_grouped:
        can_convert_to_blank = False
        for i, row in group.iterrows():
            for col_name in whitelist:
                if not is_empty_value(row[col_name]):
                    can_convert_to_blank = True
                    break
            for col_name in whitelist:
                if is_empty_value(row[col_name]) and can_convert_to_blank:
                    df.set_value(i, col_name, 0)
    return df

def eliminate_early_leavers(orig_df):
    # Eliminate subjects that don't have any records > 21
    df = orig_df.copy(deep=True)
    df.days_baseline = df.days_baseline.astype("float")  # Deathly necessary. lol
    subjects_grouped = df.groupby([gls.COL_NAME_SUBJECTKEY])

    subject_ids = []
    count = 0
    print("Number of subjects to begin with:", len(df["subjectkey"].unique()))
    for subject_id, group in subjects_grouped:
        for i, row in group.iterrows():
            # If a subject has at least one record > 21, then keep them
            if row.days_baseline > 21:
                subject_ids.append(subject_id)
                count += 1
                break

    reduced_df = df.loc[df.subjectkey.isin(subject_ids)]
    print("Number of subjects reduced to:", len(reduced_df["subjectkey"].unique()))
    return reduced_df

def create_targets(root_dir):
    output_dir = root_dir + "/output_files"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_filepath = output_dir + "/" + "stard-y.csv"

    df = pd.read_csv(root_dir + "/" + "qids01.txt", sep='\s+')
    df = df.iloc[1:]  # Drop the second row
    df = df.rename(columns=lambda x: x.lower())
    df = eliminate_early_leavers(df)

    target_df = pd.DataFrame()
    days_baseline_threshold = 22

    df.days_baseline = df.days_baseline.astype("float")
    no_remission_df = df[df.days_baseline >= days_baseline_threshold]
    no_remission_df = no_remission_df[(no_remission_df.level == "Level 3") | (no_remission_df.level == "Level 4")]

    subjects = []
    subject_keys_no_remission = []
    for name, group in no_remission_df.groupby([gls.COL_NAME_SUBJECTKEY]):
        subjects.append([name, 0])
        subject_keys_no_remission.append(name)
    target_df = target_df.append(subjects)

    df.qstot = df.qstot.astype("float")
    remission_df = df[(df.level != "Follow-Up")]
    remission_df = remission_df[(remission_df.qstot <= 5)]
    remission_df = remission_df.loc[~remission_df.subjectkey.isin(subject_keys_no_remission)] # Filter out

    subjects = []
    subject_keys_remission = []
    for name, group in remission_df.groupby([gls.COL_NAME_SUBJECTKEY]):
        subjects.append([name, 0])
        subject_keys_remission.append(name)
    target_df = target_df.append(subjects)

    remaining_df = df.loc[~df.subjectkey.isin(subject_keys_remission + subject_keys_no_remission)]
    subjects = []
    subject_keys_remaining = []
    for name, group in remaining_df.groupby([gls.COL_NAME_SUBJECTKEY]):
        subjects.append([name, 1])
        subject_keys_remaining.append(name)

    target_df = target_df.append(subjects)
    target_df = target_df.reset_index(drop=True)
    target_df = target_df.rename(columns={0: "subjectkey", 1: "target"})
    target_df = target_df.sort_values(by=["subjectkey"])
    target_df.to_csv(output_filepath)
    print("Targets saved to:", output_filepath)

def create_overlapping_targets(root_dir):
    output_dir = root_dir + "/output_files"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_filepath = output_dir + "/" + "y-stard-overlapping-targets.csv"

    orig_df = pd.read_csv(root_dir + "/" + "qids01.txt", sep='\s+')
    orig_df = orig_df.iloc[1:]  # Drop the second row
    orig_df = orig_df.rename(columns=lambda x: x.lower())
    orig_df.days_baseline = orig_df.days_baseline.astype("float")  # Deathly necessary. lol
    orig_df.qstot = orig_df.qstot.astype("float")

    CLINICIAN = "Clinician"
    SELF_RATING = "Self Rating"
    QSTOT = "qstot"

    reduced_df = eliminate_early_leavers(orig_df)

    # Determine baselines and ratios
    reduced_df.days_baseline = reduced_df.days_baseline.astype("float")  # Deathly necessary. lol
    reduced_df = reduced_df.loc[reduced_df.days_baseline <= 77]
    reduced_df = reduced_df.loc[(reduced_df.version_form == "Clinician") | (reduced_df.version_form == "Self Rating")]
    subjects_grouped = reduced_df.groupby([gls.COL_NAME_SUBJECTKEY])
    reduced_df.qstot = reduced_df.qstot.astype("float")

    baseline_cases = {}
    for subject_id, group in subjects_grouped:
        baseline_cases[subject_id] = {}
        baseline_cases[subject_id][CLINICIAN] = None
        baseline_cases[subject_id][SELF_RATING] = None
        lowest_days_baseline_clin = np.inf
        lowest_days_baseline_self = np.inf
        for i, row in group.iterrows():
            if row.version_form == "Clinician" and row.days_baseline < lowest_days_baseline_clin:
                baseline_cases[subject_id][CLINICIAN] = row
                lowest_days_baseline_clin = row.days_baseline
            elif row.version_form == "Self Rating" and row.days_baseline < lowest_days_baseline_self:
                baseline_cases[subject_id][SELF_RATING] = row
                lowest_days_baseline_self = row.days_baseline

    subject_ids = []
    y_vals = []
    for subject_id, group in subjects_grouped:
        if subject_id in baseline_cases:
            baseline_case_clin = baseline_cases[subject_id][CLINICIAN]
            baseline_case_self = baseline_cases[subject_id][SELF_RATING]
            y = 0
            for i, row in group.iterrows():
                if row.equals(baseline_case_clin) or row.equals(baseline_case_self):
                    continue
                else:
                    if row.version_form == CLINICIAN and baseline_case_clin is not None:
                        ratio = row[QSTOT] / baseline_case_clin[QSTOT]
                        if row[QSTOT] == 0 or ratio <= 0.50:
                            y = 1
                            break
                    elif row.version_form == SELF_RATING and baseline_case_self is not None:
                        ratio = row[QSTOT] / baseline_case_self[QSTOT]
                        if row[QSTOT] == 0 or ratio <= 0.50:
                            y = 1
                            break
            subject_ids.append([subject_id, y])

    new_reduced_df = pd.DataFrame()
    new_reduced_df = new_reduced_df.append(subject_ids)
    new_reduced_df = new_reduced_df.rename(columns={0: "subjectkey", 1: "target"})
    new_reduced_df = new_reduced_df.sort_values(by=["subjectkey"])
    new_reduced_df = new_reduced_df.reset_index(drop=True)
    new_reduced_df.to_csv(output_filepath)
    print("Targets saved to:", output_filepath)

if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        aggregate(sys.argv[1], verbose=False, extra=False)

    elif len(sys.argv) == 3 and sys.argv[1] == "-v" and os.path.isdir(sys.argv[2]):
        aggregate(sys.argv[2], verbose=True, extra=False)

    elif len(sys.argv) == 3 and sys.argv[1] == "-v+" and os.path.isdir(sys.argv[2]):
        aggregate(sys.argv[2], verbose=True, extra=True)

    elif len(sys.argv) == 3 and sys.argv[1] == "-cleanonly" and os.path.isdir(sys.argv[2]):
        clean(sys.argv[2])

    elif len(sys.argv) == 3 and sys.argv[1] == "-targets" and os.path.isdir(sys.argv[2]):
        create_targets(sys.argv[2])

    elif len(sys.argv) == 3 and sys.argv[1] == "-targetsoverlapping" and os.path.isdir(sys.argv[2]):
        create_overlapping_targets(sys.argv[2])

    elif len(sys.argv) == 3 and sys.argv[1] == "-runpipeline" and os.path.isdir(sys.argv[2]):
        aggregate(sys.argv[1], verbose=False, extra=False)
        create_targets(sys.argv[2])
        create_overlapping_targets(sys.argv[2])

    else:
        print("Enter valid arguments\n"
              "\t options: -v for verbose, -v+ for super verbose\n"
              "\t path: the path to a real directory\n")
