import os
import sys
import pandas as pd
import numpy as np

from utils import *

from overlapping_globals import HEADER_CONVERSION_DICT, CANBIND_OVERLAPPING_VALUE_CONVERSION_MAP,NEW_FEATURES_CANBIND,QIDS_STARD_TO_CANBIND_DICT as Q_DICT

STARD_OVERLAPPING_VALUE_CONVERSION_MAP = {
    "whitelist": ['epino', 'subjectkey', 'educat', 'pd_ag', 'pd_antis', 'gad_phx', 'anorexia', 'bulimia_0.0',
                  'ocd_phx', 'pd_noag', 'psd', 'soc_phob', 'epino', 'qlesq01', 'qlesq02', 'qlesq03', 'qlesq04',
                  'qlesq05', 'qlesq06', 'qlesq07', 'qlesq08', 'qlesq09', 'qlesq10', 'qlesq11', 'qlesq12', 'qlesq13',
                  'qlesq14', 'qlesq16', 'totqlesq', 'interview_age', 'episode_date',
                  'wsas01', 'wsas03', 'wsas02', 'wpai_totalhrs', 'wpai02', 'empl_2.0', 'empl_1.0', 'empl_3.0', 'episode_date',
                  'dage', 'empl_1.0', 'empl_3.0', 'empl_5.0', 'empl_2.0','empl_4.0', 'empl_6.0', 'marital_5.0',
                  'marital_2.0', 'marital_3.0', 'marital_1.0', 'marital_4.0', 'marital_6.0',
                  'qstot_week0_Self Rating', 'qstot_week2_Self Rating'], # Left out: 'alcoh' (one hot encoded) 'totincom' (too sparse) 'empl_8.0' 'empl_14.0' (non existent in data)
    "multiply": {
        "description": "Multiply the value by the multiple specified.",
        "col_names": {
            "wsas01": 1.25,
            "wsas03": 1.25,
            "wsas02": 1.25,
            "wpai_totalhrs": 2,
            "wpai02": 2,
            "episode_date": -1,
        }
    },
    "other": {},
    "blacklist": ['empl_5.0', 'empl_4.0'] # Use to remove after done processing
}


def convert_stard_to_overlapping(output_dir=""):
    if output_dir == "":
        output_dir = "/Users/teyden/Downloads/stard_data/output_files" # TODO temporarily hardcode
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # df = pd.read_csv(file_path)
    orig_df = pd.read_csv(output_dir + "/stard-clean-aggregated-data.csv")
    df = orig_df.drop(["Unnamed: 0"], axis=1)

    # Take whitelist columns first
    df = df[STARD_OVERLAPPING_VALUE_CONVERSION_MAP["whitelist"] + ["days_baseline"]]

    # Then process them
    for case, config in STARD_OVERLAPPING_VALUE_CONVERSION_MAP.items():
        if case == "keep":
            # Nothing to do, already grabbed from whitelist
            continue
        elif case == "multiply":
            mult_config = config[case]
            for col_name, multiple in mult_config.items():
                df[col_name] = df[col_name].apply(lambda x: x * multiple)
        else:
            df["episode_date"] = 1
            df["epino"] = 0

            # for i, row in df.iterrows():
            #     # TODO many of these cases won't work, and this is because features are based on unprocessed and processed data. Can't fix til handling that.
            #     if row["empl_4.0"] == 1 or row["empl_5.0"] == 1 or row["empl_10.0"] == 1:
            #         df.set_value(i, "empl_3.0", 1)
            #     if row["empl_7.0"] == 1:
            #         df.set_value(i, "empl_3.0", 1)
            #     if row["empl_12.0"] == 1 or row["empl_13.0"] == 1:
            #         df.set_value(i, "empl_2.0", 1)
            #     if row["dep"] == 1 or row["bip"] ==1 or row["alcohol"] == 1 or row["drug_phx"] == 1 or row["suic_phx"] == 1:
            #         df.set_value(i, "dep::bip::alcohol::drug_ph::suic_phx", 1)
            #     if row["amphet"] == 1 or row["cannibis"] ==1 or row["opioid"] == 1 or row["ax_cocaine"] == 1:
            #         df.set_value(i, "amphet::cannibis::opioid::ax_cocaine", 1)
            #     if row["dage"] < 0:
            #         df.set_value(i, "dage", row["age"] - row["dage"])
            #     if row["epino"] >= 2:
            #         df.set_value(i, "epino", 1)

    # Eliminate subjects that don't have any records > 21
    df = eliminate_early_leavers(df)
    df = df.drop(["days_baseline"], axis=1)

    df = df.sort_values(by=["subjectkey"])
    df = df.reset_index(drop=True)
    df.to_csv(output_dir + "/" + "canbind_imputed.csv")


def convert_canbind_to_overlapping(output_dir=""):
    if output_dir == "":
        output_dir = "/Users/teyden/Downloads/canbind-data-binned/output_files" # TODO temporarily hardcode
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # df = pd.read_csv(file_path)
    orig_df = pd.read_csv(output_dir + "/canbind_imputed.csv")
    df = orig_df.drop(["Unnamed: 0"], axis=1)

    # Take whitelist columns first
    df = df[CANBIND_OVERLAPPING_VALUE_CONVERSION_MAP["whitelist"]]

    # Add new features as blank
    for new_feature in NEW_FEATURES_CANBIND:
            df[new_feature] = np.nan
    
    # Then process them
    for case, config in CANBIND_OVERLAPPING_VALUE_CONVERSION_MAP.items():
        if case == "keep":
            # Nothing to do, already grabbed from whitelist
            continue
        elif case == "multiply":
            mult_config = config
            for col_name, multiple in config["col_names"].items():
                df[col_name] = df[col_name].apply(lambda x: x * multiple)
        else:
            for i, row in df.iterrows():
                if row["MINI_SBSTNC_DPNDC_NONALCHL_TIME"] == 1:
                    df.set_value(i, "MINI_SBSTNC_ABUSE_NONALCHL_TIME", 1)
                if row["MINI_ALCHL_DPNDC_TIME"] == 1:
                    df.set_value(i, "MINI_ALCHL_ABUSE_TIME", 1)
                if row["MINI_AN_TIME"] == 1:
                    df.set_value(i, "MINI_AN_BINGE_TIME", 1)
                if (row['EMPLOY_STATUS_6.0'] == 1) or (row['EMPLOY_STATUS_3.0'] == 1):
                    df.set_value(i, "EMPLOY_STATUS_1.0", 1)
                if row['EMPLOY_STATUS_4.0'] == 1:
                    df.set_value(i, "EMPLOY_STATUS_2.0", 1)
                add_new_imputed_features_canbind(df, row, i) # fill in new features
    
    # Drop columns that were used for calcs above
    for todrop in ["MINI_SBSTNC_DPNDC_NONALCHL_TIME","MINI_ALCHL_DPNDC_TIME","MINI_AN_TIME","EMPLOY_STATUS_6.0","EMPLOY_STATUS_3.0","EMPLOY_STATUS_4.0"]:
        df = df.drop([todrop], axis=1)
    
    # Filter out those without valid response/nonresponse values
    ## Already filtered so ignore
    ##df = get_valid_subjects(df)
    ##df = df.drop(["RESPOND_WK8"], axis=1)
    
    # Rename Column Headers according to dict
    df = df.rename(HEADER_CONVERSION_DICT, axis=1)
    
    # Check that all column headers have ::: to ensure they have correspondance in STAR*D
    for header in list(df.columns.values):
        if not (':::' in header):
            print('Warning! Likely unwanted column in output: ' + header)
    
    
    # Sort and output
    df = df.sort_values(by=['SUBJLABEL:::subjectkey'])
    df = df.drop(['SUBJLABEL:::subjectkey'], axis=1)
    df = df.reset_index(drop=True)
    df = df.sort_index(axis=1) # Newly added, sorts columns alphabetically so same for both matrices
    df.to_csv(output_dir + "/canbind-overlapping-X-data.csv")


def add_new_imputed_features_canbind(df, row, i):
    
    # imput_anyanxiety
    imput_anyanxiety = ['MINI_PTSD_TIME', 'MINI_PD_DX', 'MINI_AGRPHOBIA_TIME', 'MINI_SOCL_PHOBIA_DX', 'MINI_GAD_TIME']
    val = 1 if sum(row[imput_anyanxiety] == 1) > 0 else 0
    df.set_value(i, ':::imput_anyanxiety', val)
        
    # imput_QIDS_SR_perc_change
    val = round((row[Q_DICT['qids01_w2sr__qstot']] - row[Q_DICT['qids01_w0sr__qstot']]) / row[Q_DICT['qids01_w0sr__qstot']] if row[Q_DICT['qids01_w0sr__qstot']] else 0, 3)
    df.set_value(i, 'imput_QIDS_SR_perc_change:::', val)
    
    # Imputed new QIDS features
    for time in ['week0','week2']: 
        time2 = 'baseline' if time =='week0' else 'week2' #week0 is sometimes called _baseline
        
        # imput_QIDS_SR_sleep_domain
        val = round(np.nanmax(list(row[['QIDS_SR_1_' + time2,'QIDS_SR_2_' + time2,'QIDS_SR_3_' + time2,'QIDS_SR_4_' + time2]])))
        df.set_value(i, 'imput_QIDS_SR_sleep_domain_' + time + ':::', val)

        # imput_QIDS_SR_appetite_domain
        val = round(np.nanmax(list(row[['QIDS_SR_6_' + time2,'QIDS_SR_7_' + time2,'QIDS_SR_8_' + time2,'QIDS_SR_9_' + time2]])))
        df.set_value(i, 'imput_QIDS_SR_appetite_domain_' + time + ':::', val)
        
        # imput_QIDS_SR_psychomot_domain
        val = round(np.nanmax(list(row[['QIDS_SR_15_' + time2,'QIDS_SR_16_' + time2]])))
        df.set_value(i, 'imput_QIDS_SR_psychomot_domain_' + time + ':::', val)
        
        # imput_QIDS_SR_overeating
        val = round(np.nanmax(list(row[['QIDS_SR_7_' + time2,'QIDS_SR_9_' + time2]])))
        df.set_value(i, 'imput_QIDS_SR_overeating_' + time + ':::', val)

        # imput_QIDS_SR_insomnia
        val = round(np.nanmax(list(row[['QIDS_SR_1_' + time2,'QIDS_SR_2_' + time2,'QIDS_SR_3_' + time2]])))
        df.set_value(i, 'imput_QIDS_SR_insomnia_' + time + ':::', val)

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "-bothdefault":
        convert_stard_to_overlapping()
        convert_canbind_to_overlapping()

    elif len(sys.argv) == 4 and sys.argv[1] == "-both" and os.path.isdir(sys.argv[2]):
        convert_stard_to_overlapping(sys.argv[2])
        convert_canbind_to_overlapping(sys.argv[3])

    elif len(sys.argv) == 3 and sys.argv[1] == "-sd" and os.path.isdir(sys.argv[2]):
        convert_stard_to_overlapping(sys.argv[2])

    elif len(sys.argv) == 3 and sys.argv[1] == "-cb" and os.path.isdir(sys.argv[2]):
        convert_canbind_to_overlapping(sys.argv[2])

    else:
        print("Enter valid arguments\n"
              "\t options: -v for verbose, -v+ for super verbose\n"
              "\t path: the path to a real directory\n")