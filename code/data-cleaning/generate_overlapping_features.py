import os
import sys
import pandas as pd

from utils import *

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
CANBIND_OVERLAPPING_VALUE_CONVERSION_MAP = {

    "whitelist": ['MADRS_TOT_PRO_RATED_baseline', 'MADRS_TOT_PRO_RATED_week 2', 'SUBJLABEL', 'EDUC', 'HSHLD_INCOME',
                  'MINI_AGRPHOBIA_TIME', 'MINI_APD_TIME', 'MINI_GAD_TIME', 'MINI_MDE_TIME_CUR', 'MINI_BN_TIME',
                  'MINI_OCD_TIME', 'MINI_PTSD_TIME', 'MINI_SOCL_PHOBIA_DX', 'PSYHIS_FH', 'PSYHIS_MDD_AGE',
                  'PSYHIS_MDD_PREV', 'PSYHIS_MDE_NUM', 'QLESQ_1A_1_baseline_QLESQ_1B_1_baseline_merged',
                  'QLESQ_1A_2_baseline_QLESQ_1B_2_baseline_merged', 'QLESQ_1A_3_baseline_QLESQ_1B_3_baseline_merged',
                  'QLESQ_1A_4_baseline_QLESQ_1B_4_baseline_merged', 'QLESQ_1A_5_baseline_QLESQ_1B_5_baseline_merged',
                  'QLESQ_1A_6_baseline_QLESQ_1B_6_baseline_merged', 'QLESQ_1A_7_baseline_QLESQ_1B_7_baseline_merged',
                  'QLESQ_1A_8_baseline_QLESQ_1B_8_baseline_merged', 'QLESQ_1A_9_baseline_QLESQ_1B_9_baseline_merged',
                  'QLESQ_1A_10_baseline_QLESQ_1B_10_baseline_merged',
                  'QLESQ_1A_11_baseline_QLESQ_1B_11_baseline_merged',
                  'QLESQ_1A_12_baseline_QLESQ_1B_12_baseline_merged',
                  'QLESQ_1A_13_baseline_QLESQ_1B_13_baseline_merged',
                  'QLESQ_1A_14_baseline_QLESQ_1B_14_baseline_merged',
                  'QLESQ_1A_16_baseline_QLESQ_1B_16_baseline_merged', 'QLESQA_TOT_QLESQB_TOT_merged',
                  'SDS_1_1_baseline', 'SDS_2_1_baseline', 'SDS_3_1_baseline', 'LAM_2_baseline', 'LAM_3_baseline',
                  'SEX_female', 'SEX_male', 'AGE', 'MRTL_STATUS_Divorced', 'MRTL_STATUS_Domestic Partnership',
                  'MRTL_STATUS_Married', 'MRTL_STATUS_Never Married', 'MRTL_STATUS_Separated', 'MRTL_STATUS_Widowed',
                  'EMPLOY_STATUS_1.0', 'EMPLOY_STATUS_2.0', 'EMPLOY_STATUS_4.0', 'EMPLOY_STATUS_5.0',
                  'EMPLOY_STATUS_6.0', 'EMPLOY_STATUS_7.0', 'PSYHIS_MDE_EP_DUR_MO', 'MINI_AN_BINGE_TIME',
                  'MINI_ALCHL_ABUSE_TIME', 'MINI_PD_DX', 'MINI_ALCHL_DPNDC_TIME', 'MINI_SBSTNC_ABUSE_NONALCHL_TIME',
                  'MINI_AN_TIME', 'MINI_SBSTNC_DPNDC_NONALCHL_TIME'],
    "multiply": {
        "description": "Multiply the value by the multiple specified.",
        "col_names": {
            "PSYHIS_MDE_EP_DUR_MO": 30,

        }
    },
    "other": {}
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
    df.to_csv(output_dir + "/stard-overlapping-X-data.csv")


def convert_canbind_to_overlapping(output_dir=""):
    if output_dir == "":
        output_dir = "/Users/teyden/Downloads/canbind-data-binned/output_files" # TODO temporarily hardcode
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # df = pd.read_csv(file_path)
    orig_df = pd.read_csv(output_dir + "/canbind-overlapping-X-data.csv")
    df = orig_df.drop(["Unnamed: 0"], axis=1)

    # Take whitelist columns first
    df = df[CANBIND_OVERLAPPING_VALUE_CONVERSION_MAP["whitelist"] + ["RESPOND_WK8"]]

    # Then process them
    for case, config in CANBIND_OVERLAPPING_VALUE_CONVERSION_MAP.items():
        if case == "keep":
            # Nothing to do, already grabbed from whitelist
            continue
        elif case == "multiply":
            mult_config = config[case]
            for col_name, multiple in mult_config.items():
                df[col_name] = df[col_name].apply(lambda x: x * multiple)
        else:
            for i, row in df.iterrows():
                if row["MINI_SBSTNC_DPNDC_NONALCHL_TIME"] == 1:
                    df.set_value(i, "MINI_SBSTNC_ABUSE_NONALCHL_TIME", 1)
                if row["MINI_ALCHL_DPNDC_TIME"] == 1:
                    df.set_value(i, "MINI_ALCHL_ABUSE_TIME", 1)
                if row["MINI_AN_TIME"] == 1 or row["empl_13"] == 1:
                    df.set_value(i, "MINI_AN_BINGE_TIME", 1)

    # Filter out those without valid response/nonresponse values
    df = get_valid_subjects(df)
    df = df.drop(["RESPOND_WK8"], axis=1)

    df = df.sort_values(by=["subjectkey"])
    df = df.reset_index(drop=True)
    df.to_csv(output_dir + "/canbind-overlapping-X-data.csv")


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