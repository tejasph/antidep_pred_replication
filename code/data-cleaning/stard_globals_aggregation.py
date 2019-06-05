import numpy as np

ignr = np.nan

# Values in "level", "week", "days_baseline" are ordered in terms of optimal preference, as
# they should be used to filter rows of subjects based on this.
# Preference for:
#   smaller days_baseline values
#   larger week values
#   earlier level values
SCALES = {
    "rs__dm01_enroll": {
        "whitelist": ['resid', 'rtown', 'resy', 'resm', 'marital', 'spous', 'relat', 'frend', 'thous',
                      'educat', 'student', 'empl', 'volun', 'leave', 'publica', 'medicaid', 'privins',
                      'mkedc', 'enjoy', 'famim']
    },
    "rs__dm01_w0": {
        "whitelist": ['inc_curr', 'mempl', 'assist', 'massist', 'unempl', 'munempl', 'otherinc', 'minc_other',
                      'totincom'],
    },
    "rs__ccv01_w0": {
        "whitelist": ['medication1_dosage', 'suicd', 'remsn', 'raise', 'effct', 'cncn', 'prtcl', 'stmed', 'trtmt'],
    },
    "rs__ccv01_w2": {
        "whitelist": ['medication1_dosage', 'suicd', 'remsn', 'raise', 'effct', 'cncn', 'prtcl', 'stmed', 'trtmt'],
    },
    "rs__crs01": {
        "whitelist": ['heart', 'vsclr', 'hema', 'eyes', 'ugi', 'lgi', 'renal', 'genur', 'mskl', 'neuro', 'psych',
                      'respiratory', 'liverd', 'endod'],
    },
    "rs__hrsd01 ": {
        "whitelist": ['hsoin', 'hmnin', 'hemin', 'hmdsd', 'hpanx', 'hinsg', 'happt', 'hwl', 'hsanx', 'hhypc', 'hvwsf',
                      'hsuic', 'hintr', 'hengy', 'hslow', 'hagit', 'hsex', 'hdtot_r'],
    },
    "rs__mhx01": {
        "whitelist": ['psmed'],
    },
    "rs__pdsq1": {
        "whitelist": ['evy2w', 'joy2w', 'int2w', 'lap2w', 'gap2w', 'lsl2w', 'msl2w', 'jmp2w', 'trd2w', 'glt2w', 'neg2w',
                      'flr2w', 'cnt2w', 'dcn2w', 'psv2w', 'wsh2w', 'btr2w', 'tht2w', 'ser2w', 'spf2w', 'sad2y', 'apt2y',
                      'slp2y', 'trd2y', 'cd2y', 'low2y', 'hpl2y', 'trexp', 'trwit', 'tetht', 'teups', 'temem', 'tedis',
                      'teblk', 'termd', 'tefsh', 'teshk', 'tedst', 'tenmb', 'tegug', 'tegrd', 'tejmp', 'ebnge', 'ebcrl',
                      'ebfl', 'ebhgy', 'ebaln', 'ebdsg', 'ebups', 'ebdt', 'ebvmt', 'ebwgh', 'obgrm', 'obfgt', 'obvlt',
                      'obstp', 'obint', 'obcln', 'obrpt', 'obcnt', 'anhrt', 'anbrt', 'anshk', 'anrsn', 'anczy', 'ansym',
                      'anwor', 'anavd', 'pechr', 'pecnf', 'peslp', 'petlk', 'pevth', 'peimp', 'imagn', 'imspy', 'imdgr',
                      'impwr', 'imcrl', 'imvcs', 'fravd', 'frfar', 'frcwd', 'frlne', 'frbrg', 'frbus', 'frcar', 'fralo',
                      'fropn', 'franx', 'frsit', 'emwry', 'emstu', 'ematn', 'emsoc', 'emavd', 'emspk', 'emeat', 'emupr',
                      'emwrt', 'emstp', 'emqst', 'embmt', 'empty', 'emanx', 'emsit', 'dkmch', 'dkfam', 'dkfrd', 'dkcut',
                      'dkpbm', 'dkmge', 'dgmch', 'dgfam', 'dgfrd', 'dgcut', 'dgpbm', 'dgmge', 'wynrv', 'wybad', 'wysdt',
                      'wydly', 'wyrst', 'wyslp', 'wytsn', 'wycnt', 'wysnp', 'wycrl', 'phstm', 'phach', 'phsck', 'phpr',
                      'phcse', 'wiser', 'wistp', 'wiill', 'wintr', 'widr'],
    },
    "rs__phx01": {
        "whitelist": ['dage', 'epino', 'episode_date', 'ai_none', 'alcoh', 'amphet', 'cannibis', 'opioid', 'pd_ag',
                      'pd_noag', 'specphob', 'soc_phob', 'ocd_phx', 'psd', 'gad_phx', 'axi_oth', 'aii_none', 'aii_def',
                      'aii_na', 'pd_border', 'pd_depend', 'pd_antis', 'pd_paran', 'pd_nos', 'axii_oth', 'dep', 'deppar',
                      'depsib', 'depchld', 'bip', 'bippar', 'bipsib', 'bipchld', 'alcohol', 'alcpar', 'alcsib',
                      'alcchld', 'drug_phx', 'drgpar', 'drgsib', 'drgchld', 'suic_phx', 'suicpar', 'suicsib',
                      'suicchld', 'wrsms', 'anorexia', 'bulimia', 'ax_cocaine'],
    },
    "rs__qlesq01": {
        "whitelist": ['qlesq01', 'qlesq02', 'qlesq03', 'qlesq04', 'qlesq05', 'qlesq06', 'qlesq07', 'qlesq08', 'qlesq09',
                      'qlesq10', 'qlesq11', 'qlesq12', 'qlesq13', 'qlesq14', 'qlesq15', 'qlesq16', 'totqlesq'],
    },
    "rs__sfhs01": {
        "whitelist": ['sfhs01', 'sfhs02', 'sfhs03', 'sfhs04', 'sfhs05', 'sfhs06', 'sfhs07', 'sfhs08', 'sfhs09',
                      'sfhs10', 'sfhs11', 'sfhs12', 'pcs12', 'mcs12'],
    },
    "rs__side_effects01": {
        "whitelist": ['fisfq', 'fisin', 'grseb'],
    },
    "rs__ucq01": {
        "whitelist": ['ucq010', 'ucq020', 'ucq030', 'ucq080', 'ucq091', 'ucq092', 'ucq100', 'ucq110', 'ucq120',
                      'ucq130', 'ucq140', 'ucq150', 'ucq160', 'ucq170', 'ucq040', 'ucq050', 'ucq060', 'ucq070'],
    },
    "rs__wpai01": {
        "whitelist": ['wpai01', 'wpai02', 'wpai03', 'wpai04', 'wpai05', 'wpai06', 'wpai_totalhrs', 'wpai_pctmissed',
                      'wpai_pctworked', 'wpai_pctwrkimp', 'wpai_pctactimp', 'wpai_totwrkimp'],
    },
    "rs_wsas01": {
        "whitelist": ['wsas01', 'wsas02', 'wsas03', 'wsas04', 'wsas05', 'totwsas', 'wsastot'],
    },
    "rs__qids01_w0c": {
        "whitelist": ['interview_age', 'gender', 'subjectkey', 'vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc',
                      'vapin', 'vwtdc', 'vwtin', 'vcntr', 'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit',
                      'qstot'],
    },
    "rs__qids01_w0sr": {
        "whitelist": ['vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc',
                      'vapin', 'vwtdc', 'vwtin', 'vcntr', 'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit',
                      'qstot'],
    },
    "rs__qids01_w2c": {
        "whitelist": ['vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc',
                      'vapin', 'vwtdc', 'vwtin', 'vcntr', 'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit',
                      'qstot'],
    },
    "rs__qids01_w2sr": {
        "whitelist": ['vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc',
                      'vapin', 'vwtdc', 'vwtin', 'vcntr', 'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit',
                      'qstot'],
    },
    "rs_idsc01": {
        "whitelist": ['isoin', 'imnin', 'iemin', 'ihysm', 'imdsd', 'ianx', 'ipanc', 'iirtb', 'irct', 'ivrtn', 'iwrse',
                      'ienv', 'iqty', 'iapdc', 'iapin', 'iwtdc', 'iwtin', 'icntr', 'ivwsf', 'ivwfr', 'isuic', 'iintr',
                      'iplsr', 'iengy', 'isex', 'islow', 'iagit', 'ismtc', 'isymp', 'igas', 'iintp', 'ildn'],
    },
}

COL_NAME_SUBJECTKEY = "subjectkey"
COL_NAME_WEEK = "week"
COL_NAME_LEVEL = "level"
COL_NAME_DAYS_BASELINE = "days_baseline"
COL_NAME_VERSION_FORM = "version_form"

"""
Notes: Values that get converted to np.nan are being eliminated completely. Values that are converted to the ignr 
variable string still need to be determined.
"""
VALUE_CONVERSION_MAP = {
    "demo_-7": {
        "col_names": {'medicaid', 'privins', 'mkedc', 'enjoy', 'famim', 'volun', 'leave'},
        "values": {-7: ignr}
    },
    "publica": {
        "col_names": {'publica'},
        "values": {-7: ignr, 3: ignr}
    },
    "empl": {
        "col_names": {'empl'},
        "values": {15: ignr, 9: ignr, -7: ignr}
    },
    "student": {
        "col_names": {'student'},
        "values": {2: 0.5}
    },
    "educat": {
        "col_names": {'student'},
        "values": {999: ignr}
    },
    "thous": {
        "col_names": {'thous'},
        "values": {99: ignr}
    },
    "medication1_dosage": {
        "col_names": {'medication1_dosage'},
        "col_extenders": ['_Level 1_0.1', '_Level 1_2'],
        "values": {0: ignr, 999: ignr}
    },
    "crs01": {
        "col_names": {'heart', 'vsclr', 'hema', 'eyes', 'ugi', 'lgi', 'renal', 'genur', 'mskl', 'neuro', 'psych',
                      'respiratory', 'liverd', 'endod', 'hsoin', 'hmnin', 'hemin', 'hmdsd', 'hpanx', 'hinsg', 'happt',
                      'hwl', 'hsanx', 'hhypc', 'hvwsf', 'hsuic', 'hintr', 'hengy', 'hslow', 'hagit', 'hsex', 'suic_phx',
                      'drug_phx', 'alcohol', 'bip', 'dep', 'dage'},
        "values": {-9: ignr}
    },
    "blank_to_zero": {
        "col_names": {'sex_prs', 'gdiar', 'gcnst', 'gdmth', 'gnone', 'gnsea', 'gstro', 'htplp', 'htdzy', 'htchs', 'htnone',
                      'heart_prs', 'skrsh', 'skpsp', 'skich', 'sknone', 'skdry', 'nvhed', 'nvtrm', 'nvcrd', 'nvnone',
                      'nvdzy', 'nrvsy', 'eyvsn', 'earng', 'enone', 'eyear', 'urdif', 'urpn', 'urmns', 'urfrq', 'urnone',
                      'genur_prs', 'sldif', 'slnone', 'slmch', 'sleep', 'sxls', 'sxorg', 'sxerc', 'sxnone', 'oaxty',
                      'octrt', 'omal', 'orsls', 'oftge', 'odegy', 'onone', 'other_prs', 'skin_c', 'deppar', 'depsib',
                      'depchld', 'bippar', 'bipsib', 'bipchld', 'alcpar', 'alcsib', 'alcchld', 'drgpar', 'drgsib',
                      'drgchld', 'suicpar', 'suicsib', 'suicchld', 'fisfq', 'fisin', 'grseb', 'wpai02', 'wpai03',
                      'wpai04', 'wpai05', 'wpai_totalhrs', 'wpai_pctmissed', 'wpai_pctworked', 'wpai_pctwrkimp',
                      'wpai_pctactimp', 'wpai_totwrkimp', 'ucq010', 'ucq020', 'ucq030', 'ucq080', 'ucq091', 'ucq092',
                      'ucq100', 'ucq110', 'ucq120', 'ucq130', 'ucq140', 'ucq150', 'ucq160', 'ucq170', 'ucq040',
                      'ucq050', 'ucq060', 'ucq070'},
        "values": {"": 0}
    },
    "bulimia": {
        "col_names": {'bulimia'},
        "values": {0: np.nan, 1: np.nan, 2: "2/5", 5: "2/5"}
    },
    "zero_to_nan": {
        "col_names": {'ax_cocaine', 'alcoh', 'amphet', 'cannibis' , 'opioid'},
        "values": {0: np.nan}
    },
    "two_to_zero": {
        "col_names": {'wpai01', 'sfhs04', 'sfhs05', 'sfhs06', 'sfhs07', 'ucq010', 'ucq020', 'ucq080', 'ucq110',
                      'ucq120', 'ucq140', 'ucq160', 'ucq040', 'ucq060'},
        "values": {2: 0}
    },
    "sex_prs": {
        "col_names": {'sex_prs'},
        "values": {-7: 0, "": 0}
    },
    "qids01": {
        "col_names": {'vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc', 'vapin', 'vwtdc', 'vwtin', 'vcntr',
                      'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit'},
        "col_extenders": ['_week0_Self Rating', '_week2_Self Rating', '_week0_Clinician',
                          '_week2_Clinician'],
        "values": {999: 0}
    },
    "minus": {
        6: {'sfhs12', 'sfhs11', 'sfhs10', 'sfhs09', 'sfhs01'}, # Subtract 6 minus value
        3: {'sfhs02', 'sfhs03'}, # Subtract 3 minus value
        1: {'sfhs08'} # Subtract 1
    },
}

## TODO - not done.
COL_NAMES_ONE_HOT_ENCODE = {'trtmt', 'trtmt', 'gender', 'resid', 'rtown', 'marital', 'bulimia',
                                'ax_cocaine', 'alcoh', 'amphet', 'cannibis' , 'opioid', 'empl', 'volun', 'leave',
                                'publica', 'medicaid', 'privins', 'iwrse'}



ONE_HOT_ENCODE = {

}
