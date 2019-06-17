import numpy as np

ignr = np.nan

ORIGINAL_SCALE_NAMES = {
    "dm01": {
        "subjectkey_selector": "dm01_id",
        "preference": "smaller"
    },
    "ccv01": {
        "subjectkey_selector": "ccv01_id",
        "preference": "larger"
    },
    "crs01": {
    },
    "hrsd01": {
    },
    "idsc01": {
        "subjectkey_selector": "idsc01_id",
        "preference": "smaller"
    },
    "mhx01": {
        "subjectkey_selector": "mhx01_id",
        "preference": "smaller"
    },
    "pdsq01": {
        "subjectkey_selector": "pdsq01_id",
        "preference": "smaller"
    },
    "phx01": {
        "subjectkey_selector": "phx01_id",
        "preference": "smaller"
    },
    "qids01": {
        "subjectkey_selector": "days_baseline",
        "preference": "smaller"
    },
    "qlesq01": {
        "subjectkey_selector": "qlesq01",
        "preference": "smaller"
    },
    "sfhs01": {
        "subjectkey_selector": "sfhs01",
        "preference": "smaller"
    },
    "side_effects01": {
        "subjectkey_selector": "side_effects01_id",
        "preference": "larger"
    },
    "side": {

    },
    "ucq01": {
        "subjectkey_selector": "ucq01_id",
        "preference": "smaller"
    },
    "wpai01": {
        "subjectkey_selector": "wpai01",
        "preference": "smaller"
    },
    "wsas01": {
        "subjectkey_selector": "wsas01_id",
        "preference": "smaller"
    }
}

# Columns to keep for each scale
SCALES = {
    "dm01_enroll": {
        "whitelist": ['resid', 'rtown', 'resy', 'resm', 'marital', 'spous', 'relat', 'frend', 'thous',
                      'educat', 'student', 'empl', 'volun', 'leave', 'publica', 'medicaid', 'privins',
                      'mkedc', 'enjoy', 'famim'],
        "one_hot_encode": ['resid', 'rtown', 'marital', 'empl', 'volun', 'leave', 'publica', 'medicaid', 'privins']
    },
    "dm01_w0": {
        "whitelist": ['inc_curr', 'mempl', 'assist', 'massist', 'unempl', 'munempl', 'otherinc', 'minc_other',
                      'totincom'],
    },
    # "ccv01_w0": {
    #     "whitelist": ['medication1_dosage', 'suicd', 'remsn', 'raise', 'effct', 'cncn', 'prtcl', 'stmed', 'trtmt'],
    #     "one_hot_encode": ['trtmt']
    # },
    "ccv01_w2": {
        "whitelist": ['medication1_dosage', 'suicd', 'remsn', 'raise', 'effct', 'cncn', 'prtcl', 'stmed', 'trtmt'],
        "one_hot_encode": ['trtmt']
    },
    "crs01": {
        "whitelist": ['heart', 'vsclr', 'hema', 'eyes', 'ugi', 'lgi', 'renal', 'genur', 'mskl', 'neuro', 'psych',
                      'respiratory', 'liverd', 'endod'],
    },
    "hrsd01": {
        "whitelist": ['hsoin', 'hmnin', 'hemin', 'hmdsd', 'hpanx', 'hinsg', 'happt', 'hwl', 'hsanx', 'hhypc', 'hvwsf',
                      'hsuic', 'hintr', 'hengy', 'hslow', 'hagit', 'hsex', 'hdtot_r'],
    },
    "mhx01": {
        "whitelist": ['psmed'],
    },
    "pdsq01": {
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
    "phx01": {
        "whitelist": ['dage', 'epino', 'episode_date', 'ai_none', 'alcoh', 'amphet', 'cannibis', 'opioid', 'pd_ag',
                      'pd_noag', 'specphob', 'soc_phob', 'ocd_phx', 'psd', 'gad_phx', 'axi_oth', 'aii_none', 'aii_def',
                      'aii_na', 'pd_border', 'pd_depend', 'pd_antis', 'pd_paran', 'pd_nos', 'axii_oth', 'dep', 'deppar',
                      'depsib', 'depchld', 'bip', 'bippar', 'bipsib', 'bipchld', 'alcohol', 'alcpar', 'alcsib',
                      'alcchld', 'drug_phx', 'drgpar', 'drgsib', 'drgchld', 'suic_phx', 'suicpar', 'suicsib',
                      'suicchld', 'wrsms', 'anorexia', 'bulimia', 'ax_cocaine'],
        "one_hot_encode": ['alcoh', 'amphet', 'cannibis', 'opioid', 'ax_cocaine', 'bulimia']
    },
    "qlesq01": {
        "whitelist": ['qlesq01', 'qlesq02', 'qlesq03', 'qlesq04', 'qlesq05', 'qlesq06', 'qlesq07', 'qlesq08', 'qlesq09',
                      'qlesq10', 'qlesq11', 'qlesq12', 'qlesq13', 'qlesq14', 'qlesq15', 'qlesq16', 'totqlesq'],
    },
    "sfhs01": {
        "whitelist": ['sfhs01', 'sfhs02', 'sfhs03', 'sfhs04', 'sfhs05', 'sfhs06', 'sfhs07', 'sfhs08', 'sfhs09',
                      'sfhs10', 'sfhs11', 'sfhs12', 'pcs12', 'mcs12'],
    },
    "side_effects01": {
        "whitelist": ['fisfq', 'fisin', 'grseb'],
    },
    "ucq01": {
        "whitelist": ['ucq010', 'ucq020', 'ucq030', 'ucq080', 'ucq091', 'ucq092', 'ucq100', 'ucq110', 'ucq120',
                      'ucq130', 'ucq140', 'ucq150', 'ucq160', 'ucq170', 'ucq040', 'ucq050', 'ucq060', 'ucq070'],
    },
    "wpai01": {
        "whitelist": ['wpai01', 'wpai02', 'wpai03', 'wpai04', 'wpai05', 'wpai06', 'wpai_totalhrs', 'wpai_pctmissed',
                      'wpai_pctworked', 'wpai_pctwrkimp', 'wpai_pctactimp', 'wpai_totwrkimp'],
    },
    "wsas01": {
        "whitelist": ['wsas01', 'wsas02', 'wsas03', 'wsas04', 'wsas05', 'totwsas', 'wsastot'],
    },
    "qids01_w0c": {
        "whitelist": ['interview_age', 'gender', 'subjectkey', 'vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc',
                      'vapin', 'vwtdc', 'vwtin', 'vcntr', 'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit',
                      'qstot'],
        "one_hot_encode": ['gender']
    },
    "qids01_w0sr": {
        "whitelist": ['vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc',
                      'vapin', 'vwtdc', 'vwtin', 'vcntr', 'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit',
                      'qstot'],
    },
    "qids01_w2c": {
        "whitelist": ['vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc',
                      'vapin', 'vwtdc', 'vwtin', 'vcntr', 'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit',
                      'qstot'],
    },
    "qids01_w2sr": {
        "whitelist": ['vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc',
                      'vapin', 'vwtdc', 'vwtin', 'vcntr', 'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit',
                      'qstot'],
    },
    "idsc01": {
        "whitelist": ['isoin', 'imnin', 'iemin', 'ihysm', 'imdsd', 'ianx', 'ipanc', 'iirtb', 'irct', 'ivrtn', 'iwrse',
                      'ienv', 'iqty', 'iapdc', 'iapin', 'iwtdc', 'iwtin', 'icntr', 'ivwsf', 'ivwfr', 'isuic', 'iintr',
                      'iplsr', 'iengy', 'isex', 'islow', 'iagit', 'ismtc', 'isymp', 'igas', 'iintp', 'ildn'],
        "one_hot_encode": ['iwrse']
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
        "conversion_map": {-7: ignr}
    },
    "student": {
        "col_names": {'student'},
        "conversion_map": {2: 0.5}
    },
    "educat": {
        "col_names": {'student'},
        "conversion_map": {999: ignr}
    },
    "thous": {
        "col_names": {'thous'},
        "conversion_map": {99: ignr}
    },
    "medication1_dosage": {
        "col_names": {'medication1_dosage'},
        "conversion_map": {0: ignr, 999: ignr}
    },
    "-9_to_nan": {
        "col_names": {'heart', 'vsclr', 'hema', 'eyes', 'ugi', 'lgi', 'renal', 'genur', 'mskl', 'neuro', 'psych',
                      'respiratory', 'liverd', 'endod', 'hsoin', 'hmnin', 'hemin', 'hmdsd', 'hpanx', 'hinsg', 'happt',
                      'hwl', 'hsanx', 'hhypc', 'hvwsf', 'hsuic', 'hintr', 'hengy', 'hslow', 'hagit', 'hsex', 'suic_phx',
                      'drug_phx', 'alcohol', 'bip', 'dep', 'dage'},
        "conversion_map": {-9: ignr}
    },
    # "blank_to_zero": {
    #     "col_names": {'sex_prs', 'gdiar', 'gcnst', 'gdmth', 'gnone', 'gnsea', 'gstro', 'htplp', 'htdzy', 'htchs', 'htnone',
    #                   'heart_prs', 'skrsh', 'skpsp', 'skich', 'sknone', 'skdry', 'nvhed', 'nvtrm', 'nvcrd', 'nvnone',
    #                   'nvdzy', 'nrvsy', 'eyvsn', 'earng', 'enone', 'eyear', 'urdif', 'urpn', 'urmns', 'urfrq', 'urnone',
    #                   'genur_prs', 'sldif', 'slnone', 'slmch', 'sleep', 'sxls', 'sxorg', 'sxerc', 'sxnone', 'oaxty',
    #                   'octrt', 'omal', 'orsls', 'oftge', 'odegy', 'onone', 'other_prs', 'skin_c', 'deppar', 'depsib',
    #                   'depchld', 'bippar', 'bipsib', 'bipchld', 'alcpar', 'alcsib', 'alcchld', 'drgpar', 'drgsib',
    #                   'drgchld', 'suicpar', 'suicsib', 'suicchld', 'fisfq', 'fisin', 'grseb', 'wpai02', 'wpai03',
    #                   'wpai04', 'wpai05', 'wpai_totalhrs', 'wpai_pctmissed', 'wpai_pctworked', 'wpai_pctwrkimp',
    #                   'wpai_pctactimp', 'wpai_totwrkimp', 'ucq010', 'ucq020', 'ucq030', 'ucq080', 'ucq091', 'ucq092',
    #                   'ucq100', 'ucq110', 'ucq120', 'ucq130', 'ucq140', 'ucq150', 'ucq160', 'ucq170', 'ucq040',
    #                   'ucq050', 'ucq060', 'ucq070'},
    #     "conversion_map": {"": 0}
    # },
    "zero_to_nan": {
        "col_names": {'ax_cocaine', 'alcoh', 'amphet', 'cannibis' , 'opioid'},
        "conversion_map": {0: ignr}
    },
    "two_to_zero": {
        "col_names": {'wpai01', 'sfhs04', 'sfhs05', 'sfhs06', 'sfhs07', 'ucq010', 'ucq020', 'ucq080', 'ucq110',
                      'ucq120', 'ucq140', 'ucq160', 'ucq040', 'ucq060'},
        "conversion_map": {2: 0}
    },
    "sex_prs": {
        "col_names": {'sex_prs'},
        "conversion_map": {-7: 0, "": 0}
    },
    "qids01": {
        "col_names": {'vsoin', 'vmnin', 'vemin', 'vhysm', 'vmdsd', 'vapdc', 'vapin', 'vwtdc', 'vwtin', 'vcntr',
                      'vvwsf', 'vsuic', 'vintr', 'vengy', 'vslow', 'vagit'},
        "conversion_map": {999: 0}
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
    'resid', 'rtown', 'marital', 'empl', 'volun', 'leave', 'publica', 'medicaid', 'privins',
    'trtmt',
    'gender',
    'alcoh', 'amphet', 'cannibis', 'opioid',
    'ax_cocaine',
    'iwrse'
}
