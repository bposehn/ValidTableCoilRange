import os, sys
import string
import random
import time
import pickle
import re

import pandas as pd
import numpy as np

COILS_OF_INTEREST = ['Coil_A', 'Coil_C', 'Coil_D', 'PFC_1','PFC_3']
QUERY_COLUMN_NAMES = ['WBpolFCnoDC', 'q050', 'NevinsC', 'beta_pol1']

def get_coils_from_filename(filename, coil_names):
    coils = {}
    for coil_name in coil_names:
        start = filename.find(coil_name)
        end_coil_name = start + len(coil_name) + 1
        end_number = filename.find('A', end_coil_name)
        coils[coil_name] = float(filename[end_coil_name:end_number])
    
    return coils

def get_df_from_recons(recons_dir, num_expected):   
    data = np.empty((num_expected, 2*len(QUERY_COLUMN_NAMES)))

    suffixes = ['_mean', '_truth', '_1sigma']
    df_column_names = []
    for query_column_name in QUERY_COLUMN_NAMES:
        df_column_names += [query_column_name + suffix for suffix in suffixes]

    pattern = re.compile("^testcase_[0-9]+.csv")
    i_row = -1
    filenames = []
    for filepath in os.listdir(recons_dir):
        if pattern.match(filepath):
            i_row += 1
            recon_df = pd.read_csv(os.path.join(recons_dir, filepath))
            for i_column, query_column_name in enumerate(df_column_names):
                data[i_row, i_column] = recon_df[query_column_name].item()
            filenames.append(recon_df['FileName'])

    interim_df = pd.DataFrame(data = data, columns = df_column_names)
    interim_df['FileName'] = filenames 
    interim_df.set_index('FileName', inplace=True)

    return interim_df

def organize_df(recon_df: pd.DataFrame):
    out_filenames_path = ''
    out_filenames = pickle.load(open(out_filenames_path))

    num_coils = len(COILS_OF_INTEREST)
    num_configs_per_coil = 10 # plus the base!
    num_configs_per_table_axis_config = 1 + (num_coils * num_configs_per_coil)
    data = np.empty(len(recon_df), len(recon_df.columns) + num_coils + 2) # 2 for Coil Config Index, TA Config Index
    for i_filename, filename in enumerate(out_filenames):
        i_ta_config = int(i_filename / (num_configs_per_table_axis_config + 1)) 
        i_coil_config = i_filename % (num_configs_per_table_axis_config + 1) #in [0, 50], 0 is base 
        data[i_filename, 0] = i_coil_config
        data[i_filename, 1] = i_ta_config
        coil_values = get_coils_from_filename(filename, COILS_OF_INTEREST)
        for i_coil, coil_name in coil_values:
            data[i_filename, 2 + i_coil] = coil_values[coil_name]
        row = recon_df.loc[filename] # what to do if can't get it
        data[2:] = row.values

    column_names = ['Coil Config Index', 'TA Config Index'] + COILS_OF_INTEREST + recon_df.columns.values.tolist()
    organized_df = pd.DataFrame(data = data, columns = column_names)

    return organized_df

def get_truth_outside_sigma_bound(organized_df: pd.DataFrame, sigma_range: float):

    col_type = 'TOSB'
    col_descriptors = [prod[0] + '_' + prod[1] for prod in itertools.product(COILS_OF_INTEREST, QUERY_COLUMN_NAMES)]
    col_names = [col_type + coil_name for descriptor in col_descriptors]
    data = np.empty((len(organized_df, len(COILS_OF_INTEREST))))
    
    ta_config_index_colname = 'TA Config Index'

    #each TA config, each coil 
    for i_ta_config in organized_df[ta_config_index_colname].unique():
        first_i_ta_config_index = organized_df.where(organized_df[ta_config_index_colname] == i_ta_config).first_valid_index()
        base_coil_row = pd.DataFrame(data = [organized_df.iloc[first_i_ta_config_index]], columns = organized_df.columns)
        for coil_name in COILS_OF_INTEREST:
            varying_coils_df = organized_df.loc[(organized_df[coil_name] != organized_df.iloc[0][coil_name]) & (organized_df[ta_config_index_colname] == first_i_ta_config_index)]
            varying_coils_df = pd.concat([base_coil_row, varying_coils_df])




    return f'truth outside {sigma_range} sigma', row

def analyze_ta_config_set_size(organized_df):
    pass

def analyze_df(df):
    pass

if __name__ == '__main__':
