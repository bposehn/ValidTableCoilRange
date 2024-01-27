import os, sys
import string
import random
import time
import pickle
import re

import pandas as pd
import numpy as np

def get_coils_from_filename(filename, coil_names):
    coils = {}
    for coil_name in coil_names:
        start = filename.find(coil_name)
        end_coil_name = start + len(coil_name) + 1
        end_number = filename.find('A', end_coil_name)
        coils[coil_name] = float(filename[end_coil_name:end_number])
    
    return coils

def get_df_from_recons(recons_dir, num_expected):
    query_column_names = ['WBpolFCnoDC', 'q050', 'NevinsC', 'beta_pol1']
    
    data = np.empty((num_expected, 2*len(query_column_names)))

    suffixes = ['_mean', '_truth', '_1sigma']
    df_column_names = []
    for query_column_name in query_column_names:
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

    num_coils = 5
    num_configs_per_coil = 10
    num_configs_per_table_axis_config = 1 + (num_coils * num_configs_per_coil) 

    data = np.empty(len(recon_df), len(recon_df.columns)+2)
    for i_filename, filename in enumerate(out_filenames):
        i_coil_config = int(i_filename / num_configs_per_table_axis_config)
        i_ta_config = i_filename % num_configs_per_table_axis_config
        data[i_filename, 0] = i_coil_config
        data[i_filename, 1] = i_ta_config
        row = recon_df.loc[filename] # what to do if can't get it
        data[2:] = row.values

    column_names = ['Coil Config Index', 'TA Config Index'] + recon_df.columns.values.tolist()
    organized_df = pd.DataFrame(data = data, columns = column_names)

    return organized_df

def analyze_ta_config_set_size(organized_df):



def analyze_df(df):


if __name__ == '__main__':
    get_coils_from_filename('0hz-Coil_A_0A-Coil_B_0A-Coil_C_449.048268A-Coil_D_-343.1827A-PFC_1_0A-PFC_2_0A-PFC_3_0A_s0.019_dc-0.028_a0.7_b0.7_c0.33_y29257.92357454872_i0.41_b0.10_pa2_pb02.hdf5', ['Coil_D', 'PFC_1'])